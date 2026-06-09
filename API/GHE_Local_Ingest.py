# %% NOAA Global Hydro-Estimator (GHE) Enterprise Rain Rate Ingest
# Downloads the latest GHE 15-minute rain rate data from the NOAA Open Data bucket,
# applies a 2-hour Lagrangian extrapolation nowcast using pysteps optical flow, and
# stores the result (1 observed + 8 nowcast frames) as a zarr store.
# Alexander Rey, June 2025

# %% Import modules
import logging
import os
import pickle
import shutil
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone

import dask
import dask.array as da
import numpy as np
import s3fs
import xarray as xr
import zarr
from pysteps.extrapolation.semilagrangian import extrapolate
from pysteps.motion.lucaskanade import dense_lucaskanade

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    VALID_DATA_MAX,
    close_store,
    configure_zarr_limits,
    pad_to_chunk_size,
    positive_int_env,
    tune_nofile_limit,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# %% Constants
# S3 bucket with NOAA GHE Enterprise Rain Rate data (public, no auth required)
GHE_BUCKET = "noaa-ghe"

# Number of nowcast frames to generate (8 × 15 min = 2 hours)
NOWCAST_FRAMES = 8

# Number of recent observations to use for optical-flow motion estimation
# Using 4 frames (1 hour of history) gives a stable motion estimate
OPTICAL_FLOW_FRAMES = 4

# Temporal resolution of GHE data (minutes)
GHE_INTERVAL_MIN = 15

# Rain rate threshold below which values are treated as no-rain (mm/h)
# Avoids noisy near-zero rain rate pixels biasing the optical flow
GHE_MIN_RAIN_RATE = 0.1

# Name of the zarr variable group (used only for output; the zarr array holds
# variables in index order: 0=time, 1=rain_rate)
zarr_vars = ("time", "rain_rate")

# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR

forecast_process_dir = os.getenv("forecast_process_dir", default="/mnt/nvme/data/GHE")
tmp_dir = os.path.join(forecast_process_dir, "Downloads")

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/GHE")

process_chunk = CHUNK_SIZES["GHE"]
final_chunk = FINAL_CHUNK_SIZES["GHE"]

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")
zarr_store_workers = positive_int_env("zarr_store_workers", 2)
zarr_async_concurrency = positive_int_env("zarr_async_concurrency", 2)

# Authenticated S3 for writing to private bucket
s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
# Anonymous S3 for reading NOAA public bucket
s3_public = s3fs.S3FileSystem(anon=True)

tune_nofile_limit()
zarr_store_workers, zarr_async_concurrency = configure_zarr_limits(
    zarr_store_workers, zarr_async_concurrency
)

# Create / reset working directory
if os.path.exists(forecast_process_dir):
    shutil.rmtree(forecast_process_dir)
os.makedirs(forecast_process_dir)
os.makedirs(tmp_dir)

if save_type == "Download":
    os.makedirs(os.path.join(forecast_path, ingest_version), exist_ok=True)


# %% Helper functions


def _candidate_timestamps(n_candidates: int = 40) -> list[datetime]:
    """Return a list of UTC datetimes at 15-min intervals, newest first."""
    now = datetime.now(timezone.utc)
    # Round down to the nearest 15-minute mark
    base = now.replace(
        minute=(now.minute // GHE_INTERVAL_MIN) * GHE_INTERVAL_MIN,
        second=0,
        microsecond=0,
    )
    return [base - timedelta(minutes=i * GHE_INTERVAL_MIN) for i in range(n_candidates)]


def _build_s3_path(dt: datetime) -> str:
    """Build the S3 key for a given UTC datetime.

    NOAA GHE Enterprise files follow the pattern:
        s3://noaa-ghe/{year}/{julian_day}/{hour}/GHE_RR_v2.0_{YYYYMMDDTHHmm}00Z.nc
    """
    year = dt.strftime("%Y")
    doy = dt.strftime("%j")  # zero-padded Julian day (001-366)
    hour = dt.strftime("%H")
    timestamp = dt.strftime("%Y%m%dT%H%M")
    return f"{GHE_BUCKET}/{year}/{doy}/{hour}/GHE_RR_v2.0_{timestamp}00Z.nc"


def _find_ghe_files(n_frames: int = OPTICAL_FLOW_FRAMES) -> list[tuple[str, datetime]]:
    """Discover the *n_frames* most recent GHE files available on S3.

    Returns a list of (s3_path, datetime) tuples ordered newest→oldest.
    Raises RuntimeError if fewer files than required are found.
    """
    found: list[tuple[str, datetime]] = []
    # Search up to 3 × n_frames time slots back to allow for occasional missing files
    for dt in _candidate_timestamps(n_candidates=n_frames * 3):
        path = _build_s3_path(dt)
        try:
            if s3_public.exists(path):
                found.append((path, dt))
                if len(found) >= n_frames:
                    break
        except Exception as exc:
            logging.warning("Error checking %s: %s", path, exc)

    if not found:
        raise RuntimeError(
            f"No GHE files found after checking {n_frames * 3} time slots. "
            "Check network connectivity or S3 path format."
        )
    if len(found) < 2:
        raise RuntimeError(
            f"Only {len(found)} GHE file(s) found; need at least 2 for optical flow."
        )
    logging.info("Found %d GHE file(s) for processing.", len(found))
    return found


def _load_rain_rate(s3_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download and read a GHE NetCDF file.

    Returns (rain_rate_2d, lats_1d, lons_1d) where rain_rate_2d has units mm/h.
    """
    local_nc = os.path.join(tmp_dir, os.path.basename(s3_path))
    s3_public.get(s3_path, local_nc)

    with xr.open_dataset(local_nc, engine="netcdf4") as ds:
        # The primary variable is 'RainRate' (mm/h). Detect it robustly.
        rr_candidates = [v for v in ds.data_vars if "rain" in v.lower()]
        if not rr_candidates:
            rr_candidates = list(ds.data_vars)
        var_name = rr_candidates[0]
        logging.debug("Using variable '%s' from %s", var_name, s3_path)

        rr = ds[var_name].values.astype(np.float32)

        # Extract 1-D lat/lon coordinate arrays
        lat_names = [c for c in ds.coords if "lat" in c.lower()]
        lon_names = [c for c in ds.coords if "lon" in c.lower()]
        lats = ds[lat_names[0]].values.astype(np.float32)
        lons = ds[lon_names[0]].values.astype(np.float32)

    os.remove(local_nc)  # clean up immediately to save disk space

    # Ensure shape is (lat, lon) — squeeze any size-1 extra dims
    rr = np.squeeze(rr)
    if rr.ndim != 2:
        raise ValueError(f"Expected 2-D rain rate array, got shape {rr.shape}")

    # Replace fill/missing values with 0 (no rain)
    rr = np.where(np.isfinite(rr) & (rr >= 0), rr, 0.0)

    return rr, lats, lons


def _to_dbr(rain_rate: np.ndarray) -> np.ndarray:
    """Convert rain rate (mm/h) to dBR (decibel rain rate) for optical flow.

    Uses a threshold floor to avoid log(0).
    """
    r = np.where(rain_rate >= GHE_MIN_RAIN_RATE, rain_rate, GHE_MIN_RAIN_RATE)
    return 10.0 * np.log10(r)


def _from_dbr(dbr: np.ndarray) -> np.ndarray:
    """Convert dBR back to rain rate (mm/h), zeroing sub-threshold values."""
    r = 10.0 ** (dbr / 10.0)
    return np.where(r > GHE_MIN_RAIN_RATE, r, 0.0)


def _compute_nowcast(
    observations: list[np.ndarray],
    n_frames: int = NOWCAST_FRAMES,
) -> np.ndarray:
    """Estimate motion field and extrapolate precipitation field forward.

    Parameters
    ----------
    observations:
        List of rain-rate arrays (newest first) in mm/h on a regular grid.
    n_frames:
        Number of 15-min nowcast frames to produce.

    Returns
    -------
    nowcast_frames : ndarray, shape (n_frames, lat, lon)
        Extrapolated rain-rate fields in mm/h, indexed 0=+15min … n-1=+2h.
    """
    # Reverse to chronological order required by pysteps (oldest first)
    chron = list(reversed(observations))

    # Transform to dBR space for optical flow stability
    dbr_stack = np.stack([_to_dbr(r) for r in chron], axis=0)  # (T, lat, lon)

    # Estimate motion field using Lucas-Kanade optical flow
    logging.info("Estimating motion field with Lucas-Kanade optical flow …")
    velocity = dense_lucaskanade(dbr_stack)  # shape (2, lat, lon)
    logging.info("Motion field estimated. Running %d-frame extrapolation …", n_frames)

    # Semi-Lagrangian extrapolation in dBR space
    # vel_timestep=1 means velocity is in pixels-per-input-time-step
    nowcast_dbr = extrapolate(
        dbr_stack[-1],
        velocity,
        timesteps=n_frames,
        allow_nonfinite_values=True,
        vel_timestep=1,
    )  # shape (n_frames, lat, lon)

    # Back-transform to rain rate
    nowcast_rr = np.stack([_from_dbr(f) for f in nowcast_dbr], axis=0)

    logging.info("Nowcast complete: %d frames generated.", n_frames)
    return nowcast_rr


# %% Main processing
t0 = time.time()

# 1. Discover recent GHE files
logging.info("Searching for latest NOAA GHE files …")
ghe_files = _find_ghe_files(n_frames=OPTICAL_FLOW_FRAMES)
latest_path, latest_dt = ghe_files[0]
logging.info("Latest GHE observation: %s", latest_dt.isoformat())

# 2. Check whether this is newer than the stored timestamp
pickle_filename = "GHE.time.pickle"
if save_type == "S3":
    pickle_s3_path = os.path.join(forecast_path, ingest_version, pickle_filename)
    if s3.exists(pickle_s3_path):
        with s3.open(pickle_s3_path, "rb") as fh:
            previous_dt = pickle.load(fh)
        if previous_dt >= latest_dt:
            logging.info("No new GHE data (current: %s), exiting.", previous_dt)
            sys.exit()
else:
    pickle_local_path = os.path.join(
        forecast_path, ingest_version, pickle_filename
    )
    if os.path.exists(pickle_local_path):
        with open(pickle_local_path, "rb") as fh:
            previous_dt = pickle.load(fh)
        if previous_dt >= latest_dt:
            logging.info("No new GHE data (current: %s), exiting.", previous_dt)
            sys.exit()

# 3. Load observations (newest → oldest)
logging.info("Downloading %d GHE observation(s) …", len(ghe_files))
observations = []
lats_ghe = None
lons_ghe = None
for path, dt in ghe_files:
    rr, lats, lons = _load_rain_rate(path)
    observations.append(rr)
    if lats_ghe is None:
        lats_ghe = lats
        lons_ghe = lons

# 4. Run the pysteps nowcast
nowcast_frames = _compute_nowcast(observations, n_frames=NOWCAST_FRAMES)

# 5. Build the full time series: current observation + nowcast frames
#    Shape will be (1 + NOWCAST_FRAMES, lat, lon) = (9, lat, lon)
current_rr = observations[0].astype(np.float32)  # most recent observation
nowcast_rr = nowcast_frames.astype(np.float32)

# Compute Unix timestamps for each frame
frame_times = np.array(
    [
        int(
            (latest_dt + timedelta(minutes=i * GHE_INTERVAL_MIN)).timestamp()
        )
        for i in range(NOWCAST_FRAMES + 1)
    ],
    dtype=np.float32,
)

n_times = NOWCAST_FRAMES + 1  # 9 frames total
n_lat = current_rr.shape[0]
n_lon = current_rr.shape[1]
n_vars = len(zarr_vars)  # 2: time + rain_rate

# Build (vars, times, lat, lon) array
# Dim 0: variable index (0=time, 1=rain_rate)
# Dim 1: time index (0=current, 1-8=nowcast at 15-min steps)
logging.info(
    "Building zarr array: (%d vars, %d times, %d lat, %d lon) …",
    n_vars,
    n_times,
    n_lat,
    n_lon,
)

# Time variable: broadcast timestamps to full spatial extent
time_grid = np.broadcast_to(
    frame_times[np.newaxis, :, np.newaxis, np.newaxis],
    (1, n_times, n_lat, n_lon),
).astype(np.float32)

# Rain rate: stack current + nowcast along time axis
rr_all = np.concatenate(
    [current_rr[np.newaxis, np.newaxis], nowcast_rr[np.newaxis]],
    axis=1,
).astype(np.float32)  # shape (1, n_times, n_lat, n_lon)

# Combine into (n_vars, n_times, n_lat, n_lon)
dask_var_array = da.from_array(
    np.concatenate([time_grid, rr_all], axis=0),
    chunks=(n_vars, n_times, process_chunk, process_chunk),
)

# Clip rain rate to valid range (leave time dim untouched)
rr_slice = dask_var_array[1]  # shape (n_times, n_lat, n_lon)
rr_slice = da.clip(rr_slice, 0.0, VALID_DATA_MAX)
dask_var_array = da.concatenate(
    [dask_var_array[0:1], rr_slice[np.newaxis]], axis=0
)

# Re-chunk to final chunk size for efficient spatial reads
dask_var_array = dask_var_array.rechunk(
    chunks=(n_vars, n_times, final_chunk, final_chunk)
)

# Pad spatial dimensions to multiples of chunk size
dask_var_array = pad_to_chunk_size(dask_var_array, final_chunk)

# 6. Write to zarr
if save_type == "S3":
    zarr_store = zarr.storage.ZipStore(
        os.path.join(forecast_process_dir, "GHE.zarr.zip"),
        mode="a",
        compression=0,
    )
else:
    zarr_store = zarr.storage.LocalStore(
        os.path.join(forecast_process_dir, "GHE.zarr")
    )

zarr_array = zarr.create_array(
    store=zarr_store,
    shape=(
        n_vars,
        n_times,
        dask_var_array.shape[2],
        dask_var_array.shape[3],
    ),
    chunks=(n_vars, n_times, final_chunk, final_chunk),
    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3),
    dtype="float32",
    overwrite=True,
)

logging.info("Writing GHE zarr store …")
with dask.config.set(scheduler="threads", num_workers=zarr_store_workers):
    dask_var_array.to_zarr(zarr_array, overwrite=True, compute=True)

close_store(zarr_store)
logging.info("GHE zarr store written.")

# 7. Save timestamp pickle and upload / copy to final location
if save_type == "S3":
    local_pickle = os.path.join(forecast_process_dir, pickle_filename)
    with open(local_pickle, "wb") as fh:
        pickle.dump(latest_dt, fh)

    s3.put_file(
        os.path.join(forecast_process_dir, "GHE.zarr.zip"),
        os.path.join(forecast_path, ingest_version, "GHE.zarr.zip"),
    )
    s3.put_file(
        local_pickle,
        os.path.join(forecast_path, ingest_version, pickle_filename),
    )
    logging.info("GHE zarr.zip and pickle uploaded to S3.")
else:
    local_pickle = os.path.join(forecast_process_dir, pickle_filename)
    with open(local_pickle, "wb") as fh:
        pickle.dump(latest_dt, fh)
    shutil.move(
        local_pickle,
        os.path.join(forecast_path, ingest_version, pickle_filename),
    )
    shutil.copytree(
        os.path.join(forecast_process_dir, "GHE.zarr"),
        os.path.join(forecast_path, ingest_version, "GHE.zarr"),
        dirs_exist_ok=True,
    )
    logging.info("GHE zarr and pickle moved to local storage.")

# Cleanup
shutil.rmtree(forecast_process_dir)
logging.info("Cleaned up temporary directory.")

t1 = time.time()
logging.info("GHE ingest completed in %.1f s.", t1 - t0)
