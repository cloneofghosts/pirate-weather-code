# %% MRMS (Multi-Radar/Multi-Sensor) Processing script
# Downloads MRMS observations directly from the NOAA public AWS S3 bucket
# (noaa-mrms-pds) using anonymous S3 access.
#
# Products ingested (all CONUS):
#   - PrecipRate:                    Surface precipitation rate (mm/hr)
#   - PrecipFlag:                    Categorical precipitation type
#   - MergedCompositeReflectivityQC: Column-maximum composite reflectivity (dBZ);
#                                    full CONUS coverage (replaces limited-coverage
#                                    1-km AGL product)
#   - LightningFlashRateDensity:     Lightning flash rate density (flashes/km²/min)
#
# Data updates every 2 minutes.  The script checks whether the stored data is
# already current and exits early if no update is required.

# %% Import modules
import datetime
import gzip
import logging
import os
import pickle
import shutil
import sys
import time
import warnings

import numpy as np
import s3fs
import xarray as xr
import zarr

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    VALID_DATA_MAX,
    close_store,
    configure_zarr_limits,
    positive_int_env,
    tune_nofile_limit,
)

warnings.filterwarnings("ignore", "This pattern is interpreted")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# %% Setup paths and parameters
ingest_version = INGEST_VERSION_STR

forecast_process_dir = os.getenv(
    "forecast_process_dir", default="/mnt/nvme/data/MRMS"
)
tmp_dir = forecast_process_dir + "/Downloads"

forecast_path = os.getenv("forecast_path", default="/mnt/nvme/data/Prod/MRMS")

# Define the processing and final chunk size
process_chunk = CHUNK_SIZES["RTMA"]
final_chunk = FINAL_CHUNK_SIZES["RTMA"]

save_type = os.getenv("save_type", default="Download")
aws_access_key_id = os.environ.get("AWS_KEY", "")
aws_secret_access_key = os.environ.get("AWS_SECRET", "")
zarr_store_workers = positive_int_env("zarr_store_workers", 2)
zarr_async_concurrency = positive_int_env("zarr_async_concurrency", 2)

# Authenticated S3 client for the *output* bucket (if save_type == "S3")
s3_out = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

# Anonymous S3 client for the NOAA public MRMS bucket
s3_mrms = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-east-1"})

# MRMS public bucket constants
MRMS_BUCKET = "noaa-mrms-pds"

# Product definitions: {product_name: level_string}
# The level string appears in the filename, e.g. PrecipRate_00.00_YYYYMMDD-HHMMSS
MRMS_PRODUCT_LEVELS = {
    "PrecipRate": "00.00",
    "PrecipFlag": "00.00",
    "MergedCompositeReflectivityQC": "00.50",
    "LightningFlashRateDensity": "00.00",
}

tune_nofile_limit()
zarr_store_workers, zarr_async_concurrency = configure_zarr_limits(
    zarr_store_workers, zarr_async_concurrency
)

# Create working directories
if not os.path.exists(forecast_process_dir):
    os.makedirs(forecast_process_dir)
else:
    shutil.rmtree(forecast_process_dir)
    os.makedirs(forecast_process_dir)

if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

if save_type == "Download":
    if not os.path.exists(forecast_path + "/" + ingest_version):
        os.makedirs(forecast_path + "/" + ingest_version)


# %% Helper functions


def find_latest_mrms_s3_path(product: str, level: str) -> tuple[str, str]:
    """
    Find the most recently available MRMS GRIB2 file for *product* on the NOAA
    public S3 bucket.

    Searches today's and yesterday's date directories (UTC) and returns the path
    of the newest matching file together with the UTC timestamp encoded in the
    filename.

    Args:
        product: MRMS product name, e.g. ``"PrecipRate"``.
        level:   Level string used in the filename, e.g. ``"00.00"``.

    Returns:
        Tuple of (s3_path, timestamp_utc_str) where timestamp_utc_str is
        ``"YYYYMMDD-HHMMSS"``.

    Raises:
        FileNotFoundError: If no matching file is found in either date directory.
    """
    now_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    pattern = f"{product}_{level}_"

    for days_back in range(0, 2):
        date = now_utc - datetime.timedelta(days=days_back)
        date_str = date.strftime("%Y%m%d")
        prefix = f"{MRMS_BUCKET}/CONUS/{product}/{date_str}/"
        try:
            files = s3_mrms.ls(prefix)
        except Exception:
            continue

        matching = sorted(
            [f for f in files if pattern in f and f.endswith(".grib2.gz")]
        )
        if matching:
            latest = matching[-1]
            # Extract timestamp from filename: <product>_<level>_<YYYYMMDD-HHMMSS>.grib2.gz
            ts_part = os.path.basename(latest).split(pattern)[-1].replace(
                ".grib2.gz", ""
            )
            return latest, ts_part

    raise FileNotFoundError(
        f"No MRMS {product} files found in the last 2 days on s3://{MRMS_BUCKET}"
    )


def download_mrms_grib(s3_path: str, out_dir: str) -> str:
    """
    Download an MRMS .grib2.gz file from S3, decompress it, and return the
    path to the resulting .grib2 file.

    Args:
        s3_path: Full S3 path (without ``s3://`` prefix), e.g.
                 ``"noaa-mrms-pds/CONUS/PrecipRate/20250101/PrecipRate_00.00_20250101-120000.grib2.gz"``.
        out_dir: Local directory where the decompressed file will be written.

    Returns:
        Local path to the decompressed .grib2 file.
    """
    filename = os.path.basename(s3_path).replace(".grib2.gz", ".grib2")
    local_path = os.path.join(out_dir, filename)

    logging.info(f"Downloading s3://{s3_path} ...")
    with s3_mrms.open(s3_path, "rb") as f_gz:
        gz_bytes = f_gz.read()

    logging.info(f"Decompressing {filename} ...")
    with open(local_path, "wb") as f_out:
        f_out.write(gzip.decompress(gz_bytes))

    return local_path


def read_grib2_to_array(grib_path: str) -> np.ndarray:
    """
    Open a GRIB2 file with cfgrib/xarray and return the first data variable as
    a 2D float32 NumPy array with shape ``(ny, nx)``.

    Args:
        grib_path: Local path to a .grib2 file.

    Returns:
        2D float32 array.
    """
    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    )
    var_name = list(ds.data_vars)[0]
    arr = ds[var_name].values
    # Drop any leading size-1 dimensions
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(np.float32)


# %% Determine the most recent base time

t0 = time.time()

logging.info("Searching for latest MRMS PrecipRate ...")
try:
    rate_s3_path, base_ts_str = find_latest_mrms_s3_path(
        "PrecipRate", MRMS_PRODUCT_LEVELS["PrecipRate"]
    )
except FileNotFoundError as exc:
    logging.error(str(exc))
    sys.exit(1)

# Parse base timestamp
base_time = datetime.datetime.strptime(base_ts_str, "%Y%m%d-%H%M%S")
logging.info(f"Latest MRMS base time: {base_time} UTC")

# %% Check if update is needed

if save_type == "S3":
    pickle_key = forecast_path + "/" + ingest_version + "/MRMS.time.pickle"
    if s3_out.exists(pickle_key):
        with s3_out.open(pickle_key, "rb") as f:
            previous_base_time = pickle.load(f)
        if previous_base_time >= base_time:
            logging.info("No Update to MRMS, ending")
            sys.exit()
else:
    pickle_path = forecast_path + "/" + ingest_version + "/MRMS.time.pickle"
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            previous_base_time = pickle.load(f)
        if previous_base_time >= base_time:
            logging.info("No Update to MRMS, ending")
            sys.exit()

# %% Download all four products

logging.info("Downloading MRMS products from S3 ...")

arr_rate = None
arr_flag = None
arr_refl = None
arr_lightning = None

products_to_download = {
    "PrecipRate": ("PrecipRate", MRMS_PRODUCT_LEVELS["PrecipRate"]),
    "PrecipFlag": ("PrecipFlag", MRMS_PRODUCT_LEVELS["PrecipFlag"]),
    "MergedCompositeReflectivityQC": (
        "MergedCompositeReflectivityQC",
        MRMS_PRODUCT_LEVELS["MergedCompositeReflectivityQC"],
    ),
    "LightningFlashRateDensity": (
        "LightningFlashRateDensity",
        MRMS_PRODUCT_LEVELS["LightningFlashRateDensity"],
    ),
}

for key, (product, level) in products_to_download.items():
    try:
        s3_path, _ = find_latest_mrms_s3_path(product, level)
        grib_path = download_mrms_grib(s3_path, tmp_dir)
        arr = read_grib2_to_array(grib_path)
        logging.info(f"{product}: shape={arr.shape}")
        if key == "PrecipRate":
            arr_rate = arr
        elif key == "PrecipFlag":
            arr_flag = arr
        elif key == "MergedCompositeReflectivityQC":
            arr_refl = arr
        elif key == "LightningFlashRateDensity":
            arr_lightning = arr
    except Exception:
        logging.exception(f"Failed to download/process {product}; using NaN fill")

logging.info("All MRMS products processed.")

# If PrecipRate is missing we cannot proceed (it defines the grid shape)
if arr_rate is None:
    logging.error("PrecipRate download failed – cannot continue")
    sys.exit(1)

ny, nx = arr_rate.shape

# Create NaN fill arrays for any products that failed
if arr_flag is None:
    arr_flag = np.full((ny, nx), np.nan, dtype=np.float32)
if arr_refl is None:
    arr_refl = np.full((ny, nx), np.nan, dtype=np.float32)
if arr_lightning is None:
    arr_lightning = np.full((ny, nx), np.nan, dtype=np.float32)

# %% Validate and clean data

# Clip and mask invalid values (MRMS fill is typically -999)
arr_rate = np.where(arr_rate < -100, np.nan, np.clip(arr_rate, 0, VALID_DATA_MAX))
arr_flag = np.where(arr_flag < -100, np.nan, np.clip(arr_flag, -3, 200))
arr_refl = np.where(arr_refl < -100, np.nan, np.clip(arr_refl, -30, 80))
arr_lightning = np.where(
    arr_lightning < -100, np.nan, np.clip(arr_lightning, 0, VALID_DATA_MAX)
)

# %% Build Unix timestamp

base_time_unix = np.int64(
    (
        np.datetime64(base_time, "s")
        - np.datetime64("1970-01-01T00:00:00", "s")
    )
    / np.timedelta64(1, "s")
)
logging.info(f"MRMS base time (Unix): {base_time_unix}")

# %% Build the Zarr store
# Shape: (num_vars=5, 1, ny, nx)
# Variable order: time, precip_rate, precip_flag, refl_comp, lightning
num_vars = 5
zarr_data = np.full((num_vars, 1, ny, nx), np.nan, dtype=np.float32)
zarr_data[0, 0, :, :] = base_time_unix  # time
zarr_data[1, 0, :, :] = arr_rate        # precip_rate (mm/hr)
zarr_data[2, 0, :, :] = arr_flag        # precip_flag (categorical)
zarr_data[3, 0, :, :] = arr_refl        # refl_comp (dBZ)
zarr_data[4, 0, :, :] = arr_lightning   # lightning (flashes/km²/min)

logging.info(
    f"Zarr data shape: {zarr_data.shape}, "
    f"non-NaN rate pixels: {int(np.sum(~np.isnan(zarr_data[1, 0])))}"
)

# %% Save to Zarr

zarr_path = os.path.join(forecast_process_dir, "MRMS.zarr")

compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
zarr_out = zarr.open(
    zarr.storage.LocalStore(zarr_path),
    mode="w",
    shape=zarr_data.shape,
    chunks=(num_vars, 1, final_chunk, final_chunk),
    dtype="f4",
    compressor=compressor,
)
zarr_out[:] = zarr_data
close_store(zarr_out)

logging.info(f"Zarr store written to: {zarr_path}")

# %% Move to final location or upload to S3

if save_type == "S3":
    s3_zarr_path = forecast_path + "/" + ingest_version + "/MRMS.zarr"
    logging.info(f"Uploading MRMS zarr to S3: {s3_zarr_path}")
    s3_out.put(zarr_path, s3_zarr_path, recursive=True)

    with s3_out.open(
        forecast_path + "/" + ingest_version + "/MRMS.time.pickle", "wb"
    ) as f:
        pickle.dump(base_time, f)
    logging.info("MRMS zarr uploaded to S3 and timestamp saved.")

else:
    final_zarr_path = forecast_path + "/" + ingest_version + "/MRMS.zarr"
    if os.path.exists(final_zarr_path):
        shutil.rmtree(final_zarr_path)
    shutil.copytree(zarr_path, final_zarr_path)

    with open(
        forecast_path + "/" + ingest_version + "/MRMS.time.pickle", "wb"
    ) as file:
        pickle.dump(base_time, file)
    logging.info(f"MRMS zarr copied to: {final_zarr_path}")

t_elapsed = time.time() - t0
logging.info(f"MRMS ingest completed in {t_elapsed:.1f}s")
