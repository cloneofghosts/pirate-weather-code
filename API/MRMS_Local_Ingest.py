# %% MRMS (Multi-Radar/Multi-Sensor) Processing script using Herbie and Zarr
# Ingests current MRMS observations: precipitation rate, precipitation type flags,
# reflectivity at 1 km AGL, and lightning flash rate density.
# Data updates every 2 minutes; this script checks for new data and exits if
# the stored version is already current.

# %% Import modules
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
from herbie import Herbie
from herbie.fast import Herbie_latest

from API.constants.shared_const import INGEST_VERSION_STR
from API.ingest_utils import (
    CHUNK_SIZES,
    FINAL_CHUNK_SIZES,
    VALID_DATA_MAX,
    close_store,
    configure_zarr_limits,
    make_herbie_save_dir,
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

s3 = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)
tune_nofile_limit()
zarr_store_workers, zarr_async_concurrency = configure_zarr_limits(
    zarr_store_workers, zarr_async_concurrency
)

# Create new directory for processing if it does not exist
if not os.path.exists(forecast_process_dir):
    os.makedirs(forecast_process_dir)
else:
    # If it does exist, remove it to start fresh
    shutil.rmtree(forecast_process_dir)
    os.makedirs(forecast_process_dir)

if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

if save_type == "Download":
    if not os.path.exists(forecast_path + "/" + ingest_version):
        os.makedirs(forecast_path + "/" + ingest_version)

herbie_save_dir = make_herbie_save_dir(tmp_dir)

# %% Define base time from the most recent MRMS run
# MRMS updates every 2 minutes; search the last 5 runs
t0 = time.time()

latest_run = Herbie_latest(
    model="mrms",
    n=5,
    freq="2min",
    product="PrecipRate",
    verbose=True,
    priority=["aws", "nomdas"],
    save_dir=herbie_save_dir,
)

base_time = latest_run.date
logging.info(f"Checking for new MRMS data for base time: {base_time}")

# Check if this is newer than the current stored file
if save_type == "S3":
    if s3.exists(forecast_path + "/" + ingest_version + "/MRMS.time.pickle"):
        with s3.open(
            forecast_path + "/" + ingest_version + "/MRMS.time.pickle", "rb"
        ) as f:
            previous_base_time = pickle.load(f)
        if previous_base_time >= base_time:
            logging.info("No Update to MRMS, ending")
            sys.exit()
else:
    if os.path.exists(forecast_path + "/" + ingest_version + "/MRMS.time.pickle"):
        with open(
            forecast_path + "/" + ingest_version + "/MRMS.time.pickle", "rb"
        ) as file:
            previous_base_time = pickle.load(file)
        if previous_base_time >= base_time:
            logging.info("No Update to MRMS, ending")
            sys.exit()

# Zarr variable order: time, precip_rate, precip_flag, refl_1km, lightning
zarr_vars = (
    "time",
    "unknown",   # PrecipRate (mm/hr) – GRIB shortName varies by product
    "unknown",   # PrecipFlag categorical type
    "unknown",   # Reflectivity at 1 km AGL (dBZ)
    "unknown",   # Lightning flash rate density (flashes/km²/min)
)

# %% Download MRMS PrecipRate
logging.info("Downloading MRMS PrecipRate ...")
fh_precip_rate = Herbie(
    base_time,
    model="mrms",
    product="PrecipRate",
    verbose=False,
    priority=["aws", "nomdas"],
    save_dir=herbie_save_dir,
)
fh_precip_rate.download(verbose=True)

# %% Download MRMS PrecipFlag
logging.info("Downloading MRMS PrecipFlag ...")
fh_precip_flag = Herbie(
    base_time,
    model="mrms",
    product="PrecipFlag",
    verbose=False,
    priority=["aws", "nomdas"],
    save_dir=herbie_save_dir,
)
fh_precip_flag.download(verbose=True)

# %% Download MRMS Reflectivity at 1 km AGL
logging.info("Downloading MRMS Reflectivity at 1 km AGL ...")
fh_refl = Herbie(
    base_time,
    model="mrms",
    product="MergedReflectivityQC_01.00",
    verbose=False,
    priority=["aws", "nomdas"],
    save_dir=herbie_save_dir,
)
fh_refl.download(verbose=True)

# %% Download MRMS LightningFlashRateDensity
logging.info("Downloading MRMS LightningFlashRateDensity ...")
fh_lightning = Herbie(
    base_time,
    model="mrms",
    product="LightningFlashRateDensity",
    verbose=False,
    priority=["aws", "nomdas"],
    save_dir=herbie_save_dir,
)
fh_lightning.download(verbose=True)

logging.info("All MRMS GRIB files downloaded successfully.")

# %% Load datasets into xarray
ds_rate = fh_precip_rate.xarray()
ds_flag = fh_precip_flag.xarray()
ds_refl = fh_refl.xarray()
ds_lightning = fh_lightning.xarray()

# Ensure they are all single DataArrays / Datasets with a consistent grid
if isinstance(ds_rate, list):
    ds_rate = ds_rate[0]
if isinstance(ds_flag, list):
    ds_flag = ds_flag[0]
if isinstance(ds_refl, list):
    ds_refl = ds_refl[0]
if isinstance(ds_lightning, list):
    ds_lightning = ds_lightning[0]

# %% Extract 2D arrays
# MRMS GRIB2 typically uses (y, x) or (latitude, longitude) dimensions
def _get_2d(ds):
    """Return the first 2D data variable as a numpy float32 array."""
    if isinstance(ds, xr.Dataset):
        var_name = list(ds.data_vars)[0]
        arr = ds[var_name].values
    else:
        arr = ds.values
    # Drop any leading dimensions of size 1
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(np.float32)


arr_rate = _get_2d(ds_rate)
arr_flag = _get_2d(ds_flag)
arr_refl = _get_2d(ds_refl)
arr_lightning = _get_2d(ds_lightning)

logging.info(f"MRMS grid shape: {arr_rate.shape}")

# Validate data ranges
arr_rate = np.clip(arr_rate, 0, VALID_DATA_MAX)
arr_flag = np.clip(arr_flag, 0, 200)
arr_refl = np.clip(arr_refl, -30, 80)
arr_lightning = np.clip(arr_lightning, 0, VALID_DATA_MAX)

# Replace fill values (-999, -9999) with NaN
for arr in (arr_rate, arr_flag, arr_refl, arr_lightning):
    arr[arr < -100] = np.nan

# %% Build Unix timestamp
base_time_unix = np.int64(
    (np.datetime64(base_time, "s") - np.datetime64("1970-01-01T00:00:00", "s"))
    / np.timedelta64(1, "s")
)
logging.info(f"MRMS base time (Unix): {base_time_unix}")

ny, nx = arr_rate.shape

# %% Build the Zarr store
# Shape: (num_vars=5, num_times=1, ny, nx)
num_vars = 5
zarr_data = np.full((num_vars, 1, ny, nx), np.nan, dtype=np.float32)
zarr_data[0, 0, :, :] = base_time_unix   # time
zarr_data[1, 0, :, :] = arr_rate          # precip_rate (mm/hr)
zarr_data[2, 0, :, :] = arr_flag          # precip_flag (categorical)
zarr_data[3, 0, :, :] = arr_refl          # refl_1km (dBZ)
zarr_data[4, 0, :, :] = arr_lightning     # lightning (flashes/km²/min)

logging.info(
    f"Zarr data shape: {zarr_data.shape}, "
    f"non-NaN rate pixels: {np.sum(~np.isnan(zarr_data[1, 0]))}"
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
    s3.put(zarr_path, s3_zarr_path, recursive=True)

    with s3.open(
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
