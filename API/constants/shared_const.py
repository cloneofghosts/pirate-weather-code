"""
Shared constants
"""

import numpy as np

# Invalid data
MISSING_DATA = np.nan

# Minimum reflectivity threshold (dBZ)
REFC_THRESHOLD = 5.0

# Ingest version
INGEST_VERSION_STR = "v30"

# Convert Kelvin to Celsius
KELVIN_TO_CELSIUS = 273.15

HISTORY_PERIODS = {
    "NBM": 48,
    "HRRR": 48,
    "HRRR_6H": 48,
    "GFS": 288,  # GFS has a 12-day history, allowing 10 days of local retrievals. Beyond that is Google ERA5
    "GEFS": 48,
    "ECMWF": 48,
    "NBM_Fire": 48,
    "DWD_MOSMIX": 48,  # History period offset (like other models)
}

# MRMS lightning flash rate density threshold (flashes/km²/min) for thunderstorm detection.
# Any value at or above this threshold combined with active precipitation indicates a thunderstorm.
MRMS_LIGHTNING_THUNDERSTORM_THRESHOLD = 0.1
