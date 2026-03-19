# US bounding box for cull
US_BOUNDING_BOX = {
    "top": 49.3457868,  # north lat
    "left": -124.7844079,  # west long
    "right": -66.9513812,  # east long
    "bottom": 24.7433195,  # south lat
}
# Grid boundary constants for model index checks
HRRR_X_MIN = 1
HRRR_Y_MIN = 1
HRRR_X_MAX = 1799
HRRR_Y_MAX = 1059

NBM_X_MIN = 1
NBM_Y_MIN = 1
NBM_X_MAX = 2344
NBM_Y_MAX = 1596

RTMA_RU_X_MIN = 1
RTMA_RU_Y_MIN = 1
RTMA_RU_X_MAX = 2344
RTMA_RU_Y_MAX = 1596
RTMA_RU_CENTRAL_LONG = 265.0
RTMA_RU_CENTRAL_LAT = 25.0
RTMA_RU_PARALLEL = 25.0
RTMA_RU_AXIS = 6371200
RTMA_RU_MIN_X = -3271152.8
RTMA_RU_MIN_Y = -263793.46
RTMA_RU_DELTA = 2539.703000

# MRMS CONUS regular lat/lon grid constants
# Coverage: 20.005°N–54.995°N, 129.995°W–60.005°W
# Resolution: 0.01° × 0.01°, grid size: 3500 rows × 7000 columns
MRMS_LAT_MIN = 20.005   # southernmost latitude (°N)
MRMS_LAT_MAX = 54.995   # northernmost latitude (°N)
MRMS_LON_MIN = -129.995  # westernmost longitude (°W, negative)
MRMS_LON_MAX = -60.005   # easternmost longitude (°W, negative)
MRMS_DELTA = 0.01        # grid spacing in degrees
MRMS_NY = 3500           # number of latitude rows
MRMS_NX = 7000           # number of longitude columns
