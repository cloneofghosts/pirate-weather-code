"""Tests for MRMS ingest script and integration."""

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

MRMS_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "API" / "MRMS_Local_Ingest.py"
)


def test_mrms_script_exists():
    """Test that MRMS_Local_Ingest.py exists."""
    assert MRMS_SCRIPT_PATH.exists(), f"MRMS script not found at {MRMS_SCRIPT_PATH}"
    assert MRMS_SCRIPT_PATH.is_file(), "MRMS script is not a file"


def test_mrms_script_is_valid_python():
    """Test that MRMS_Local_Ingest.py is valid Python syntax."""
    script_content = MRMS_SCRIPT_PATH.read_text()
    try:
        ast.parse(script_content)
    except SyntaxError as e:
        pytest.fail(f"MRMS script has invalid Python syntax: {e}")


def test_mrms_script_python_check():
    """Test that the MRMS script passes py_compile."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(MRMS_SCRIPT_PATH)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script failed to compile: {result.stderr}"


def test_mrms_script_has_required_imports():
    """Test that the MRMS script has all required imports."""
    script_content = MRMS_SCRIPT_PATH.read_text()
    required_imports = [
        "import numpy",
        "import s3fs",
        "import xarray",
        "import zarr",
        "from herbie import Herbie",
        "from herbie.fast import Herbie_latest",
    ]
    for import_stmt in required_imports:
        assert import_stmt in script_content, f"Missing required import: {import_stmt}"


def test_mrms_script_has_required_components():
    """Test that the MRMS script contains expected components."""
    script_content = MRMS_SCRIPT_PATH.read_text()
    assert "zarr_vars" in script_content, "Missing zarr_vars definition"
    assert "Herbie_latest" in script_content, "Missing Herbie_latest usage"
    assert "base_time" in script_content, "Missing base_time variable"
    assert "PrecipRate" in script_content, "Missing PrecipRate product"
    assert "PrecipFlag" in script_content, "Missing PrecipFlag product"
    assert "Reflectivity" in script_content, "Missing Reflectivity product"
    assert "Lightning" in script_content, "Missing Lightning product"


def test_mrms_constants_defined():
    """Test that MRMS constants are defined in model_const.py."""
    from API.constants.model_const import MRMS

    assert "precip_rate" in MRMS, "MRMS missing precip_rate index"
    assert "precip_flag" in MRMS, "MRMS missing precip_flag index"
    assert "refl_1km" in MRMS, "MRMS missing refl_1km index"
    assert "lightning" in MRMS, "MRMS missing lightning index"
    # Indices must be non-zero (variable index 0 in the zarr array is reserved for the
    # time variable; all other variables must use indices ≥ 1 to avoid collision)
    for key, idx in MRMS.items():
        assert idx > 0, f"MRMS[{key!r}] = {idx}, must be > 0 (zarr variable index 0 is time)"


def test_mrms_grid_constants_defined():
    """Test that MRMS grid constants are defined in grid_const.py."""
    from API.constants.grid_const import (
        MRMS_DELTA,
        MRMS_LAT_MAX,
        MRMS_LAT_MIN,
        MRMS_LON_MAX,
        MRMS_LON_MIN,
        MRMS_NX,
        MRMS_NY,
    )

    assert MRMS_LAT_MIN < MRMS_LAT_MAX, "MRMS_LAT_MIN should be < MRMS_LAT_MAX"
    assert MRMS_LON_MIN < MRMS_LON_MAX, "MRMS_LON_MIN should be < MRMS_LON_MAX"
    assert MRMS_DELTA > 0, "MRMS_DELTA should be positive"
    # Validate that grid dimensions match lat/lon range
    expected_ny = round((MRMS_LAT_MAX - MRMS_LAT_MIN) / MRMS_DELTA) + 1
    expected_nx = round((MRMS_LON_MAX - MRMS_LON_MIN) / MRMS_DELTA) + 1
    assert abs(MRMS_NY - expected_ny) <= 1, (
        f"MRMS_NY={MRMS_NY} doesn't match expected {expected_ny}"
    )
    assert abs(MRMS_NX - expected_nx) <= 1, (
        f"MRMS_NX={MRMS_NX} doesn't match expected {expected_nx}"
    )


def test_map_mrms_flag_to_ptype():
    """Test that map_mrms_flag_to_ptype returns correct precipitation types."""
    from API.api_utils import map_mrms_flag_to_ptype

    assert map_mrms_flag_to_ptype(0) == "none", "Flag 0 should be 'none'"
    assert map_mrms_flag_to_ptype(1) == "rain", "Flag 1 (Rain) should be 'rain'"
    assert map_mrms_flag_to_ptype(2) == "hail", "Flag 2 (Hail) should be 'hail'"
    assert map_mrms_flag_to_ptype(3) == "rain", "Flag 3 (Big Drops) should be 'rain'"
    assert map_mrms_flag_to_ptype(4) == "hail", "Flag 4 (Rain+Hail) should be 'hail'"
    assert map_mrms_flag_to_ptype(5) == "hail", "Flag 5 (Rain+Hail) should be 'hail'"
    assert map_mrms_flag_to_ptype(7) == "snow", "Flag 7 (Graupel) should be 'snow'"
    assert map_mrms_flag_to_ptype(8) == "snow", "Flag 8 (Snow) should be 'snow'"
    assert map_mrms_flag_to_ptype(9) == "snow", "Flag 9 (Dry Snow) should be 'snow'"
    assert map_mrms_flag_to_ptype(10) == "snow", "Flag 10 (Wet Snow) should be 'snow'"
    assert map_mrms_flag_to_ptype(11) == "snow", "Flag 11 (Ice Crystals) should be 'snow'"
    assert map_mrms_flag_to_ptype(12) == "rain", "Flag 12 (Drizzle) should be 'rain'"
    assert map_mrms_flag_to_ptype(91) == "rain", "Flag 91 (Tropical) should be 'rain'"
    assert map_mrms_flag_to_ptype(96) == "none", "Flag 96 (Biological) should be 'none'"


def test_mrms_nowcasting_blending():
    """Test that MRMS nowcasting blending logic produces correct output."""
    from API.constants.model_const import MRMS
    from API.minutely.builder import _calculate_intensity

    # Create a fake mrms_data: shape (1, 5) => 1 time step, 5 variables
    # time=0, precip_rate=5.0 mm/hr, precip_flag=1 (rain), refl_1km=35, lightning=0
    mrms_data = np.zeros((1, 5), dtype=np.float32)
    mrms_data[0, 0] = 0.0       # UNIX timestamp = 0 (t=0)
    mrms_data[0, MRMS["precip_rate"]] = 5.0
    mrms_data[0, MRMS["precip_flag"]] = 1
    mrms_data[0, MRMS["refl_1km"]] = 35.0
    mrms_data[0, MRMS["lightning"]] = 0.0

    # 61 minutes starting at t=0, 1 minute apart
    minute_array_grib = np.arange(0, 61 * 60, 60, dtype=float)

    precipTypes = np.array(["rain"] * 61, dtype="U5")

    # Use MRMS with no other sources active
    source_list = ["mrms"]

    intensity, _, _ = _calculate_intensity(
        source_list=source_list,
        precipTypes=precipTypes,
        hrrrSubHInterpolation=None,
        nbmMinuteInterpolation=None,
        dwd_mosmix_MinuteInterpolation=None,
        ecmwfMinuteInterpolation=None,
        gefsMinuteInterpolation=None,
        gfsMinuteInterpolation=None,
        era5_MinuteInterpolation=None,
        mrms_data=mrms_data,
        minute_array_grib=minute_array_grib,
    )

    # At t=0, intensity should be mrms_rate * (1-0) + MISSING_DATA*0 = ~5.0
    # (since model_intensity is MISSING_DATA when no model source active)
    assert intensity[0] == pytest.approx(5.0, abs=0.1), (
        f"At t=0, expected MRMS rate ~5.0 but got {intensity[0]}"
    )
    # After 15 minutes (index 15), MRMS weight = 0 → intensity should approach MISSING_DATA or 0
    # (since there's no model source, the blend falls back to MRMS with decaying weight)


def test_mrms_ptype_fallback():
    """Test that MRMS precipitation type is used as fallback when HRRR/NBM unavailable."""
    from API.constants.model_const import MRMS
    from API.minutely.builder import _calculate_precip_type_probs

    # Snow flag
    mrms_data_snow = np.zeros((1, 5), dtype=np.float32)
    mrms_data_snow[0, MRMS["precip_flag"]] = 8  # Snow

    result = _calculate_precip_type_probs(
        source_list=["mrms"],
        hrrrSubHInterpolation=None,
        nbmMinuteInterpolation=None,
        dwd_mosmix_MinuteInterpolation=None,
        ecmwfMinuteInterpolation=None,
        gefsMinuteInterpolation=None,
        gfsMinuteInterpolation=None,
        era5_MinuteInterpolation=None,
        lat=40.0,
        lon=-90.0,
        mrms_data=mrms_data_snow,
    )

    # Column 1 = snow probability; should be 1.0 for all minutes
    assert np.all(result[:, 1] == 1.0), (
        "MRMS snow flag should set snow probability to 1.0"
    )

    # Rain flag
    mrms_data_rain = np.zeros((1, 5), dtype=np.float32)
    mrms_data_rain[0, MRMS["precip_flag"]] = 1  # Rain

    result_rain = _calculate_precip_type_probs(
        source_list=["mrms"],
        hrrrSubHInterpolation=None,
        nbmMinuteInterpolation=None,
        dwd_mosmix_MinuteInterpolation=None,
        ecmwfMinuteInterpolation=None,
        gefsMinuteInterpolation=None,
        gfsMinuteInterpolation=None,
        era5_MinuteInterpolation=None,
        lat=40.0,
        lon=-90.0,
        mrms_data=mrms_data_rain,
    )

    # Column 4 = rain probability
    assert np.all(result_rain[:, 4] == 1.0), (
        "MRMS rain flag should set rain probability to 1.0"
    )

    # No precip flag
    mrms_data_none = np.zeros((1, 5), dtype=np.float32)
    mrms_data_none[0, MRMS["precip_flag"]] = 0  # No precip

    result_none = _calculate_precip_type_probs(
        source_list=["mrms"],
        hrrrSubHInterpolation=None,
        nbmMinuteInterpolation=None,
        dwd_mosmix_MinuteInterpolation=None,
        ecmwfMinuteInterpolation=None,
        gefsMinuteInterpolation=None,
        gfsMinuteInterpolation=None,
        era5_MinuteInterpolation=None,
        lat=40.0,
        lon=-90.0,
        mrms_data=mrms_data_none,
    )

    # All zero when no precip
    assert np.all(result_none == 0.0), (
        "MRMS no-precip flag should result in all-zero probability columns"
    )


def test_mrms_not_used_when_hrrr_available():
    """Test that MRMS is NOT used when HRRR SubH is available (priority check)."""
    from API.constants.model_const import HRRR_SUBH, MRMS
    from API.minutely.builder import _calculate_precip_type_probs

    # MRMS says snow, HRRR SubH says rain
    mrms_data_snow = np.zeros((1, 5), dtype=np.float32)
    mrms_data_snow[0, MRMS["precip_flag"]] = 8  # Snow

    # Mock HRRR SubH: all rain (column 4 in InterTminute = rain in HRRR_SUBH mapping)
    n_minutes = 61
    hrrr_cols = max(HRRR_SUBH.values()) + 1
    hrrrSubH = np.zeros((n_minutes, hrrr_cols), dtype=np.float32)
    # HRRR_SUBH["rain"] corresponds to InterTminute[:, rain_col]
    hrrrSubH[:, HRRR_SUBH["rain"]] = 0.8  # 80% rain probability

    result = _calculate_precip_type_probs(
        source_list=["hrrrsubh", "mrms"],
        hrrrSubHInterpolation=hrrrSubH,
        nbmMinuteInterpolation=None,
        dwd_mosmix_MinuteInterpolation=None,
        ecmwfMinuteInterpolation=None,
        gefsMinuteInterpolation=None,
        gfsMinuteInterpolation=None,
        era5_MinuteInterpolation=None,
        lat=40.0,
        lon=-90.0,
        mrms_data=mrms_data_snow,
    )

    # Should use HRRR SubH values, not MRMS (snow should not dominate)
    assert np.all(result[:, 1] == 0.0), (
        "When HRRR SubH is available, MRMS snow should not override"
    )
    rain_col = HRRR_SUBH["rain"] - 7  # convert to InterTminute index
    assert np.all(result[:, rain_col] == pytest.approx(0.8)), (
        "HRRR SubH rain probability should be used when available"
    )


def test_zarr_stores_has_mrms():
    """Test that ZarrStores dataclass includes MRMS_Zarr field."""
    from API.io.zarr_reader import ZarrStores

    stores = ZarrStores()
    assert hasattr(stores, "MRMS_Zarr"), "ZarrStores should have MRMS_Zarr attribute"
    assert stores.MRMS_Zarr is None, "MRMS_Zarr should default to None"


def test_zarr_sources_has_mrms():
    """Test that ZarrSources dataclass includes mrms field."""
    # ZarrSources requires positional args; test only mrms default
    import dataclasses

    from API.request.grid_indexing import ZarrSources

    fields = {f.name: f for f in dataclasses.fields(ZarrSources)}
    assert "mrms" in fields, "ZarrSources should have mrms field"


def test_grid_indexing_result_has_mrms():
    """Test that GridIndexingResult dataclass includes MRMS fields."""
    import dataclasses

    from API.request.grid_indexing import GridIndexingResult

    field_names = {f.name for f in dataclasses.fields(GridIndexingResult)}
    assert "dataOut_mrms" in field_names
    assert "x_mrms" in field_names
    assert "y_mrms" in field_names
    assert "mrms_lat" in field_names
    assert "mrms_lon" in field_names
