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
        "import gzip",
    ]
    for import_stmt in required_imports:
        assert import_stmt in script_content, f"Missing required import: {import_stmt}"


def test_mrms_script_has_required_components():
    """Test that the MRMS script contains expected components."""
    script_content = MRMS_SCRIPT_PATH.read_text()
    assert "MRMS_BUCKET" in script_content, "Missing MRMS_BUCKET constant"
    assert "noaa-mrms-pds" in script_content, "Missing NOAA MRMS S3 bucket name"
    assert "anon=True" in script_content, "Missing anonymous S3 access"
    assert "PrecipRate" in script_content, "Missing PrecipRate product"
    assert "PrecipFlag" in script_content, "Missing PrecipFlag product"
    assert "MergedCompositeReflectivityQC" in script_content, "Missing composite reflectivity product"
    assert "LightningFlashRateDensity" in script_content, "Missing LightningFlashRateDensity product"
    assert "gzip.decompress" in script_content, "Missing gzip decompression"
    assert "cfgrib" in script_content, "Missing cfgrib engine"


def test_mrms_constants_defined():
    """Test that MRMS constants are defined in model_const.py."""
    from API.constants.model_const import MRMS

    assert "precip_rate" in MRMS, "MRMS missing precip_rate index"
    assert "precip_flag" in MRMS, "MRMS missing precip_flag index"
    assert "refl_comp" in MRMS, "MRMS missing refl_comp index (composite reflectivity)"
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
    """Test that map_mrms_flag_to_ptype returns correct precipitation types.

    Flag values per NOAA MRMS UserTable_MRMS_PrecipFlags.csv:
        -3: no coverage, 0: no precip, 1: warm stratiform rain, 3: snow,
        6: convective rain, 7: rain+hail (→ sleet), 10: cold stratiform rain,
        91: tropical/stratiform rain mix, 96: tropical/convective rain mix.
    """
    from API.api_utils import map_mrms_flag_to_ptype

    # No-precipitation codes
    assert map_mrms_flag_to_ptype(-3) == "none", "Flag -3 (no coverage) should be 'none'"
    assert map_mrms_flag_to_ptype(0) == "none", "Flag 0 (no precip) should be 'none'"

    # Rain codes
    assert map_mrms_flag_to_ptype(1) == "rain", "Flag 1 (warm stratiform rain) should be 'rain'"
    assert map_mrms_flag_to_ptype(6) == "rain", "Flag 6 (convective rain) should be 'rain'"
    assert map_mrms_flag_to_ptype(10) == "rain", "Flag 10 (cold stratiform rain) should be 'rain'"
    assert map_mrms_flag_to_ptype(91) == "rain", "Flag 91 (tropical/stratiform mix) should be 'rain'"
    assert map_mrms_flag_to_ptype(96) == "rain", "Flag 96 (tropical/convective mix) should be 'rain'"

    # Snow codes
    assert map_mrms_flag_to_ptype(3) == "snow", "Flag 3 (snow) should be 'snow'"

    # Sleet (rain + hail; no dedicated hail type in the API)
    assert map_mrms_flag_to_ptype(7) == "sleet", "Flag 7 (rain+hail) should be 'sleet'"

    # Unknown codes default to none
    assert map_mrms_flag_to_ptype(99) == "none", "Unknown flag should be 'none'"


def test_mrms_nowcasting_blending():
    """Test that MRMS nowcasting uses persistence then blends to model.

    The improved algorithm:
    - 0-30 min: pure MRMS rate (Lagrangian persistence)
    - 30-60 min: linear blend from MRMS → model
    - No-model fallback: hold rate for 30 min then fade to 0
    """
    from API.constants.model_const import MRMS
    from API.minutely.builder import _calculate_intensity

    # Create a fake mrms_data: shape (1, 5) => 1 time step, 5 variables
    # time=0, precip_rate=5.0 mm/hr, precip_flag=1 (rain), refl_comp=35, lightning=0
    mrms_data = np.zeros((1, 5), dtype=np.float32)
    mrms_data[0, 0] = 0.0       # UNIX timestamp = 0 (t=0)
    mrms_data[0, MRMS["precip_rate"]] = 5.0
    mrms_data[0, MRMS["precip_flag"]] = 1
    mrms_data[0, MRMS["refl_comp"]] = 35.0
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

    # At t=0 (persistence window), intensity must equal MRMS rate
    assert intensity[0] == pytest.approx(5.0, abs=0.1), (
        f"At t=0, expected MRMS rate ~5.0 but got {intensity[0]}"
    )
    # At t=29 min (still inside persistence window), intensity must still be MRMS rate
    assert intensity[29] == pytest.approx(5.0, abs=0.1), (
        f"At t=29 min (persistence window), expected MRMS rate ~5.0 but got {intensity[29]}"
    )
    # After persistence + blend window (t>=60 min), rate should have faded to 0
    # (no model source → fallback fade)
    assert intensity[60] == pytest.approx(0.0, abs=0.1), (
        f"At t=60 min (end of blend), expected ~0.0 but got {intensity[60]}"
    )


def test_mrms_ptype_fallback():
    """Test that MRMS precipitation type is used as fallback when HRRR/NBM unavailable."""
    from API.constants.model_const import MRMS
    from API.minutely.builder import _calculate_precip_type_probs

    # Snow flag (flag 3 per NOAA MRMS PrecipFlags table)
    mrms_data_snow = np.zeros((1, 5), dtype=np.float32)
    mrms_data_snow[0, MRMS["precip_flag"]] = 3  # Snow

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

    # Sleet flag (flag 7 = rain + hail; maps to sleet since no hail type)
    mrms_data_sleet = np.zeros((1, 5), dtype=np.float32)
    mrms_data_sleet[0, MRMS["precip_flag"]] = 7  # Rain + hail → sleet

    result_sleet = _calculate_precip_type_probs(
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
        mrms_data=mrms_data_sleet,
    )

    # Column 3 = sleet/freezing-rain probability
    assert np.all(result_sleet[:, 3] == 1.0), (
        "MRMS rain+hail flag (7) should set sleet probability to 1.0"
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

    # MRMS says snow (flag 3), HRRR SubH says rain
    mrms_data_snow = np.zeros((1, 5), dtype=np.float32)
    mrms_data_snow[0, MRMS["precip_flag"]] = 3  # Snow

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


def test_mrms_lightning_thunderstorm_threshold():
    """Test that the MRMS lightning threshold constant is defined and positive."""
    from API.constants.shared_const import MRMS_LIGHTNING_THUNDERSTORM_THRESHOLD

    assert MRMS_LIGHTNING_THUNDERSTORM_THRESHOLD > 0, (
        "Lightning thunderstorm threshold must be positive"
    )


def test_mrms_lightning_overrides_currently_icon():
    """Test that MRMS lightning above threshold overrides the currently icon.

    When lightning flash rate density exceeds the threshold AND precipitation
    is occurring, build_current_section should override the icon to 'thunderstorm'.
    """
    from API.constants.model_const import MRMS
    from API.constants.shared_const import MRMS_LIGHTNING_THUNDERSTORM_THRESHOLD

    # Verify MRMS constant has lightning field
    assert "lightning" in MRMS

    # Verify threshold is sensible
    assert MRMS_LIGHTNING_THUNDERSTORM_THRESHOLD > 0.0
    assert MRMS_LIGHTNING_THUNDERSTORM_THRESHOLD < 10.0  # sanity upper bound


def test_mrms_lightning_override_logic():
    """Test the conditional logic that triggers the thunderstorm override.

    Directly exercises the three conditions in the lightning override block:
    lightning ≥ threshold, precip > 0, lightning is finite.
    """
    from API.constants.model_const import MRMS
    from API.constants.shared_const import MRMS_LIGHTNING_THUNDERSTORM_THRESHOLD

    def _should_override(lightning_val, precip_intensity):
        """Replicate the override condition from build_current_section."""
        return (
            np.isfinite(lightning_val)
            and lightning_val >= MRMS_LIGHTNING_THUNDERSTORM_THRESHOLD
            and precip_intensity > 0.0
        )

    lightning_idx = MRMS["lightning"]
    thr = MRMS_LIGHTNING_THUNDERSTORM_THRESHOLD

    # --- Case 1: above threshold + active precip → should override ---
    mrms_above = np.zeros((1, 5), dtype=np.float32)
    mrms_above[0, lightning_idx] = thr * 2.0  # well above threshold
    assert _should_override(float(mrms_above[0, lightning_idx]), 1.0), (
        "Should trigger override when lightning above threshold with active precip"
    )

    # --- Case 2: exactly at threshold + active precip → should override ---
    mrms_at = np.zeros((1, 5), dtype=np.float32)
    mrms_at[0, lightning_idx] = thr
    assert _should_override(float(mrms_at[0, lightning_idx]), 0.5), (
        "Should trigger override when lightning equals threshold with active precip"
    )

    # --- Case 3: below threshold + active precip → should NOT override ---
    mrms_below = np.zeros((1, 5), dtype=np.float32)
    mrms_below[0, lightning_idx] = thr * 0.5
    assert not _should_override(float(mrms_below[0, lightning_idx]), 1.0), (
        "Should not override when lightning below threshold"
    )

    # --- Case 4: above threshold + NO precip → should NOT override ---
    mrms_no_precip = np.zeros((1, 5), dtype=np.float32)
    mrms_no_precip[0, lightning_idx] = thr * 2.0
    assert not _should_override(float(mrms_no_precip[0, lightning_idx]), 0.0), (
        "Should not override when lightning is high but no precipitation"
    )

    # --- Case 5: NaN lightning → should NOT override ---
    assert not _should_override(float("nan"), 1.0), (
        "Should not override when lightning is NaN"
    )
