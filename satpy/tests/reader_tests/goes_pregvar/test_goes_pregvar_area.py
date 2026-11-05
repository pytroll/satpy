"""Tests for the pre-GVAR VISSR/VAS AREA file handler (``goes_pregvar_area``)."""

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS
from pyresample.geometry import AreaDefinition, SwathDefinition

from satpy import Scene
from satpy.tests.utils import skip_numba_unstable_if_missing

try:
    from satpy.readers.goes_pregvar.goes_pregvar_area import GOESPreGVARAreaFileHandler
    from satpy.readers.goes_pregvar.goes_pregvar_base import AreaFile
except ImportError as err:
    if skip_numba_unstable_if_missing():
        pytest.skip(f"Numba is not compatible with unstable NumPy: {err!s}", allow_module_level=True)
    raise

from satpy.tests.reader_tests.goes_pregvar.fake_area_files import (
    AAA_AB1,
    AAA_AB2,
    AAA_COUNTS,
    AAA_IFAB,
    CAL_WORDS,
    DIR_WORDS,
    EARTH_SEMIMAJOR_KM,
    GOES3_IR,
    GOES5_IR,
    GOES6_IR,
    GOES7_IR,
    IR_WINDOW,
    LIMB_WINDOW,
    MCIDAS_MISSING,
    NADIR_COL,
    NADIR_ROW,
    NAV_WORDS,
    SHAPE,
    SUB_LON,
    W_CAL_OFFSET,
    FakeArea,
    aaa_spec,
    make_nav_words,
    mode_a_data,
    mode_aa_spec,
    patch_dir_word,
    planck_temperature,
    write_area_file,
)

# NOTE: the ``mode_a_ir_file``, ``mode_a_vis_file``, ``mode_aa_file`` and ``goes7_aaa_file``
# fixtures come from the ``conftest.py`` next to this module.


def make_handler(path, **kwargs):
    """Create a file handler for the given fake file."""
    return GOESPreGVARAreaFileHandler(path, {}, {"file_type": "area_ir"}, **kwargs)


def get(handler, name, calibration=None, **info):
    """Load a dataset from a handler like the YAML reader would."""
    dataset_id = {"name": name, "calibration": calibration}
    return handler.get_dataset(dataset_id, {"name": name, **info})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModeAIR:
    """Single-band IR files."""

    def test_brightness_temperature(self, mode_a_ir_file):
        """Apply the documented two-piece VISSR formula."""
        res = get(make_handler(mode_a_ir_file), "IR", "brightness_temperature", band="IR")
        counts = np.array([0, 100, 175, 176, 200, 255])
        expected = np.where(counts >= 176, 418.0 - counts, 330.0 - counts / 2.0)
        assert res.attrs["units"] == "K"
        assert res.dims == ("y", "x")
        np.testing.assert_allclose(res.values[NADIR_ROW, :6], expected)

    def test_counts(self, mode_a_ir_file):
        """Return the raw counts."""
        res = get(make_handler(mode_a_ir_file), "IR", "counts", band="IR")
        assert res.attrs["units"] == "1"
        np.testing.assert_array_equal(res.values[NADIR_ROW, :6], [0, 100, 175, 176, 200, 255])

    def test_unsupported_calibration(self, mode_a_ir_file):
        """Reject calibrations the format does not have."""
        with pytest.raises(ValueError, match="Unsupported calibration"):
            get(make_handler(mode_a_ir_file), "IR", "radiance", band="IR")

    def test_dataset_of_the_other_band_is_none(self, mode_a_ir_file):
        """Return None for a VIS dataset requested from an IR file."""
        assert get(make_handler(mode_a_ir_file), "VIS", "counts", band="VIS") is None

    def test_attributes(self, mode_a_ir_file):
        """Platform and sensor names are set on the data."""
        res = get(make_handler(mode_a_ir_file), "IR", band="IR")
        assert res.attrs["platform_name"] == "GOES-3"
        assert res.attrs["sensor"] == "goes_pregvar"


class TestModeAVIS:
    """Single-band VIS files."""

    def test_counts(self, mode_a_vis_file):
        """Return the raw counts."""
        res = get(make_handler(mode_a_vis_file), "VIS", "counts", band="VIS")
        np.testing.assert_array_equal(res.values[NADIR_ROW, :6], [0, 4, 128, 252, 252, 255])

    def test_unnormalized_reflectance(self, mode_a_vis_file):
        """Scale the 6-bit counts to percent and clip to 0-100."""
        res = get(make_handler(mode_a_vis_file), "VIS", "unnormalized_reflectance", band="VIS")
        expected = np.array([0, 4, 128, 252, 252, 255]) / 252.0 * 100.0
        assert res.attrs["units"] == "%"
        np.testing.assert_allclose(res.values[NADIR_ROW, :6], np.clip(expected, 0, 100), rtol=1e-6)

    def test_unsupported_calibration(self, mode_a_vis_file):
        """Reject calibrations the format does not have."""
        with pytest.raises(ValueError, match="Unsupported calibration"):
            get(make_handler(mode_a_vis_file), "VIS", "brightness_temperature", band="VIS")


class TestNavigation:
    """Geolocation, time and sampling attributes."""

    @pytest.fixture
    def handler(self, mode_a_ir_file):
        """Create a handler for the IR window around nadir."""
        return make_handler(mode_a_ir_file)

    def test_lonlat_at_nadir(self, handler):
        """The pixel nearest nadir sits at the sub-satellite point."""
        lon = get(handler, "longitude").values
        lat = get(handler, "latitude").values
        assert lon.shape == SHAPE
        assert lon[NADIR_ROW, NADIR_COL] == pytest.approx(SUB_LON, abs=0.1)
        assert lat[NADIR_ROW, NADIR_COL] == pytest.approx(0.0, abs=0.1)

    def test_lonlat_metadata(self, handler):
        """Longitude and latitude carry the nadir line spacing as resolution."""
        for name in ("longitude", "latitude"):
            res = get(handler, name, standard_name=name)
            assert res.attrs["standard_name"] == name
            assert res.attrs["resolution"] == pytest.approx(6860.0, rel=1e-3)

    def test_times(self, handler):
        """Start time is the nominal time; the end time cannot be known."""
        assert handler.start_time.isoformat() == "1992-08-23T13:01:00"
        assert handler.end_time == handler.start_time

    def test_nadir_sampling(self, handler):
        """Nadir spacing follows from the scan angle per line and element."""
        nadir = handler._get_nadir_sampling()
        altitude = nadir["altitude"]
        assert nadir["sub_lon"] == pytest.approx(SUB_LON, abs=1e-3)
        assert nadir["sub_lat"] == pytest.approx(0.0, abs=1e-6)
        assert altitude == pytest.approx(42164.17e3 - EARTH_SEMIMAJOR_KM * 1e3, abs=10)
        radlin = np.deg2rad(20.0) / (8 * 1821 - 1)
        radele = np.deg2rad(18.0 + 21.0 / 60.0) / (15288 - 1)
        assert nadir["line_step"] == pytest.approx(altitude * radlin * 8)
        assert nadir["elem_step"] == pytest.approx(altitude * radele * 4)

    def test_uniform_sampling_area(self, handler):
        """Square full disk geos grid with the line spacing as pixel size."""
        area = get(handler, "IR", band="IR").attrs["area_def_uniform_sampling"]
        nadir = handler._get_nadir_sampling()
        assert isinstance(area, AreaDefinition)
        assert area.shape[0] == area.shape[1]
        assert area.shape[0] == pytest.approx(1584, abs=2)
        assert area.pixel_size_x == pytest.approx(nadir["line_step"], rel=1e-3)
        assert area.pixel_size_y == pytest.approx(nadir["line_step"], rel=1e-3)
        expected_crs = CRS(proj="geos", lon_0=nadir["sub_lon"], h=nadir["altitude"],
                           a=6378137.0, b=6356752.314245, sweep="y", units="m")
        assert area.crs == expected_crs
        assert nadir["sub_lon"] == pytest.approx(SUB_LON, abs=1e-3)

    def test_uniform_sampling_area_is_symmetric(self, handler):
        """The grid covers the full disk and is centered on the sub-satellite point."""
        area = handler._get_area_def_uniform_sampling()
        left, bottom, right, top = area.area_extent
        assert (left, bottom) == (-right, -top)
        assert right == pytest.approx(5.43e6, rel=2e-3)

    def test_skew_word_missing_value_is_ignored(self, tmp_path):
        """A McIDAS missing value in the skew word means no skew."""
        data = mode_a_data([100])
        plain = FakeArea(GOES3_IR, data, **IR_WINDOW)
        missing = FakeArea(GOES3_IR, data, nav_words=make_nav_words(skew_word=MCIDAS_MISSING), **IR_WINDOW)
        lonlat = []
        for name, spec in (("a", plain), ("b", missing)):
            handler = make_handler(write_area_file(tmp_path / f"goes03.1979.234.15{name}.AREA_IR", spec))
            lonlat.append((get(handler, "longitude").values, get(handler, "latitude").values))
        np.testing.assert_allclose(lonlat[0][0], lonlat[1][0])
        np.testing.assert_allclose(lonlat[0][1], lonlat[1][1])


class TestSpaceMask:
    """Masking of pixels that look into space."""

    @pytest.fixture
    def limb_file(self, tmp_path):
        """IR file whose first rows are in space and last rows on the earth."""
        spec = FakeArea(GOES3_IR, mode_a_data([100]), **LIMB_WINDOW)
        return write_area_file(tmp_path / "goes03.1979.234.1515.AREA_IR", spec)

    def test_space_is_masked(self, limb_file):
        """Space pixels are NaN where the navigation has no solution."""
        handler = make_handler(limb_file)
        res = get(handler, "IR", "brightness_temperature", band="IR").values
        lon = get(handler, "longitude").values
        assert np.isnan(res[0]).all()
        assert np.isfinite(res[-1]).all()
        np.testing.assert_array_equal(np.isnan(res), np.isnan(lon))

    def test_mask_can_be_disabled(self, limb_file):
        """Keep the values of space pixels without masking."""
        handler = make_handler(limb_file, mask_space=False)
        res = get(handler, "IR", "brightness_temperature", band="IR").values
        assert np.isfinite(res).all()


class TestModeAA:
    """Mode AA (VAS) files with per-line calibration entries in the line prefix."""

    def test_brightness_temperature(self, mode_aa_file):
        """Subtract the offset, scale to radiance and invert the Planck function."""
        res = get(make_handler(mode_aa_file), "VAS_CH8", "brightness_temperature", band="IR", vas_channel=8)
        corrected = 10000 - 400  # already a multiple of 8
        radiance = corrected * 2.0 ** (0 - 7)
        expected = planck_temperature(radiance, (8512, 1286, 0.3212, 0.9974))
        assert res.attrs["units"] == "K"
        assert res.values[NADIR_ROW, NADIR_COL] == pytest.approx(expected, abs=1e-3)
        assert 200 < expected < 330

    def test_channel_missing_on_all_lines_is_nan(self, mode_aa_file):
        """A band without calibration entry has no data."""
        res = get(make_handler(mode_aa_file), "VAS_CH7", "brightness_temperature", band="IR", vas_channel=7)
        assert np.isnan(res.values).all()

    def test_zero_counts_are_nan(self, tmp_path):
        """Counts of zero are the no-data value."""
        spec = mode_aa_spec(raw=0)
        handler = make_handler(write_area_file(tmp_path / "goes05.1983.229.2231-2.AREA_IR", spec))
        res = get(handler, "VAS_CH8", "brightness_temperature", band="IR", vas_channel=8)
        assert np.isnan(res.values).all()

    @pytest.mark.parametrize("calibration", ["radiance", "counts"])
    def test_unsupported_calibrations(self, mode_aa_file, calibration):
        """Only brightness temperature can be derived for Mode AA."""
        with pytest.raises(ValueError, match="calibration"):
            get(make_handler(mode_aa_file), "VAS_CH8", calibration, band="IR", vas_channel=8)

    def test_channel_is_required(self, mode_aa_file):
        """The dataset must name the physical channel."""
        with pytest.raises(ValueError, match="vas_channel"):
            get(make_handler(mode_aa_file), "IR", "brightness_temperature", band="IR")


class TestModeAAA:
    """Mode AAA (VAS) files with a calibration block and a per-line band list."""

    @pytest.mark.parametrize(("channel", "row"), [(8, 0), (8, 1), (7, 0), (7, 1), (10, 1)])
    def test_counts_follow_the_per_line_band_list(self, goes7_aaa_file, channel, row):
        """Find each channel in whichever band slot it is on that line."""
        res = get(make_handler(goes7_aaa_file), f"VAS_CH{channel}", "counts", band="IR", vas_channel=channel)
        assert (res.values[row] == AAA_COUNTS[channel]).all()

    def test_radiance(self, goes7_aaa_file):
        """Apply the radiance equation of the calibration block."""
        res = get(make_handler(goes7_aaa_file), "VAS_CH8", "radiance", band="IR", vas_channel=8)
        expected = (AAA_AB2 * AAA_COUNTS[8] / 32.0 - AAA_AB1) / 2.0 ** (15 - AAA_IFAB)
        assert res.attrs["units"] == "mW m-2 sr-1 (cm-1)-1"
        np.testing.assert_allclose(res.values[NADIR_ROW], expected, rtol=1e-6)

    @pytest.mark.parametrize(
        ("sensor_source", "constants"),
        [
            (GOES6_IR, (8438.0, 1283.0, 0.3263, 0.9973)),
            (GOES7_IR, (8525.0, 1287.0, 0.3408, 0.9973)),
        ])
    def test_brightness_temperature_uses_satellite_constants(self, tmp_path, sensor_source, constants):
        """GOES-6 and GOES-7 have their own Planck constants."""
        path = write_area_file(tmp_path / "goes07.1992.236.1301.AREA_IR", aaa_spec(sensor_source))
        res = get(make_handler(path), "VAS_CH8", "brightness_temperature", band="IR", vas_channel=8)
        expected = planck_temperature(AAA_COUNTS[8], constants)
        assert res.attrs["units"] == "K"
        assert res.values[NADIR_ROW, NADIR_COL] == pytest.approx(expected, abs=1e-3)

    def test_satellite_without_constants(self, tmp_path):
        """Do not guess constants for satellites that have none."""
        path = write_area_file(tmp_path / "goes05.1992.236.1301.AREA_IR", aaa_spec(GOES5_IR))
        with pytest.raises(NotImplementedError, match="GOES-6 and GOES-7"):
            get(make_handler(path), "VAS_CH8", "brightness_temperature", band="IR", vas_channel=8)

    def test_unsupported_calibration(self, goes7_aaa_file):
        """Reject calibrations the format does not have."""
        with pytest.raises(ValueError, match="Unsupported calibration"):
            get(make_handler(goes7_aaa_file), "VAS_CH8", "reflectance", band="IR", vas_channel=8)

    def test_channel_is_required(self, goes7_aaa_file):
        """The dataset must name the physical channel."""
        with pytest.raises(ValueError, match="vas_channel"):
            get(make_handler(goes7_aaa_file), "IR", "brightness_temperature", band="IR")

    def test_channel_absent_on_some_lines_is_nan(self, tmp_path):
        """Lines that do not carry the channel have no data."""
        spec = aaa_spec(GOES7_IR)
        spec.prefixes[1, 632:632 + 3] = (7, 10, 0)  # channel 8 is missing on odd lines
        handler = make_handler(write_area_file(tmp_path / "goes07.1992.236.1301.AREA_IR", spec))
        res = get(handler, "VAS_CH8", "brightness_temperature", band="IR", vas_channel=8)
        assert np.isfinite(res.values[0]).all()
        assert np.isnan(res.values[1]).all()

    def test_band_list_is_required(self, tmp_path):
        """Without a usable band list the channels of an AAA file cannot be told apart."""
        spec = aaa_spec(GOES7_IR)
        spec.prefixes = spec.prefixes[:, :633]
        handler = make_handler(write_area_file(tmp_path / "goes07.1992.236.1301.AREA_IR", spec))
        with pytest.raises(ValueError, match="band list"):
            get(handler, "VAS_CH8", "brightness_temperature", band="IR", vas_channel=8)


class TestAvailableDatasets:
    """Advertise only the datasets a file can produce."""

    @staticmethod
    def _availability(path):
        handler = GOESPreGVARAreaFileHandler(path, {}, {"file_type": "area_ir"})
        infos = [
            {"name": "IR", "file_type": ["area_ir"], "band": "IR"},
            {"name": "VAS_CH8", "file_type": ["area_ir"], "band": "IR", "vas_channel": 8},
        ]
        result = handler.available_datasets([(None, info) for info in infos])
        return {info["name"]: available for available, info in result}

    def test_single_band_file(self, mode_a_ir_file):
        """The generic IR dataset is available, the sounder channels are not."""
        assert self._availability(mode_a_ir_file) == {"IR": True, "VAS_CH8": False}

    def test_multiband_aaa_file(self, goes7_aaa_file):
        """The sounder channel is available, the generic IR dataset is not."""
        assert self._availability(goes7_aaa_file) == {"IR": False, "VAS_CH8": True}

    def test_multiband_mode_aa_file(self, mode_aa_file):
        """Only the channels with a calibration entry are available."""
        assert self._availability(mode_aa_file) == {"IR": False, "VAS_CH8": True}


class TestFileVariants:
    """Truncated files and byte order."""

    def test_truncated_file_uses_the_lines_present(self, tmp_path):
        """Data, navigation and mask all use the number of lines that are really there."""
        spec = FakeArea(GOES3_IR, mode_a_data([100]), claimed_lines=SHAPE[0] + 3, **IR_WINDOW)
        handler = make_handler(write_area_file(tmp_path / "goes03.1979.234.1515.AREA_IR", spec))
        assert get(handler, "IR", "counts", band="IR").shape == SHAPE
        assert get(handler, "longitude").shape == SHAPE

    @pytest.mark.parametrize("endian", [">", "<"])
    def test_byte_order_is_detected(self, tmp_path, endian):
        """Read big and little endian files alike."""
        spec = FakeArea(GOES3_IR, mode_a_data([100, 200]), endian=endian, **IR_WINDOW)
        handler = make_handler(write_area_file(tmp_path / "goes03.1979.234.1515.AREA_IR", spec))
        counts = get(handler, "IR", "counts", band="IR")
        np.testing.assert_array_equal(counts.values[0, :2], [100, 200])
        assert get(handler, "longitude").values[NADIR_ROW, NADIR_COL] == pytest.approx(SUB_LON, abs=0.1)

    def test_two_byte_data_follows_the_byte_order(self, tmp_path):
        """Two byte counts are read in the byte order of the file."""
        data = np.full((SHAPE[0], SHAPE[1], 3), 1000)
        spec = aaa_spec(GOES7_IR)
        spec.endian = "<"
        spec.data = np.where(np.arange(3)[None, None, :] == 1, 3000, data)
        handler = make_handler(write_area_file(tmp_path / "goes07.1992.236.1301.AREA_IR", spec))
        assert AreaFile(handler.filename).read_channel(7)[0].max() == 3000


class TestMissingNavigation:
    """Files without usable navigation cannot be read."""

    def test_no_orbital_elements(self, tmp_path):
        """Pre-1980 files have a zeroed navigation block."""
        nav = make_nav_words()
        nav[6:12] = 0
        spec = FakeArea(GOES3_IR, mode_a_data([100]), nav_words=nav, **IR_WINDOW)
        handler = make_handler(write_area_file(tmp_path / "goes03.1979.234.1515.AREA_IR", spec))
        with pytest.raises(ValueError, match="orbital"):
            get(handler, "longitude")


class TestHandlerAvailability:
    """Helpers behind available_datasets()."""

    def test_dataset_without_channel_is_available(self, mode_a_ir_file):
        """Datasets that name no sounder channel need no channel calibration."""
        assert make_handler(mode_a_ir_file)._vas_channel_available({})

    def test_sounder_channel_of_single_band_file(self, mode_a_ir_file):
        """A single-band file has no sounder channels."""
        assert not make_handler(mode_a_ir_file)._vas_channel_available({"vas_channel": 8})

    def test_known_availability_is_kept(self, mode_a_ir_file):
        """Availability decided earlier is passed on."""
        assert make_handler(mode_a_ir_file)._dataset_availability(True, {"file_type": "area_ir"})

    def test_other_file_types_are_left_alone(self, mode_a_ir_file):
        """Datasets of another file type stay undecided."""
        assert make_handler(mode_a_ir_file)._dataset_availability(None, {"file_type": "other"}) is None

    def test_missing_lines_are_padded_with_nan(self, mode_a_ir_file):
        """Pad a result that has fewer lines than the file."""
        data = np.ones((SHAPE[0] - 2, SHAPE[1]), dtype=np.float32)
        res = make_handler(mode_a_ir_file)._pad_missing_lines(data)
        assert res.shape == SHAPE
        assert np.isnan(res[-2:]).all()
        assert (res[:-2] == 1).all()


class TestScene:
    """Load data through a Scene with the YAML reader definition."""

    def test_load_ir_and_vis(self, mode_a_ir_file, mode_a_vis_file):
        """Load calibrated bands with their coordinates and areas."""
        scn = Scene(filenames=[mode_a_ir_file, mode_a_vis_file], reader="goes_pregvar_area")
        scn.load(["IR", "VIS"])
        ir, vis = scn["IR"], scn["VIS"]
        assert ir.attrs["units"] == "K"
        assert vis.attrs["units"] == "%"
        assert ir.shape == SHAPE
        assert ir.attrs["platform_name"] == "GOES-3"
        assert ir.attrs["sensor"] == "goes_pregvar"
        assert isinstance(ir.attrs["area"], SwathDefinition)
        assert isinstance(ir.attrs["area_def_uniform_sampling"], AreaDefinition)
        assert ir.attrs["start_time"] == vis.attrs["start_time"]

    def test_available_dataset_names(self, goes7_aaa_file):
        """List the sounder channels, but not the generic IR dataset, for a multiband file."""
        scn = Scene(filenames=[goes7_aaa_file], reader="goes_pregvar_area")
        names = scn.available_dataset_names()
        assert "VAS_CH8" in names
        assert "IR" not in names

    def test_load_sounder_channel(self, goes7_aaa_file):
        """Load a sounder channel in brightness temperature."""
        scn = Scene(filenames=[goes7_aaa_file], reader="goes_pregvar_area")
        scn.load(["VAS_CH8"])
        res = scn["VAS_CH8"]
        assert isinstance(res, xr.DataArray)
        assert res.attrs["units"] == "K"
        assert np.isfinite(res.values[NADIR_ROW, NADIR_COL])

    def test_aaa_channel_without_calibration_block(self, tmp_path):
        """A channel is not available when the AAA calibration block is too short."""
        path = write_area_file(tmp_path / "goes07.1992.236.1301.AREA_IR", aaa_spec(GOES7_IR))
        patch_dir_word(path, W_CAL_OFFSET, (DIR_WORDS + NAV_WORDS + CAL_WORDS) * 4 - 40)
        assert not make_handler(path)._vas_channel_available({"vas_channel": 8})

    def test_mode_aa_channel_without_calibration_entries(self, tmp_path):
        """A channel is not available when the Mode AA calibration region is too short."""
        spec = mode_aa_spec()
        spec.cal_len = 50
        path = write_area_file(tmp_path / "goes05.1983.229.2231-2.AREA_IR", spec)
        assert not make_handler(path)._vas_channel_available({"vas_channel": 8})
