"""Tests for the pre-GVAR VISSR/VAS AREA reader (``goes_pregvar_area``)."""

import dataclasses

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

# Area directory word indices (0-based)
_W_SENSOR = 2
_W_DATE = 3
_W_TIME = 4
_W_UL_LINE = 5
_W_UL_ELEM = 6
_W_NLINES = 8
_W_NELEM = 9
_W_BYTES = 10
_W_LINE_RES = 11
_W_ELEM_RES = 12
_W_NBANDS = 13
_W_PREFIX_LEN = 14
_W_DATA_OFFSET = 33
_W_NAV_OFFSET = 34
_W_VALCODE = 35
_W_DOC_LEN = 48
_W_CAL_LEN = 49
_W_SOURCE_TYPE = 51
_W_CAL_TYPE = 52
_W_CAL_OFFSET = 62

_DIR_WORDS = 64
_NAV_WORDS = 128
_CAL_WORDS = 128

SUB_LON = -112.1
EARTH_SEMIMAJOR_KM = 6378.137
MCIDAS_MISSING = -2139062144  # 0x80808080 as a signed 32-bit integer

# sensor source numbers from the AREA directory
GOES3_IR = 25
GOES3_VIS = 24
GOES5_IR = 29
GOES6_IR = 31
GOES7_IR = 33


@dataclasses.dataclass
class FakeArea:
    """Description of a fake AREA file."""

    sensor_source: int
    data: np.ndarray  # (lines, elements, bands)
    source_type: str = "VISR"
    cal_type: str = "BRIT"
    ul_line: int = 1
    ul_element: int = 1
    line_res: int = 8
    elem_res: int = 4
    bytes_per_elem: int = 1
    prefixes: np.ndarray | None = None  # (lines, prefix_len) uint8
    cal_words: np.ndarray | None = None
    valcode: int = 0
    doc_len: int = 0
    cal_len: int = 0
    claimed_lines: int | None = None
    endian: str = ">"
    nav_words: np.ndarray | None = None


def make_nav_words(skew_word=0):
    """Nav block of a circular equatorial orbit with a polar spin axis.

    The scan geometry gives about 0.86 km between full-resolution lines and
    0.75 km between full-resolution elements at nadir, like the real files.
    """
    words = np.zeros(_NAV_WORDS, dtype=np.int64)
    words[0] = int.from_bytes(b"GOES", "big")
    words[1] = 92236  # SSYYDDD
    words[2] = 130100  # nominal HHMMSS
    words[4] = 920823  # epoch YYMMDD
    words[5] = 130000  # epoch HHMMSS
    words[6] = 4216417  # semimajor axis [km * 100]
    words[7] = 0  # eccentricity * 1e6
    words[8] = 0  # inclination * 1000
    words[9] = 0  # mean anomaly * 1000
    words[10] = 0  # argument of perigee * 1000
    words[11] = 54973  # right ascension of ascending node * 1000
    words[12] = 900000  # spin axis declination, DDDMMSS
    words[13] = 0  # spin axis right ascension, DDDMMSS
    words[14] = 7285  # picture center line
    words[15] = 600000  # spin period [us]
    words[16] = 200000  # degrees of line sweep, DDDMMSS (20 deg)
    words[17] = 8 * 100000 + 1821  # NNLLLLL: 8 detectors, 1821 scans
    words[18] = 182100  # degrees of element sweep, DDDMMSS (18 deg 21 min)
    words[19] = 15288  # elements per line
    words[28] = skew_word
    return words


def _pack(words, endian):
    return np.asarray(words).astype(f"{endian}i4").tobytes()


def _directory(spec, nav_offset, cal_offset, data_offset):
    words = np.zeros(_DIR_WORDS, dtype=np.int64)
    nlines = spec.data.shape[0]
    words[_W_SENSOR] = spec.sensor_source
    words[_W_DATE] = 92236
    words[_W_TIME] = 130100
    words[_W_UL_LINE] = spec.ul_line
    words[_W_UL_ELEM] = spec.ul_element
    words[_W_NLINES] = spec.claimed_lines or nlines
    words[_W_NELEM] = spec.data.shape[1]
    words[_W_BYTES] = spec.bytes_per_elem
    words[_W_LINE_RES] = spec.line_res
    words[_W_ELEM_RES] = spec.elem_res
    words[_W_NBANDS] = spec.data.shape[2]
    words[_W_PREFIX_LEN] = 0 if spec.prefixes is None else spec.prefixes.shape[1]
    words[_W_DATA_OFFSET] = data_offset
    words[_W_NAV_OFFSET] = nav_offset
    words[_W_VALCODE] = spec.valcode
    words[_W_DOC_LEN] = spec.doc_len
    words[_W_CAL_LEN] = spec.cal_len
    words[_W_CAL_OFFSET] = cal_offset
    raw = bytearray(_pack(words, spec.endian))
    # source and calibration types are plain ASCII, independent of byte order
    raw[_W_SOURCE_TYPE * 4:_W_SOURCE_TYPE * 4 + 4] = spec.source_type.ljust(4).encode("ascii")
    raw[_W_CAL_TYPE * 4:_W_CAL_TYPE * 4 + 4] = spec.cal_type.ljust(4).encode("ascii")
    return bytes(raw)


def write_area_file(path, spec):
    """Write a fake AREA file and return its path as a string."""
    nav_offset = _DIR_WORDS * 4
    cal_offset = nav_offset + _NAV_WORDS * 4 if spec.cal_words is not None else 0
    data_offset = (cal_offset + _CAL_WORDS * 4) if spec.cal_words is not None else nav_offset + _NAV_WORDS * 4

    nav_words = make_nav_words() if spec.nav_words is None else spec.nav_words
    nav = b"GOES" + _pack(nav_words[1:], spec.endian)

    blocks = [_directory(spec, nav_offset, cal_offset, data_offset), nav]
    if spec.cal_words is not None:
        blocks.append(_pack(spec.cal_words, spec.endian))
    pixels = spec.data.astype(f"{spec.endian}u{spec.bytes_per_elem}")
    for line in range(spec.data.shape[0]):
        if spec.prefixes is not None:
            blocks.append(spec.prefixes[line].astype(np.uint8).tobytes())
        blocks.append(pixels[line].tobytes())
    path.write_bytes(b"".join(blocks))
    return str(path)


# ---------------------------------------------------------------------------
# Fake files
# ---------------------------------------------------------------------------

# 16 lines x 24 elements around nadir (nadir is at row 8, column 12 for IR)
IR_WINDOW = {"ul_line": 7213, "ul_element": 7592, "line_res": 8, "elem_res": 4}
VIS_WINDOW = {"ul_line": 7276, "ul_element": 7631, "line_res": 1, "elem_res": 1}
# 16 lines straddling the northern limb; the first rows look into space
LIMB_WINDOW = {"ul_line": 900, "ul_element": 7592, "line_res": 8, "elem_res": 4}
SHAPE = (16, 24)
NADIR_ROW, NADIR_COL = 8, 12


def _mode_a_data(values):
    """Single band image where each column holds one of the given values."""
    columns = np.resize(np.asarray(values, dtype=np.uint8), SHAPE[1])
    return np.tile(columns[None, :, None], (SHAPE[0], 1, 1))


@pytest.fixture
def mode_a_ir_file(tmp_path):
    """Mode A IR file where the counts cover both branches of the formula."""
    spec = FakeArea(GOES3_IR, _mode_a_data([0, 100, 175, 176, 200, 255]), **IR_WINDOW)
    return write_area_file(tmp_path / "goes03.1979.234.1515.AREA_IR", spec)


@pytest.fixture
def mode_a_vis_file(tmp_path):
    """Mode A VIS file; counts are 6-bit values shifted left by two."""
    spec = FakeArea(GOES3_VIS, _mode_a_data([0, 4, 128, 252, 252, 255]), cal_type="RAW", **VIS_WINDOW)
    return write_area_file(tmp_path / "goes03.1979.234.1515.AREA_VIS", spec)


def _mode_aa_spec(raw=10000, channel_codes=(8,) + (0,) * 12):
    """Mode AA file: calibration entries in every line's prefix, 2-byte data."""
    nlines, nelem = SHAPE
    prefix_len = 4 + 512 + 116 + 4
    prefixes = np.zeros((nlines, prefix_len), dtype=np.uint8)
    for line in range(nlines):
        entries = np.zeros((13, 4), dtype=">i2")
        entries[:, 0] = channel_codes
        entries[0, 2] = 0  # delta-F code 0 -> delta-F = 0
        entries[0, 3] = 400  # space-view offset
        prefixes[line, 4 + 512 + 12:4 + 512 + 116] = np.frombuffer(entries.tobytes(), dtype=np.uint8)
        prefixes[line, 632] = channel_codes[0]  # band list: raw channel code of the one band
    data = np.full((nlines, nelem, 1), raw)
    return FakeArea(
        GOES5_IR, data, source_type="VAS", cal_type="RAW", bytes_per_elem=2, prefixes=prefixes,
        valcode=1, doc_len=512, cal_len=116, **IR_WINDOW)


@pytest.fixture
def mode_aa_file(tmp_path):
    """Mode AA file with band 8 calibrated on every line."""
    return write_area_file(tmp_path / "goes05.1983.229.2231-2.AREA_IR", _mode_aa_spec())


# AAA fake: three bands whose order differs between lines
AAA_BAND_ORDERS = ((8, 7, 10), (7, 8, 10))
AAA_COUNTS = {7: 3000, 8: 4000, 10: 5000}
AAA_AB1, AAA_AB2, AAA_IFAB = 0, 32, 15  # radiance = counts for these coefficients


def _aaa_cal_words(sensor_source):
    words = np.zeros(_CAL_WORDS, dtype=np.int64)
    words[0] = sensor_source
    for channel in range(1, 39):
        words[3 + 2 * (channel - 1)] = AAA_AB1
        words[3 + 2 * (channel - 1) + 1] = AAA_AB2
        words[79 + channel - 1] = AAA_IFAB
    return words


def _aaa_spec(sensor_source):
    nlines, nelem = SHAPE
    prefix_len = 632 + 4
    prefixes = np.zeros((nlines, prefix_len), dtype=np.uint8)
    data = np.zeros((nlines, nelem, 3), dtype=np.int64)
    for line in range(nlines):
        order = AAA_BAND_ORDERS[line % 2]
        prefixes[line, 632:632 + 3] = order
        for slot, channel in enumerate(order):
            data[line, :, slot] = AAA_COUNTS[channel]
    return FakeArea(
        sensor_source, data, source_type="AAA", cal_type="RAW", bytes_per_elem=2,
        prefixes=prefixes, cal_words=_aaa_cal_words(sensor_source), **IR_WINDOW)


@pytest.fixture
def goes7_aaa_file(tmp_path):
    """GOES-7 AAA file with channels 8, 7 and 10 in changing order."""
    return write_area_file(tmp_path / "goes07.1992.236.1301.AREA_IR", _aaa_spec(GOES7_IR))


def make_handler(path, **kwargs):
    """Create a file handler for the given fake file."""
    return GOESPreGVARAreaFileHandler(path, {}, {"file_type": "area_ir"}, **kwargs)


def get(handler, name, calibration=None, **info):
    """Load a dataset from a handler like the YAML reader would."""
    dataset_id = {"name": name, "calibration": calibration}
    return handler.get_dataset(dataset_id, {"name": name, **info})


def planck_temperature(radiance, constants):
    """Planck inversion with a band correction, written out independently of the reader."""
    fk1, fk2, tc1, tc2 = constants
    return (fk2 / np.log(1.0 + fk1 / radiance) - tc1) / tc2


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
        data = _mode_a_data([100])
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
        spec = FakeArea(GOES3_IR, _mode_a_data([100]), **LIMB_WINDOW)
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
        spec = _mode_aa_spec(raw=0)
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
        path = write_area_file(tmp_path / "goes07.1992.236.1301.AREA_IR", _aaa_spec(sensor_source))
        res = get(make_handler(path), "VAS_CH8", "brightness_temperature", band="IR", vas_channel=8)
        expected = planck_temperature(AAA_COUNTS[8], constants)
        assert res.attrs["units"] == "K"
        assert res.values[NADIR_ROW, NADIR_COL] == pytest.approx(expected, abs=1e-3)

    def test_satellite_without_constants(self, tmp_path):
        """Do not guess constants for satellites that have none."""
        path = write_area_file(tmp_path / "goes05.1992.236.1301.AREA_IR", _aaa_spec(GOES5_IR))
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
        spec = _aaa_spec(GOES7_IR)
        spec.prefixes[1, 632:632 + 3] = (7, 10, 0)  # channel 8 is missing on odd lines
        handler = make_handler(write_area_file(tmp_path / "goes07.1992.236.1301.AREA_IR", spec))
        res = get(handler, "VAS_CH8", "brightness_temperature", band="IR", vas_channel=8)
        assert np.isfinite(res.values[0]).all()
        assert np.isnan(res.values[1]).all()


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
        spec = FakeArea(GOES3_IR, _mode_a_data([100]), claimed_lines=SHAPE[0] + 3, **IR_WINDOW)
        handler = make_handler(write_area_file(tmp_path / "goes03.1979.234.1515.AREA_IR", spec))
        assert get(handler, "IR", "counts", band="IR").shape == SHAPE
        assert get(handler, "longitude").shape == SHAPE

    @pytest.mark.parametrize("endian", [">", "<"])
    def test_byte_order_is_detected(self, tmp_path, endian):
        """Read big and little endian files alike."""
        spec = FakeArea(GOES3_IR, _mode_a_data([100, 200]), endian=endian, **IR_WINDOW)
        handler = make_handler(write_area_file(tmp_path / "goes03.1979.234.1515.AREA_IR", spec))
        counts = get(handler, "IR", "counts", band="IR")
        np.testing.assert_array_equal(counts.values[0, :2], [100, 200])
        assert get(handler, "longitude").values[NADIR_ROW, NADIR_COL] == pytest.approx(SUB_LON, abs=0.1)

    def test_two_byte_data_follows_the_byte_order(self, tmp_path):
        """Two byte counts are read in the byte order of the file."""
        data = np.full((SHAPE[0], SHAPE[1], 3), 1000)
        spec = _aaa_spec(GOES7_IR)
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
        spec = FakeArea(GOES3_IR, _mode_a_data([100]), nav_words=nav, **IR_WINDOW)
        handler = make_handler(write_area_file(tmp_path / "goes03.1979.234.1515.AREA_IR", spec))
        with pytest.raises(ValueError, match="orbital"):
            get(handler, "longitude")


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
