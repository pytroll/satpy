"""Tests for the pre-GVAR VISSR/VAS AREA file handler (``goes_pregvar_area``)."""

import base64
import datetime as dt
import gzip
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS
from pyresample.geometry import AreaDefinition, SwathDefinition
from satpy import Scene
from satpy.tests.utils import skip_numba_unstable_if_missing

try:
    from satpy.readers.goes_pregvar import goes_pregvar_area as handler_module
    from satpy.readers.goes_pregvar.goes_pregvar_area import GOESPreGVARAreaFileHandler
except ImportError as err:
    if skip_numba_unstable_if_missing():
        pytest.skip(f"Numba is not compatible with unstable NumPy: {err!s}", allow_module_level=True)
    raise

SHAPE = (4, 6)
NADIR = {"sub_lon": -112.1, "sub_lat": 0.0, "altitude": 35786033.0, "line_step": 6860.26, "elem_step": 2998.92}
BT_A, BT_AA, BT_AAA, RADIANCE, CHANNEL_COUNTS = 250.0, 251.0, 252.0, 7.5, 90

# A 1152 byte AREA file made with pyarea's own fake file writer (tests/fake_area_files.py in pyarea):
# FakeArea(GOES7_IR, mode_a_data([0, 100, 175, 176, 200, 255]), **IR_WINDOW), a 16 x 24 IR window of GOES-7
# (gzip, base64).
REAL_MODE_A_IR = (
    "H4sIAAAAAAAC/2NggANFBsYMHwbGPyYMDDK6DAyyK6DiAkAsAcSMQMwBxCxQNhUBM1WMDPMMDnIK8gwhRY+7v2swwt9AwMfznYHx9wUGh7BETNXX9jLw7l4AYcuUMnCqH2Bg5nVg4LGRZWA6DrTXegfDKBhaIGX9hhP/R8mBIgETXmURgAQAAA=="
)


class FakeAreaFile:
    """The part of ``pyarea.AreaFile`` the handler uses, with a different value from every reader."""

    source = "fake.AREA"
    nominal_datetime = dt.datetime(1992, 8, 23, 13, 1)

    def __init__(self, band="IR", source_type="VISR", satellite="GOES-3", nan_rows=0, band_list=True,
                 aaa_cal=None, mode_aa_cal=None):
        """Describe the file; the first ``nan_rows`` rows are in space."""
        self.directory = SimpleNamespace(band=band, satellite=satellite)
        self.source_type = source_type
        self.band_list = np.zeros(SHAPE[0]) if band_list else None
        self.aaa_cal = aaa_cal
        self.mode_aa_cal = mode_aa_cal
        self.nan_rows = nan_rows
        self.calls = []

    def _result(self, call, value, dtype=np.float32):
        self.calls.append(call)
        return np.full(SHAPE, value, dtype=dtype)

    def read_data(self):
        """Return counts."""
        return self._result(("read_data",), 100, np.uint8)

    def read_ir_temperature_a(self):
        """Mode A temperature."""
        return self._result(("read_ir_temperature_a",), BT_A)

    def read_vis(self, calibration="counts"):
        """Mode A visible, the percentage being 30 and the counts 128."""
        return self._result(("read_vis", calibration), 30.0 if calibration == "unnormalized_reflectance" else 128.0)

    def read_ir_temperature_aa(self, channel):
        """Mode AA temperature."""
        return self._result(("read_ir_temperature_aa", channel), BT_AA)

    def read_ir_temperature_aaa(self, channel):
        """Mode AAA temperature."""
        return self._result(("read_ir_temperature_aaa", channel), BT_AAA)

    def read_ir_radiance(self, channel):
        """Mode AAA radiance."""
        return self._result(("read_ir_radiance", channel), RADIANCE)

    def read_channel(self, channel):
        """Mode AAA counts and their validity."""
        return self._result(("read_channel", channel), CHANNEL_COUNTS, np.uint16), None

    def get_nadir_sampling(self):
        """Return the nadir sampling of a GOES-7 IR file."""
        return dict(NADIR)

    def get_lonlats(self):
        """Longitude and latitude, NaN for the rows in space."""
        lat, lon = np.meshgrid(np.linspace(-10, 10, SHAPE[0]), np.linspace(-120, -100, SHAPE[1]), indexing="ij")
        lat[:self.nan_rows] = np.nan
        lon[:self.nan_rows] = np.nan
        return lon, lat


@pytest.fixture
def make_handler(monkeypatch):
    """Create a handler on top of a fake AreaFile."""
    def _make(fake=None, **kwargs):
        fake = fake or FakeAreaFile()
        monkeypatch.setattr(handler_module, "AreaFile", lambda filename: fake)
        handler = GOESPreGVARAreaFileHandler("fake.AREA_IR", {}, {"file_type": "area_ir"}, **kwargs)
        handler.fake = fake
        return handler
    return _make


def get(handler, name, calibration=None, **info):
    """Load a dataset from a handler like the YAML reader would."""
    dataset_id = {"name": name, "calibration": calibration}
    return handler.get_dataset(dataset_id, {"name": name, **info})


class TestModeAIR:
    """Single-band IR files."""

    @pytest.mark.parametrize("calibration", ["brightness_temperature", None])
    def test_brightness_temperature(self, make_handler, calibration):
        """Use pyarea's Mode A temperature."""
        handler = make_handler()
        res = get(handler, "IR", calibration, band="IR")
        assert res.attrs["units"] == "K"
        assert res.dims == ("y", "x")
        assert (res.values == BT_A).all()
        assert handler.fake.calls == [("read_ir_temperature_a",)]

    def test_counts(self, make_handler):
        """Return the raw counts as floats."""
        res = get(make_handler(), "IR", "counts", band="IR")
        assert res.attrs["units"] == "1"
        assert res.dtype == np.float32
        assert (res.values == 100).all()

    def test_unsupported_calibration(self, make_handler):
        """Reject calibrations the format does not have."""
        with pytest.raises(ValueError, match="Unsupported calibration"):
            get(make_handler(), "IR", "radiance", band="IR")

    def test_dataset_of_the_other_band_is_none(self, make_handler):
        """Return None for a VIS dataset requested from an IR file."""
        assert get(make_handler(), "VIS", "counts", band="VIS") is None

    def test_attributes(self, make_handler):
        """Platform and sensor names are set on the data, which keeps the YAML keys."""
        res = get(make_handler(), "IR", band="IR", wavelength=[10.5, 11.5, 12.5])
        assert res.attrs["platform_name"] == "GOES-3"
        assert res.attrs["sensor"] == "goes_pregvar"
        assert res.attrs["wavelength"] == [10.5, 11.5, 12.5]
        assert "_calibration_note" not in res.attrs


class TestModeAVIS:
    """Single-band VIS files."""

    @pytest.fixture
    def handler(self, make_handler):
        """Create a handler for a VIS file."""
        return make_handler(FakeAreaFile(band="VIS"))

    @pytest.mark.parametrize("calibration", ["counts", None])
    def test_counts(self, handler, calibration):
        """Return the counts."""
        res = get(handler, "VIS", calibration, band="VIS")
        assert res.attrs["units"] == "1"
        assert (res.values == 128).all()
        assert handler.fake.calls == [("read_vis", "counts")]

    def test_unnormalized_reflectance(self, handler):
        """Ask pyarea for the percentage and note that it is relative."""
        res = get(handler, "VIS", "unnormalized_reflectance", band="VIS")
        assert res.attrs["units"] == "%"
        assert (res.values == 30.0).all()
        assert handler.fake.calls == [("read_vis", "unnormalized_reflectance")]
        assert "not an independently" in res.attrs["_calibration_note"]

    def test_unsupported_calibration(self, handler):
        """Reject calibrations the format does not have."""
        with pytest.raises(ValueError, match="Unsupported calibration"):
            get(handler, "VIS", "brightness_temperature", band="VIS")


class TestNavigation:
    """Geolocation, time and sampling attributes."""

    @pytest.fixture
    def handler(self, make_handler):
        """Create a handler for an IR file."""
        return make_handler()

    def test_lonlat(self, handler):
        """Longitude and latitude are the ones from pyarea, as float32."""
        lon, lat = (get(handler, name).values for name in ("longitude", "latitude"))
        assert lon.shape == lat.shape == SHAPE
        assert lon.dtype == lat.dtype == np.float32
        assert lon[0, 0] == pytest.approx(-120.0)
        assert lat[-1, 0] == pytest.approx(10.0)

    def test_lonlat_metadata(self, handler):
        """Longitude and latitude carry the nadir line spacing as resolution."""
        for name in ("longitude", "latitude"):
            res = get(handler, name, standard_name=name)
            assert res.attrs["standard_name"] == name
            assert res.attrs["resolution"] == NADIR["line_step"]

    def test_times(self, handler):
        """Start time is the nominal time; the end time cannot be known."""
        assert handler.start_time.isoformat() == "1992-08-23T13:01:00"
        assert handler.end_time == handler.start_time

    def test_nadir_sampling_is_read_once(self, handler, monkeypatch):
        """Use the same nadir sampling for every dataset."""
        calls = []
        monkeypatch.setattr(handler.fake, "get_nadir_sampling", lambda: calls.append(1) or dict(NADIR))
        get(handler, "longitude")
        get(handler, "IR", band="IR")
        assert len(calls) == 1

    def test_uniform_sampling_area(self, handler):
        """Square full disk geos grid with the line spacing as pixel size."""
        area = get(handler, "IR", band="IR").attrs["area_def_uniform_sampling"]
        assert isinstance(area, AreaDefinition)
        assert area.shape[0] == area.shape[1]
        assert area.shape[0] == pytest.approx(1584, abs=2)
        assert area.pixel_size_x == pytest.approx(NADIR["line_step"], rel=1e-3)
        assert area.pixel_size_y == pytest.approx(NADIR["line_step"], rel=1e-3)
        expected_crs = CRS(proj="geos", lon_0=NADIR["sub_lon"], h=NADIR["altitude"],
                           a=6378137.0, b=6356752.314245, sweep="y", units="m")
        assert area.crs == expected_crs

    def test_uniform_sampling_area_is_symmetric(self, handler):
        """The grid covers the full disk and is centered on the sub-satellite point."""
        left, bottom, right, top = handler._get_area_def_uniform_sampling().area_extent
        assert (left, bottom) == (-right, -top)
        assert right == pytest.approx(5.43e6, rel=2e-3)


class TestSpaceMask:
    """Masking of pixels that look into space."""

    @pytest.fixture
    def limb_file(self):
        """IR file whose first row is in space."""
        return FakeAreaFile(nan_rows=1)

    def test_space_is_masked(self, make_handler, limb_file):
        """Space pixels are NaN where the navigation has no solution."""
        handler = make_handler(limb_file)
        res = get(handler, "IR", "brightness_temperature", band="IR").values
        lon = get(handler, "longitude").values
        assert np.isnan(res[0]).all()
        assert np.isfinite(res[1:]).all()
        np.testing.assert_array_equal(np.isnan(res), np.isnan(lon))

    def test_mask_can_be_disabled(self, make_handler, limb_file):
        """Keep the values of space pixels without masking."""
        res = get(make_handler(limb_file, mask_space=False), "IR", "brightness_temperature", band="IR").values
        assert np.isfinite(res).all()


class TestModeAA:
    """Mode AA (VAS) files."""

    @pytest.fixture
    def handler(self, make_handler):
        """Create a handler for a Mode AA file."""
        return make_handler(FakeAreaFile(source_type="VAS"))

    @pytest.mark.parametrize("calibration", ["brightness_temperature", None])
    def test_brightness_temperature(self, handler, calibration):
        """Ask pyarea for the requested channel."""
        res = get(handler, "VAS_CH8", calibration, band="IR", vas_channel=8)
        assert res.attrs["units"] == "K"
        assert (res.values == BT_AA).all()
        assert handler.fake.calls == [("read_ir_temperature_aa", 8)]

    @pytest.mark.parametrize("calibration", ["radiance", "counts"])
    def test_unsupported_calibrations(self, handler, calibration):
        """Only brightness temperature can be derived for Mode AA."""
        with pytest.raises(ValueError, match="calibration"):
            get(handler, "VAS_CH8", calibration, band="IR", vas_channel=8)

    def test_channel_is_required(self, handler):
        """The dataset must name the physical channel."""
        with pytest.raises(ValueError, match="vas_channel"):
            get(handler, "IR", "brightness_temperature", band="IR")


class TestModeAAA:
    """Mode AAA (VAS) files."""

    @pytest.fixture
    def handler(self, make_handler):
        """Create a handler for a Mode AAA file."""
        return make_handler(FakeAreaFile(source_type="AAA"))

    def test_counts(self, handler):
        """Take the counts of the channel, wherever its band slot is."""
        res = get(handler, "VAS_CH8", "counts", band="IR", vas_channel=8)
        assert res.attrs["units"] == "1"
        assert (res.values == CHANNEL_COUNTS).all()
        assert handler.fake.calls == [("read_channel", 8)]

    @pytest.mark.parametrize("calibration", ["brightness_temperature", None])
    def test_brightness_temperature(self, handler, calibration):
        """Ask pyarea for the temperature of the channel."""
        res = get(handler, "VAS_CH8", calibration, band="IR", vas_channel=8)
        assert res.attrs["units"] == "K"
        assert (res.values == BT_AAA).all()
        assert handler.fake.calls == [("read_ir_temperature_aaa", 8)]

    def test_radiance(self, handler):
        """Ask pyarea for the radiance of the channel."""
        res = get(handler, "VAS_CH8", "radiance", band="IR", vas_channel=8)
        assert res.attrs["units"] == "mW m-2 sr-1 (cm-1)-1"
        assert (res.values == RADIANCE).all()
        assert handler.fake.calls == [("read_ir_radiance", 8)]

    def test_unsupported_calibration(self, handler):
        """Reject calibrations the format does not have."""
        with pytest.raises(ValueError, match="Unsupported calibration"):
            get(handler, "VAS_CH8", "reflectance", band="IR", vas_channel=8)

    def test_channel_is_required(self, handler):
        """The dataset must name the physical channel."""
        with pytest.raises(ValueError, match="vas_channel"):
            get(handler, "IR", "brightness_temperature", band="IR")

    def test_band_list_is_required(self, make_handler):
        """Without a band list the channels of an AAA file cannot be told apart."""
        handler = make_handler(FakeAreaFile(source_type="AAA", band_list=False))
        with pytest.raises(ValueError, match="band list"):
            get(handler, "VAS_CH8", "brightness_temperature", band="IR", vas_channel=8)
        assert handler.fake.calls == []


class TestAvailableDatasets:
    """Advertise only the datasets a file can produce."""

    @staticmethod
    def _availability(handler, *channels):
        infos = [{"name": "IR", "file_type": ["area_ir"], "band": "IR"}]
        infos += [{"name": f"VAS_CH{ch}", "file_type": ["area_ir"], "band": "IR", "vas_channel": ch}
                  for ch in channels]
        result = handler.available_datasets([(None, info) for info in infos])
        return {info["name"]: available for available, info in result}

    def test_single_band_file(self, make_handler):
        """The generic IR dataset is available, the sounder channels are not."""
        assert self._availability(make_handler(), 8) == {"IR": True, "VAS_CH8": False}

    def test_multiband_aaa_file(self, make_handler):
        """Only the channels with constants in the calibration block are available, 13 borrowing from 8."""
        handler = make_handler(FakeAreaFile(source_type="AAA", aaa_cal={"iab": {8: (1, 2, 3), 10: (1, 2, 3)}}))
        assert self._availability(handler, 7, 8, 10, 13) == {
            "IR": False, "VAS_CH7": False, "VAS_CH8": True, "VAS_CH10": True, "VAS_CH13": True}

    def test_aaa_file_without_calibration_block(self, make_handler):
        """No calibration block, no channels."""
        handler = make_handler(FakeAreaFile(source_type="AAA"))
        assert self._availability(handler, 8) == {"IR": False, "VAS_CH8": False}

    def test_multiband_mode_aa_file(self, make_handler):
        """Only the channels with a calibration entry in some line are available; 13 never is."""
        cal = {"channel": np.array([8, 0, 0, 8, 0])}  # raw channel codes, 0 unused
        handler = make_handler(FakeAreaFile(source_type="VAS", mode_aa_cal=cal))
        assert self._availability(handler, 7, 8, 13) == {"IR": False, "VAS_CH7": False, "VAS_CH8": True,
                                                         "VAS_CH13": False}

    def test_mode_aa_file_without_calibration_entries(self, make_handler):
        """No calibration entries, no channels."""
        handler = make_handler(FakeAreaFile(source_type="VAS"))
        assert self._availability(handler, 8) == {"IR": False, "VAS_CH8": False}

    def test_known_availability_is_kept(self, make_handler):
        """Availability decided earlier is passed on."""
        assert make_handler()._dataset_availability(True, {"file_type": "area_ir"})
        assert not make_handler()._dataset_availability(False, {"file_type": "area_ir"})

    def test_other_file_types_are_left_alone(self, make_handler):
        """Datasets of another file type stay undecided."""
        assert make_handler()._dataset_availability(None, {"file_type": "other"}) is None

    def test_sounder_channel_of_single_band_file(self, make_handler):
        """A single-band file has no sounder channels."""
        assert not make_handler()._vas_channel_available({"vas_channel": 8})

    def test_dataset_without_channel_is_available(self, make_handler):
        """Datasets that name no sounder channel need no channel calibration."""
        assert make_handler()._vas_channel_available({})


class TestScene:
    """Load data through a Scene with the YAML reader definition."""

    @pytest.fixture
    def files(self, tmp_path):
        """Empty files with the names of an IR and a VIS image."""
        paths = [tmp_path / "goes03.1992.236.1301.AREA_IR", tmp_path / "goes03.1992.236.1301.AREA_VIS"]
        for path in paths:
            path.touch()
        return [str(path) for path in paths]

    @staticmethod
    def _fakes(monkeypatch, ir, vis=None):
        fakes = {"AREA_IR": ir, "AREA_VIS": vis}
        monkeypatch.setattr(handler_module, "AreaFile", lambda filename: fakes[str(filename).rsplit(".", 1)[1]])

    def test_load_ir_and_vis(self, files, monkeypatch):
        """Load calibrated bands with their coordinates and areas."""
        self._fakes(monkeypatch, FakeAreaFile(), FakeAreaFile(band="VIS"))
        scn = Scene(filenames=files, reader="goes_pregvar_area")
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

    def test_available_dataset_names(self, files, monkeypatch):
        """List the sounder channels, but not the generic IR dataset, for a multiband file."""
        cal = {"iab": {8: (1, 2, 3)}}
        self._fakes(monkeypatch, FakeAreaFile(source_type="AAA", aaa_cal=cal))
        scn = Scene(filenames=files[:1], reader="goes_pregvar_area")
        names = scn.available_dataset_names()
        assert "VAS_CH8" in names
        assert "IR" not in names
        assert "VAS_CH7" not in names

    def test_load_sounder_channel(self, files, monkeypatch):
        """Load a sounder channel in brightness temperature."""
        cal = {"iab": {8: (1, 2, 3)}}
        self._fakes(monkeypatch, FakeAreaFile(source_type="AAA", aaa_cal=cal))
        scn = Scene(filenames=files[:1], reader="goes_pregvar_area")
        scn.load(["VAS_CH8"])
        res = scn["VAS_CH8"]
        assert isinstance(res, xr.DataArray)
        assert res.attrs["units"] == "K"
        assert (res.values == BT_AAA).all()


def test_real_pyarea_file(tmp_path):
    """Read a real (tiny) Mode A IR file through pyarea: the fake above must match what pyarea does."""
    path = tmp_path / "goes07.1992.236.1301.AREA_IR"
    path.write_bytes(gzip.decompress(base64.b64decode(REAL_MODE_A_IR)))
    scn = Scene(filenames=[str(path)], reader="goes_pregvar_area")
    assert "IR" in scn.available_dataset_names()
    scn.load(["IR", "longitude"])
    ir = scn["IR"]
    assert ir.shape == (16, 24)
    assert ir.attrs["platform_name"] == "GOES-7"
    assert ir.attrs["units"] == "K"
    # counts 0, 100, 175, 176, 200, 255 by the documented VISSR formula: 330 - c / 2 below 176, 418 - c from there
    np.testing.assert_allclose(ir.values[8, :6], [330.0, 280.0, 242.5, 242.0, 218.0, 163.0])
    assert np.isfinite(scn["longitude"].values).all()
    assert ir.attrs["start_time"].isoformat() == "1992-08-23T13:01:00"
