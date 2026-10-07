"""Unit tests for the GMS-1..4 VISSR reader."""

import datetime as dt
import gzip

import numpy as np
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition

import satpy.readers.gms.gms4_vissr_format as fmt
import satpy.readers.gms.gms4_vissr_l1b as vissr
import satpy.readers.gms.gms_vissr_navigation as nav
import satpy.tests.reader_tests.gms.test_gms4_vissr_data as real_world
from satpy import Scene
from satpy.tests.utils import make_dataid

IR_BLOCK_LEN = fmt.IR_BLOCK_LEN
VIS_BLOCK_LEN = fmt.VIS_BLOCK_LEN
NUM_TEST_PIXELS = 4
NUM_TEST_LINES = 2


@pytest.fixture(params=[True, False])
def with_compression(request):
    """Enable gzip compression for the written test file."""
    return request.param


@pytest.fixture
def open_function(with_compression):
    """Get the open function used to write the test file."""
    return gzip.open if with_compression else open


@pytest.fixture(params=[fmt.IR_CHANNEL, fmt.VIS_CHANNEL])
def channel(request):
    """Parametrize over both channels."""
    return request.param


@pytest.fixture(autouse=True)
def _patch_num_pixels(monkeypatch):
    """Shrink the per-line pixel count so test files stay small.

    Only the per-line dtypes are patched -- IR_BLOCK_LEN/VIS_BLOCK_LEN
    and the IMAGE_PARAMS_*/IMAGE_DATA offset dicts are computed once at
    module-import time from the *original* constants, so they can't be
    shrunk for tests without breaking every other offset in the file;
    this mirrors how satpy's own gms5 tests handle the same constraint.
    """
    image_data_block_ir = np.dtype([
        ("LCW", fmt.LINE_CONTROL_WORD),
        ("DOC", fmt.U1, (256,)),
        ("image_data", fmt.U1, (NUM_TEST_PIXELS,)),
    ])
    image_data_block_vis = np.dtype([
        ("LCW", fmt.LINE_CONTROL_WORD),
        ("DOC", fmt.U1, (64,)),
        ("image_data", fmt.U1, (NUM_TEST_PIXELS,)),
    ])
    monkeypatch.setitem(fmt.IMAGE_DATA[fmt.IR_CHANNEL], "dtype", image_data_block_ir)
    monkeypatch.setitem(fmt.IMAGE_DATA[fmt.VIS_CHANNEL], "dtype", image_data_block_vis)


class VissrFileWriter:
    """Write a real GMS-1..4 VISSR archive file to disk, at the format's real absolute offsets."""

    def __init__(self, channel, open_function):
        """Store the channel and open function (plain open or gzip.open) to write with."""
        self.channel = channel
        self.open_function = open_function
        self.params = fmt.IMAGE_DATA[channel]["params"]
        self.image_data_offset = fmt.IMAGE_DATA[channel]["offset"]
        self.image_data_dtype = fmt.IMAGE_DATA[channel]["dtype"]

    def write(self, filename, contents):
        """Write mode/coordinate-conversion/calibration blocks and image data lines to *filename*.

        *contents* is a dict with keys "mode", "coordinate_conversion",
        "calibration", "image_data" -- see the file_contents fixture -- and
        optionally "attitude_prediction", "orbit_prediction_1" and
        "orbit_prediction_2" (all zero in the file otherwise).
        """
        # Written in ascending-offset order (NOT a "logical" order): the
        # calibration block's real offset sits before
        # coordinate_conversion's in both channels' actual file layout.
        with self.open_function(filename, "wb") as fd:
            self._write_at(fd, self.params["mode"]["offset"], contents["mode"])
            cal_key = "ir_calibration" if self.channel == fmt.IR_CHANNEL else "vis_calibration"
            self._write_at(fd, self.params[cal_key]["offset"], contents["calibration"])
            self._write_at(fd, self.params["coordinate_conversion"]["offset"], contents["coordinate_conversion"])
            for key in ("attitude_prediction", "orbit_prediction_1", "orbit_prediction_2"):
                if key in contents:
                    self._write_at(fd, self.params[key]["offset"], contents[key])
            self._write_at(fd, self.image_data_offset, contents["image_data"])

    @staticmethod
    def _write_at(fd, offset, struct_array):
        pos = fd.tell()
        if offset > pos:
            fd.write(b"\x00" * (offset - pos))
        elif offset < pos:
            raise ValueError(f"Offset {offset} is before current file position {pos} -- write order is wrong.")
        fd.write(struct_array.tobytes())


@pytest.fixture
def mode_block():
    """Get a mode block with the real spin rate, satellite position and satellite name."""
    mode = np.zeros(1, dtype=fmt.MODE_BLOCK)
    mode["satellite_name"] = real_world.SATELLITE_NAME.encode()
    for name, value in real_world.MODE.items():
        mode[name] = value
    return mode


@pytest.fixture
def coord_block(channel):
    """Get a coordinate conversion block with the real scanning geometry.

    The real image centre (line 1384, pixel 3344.5 for IR) lies far outside
    the tiny test image, which would be navigated to space (NaN). Therefore
    the centre is moved next to the test image. This keeps the navigation
    finite without changing anything else about the geometry.
    """
    coord = real_coord_block()
    suffix = "ir" if channel == fmt.IR_CHANNEL else "vis"
    coord[f"central_line_{suffix}"] = 1.0
    coord[f"central_pixel_{suffix}"] = 1.5
    return coord


def real_coord_block():
    """Get a coordinate conversion block with the unmodified real scanning geometry."""
    coord = np.zeros(1, dtype=fmt.COORDINATE_CONVERSION_PARAMETERS)
    for name, value in real_world.COORDINATE_CONVERSION.items():
        coord[name] = value
    return coord


@pytest.fixture
def navigation_blocks():
    """Get the real attitude and orbit predictions."""
    attitude = np.zeros(1, dtype=fmt.ATTITUDE_PREDICTION)
    attitude["data"] = real_world.ATTITUDE_PREDICTION
    orbit_1 = np.zeros(1, dtype=fmt.ORBIT_PREDICTION)
    orbit_1["data"] = real_world.ORBIT_PREDICTION_1
    orbit_2 = np.zeros(1, dtype=fmt.ORBIT_PREDICTION)
    orbit_2["data"] = real_world.ORBIT_PREDICTION_2
    return {"attitude_prediction": attitude, "orbit_prediction_1": orbit_1, "orbit_prediction_2": orbit_2}


@pytest.fixture
def cal_block(channel):
    """Get a populated calibration block with a simple, easy-to-check-by-hand LUT."""
    if channel == fmt.IR_CHANNEL:
        cal = np.zeros(1, dtype=fmt.IR_CALIBRATION)
        # count N -> N kelvin, so results are trivially predictable
        cal["conversion_table_of_equivalent_black_body_temperature"] = np.arange(256, dtype=np.float32)
        return cal
    cal = np.zeros(1, dtype=fmt.VIS_CALIBRATION)
    # count N -> N/100 (so *100 postproc below gives back N exactly)
    cal["vis1_calibration_table"]["brightness_albedo_conversion_table"] = np.arange(64, dtype=np.float32) / 100.0
    return cal


@pytest.fixture
def image_lines(channel):
    """Get NUM_TEST_LINES lines of image data, in pairs (the format stores 2 lines per raw block)."""
    dtype = fmt.IMAGE_DATA[channel]["dtype"]
    lines = np.zeros(NUM_TEST_LINES, dtype=dtype)
    for i in range(NUM_TEST_LINES):
        lines[i]["LCW"]["line_number"] = i + 1
        lines[i]["LCW"]["scan_time"] = real_world.COORDINATE_CONVERSION["scheduled_observation_time"] + i * 1e-6
        lines[i]["LCW"]["west_side_earth_edge"] = 0
        lines[i]["LCW"]["east_side_earth_edge"] = NUM_TEST_PIXELS - 1
        lines[i]["image_data"] = np.arange(i, i + NUM_TEST_PIXELS, dtype=np.uint8)
    return lines


@pytest.fixture
def header_blocks(mode_block, coord_block, cal_block, navigation_blocks):
    """Bundle the header blocks of a VISSR file into one dict."""
    return {
        "mode": mode_block,
        "coordinate_conversion": coord_block,
        "calibration": cal_block,
        **navigation_blocks,
    }


@pytest.fixture
def file_contents(header_blocks, image_lines):
    """Get the contents of a VISSR file: header blocks and image data."""
    return {**header_blocks, "image_data": image_lines}


@pytest.fixture
def vissr_filename(tmp_path, channel, with_compression):
    """Construct a real, channel-appropriately-prefixed test filename."""
    prefix = "IR" if channel == fmt.IR_CHANNEL else "VS"
    name = f"{prefix}901110.Z23"
    if with_compression:
        name += ".gz"
    return tmp_path / name


@pytest.fixture
def vissr_file(vissr_filename, channel, open_function, file_contents):
    """Write a real VISSR test file to disk and return its path."""
    VissrFileWriter(channel, open_function).write(vissr_filename, file_contents)
    return vissr_filename


@pytest.fixture
def file_handler(vissr_file):
    """Get a real file handler pointed at the on-disk test file."""
    return vissr.GmsVissrFileHandler(vissr_file, {}, {})


@pytest.fixture
def dataset_id(channel):
    """Get the dataset ID matching the test file's channel."""
    if channel == fmt.IR_CHANNEL:
        return make_dataid(name="IR", calibration="brightness_temperature", resolution=5000)
    return make_dataid(name="VIS", calibration="unnormalized_reflectance", resolution=1250)


def _expected_counts(channel):
    """Get the counts of the test image: line i holds i, i+1, ..."""
    return np.array([np.arange(i, i + NUM_TEST_PIXELS) for i in range(NUM_TEST_LINES)], dtype=np.float32)


def _expected_lons_lats(channel):
    """Get the longitudes and latitudes of the test image.

    Snapshot of the navigation of the real attitude/orbit predictions for the
    tiny test image (see the coord_block fixture), not an independent
    reference. They are symmetric in the sense that the longitude grows
    eastwards and the latitude shrinks southwards with increasing pixel
    and line number, with one scan step of about 0.045 (IR) / 0.011 (VIS)
    degrees between the lines.
    """
    if channel == fmt.IR_CHANNEL:
        lons = [[139.668, 139.683, 139.699, 139.714]] * 2
        lats = [[1.370] * NUM_TEST_PIXELS, [1.325] * NUM_TEST_PIXELS]
    else:
        lons = [[139.672, 139.680, 139.687, 139.695], [139.672, 139.679, 139.687, 139.695]]
        lats = [[1.370] * NUM_TEST_PIXELS, [1.359] * NUM_TEST_PIXELS]
    return np.array(lons, dtype=np.float32), np.array(lats, dtype=np.float32)


def _expected_area_def(dataset_id, channel):
    """Get the expected area definition: full disk, square, one pixel per scan step."""
    suffix = "ir" if channel == fmt.IR_CHANNEL else "vis"
    pixel_size = real_world.COORDINATE_CONVERSION[f"stepping_angle_{suffix}"] * real_world.MODE["satellite_height"]
    resolution = "5km" if channel == fmt.IR_CHANNEL else "1km"
    return AreaDefinition(
        f"gms-4_vissr_western-pacific_{resolution}",
        f"GMS-4 VISSR Western Pacific area definition with {resolution[:-2]} km resolution",
        f"gms-4_vissr_western-pacific_{resolution}",
        {
            "proj": "geos",
            "lon_0": real_world.MODE["ssp_longitude"],
            "h": real_world.MODE["satellite_height"],
            "a": nav.EARTH_EQUATORIAL_RADIUS,
            "b": nav.EARTH_POLAR_RADIUS,
            "units": "m",
        },
        NUM_TEST_LINES,
        NUM_TEST_LINES,
        (-0.5 * pixel_size, -0.5 * pixel_size, 1.5 * pixel_size, 1.5 * pixel_size),
    )


class TestFileHandler:
    """Test the file handler end-to-end against a real file on disk."""

    @pytest.fixture
    def ds_info(self, dataset_id):
        """Get the dataset info as provided by the YAML file."""
        return {"name": dataset_id["name"], "units": "K", "standard_name": "toa_brightness_temperature"}

    @pytest.fixture
    def expected(self, channel):
        """Get the expected dataset, calibrated with the test calibration table (see cal_block fixture)."""
        lons, lats = _expected_lons_lats(channel)
        return xr.DataArray(
            _expected_counts(channel),
            dims=("y", "x"),
            coords={"longitude": (("y", "x"), lons), "latitude": (("y", "x"), lats)},
        )

    def test_get_dataset(self, file_handler, dataset_id, ds_info, expected):
        """Test data, dimensions and coordinates of the calibrated dataset."""
        dataset = file_handler.get_dataset(dataset_id, ds_info)
        xr.testing.assert_allclose(dataset, expected, rtol=0, atol=1e-3)

    def test_get_dataset_counts(self, file_handler, dataset_id, ds_info, expected):
        """Test that counts are returned if requested."""
        counts_id = make_dataid(name=dataset_id["name"], calibration="counts", resolution=dataset_id["resolution"])
        dataset = file_handler.get_dataset(counts_id, ds_info)
        xr.testing.assert_allclose(dataset, expected, rtol=0, atol=1e-3)

    def test_get_dataset_attributes(self, file_handler, dataset_id, ds_info, channel):
        """Test the dataset attributes, especially the area definition."""
        dataset = file_handler.get_dataset(dataset_id, ds_info)
        assert dataset.name == dataset_id["name"]
        assert dataset.attrs["units"] == "K"
        assert dataset.attrs["standard_name"] == "toa_brightness_temperature"
        assert dataset.attrs["platform_name"] == "GMS-4"
        area_def = dataset.attrs["area_def_uniform_sampling"]
        expected_area_def = _expected_area_def(dataset_id, channel)
        assert area_def.area_id == expected_area_def.area_id
        assert area_def.description == expected_area_def.description
        assert area_def.shape == expected_area_def.shape
        assert area_def.crs == expected_area_def.crs
        np.testing.assert_allclose(area_def.area_extent, expected_area_def.area_extent, rtol=1e-6)

    def test_coordinate_attributes(self, file_handler, dataset_id, ds_info):
        """Test that lon/lat coords are tagged for Satpy's generic SwathDefinition auto-building."""
        dataset = file_handler.get_dataset(dataset_id, ds_info)
        for name in ("longitude", "latitude"):
            assert dataset.coords[name].attrs["standard_name"] == name
            assert dataset.coords[name].attrs["resolution"] > 0
        assert "area" not in dataset.attrs  # built generically by Satpy, not by us

    def test_get_dataset_wrong_channel_returns_none(self, file_handler, channel):
        """Test that requesting the other channel from this file returns None."""
        other = "VIS" if channel == fmt.IR_CHANNEL else "IR"
        result = file_handler.get_dataset(make_dataid(name=other), {"name": other})
        assert result is None

    def test_start_and_end_time(self, file_handler):
        """Test that the times are decoded from the scheduled observation time and the scan times."""
        mjd = real_world.COORDINATE_CONVERSION["scheduled_observation_time"]
        expected_start = dt.datetime(1858, 11, 17) + dt.timedelta(days=mjd)
        assert file_handler.start_time == expected_start
        assert dt.timedelta(0) <= file_handler.end_time - file_handler.start_time < dt.timedelta(seconds=1)

    def test_sensor_names(self, file_handler):
        """Test that the sensor name is the one defined in the YAML file."""
        assert file_handler.sensor_names == {"gms4-vissr"}

    def test_combine_info_single_file(self, file_handler):
        """Test that combine_info() with a single info dict is a no-op passthrough."""
        info = {"name": "IR", "resolution": 5000}
        assert file_handler.combine_info([info]) is info

    def test_get_area_def_raises_not_implemented(self, file_handler, dataset_id):
        """Test that get_area_def() isn't overridden: Satpy builds a SwathDefinition from the coordinates."""
        with pytest.raises(NotImplementedError):
            file_handler.get_area_def(dataset_id)


class TestSpaceMask:
    """Test masking of space with the earth edges of the line control words (IR channel only)."""

    @pytest.fixture
    def channel(self):
        """Pin this class to the IR channel."""
        return fmt.IR_CHANNEL

    def test_space_is_masked(self, vissr_filename, open_function, file_contents, dataset_id):
        """Test that pixels outside of the earth edges and lines without edges are NaN."""
        file_contents["image_data"]["LCW"]["west_side_earth_edge"] = [1, -1]
        file_contents["image_data"]["LCW"]["east_side_earth_edge"] = [2, -1]
        VissrFileWriter(fmt.IR_CHANNEL, open_function).write(vissr_filename, file_contents)

        handler = vissr.GmsVissrFileHandler(vissr_filename, {}, {})
        dataset = handler.get_dataset(dataset_id, {"name": "IR"})
        expected = np.array([[np.nan, 1, 2, np.nan], [np.nan] * 4], dtype=np.float32)
        np.testing.assert_array_equal(dataset.values, expected)


class TestScene:
    """Test the reader through the Scene, including the YAML file."""

    @pytest.fixture
    def scene(self, vissr_file):
        """Get a scene with the test file."""
        return Scene(filenames=[str(vissr_file)], reader="gms4-vissr_l1b")

    @pytest.mark.parametrize("calibration", ["counts", None])
    def test_load(self, scene, channel, calibration, file_handler):
        """Test loading a channel with the YAML attributes, in counts and in the default calibration."""
        name = "IR" if channel == fmt.IR_CHANNEL else "VIS"
        kwargs = {} if calibration is None else {"calibration": calibration}
        scene.load([name], **kwargs)
        dataset = scene[name]

        assert dataset.shape == (NUM_TEST_LINES, NUM_TEST_PIXELS)
        assert dataset.attrs["sensor"] == "gms4-vissr"
        assert dataset.attrs["start_time"] == file_handler.start_time
        assert dataset.attrs["end_time"] == file_handler.end_time
        if calibration == "counts":
            assert dataset.attrs["units"] == 1
            np.testing.assert_allclose(dataset.values, _expected_counts(channel), atol=1e-3)
        else:
            assert dataset.attrs["units"] == ("K" if channel == fmt.IR_CHANNEL else "%")


class TestSpinRateFallback:
    """Test the fallback to the daily mean spin rate.

    Some older GMS-1/2/3 archives don't populate mode["spin_rate"] (reads
    back as 0.0, physically impossible for a spin-scan imager), which used to
    cause a ZeroDivisionError in the per-pixel observation time calculation of
    the navigation module.
    """

    @pytest.fixture
    def zero_mode_spin_rate_contents(self, file_contents):
        """Get file_contents with mode.spin_rate zeroed out, simulating the real GMS-3 archive gap."""
        file_contents["mode"]["spin_rate"] = 0.0
        return file_contents

    @pytest.fixture
    def l1b_with_zero_mode_spin_rate(self, vissr_filename, channel, open_function, zero_mode_spin_rate_contents):
        """Get a file where mode.spin_rate reads back as 0."""
        VissrFileWriter(channel, open_function).write(vissr_filename, zero_mode_spin_rate_contents)
        return vissr.GmsVissrL1bFile(vissr_filename)

    def test_navigation_succeeds(self, l1b_with_zero_mode_spin_rate, channel):
        """Test that the navigation works without the spin rate of the mode block."""
        lons, lats = l1b_with_zero_mode_spin_rate.navigate_dask()
        assert np.all(np.isfinite(lons.compute()))
        assert np.all(np.isfinite(lats.compute()))

    def test_uses_daily_mean_spin_rate_value(self, l1b_with_zero_mode_spin_rate):
        """Test that the resolved spinning rate is the daily mean spin rate."""
        nav_params = l1b_with_zero_mode_spin_rate._build_navigation_parameters()
        daily_mean = real_world.COORDINATE_CONVERSION["daily_mean_spin_rate"]
        assert nav_params.static.scan_params.spinning_rate == pytest.approx(daily_mean)

    @pytest.fixture
    def no_spin_rate_vissr_file(self, vissr_filename, channel, open_function, zero_mode_spin_rate_contents):
        """Write a file where NEITHER spin rate source is populated."""
        zero_mode_spin_rate_contents["coordinate_conversion"]["daily_mean_spin_rate"] = 0.0
        VissrFileWriter(channel, open_function).write(vissr_filename, zero_mode_spin_rate_contents)
        return vissr_filename

    def test_raises_when_neither_spin_rate_source_is_populated(self, no_spin_rate_vissr_file, dataset_id):
        """Test that a clear error is raised (not a silent bad value) when both sources are missing."""
        handler = vissr.GmsVissrFileHandler(no_spin_rate_vissr_file, {}, {})
        with pytest.raises(ValueError, match="spin rate"):
            handler.get_dataset(dataset_id, {"name": dataset_id["name"]})


class TestRealWorldNavigation:
    """Test navigation with the real attitude/orbit predictions and geometry of a GMS archive file.

    The expected lon/lat values are a snapshot of what this reader computed
    for the real file the test data was extracted from (see
    test_gms4_vissr_data.py). The navigation was verified once by resampling
    that scene and overlaying coastlines, but the numbers are NOT an
    independent reference such as JMA's Msial, so this guards the navigation
    logic (parameter mapping, units, matrix order, orbit table assembly, spin
    rate handling) against regressions.

    It does not verify the header offsets in gms4_vissr_format: the test file
    is written at the same offsets the reader reads from. Those were checked
    against real GMS-1/GMS-3 files when the format was written.
    """

    @pytest.fixture
    def coord_block(self):
        """Get the unmodified real scanning geometry, including the real image centre."""
        return real_coord_block()

    def test_navigation_matches_snapshot(self, vissr_file, channel):
        """Test that lon/lat of the snapshot pixels are reproduced."""
        l1b = vissr.GmsVissrL1bFile(vissr_file)
        nav_params = l1b._build_navigation_parameters()
        snapshot = real_world.NAVIGATION_SNAPSHOT[channel]

        lons, lats = [], []
        for line, pixel, _, _ in snapshot:
            lon, lat = nav.get_lons_lats(np.array([float(line)]), np.array([float(pixel)]), nav_params)
            lons.append(float(np.asarray(lon)[0, 0]))
            lats.append(float(np.asarray(lat)[0, 0]))

        np.testing.assert_allclose(lons, [row[2] for row in snapshot], atol=1e-4)
        np.testing.assert_allclose(lats, [row[3] for row in snapshot], atol=1e-4)

    def test_orbit_prediction_assembly(self, vissr_file):
        """Test that the orbit table is the first 8 entries of block 1 followed by block 2."""
        l1b = vissr.GmsVissrL1bFile(vissr_file)
        nav_params = l1b._build_navigation_parameters()
        expected_times = np.concatenate([
            real_world.ORBIT_PREDICTION_1["prediction_time_mjd"][:8],
            real_world.ORBIT_PREDICTION_2["prediction_time_mjd"],
        ])
        np.testing.assert_array_equal(nav_params.predicted.orbit.prediction_times, expected_times)


class TestChannelDetection:
    """Test GmsVissrL1bFile's filename-based channel sniffing."""

    def test_detects_ir_from_prefix(self, tmp_path, file_contents):
        """Test that a filename starting with IR is detected as the IR channel."""
        path = tmp_path / "IR901110.Z23"
        VissrFileWriter(fmt.IR_CHANNEL, open).write(path, file_contents)
        l1b = vissr.GmsVissrL1bFile(path)
        assert l1b.channel == fmt.IR_CHANNEL

    def test_detects_vis_from_prefix(self, tmp_path, file_contents, channel):
        """Test that a filename starting with VS is detected as the VIS channel."""
        if channel != fmt.VIS_CHANNEL:
            pytest.skip("only meaningful for the VIS-shaped fixture data")
        path = tmp_path / "VS901110.Z23"
        VissrFileWriter(fmt.VIS_CHANNEL, open).write(path, file_contents)
        l1b = vissr.GmsVissrL1bFile(path)
        assert l1b.channel == fmt.VIS_CHANNEL
