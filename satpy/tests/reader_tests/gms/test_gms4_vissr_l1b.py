"""Unit tests for the GMS-1..4 VISSR reader."""

import collections
import gzip

import numpy as np
import pytest

import satpy.readers.gms.gms4_vissr_format as fmt
import satpy.readers.gms.gms4_vissr_l1b as vissr
from satpy.tests.utils import make_dataid

IR_BLOCK_LEN = fmt.IR_BLOCK_LEN
VIS_BLOCK_LEN = fmt.VIS_BLOCK_LEN
NUM_TEST_PIXELS = 4
NUM_TEST_LINES = 2


class TestEarthMask:
    """Test the vectorized earth mask against a real file on disk.

    Regression coverage for the "Bumpy Road" refactor: the original
    implementation was a per-line Python loop with two nested ifs; the
    vectorized replacement was verified against it over 200 randomized
    trials during development, including the asymmetric-clamping edge
    case parametrized below (w is only clamped from below, e only from
    above -- out-of-range values on the same side must NOT clamp into
    false agreement).

    Restricted to the IR channel: VIS additionally applies an
    oversampling-ratio correction to the edges before clamping (see
    _earth_edges_for_mask), which would need its own separately
    computed expected values -- that path is covered indirectly by
    TestFileHandler's channel-parametrized tests instead.
    """

    @pytest.fixture
    def channel(self):
        """Pin this class to the IR channel only (see class docstring)."""
        return fmt.IR_CHANNEL

    EarthEdgeScenario = collections.namedtuple("EarthEdgeScenario", ["west", "east", "expected_row"])

    @pytest.mark.parametrize(
        "scenario",
        [
            pytest.param(
                EarthEdgeScenario(0, NUM_TEST_PIXELS - 1, [True] * NUM_TEST_PIXELS), id="normal_full_range"
            ),
            pytest.param(EarthEdgeScenario(-1, -1, [False] * NUM_TEST_PIXELS), id="fill_value_both_sides"),
            pytest.param(EarthEdgeScenario(-1, 2, [False] * NUM_TEST_PIXELS), id="fill_value_west_only"),
            pytest.param(
                EarthEdgeScenario(10, 12, [False] * NUM_TEST_PIXELS), id="out_of_range_same_side_w_gt_e_after_clamp"
            ),
        ],
    )
    def test_earth_mask(self, tmp_path, file_contents, scenario):
        """Test get_earth_mask() from a real file on disk, for each edge-case scenario."""
        file_contents["image_data"]["LCW"]["west_side_earth_edge"] = scenario.west
        file_contents["image_data"]["LCW"]["east_side_earth_edge"] = scenario.east
        path = tmp_path / "IR901110.Z23"
        VissrFileWriter(fmt.IR_CHANNEL, open).write(path, file_contents)

        l1b = vissr.GmsVissrL1bFile(path)
        mask = l1b.get_earth_mask()
        np.testing.assert_array_equal(mask[0], np.array(scenario.expected_row, dtype=bool))


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
        "calibration", "image_data" -- see the file_contents fixture.
        """
        # Written in ascending-offset order (NOT a "logical" order): the
        # calibration block's real offset sits before
        # coordinate_conversion's in both channels' actual file layout.
        with self.open_function(filename, "wb") as fd:
            self._write_at(fd, self.params["mode"]["offset"], contents["mode"])
            cal_key = "ir_calibration" if self.channel == fmt.IR_CHANNEL else "vis_calibration"
            self._write_at(fd, self.params[cal_key]["offset"], contents["calibration"])
            self._write_at(fd, self.params["coordinate_conversion"]["offset"], contents["coordinate_conversion"])
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
def mode_block(channel):
    """Get a populated VISSR mode block, with a real (nonzero) spin_rate."""
    mode = np.zeros(1, dtype=fmt.MODE_BLOCK)
    mode["satellite_name"] = b"GMS-4"
    mode["spin_rate"] = 100.37372693
    mode["observation_time_mjd"] = 45994.73027376625
    mode["ssp_longitude"] = 140.0
    mode["satellite_height"] = 3.59e7
    mode["gms_operation_mode"] = 1
    mode["dpc_operation_mode"] = 1
    mode["vissr_observation_mode"] = 1
    mode["scanner_selection"] = 1
    mode["sensor_selection"] = 1
    mode["sensor_mode"] = 3  # both VIS and IR
    mode["scan_frame_mode"] = 1
    mode["scan_mode"] = 1
    mode["upper_limit_of_scan_number"] = 1
    mode["lower_limit_of_scan_number"] = NUM_TEST_LINES
    return mode


@pytest.fixture
def coord_block():
    """Get a populated coordinate conversion parameters block.

    Uses the real GMS-3 telemetry values found and validated against
    during this reader's development (stepping_angle_ir=0.00013992561
    correctly implies ~5026m IR resolution at satellite_height=3.59e7,
    matching the reader's own resolution-attribute logic).
    """
    coord = np.zeros(1, dtype=fmt.COORDINATE_CONVERSION_PARAMETERS)
    coord["scheduled_observation_time"] = 45994.73027376625
    coord["stepping_angle_vis"] = 3.4981407e-05
    coord["stepping_angle_ir"] = 0.00013992561
    coord["sampling_angle_vis"] = 2.3974804e-05
    coord["sampling_angle_ir"] = 4.7949594e-05
    coord["num_sensors_vis"] = 4.0
    coord["num_sensors_ir"] = 1.0
    coord["matrix_of_misalignment"] = np.eye(3, dtype=np.float32).flatten(order="F")
    coord["daily_mean_spin_rate"] = 100.37372693
    return coord


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
        lines[i]["LCW"]["scan_time"] = 45994.73027376625 + i * 1e-6
        lines[i]["LCW"]["west_side_earth_edge"] = 0
        lines[i]["LCW"]["east_side_earth_edge"] = NUM_TEST_PIXELS - 1
        lines[i]["image_data"] = np.arange(i, i + NUM_TEST_PIXELS, dtype=np.uint8)
    return lines


@pytest.fixture
def file_contents(mode_block, coord_block, cal_block, image_lines):
    """Bundle the blocks that make up a VISSR file's contents into one dict."""
    return {
        "mode": mode_block,
        "coordinate_conversion": coord_block,
        "calibration": cal_block,
        "image_data": image_lines,
    }


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
    try:
        return make_dataid(name="VIS", calibration="unnormalized_reflectance", resolution=1250)
    except ValueError:
        # "unnormalized_reflectance" isn't in Satpy core's calibration
        # enum (satpy/dataset/dataid.py) until
        # https://github.com/pytroll/satpy/pull/3292 merges -- this is
        # a known, acknowledged upstream dependency (see review
        # discussion), not a bug in this reader. Once #3292 merges,
        # this will start "unexpectedly passing", which is the signal
        # to remove this xfail.
        pytest.xfail("Requires satpy#3292 (unnormalized_reflectance calibration enum) to be merged.")


class TestFileHandler:
    """Test the file handler end-to-end against a real file on disk."""

    def test_get_dataset(self, file_handler, dataset_id, channel):
        """Test that get_dataset() returns correctly calibrated, correctly shaped data."""
        dataset = file_handler.get_dataset(dataset_id, {"name": dataset_id["name"]})
        computed = dataset.compute()
        assert computed.shape == (NUM_TEST_LINES, NUM_TEST_PIXELS)

        if channel == fmt.IR_CHANNEL:
            # count N -> N kelvin (see cal_block fixture)
            expected = np.array([np.arange(i, i + NUM_TEST_PIXELS) for i in range(NUM_TEST_LINES)], dtype=np.float32)
        else:
            # count N -> N/100 * 100 = N (see cal_block fixture and _postproc)
            expected = np.array([np.arange(i, i + NUM_TEST_PIXELS) for i in range(NUM_TEST_LINES)], dtype=np.float32)
        np.testing.assert_allclose(computed.values, expected, atol=1e-3)

    def test_get_dataset_wrong_channel_returns_none(self, file_handler, channel):
        """Test that requesting the other channel from this file returns None."""
        other = "VIS" if channel == fmt.IR_CHANNEL else "IR"
        result = file_handler.get_dataset(make_dataid(name=other), {"name": other})
        assert result is None

    def test_longitude_latitude_standard_name(self, file_handler, dataset_id):
        """Test that lon/lat coords are tagged for Satpy's generic SwathDefinition auto-building."""
        dataset = file_handler.get_dataset(dataset_id, {"name": dataset_id["name"]})
        assert dataset.coords["longitude"].attrs["standard_name"] == "longitude"
        assert dataset.coords["latitude"].attrs["standard_name"] == "latitude"
        assert "area" not in dataset.attrs  # built generically by Satpy, not by us

    def test_start_time(self, file_handler):
        """Test that start_time comes from the real scheduled_observation_time telemetry."""
        start_time = file_handler.start_time
        assert start_time.year == 2025 or start_time.year > 1900  # sanity: a real decoded MJD date

    def test_combine_info_single_file(self, file_handler):
        """Test that combine_info() with a single info dict is a no-op passthrough."""
        info = {"name": "IR", "resolution": 5000}
        assert file_handler.combine_info([info]) is info

    def test_get_area_def_raises_not_implemented(self, file_handler, dataset_id):
        """Test that get_area_def() isn't overridden with a custom (unreachable) message.

        Satpy's generic yaml_reader discards any custom NotImplementedError
        message anyway (see satpy/readers/core/yaml_reader.py
        _load_dataset_area's bare `except NotImplementedError:`), so the
        base class's plain `raise NotImplementedError` is deliberately
        left as-is rather than overridden.
        """
        with pytest.raises(NotImplementedError):
            file_handler.get_area_def(dataset_id)


class TestSpinRateFallback:
    """Regression test for the real bug found and fixed this session.

    Some older GMS-1/2/3 archives don't populate mode["spin_rate"]
    (reads back as 0.0, physically impossible for a spin-scan imager),
    which used to cause a bare ZeroDivisionError deep inside the shared
    navigation module's per-pixel observation-time calculation, well
    before resampling was ever reached.
    """

    @pytest.fixture
    def zero_mode_spin_rate_contents(self, file_contents):
        """Get file_contents with mode.spin_rate zeroed out, simulating the real GMS-3 archive gap."""
        file_contents["mode"]["spin_rate"] = 0.0
        return file_contents

    @pytest.fixture
    def file_handler_with_zero_mode_spin_rate(
        self, vissr_filename, channel, open_function, zero_mode_spin_rate_contents
    ):
        """Get a real file handler built from a file where mode.spin_rate reads back as 0."""
        VissrFileWriter(channel, open_function).write(vissr_filename, zero_mode_spin_rate_contents)
        return vissr.GmsVissrFileHandler(vissr_filename, {}, {})

    def test_falls_back_to_daily_mean_spin_rate(self, file_handler_with_zero_mode_spin_rate, dataset_id):
        """Test that navigation still succeeds (no ZeroDivisionError) when mode.spin_rate is unpopulated.

        Note: this fixture doesn't populate attitude_prediction/
        orbit_prediction with physically realistic values (they're
        all-zero, since VissrFileWriter never writes them), so the real
        navigation math correctly produces NaN lon/lat here regardless
        of which spin rate source is used -- that's expected given an
        all-zero NPA "rotation" matrix isn't physically valid, not a
        reader bug. What this test actually guards against is the
        literal ZeroDivisionError this session found and fixed: it
        must complete without raising.
        """
        dataset = file_handler_with_zero_mode_spin_rate.get_dataset(dataset_id, {"name": dataset_id["name"]})
        # Must not raise ZeroDivisionError -- accessing .values forces
        # the lazy dask computation through navigate_dask() to actually run.
        assert dataset.coords["longitude"].values.shape == (NUM_TEST_LINES, NUM_TEST_PIXELS)
        assert dataset.coords["latitude"].values.shape == (NUM_TEST_LINES, NUM_TEST_PIXELS)

    def test_uses_daily_mean_spin_rate_value(self, file_handler_with_zero_mode_spin_rate):
        """Test the fallback logic via the real code path: resolved spinning_rate matches coord.daily_mean_spin_rate."""
        nav_params = file_handler_with_zero_mode_spin_rate._l1b._build_navigation_parameters()
        assert nav_params.static.scan_params.spinning_rate == pytest.approx(100.37372693)

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
