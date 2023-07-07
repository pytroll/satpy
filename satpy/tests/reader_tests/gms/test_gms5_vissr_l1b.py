"""Unit tests for GMS-5 VISSR reader."""

import datetime as dt
import gzip

import fsspec
import numpy as np
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition

import satpy.tests.reader_tests.gms.test_gms5_vissr_data as real_world
from satpy.readers import FSFile
from satpy.tests.reader_tests.utils import get_jit_methods, skip_numba_unstable_if_missing
from satpy.tests.utils import make_dataid

try:
    import satpy.readers.gms.gms5_vissr_format as fmt
    import satpy.readers.gms.gms5_vissr_l1b as vissr
    import satpy.readers.gms.gms5_vissr_navigation as nav
except ImportError as err:
    if skip_numba_unstable_if_missing():
        pytest.skip(f"Numba is not compatible with unstable NumPy: {err!s}", allow_module_level=True)
    raise


@pytest.fixture(params=[False, True], autouse=True)
def disable_jit(request, monkeypatch):
    """Run tests with jit enabled and disabled.

    Reason: Coverage report is only accurate with jit disabled.
    """
    if request.param:
        jit_methods = get_jit_methods(vissr)
        for name, method in jit_methods.items():
            monkeypatch.setattr(name, method.py_func)


class TestEarthMask:
    """Test getting the earth mask."""

    def test_get_earth_mask(self):
        """Test getting the earth mask."""
        first_earth_pixels = np.array([-1, 1, 0, -1])
        last_earth_pixels = np.array([-1, 3, 2, -1])
        edges = first_earth_pixels, last_earth_pixels
        mask_exp = np.array(
            [[0, 0, 0, 0],
             [0, 1, 1, 1],
             [1, 1, 1, 0],
             [0, 0, 0, 0]]
        )
        mask = vissr.get_earth_mask(mask_exp.shape, edges)
        np.testing.assert_equal(mask, mask_exp)


class TestFileHandler:
    """Test VISSR file handler."""

    @pytest.fixture(autouse=True)
    def patch_number_of_pixels_per_scanline(self, monkeypatch):
        """Patch data types so that each scanline has two pixels."""
        num_pixels = 2
        IMAGE_DATA_BLOCK_IR = np.dtype(
            [
                ("LCW", fmt.LINE_CONTROL_WORD),
                ("DOC", fmt.U1, (256,)),
                ("image_data", fmt.U1, num_pixels),
            ]
        )
        IMAGE_DATA_BLOCK_VIS = np.dtype(
            [
                ("LCW", fmt.LINE_CONTROL_WORD),
                ("DOC", fmt.U1, (64,)),
                ("image_data", fmt.U1, (num_pixels,)),
            ]
        )
        IMAGE_DATA = {
            fmt.VIS_CHANNEL: {
                "offset": 6 * fmt.BLOCK_SIZE_VIS,
                "dtype": IMAGE_DATA_BLOCK_VIS,
            },
            fmt.IR_CHANNEL: {
                "offset": 18 * fmt.BLOCK_SIZE_IR,
                "dtype": IMAGE_DATA_BLOCK_IR,
            },
        }
        monkeypatch.setattr(
            "satpy.readers.gms.gms5_vissr_format.IMAGE_DATA_BLOCK_IR", IMAGE_DATA_BLOCK_IR
        )
        monkeypatch.setattr(
            "satpy.readers.gms.gms5_vissr_format.IMAGE_DATA_BLOCK_VIS", IMAGE_DATA_BLOCK_VIS
        )
        monkeypatch.setattr("satpy.readers.gms.gms5_vissr_format.IMAGE_DATA", IMAGE_DATA)

    @pytest.fixture(
        params=[
            make_dataid(name="VIS", calibration="reflectance", resolution=1250),
            make_dataid(
                name="IR1", calibration="brightness_temperature", resolution=5000
            ),
            make_dataid(name="IR1", calibration="counts", resolution=5000),
        ]
    )
    def dataset_id(self, request):
        """Get dataset ID."""
        return request.param

    @pytest.fixture(params=[True, False])
    def mask_space(self, request):
        """Mask space pixels."""
        return request.param

    @pytest.fixture(params=[True, False])
    def with_compression(self, request):
        """Enable compression."""
        return request.param

    @pytest.fixture
    def open_function(self, with_compression):
        """Get open function for writing test files."""
        return gzip.open if with_compression else open

    @pytest.fixture
    def vissr_file(self, dataset_id, file_contents, open_function, tmp_path):
        """Get test VISSR file."""
        filename = tmp_path / "vissr_file"
        ch_type = fmt.CHANNEL_TYPES[dataset_id["name"]]
        writer = VissrFileWriter(ch_type, open_function)
        writer.write(filename, file_contents)
        return filename

    @pytest.fixture
    def file_contents(self, control_block, image_parameters, image_data):
        """Get VISSR file contents."""
        return {
            "control_block": control_block,
            "image_parameters": image_parameters,
            "image_data": image_data,
        }

    @pytest.fixture
    def control_block(self, dataset_id):
        """Get VISSR control block."""
        block_size = {"IR1": 16, "VIS": 4}
        ctrl_block = np.zeros(1, dtype=fmt.CONTROL_BLOCK)
        ctrl_block["parameter_block_size"] = block_size[dataset_id["name"]]
        ctrl_block["available_block_size_of_image_data"] = 2
        return ctrl_block

    @pytest.fixture
    def image_parameters(self, mode_block, cal_params, nav_params):
        """Get VISSR image parameters."""
        image_params = {"mode": mode_block}
        image_params.update(cal_params)
        image_params.update(nav_params)
        return image_params

    @pytest.fixture
    def nav_params(
        self,
        coordinate_conversion,
        attitude_prediction,
        orbit_prediction,
    ):
        """Get navigation parameters."""
        nav_params = {}
        nav_params.update(attitude_prediction)
        nav_params.update(orbit_prediction)
        nav_params.update(coordinate_conversion)
        return nav_params

    @pytest.fixture
    def cal_params(
        self,
        vis_calibration,
        ir1_calibration,
        ir2_calibration,
        wv_calibration,
    ):
        """Get calibration parameters."""
        return {
            "vis_calibration": vis_calibration,
            "ir1_calibration": ir1_calibration,
            "ir2_calibration": ir2_calibration,
            "wv_calibration": wv_calibration,
        }

    @pytest.fixture
    def mode_block(self):
        """Get VISSR mode block."""
        mode = np.zeros(1, dtype=fmt.MODE_BLOCK)
        mode["satellite_name"] = b"GMS-5       "
        mode["spin_rate"] = 99.21774
        mode["observation_time_mjd"] = 50000.0
        mode["ssp_longitude"] = 140.0
        mode["satellite_height"] = 123456.0
        mode["ir_frame_parameters"]["number_of_lines"] = 2
        mode["ir_frame_parameters"]["number_of_pixels"] = 2
        mode["vis_frame_parameters"]["number_of_lines"] = 2
        mode["vis_frame_parameters"]["number_of_pixels"] = 2
        return mode

    @pytest.fixture
    def coordinate_conversion(self, coord_conv, simple_coord_conv_table):
        """Get all coordinate conversion parameters."""
        return {
            "coordinate_conversion": coord_conv,
            "simple_coordinate_conversion_table": simple_coord_conv_table
        }

    @pytest.fixture
    def coord_conv(self):
        """Get parameters for coordinate conversions.

        Adjust pixel offset so that the first column is at the image center.
        This has the advantage that we can test with very small 2x2 images.
        Otherwise, all pixels would be in space.
        """
        conv = np.zeros(1, dtype=fmt.COORDINATE_CONVERSION_PARAMETERS)

        cline = conv["central_line_number_of_vissr_frame"]
        cline["IR1"] = 1378.5
        cline["VIS"] = 5513.0

        cpix = conv["central_pixel_number_of_vissr_frame"]
        cpix["IR1"] = 0.5  # instead of 1672.5
        cpix["VIS"] = 0.5  # instead of 6688.5

        conv['scheduled_observation_time'] = 50130.979089568464

        nsensors = conv["number_of_sensor_elements"]
        nsensors["IR1"] = 1
        nsensors["VIS"] = 4

        sampling_angle = conv["sampling_angle_along_pixel"]
        sampling_angle["IR1"] = 9.5719995e-05
        sampling_angle["VIS"] = 2.3929999e-05

        stepping_angle = conv["stepping_angle_along_line"]
        stepping_angle["IR1"] = 0.00014000005
        stepping_angle["VIS"] = 3.5000005e-05

        conv["matrix_of_misalignment"] = np.array(
            [[9.9999917e-01, -5.1195198e-04, -1.2135329e-03],
             [5.1036407e-04, 9.9999905e-01, -1.3083406e-03],
             [1.2142011e-03, 1.3077201e-03, 9.9999845e-01]],
            dtype=np.float32
        )

        conv["parameters"]["equatorial_radius"] = 6377397.0
        conv["parameters"]["oblateness_of_earth"] = 0.003342773

        conv["orbital_parameters"]["longitude_of_ssp"] = 141.0
        conv["orbital_parameters"]["latitude_of_ssp"] = 1.0
        return conv

    @pytest.fixture
    def attitude_prediction(self):
        """Get attitude prediction."""
        att_pred = np.zeros(1, dtype=fmt.ATTITUDE_PREDICTION)
        att_pred["data"] = real_world.ATTITUDE_PREDICTION
        return {"attitude_prediction": att_pred}

    @pytest.fixture
    def orbit_prediction(self, orbit_prediction_1, orbit_prediction_2):
        """Get predictions of orbital parameters."""
        return {
            "orbit_prediction_1": orbit_prediction_1,
            "orbit_prediction_2": orbit_prediction_2
        }

    @pytest.fixture
    def orbit_prediction_1(self):
        """Get first block of orbit prediction data."""
        orb_pred = np.zeros(1, dtype=fmt.ORBIT_PREDICTION)
        orb_pred["data"] = real_world.ORBIT_PREDICTION_1
        return orb_pred

    @pytest.fixture
    def orbit_prediction_2(self):
        """Get second block of orbit prediction data."""
        orb_pred = np.zeros(1, dtype=fmt.ORBIT_PREDICTION)
        orb_pred["data"] = real_world.ORBIT_PREDICTION_2
        return orb_pred

    @pytest.fixture
    def vis_calibration(self):
        """Get VIS calibration block."""
        vis_cal = np.zeros(1, dtype=fmt.VIS_CALIBRATION)
        table = vis_cal["vis1_calibration_table"]["brightness_albedo_conversion_table"]
        table[0, 0:4] = np.array([0, 0.25, 0.5, 1])
        return vis_cal

    @pytest.fixture
    def ir1_calibration(self):
        """Get IR1 calibration block."""
        cal = np.zeros(1, dtype=fmt.IR_CALIBRATION)
        table = cal["conversion_table_of_equivalent_black_body_temperature"]
        table[0, 0:4] = np.array([0, 100, 200, 300])
        return cal

    @pytest.fixture
    def ir2_calibration(self):
        """Get IR2 calibration block."""
        cal = np.zeros(1, dtype=fmt.IR_CALIBRATION)
        return cal

    @pytest.fixture
    def wv_calibration(self):
        """Get WV calibration block."""
        cal = np.zeros(1, dtype=fmt.IR_CALIBRATION)
        return cal

    @pytest.fixture
    def simple_coord_conv_table(self):
        """Get simple coordinate conversion table."""
        table = np.zeros(1, dtype=fmt.SIMPLE_COORDINATE_CONVERSION_TABLE)
        table["satellite_height"] = 123457.0
        return table

    @pytest.fixture
    def image_data(self, dataset_id, image_data_ir1, image_data_vis):
        """Get VISSR image data."""
        data = {"IR1": image_data_ir1, "VIS": image_data_vis}
        return data[dataset_id["name"]]

    @pytest.fixture
    def image_data_ir1(self):
        """Get IR1 image data."""
        image_data = np.zeros(2, fmt.IMAGE_DATA_BLOCK_IR)
        image_data["LCW"]["line_number"] = [686, 2089]
        image_data["LCW"]["scan_time"] = [50000, 50000]
        image_data["LCW"]["west_side_earth_edge"] = [0, 0]
        image_data["LCW"]["east_side_earth_edge"] = [1, 1]
        image_data["image_data"] = [[0, 1], [2, 3]]
        return image_data

    @pytest.fixture
    def image_data_vis(self):
        """Get VIS image data."""
        image_data = np.zeros(2, fmt.IMAGE_DATA_BLOCK_VIS)
        image_data["LCW"]["line_number"] = [2744, 8356]
        image_data["LCW"]["scan_time"] = [50000, 50000]
        image_data["LCW"]["west_side_earth_edge"] = [-1, 0]
        image_data["LCW"]["east_side_earth_edge"] = [-1, 1]
        image_data["image_data"] = [[0, 1], [2, 3]]
        return image_data

    @pytest.fixture
    def vissr_file_like(self, vissr_file, with_compression):
        """Get file-like object for VISSR test file."""
        if with_compression:
            open_file = fsspec.open(vissr_file, compression="gzip")
            return FSFile(open_file)
        return vissr_file

    @pytest.fixture
    def file_handler(self, vissr_file_like, mask_space):
        """Get file handler to be tested."""
        return vissr.GMS5VISSRFileHandler(
            vissr_file_like, {}, {}, mask_space=mask_space
        )

    @pytest.fixture
    def vis_refl_exp(self, mask_space, lons_lats_exp):
        """Get expected VIS reflectance."""
        lons, lats = lons_lats_exp
        if mask_space:
            data = [[np.nan, np.nan], [50, 100]]
        else:
            data = [[0, 25], [50, 100]]
        return xr.DataArray(
            data,
            dims=("y", "x"),
            coords={
                "lon": lons,
                "lat": lats,
                "acq_time": (
                    "y",
                    [dt.datetime(1995, 10, 10), dt.datetime(1995, 10, 10)],
                ),
                "line_number": ("y", [2744, 8356]),
            },
        )

    @pytest.fixture
    def ir1_counts_exp(self, lons_lats_exp):
        """Get expected IR1 counts."""
        lons, lats = lons_lats_exp
        return xr.DataArray(
            [[0, 1], [2, 3]],
            dims=("y", "x"),
            coords={
                "lon": lons,
                "lat": lats,
                "acq_time": (
                    "y",
                    [dt.datetime(1995, 10, 10), dt.datetime(1995, 10, 10)],
                ),
                "line_number": ("y", [686, 2089]),
            },
        )

    @pytest.fixture
    def ir1_bt_exp(self, lons_lats_exp):
        """Get expected IR1 brightness temperature."""
        lons, lats = lons_lats_exp
        return xr.DataArray(
            [[0, 100], [200, 300]],
            dims=("y", "x"),
            coords={
                "lon": lons,
                "lat": lats,
                "acq_time": (
                    "y",
                    [dt.datetime(1995, 10, 10), dt.datetime(1995, 10, 10)],
                ),
                "line_number": ("y", [686, 2089]),
            },
        )

    @pytest.fixture
    def lons_lats_exp(self, dataset_id):
        """Get expected lon/lat coordinates.

        Computed with JMA's Msial library for 2 pixels near the central column
        (6688.5/1672.5 for VIS/IR).

        VIS:

        pix = [6688, 6688, 6689, 6689]
        lin = [2744, 8356, 2744, 8356]

        IR1:

        pix = [1672, 1672, 1673, 1673]
        lin = [686, 2089, 686, 2089]
        """
        expectations = {
            "IR1": {
                "lons": [[139.680120, 139.718902],
                         [140.307367, 140.346062]],
                "lats": [[35.045132, 35.045361],
                         [-34.971012, -34.970738]]
            },
            "VIS": {
                "lons": [[139.665133, 139.674833],
                         [140.292579, 140.302249]],
                "lats": [[35.076113, 35.076170],
                         [-34.940439, -34.940370]]
            }
        }
        exp = expectations[dataset_id["name"]]
        lons = xr.DataArray(exp["lons"], dims=("y", "x"))
        lats = xr.DataArray(exp["lats"], dims=("y", "x"))
        return lons, lats

    @pytest.fixture
    def dataset_exp(self, dataset_id, ir1_counts_exp, ir1_bt_exp, vis_refl_exp):
        """Get expected dataset."""
        ir1_counts_id = make_dataid(name="IR1", calibration="counts", resolution=5000)
        ir1_bt_id = make_dataid(
            name="IR1", calibration="brightness_temperature", resolution=5000
        )
        vis_refl_id = make_dataid(
            name="VIS", calibration="reflectance", resolution=1250
        )
        expectations = {
            ir1_counts_id: ir1_counts_exp,
            ir1_bt_id: ir1_bt_exp,
            vis_refl_id: vis_refl_exp,
        }
        return expectations[dataset_id]

    @pytest.fixture
    def area_def_exp(self, dataset_id):
        """Get expected area definition."""
        if dataset_id["name"] == "IR1":
            resol = 5
            size = 2366
            extent = (-20438.1468, -20438.1468, 20455.4306, 20455.4306)
        else:
            resol = 1
            size = 9464
            extent = (-20444.6235, -20444.6235, 20448.9445, 20448.9445)
        area_id = f"gms-5_vissr_western-pacific_{resol}km"
        desc = f"GMS-5 VISSR Western Pacific area definition with {resol} km resolution"
        return AreaDefinition(
            area_id=area_id,
            description=desc,
            proj_id=area_id,
            projection={
                "a": nav.EARTH_EQUATORIAL_RADIUS,
                "b": nav.EARTH_POLAR_RADIUS,
                "h": "123456",
                "lon_0": "140",
                "no_defs": "None",
                "proj": "geos",
                "type": "crs",
                "units": "m",
                "x_0": "0",
                "y_0": "0",
            },
            area_extent=extent,
            width=size,
            height=size,
        )

    @pytest.fixture
    def attrs_exp(self, area_def_exp):
        """Get expected dataset attributes."""
        return {
            "yaml": "info",
            "platform": "GMS-5",
            "sensor": "VISSR",
            "time_parameters": {
                "nominal_start_time": dt.datetime(1995, 10, 10),
                "nominal_end_time": dt.datetime(1995, 10, 10, 0, 25),
            },
            "orbital_parameters": {
                "satellite_nominal_longitude": 140.0,
                "satellite_nominal_latitude": 0.0,
                "satellite_nominal_altitude": 123456.0,
                "satellite_actual_longitude": 141.0,
                "satellite_actual_latitude": 1.0,
                "satellite_actual_altitude": 123457.0,
            },
            "area_def_uniform_sampling": area_def_exp,
        }

    def test_get_dataset(self, file_handler, dataset_id, dataset_exp, attrs_exp):
        """Test getting the dataset."""
        dataset = file_handler.get_dataset(dataset_id, {"yaml": "info"})
        xr.testing.assert_allclose(dataset.compute(), dataset_exp, atol=1e-6)
        assert dataset.attrs == attrs_exp

    def test_time_attributes(self, file_handler, attrs_exp):
        """Test the file handler's time attributes."""
        start_time_exp = attrs_exp["time_parameters"]["nominal_start_time"]
        end_time_exp = attrs_exp["time_parameters"]["nominal_end_time"]
        assert file_handler.start_time == start_time_exp
        assert file_handler.end_time == end_time_exp


class TestCorruptFile:
    """Test reading corrupt files."""

    @pytest.fixture
    def file_contents(self):
        """Get corrupt file contents (all zero)."""
        control_block = np.zeros(1, dtype=fmt.CONTROL_BLOCK)
        image_data = np.zeros(1, dtype=fmt.IMAGE_DATA_BLOCK_IR)
        return {
            "control_block": control_block,
            "image_parameters": {},
            "image_data": image_data,
        }

    @pytest.fixture
    def corrupt_file(self, file_contents, tmp_path):
        """Write corrupt VISSR file to disk."""
        filename = tmp_path / "my_vissr_file"
        writer = VissrFileWriter(ch_type="VIS", open_function=open)
        writer.write(filename, file_contents)
        return filename

    def test_corrupt_file(self, corrupt_file):
        """Test reading a corrupt file."""
        with pytest.raises(ValueError, match=r".* corrupt .*"):
            vissr.GMS5VISSRFileHandler(corrupt_file, {}, {})


class VissrFileWriter:
    """Write data in VISSR archive format."""

    image_params_order = [
        "mode",
        "coordinate_conversion",
        "attitude_prediction",
        "orbit_prediction_1",
        "orbit_prediction_2",
        "vis_calibration",
        "ir1_calibration",
        "ir2_calibration",
        "wv_calibration",
        "simple_coordinate_conversion_table",
    ]

    def __init__(self, ch_type, open_function):
        """Initialize the writer.

        Args:
            ch_type: Channel type (VIS or IR)
            open_function: Open function to be used (e.g. open or gzip.open)
        """
        self.ch_type = ch_type
        self.open_function = open_function

    def write(self, filename, contents):
        """Write file contents to disk."""
        with self.open_function(filename, mode="wb") as fd:
            self._write_control_block(fd, contents)
            self._write_image_parameters(fd, contents)
            self._write_image_data(fd, contents)

    def _write_control_block(self, fd, contents):
        self._write(fd, contents["control_block"])

    def _write_image_parameters(self, fd, contents):
        for name in self.image_params_order:
            im_param = contents["image_parameters"].get(name)
            if im_param:
                self._write_image_parameter(fd, im_param, name)

    def _write_image_parameter(self, fd, im_param, name):
        offset = fmt.IMAGE_PARAMS[name]["offset"][self.ch_type]
        self._write(fd, im_param, offset)

    def _write_image_data(self, fd, contents):
        offset = fmt.IMAGE_DATA[self.ch_type]["offset"]
        self._write(fd, contents["image_data"], offset)

    def _write(self, fd, data, offset=None):
        """Write data to file.

        If specified, prepend with 'offset' placeholder bytes.
        """
        if offset:
            self._fill(fd, offset)
        fd.write(data.tobytes())

    def _fill(self, fd, target_byte):
        """Write placeholders from current position to target byte."""
        nbytes = target_byte - fd.tell()
        fd.write(b" " * nbytes)
