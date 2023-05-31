"""Unit tests for GMS-5 VISSR reader."""

import datetime as dt
import gzip

import fsspec
import numpy as np
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition

import satpy.readers.gms5_vissr_l1b as vissr
import satpy.tests.reader_tests.test_gms5_vissr_data as real_world
from satpy.readers import FSFile
from satpy.tests.reader_tests.utils import get_jit_methods
from satpy.tests.utils import make_dataid


@pytest.fixture(params=[False, True], autouse=True)
def disable_jit(request, monkeypatch):
    """Run tests with jit enabled and disabled.

    Reason: Coverage report is only accurate with jit disabled.
    """
    if request.param:
        jit_methods = get_jit_methods(vissr)
        for name, method in jit_methods.items():
            monkeypatch.setattr(
                name,
                method.py_func
            )


class TestEarthMask:
    """Test getting the earth mask."""

    def test_get_earth_mask(self):
        """Test getting the earth mask."""
        first_earth_pixels = np.array([-1, 1, 0, -1])
        last_earth_pixels = np.array([-1, 3, 2, -1])
        edges = first_earth_pixels, last_earth_pixels
        # fmt: off
        mask_exp = np.array(
            [[0, 0, 0, 0],
             [0, 1, 1, 1],
             [1, 1, 1, 0],
             [0, 0, 0, 0]]
        )
        # fmt: on
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
                ("LCW", vissr.LINE_CONTROL_WORD),
                ("DOC", vissr.U1, (256,)),
                ("image_data", vissr.U1, num_pixels),
            ]
        )
        IMAGE_DATA_BLOCK_VIS = np.dtype(
            [
                ("LCW", vissr.LINE_CONTROL_WORD),
                ("DOC", vissr.U1, (64,)),
                ("image_data", vissr.U1, (num_pixels,)),
            ]
        )
        IMAGE_DATA = {
            vissr.VIS_CHANNEL: {
                "offset": 6 * vissr.BLOCK_SIZE_VIS,
                "dtype": IMAGE_DATA_BLOCK_VIS,
            },
            vissr.IR_CHANNEL: {
                "offset": 18 * vissr.BLOCK_SIZE_IR,
                "dtype": IMAGE_DATA_BLOCK_IR,
            },
        }
        monkeypatch.setattr(
            "satpy.readers.gms5_vissr_l1b.IMAGE_DATA_BLOCK_IR", IMAGE_DATA_BLOCK_IR
        )
        monkeypatch.setattr(
            "satpy.readers.gms5_vissr_l1b.IMAGE_DATA_BLOCK_VIS", IMAGE_DATA_BLOCK_VIS
        )
        monkeypatch.setattr("satpy.readers.gms5_vissr_l1b.IMAGE_DATA", IMAGE_DATA)

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
        ch_type = vissr.CHANNEL_TYPES[dataset_id["name"]]
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
        ctrl_block = np.zeros(1, dtype=vissr.CONTROL_BLOCK)
        ctrl_block["parameter_block_size"] = block_size[dataset_id["name"]]
        ctrl_block["available_block_size_of_image_data"] = 2
        return ctrl_block

    @pytest.fixture
    def image_parameters(
        self,
        mode_block,
        coordinate_conversion,
        attitude_prediction,
        orbit_prediction_1,
        orbit_prediction_2,
        vis_calibration,
        ir1_calibration,
        ir2_calibration,
        wv_calibration,
        simple_coordinate_conversion_table,
    ):
        """Get VISSR image parameters."""
        return {
            "mode": mode_block,
            "coordinate_conversion": coordinate_conversion,
            "attitude_prediction": attitude_prediction,
            "orbit_prediction_1": orbit_prediction_1,
            "orbit_prediction_2": orbit_prediction_2,
            "vis_calibration": vis_calibration,
            "ir1_calibration": ir1_calibration,
            "ir2_calibration": ir2_calibration,
            "wv_calibration": wv_calibration,
            "simple_coordinate_conversion_table": simple_coordinate_conversion_table,
        }

    @pytest.fixture
    def mode_block(self):
        """Get VISSR mode block."""
        mode = np.zeros(1, dtype=vissr.MODE_BLOCK)
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
    def coordinate_conversion(self):
        """Get parameters for coordinate conversions.

        Adjust pixel offset so that the first column is at the image center.
        This has the advantage that we can test with very small 2x2 images.
        Otherwise, all pixels would be in space.
        """
        # fmt: off
        conv = np.zeros(1, dtype=vissr.COORDINATE_CONVERSION_PARAMETERS)

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
        # fmt: on
        return conv

    @pytest.fixture
    def attitude_prediction(self):
        """Get attitude prediction."""
        att_pred = np.zeros(1, dtype=vissr.ATTITUDE_PREDICTION)
        att_pred["data"] = real_world.ATTITUDE_PREDICTION
        return att_pred

    @pytest.fixture
    def orbit_prediction_1(self):
        """Get first block of orbit prediction data."""
        orb_pred = np.zeros(1, dtype=vissr.ORBIT_PREDICTION)
        orb_pred["data"] = real_world.ORBIT_PREDICTION_1
        return orb_pred

    @pytest.fixture
    def orbit_prediction_2(self):
        """Get second block of orbit prediction data."""
        orb_pred = np.zeros(1, dtype=vissr.ORBIT_PREDICTION)
        orb_pred["data"] = real_world.ORBIT_PREDICTION_2
        return orb_pred

    @pytest.fixture
    def vis_calibration(self):
        """Get VIS calibration block."""
        vis_cal = np.zeros(1, dtype=vissr.VIS_CALIBRATION)
        table = vis_cal["vis1_calibration_table"]["brightness_albedo_conversion_table"]
        table[0, 0:4] = np.array([0, 0.25, 0.5, 1])
        return vis_cal

    @pytest.fixture
    def ir1_calibration(self):
        """Get IR1 calibration block."""
        cal = np.zeros(1, dtype=vissr.IR_CALIBRATION)
        table = cal["conversion_table_of_equivalent_black_body_temperature"]
        table[0, 0:4] = np.array([0, 100, 200, 300])
        return cal

    @pytest.fixture
    def ir2_calibration(self):
        """Get IR2 calibration block."""
        cal = np.zeros(1, dtype=vissr.IR_CALIBRATION)
        return cal

    @pytest.fixture
    def wv_calibration(self):
        """Get WV calibration block."""
        cal = np.zeros(1, dtype=vissr.IR_CALIBRATION)
        return cal

    @pytest.fixture
    def simple_coordinate_conversion_table(self):
        """Get simple coordinate conversion table."""
        table = np.zeros(1, dtype=vissr.SIMPLE_COORDINATE_CONVERSION_TABLE)
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
        image_data = np.zeros(2, vissr.IMAGE_DATA_BLOCK_IR)
        image_data["LCW"]["line_number"] = [686, 2089]
        image_data["LCW"]["scan_time"] = [50000, 50000]
        image_data["LCW"]["west_side_earth_edge"] = [0, 0]
        image_data["LCW"]["east_side_earth_edge"] = [1, 1]
        image_data["image_data"] = [[0, 1], [2, 3]]
        return image_data

    @pytest.fixture
    def image_data_vis(self):
        """Get VIS image data."""
        image_data = np.zeros(2, vissr.IMAGE_DATA_BLOCK_VIS)
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
            data = [[np.nan, np.nan], [0.5, 1]]
        else:
            data = [[0, 0.25], [0.5, 1]]
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
        # fmt: off
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
        # fmt: on
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
            extent = (
                -8.641922536247211,
                -8.641922536247211,
                25.925767608741637,
                25.925767608741637,
            )
        else:
            resol = 1
            extent = (
                -2.1604801323784297,
                -2.1604801323784297,
                6.481440397135289,
                6.481440397135289,
            )
        area_id = f"gms-5_vissr_western-pacific_{resol}km"
        desc = f"GMS-5 VISSR Western Pacific area definition with {resol} km resolution"
        return AreaDefinition(
            area_id=area_id,
            description=desc,
            proj_id=area_id,
            projection={
                "ellps": "SGS85",
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
            width=2,
            height=2,
        )

    @pytest.fixture
    def attrs_exp(self, area_def_exp):
        """Get expected dataset attributes."""
        return {
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
        dataset = file_handler.get_dataset(dataset_id, None)
        xr.testing.assert_allclose(dataset.compute(), dataset_exp, atol=1e-6)
        self._assert_attrs_equal(dataset.attrs, attrs_exp)

    def test_time_attributes(self, file_handler, attrs_exp):
        """Test the file handler's time attributes."""
        start_time_exp = attrs_exp["time_parameters"]["nominal_start_time"]
        end_time_exp = attrs_exp["time_parameters"]["nominal_end_time"]
        assert file_handler.start_time == start_time_exp
        assert file_handler.end_time == end_time_exp

    def _assert_attrs_equal(self, attrs_tst, attrs_exp):
        area_tst = attrs_tst.pop("area_def_uniform_sampling")
        area_exp = attrs_exp.pop("area_def_uniform_sampling")
        assert attrs_tst == attrs_exp
        self._assert_areas_close(area_tst, area_exp)

    def _assert_areas_close(self, area_tst, area_exp):
        lons_tst, lats_tst = area_tst.get_lonlats()
        lons_exp, lats_exp = area_exp.get_lonlats()
        np.testing.assert_allclose(lons_tst, lons_exp)
        np.testing.assert_allclose(lats_tst, lats_exp)


class TestCorruptFile:
    """Test reading corrupt files."""

    @pytest.fixture
    def file_contents(self):
        """Get corrupt file contents (all zero)."""
        control_block = np.zeros(1, dtype=vissr.CONTROL_BLOCK)
        image_data = np.zeros(1, dtype=vissr.IMAGE_DATA_BLOCK_IR)
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
        with pytest.raises(ValueError, match=r'.* corrupt .*'):
            vissr.GMS5VISSRFileHandler(corrupt_file, {}, {})


class VissrFileWriter:
    """Write data in VISSR archive format."""

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
        for key, im_param in contents["image_parameters"].items():
            offset = vissr.IMAGE_PARAMS[key]["offset"][self.ch_type]
            self._write(fd, im_param, offset)

    def _write_image_data(self, fd, contents):
        offset = vissr.IMAGE_DATA[self.ch_type]["offset"]
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
