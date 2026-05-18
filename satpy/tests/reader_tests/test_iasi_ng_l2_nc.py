# Copyright (c) 2022 Satpy developers
#
# satpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# satpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Unit tests for the IASI NG L2 reader using temporary NetCDF files."""
import os
import re
from datetime import datetime

import h5py
import numpy as np
import pytest
import xarray as xr

from satpy import Scene
from satpy.readers.core.loading import load_reader
from satpy.readers.iasi_ng_l2_nc import IASINGL2NCFileHandler

d_lff = ("n_lines", "n_for", "n_fov")

DATA_DESC = [
    {
        "group": "data/diagnostics",
        "attrs": ["missing_value"],
        "variables": {
            "fg_cost": [d_lff, "float32", (3.4e38,)],
            "nbr_iterations": [d_lff, "int32", (2147483647,)],
            "rt_cost_x": [d_lff, "float32", (3.4e38,)],
            "rt_cost_y": [d_lff, "float32", (3.4e38,)],
        },
    },
    {
        "group": "data/geolocation_information",
        "attrs": [
            "units",
            "valid_min",
            "valid_max",
            "scale_factor",
            "add_offset",
            "missing_value",
        ],
        "variables": {
            "onboard_utc": [
                ("n_lines", "n_for"),
                "float64",
                ("seconds since 2020-01-01 00:00:00.000", -1e9, 1e9, None, None, -9e9),
            ],
            "sounder_pixel_latitude": [
                d_lff,
                "int32",
                ("degrees_north", -18e8, 18e8, 5.0e-8, 0.0, -2147483648),
            ],
            "sounder_pixel_longitude": [
                d_lff,
                "int32",
                ("degrees_east", -18.432e8, 18.432e8, 9.765625e-8, 0.0, -2147483648),
            ],
            "sounder_pixel_sun_azimuth": [
                d_lff,
                "int32",
                ("degrees", -18.432e8, 18.432e8, 9.765625e-8, 0.0, -2147483648),
            ],
            "sounder_pixel_sun_zenith": [
                d_lff,
                "int32",
                ("degrees", -18e8, 18e8, 5.0e-8, 90.0, -2147483648),
            ],
            "sounder_pixel_azimuth": [
                d_lff,
                "int32",
                ("degrees", -18.432e8, 18.432e8, 9.765625e-8, 0.0, -2147483648),
            ],
            "sounder_pixel_zenith": [
                d_lff,
                "int32",
                ("degrees", -18e8, 18e8, 2.5e-8, 45.0, -2147483648),
            ],
        },
    },
    {
        "group": "data/statistical_retrieval",
        "attrs": ["units", "valid_min", "valid_max", "missing_value"],
        "variables": {
            "air_temperature": [d_lff, "float32", ("K", 100.0, 400.0, 3.4e38)],
            "atmosphere_mass_content_of_water": [
                d_lff,
                "float32",
                ("kg.m-2", 0, 300, 3.4e38),
            ],
            "qi_air_temperature": [d_lff, "float32", ("", 0, 25, 3.4e38)],
            "qi_specific_humidity": [d_lff, "float32", ("", 0, 25, 3.4e38)],
            "specific_humidity": [d_lff, "float32", ("kg/kg", 0, 1, 3.4e38)],
            "surface_air_temperature": [d_lff, "float32", ("K", 100, 400, 3.4e38)],
            "surface_specific_humidity": [d_lff, "float32", ("K", 100, 400, 3.4e38)],
        },
    },
    {
        "group": "data/surface_info",
        "attrs": ["units", "valid_min", "valid_max", "missing_value"],
        "variables": {
            "height": [d_lff, "float32", ("m", -418, 8848, 3.4e38)],
            "height_std": [d_lff, "float32", ("m", 0, 999.0, 3.4e38)],
            "ice_fraction": [d_lff, "float32", ("%", 0, 100.0, 3.4e38)],
            "land_fraction": [d_lff, "float32", ("%", 0, 100.0, 3.4e38)],
        },
    },
]


def create_random_data(shape, dtype, attribs):
    """Create a random array with potentially missing values."""
    rng = np.random.default_rng()
    if dtype.startswith("float"):
        vmin = attribs.get("valid_min", -1e16)
        vmax = attribs.get("valid_max", 1e16)
        data = rng.uniform(vmin, vmax, shape).astype(dtype)
    else:
        vmin = attribs.get("valid_min", -2147483647)
        vmax = attribs.get("valid_max", 2147483647)
        data = rng.integers(vmin, vmax, shape, dtype=dtype)

    if "missing_value" in attribs:
        mask = rng.random(shape) < 0.05
        data[mask] = attribs["missing_value"]

    return data


def create_controlled_data(shape, dtype, attribs):
    """Create data with controlled values for testing scaling and masking."""
    data = np.zeros(shape, dtype=dtype)

    indices = np.indices(shape)
    rows, cols, fovs = indices

    if dtype.startswith("int"):
        valid_max = int(attribs.get("valid_max", 1000))
        data = ((rows * 100) + (cols * 10) + fovs) % valid_max
    else:
        valid_max = float(attribs.get("valid_max", 100.0))
        data = ((rows * 10.0) + (cols * 1.0) + (fovs / 10.0)) % valid_max

    if "missing_value" in attribs:
        data[0, 0, 0] = attribs["missing_value"]
        data[2, 2, 2] = attribs["missing_value"]

    return data


def get_valid_attribs(anames, alist):
    """Retrieve only the valid attributes from a list."""
    attribs = {}
    for idx, val in enumerate(alist):
        if val is not None:
            attribs[anames[idx]] = val

    return attribs


def add_dataset_variable(grp, var_config, dim_refs):
    """Add a variable to a given dataset group."""
    dnames = var_config["dims"]
    vname = var_config["name"]
    tname = var_config["dtype"]
    anames = var_config["allowed_attrs"]
    attribs = get_valid_attribs(anames, var_config["attrs"])

    shape = tuple([dim_refs[dname].size for dname in dnames])

    if vname in ["sounder_pixel_latitude", "sounder_pixel_longitude"]:
        arr = create_controlled_data(shape, tname, attribs)
    else:
        arr = create_random_data(shape, tname, attribs)

    dset = grp.create_dataset(vname, data=arr)

    for i, dname in enumerate(dnames):
        dset.dims[i].attach_scale(dim_refs[dname])

    for attr_name, attr_value in attribs.items():
        if attr_name in ["valid_min", "valid_max", "missing_value"]:
            attr_value = np.dtype(tname).type(attr_value)
        dset.attrs[attr_name] = attr_value


def write_fake_iasing_l2_file(output_path):
    """Create a fake IASI-NG L2 file with all required variables and attributes."""
    n_lines = 10
    n_for = 14
    n_fov = 16
    dims = {"n_lines": n_lines, "n_for": n_for, "n_fov": n_fov}

    with h5py.File(output_path, "w") as ds:
        grp = ds.create_group("data")
        dim_refs = {}
        for dim_name, size in dims.items():
            dim_dset = grp.create_dataset(dim_name, (size,), dtype="int32")
            dim_dset.make_scale(dim_name)
            dim_dset[:] = range(size)
            dim_refs[dim_name] = dim_dset

        for grp_desc in DATA_DESC:
            prefix = grp_desc["group"]
            grp = ds.create_group(prefix)

            for vname, vdesc in grp_desc["variables"].items():
                var_config = {
                    "name": vname,
                    "dims": vdesc[0],
                    "dtype": vdesc[1],
                    "attrs": vdesc[2],
                    "allowed_attrs": grp_desc["attrs"],
                }

                add_dataset_variable(grp, var_config, dim_refs)

    return os.fspath(output_path)


class TestIASINGL2NCReader:
    """Test class for the IASI NG L2 reader."""

    reader_name = "iasi_ng_l2_nc"
    file_prefix = "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-IAS-02"
    file_suffix = "C_EUMT_20170616120000_G_V_20070912084329_20070912084600_O_N____.nc"

    def setup_method(self):
        """Set up the reader configuration."""
        from satpy._config import config_search_paths

        self.reader_configs = config_search_paths(
            os.path.join("readers", self.reader_name + ".yaml")
        )

    @pytest.fixture
    def twv_handler(self, tmp_path):
        """Create a simple default handler on a TWV product."""
        twv_filename = f"{self.file_prefix}-TWV_{self.file_suffix}"
        output_path = write_fake_iasing_l2_file(tmp_path / twv_filename)

        return self._create_file_handler(output_path)

    @pytest.fixture
    def twv_scene(self, tmp_path):
        """Create a simple satpy scene on a TWV product."""
        twv_filename = f"{self.file_prefix}-TWV_{self.file_suffix}"
        output_path = write_fake_iasing_l2_file(tmp_path / twv_filename)
        return Scene(filenames=[output_path], reader=self.reader_name)

    def _create_file_handler(self, filename):
        """Create an handler for the given file checking that it can be parsed correctly."""
        reader = load_reader(self.reader_configs)

        files = reader.select_files_from_pathnames([filename])
        assert len(files) == 1, "File should be recognized by the reader"

        reader.create_filehandlers(files)

        assert len(reader.file_handlers) == 1

        assert self.reader_name in reader.file_handlers

        handlers = reader.file_handlers[self.reader_name]

        assert len(handlers) == 1
        assert isinstance(handlers[0], IASINGL2NCFileHandler)

        return handlers[0]

    def test_filename_matching(self, tmp_path):
        """Test filename matching against some random name."""
        prefix = "W_fr-meteo-sat,GRAL,MTI1-IASING-2"
        suffix = (
            "C_EUMS_20220101120000_LEO_O_D_20220101115425_20220101115728_____W______.nc"
        )
        filename = f"{prefix}-l2p_{suffix}"
        output_path = write_fake_iasing_l2_file(tmp_path / filename)

        self._create_file_handler(output_path)

    def test_real_filename_matching(self, tmp_path):
        """Test that we will match an actual IASI NG L2 product file name."""
        supported_types = ["TWV", "CLD", "GHG", "SFC", "O3_", "CO_"]

        for ptype in supported_types:
            filename = f"{self.file_prefix}-{ptype}_{self.file_suffix}"
            output_path = write_fake_iasing_l2_file(tmp_path / filename)
            handler = self._create_file_handler(output_path)

            assert handler.filename_info["oflag"] == "C"
            assert handler.filename_info["originator"] == "EUMT"
            assert handler.filename_info["product_type"] == ptype

    def test_sensing_times(self, twv_handler):
        """Test that we read the sensing start/end times correctly from filename."""
        assert twv_handler.start_time == datetime(2007, 9, 12, 8, 43, 29)
        assert twv_handler.end_time == datetime(2007, 9, 12, 8, 46, 0)

    def test_sensor_names(self, twv_handler):
        """Test that the handler reports iasi_ng as sensor."""
        assert twv_handler.sensor_names == {"iasi_ng"}

    def test_available_datasets(self, twv_scene):
        """Test the list of available datasets in scene."""
        dnames = twv_scene.available_dataset_names()

        expected_names = [
            "onboard_utc",
            "sounder_pixel_latitude",
            "sounder_pixel_longitude",
            "sounder_pixel_azimuth",
            "sounder_pixel_zenith",
            "sounder_pixel_sun_azimuth",
            "sounder_pixel_sun_zenith",
            "air_temperature",
            "atmosphere_mass_content_of_water",
            "qi_air_temperature",
            "qi_specific_humidity",
            "specific_humidity",
            "surface_air_temperature",
            "surface_specific_humidity",
        ]

        for dname in expected_names:
            assert dname in dnames

    def test_latlon_datasets(self, twv_scene):
        """Test loading the latitude/longitude dataset."""
        twv_scene.load(["sounder_pixel_latitude", "sounder_pixel_longitude"])
        lat = twv_scene["sounder_pixel_latitude"]
        lon = twv_scene["sounder_pixel_longitude"]

        assert lat.dims == ("n_lines", "n_for", "n_fov")
        assert lon.dims == ("n_lines", "n_for", "n_fov")

        assert lat.dtype == np.float64
        assert lon.dtype == np.float64

        assert np.nanmin(lat) >= -90.0
        assert np.nanmax(lat) <= 90.0

        assert np.nanmin(lon) >= -180.0
        assert np.nanmax(lon) <= 180.0

    def test_scaling_latitude(self, tmp_path):
        """Test that latitude values are correctly scaled and masked."""
        twv_filename = f"{self.file_prefix}-TWV_{self.file_suffix}"
        output_path = write_fake_iasing_l2_file(tmp_path / twv_filename)

        scene = Scene(filenames=[output_path], reader=self.reader_name)
        scene.load(["sounder_pixel_latitude"])
        lat = scene["sounder_pixel_latitude"]

        with h5py.File(output_path, "r") as f:
            raw_lat = f["data/geolocation_information/sounder_pixel_latitude"][:]
            scale_factor = f["data/geolocation_information/sounder_pixel_latitude"].attrs[
                "scale_factor"
            ]
            add_offset = f["data/geolocation_information/sounder_pixel_latitude"].attrs[
                "add_offset"
            ]
            missing_value = f[
                "data/geolocation_information/sounder_pixel_latitude"
            ].attrs["missing_value"]

        assert np.isnan(lat.values[0, 0, 0])

        valid_point = np.where(raw_lat != missing_value)
        if len(valid_point[0]) > 0:
            i, j, k = valid_point[0][0], valid_point[1][0], valid_point[2][0]
            expected_value = raw_lat[i, j, k] * scale_factor + add_offset
            np.testing.assert_almost_equal(lat.values[i, j, k], expected_value)

        assert lat.attrs["units"] == "degrees_north"

    def test_mask_propagation(self, twv_scene):
        """Test that masks are properly propagated to derived datasets."""
        twv_scene.load(["sounder_pixel_latitude", "sounder_pixel_longitude"])
        lat = twv_scene["sounder_pixel_latitude"]
        lon = twv_scene["sounder_pixel_longitude"]

        assert np.isnan(lat.values[0, 0, 0])
        assert np.isnan(lon.values[0, 0, 0])

        assert np.isnan(lat.values[2, 2, 2])
        assert np.isnan(lon.values[2, 2, 2])

    def test_onboard_utc_dataset(self, twv_scene):
        """Test loading the onboard_utc dataset."""
        twv_scene.load(["onboard_utc", "sounder_pixel_latitude"])
        dset = twv_scene["onboard_utc"]

        assert dset.dims == ("n_lines", "n_for")

        assert dset.dtype == np.dtype("datetime64[ns]")

    def test_nbr_iterations_dataset(self, twv_scene):
        """Test loading the nbr_iterations dataset."""
        twv_scene.load(["nbr_iterations"])
        dset = twv_scene["nbr_iterations"]

        assert len(dset.dims) == 3
        assert dset.dtype == np.float64

    def test_register_dataset(self, twv_handler):
        """Test the register_dataset method."""
        twv_handler.dataset_infos = {}
        twv_handler.register_dataset("test_dataset", {"attr": "value"})
        assert "test_dataset" in twv_handler.dataset_infos
        assert twv_handler.dataset_infos["test_dataset"]["attr"] == "value"

        with pytest.raises(KeyError):
            twv_handler.register_dataset("test_dataset", {"attr": "new_value"})

    def test_process_dimension(self, twv_handler):
        """Test the process_dimension method."""
        twv_handler.dimensions_desc = {}
        twv_handler.process_dimension("group/dimension/test_dim", 10)
        assert twv_handler.dimensions_desc["test_dim"] == 10

        with pytest.raises(KeyError):
            twv_handler.process_dimension("group/dimension/test_dim", 20)

    def test_process_attribute(self, twv_handler):
        """Test the process_attribute method."""
        twv_handler.variable_desc = {"test_var": {"attribs": {}}}
        twv_handler.process_attribute("test_var/attr/test_attr", "value")
        assert twv_handler.variable_desc["test_var"]["attribs"]["test_attr"] == "value"

    def test_process_variable(self, twv_handler):
        """Test the process_variable method."""
        twv_handler.file_content = {
            "group/test_var/shape": (10, 10),
            "group/test_var/dimensions": ("x", "y"),
            "group/test_var/dtype": "float32",
        }
        twv_handler.ignored_patterns = []
        twv_handler.process_variable("group/test_var")
        assert "group/test_var" in twv_handler.variable_desc

    def test_ignore_scalar_variable(self, twv_handler):
        """Test ignoring of scalar variable."""
        twv_handler.file_content = {
            "group/test_var/shape": (1, 1),
            "group/test_var/dimensions": ("x", "y"),
            "group/test_var/dtype": "float32",
        }
        twv_handler.ignored_patterns = []
        twv_handler.process_variable("group/test_var")
        assert "group/test_var" not in twv_handler.variable_desc

    def test_ignore_pattern_variable(self, twv_handler):
        """Test ignoring of pattern in variable."""
        twv_handler.file_content = {
            "group/test_var/shape": (10, 10),
            "group/test_var/dimensions": ("x", "y"),
            "group/test_var/dtype": "float32",
        }
        twv_handler.ignored_patterns = [re.compile(r"test_")]
        twv_handler.process_variable("group/test_var")
        assert "group/test_var" not in twv_handler.variable_desc

    def test_parse_file_content(self, twv_handler):
        """Test the parse_file_content method."""
        twv_handler.file_content = {
            "dim/dimension/test_dim": 10,
            "var/attr/test_attr": "value",
            "grp/test_var": "my_data",
            "grp/test_var/shape": (10, 10),
            "grp/test_var/dimensions": ("x", "y"),
            "grp/test_var/dtype": "float32",
        }

        twv_handler.parse_file_content()
        assert "test_dim" in twv_handler.dimensions_desc
        assert "grp/test_var" in twv_handler.variable_desc

    def test_register_available_datasets(self, twv_handler):
        """Test the register_available_datasets method."""
        twv_handler.dataset_infos = None
        twv_handler.variable_desc = {
            "var/test_var": {"var_name": "test_var", "attribs": {"units": "test_unit"}}
        }
        twv_handler.dataset_aliases = {}
        twv_handler.register_available_datasets()
        assert "test_var" in twv_handler.dataset_infos

    def test_ignored_register_available_datasets(self, twv_handler):
        """Test ignoring register_available_datasets method if done already."""
        twv_handler.variable_desc = {
            "var/test_var": {"var_name": "test_var", "attribs": {"units": "test_unit"}}
        }
        twv_handler.dataset_aliases = {}
        twv_handler.register_available_datasets()
        assert "test_var" not in twv_handler.dataset_infos

    def test_register_available_datasets_alias(self, twv_handler):
        """Test the register_available_datasets method with alias."""
        twv_handler.dataset_infos = None
        twv_handler.variable_desc = {
            "var/test_var": {"var_name": "test_var", "attribs": {"units": "test_unit"}}
        }
        twv_handler.dataset_aliases = {re.compile(r"var/(.+)$"): "${VAR_NAME}_oem"}
        twv_handler.register_available_datasets()
        assert "test_var_oem" in twv_handler.dataset_infos

    def test_get_dataset_infos(self, twv_handler):
        """Test the get_dataset_infos method."""
        twv_handler.dataset_infos = {"test_dataset": {"attr": "value"}}
        assert twv_handler.get_dataset_infos("test_dataset") == {"attr": "value"}

        with pytest.raises(KeyError):
            twv_handler.get_dataset_infos("non_existent_dataset")

    def test_variable_path_exists(self, twv_handler):
        """Test the variable_path_exists method."""
        twv_handler.file_content = {"test_var": "dummy"}
        assert twv_handler.variable_path_exists("test_var")
        assert not twv_handler.variable_path_exists("/attr/test_var")
        assert not twv_handler.variable_path_exists("test_var/dtype")
        assert not twv_handler.variable_path_exists("/grp/a_non_existing_var")

    def test_convert_to_datetime(self, twv_handler):
        """Test the convert_to_datetime method."""
        data = xr.DataArray(np.array([0, 86400]), dims=("time",))
        ds_info = {"seconds_since_epoch": "2000-01-01 00:00:00"}
        result = twv_handler.convert_to_datetime(data, ds_info)
        assert result[0].values == np.datetime64("2000-01-01T00:00:00")
        assert result[1].values == np.datetime64("2000-01-02T00:00:00")

    def test_get_transformed_dataset(self, twv_handler):
        """Test the get_transformed_dataset method."""
        ds_info = {
            "location": "test_var",
            "seconds_since_epoch": "2000-01-01 00:00:00",
        }
        twv_handler.variable_path_exists = lambda _arg: True
        twv_handler.file_content["test_var"] = xr.DataArray(
            np.array([[0, 86400]]), dims=("x", "y")
        )

        twv_handler.dimensions_desc = {"n_fov": 2}

        result = twv_handler.get_transformed_dataset(ds_info)
        assert result.shape == (1, 2)
        assert result.dtype == "datetime64[ns]"

    def test_get_transformed_dataset_failure(self, twv_handler):
        """Test the get_transformed_dataset method fails on invalid path."""
        ds_info = {
            "location": "test_var",
            "seconds_since_epoch": "2000-01-01 00:00:00",
        }

        with pytest.raises(KeyError):
            twv_handler.get_transformed_dataset(ds_info)

    def test_get_dataset(self, twv_handler):
        """Test the get_dataset method."""
        twv_handler.dataset_infos = {
            "test_dataset": {"name": "test_dataset", "location": "test_var"}
        }
        twv_handler.get_transformed_dataset = lambda _arg: xr.DataArray([1, 2, 3])

        result = twv_handler.get_dataset({"name": "test_dataset"})
        assert result.equals(xr.DataArray([1, 2, 3]))

        assert twv_handler.get_dataset({"name": "non_existent_dataset"}) is None
