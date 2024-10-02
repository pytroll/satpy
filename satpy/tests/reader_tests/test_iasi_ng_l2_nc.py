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

"""Unit tests on the IASI NG L2 reader using the conventional mock constructed context."""
import os
import re
from datetime import datetime
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from satpy import Scene
from satpy.readers import load_reader
from satpy.readers.iasi_ng_l2_nc import IASINGL2NCFileHandler
from satpy.tests.reader_tests.test_netcdf_utils import FakeNetCDF4FileHandler

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


class FakeIASINGFileHandlerBase(FakeNetCDF4FileHandler):
    """Fake base class for IASI NG handler."""

    chunks = (10, 10, 10)

    def inject_missing_value(self, dask_array, attribs):
        """Inject missing value in data array."""
        # Force setting a few elements to this missing value (but still lazily with
        # dask map_blocks function)
        missing_value = attribs["missing_value"]

        # Ratio of elements to replace with missing value:
        missing_ratio = 0.05

        def set_missing_values(block):
            # Generate random indices to set as missing values within this block
            block_shape = block.shape

            block_size = np.prod(block_shape)
            block_num_to_replace = int(block_size * missing_ratio)

            # Create a Generator instance
            rng = np.random.default_rng()

            # Generate unique random indices to set as missing values within this block
            flat_indices = rng.choice(block_size, block_num_to_replace, replace=False)
            unraveled_indices = np.unravel_index(flat_indices, block_shape)
            block[unraveled_indices] = missing_value

            return block

        # Apply the function lazily to each block
        dask_array = dask_array.map_blocks(set_missing_values, dtype=dask_array.dtype)

        return dask_array

    def generate_dask_array(self, dims, dtype, attribs):
        """Generate some random dask array of the given dtype."""
        shape = [self.dims[k] for k in dims]
        chunks = [10] * len(dims)

        if dtype == "int32":
            rand_min = -2147483647
            rand_max = 2147483647
            dask_array = da.random.randint(
                rand_min, rand_max, size=shape, chunks=chunks, dtype=np.int32
            )
        elif dtype == "float64" or dtype == "float32":
            dask_array = da.random.random(shape, chunks=chunks)

            if dtype == "float32":
                dask_array = dask_array.astype(np.float32)

            # Max flaot32 value is 3.4028235e38, we use a smaller value
            # by default here:
            vmin = attribs.get("valid_min", -1e16)
            vmax = attribs.get("valid_max", 1e16)
            rand_min = vmin - (vmax - vmin) * 0.1
            rand_max = vmax + (vmax - vmin) * 0.1

            # Scale and shift to the desired range [min_val, max_val]
            dask_array = dask_array * (rand_max - rand_min) + rand_min
        else:
            raise ValueError(f"Unsupported data type: {dtype}")

        return dask_array

    def add_rand_data(self, desc):
        """Build a random DataArray from a given description."""
        # Create a lazy dask array with random int32 values
        dtype = desc.get("data_type", "int32")
        dims = desc["dims"]
        attribs = desc["attribs"]
        key = desc["key"]

        dask_array = self.generate_dask_array(dims, dtype, attribs)
        # Define the chunk size for dask array

        if "missing_value" in attribs:
            dask_array = self.inject_missing_value(dask_array, attribs)

        # Wrap the dask array with xarray.DataArray
        data_array = xr.DataArray(dask_array, dims=dims)

        data_array.attrs.update(attribs)

        self.content[key] = data_array
        self.content[key + "/shape"] = data_array.shape
        self.content[key + "/dtype"] = dask_array.dtype
        self.content[key + "/dimensions"] = data_array.dims

        for aname, val in attribs.items():
            self.content[key + "/attr/" + aname] = val

    def get_valid_attribs(self, anames, alist):
        """Retrieve only the valid attributes from a list."""
        attribs = {}
        for idx, val in enumerate(alist):
            # AEnsure we do not inject "None" attribute values:
            if val is not None:
                attribs[anames[idx]] = val
        return attribs

    def get_test_content(self, _filename, _filename_info, _filetype_info):
        """Get the content of the test data.

        Here we generate the default content we want to provide depending
        on the provided filename infos.
        """
        n_lines = 10
        n_for = 14
        n_fov = 16
        self.dims = {"n_lines": n_lines, "n_for": n_for, "n_fov": n_fov}

        self.content = {}

        # Note: below we use the full range of int32 to generate the random
        # values, we expect the handler to "fix" out of range values replacing
        # them with NaNs.
        for grp_desc in DATA_DESC:
            # get the group prefix:
            prefix = grp_desc["group"]
            anames = grp_desc["attrs"]
            for vname, vdesc in grp_desc["variables"].items():
                # For each variable we create a dataset descriptor:
                attribs = self.get_valid_attribs(anames, vdesc[2])

                desc = {
                    "key": f"{prefix}/{vname}",
                    "dims": vdesc[0],
                    "data_type": vdesc[1],
                    "attribs": attribs,
                }

                self.add_rand_data(desc)

        # We should also register the dimensions as we will need access to "n_fov" for instance:
        for key, val in self.dims.items():
            self.content[f"data/dimension/{key}"] = val

        return self.content


class TestIASINGL2NCReader:
    """Main test class for the IASI NG L2 reader."""

    reader_name = "iasi_ng_l2_nc"

    file_prefix = "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-IAS-02"
    file_suffix = "C_EUMT_20170616120000_G_V_20070912084329_20070912084600_O_N____.nc"
    twv_filename = f"{file_prefix}-TWV_{file_suffix}"

    def setup_method(self):
        """Setup the reade config."""
        from satpy._config import config_search_paths

        self.reader_configs = config_search_paths(
            os.path.join("readers", self.reader_name + ".yaml")
        )

    @pytest.fixture(autouse=True, scope="class")
    def fake_handler(self):
        """Wrap NetCDF4 FileHandler with our own fake handler."""
        patch_ctx = mock.patch.object(
            IASINGL2NCFileHandler, "__bases__", (FakeIASINGFileHandlerBase,)
        )

        with patch_ctx:
            patch_ctx.is_local = True
            yield patch_ctx

    @pytest.fixture
    def twv_handler(self):
        """Create a simple (and fake) default handler on a TWV product."""
        return self._create_file_handler(self.twv_filename)

    @pytest.fixture
    def twv_scene(self):
        """Create a simple (and fake) satpy scene on a TWV product."""
        return Scene(filenames=[self.twv_filename], reader=self.reader_name)

    def _create_file_handler(self, filename):
        """Create an handler for the given file checking that it can be parsed correctly."""
        reader = load_reader(self.reader_configs)

        # Test if the file is recognized by the reader
        files = reader.select_files_from_pathnames([filename])
        assert len(files) == 1, "File should be recognized by the reader"

        # Create the file handler:
        reader.create_filehandlers(files)

        # We should have our handler now:
        assert len(reader.file_handlers) == 1

        assert self.reader_name in reader.file_handlers

        handlers = reader.file_handlers[self.reader_name]

        # We should have a single handler for a single file:
        assert len(handlers) == 1
        assert isinstance(handlers[0], IASINGL2NCFileHandler)

        return handlers[0]

    def test_filename_matching(self):
        """Test filename matching against some random name."""
        # Example filename
        prefix = "W_fr-meteo-sat,GRAL,MTI1-IASING-2"
        suffix = (
            "C_EUMS_20220101120000_LEO_O_D_20220101115425_20220101115728_____W______.nc"
        )
        filename = f"{prefix}-l2p_{suffix}"

        self._create_file_handler(filename)

    def test_real_filename_matching(self):
        """Test that we will match an actual IASI NG L2 product file name."""
        # Below we test the TWV,CLD,GHG,SFC,O3_ and CO_ products:
        ptypes = ["TWV", "CLD", "GHG", "SFC", "O3_", "CO_"]

        for ptype in ptypes:
            filename = f"{self.file_prefix}-{ptype}_{self.file_suffix}"
            handler = self._create_file_handler(filename)

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

        # Should be 2D now:
        assert len(lat.dims) == 2
        assert len(lon.dims) == 2
        assert lat.dims[0] == "x"
        assert lat.dims[1] == "y"
        assert lon.dims[0] == "x"
        assert lon.dims[1] == "y"

        # Should have been converted to float64:
        assert lat.dtype == np.float64
        assert lon.dtype == np.float64

        # All valid lat values should be in range [-90.0,90.0]
        assert np.nanmin(lat) >= -90.0
        assert np.nanmax(lat) <= 90.0

        # All valid lon values should be in range [-180.0,180.0]
        assert np.nanmin(lon) >= -180.0
        assert np.nanmax(lon) <= 180.0

    def test_onboard_utc_dataset(self, twv_scene):
        """Test loading the onboard_utc dataset."""
        twv_scene.load(["onboard_utc", "sounder_pixel_latitude"])
        dset = twv_scene["onboard_utc"]

        # Should be 2D now:
        assert len(dset.dims) == 2
        assert dset.dims[0] == "x"
        assert dset.dims[1] == "y"

        # Should have been converted to datetime:
        assert dset.dtype == np.dtype("datetime64[ns]")

        # Should also have the same size as "lattitude" for instance:
        lat = twv_scene["sounder_pixel_latitude"]

        assert lat.shape == dset.shape

    def test_nbr_iterations_dataset(self, twv_scene):
        """Test loading the nbr_iterations dataset."""
        twv_scene.load(["nbr_iterations"])
        dset = twv_scene["nbr_iterations"]

        assert len(dset.dims) == 2
        # Should still be in int32 type:
        assert dset.dtype == np.int32

    def test_register_dataset(self, twv_handler):
        """Test the register_dataset method."""
        twv_handler.dataset_infos = {}
        twv_handler.register_dataset("test_dataset", {"attr": "value"})
        assert "test_dataset" in twv_handler.dataset_infos
        assert twv_handler.dataset_infos["test_dataset"]["attr"] == "value"

        # Should refuse to register the same dataset again afterwards:
        with pytest.raises(KeyError):
            twv_handler.register_dataset("test_dataset", {"attr": "new_value"})

    def test_process_dimension(self, twv_handler):
        """Test the process_dimension method."""
        twv_handler.dimensions_desc = {}
        twv_handler.process_dimension("group/dimension/test_dim", 10)
        assert twv_handler.dimensions_desc["test_dim"] == 10

        # Should refuse to process the same dim again:
        with pytest.raises(KeyError):
            twv_handler.process_dimension("group/dimension/test_dim", 20)

    def test_process_attribute(self, twv_handler):
        """Test the process_attribute method."""
        twv_handler.variable_desc = {"test_var": {"attribs": {}}}
        twv_handler.process_attribute("test_var/attr/test_attr", "value")
        assert twv_handler.variable_desc["test_var"]["attribs"]["test_attr"] == "value"

    def test_process_variable(self, twv_handler):
        """Test the process_variable method."""
        # Note: we only support variables inside a group,
        # because this is always the case in the IASI-NG file format.
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
        # Note: we only support variables inside a group,
        # because this is always the case in the IASI-NG file format.
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
        # Note: we only support variables inside a group,
        # because this is always the case in the IASI-NG file format.
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
        twv_handler.file_content = {"test_var": mock.MagicMock()}
        assert twv_handler.variable_path_exists("test_var")
        assert not twv_handler.variable_path_exists("/attr/test_var")
        assert not twv_handler.variable_path_exists("test_var/dtype")

    def test_convert_data_type(self, twv_handler):
        """Test the convert_data_type method."""
        data = xr.DataArray(np.array([1, 2, 3], dtype=np.float32))
        result = twv_handler.convert_data_type(data, dtype="float64")
        assert result.dtype == np.float64

        # Should stay as f32 if dtype is auto:
        data.attrs["scale_factor"] = 2.0
        result = twv_handler.convert_data_type(data, dtype="auto")
        assert result.dtype == np.float32

    def test_apply_fill_value(self, twv_handler):
        """Test the apply_fill_value method."""
        data = xr.DataArray(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            attrs={"valid_min": 2.0, "valid_max": 4.0},
        )
        result = twv_handler.apply_fill_value(data)
        assert np.isnan(result[0])
        assert np.isnan(result[4])

    def test_apply_rescaling(self, twv_handler):
        """Test the apply_rescaling method."""
        data = xr.DataArray(
            np.array([1, 2, 3]), attrs={"scale_factor": 2, "add_offset": 1}
        )
        result = twv_handler.apply_rescaling(data)
        np.testing.assert_array_equal(result, [3, 5, 7])

    def test_apply_reshaping(self, twv_handler):
        """Test the apply_reshaping method."""
        data = xr.DataArray(np.ones((2, 3, 4)), dims=("a", "b", "c"))
        result = twv_handler.apply_reshaping(data)
        assert result.dims == ("x", "y")
        assert result.shape == (2, 12)

    def test_convert_to_datetime(self, twv_handler):
        """Test the convert_to_datetime method."""
        data = xr.DataArray(np.array([0, 86400]), dims=("time",))
        ds_info = {"seconds_since_epoch": "2000-01-01 00:00:00"}
        result = twv_handler.convert_to_datetime(data, ds_info)
        assert result[0].values == np.datetime64("2000-01-01T00:00:00")
        assert result[1].values == np.datetime64("2000-01-02T00:00:00")

    def test_apply_broadcast(self, twv_handler):
        """Test the apply_broadcast method."""
        data = xr.DataArray(np.array([1, 2, 3]), dims=("x",))
        twv_handler.dimensions_desc = {"n_fov": 2}
        ds_info = {"broadcast_on_dim": "n_fov"}
        result = twv_handler.apply_broadcast(data, ds_info)
        assert result.shape == (6,)

    def test_get_transformed_dataset(self, twv_handler):
        """Test the get_transformed_dataset method."""
        ds_info = {
            "location": "test_var",
            "seconds_since_epoch": "2000-01-01 00:00:00",
            "broadcast_on_dim": "n_fov",
        }
        twv_handler.variable_path_exists = mock.MagicMock(return_value=True)
        twv_handler.file_content["test_var"] = xr.DataArray(
            np.array([[0, 86400]]), dims=("x", "y")
        )

        twv_handler.dimensions_desc = {"n_fov": 2}

        result = twv_handler.get_transformed_dataset(ds_info)
        assert result.shape == (1, 4)
        assert result.dtype == "datetime64[ns]"

    def test_get_dataset(self, twv_handler):
        """Test the get_dataset method."""
        twv_handler.dataset_infos = {
            "test_dataset": {"name": "test_dataset", "location": "test_var"}
        }
        twv_handler.get_transformed_dataset = mock.MagicMock(
            return_value=xr.DataArray([1, 2, 3])
        )

        result = twv_handler.get_dataset({"name": "test_dataset"})
        assert result.equals(xr.DataArray([1, 2, 3]))

        assert twv_handler.get_dataset({"name": "non_existent_dataset"}) is None
