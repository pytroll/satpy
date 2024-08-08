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
import logging
import os
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

logger = logging.getLogger(__name__)

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
            "unit",
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
                ("seconds", -1e9, 1e9, None, None, -9e9),
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
        "attrs": ["unit", "valid_min", "valid_max", "missing_value"],
        "variables": {
            "air_temperature": [d_lff, "float32", ("K", 100.0, 400.0, 3.4e38)],
            "atmosphere_mass_content_of_water": [
                d_lff,
                "float32",
                ("kg.m-2", 0, 300, 3.4e38),
            ],
            "qi_air_temperature": [d_lff, "float32", ("", 0, 25, 3.4e38)],
            "qi_speciﬁc_humidity": [d_lff, "float32", ("", 0, 25, 3.4e38)],
            "speciﬁc_humidity": [d_lff, "float32", ("kg/kg", 0, 1, 3.4e38)],
            "surface_air_temperature": [d_lff, "float32", ("K", 100, 400, 3.4e38)],
            "surface_speciﬁc_humidity": [d_lff, "float32", ("K", 100, 400, 3.4e38)],
        },
    },
]


class FakeIASINGFileHandlerBase(FakeNetCDF4FileHandler):
    """Fake base class for IASI NG handler"""

    chunks = (10, 10, 10)

    def add_rand_data(self, desc):
        """
        Build a random DataArray from a given description, and add it as content.
        """

        # Create a lazy dask array with random int32 values
        dtype = desc.get("data_type", "int32")
        dims = desc["dims"]
        key = desc["key"]

        shape = [self.dims[k] for k in dims]

        # Define the chunk size for dask array
        chunks = [10] * len(dims)
        attribs = desc["attribs"]

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

        if "missing_value" in attribs:
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

                # Generate unique random indices to set as missing values within this block
                flat_indices = np.random.choice(
                    block_size, block_num_to_replace, replace=False
                )
                unraveled_indices = np.unravel_index(flat_indices, block_shape)
                block[unraveled_indices] = missing_value

                return block

            # Apply the function lazily to each block
            dask_array = dask_array.map_blocks(set_missing_values, dtype=dask_array.dtype)

        # Wrap the dask array with xarray.DataArray
        data_array = xr.DataArray(dask_array, dims=dims)

        data_array.attrs.update(attribs)

        self.content[key] = data_array

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
                attribs = {}
                for idx, val in enumerate(vdesc[2]):
                    # AEnsure we do not inject "None" attribute values:
                    if val is not None:
                        attribs[anames[idx]] = val

                desc = {
                    "key": f"{prefix}/{vname}",
                    "dims": vdesc[0],
                    "data_type": vdesc[1],
                    "attribs": attribs,
                }

                self.add_rand_data(desc)

        return self.content


class TestIASINGL2NCReader:
    """Main test class for the IASI NG L2 reader."""

    reader_name = "iasi_ng_l2_nc"

    def setup_method(self):
        """Setup the reade config"""
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

    @pytest.fixture()
    def twv_handler(self):
        """Create a simple (and fake) default handler on a TWV product"""
        filename = "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-IAS-02-TWV_C_EUMT_20170616120000_G_V_20070912084329_20070912084600_O_N____.nc"
        return self._create_file_handler(filename)

    @pytest.fixture()
    def twv_scene(self):
        """Create a simple (and fake) satpy scene on a TWV product"""
        filename = "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-IAS-02-TWV_C_EUMT_20170616120000_G_V_20070912084329_20070912084600_O_N____.nc"
        return Scene(filenames=[filename], reader=self.reader_name)

    def _create_file_handler(self, filename):
        """Create an handler for the given file checking that it can
        be parsed correctly"""

        reader = load_reader(self.reader_configs)

        # Test if the file is recognized by the reader
        files = reader.select_files_from_pathnames([filename])
        assert len(files) == 1, "File should be recognized by the reader"

        # Create the file handler:
        reader.create_filehandlers(files)

        # We should have our handler now:
        assert len(reader.file_handlers) == 1

        # logger.info("File handlers are: %s", reader.file_handlers)
        assert self.reader_name in reader.file_handlers

        handlers = reader.file_handlers[self.reader_name]

        # We should have a single handler for a single file:
        assert len(handlers) == 1
        assert isinstance(handlers[0], IASINGL2NCFileHandler)

        return handlers[0]

    def test_filename_matching(self):
        """Test filename matching against some random name"""

        # Example filename
        filename = "W_fr-meteo-sat,GRAL,MTI1-IASING-2-l2p_C_EUMS_20220101120000_LEO_O_D_20220101115425_20220101115728_____W______.nc"

        self._create_file_handler(filename)

    def test_real_filename_matching(self):
        """Test that we will match an actual IASI NG L2 product file name"""

        # Below we test the TWV,CLD,GHG and SFC products:
        filenames = {
            "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-IAS-02-TWV_C_EUMT_20170616120000_G_V_20070912084329_20070912084600_O_N____.nc",
            "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-IAS-02-CLD_C_EUMT_20170616120000_G_V_20070912094037_20070912094308_O_N____.nc",
            "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-IAS-02-GHG_C_EUMT_20170616120000_G_V_20070912090651_20070912090922_O_N____.nc",
            "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-IAS-02-SFC_C_EUMT_20170616120000_G_V_20070912100911_20070912101141_O_N____.nc",
        }

        for filename in filenames:
            self._create_file_handler(filename)

    def test_sensing_times(self, twv_handler):
        """Test that we read the sensing start/end times correctly
        from filename"""

        assert twv_handler.start_time == datetime(2007, 9, 12, 8, 43, 29)
        assert twv_handler.end_time == datetime(2007, 9, 12, 8, 46, 0)

    def test_sensor_names(self, twv_handler):
        """Test that the handler reports iasi_ng as sensor"""

        assert twv_handler.sensor_names == {"iasi_ng"}

    def test_available_datasets(self, twv_scene):
        """Test the list of available datasets in scene"""

        dnames = twv_scene.available_dataset_names()

        expected_names = [
            "onboard_utc",
            "latitude",
            "longitude",
            "azimuth",
            "zenith",
            "sun_azimuth",
            "sun_zenith",
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

    def test_latitude_dataset(self, twv_scene):
        """Test loading the latitude dataset"""

        twv_scene.load(["latitude"])
        dset = twv_scene["latitude"]

        # Should be 2D now:
        assert len(dset.dims) == 2
        assert dset.dims[0] == "x"
        assert dset.dims[1] == "y"

        # Should have been converted to float64:
        assert dset.dtype == np.float64

        # All valid values should be in range [-90.0,90.0]
        vmin = np.nanmin(dset)
        vmax = np.nanmax(dset)

        assert vmin >= -90.0
        assert vmax <= 90.0

    def test_longitude_dataset(self, twv_scene):
        """Test loading the longitude dataset"""

        twv_scene.load(["longitude"])
        dset = twv_scene["longitude"]

        # Should be 2D now:
        assert len(dset.dims) == 2
        assert dset.dims[0] == "x"
        assert dset.dims[1] == "y"

        # Should have been converted to float64:
        assert dset.dtype == np.float64

        # All valid values should be in range [-90.0,90.0]
        vmin = np.nanmin(dset)
        vmax = np.nanmax(dset)

        assert vmin >= -180.0
        assert vmax <= 180.0

    def test_onboard_utc_dataset(self, twv_scene):
        """Test loading the onboard_utc dataset"""

        twv_scene.load(["onboard_utc"])
        dset = twv_scene["onboard_utc"]

        # Should be 2D now:
        assert len(dset.dims) == 2
        assert dset.dims[0] == "x"
        assert dset.dims[1] == "y"

        # Should have been converted to datetime:
        assert dset.dtype == np.dtype("datetime64[ns]")

    def test_nbr_iterations_dataset(self, twv_scene):
        """Test loading the nbr_iterations dataset"""

        twv_scene.load(["nbr_iterations"])
        dset = twv_scene["nbr_iterations"]

        assert len(dset.dims) == 2
        # Should still be in int32 type:
        assert dset.dtype == np.int32
