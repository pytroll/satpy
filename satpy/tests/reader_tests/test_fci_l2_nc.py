#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Satpy developers
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

"""The fci_cld_l2_nc reader tests package."""

import os
import unittest
import uuid
from contextlib import suppress
from unittest import mock

import numpy as np
import pytest
from netCDF4 import Dataset
from pyresample import geometry

from satpy.readers.fci_l2_nc import FciL2NCAMVFileHandler, FciL2NCFileHandler, FciL2NCSegmentFileHandler
from satpy.tests.utils import make_dataid

AREA_DEF = geometry.AreaDefinition(
    "mtg_fci_fdss_2km",
    "MTG FCI Full Disk Scanning Service area definition with 2 km resolution",
    "",
    {"h": 35786400., "lon_0": 0.0, "ellps": "WGS84", "proj": "geos", "units": "m"},
    5568,
    5568,
    (-5567999.9942, 5567999.9942, 5567999.9942, -5567999.9942)
)

SEG_AREA_DEF = geometry.AreaDefinition(
    "mtg_fci_fdss_32km",
    "MTG FCI Full Disk Scanning Service area definition with 32 km resolution",
    "",
    {"h": 35786400., "lon_0": 0.0, "ellps": "WGS84", "proj": "geos", "units": "m"},
    348,
    348,
    (-5567999.9942, 5567999.9942, 5567999.9942, -5567999.9942)
)


class TestFciL2NCFileHandler(unittest.TestCase):
    """Test the FciL2NCFileHandler reader."""

    def setUp(self):
        """Set up the test by creating a test file and opening it with the reader."""
        # Easiest way to test the reader is to create a test netCDF file on the fly
        # Create unique filenames to prevent race conditions when tests are run in parallel
        self.test_file = str(uuid.uuid4()) + ".nc"
        with Dataset(self.test_file, "w") as nc:
            # Create dimensions
            nc.createDimension("number_of_columns", 10)
            nc.createDimension("number_of_rows", 100)
            nc.createDimension("maximum_number_of_layers", 2)

            # add global attributes
            nc.data_source = "TEST_DATA_SOURCE"
            nc.platform = "TEST_PLATFORM"

            # Add datasets
            x = nc.createVariable("x", np.float32, dimensions=("number_of_columns",))
            x.standard_name = "projection_x_coordinate"
            x[:] = np.arange(10)

            y = nc.createVariable("y", np.float32, dimensions=("number_of_rows",))
            y.standard_name = "projection_y_coordinate"
            y[:] = np.arange(100)

            s = nc.createVariable("product_quality", np.int8)
            s[:] = 99.

            one_layer_dataset = nc.createVariable("test_one_layer", np.float32,
                                                  dimensions=("number_of_rows", "number_of_columns"))
            one_layer_dataset[:] = np.ones((100, 10))
            one_layer_dataset.test_attr = "attr"
            one_layer_dataset.units = "test_units"

            two_layers_dataset = nc.createVariable("test_two_layers", np.float32,
                                                   dimensions=("maximum_number_of_layers",
                                                               "number_of_rows",
                                                               "number_of_columns"))
            two_layers_dataset[0, :, :] = np.ones((100, 10))
            two_layers_dataset[1, :, :] = 2 * np.ones((100, 10))
            two_layers_dataset.unit = "test_unit"

            mtg_geos_projection = nc.createVariable("mtg_geos_projection", int, dimensions=())
            mtg_geos_projection.longitude_of_projection_origin = 0.0
            mtg_geos_projection.semi_major_axis = 6378137.
            mtg_geos_projection.inverse_flattening = 298.257223563
            mtg_geos_projection.perspective_point_height = 35786400.

            # Add enumerated type
            enum_dict = {"False": 0, "True": 1}
            bool_type = nc.createEnumType(np.uint8,"bool_t",enum_dict)
            nc.createVariable("quality_flag", bool_type,
                              dimensions=("number_of_rows", "number_of_columns"))

        self.fh = FciL2NCFileHandler(filename=self.test_file, filename_info={}, filetype_info={})

    def tearDown(self):
        """Remove the previously created test file."""
        # First delete the file handler, forcing the file to be closed if still open
        del self.fh
        # Then we can safely remove the file from the system
        with suppress(OSError):
            os.remove(self.test_file)

    def test_all_basic(self):
        """Test all basic functionalities."""
        assert self.fh.spacecraft_name == "TEST_PLATFORM"
        assert self.fh.sensor_name == "test_data_source"
        assert self.fh.ssp_lon == 0.0

        global_attributes = self.fh._get_global_attributes()
        expected_global_attributes = {
            "filename": self.test_file,
            "spacecraft_name": "TEST_PLATFORM",
            "ssp_lon": 0.0,
            "sensor": "test_data_source",
            "platform_name": "TEST_PLATFORM"
        }
        assert global_attributes == expected_global_attributes

    @mock.patch("satpy.readers.fci_l2_nc.geometry.AreaDefinition")
    @mock.patch("satpy.readers.fci_l2_nc.make_ext")
    def test_area_definition(self, me_, gad_):
        """Test the area definition computation."""
        self.fh._compute_area_def(make_dataid(name="test_area_def", resolution=2000))

        # Asserts that the make_ext function was called with the correct arguments
        me_.assert_called_once()
        args, kwargs = me_.call_args
        np.testing.assert_allclose(args, [-0.0, -515.6620, 5672.28217, 0.0, 35786400.])

        proj_dict = {"a": 6378137.,
                     "lon_0": 0.0,
                     "h": 35786400,
                     "rf": 298.257223563,
                     "proj": "geos",
                     "units": "m",
                     "sweep": "y"}

        # Asserts that the get_area_definition function was called with the correct arguments
        gad_.assert_called_once()
        args, kwargs = gad_.call_args
        assert args[0] == "mtg_fci_fdss_2km"
        assert args[1] == "MTG FCI Full Disk Scanning Service area definition with 2 km resolution"
        assert args[2] == ""
        assert args[3] == proj_dict
        assert args[4] == 10
        assert args[5] == 100

    def test_dataset(self):
        """Test the correct execution of the get_dataset function with a valid nc_key."""
        dataset = self.fh.get_dataset(make_dataid(name="test_one_layer", resolution=2000),
                                      {"name": "test_one_layer",
                                       "nc_key": "test_one_layer",
                                       "fill_value": -999,
                                       "file_type": "test_file_type"})

        np.testing.assert_allclose(dataset.values, np.ones((100, 10)))
        assert dataset.attrs["test_attr"] == "attr"
        assert dataset.attrs["fill_value"] == -999

    def test_dataset_with_layer(self):
        """Check the correct execution of the get_dataset function with a valid nc_key & layer."""
        dataset = self.fh.get_dataset(make_dataid(name="test_two_layers", resolution=2000),
                                      {"name": "test_two_layers",
                                       "nc_key": "test_two_layers", "layer": 1,
                                       "fill_value": -999,
                                       "file_type": "test_file_type"})
        np.testing.assert_allclose(dataset.values, 2 * np.ones((100, 10)))
        assert dataset.attrs["spacecraft_name"] == "TEST_PLATFORM"

    def test_dataset_with_invalid_filekey(self):
        """Test the correct execution of the get_dataset function with an invalid nc_key."""
        invalid_dataset = self.fh.get_dataset(make_dataid(name="test_invalid", resolution=2000),
                                              {"name": "test_invalid",
                                               "nc_key": "test_invalid",
                                               "fill_value": -999,
                                               "file_type": "test_file_type"})
        assert invalid_dataset is None

    def test_dataset_with_total_cot(self):
        """Test the correct execution of the get_dataset function for total COT (add contributions from two layers)."""
        dataset = self.fh.get_dataset(make_dataid(name="retrieved_cloud_optical_thickness", resolution=2000),
                                      {"name": "retrieved_cloud_optical_thickness",
                                       "nc_key": "test_two_layers",
                                       "fill_value": -999,
                                       "file_type": "test_file_type"})
        # Checks that the function returns None
        expected_sum = np.empty((100, 10))
        expected_sum[:] = np.log10(10**2 + 10**1)
        np.testing.assert_allclose(dataset.values, expected_sum)

    def test_dataset_with_scalar(self):
        """Test the execution of the get_dataset function for scalar values."""
        # Checks returned scalar value
        dataset = self.fh.get_dataset(make_dataid(name="test_scalar"),
                                      {"name": "product_quality",
                                       "nc_key": "product_quality",
                                       "file_type": "test_file_type"})
        assert dataset.values == 99.0

        # Checks that no AreaDefintion is implemented for scalar values
        with pytest.raises(NotImplementedError):
            self.fh.get_area_def(None)

    def test_emumerations(self):
        """Test the conversion of enumerated type information into flag_values and flag_meanings."""
        dataset = self.fh.get_dataset(make_dataid(name="test_enum", resolution=2000),
                                      {"name": "quality_flag",
                                       "nc_key": "quality_flag",
                                       "file_type": "test_file_type",
                                       "import_enum_information": True})
        attributes = dataset.attrs
        assert "flag_values" in attributes
        assert attributes["flag_values"] == [0,1]
        assert "flag_meanings" in attributes
        assert attributes["flag_meanings"] == ["False","True"]

    def test_units_from_file(self):
        """Test units extraction from NetCDF file."""
        dataset = self.fh.get_dataset(make_dataid(name="test_units_from_file", resolution=2000),
                                      {"name": "test_one_layer",
                                       "nc_key": "test_one_layer",
                                       "file_type": "test_file_type"})
        assert dataset.attrs["units"] == "test_units"

    def test_unit_from_file(self):
        """Test that a unit stored with attribute `unit` in the file is assigned to the `units` attribute."""
        dataset = self.fh.get_dataset(make_dataid(name="test_unit_from_file", resolution=2000),
                                      {"name": "test_two_layers",
                                       "nc_key": "test_two_layers", "layer": 1,
                                       "file_type": "test_file_type"})
        assert dataset.attrs["units"] == "test_unit"

    def test_units_from_yaml(self):
        """Test units extraction from yaml file."""
        dataset = self.fh.get_dataset(make_dataid(name="test_units_from_yaml", resolution=2000),
                                      {"name": "test_one_layer",
                                       "units": "test_unit_from_yaml",
                                       "nc_key": "test_one_layer",
                                       "file_type": "test_file_type"})
        assert dataset.attrs["units"] == "test_unit_from_yaml"

    def test_units_none_conversion(self):
        """Test that a units stored as 'none' is converted to None."""
        dataset = self.fh.get_dataset(make_dataid(name="test_units_none_conversion", resolution=2000),
                                      {"name": "test_one_layer",
                                       "units": "none",
                                       "nc_key": "test_one_layer",
                                       "file_type": "test_file_type"})
        assert dataset.attrs["units"] is None


class TestFciL2NCSegmentFileHandler(unittest.TestCase):
    """Test the FciL2NCSegmentFileHandler reader."""

    def setUp(self):
        """Set up the test by creating a test file and opening it with the reader."""
        # Easiest way to test the reader is to create a test netCDF file on the fly
        self.seg_test_file = str(uuid.uuid4()) + ".nc"
        with Dataset(self.seg_test_file, "w") as nc:
            # Create dimensions
            nc.createDimension("number_of_FoR_cols", 348)
            nc.createDimension("number_of_FoR_rows", 348)
            nc.createDimension("number_of_channels", 8)
            nc.createDimension("number_of_categories", 6)

            # add global attributes
            nc.data_source = "TEST_FCI_DATA_SOURCE"
            nc.platform = "TEST_FCI_PLATFORM"

            # Add datasets
            x = nc.createVariable("x", np.float32, dimensions=("number_of_FoR_cols",))
            x.standard_name = "projection_x_coordinate"
            x[:] = np.arange(348)

            y = nc.createVariable("y", np.float32, dimensions=("number_of_FoR_rows",))
            y.standard_name = "projection_y_coordinate"
            y[:] = np.arange(348)

            s = nc.createVariable("product_quality", np.int8)
            s[:] = 99.

            chans = nc.createVariable("channels", np.float32, dimensions=("number_of_channels",))
            chans.standard_name = "fci_channels"
            chans[:] = np.arange(8)

            cats = nc.createVariable("categories", np.float32, dimensions=("number_of_categories",))
            cats.standard_name = "product_categories"
            cats[:] = np.arange(6)

            test_dataset = nc.createVariable("test_values", np.float32,
                                             dimensions=("number_of_FoR_rows", "number_of_FoR_cols",
                                                         "number_of_channels", "number_of_categories"))

            test_dataset[:] = self._get_unique_array(range(8), range(6))
            test_dataset.test_attr = "attr"
            test_dataset.units = "test_units"

    def tearDown(self):
        """Remove the previously created test file."""
        # First delete the fh, forcing the file to be closed if still open
        del self.fh
        # Then can safely remove it from the system
        with suppress(OSError):
            os.remove(self.seg_test_file)

    def test_all_basic(self):
        """Test all basic functionalities."""
        self.fh = FciL2NCSegmentFileHandler(filename=self.seg_test_file, filename_info={}, filetype_info={})

        assert self.fh.spacecraft_name == "TEST_FCI_PLATFORM"
        assert self.fh.sensor_name == "test_fci_data_source"
        assert self.fh.ssp_lon == 0.0

        global_attributes = self.fh._get_global_attributes()

        expected_global_attributes = {
            "filename": self.seg_test_file,
            "spacecraft_name": "TEST_FCI_PLATFORM",
            "ssp_lon": 0.0,
            "sensor": "test_fci_data_source",
            "platform_name": "TEST_FCI_PLATFORM"
        }
        assert global_attributes == expected_global_attributes

    def test_dataset(self):
        """Test the correct execution of the get_dataset function with valid nc_key."""
        self.fh = FciL2NCSegmentFileHandler(filename=self.seg_test_file, filename_info={}, filetype_info={})

        # Checks the correct execution of the get_dataset function with a valid nc_key
        dataset = self.fh.get_dataset(make_dataid(name="test_values", resolution=32000),
                                      {"name": "test_values",
                                       "nc_key": "test_values",
                                       "fill_value": -999, })
        expected_dataset = self._get_unique_array(range(8), range(6))
        np.testing.assert_allclose(dataset.values, expected_dataset)
        assert dataset.attrs["test_attr"] == "attr"
        assert dataset.attrs["units"] == "test_units"
        assert dataset.attrs["fill_value"] == -999

        # Checks that no AreaDefintion is implemented
        with pytest.raises(NotImplementedError):
            self.fh.get_area_def(None)

    def test_dataset_with_invalid_filekey(self):
        """Test the correct execution of the get_dataset function with an invalid nc_key."""
        self.fh = FciL2NCSegmentFileHandler(filename=self.seg_test_file, filename_info={}, filetype_info={})

        # Checks the correct execution of the get_dataset function with an invalid nc_key
        invalid_dataset = self.fh.get_dataset(make_dataid(name="test_invalid", resolution=32000),
                                              {"name": "test_invalid",
                                               "nc_key": "test_invalid",
                                               "fill_value": -999, })
        # Checks that the function returns None
        assert invalid_dataset is None

    def test_dataset_with_adef(self):
        """Test the correct execution of the get_dataset function with `with_area_definition=True`."""
        self.fh = FciL2NCSegmentFileHandler(filename=self.seg_test_file, filename_info={}, filetype_info={},
                                            with_area_definition=True)

        # Checks the correct execution of the get_dataset function with a valid nc_key
        dataset = self.fh.get_dataset(make_dataid(name="test_values", resolution=32000),
                                      {"name": "test_values",
                                       "nc_key": "test_values",
                                       "fill_value": -999,
                                       "coordinates": ("test_lon", "test_lat"), })
        expected_dataset = self._get_unique_array(range(8), range(6))
        np.testing.assert_allclose(dataset.values, expected_dataset)
        assert dataset.attrs["test_attr"] == "attr"
        assert dataset.attrs["units"] == "test_units"
        assert dataset.attrs["fill_value"] == -999

        # Checks returned AreaDefinition against reference
        adef = self.fh.get_area_def(None)
        assert adef == SEG_AREA_DEF

    def test_dataset_with_adef_and_wrongs_dims(self):
        """Test the correct execution of the get_dataset function with dims that don't match expected AreaDefinition."""
        self.fh = FciL2NCSegmentFileHandler(filename=self.seg_test_file, filename_info={}, filetype_info={},
                                            with_area_definition=True)
        with pytest.raises(NotImplementedError):
            self.fh.get_dataset(make_dataid(name="test_wrong_dims", resolution=6000),
                                {"name": "test_wrong_dims", "nc_key": "test_values", "fill_value": -999}
                                )

    def test_dataset_with_scalar(self):
        """Test the execution of the get_dataset function for scalar values."""
        self.fh = FciL2NCSegmentFileHandler(filename=self.seg_test_file, filename_info={}, filetype_info={})
        # Checks returned scalar value
        dataset = self.fh.get_dataset(make_dataid(name="test_scalar"),
                                      {"name": "product_quality",
                                       "nc_key": "product_quality",
                                       "file_type": "test_file_type"})
        assert dataset.values == 99.0

        # Checks that no AreaDefintion is implemented for scalar values
        with pytest.raises(NotImplementedError):
            self.fh.get_area_def(None)

    def test_dataset_slicing_catid(self):
        """Test the correct execution of the _slice_dataset function with 'category_id' set."""
        self.fh = FciL2NCSegmentFileHandler(filename=self.seg_test_file, filename_info={}, filetype_info={})

        dataset = self.fh.get_dataset(make_dataid(name="test_values", resolution=32000),
                                      {"name": "test_values",
                                       "nc_key": "test_values",
                                       "fill_value": -999,
                                       "category_id": 5})
        expected_dataset = self._get_unique_array(range(8), 5)
        np.testing.assert_allclose(dataset.values, expected_dataset)

    def test_dataset_slicing_chid_catid(self):
        """Test the correct execution of the _slice_dataset function with 'channel_id' and 'category_id' set."""
        self.fh = FciL2NCSegmentFileHandler(filename=self.seg_test_file, filename_info={}, filetype_info={})

        dataset = self.fh.get_dataset(make_dataid(name="test_values", resolution=32000),
                                      {"name": "test_values",
                                       "nc_key": "test_values",
                                       "fill_value": -999,
                                       "channel_id": 0, "category_id": 1})
        expected_dataset = self._get_unique_array(0, 1)
        np.testing.assert_allclose(dataset.values, expected_dataset)

    def test_dataset_slicing_visid_catid(self):
        """Test the correct execution of the _slice_dataset function with 'vis_channel_id' and 'category_id' set."""
        self.fh = FciL2NCSegmentFileHandler(filename=self.seg_test_file, filename_info={}, filetype_info={})

        self.fh.nc = self.fh.nc.rename_dims({"number_of_channels": "number_of_vis_channels"})
        dataset = self.fh.get_dataset(make_dataid(name="test_values", resolution=32000),
                                      {"name": "test_values",
                                       "nc_key": "test_values",
                                       "fill_value": -999,
                                       "vis_channel_id": 3, "category_id": 3})
        expected_dataset = self._get_unique_array(3, 3)
        np.testing.assert_allclose(dataset.values, expected_dataset)

    def test_dataset_slicing_irid(self):
        """Test the correct execution of the _slice_dataset function with 'ir_channel_id' set."""
        self.fh = FciL2NCSegmentFileHandler(filename=self.seg_test_file, filename_info={}, filetype_info={})

        self.fh.nc = self.fh.nc.rename_dims({"number_of_channels": "number_of_ir_channels"})
        dataset = self.fh.get_dataset(make_dataid(name="test_values", resolution=32000),
                                      {"name": "test_values",
                                       "nc_key": "test_values",
                                       "fill_value": -999,
                                       "ir_channel_id": 4})
        expected_dataset = self._get_unique_array(4, range(6))
        np.testing.assert_allclose(dataset.values, expected_dataset)

    @staticmethod
    def _get_unique_array(iarr, jarr):
        if not hasattr(iarr, "__iter__"):
            iarr = [iarr]

        if not hasattr(jarr, "__iter__"):
            jarr = [jarr]

        array = np.zeros((348, 348, 8, 6))
        for i in iarr:
            for j in jarr:
                array[:, :, i, j] = (i * 10) + j

        array = array[:, :, list(iarr), :]
        array = array[:, :, :, list(jarr)]

        return np.squeeze(array)


class TestFciL2NCReadingByteData(unittest.TestCase):
    """Test the FciL2NCFileHandler when reading and extracting byte data."""

    def setUp(self):
        """Set up the test by creating a test file and opening it with the reader."""
        # Easiest way to test the reader is to create a test netCDF file on the fly
        self.test_byte_file = str(uuid.uuid4()) + ".nc"
        with Dataset(self.test_byte_file, "w") as nc_byte:
            # Create dimensions
            nc_byte.createDimension("number_of_columns", 1)
            nc_byte.createDimension("number_of_rows", 1)

            # add global attributes
            nc_byte.data_source = "TEST_DATA_SOURCE"
            nc_byte.platform = "TEST_PLATFORM"

            # Add datasets
            x = nc_byte.createVariable("x", np.float32, dimensions=("number_of_columns",))
            x.standard_name = "projection_x_coordinate"
            x[:] = np.arange(1)

            y = nc_byte.createVariable("y", np.float32, dimensions=("number_of_rows",))
            x.standard_name = "projection_y_coordinate"
            y[:] = np.arange(1)

            mtg_geos_projection = nc_byte.createVariable("mtg_geos_projection", int, dimensions=())
            mtg_geos_projection.longitude_of_projection_origin = 0.0
            mtg_geos_projection.semi_major_axis = 6378137.
            mtg_geos_projection.inverse_flattening = 298.257223563
            mtg_geos_projection.perspective_point_height = 35786400.

            test_dataset = nc_byte.createVariable("cloud_mask_test_flag", np.float32,
                                                  dimensions=("number_of_rows", "number_of_columns",))

            # This number was chosen as we know the expected byte values
            test_dataset[:] = 4544767

        self.byte_reader = FciL2NCFileHandler(
            filename=self.test_byte_file,
            filename_info={},
            filetype_info={}
        )

    def tearDown(self):
        """Remove the previously created test file."""
        # First delete the file handler, forcing the file to be closed if still open
        del self.byte_reader
        # Then can safely remove it from the system
        with suppress(OSError):
            os.remove(self.test_byte_file)

    def test_byte_extraction(self):
        """Test the execution of the get_dataset function."""
        # Value of 1 is expected to be returned for this test
        dataset = self.byte_reader.get_dataset(make_dataid(name="cloud_mask_test_flag", resolution=2000),
                                               {"name": "cloud_mask_test_flag",
                                                "nc_key": "cloud_mask_test_flag",
                                                "fill_value": -999,
                                                "file_type": "nc_fci_test_clm",
                                                "extract_byte": 1,
                                                })

        assert dataset.values == 1

        # Value of 0 is expected fto be returned or this test
        dataset = self.byte_reader.get_dataset(make_dataid(name="cloud_mask_test_flag", resolution=2000),
                                               {"name": "cloud_mask_test_flag",
                                                "nc_key": "cloud_mask_test_flag",
                                                "fill_value": -999, "mask_value": 0.,
                                                "file_type": "nc_fci_test_clm",
                                                "extract_byte": 23,
                                                })

        assert dataset.values == 0


@pytest.fixture(scope="module")
def amv_file(tmp_path_factory):
    """Create an AMV file."""
    test_file = tmp_path_factory.mktemp("data") / "fci_l2_amv.nc"

    with Dataset(test_file, "w") as nc:
        # Create dimensions
        nc.createDimension("number_of_winds", 50000)

        # add global attributes
        nc.data_source = "TEST_DATA_SOURCE"
        nc.platform = "TEST_PLATFORM"

        # Add datasets
        latitude = nc.createVariable("latitude", np.float32, dimensions=("number_of_winds",))
        latitude[:] = np.arange(50000)

        longitude = nc.createVariable("y", np.float32, dimensions=("number_of_winds",))
        longitude[:] = np.arange(50000)

        qi = nc.createVariable("product_quality", np.int8)
        qi[:] = 99.

        test_dataset = nc.createVariable("test_dataset", np.float32,
                                                dimensions="number_of_winds")
        test_dataset[:] = np.ones(50000)
        test_dataset.test_attr = "attr"
        test_dataset.units = "test_units"

        mtg_geos_projection = nc.createVariable("mtg_geos_projection", int, dimensions=())
        mtg_geos_projection.longitude_of_projection_origin = 0.0
        mtg_geos_projection.semi_major_axis = 6378137.
        mtg_geos_projection.inverse_flattening = 298.257223563
        mtg_geos_projection.perspective_point_height = 35786400.
    return test_file


@pytest.fixture(scope="module")
def amv_filehandler(amv_file):
    """Create an AMV filehandler."""
    return FciL2NCAMVFileHandler(filename=amv_file,
                                 filename_info={"channel":"test_channel"},
                                 filetype_info={}
                                )


class TestFciL2NCAMVFileHandler:
    """Test the FciL2NCAMVFileHandler reader."""

    def test_all_basic(self, amv_filehandler, amv_file):
        """Test all basic functionalities."""
        assert amv_filehandler.spacecraft_name == "TEST_PLATFORM"
        assert amv_filehandler.sensor_name == "test_data_source"
        assert amv_filehandler.ssp_lon == 0.0

        global_attributes = amv_filehandler._get_global_attributes(product_type="amv")
        expected_global_attributes = {
            "filename": amv_file,
            "spacecraft_name": "TEST_PLATFORM",
            "sensor": "test_data_source",
            "platform_name": "TEST_PLATFORM",
            "channel": "test_channel",
            "ssp_lon": 0.0,
        }
        assert global_attributes == expected_global_attributes

    def test_dataset(self, amv_filehandler):
        """Test the correct execution of the get_dataset function with a valid nc_key."""
        dataset = amv_filehandler.get_dataset(make_dataid(name="test_dataset", resolution=2000),
                                      {"name": "test_dataset",
                                       "nc_key": "test_dataset",
                                       "fill_value": -999,
                                       "file_type": "test_file_type"})
        np.testing.assert_allclose(dataset.values, np.ones(50000))
        assert dataset.attrs["test_attr"] == "attr"
        assert dataset.attrs["units"] == "test_units"
        assert dataset.attrs["fill_value"] == -999

    def test_dataset_with_invalid_filekey(self, amv_filehandler):
         """Test the correct execution of the get_dataset function with an invalid nc_key."""
         invalid_dataset = amv_filehandler.get_dataset(make_dataid(name="test_invalid", resolution=2000),
                                               {"name": "test_invalid",
                                                "nc_key": "test_invalid",
                                                "fill_value": -999,
                                                "file_type": "test_file_type"})
         assert invalid_dataset is None
