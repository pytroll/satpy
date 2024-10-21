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

"""EUM L2 GRIB-reader test package."""

import datetime
import sys
from unittest import mock

import numpy as np
import pytest

from satpy.tests.utils import make_dataid

# Dictionary to be used as fake GRIB message
FAKE_SEVIRI_MESSAGE = {
    "longitudeOfSubSatellitePointInDegrees": 9.5,
    "dataDate": 20191020,
    "dataTime": 1745,
    "Nx": 1000,
    "Ny": 1200,
    "earthMajorAxis": 6400.,
    "earthMinorAxis": 6300.,
    "NrInRadiusOfEarth": 6.,
    "XpInGridLengths": 500,
    "parameterNumber": 30,
    "missingValue": 9999,
}

FAKE_FCI_MESSAGE = {
    "longitudeOfSubSatellitePointInDegrees": 0.0,
    "dataDate": 20191020,
    "dataTime": 1745,
    "Nx": 5568,
    "Ny": 5568,
    "earthMajorAxis": 6378140.,
    "earthMinorAxis": 6356755.,
    "NrInRadiusOfEarth": 6.6107,
    "XpInGridLengths": 2784.0,
    "parameterNumber": 30,
    "missingValue": 9999,
}

# List to be used as fake GID source
FAKE_GID = [0, 1, 2, 3, None]


@pytest.fixture
@mock.patch("satpy.readers.eum_l2_grib.ec")
def setup_reader(ec_):
    """Set up the test by creating a mocked eccodes library."""
    fake_gid_generator = (i for i in FAKE_GID)
    ec_.codes_grib_new_from_file.side_effect = lambda fh: next(fake_gid_generator)
    return ec_


def common_checks(ec_, reader, mock_file, dataset_id):
    """Commmon checks for fci and seviri data."""
    # Checks that the codes_grib_multi_support_on function has been called
    ec_.codes_grib_multi_support_on.assert_called()

    # Restarts the id generator and clears the call history
    fake_gid_generator = (i for i in FAKE_GID)
    ec_.codes_grib_new_from_file.side_effect = lambda fh: next(fake_gid_generator)
    ec_.codes_grib_new_from_file.reset_mock()
    ec_.codes_release.reset_mock()

    # Checks the correct execution of the get_dataset function with a valid parameter_number
    valid_dataset = reader.get_dataset(dataset_id, {"parameter_number": 30})
    # Checks the correct file open call
    mock_file.assert_called_with("test.grib", "rb")
    # Checks that the dataset has been created as a DataArray object
    assert valid_dataset._extract_mock_name() == "xr.DataArray()"
    # Checks that codes_release has been called after each codes_grib_new_from_file call
    # (except after the last one which has returned a None)
    assert ec_.codes_grib_new_from_file.call_count == ec_.codes_release.call_count + 1

    # Restarts the id generator and clears the call history
    fake_gid_generator = (i for i in FAKE_GID)
    ec_.codes_grib_new_from_file.side_effect = lambda fh: next(fake_gid_generator)
    ec_.codes_grib_new_from_file.reset_mock()
    ec_.codes_release.reset_mock()

    # Checks the correct execution of the get_dataset function with an invalid parameter_number
    invalid_dataset = reader.get_dataset(dataset_id, {"parameter_number": 50})
    # Checks that the function returns None
    assert invalid_dataset is None
    # Checks that codes_release has been called after each codes_grib_new_from_file call
    # (except after the last one which has returned a None)
    assert ec_.codes_grib_new_from_file.call_count == ec_.codes_release.call_count + 1


@pytest.mark.skipif(sys.platform.startswith("win"), reason="'eccodes' not supported on Windows")
@mock.patch("satpy.readers.eum_l2_grib.xr")
@mock.patch("satpy.readers.eum_l2_grib.da")
def test_seviri_data_reading(da_, xr_, setup_reader):
    """Test the reading of data from the product."""
    from satpy.readers.eum_l2_grib import EUML2GribFileHandler
    from satpy.utils import get_legacy_chunk_size
    ec_ = setup_reader
    chunk_size = get_legacy_chunk_size()

    with mock.patch("builtins.open", mock.mock_open()) as mock_file:
        with mock.patch("satpy.readers.eum_l2_grib.ec", ec_):
            ec_.codes_get_values.return_value = np.ones(1000 * 1200)
            ec_.codes_get.side_effect = lambda gid, key: FAKE_SEVIRI_MESSAGE[key]
            reader = EUML2GribFileHandler(
                filename="test.grib",
                filename_info={
                    "spacecraft": "MET11",
                    "start_time": datetime.datetime(year=2020, month=10, day=20,
                                                    hour=19, minute=45, second=0)
                },
                filetype_info={
                    "file_type": "seviri"
                }
            )

            dataset_id = make_dataid(name="dummmy", resolution=3000)

            # Check that end_time is None for SEVIRI before the dataset has been loaded
            assert reader.end_time is None

            common_checks(ec_, reader, mock_file, dataset_id)

            # Check that end_time is now a valid datetime.datetime object after the dataset has been loaded
            assert reader.end_time == datetime.datetime(year=2020, month=10, day=20,
                                                        hour=19, minute=50, second=0)


            # Checks the correct execution of the _get_global_attributes and _get_metadata_from_msg functions
            attributes = reader._get_attributes()
            expected_attributes = {
                "orbital_parameters": {
                    "projection_longitude": 9.5
                },
                "sensor": "seviri",
                "platform_name": "Meteosat-11"
            }
            assert attributes == expected_attributes

            # Checks the reading of an array from the message
            reader._get_xarray_from_msg(0)

            # Checks that dask.array has been called with the correct arguments
            name, args, kwargs = da_.mock_calls[0]
            assert np.all(args[0] == np.ones((1200, 1000)))
            assert args[1] == chunk_size

            # Checks that xarray.DataArray has been called with the correct arguments
            name, args, kwargs = xr_.mock_calls[0]
            assert kwargs["dims"] == ("y", "x")

            # Checks the correct execution of the _get_proj_area function
            pdict, area_dict = reader._get_proj_area(0)

            expected_pdict = {
                "a": 6400000.,
                "b": 6300000.,
                "h": 32000000.,
                "ssp_lon": 9.5,
                "nlines": 1000,
                "ncols": 1200,
                "a_name": "msg_seviri_rss_3km",
                "a_desc": "MSG SEVIRI Rapid Scanning Service area definition with 3 km resolution",
                "p_id": "",
            }
            assert pdict == expected_pdict
            expected_area_dict = {
                "center_point": 500,
                "north": 1200,
                "east": 1,
                "west": 1000,
                "south": 1,
            }
            assert area_dict == expected_area_dict

            # Checks the correct execution of the get_area_def function
            with mock.patch("satpy.readers.eum_l2_grib.seviri_calculate_area_extent",
                            mock.Mock(name="seviri_calculate_area_extent")) as cae:
                with mock.patch("satpy.readers.eum_l2_grib.get_area_definition", mock.Mock()) as gad:
                    dataset_id = make_dataid(name="dummmy", resolution=400.)
                    reader.get_area_def(dataset_id)
                    # Asserts that seviri_calculate_area_extent has been called with the correct arguments
                    expected_args = ({"center_point": 500, "east": 1, "west": 1000, "south": 1, "north": 1200,
                                      "column_step": 400., "line_step": 400.},)
                    name, args, kwargs = cae.mock_calls[0]
                    assert args == expected_args
                    # Asserts that get_area_definition has been called with the correct arguments
                    name, args, kwargs = gad.mock_calls[0]
                    assert args[0] == expected_pdict
                    # The second argument must be the return result of seviri_calculate_area_extent
                    assert args[1]._extract_mock_name() == "seviri_calculate_area_extent()"


@pytest.mark.skipif(sys.platform.startswith("win"), reason="'eccodes' not supported on Windows")
@mock.patch("satpy.readers.eum_l2_grib.xr")
@mock.patch("satpy.readers.eum_l2_grib.da")
def test_fci_data_reading(da_, xr_, setup_reader):
    """Test the reading of fci data from the product."""
    from satpy.readers.eum_l2_grib import EUML2GribFileHandler
    from satpy.utils import get_legacy_chunk_size
    ec_ = setup_reader
    chunk_size = get_legacy_chunk_size()

    with mock.patch("builtins.open", mock.mock_open()) as mock_file:
        with mock.patch("satpy.readers.eum_l2_grib.ec", ec_):
            ec_.codes_get_values.return_value = np.ones(5568 * 5568)
            ec_.codes_get.side_effect = lambda gid, key: FAKE_FCI_MESSAGE[key]
            reader = EUML2GribFileHandler(
                filename="test.grib",
                filename_info={
                    "spacecraft_id": "1",
                    "start_time": datetime.datetime(year=2020, month=10, day=20,
                                                    hour=19, minute=40, second=0),
                    "end_time": datetime.datetime(year=2020, month=10, day=20,
                                                  hour=19, minute=50, second=0)
                },
                filetype_info={
                    "file_type": "fci"
                }
            )

            dataset_id = make_dataid(name="dummmy", resolution=2000)

            # Check end_time
            assert reader.end_time == datetime.datetime(year=2020, month=10, day=20,
                                                        hour=19, minute=50, second=0)

            common_checks(ec_, reader, mock_file, dataset_id)

            # Checks the correct execution of the _get_global_attributes and _get_metadata_from_msg functions
            attributes = reader._get_attributes()
            expected_attributes = {
                "orbital_parameters": {
                    "projection_longitude": 0.0
                },
                "sensor": "fci",
                "platform_name": "MTG-i1"
            }
            assert attributes == expected_attributes

            # Checks the reading of an array from the message
            reader._get_xarray_from_msg(0)

            # Checks that dask.array has been called with the correct arguments
            name, args, kwargs = da_.mock_calls[0]
            assert np.all(args[0] == np.ones((5568, 5568)))
            assert args[1] == chunk_size

            # Checks that xarray.DataArray has been called with the correct arguments
            name, args, kwargs = xr_.mock_calls[0]
            assert kwargs["dims"] == ("y", "x")

            # Checks the correct execution of the _get_proj_area function
            pdict, area_dict = reader._get_proj_area(0)

            expected_pdict = {
                "a": 6378140000.0,
                "b": 6356755000.0,
                "h": 35785830098.0,
                "ssp_lon": 0.0,
                "nlines": 5568,
                "ncols": 5568,
                "a_name": "msg_fci_fdss_2km",
                "a_desc": "MSG FCI Full Disk Scanning Service area definition with 2 km resolution",
                "p_id": ""
            }
            assert pdict == expected_pdict
            expected_area_dict = {
                "nlines": 5568,
                "ncols": 5568
            }
            assert area_dict == expected_area_dict

            # Checks the correct execution of the get_area_def function
            with mock.patch("satpy.readers.eum_l2_grib.fci_calculate_area_extent",
                            mock.Mock(name="fci_calculate_area_extent")) as cae:
                with mock.patch("satpy.readers.eum_l2_grib.get_area_definition", mock.Mock()) as gad:
                    dataset_id = make_dataid(name="dummmy", resolution=2000.)
                    reader.get_area_def(dataset_id)
                    # Asserts that seviri_calculate_area_extent has been called with the correct arguments
                    expected_args = ({"nlines": 5568, "ncols": 5568,
                                      "column_step": 2000., "line_step": 2000.},)
                    name, args, kwargs = cae.mock_calls[0]
                    assert args == expected_args
                    # Asserts that get_area_definition has been called with the correct arguments
                    name, args, kwargs = gad.mock_calls[0]
                    assert args[0] == expected_pdict
                    # The second argument must be the return result of seviri_calculate_area_extent
                    assert args[1]._extract_mock_name() == "fci_calculate_area_extent()"
