#!/usr/bin/env python
# Copyright (c) 2016-2025 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Tests for reader base and utility modules that were moved to core sub-package raise warnings."""

import pytest


def test_abi_base_warns():
    """Test that there's a warning when importing from ABI base from the old location."""
    with pytest.warns(UserWarning):
        from satpy.readers import abi_base
        getattr(abi_base, "NC_ABI_BASE")


def test_fci_base_warns():
    """Test that there's a warning when importing from FCI base from the old location."""
    with pytest.warns(UserWarning):
        from satpy.readers import fci_base
        getattr(fci_base, "calculate_area_extent")


@pytest.mark.parametrize("name",
                         ["timecds2datetime",
                          "recarray2dict",
                          "get_service_mode",
                         ]
                         )
def test_eum_base_warns(name):
    """Test that there's a warning when importing from EUM base from the old location."""
    with pytest.warns(UserWarning):
        from satpy.readers import eum_base
        getattr(eum_base, name)


def test_fy4_base_warns():
    """Test that there's a warning when importing from FY4 base from the old location."""
    with pytest.warns(UserWarning):
        from satpy.readers import fy4_base
        getattr(fy4_base, "FY4Base")


@pytest.mark.parametrize("name",
                         ["from_sds",
                          "HDF4FileHandler",
                          "SDS",
                         ]
                         )
def test_hdf4_utils_warns(name):
    """Test that there's a warning when importing from hdf4 utils from the old location."""
    with pytest.warns(UserWarning):
        from satpy.readers import hdf4_utils
        getattr(hdf4_utils, name)


@pytest.mark.parametrize("name",
                         ["from_h5_array",
                          "HDF5FileHandler",
                         ]
                         )
def test_hdf5_utils_warns(name):
    """Test that there's a warning when importing from hdf5 utils from the old location."""
    with pytest.warns(UserWarning):
        from satpy.readers import hdf5_utils
        getattr(hdf5_utils, name)


@pytest.mark.parametrize("name",
                         ["interpolate",
                          "HDFEOSBaseFileReader",
                          "HDFEOSGeoReader",
                         ]
                         )
def test_hdfeos_base_warns(name):
    """Test that there's a warning when importing from hdfeos base from the old location."""
    with pytest.warns(UserWarning):
        from satpy.readers import hdfeos_base
        getattr(hdfeos_base, name)


@pytest.mark.parametrize("name",
                         ["decompress_file",
                          "decompress_buffer",
                          "get_header_id",
                          "get_header_content",
                          "HRITFileHandler",
                          "HRITSegment",
                         ]
                         )
def test_hrit_base_warns(name):
    """Test that there's a warning when importing from HRIT base from the old location."""
    with pytest.warns(UserWarning):
        from satpy.readers import hrit_base
        getattr(hrit_base, name)


@pytest.mark.parametrize("name",
                         ["LINCFileHandler",
                         ]
                         )
def test_li_base_nc_warns(name):
    """Test that there's a warning when importing from LI netCDF4 base from the old location."""
    with pytest.warns(UserWarning):
        from satpy.readers import li_base_nc
        getattr(li_base_nc, name)


@pytest.mark.parametrize("name",
                         ["get_data_as_xarray",
                          "NetCDF4FileHandler",
                          "NetCDF4FsspecFileHandler",
                         ]
                         )
def test_netcdf_utils_warns(name):
    """Test that there's a warning when importing from netCDF4 utils from the old location."""
    with pytest.warns(UserWarning):
        from satpy.readers import netcdf_utils
        getattr(netcdf_utils, name)
