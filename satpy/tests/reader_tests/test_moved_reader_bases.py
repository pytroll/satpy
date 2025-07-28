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
    from satpy.readers import abi_base

    with pytest.warns(UserWarning, match=".*has been moved.*"):
        getattr(abi_base, "NC_ABI_BASE")


def test_fci_base_warns():
    """Test that there's a warning when importing from FCI base from the old location."""
    from satpy.readers import fci_base

    with pytest.warns(UserWarning, match=".*has been moved.*"):
        getattr(fci_base, "calculate_area_extent")


@pytest.mark.parametrize("name",
                         ["timecds2datetime",
                          "recarray2dict",
                          "get_service_mode",
                         ]
                         )
def test_eum_base_warns(name):
    """Test that there's a warning when importing from EUM base from the old location."""
    from satpy.readers import eum_base

    with pytest.warns(UserWarning, match=".*has been moved.*"):
        getattr(eum_base, name)


def test_fy4_base_warns():
    """Test that there's a warning when importing from FY4 base from the old location."""
    from satpy.readers import fy4_base

    with pytest.warns(UserWarning, match=".*has been moved.*"):
        getattr(fy4_base, "FY4Base")


@pytest.mark.parametrize("name",
                         ["from_sds",
                          "HDF4FileHandler",
                          "SDS",
                         ]
                         )
def test_hdf4_utils_warns(name):
    """Test that there's a warning when importing from hdf4 utils from the old location."""
    from satpy.readers import hdf4_utils

    with pytest.warns(UserWarning, match=".*has been moved.*"):
        getattr(hdf4_utils, name)


@pytest.mark.parametrize("name",
                         ["from_h5_array",
                          "HDF5FileHandler",
                         ]
                         )
def test_hdf5_utils_warns(name):
    """Test that there's a warning when importing from hdf5 utils from the old location."""
    from satpy.readers import hdf5_utils

    with pytest.warns(UserWarning, match=".*has been moved.*"):
        getattr(hdf5_utils, name)


@pytest.mark.parametrize("name",
                         ["interpolate",
                          "HDFEOSBaseFileReader",
                          "HDFEOSGeoReader",
                         ]
                         )
def test_hdfeos_base_warns(name):
    """Test that there's a warning when importing from hdfeos base from the old location."""
    from satpy.readers import hdfeos_base

    with pytest.warns(UserWarning, match=".*has been moved.*"):
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
    from satpy.readers import hrit_base

    with pytest.warns(UserWarning, match=".*has been moved.*"):
        getattr(hrit_base, name)


@pytest.mark.parametrize("name",
                         ["LINCFileHandler",
                         ]
                         )
def test_li_base_nc_warns(name):
    """Test that there's a warning when importing from LI netCDF4 base from the old location."""
    from satpy.readers import li_base_nc

    with pytest.warns(UserWarning, match=".*has been moved.*"):
        getattr(li_base_nc, name)


@pytest.mark.parametrize("name",
                         ["get_data_as_xarray",
                          "NetCDF4FileHandler",
                          "NetCDF4FsspecFileHandler",
                         ]
                         )
def test_netcdf_utils_warns(name):
    """Test that there's a warning when importing from netCDF4 utils from the old location."""
    from satpy.readers import netcdf_utils

    with pytest.warns(UserWarning, match=".*has been moved.*"):
        getattr(netcdf_utils, name)


@pytest.mark.parametrize("name",
                         ["get_cds_time",
                          "add_scanline_acq_time",
                          "dec10216",
                          "chebyshev",
                          "chebyshev_3d",
                          "get_satpos",
                          "calculate_area_extent",
                          "create_coef_dict",
                          "get_padding_area",
                          "pad_data_horizontally",
                          "pad_data_vertically",
                          "mask_bad_quality",
                          "round_nom_time",
                          "MpefProductHeader",
                          "SEVIRICalibrationAlgorithm",
                          "SEVIRICalibrationHandler",
                          "NoValidOrbitParams",
                          "OrbitPolynomial",
                          "OrbitPolynomialFinder",
                          "NominalCoefficients",
                          "GsicsCoefficients",
                          "MeirinkCoefficients",
                         ]
                         )
def test_seviri_base_warns(name):
    """Test that there's a warning when importing from SEVIRI base from the old location."""
    from satpy.readers import seviri_base

    with pytest.warns(UserWarning, match=".*has been moved.*"):
        getattr(seviri_base, name)


@pytest.mark.parametrize("name",
                         ["ViiNCBaseFileHandler",
                         ]
                         )
def test_vii_base_nc_warns(name):
    """Test that there's a warning when importing from VII NetCDF4 base from the old location."""
    from satpy.readers import vii_base_nc

    with pytest.warns(UserWarning, match=".*has been moved.*"):
        getattr(vii_base_nc, name)


@pytest.mark.parametrize("name",
                         ["JPSS_SDR_FileHandler",
                          "DATASET_KEYS",
                          "ATMS_DATASET_KEYS",
                          "VIIRS_DATASET_KEYS",
                         ]
                         )
def test_viirs_atms_sdr_warns(name):
    """Test that there's a warning when importing from VIIRS ATMS SRD base from the old location."""
    from satpy.readers import viirs_atms_sdr_base

    with pytest.warns(UserWarning, match=".*has been moved.*"):
        getattr(viirs_atms_sdr_base, name)


@pytest.mark.parametrize("name",
                         ["C1",
                          "C2",
                          "TIE_POINTS_FACTOR",
                          "SCAN_ALT_TIE_POINTS",
                          "MEAN_EARTH_RADIUS",
                         ]
                         )
def test_vii_utils_warns(name):
    """Test that there's a warning when importing from VII utils from the old location."""
    from satpy.readers import vii_utils

    with pytest.warns(UserWarning, match=".*has been moved.*"):
        getattr(vii_utils, name)


@pytest.mark.parametrize("name",
                         ["listify_string",
                          "load_yaml_configs",
                          "split_integer_in_most_equal_parts",
                          "AbstractYAMLReader",
                          "GenericYAMLReader",
                          "FileYAMLReader",
                          "GEOFlippableFileYAMLReader",
                          "GEOSegmentYAMLReader",
                          "GEOVariableSegmentYAMLReader",
                         ]
                         )
def test_yaml_reader_warns(name):
    """Test that there's a warning when importing from YAML reader from the old location."""
    from satpy.readers import yaml_reader

    with pytest.warns(UserWarning, match=".*has been moved.*"):
        getattr(yaml_reader, name)


@pytest.mark.parametrize("name",
                         ["process_delimiter",
                          "process_field",
                          "process_array",
                          "to_dtype",
                          "to_scaled_dtype",
                          "to_scales",
                          "parse_format",
                          "XMLFormat",
                         ]
                         )
def test_xmlformat_warns(name):
    """Test that there's a warning when importing from xmlformat from the old location."""
    from satpy.readers import xmlformat

    with pytest.warns(UserWarning, match=".*has been moved.*"):
        getattr(xmlformat, name)


@pytest.mark.parametrize("name",
                         ["open_dataset",
                          "BaseFileHandler",
                         ]
                         )
def test_file_handlers_warns(name):
    """Test that there's a warning when importing from file_handlers from the old location."""
    from satpy.readers import file_handlers

    with pytest.warns(UserWarning, match=".*has been moved.*"):
        getattr(file_handlers, name)
