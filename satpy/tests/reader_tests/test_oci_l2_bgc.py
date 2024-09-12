# Copyright (c) 2024 Satpy developers
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
"""Tests for the 'oci_l2_bgc' reader."""

import numpy as np
import pytest
from pyresample.geometry import SwathDefinition

from satpy import Scene, available_readers

from .test_seadas_l2 import _create_seadas_chlor_a_netcdf_file

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path_factory


@pytest.fixture(scope="module")
def oci_l2_bgc_netcdf(tmp_path_factory):
    """Create MODIS SEADAS NetCDF file."""
    filename = "PACE_OCI.20211118T175853.L2.OC_BGC.V2_0.NRT.nc4"
    full_path = str(tmp_path_factory.mktemp("oci_l2_bgc") / filename)
    return _create_seadas_chlor_a_netcdf_file(full_path, "PACE", "OCI")


class TestSEADAS:
    """Test the OCI L2 file reader."""

    def test_available_reader(self):
        """Test that OCI L2 reader is available."""
        assert "oci_l2_bgc" in available_readers()

    def test_scene_available_datasets(self, oci_l2_bgc_netcdf):
        """Test that datasets are available."""
        scene = Scene(reader="oci_l2_bgc", filenames=oci_l2_bgc_netcdf)
        available_datasets = scene.all_dataset_names()
        assert len(available_datasets) > 0
        assert "chlor_a" in available_datasets

    @pytest.mark.parametrize("apply_quality_flags", [False, True])
    def test_load_chlor_a(self, oci_l2_bgc_netcdf, apply_quality_flags):
        """Test that we can load 'chlor_a'."""
        reader_kwargs = {"apply_quality_flags": apply_quality_flags}
        scene = Scene(reader="oci_l2_bgc", filenames=oci_l2_bgc_netcdf, reader_kwargs=reader_kwargs)
        scene.load(["chlor_a"])
        data_arr = scene["chlor_a"]
        assert data_arr.dims == ("y", "x")
        assert data_arr.attrs["platform_name"] == "PACE"
        assert data_arr.attrs["sensor"] == {"oci"}
        assert data_arr.attrs["units"] == "mg m^-3"
        assert data_arr.dtype.type == np.float32
        assert isinstance(data_arr.attrs["area"], SwathDefinition)
        assert data_arr.attrs["rows_per_scan"] == 0
        data = data_arr.data.compute()
        if apply_quality_flags:
            assert np.isnan(data[2, 2])
            assert np.count_nonzero(np.isnan(data)) == 1
        else:
            assert np.count_nonzero(np.isnan(data)) == 0
