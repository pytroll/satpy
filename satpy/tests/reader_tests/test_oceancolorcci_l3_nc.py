#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Satpy developers
#
# This file is part of Satpy.
#
# Satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# Satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Module for testing the satpy.readers.oceancolorcci_l3_nc module."""

import os
from datetime import datetime

import numpy as np
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path


@pytest.fixture()
def fake_dataset():
    """Create a CLAAS-like test dataset."""
    adg = xr.DataArray(
        [[1.0, 0.47, 4.5, 1.2], [0.2, 0, 1.3, 1.3]],
        dims=("y", "x")
    )
    atot = xr.DataArray(
        [[0.001, 0.08, 23.4, 0.1], [2.1, 1.2, 4.7, 306.]],
        dims=("y", "x")
    )
    kd = xr.DataArray(
        [[0.8, 0.01, 5.34, 1.23], [0.4, 1.0, 3.2, 1.23]],
        dims=("y", "x")
    )
    nobs = xr.DataArray(
        [[5, 118, 5, 100], [0, 15, 0, 1]],
        dims=("y", "x"),
        attrs={'_FillValue': 0}
    )
    nobs_filt = xr.DataArray(
        [[5, 118, 5, 100], [np.nan, 15, np.nan, 1]],
        dims=("y", "x"),
        attrs={'_FillValue': 0}
    )
    watcls = xr.DataArray(
        [[12.2, 0.01, 6.754, 5.33], [12.5, 101.5, 103.5, 204.]],
        dims=("y", "x")
    )
    attrs = {
        "geospatial_lon_resolution": "90",
        "geospatial_lat_resolution": "90",
        "geospatial_lon_min": -180.,
        "geospatial_lon_max": 180.,
        "geospatial_lat_min": -90.,
        "geospatial_lat_max": 90.,
        "time_coverage_start": "202108010000Z",
        "time_coverage_end": "202108312359Z",
    }
    return xr.Dataset(
        {
            "adg_490": adg,
            "water_class10": watcls,
            "SeaWiFS_nobs_sum": nobs,
            "test_nobs": nobs_filt,
            "kd_490": kd,
            "atot_665": atot,
        },
        attrs=attrs
    )


ds_dict = {'adg_490': 'adg_490',
           'water_class10': 'water_class10',
           'seawifs_nobs_sum': 'test_nobs',
           'kd_490': 'kd_490',
           'atot_665': 'atot_665'}

ds_list_all = ['adg_490', 'water_class10', 'seawifs_nobs_sum', 'kd_490', 'atot_665']
ds_list_iop = ['adg_490', 'water_class10', 'seawifs_nobs_sum', 'atot_665']
ds_list_kd = ['kd_490', 'water_class10', 'seawifs_nobs_sum']


@pytest.fixture
def fake_file_dict(fake_dataset, tmp_path):
    """Write a fake dataset to file."""
    fdict = {}
    filename = tmp_path / "ESACCI-OC-L3S-OC_PRODUCTS-MERGED-10M_MONTHLY_4km_GEO_PML_OCx_QAA-202112-fv5.0.nc"
    fake_dataset.to_netcdf(filename)
    fdict['bad_month'] = filename

    filename = tmp_path / "ESACCI-OC-L3S-OC_PRODUCTS-MERGED-2D_DAILY_4km_GEO_PML_OCx_QAA-202112-fv5.0.nc"
    fake_dataset.to_netcdf(filename)
    fdict['bad_day'] = filename

    filename = tmp_path / "ESACCI-OC-L3S-OC_PRODUCTS-MERGED-1M_MONTHLY_4km_GEO_PML_OCx_QAA-202112-fv5.0.nc"
    fake_dataset.to_netcdf(filename)
    fdict['ocprod_1m'] = filename

    filename = tmp_path / "ESACCI-OC-L3S-OC_PRODUCTS-MERGED-5D_DAILY_4km_GEO_PML_OCx_QAA-202112-fv5.0.nc"
    fake_dataset.to_netcdf(filename)
    fdict['ocprod_5d'] = filename

    filename = tmp_path / "ESACCI-OC-L3S-IOP-MERGED-8D_DAILY_4km_GEO_PML_RRS-20211117-fv5.0.nc"
    fake_dataset.to_netcdf(filename)
    fdict['iop_8d'] = filename

    filename = tmp_path / "ESACCI-OC-L3S-IOP-MERGED-1D_DAILY_4km_GEO_PML_OCx-202112-fv5.0.nc"
    fake_dataset.to_netcdf(filename)
    fdict['iop_1d'] = filename

    filename = tmp_path / "ESACCI-OC-L3S-K_490-MERGED-1D_DAILY_4km_GEO_PML_RRS-20210113-fv5.0.nc"
    fake_dataset.to_netcdf(filename)
    fdict['k490_1d'] = filename

    yield fdict


class TestOCCCIReader:
    """Test the Ocean Color reader."""

    def setup(self):
        """Set up the reader tests."""
        from satpy._config import config_search_paths

        self.yaml_file = "oceancolorcci_l3_nc.yaml"
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))

    def _create_reader_for_resolutions(self, filename):
        from satpy.readers import load_reader
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filename)
        assert len(filename) == len(files)
        reader.create_filehandlers(files)
        # Make sure we have some files
        assert reader.file_handlers
        return reader

    @pytest.fixture
    def area_exp(self):
        """Get expected area definition."""
        proj_dict = {'datum': 'WGS84', 'no_defs': 'None', 'proj': 'longlat', 'type': 'crs'}

        return AreaDefinition(
            area_id="gridded_occci",
            description="Full globe gridded area",
            proj_id="longlat",
            projection=proj_dict,
            area_extent=(-180., -90., 180., 90.),
            width=4,
            height=2,
        )

    def test_get_area_def(self, area_exp, fake_file_dict):
        """Test area definition."""
        reader = self._create_reader_for_resolutions([fake_file_dict['ocprod_1m']])
        res = reader.load([ds_list_all[0]])
        area = res[ds_list_all[0]].attrs['area']

        assert area.area_id == area_exp.area_id
        assert area.area_extent == area_exp.area_extent
        assert area.width == area_exp.width
        assert area.height == area_exp.height
        assert area.proj_dict == area_exp.proj_dict

    def test_bad_fname(self, fake_dataset, fake_file_dict):
        """Test case where an incorrect composite period is given."""
        reader = self._create_reader_for_resolutions([fake_file_dict['bad_month']])
        res = reader.load([ds_list_all[0]])
        assert len(res) == 0
        reader = self._create_reader_for_resolutions([fake_file_dict['bad_day']])
        res = reader.load([ds_list_all[0]])
        assert len(res) == 0

    def test_get_dataset_monthly_allprods(self, fake_dataset, fake_file_dict):
        """Test dataset loading."""
        reader = self._create_reader_for_resolutions([fake_file_dict['ocprod_1m']])
        # Check how many datasets are available. This file contains all of them.
        assert len(list(reader.available_dataset_names)) == 94
        res = reader.load(ds_list_all)
        assert len(res) == len(ds_list_all)
        for curds in ds_list_all:
            np.testing.assert_allclose(res[curds].values, fake_dataset[ds_dict[curds]].values)
            assert res[curds].attrs['sensor'] == 'merged'
            assert res[curds].attrs['composite_period'] == 'monthly'

    def test_get_dataset_8d_iopprods(self, fake_dataset, fake_file_dict):
        """Test dataset loading."""
        reader = self._create_reader_for_resolutions([fake_file_dict['iop_8d']])
        # Check how many datasets are available. This file contains all of them.
        assert len(list(reader.available_dataset_names)) == 70
        res = reader.load(ds_list_iop)
        assert len(res) == len(ds_list_iop)
        for curds in ds_list_iop:
            np.testing.assert_allclose(res[curds].values, fake_dataset[ds_dict[curds]].values)
            assert res[curds].attrs['sensor'] == 'merged'
            assert res[curds].attrs['composite_period'] == '8-day'

    def test_get_dataset_1d_kprods(self, fake_dataset, fake_file_dict):
        """Test dataset loading."""
        reader = self._create_reader_for_resolutions([fake_file_dict['k490_1d']])
        # Check how many datasets are available. This file contains all of them.
        assert len(list(reader.available_dataset_names)) == 25
        res = reader.load(ds_list_kd)
        assert len(res) == len(ds_list_kd)
        for curds in ds_list_kd:
            np.testing.assert_allclose(res[curds].values, fake_dataset[ds_dict[curds]].values)
            assert res[curds].attrs['sensor'] == 'merged'
            assert res[curds].attrs['composite_period'] == 'daily'

    def test_get_dataset_5d_allprods(self, fake_dataset, fake_file_dict):
        """Test dataset loading."""
        reader = self._create_reader_for_resolutions([fake_file_dict['ocprod_5d']])
        # Check how many datasets are available. This file contains all of them.
        assert len(list(reader.available_dataset_names)) == 94
        res = reader.load(ds_list_all)
        assert len(res) == len(ds_list_all)
        for curds in ds_list_all:
            np.testing.assert_allclose(res[curds].values, fake_dataset[ds_dict[curds]].values)
            assert res[curds].attrs['sensor'] == 'merged'
            assert res[curds].attrs['composite_period'] == '5-day'

    def test_start_time(self, fake_file_dict):
        """Test start time property."""
        reader = self._create_reader_for_resolutions([fake_file_dict['k490_1d']])
        assert reader.start_time == datetime(2021, 8, 1, 0, 0, 0)

    def test_end_time(self, fake_file_dict):
        """Test end time property."""
        reader = self._create_reader_for_resolutions([fake_file_dict['iop_8d']])
        assert reader.end_time == datetime(2021, 8, 31, 23, 59, 0)
