#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2023 Satpy developers
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
"""Tests CF-compliant DataArray creation."""
import datetime

import pytest
import xarray as xr
from pyresample import AreaDefinition


def test_empty_collect_cf_datasets():
    """Test that if no DataArrays, collect_cf_datasets raise error."""
    from satpy.writers.cf.datasets import collect_cf_datasets

    with pytest.raises(RuntimeError):
        collect_cf_datasets(list_dataarrays=[])


class TestCollectCfDatasets:
    """Test case for collect_cf_dataset."""

    def test_collect_cf_dataarrays(self):
        """Test collecting CF datasets from a DataArray objects."""
        from satpy.writers.cf.datasets import _collect_cf_dataset

        geos = AreaDefinition(
            area_id='geos',
            description='geos',
            proj_id='geos',
            projection={'proj': 'geos', 'h': 35785831., 'a': 6378169., 'b': 6356583.8},
            width=2, height=2,
            area_extent=[-1, -1, 1, 1])

        # Define test datasets
        data = [[1, 2], [3, 4]]
        y = [1, 2]
        x = [1, 2]
        time = [1, 2]
        tstart = datetime.datetime(2019, 4, 1, 12, 0)
        tend = datetime.datetime(2019, 4, 1, 12, 15)
        list_dataarrays = [xr.DataArray(data=data, dims=('y', 'x'), coords={'y': y, 'x': x, 'acq_time': ('y', time)},
                                        attrs={'name': 'var1', 'start_time': tstart, 'end_time': tend, 'area': geos}),
                           xr.DataArray(data=data, dims=('y', 'x'), coords={'y': y, 'x': x, 'acq_time': ('y', time)},
                                        attrs={'name': 'var2', 'long_name': 'variable 2'})]

        # Collect datasets
        ds = _collect_cf_dataset(list_dataarrays, include_lonlats=True)

        # Test results
        assert len(ds.keys()) == 3
        assert set(ds.keys()) == {'var1', 'var2', 'geos'}

        da_var1 = ds['var1']
        da_var2 = ds['var2']
        assert da_var1.name == 'var1'
        assert da_var1.attrs['grid_mapping'] == 'geos'
        assert da_var1.attrs['long_name'] == 'var1'
        # variable 2
        assert 'grid_mapping' not in da_var2.attrs
        assert da_var2.attrs['long_name'] == 'variable 2'

    def test_collect_cf_dataarrays_with_latitude_named_lat(self):
        """Test collecting CF datasets with latitude named lat."""
        from satpy.writers.cf.datasets import _collect_cf_dataset

        data = [[75, 2], [3, 4]]
        y = [1, 2]
        x = [1, 2]
        geos = AreaDefinition(
            area_id='geos',
            description='geos',
            proj_id='geos',
            projection={'proj': 'geos', 'h': 35785831., 'a': 6378169., 'b': 6356583.8},
            width=2, height=2,
            area_extent=[-1, -1, 1, 1])
        datasets = {
            'var1': xr.DataArray(data=data,
                                 dims=('y', 'x'),
                                 coords={'y': y, 'x': x}),
            'var2': xr.DataArray(data=data,
                                 dims=('y', 'x'),
                                 coords={'y': y, 'x': x}),
            'lat': xr.DataArray(data=data,
                                dims=('y', 'x'),
                                coords={'y': y, 'x': x}),
            'lon': xr.DataArray(data=data,
                                dims=('y', 'x'),
                                coords={'y': y, 'x': x})}
        datasets['lat'].attrs['standard_name'] = 'latitude'
        datasets['var1'].attrs['standard_name'] = 'dummy'
        datasets['var2'].attrs['standard_name'] = 'dummy'
        datasets['var2'].attrs['area'] = geos
        datasets['var1'].attrs['area'] = geos
        datasets['lat'].attrs['name'] = 'lat'
        datasets['var1'].attrs['name'] = 'var1'
        datasets['var2'].attrs['name'] = 'var2'
        datasets['lon'].attrs['name'] = 'lon'

        datasets_list = [datasets[key] for key in datasets.keys()]
        datasets_list_no_latlon = [datasets[key] for key in ['var1', 'var2']]

        # Collect datasets
        ds = _collect_cf_dataset(datasets_list, include_lonlats=True)
        ds2 = _collect_cf_dataset(datasets_list_no_latlon, include_lonlats=True)

        # Test results
        assert len(ds.keys()) == 5
        assert set(ds.keys()) == {'var1', 'var2', 'lon', 'lat', 'geos'}
        with pytest.raises(KeyError):
            ds['var1'].attrs["latitude"]
        with pytest.raises(KeyError):
            ds['var1'].attrs["longitude"]
        assert ds2['var1']['latitude'].attrs['name'] == 'latitude'
        assert ds2['var1']['longitude'].attrs['name'] == 'longitude'
