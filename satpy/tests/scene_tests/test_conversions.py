# Copyright (c) 2010-2023 Satpy developers
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
"""Unit tests for Scene conversion functionality."""
from datetime import datetime

import pytest
import xarray as xr
from dask import array as da

from satpy import Scene

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - include_test_etc


@pytest.mark.usefixtures("include_test_etc")
class TestSceneSerialization:
    """Test the Scene serialization."""

    def test_serialization_with_readers_and_data_arr(self):
        """Test that dask can serialize a Scene with readers."""
        from distributed.protocol import deserialize, serialize

        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['ds1'])
        cloned_scene = deserialize(*serialize(scene))
        assert scene._readers.keys() == cloned_scene._readers.keys()
        assert scene.all_dataset_ids == scene.all_dataset_ids


class TestSceneConversions:
    """Test Scene conversion to geoviews, xarray, etc."""

    def test_to_xarray_dataset_with_empty_scene(self):
        """Test converting empty Scene to xarray dataset."""
        scn = Scene()
        xrds = scn.to_xarray_dataset()
        assert isinstance(xrds, xr.Dataset)
        assert len(xrds.variables) == 0
        assert len(xrds.coords) == 0

    def test_to_xarray_with_empty_scene(self):
        """Test converting empty Scene to xarray."""
        scn = Scene()
        ds = scn.to_xarray()
        assert isinstance(ds, xr.Dataset)
        assert len(ds.variables) == 0
        assert len(ds.coords) == 0

    def test_to_xarray_with_single_area_scene(self):
        """Test converting single area Scene to xarray dataset."""
        from pyresample.geometry import AreaDefinition

        area = AreaDefinition('test', 'test', 'test',
                              {'proj': 'geos', 'lon_0': -95.5, 'h': 35786023.0},
                              2, 2, [-200, -200, 200, 200])
        data_array = xr.DataArray(da.zeros((2, 2), chunks=-1),
                                  dims=('y', 'x'),
                                  attrs={'start_time': datetime(2018, 1, 1), 'area': area})

        scn = Scene()
        scn['var1'] = data_array
        ds = scn.to_xarray()

        # Assert dataset type
        assert isinstance(ds, xr.Dataset)
        # Assert var1 is a Dataset variables
        assert "var1" in ds.data_vars

        # Assert by default it include lon lats
        assert "latitude" in ds.coords
        assert "longitude" in ds.coords

        # Assert include_lonlats=False works
        ds = scn.to_xarray(include_lonlats=False)
        assert "latitude" not in ds.coords
        assert "longitude" not in ds.coords

    def test_to_xarray_with_multiple_area_scene(self):
        """Test converting muiltple area Scene to xarray."""
        from pyresample.geometry import AreaDefinition

        area1 = AreaDefinition('test', 'test', 'test',
                               {'proj': 'geos', 'lon_0': -95.5, 'h': 35786023.0},
                               2, 2, [-200, -200, 200, 200])
        area2 = AreaDefinition('test', 'test', 'test',
                               {'proj': 'geos', 'lon_0': -95.5, 'h': 35786023.0},
                               4, 4, [-200, -200, 200, 200])

        data_array1 = xr.DataArray(da.zeros((2, 2), chunks=-1),
                                   dims=('y', 'x'),
                                   attrs={'start_time': datetime(2018, 1, 1), 'area': area1})
        data_array2 = xr.DataArray(da.zeros((4, 4), chunks=-1),
                                   dims=('y', 'x'),
                                   attrs={'start_time': datetime(2018, 1, 1), 'area': area2})
        scn = Scene()
        scn['var1'] = data_array1
        scn['var2'] = data_array2

        # TODO: in future adapt for DataTree implementation
        with pytest.raises(NotImplementedError):
            _ = scn.to_xarray()

    def test_geoviews_basic_with_area(self):
        """Test converting a Scene to geoviews with an AreaDefinition."""
        from pyresample.geometry import AreaDefinition
        scn = Scene()
        area = AreaDefinition('test', 'test', 'test',
                              {'proj': 'geos', 'lon_0': -95.5, 'h': 35786023.0},
                              2, 2, [-200, -200, 200, 200])
        scn['ds1'] = xr.DataArray(da.zeros((2, 2), chunks=-1), dims=('y', 'x'),
                                  attrs={'start_time': datetime(2018, 1, 1),
                                         'area': area})
        gv_obj = scn.to_geoviews()
        # we assume that if we got something back, geoviews can use it
        assert gv_obj is not None

    def test_geoviews_basic_with_swath(self):
        """Test converting a Scene to geoviews with a SwathDefinition."""
        from pyresample.geometry import SwathDefinition
        scn = Scene()
        lons = xr.DataArray(da.zeros((2, 2)))
        lats = xr.DataArray(da.zeros((2, 2)))
        area = SwathDefinition(lons, lats)
        scn['ds1'] = xr.DataArray(da.zeros((2, 2), chunks=-1), dims=('y', 'x'),
                                  attrs={'start_time': datetime(2018, 1, 1),
                                         'area': area})
        gv_obj = scn.to_geoviews()
        # we assume that if we got something back, geoviews can use it
        assert gv_obj is not None
