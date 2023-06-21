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
from __future__ import annotations

from datetime import datetime
from typing import Iterable

import numpy as np
import pytest
import xarray as xr
from dask import array as da

from satpy import Scene

try:
    from datatree import DataTree
except ImportError:
    DataTree = None


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


def test_to_datatree_no_datatree(monkeypatch):
    """Test that datatree not being installed causes an exception with a nice message."""
    from satpy import _scene_converters
    monkeypatch.setattr(_scene_converters, "DataTree", None)
    scn = Scene()
    with pytest.raises(ImportError, match='xarray-datatree'):
        scn.to_xarray_datatree()


@pytest.mark.skipif(DataTree is None, reason="Optional 'xarray-datatree' library is not installed")
class TestToDataTree:
    """Test Scene conversion to an xarray DataTree."""

    def test_empty_scene(self):
        """Test that an empty Scene can be converted to a DataTree."""
        from datatree import DataTree

        scn = Scene()
        data_tree = scn.to_xarray_datatree()
        assert isinstance(data_tree, DataTree)
        assert len(data_tree) == 0

    @pytest.mark.parametrize(
        ("input_metadatas", "kwargs", "expected_groups"),
        [
            ([{"platform_name": "GOES-16", "sensor": "abi", "name": "ds1"}],
             {},
             {"/": (1, 0), "GOES-16": (1, 0), "GOES-16/abi": (0, 1)}),
            ([
                 {"platform_name": "GOES-16", "sensor": "abi", "name": "ds1"},
                 {"platform_name": "GOES-16", "sensor": "abi", "name": "ds2"},
             ],
             {},
             {"/": (1, 0), "GOES-16": (1, 0), "GOES-16/abi": (0, 2)}),
            ([
                 {"platform_name": "GOES-16", "sensor": "abi", "name": "ds1"},
                 {"platform_name": "GOES-16", "sensor": "abi", "name": "ds2"},
                 {"platform_name": "GOES-18", "sensor": "abi", "name": "ds3"},
                 {"platform_name": "GOES-18", "sensor": "abi", "name": "ds4"},
             ],
             {},
             {"/": (2, 0), "GOES-16": (1, 0), "GOES-16/abi": (0, 2), "GOES-18": (1, 0), "GOES-18/abi": (0, 2)}),
            ([
                 {"platform_name": "GOES-16", "sensor": "abi", "name": "ds1"},
                 {"platform_name": "GOES-16", "sensor": "glm", "name": "ds2"},
             ],
             {},
             {"/": (1, 0), "GOES-16": (2, 0), "GOES-16/abi": (0, 1), "GOES-16/glm": (0, 1)}),
            ([
                 {"platform_name": "GOES-16", "sensor": "abi", "name": "ds1"},
                 {"platform_name": "GOES-16", "sensor": "glm", "name": "ds2"},
             ],
             {"group_keys": ("sensor",)},
             {"/": (2, 0), "abi": (0, 1), "glm": (0, 1)}),
        ],
    )
    def test_basic_groupings(
            self,
            input_metadatas: Iterable[dict],
            kwargs: dict,
            expected_groups: dict[str, tuple[int, int]]
    ):
        """Test a Scene with a single DataArray being converted to a DataTree."""
        from datatree import DataTree

        scn = Scene()
        for input_metadata in input_metadatas:
            data_arr = xr.DataArray(da.zeros((10, 5), dtype=np.float32),
                                    attrs=input_metadata)
            scn[data_arr.attrs["name"]] = data_arr

        data_tree = scn.to_xarray_datatree(**kwargs)
        assert isinstance(data_tree, DataTree)

        for exp_group, (num_child_nodes, num_child_arrs) in expected_groups.items():
            group = data_tree[exp_group]
            assert len(group.children) == num_child_nodes
            assert len(group.data_vars) == num_child_arrs
            for data_arr in group.data_vars.values():
                assert isinstance(data_arr.data, da.Array)
