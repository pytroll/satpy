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
"""Unit tests for data access methods and properties of the Scene class."""
import math

import numpy as np
import pytest
import xarray as xr
from dask import array as da

from satpy import Scene
from satpy.dataset.dataid import default_id_keys_config
from satpy.tests.utils import FAKE_FILEHANDLER_END, FAKE_FILEHANDLER_START, make_cid, make_dataid

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - include_test_etc


@pytest.mark.usefixtures("include_test_etc")
class TestDataAccessMethods:
    """Test the scene class."""

    @pytest.mark.parametrize(
        ("reader", "filenames", "exp_sensors"),
        [
            ("fake1", ["fake1_1.txt"], {"fake_sensor"}),
            (None, {"fake1": ["fake1_1.txt"], "fake2_1ds": ["fake2_1ds_1.txt"]}, {"fake_sensor", "fake_sensor2"}),
        ]
    )
    def test_sensor_names_readers(self, reader, filenames, exp_sensors):
        """Test that Scene sensor_names handles different cases properly."""
        scene = Scene(reader=reader, filenames=filenames)
        assert scene.start_time == FAKE_FILEHANDLER_START
        assert scene.end_time == FAKE_FILEHANDLER_END
        assert scene.sensor_names == exp_sensors

    @pytest.mark.parametrize(
        ("include_reader", "added_sensor", "exp_sensors"),
        [
            (False, "my_sensor", {"my_sensor"}),
            (True, "my_sensor", {"my_sensor", "fake_sensor"}),
            (False, {"my_sensor"}, {"my_sensor"}),
            (True, {"my_sensor"}, {"my_sensor", "fake_sensor"}),
            (False, {"my_sensor1", "my_sensor2"}, {"my_sensor1", "my_sensor2"}),
            (True, {"my_sensor1", "my_sensor2"}, {"my_sensor1", "my_sensor2", "fake_sensor"}),
        ]
    )
    def test_sensor_names_added_datasets(self, include_reader, added_sensor, exp_sensors):
        """Test that Scene sensor_names handles contained sensors properly."""
        if include_reader:
            scene = Scene(reader="fake1", filenames=["fake1_1.txt"])
        else:
            scene = Scene()

        scene["my_ds"] = xr.DataArray([], attrs={"sensor": added_sensor})
        assert scene.sensor_names == exp_sensors

    def test_iter(self):
        """Test iteration over the scene."""
        scene = Scene()
        scene["1"] = xr.DataArray(np.arange(5))
        scene["2"] = xr.DataArray(np.arange(5))
        scene["3"] = xr.DataArray(np.arange(5))
        for x in scene:
            assert isinstance(x, xr.DataArray)

    def test_iter_by_area_swath(self):
        """Test iterating by area on a swath."""
        from pyresample.geometry import SwathDefinition
        scene = Scene()
        sd = SwathDefinition(lons=np.arange(5), lats=np.arange(5))
        scene["1"] = xr.DataArray(np.arange(5), attrs={'area': sd})
        scene["2"] = xr.DataArray(np.arange(5), attrs={'area': sd})
        scene["3"] = xr.DataArray(np.arange(5))
        for area_obj, ds_list in scene.iter_by_area():
            ds_list_names = set(ds['name'] for ds in ds_list)
            if area_obj is sd:
                assert ds_list_names == {'1', '2'}
            else:
                assert area_obj is None
                assert ds_list_names == {'3'}

    def test_bad_setitem(self):
        """Test setting an item wrongly."""
        scene = Scene()
        pytest.raises(ValueError, scene.__setitem__, '1', np.arange(5))

    def test_setitem(self):
        """Test setting an item."""
        from satpy.tests.utils import make_dataid
        scene = Scene()
        scene["1"] = ds1 = xr.DataArray(np.arange(5))
        expected_id = make_cid(**ds1.attrs)
        assert set(scene._datasets.keys()) == {expected_id}
        assert set(scene._wishlist) == {expected_id}

        did = make_dataid(name='oranges')
        scene[did] = ds1
        assert 'oranges' in scene
        nparray = np.arange(5*5).reshape(5, 5)
        with pytest.raises(ValueError):
            scene['apples'] = nparray
        assert 'apples' not in scene
        did = make_dataid(name='apples')
        scene[did] = nparray
        assert 'apples' in scene

    def test_getitem(self):
        """Test __getitem__ with names only."""
        scene = Scene()
        scene["1"] = ds1 = xr.DataArray(np.arange(5))
        scene["2"] = ds2 = xr.DataArray(np.arange(5))
        scene["3"] = ds3 = xr.DataArray(np.arange(5))
        assert scene['1'] is ds1
        assert scene['2'] is ds2
        assert scene['3'] is ds3
        pytest.raises(KeyError, scene.__getitem__, '4')
        assert scene.get('3') is ds3
        assert scene.get('4') is None

    def test_getitem_modifiers(self):
        """Test __getitem__ with names and modifiers."""
        # Return least modified item
        scene = Scene()
        scene['1'] = ds1_m0 = xr.DataArray(np.arange(5))
        scene[make_dataid(name='1', modifiers=('mod1',))
              ] = xr.DataArray(np.arange(5))
        assert scene['1'] is ds1_m0
        assert len(list(scene.keys())) == 2

        scene = Scene()
        scene['1'] = ds1_m0 = xr.DataArray(np.arange(5))
        scene[make_dataid(name='1', modifiers=('mod1',))
              ] = xr.DataArray(np.arange(5))
        scene[make_dataid(name='1', modifiers=('mod1', 'mod2'))
              ] = xr.DataArray(np.arange(5))
        assert scene['1'] is ds1_m0
        assert len(list(scene.keys())) == 3

        scene = Scene()
        scene[make_dataid(name='1', modifiers=('mod1', 'mod2'))
              ] = ds1_m2 = xr.DataArray(np.arange(5))
        scene[make_dataid(name='1', modifiers=('mod1',))
              ] = ds1_m1 = xr.DataArray(np.arange(5))
        assert scene['1'] is ds1_m1
        assert scene[make_dataid(name='1', modifiers=('mod1', 'mod2'))] is ds1_m2
        pytest.raises(KeyError, scene.__getitem__,
                      make_dataid(name='1', modifiers=tuple()))
        assert len(list(scene.keys())) == 2

    def test_getitem_slices(self):
        """Test __getitem__ with slices."""
        from pyresample.geometry import AreaDefinition, SwathDefinition
        from pyresample.utils import proj4_str_to_dict
        scene1 = Scene()
        scene2 = Scene()
        proj_dict = proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                                      '+lon_0=-95. +lat_0=25 +lat_1=25 '
                                      '+units=m +no_defs')
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            200,
            400,
            (-1000., -1500., 1000., 1500.),
        )
        swath_def = SwathDefinition(lons=np.zeros((5, 10)),
                                    lats=np.zeros((5, 10)))
        scene1["1"] = scene2["1"] = xr.DataArray(np.zeros((5, 10)))
        scene1["2"] = scene2["2"] = xr.DataArray(np.zeros((5, 10)),
                                                 dims=('y', 'x'))
        scene1["3"] = xr.DataArray(np.zeros((5, 10)), dims=('y', 'x'),
                                   attrs={'area': area_def})
        anc_vars = [xr.DataArray(np.ones((5, 10)),
                                 attrs={'name': 'anc_var', 'area': area_def})]
        attrs = {'ancillary_variables': anc_vars, 'area': area_def}
        scene1["3a"] = xr.DataArray(np.zeros((5, 10)),
                                    dims=('y', 'x'),
                                    attrs=attrs)
        scene2["4"] = xr.DataArray(np.zeros((5, 10)), dims=('y', 'x'),
                                   attrs={'area': swath_def})
        anc_vars = [xr.DataArray(np.ones((5, 10)),
                                 attrs={'name': 'anc_var', 'area': swath_def})]
        attrs = {'ancillary_variables': anc_vars, 'area': swath_def}
        scene2["4a"] = xr.DataArray(np.zeros((5, 10)),
                                    dims=('y', 'x'),
                                    attrs=attrs)
        new_scn1 = scene1[2:5, 2:8]
        new_scn2 = scene2[2:5, 2:8]
        for new_scn in [new_scn1, new_scn2]:
            # datasets without an area don't get sliced
            assert new_scn['1'].shape == (5, 10)
            assert new_scn['2'].shape == (5, 10)

        assert new_scn1['3'].shape == (3, 6)
        assert 'area' in new_scn1['3'].attrs
        assert new_scn1['3'].attrs['area'].shape == (3, 6)
        assert new_scn1['3a'].shape == (3, 6)
        a_var = new_scn1['3a'].attrs['ancillary_variables'][0]
        assert a_var.shape == (3, 6)

        assert new_scn2['4'].shape == (3, 6)
        assert 'area' in new_scn2['4'].attrs
        assert new_scn2['4'].attrs['area'].shape == (3, 6)
        assert new_scn2['4a'].shape == (3, 6)
        a_var = new_scn2['4a'].attrs['ancillary_variables'][0]
        assert a_var.shape == (3, 6)

    def test_contains(self):
        """Test contains."""
        scene = Scene()
        scene["1"] = xr.DataArray(np.arange(5),
                                  attrs={'wavelength': (0.1, 0.2, 0.3),
                                         '_satpy_id_keys': default_id_keys_config})
        assert '1' in scene
        assert 0.15 in scene
        assert '2' not in scene
        assert 0.31 not in scene

        scene = Scene()
        scene['blueberry'] = xr.DataArray(np.arange(5))
        scene['blackberry'] = xr.DataArray(np.arange(5))
        scene['strawberry'] = xr.DataArray(np.arange(5))
        scene['raspberry'] = xr.DataArray(np.arange(5))
        #  deepcode ignore replace~keys~list~compare: This is on purpose
        assert make_cid(name='blueberry') in scene.keys()
        assert make_cid(name='blueberry') in scene
        assert 'blueberry' in scene
        assert 'blueberry' not in scene.keys()

    def test_delitem(self):
        """Test deleting an item."""
        scene = Scene()
        scene["1"] = xr.DataArray(np.arange(5),
                                  attrs={'wavelength': (0.1, 0.2, 0.3),
                                         '_satpy_id_keys': default_id_keys_config})
        scene["2"] = xr.DataArray(np.arange(5),
                                  attrs={'wavelength': (0.4, 0.5, 0.6),
                                         '_satpy_id_keys': default_id_keys_config})
        scene["3"] = xr.DataArray(np.arange(5),
                                  attrs={'wavelength': (0.7, 0.8, 0.9),
                                         '_satpy_id_keys': default_id_keys_config})
        del scene['1']
        del scene['3']
        del scene[0.45]
        assert not scene._wishlist
        assert not list(scene._datasets.keys())
        pytest.raises(KeyError, scene.__delitem__, 0.2)


def _create_coarest_finest_data_array(shape, area_def, attrs=None):
    data_arr = xr.DataArray(
        da.arange(math.prod(shape)).reshape(shape),
        attrs={
            'area': area_def,
        })
    if attrs:
        data_arr.attrs.update(attrs)
    return data_arr


def _create_coarsest_finest_area_def(shape, extents):
    from pyresample import AreaDefinition
    proj_str = '+proj=lcc +datum=WGS84 +ellps=WGS84 +lon_0=-95. +lat_0=25 +lat_1=25 +units=m +no_defs'
    area_def = AreaDefinition(
        'test',
        'test',
        'test',
        proj_str,
        shape[1],
        shape[0],
        extents,
    )
    return area_def


def _create_coarsest_finest_swath_def(shape, extents, name_suffix):
    from pyresample import SwathDefinition
    if len(shape) == 1:
        lons_arr = da.linspace(extents[0], extents[2], shape[0], dtype=np.float32)
        lats_arr = da.linspace(extents[1], extents[3], shape[0], dtype=np.float32)
    else:
        lons_arr = da.repeat(da.linspace(extents[0], extents[2], shape[1], dtype=np.float32)[None, :], shape[0], axis=0)
        lats_arr = da.repeat(da.linspace(extents[1], extents[3], shape[0], dtype=np.float32)[:, None], shape[1], axis=1)
    lons_data_arr = xr.DataArray(lons_arr, attrs={"name": f"longitude{name_suffix}"})
    lats_data_arr = xr.DataArray(lats_arr, attrs={"name": f"latitude1{name_suffix}"})
    return SwathDefinition(lons_data_arr, lats_data_arr)


class TestFinestCoarsestArea:
    """Test the Scene logic for finding the finest and coarsest area."""

    @pytest.mark.parametrize(
        ("coarse_area", "fine_area"),
        [
            (_create_coarsest_finest_area_def((2, 5), (1000.0, 1500.0, -1000.0, -1500.0)),
             _create_coarsest_finest_area_def((4, 10), (1000.0, 1500.0, -1000.0, -1500.0))),
            (_create_coarsest_finest_area_def((2, 5), (-1000.0, -1500.0, 1000.0, 1500.0)),
             _create_coarsest_finest_area_def((4, 10), (-1000.0, -1500.0, 1000.0, 1500.0))),
            (_create_coarsest_finest_swath_def((2, 5), (1000.0, 1500.0, -1000.0, -1500.0), "1"),
             _create_coarsest_finest_swath_def((4, 10), (1000.0, 1500.0, -1000.0, -1500.0), "1")),
            (_create_coarsest_finest_swath_def((5,), (1000.0, 1500.0, -1000.0, -1500.0), "1"),
             _create_coarsest_finest_swath_def((10,), (1000.0, 1500.0, -1000.0, -1500.0), "1")),
        ]
    )
    def test_coarsest_finest_area_different_shape(self, coarse_area, fine_area):
        """Test 'coarsest_area' and 'finest_area' methods for upright areas."""
        ds1 = _create_coarest_finest_data_array(coarse_area.shape, coarse_area, {"wavelength": (0.1, 0.2, 0.3)})
        ds2 = _create_coarest_finest_data_array(fine_area.shape, fine_area, {"wavelength": (0.4, 0.5, 0.6)})
        ds3 = _create_coarest_finest_data_array(fine_area.shape, fine_area, {"wavelength": (0.7, 0.8, 0.9)})
        scn = Scene()
        scn["1"] = ds1
        scn["2"] = ds2
        scn["3"] = ds3

        assert scn.coarsest_area() is coarse_area
        assert scn.finest_area() is fine_area
        assert scn.coarsest_area(['2', '3']) is fine_area

    @pytest.mark.parametrize(
        ("area_def", "shifted_area"),
        [
            (_create_coarsest_finest_area_def((2, 5), (-1000.0, -1500.0, 1000.0, 1500.0)),
             _create_coarsest_finest_area_def((2, 5), (-900.0, -1400.0, 1100.0, 1600.0))),
            (_create_coarsest_finest_swath_def((2, 5), (-1000.0, -1500.0, 1000.0, 1500.0), "1"),
             _create_coarsest_finest_swath_def((2, 5), (-900.0, -1400.0, 1100.0, 1600.0), "2")),
        ],
    )
    def test_coarsest_finest_area_same_shape(self, area_def, shifted_area):
        """Test that two areas with the same shape are consistently returned.

        If two geometries (ex. two AreaDefinitions or two SwathDefinitions)
        have the same resolution (shape) but different
        coordinates, which one has the finer resolution would ultimately be
        determined by the semi-random ordering of the internal container of
        the Scene (a dict) if only pixel resolution was compared. This test
        makes sure that it is always the same object returned.

        """
        ds1 = _create_coarest_finest_data_array(area_def.shape, area_def)
        ds2 = _create_coarest_finest_data_array(area_def.shape, shifted_area)
        scn = Scene()
        scn["ds1"] = ds1
        scn["ds2"] = ds2
        course_area1 = scn.coarsest_area()

        scn = Scene()
        scn["ds2"] = ds2
        scn["ds1"] = ds1
        coarse_area2 = scn.coarsest_area()
        # doesn't matter what order they were added, this should be the same area
        assert coarse_area2 is course_area1


@pytest.mark.usefixtures("include_test_etc")
class TestComputePersist:
    """Test methods that compute the internal data in some way."""

    def test_compute_pass_through(self):
        """Test pass through of xarray compute."""
        import numpy as np
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['ds1'])
        scene = scene.compute()
        assert isinstance(scene['ds1'].data, np.ndarray)

    def test_persist_pass_through(self):
        """Test pass through of xarray persist."""
        from dask.array.utils import assert_eq
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['ds1'])
        scenep = scene.persist()
        assert_eq(scene['ds1'].data, scenep['ds1'].data)
        assert set(scenep['ds1'].data.dask).issubset(scene['ds1'].data.dask)
        assert len(scenep["ds1"].data.dask) == scenep['ds1'].data.npartitions

    def test_chunk_pass_through(self):
        """Test pass through of xarray chunk."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['ds1'])
        scene = scene.chunk(chunks=2)
        assert scene['ds1'].data.chunksize == (2, 2)
