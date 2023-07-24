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
"""Unit tests for resampling and crop-related functionality in scene.py."""
from unittest import mock

import numpy as np
import pytest
import xarray as xr
from dask import array as da

from satpy import Scene
from satpy.dataset.dataid import default_id_keys_config
from satpy.tests.utils import make_cid, make_dataid

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - include_test_etc


class TestSceneCrop:
    """Test creating new Scenes by cropping an existing Scene."""

    def test_crop(self):
        """Test the crop method."""
        from pyresample.geometry import AreaDefinition
        scene1 = Scene()
        area_extent = (-5570248.477339745, -5561247.267842293, 5567248.074173927,
                       5570248.477339745)
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'h': 35785831.0,
                     'lon_0': 0.0, 'proj': 'geos', 'units': 'm'}
        x_size = 3712
        y_size = 3712
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            x_size,
            y_size,
            area_extent,
        )
        area_def2 = AreaDefinition(
            'test2',
            'test2',
            'test2',
            proj_dict,
            x_size // 2,
            y_size // 2,
            area_extent,
            )
        scene1["1"] = xr.DataArray(np.zeros((y_size, x_size)))
        scene1["2"] = xr.DataArray(np.zeros((y_size, x_size)), dims=('y', 'x'))
        scene1["3"] = xr.DataArray(np.zeros((y_size, x_size)), dims=('y', 'x'),
                                   attrs={'area': area_def})
        scene1["4"] = xr.DataArray(np.zeros((y_size // 2, x_size // 2)), dims=('y', 'x'),
                                   attrs={'area': area_def2})

        # by area
        crop_area = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            x_size,
            y_size,
            (area_extent[0] + 10000., area_extent[1] + 500000.,
             area_extent[2] - 10000., area_extent[3] - 500000.)
        )
        new_scn1 = scene1.crop(crop_area)
        assert '1' in new_scn1
        assert '2' in new_scn1
        assert '3' in new_scn1
        assert new_scn1['1'].shape == (y_size, x_size)
        assert new_scn1['2'].shape == (y_size, x_size)
        assert new_scn1['3'].shape == (3380, 3708)
        assert new_scn1['4'].shape == (1690, 1854)

        # by lon/lat bbox
        new_scn1 = scene1.crop(ll_bbox=(-20., -5., 0, 0))
        assert '1' in new_scn1
        assert '2' in new_scn1
        assert '3' in new_scn1
        assert new_scn1['1'].shape == (y_size, x_size)
        assert new_scn1['2'].shape == (y_size, x_size)
        assert new_scn1['3'].shape == (184, 714)
        assert new_scn1['4'].shape == (92, 357)

        # by x/y bbox
        new_scn1 = scene1.crop(xy_bbox=(-200000., -100000., 0, 0))
        assert '1' in new_scn1
        assert '2' in new_scn1
        assert '3' in new_scn1
        assert new_scn1['1'].shape == (y_size, x_size)
        assert new_scn1['2'].shape == (y_size, x_size)
        assert new_scn1['3'].shape == (36, 70)
        assert new_scn1['4'].shape == (18, 35)

    def test_crop_epsg_crs(self):
        """Test the crop method when source area uses an EPSG code."""
        from pyresample.geometry import AreaDefinition

        scene1 = Scene()
        area_extent = (699960.0, 5390220.0, 809760.0, 5500020.0)
        x_size = 3712
        y_size = 3712
        area_def = AreaDefinition(
            'test', 'test', 'test',
            "EPSG:32630",
            x_size,
            y_size,
            area_extent,
        )
        scene1["1"] = xr.DataArray(np.zeros((y_size, x_size)), dims=('y', 'x'),
                                   attrs={'area': area_def})
        # by x/y bbox
        new_scn1 = scene1.crop(xy_bbox=(719695.7781587119, 5427887.407618969, 725068.1609052602, 5433708.364368956))
        assert '1' in new_scn1
        assert new_scn1['1'].shape == (198, 182)

    def test_crop_rgb(self):
        """Test the crop method on multi-dimensional data."""
        from pyresample.geometry import AreaDefinition
        scene1 = Scene()
        area_extent = (-5570248.477339745, -5561247.267842293, 5567248.074173927,
                       5570248.477339745)
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'h': 35785831.0,
                     'lon_0': 0.0, 'proj': 'geos', 'units': 'm'}
        x_size = 3712
        y_size = 3712
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            x_size,
            y_size,
            area_extent,
        )
        area_def2 = AreaDefinition(
            'test2',
            'test2',
            'test2',
            proj_dict,
            x_size // 2,
            y_size // 2,
            area_extent,
            )
        scene1["1"] = xr.DataArray(np.zeros((3, y_size, x_size)),
                                   dims=('bands', 'y', 'x'),
                                   attrs={'area': area_def})
        scene1["2"] = xr.DataArray(np.zeros((y_size // 2, 3, x_size // 2)),
                                   dims=('y', 'bands', 'x'),
                                   attrs={'area': area_def2})

        # by lon/lat bbox
        new_scn1 = scene1.crop(ll_bbox=(-20., -5., 0, 0))
        assert '1' in new_scn1
        assert '2' in new_scn1
        assert 'bands' in new_scn1['1'].dims
        assert 'bands' in new_scn1['2'].dims
        assert new_scn1['1'].shape == (3, 184, 714)
        assert new_scn1['2'].shape == (92, 3, 357)


@pytest.mark.usefixtures("include_test_etc")
class TestSceneResampling:
    """Test resampling a Scene to another Scene object."""

    def _fake_resample_dataset(self, dataset, dest_area, **kwargs):
        """Return copy of dataset pretending it was resampled."""
        return dataset.copy()

    def _fake_resample_dataset_force_20x20(self, dataset, dest_area, **kwargs):
        """Return copy of dataset pretending it was resampled to (20, 20) shape."""
        data = np.zeros((20, 20))
        attrs = dataset.attrs.copy()
        attrs['area'] = dest_area
        return xr.DataArray(
            data,
            dims=('y', 'x'),
            attrs=attrs,
        )

    @mock.patch('satpy.scene.resample_dataset')
    @pytest.mark.parametrize('datasets', [
        None,
        ('comp13', 'ds5', 'ds2'),
    ])
    def test_resample_scene_copy(self, rs, datasets):
        """Test that the Scene is properly copied during resampling.

        The Scene that is created as a copy of the original Scene should not
        be able to affect the original Scene object.

        """
        from pyresample.geometry import AreaDefinition
        rs.side_effect = self._fake_resample_dataset_force_20x20

        proj_str = ('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                    '+lon_0=-95. +lat_0=25 +lat_1=25 +units=m +no_defs')
        area_def = AreaDefinition('test', 'test', 'test', proj_str, 5, 5, (-1000., -1500., 1000., 1500.))
        area_def.get_area_slices = mock.MagicMock()
        scene = Scene(filenames=['fake1_1.txt', 'fake1_highres_1.txt'], reader='fake1')

        scene.load(['comp19'])
        new_scene = scene.resample(area_def, datasets=datasets)
        new_scene['new_ds'] = new_scene['comp19'].copy()

        scene.load(['ds1'])

        comp19_node = scene._dependency_tree['comp19']
        ds5_mod_id = make_dataid(name='ds5', modifiers=('res_change',))
        ds5_node = scene._dependency_tree[ds5_mod_id]
        comp13_node = scene._dependency_tree['comp13']

        assert comp13_node.data[1][0] is comp19_node.data[1][0]
        assert comp13_node.data[1][0] is ds5_node
        pytest.raises(KeyError, scene._dependency_tree.__getitem__, 'new_ds')

        # comp19 required resampling to produce so we should have its 3 deps
        # 1. comp13
        # 2. ds5
        # 3. ds2
        # Then we loaded ds1 separately so we should have
        # 4. ds1
        loaded_ids = list(scene.keys())
        assert len(loaded_ids) == 4
        for name in ('comp13', 'ds5', 'ds2', 'ds1'):
            assert any(x['name'] == name for x in loaded_ids)

        loaded_ids = list(new_scene.keys())
        assert len(loaded_ids) == 2
        assert loaded_ids[0] == make_cid(name='comp19')
        assert loaded_ids[1] == make_cid(name='new_ds')

    @mock.patch('satpy.scene.resample_dataset')
    def test_resample_scene_preserves_requested_dependencies(self, rs):
        """Test that the Scene is properly copied during resampling.

        The Scene that is created as a copy of the original Scene should not
        be able to affect the original Scene object.

        """
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict

        rs.side_effect = self._fake_resample_dataset
        proj_dict = proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                                      '+lon_0=-95. +lat_0=25 +lat_1=25 '
                                      '+units=m +no_defs')
        area_def = AreaDefinition('test', 'test', 'test', proj_dict, 5, 5, (-1000., -1500., 1000., 1500.))
        area_def.get_area_slices = mock.MagicMock()
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')

        # Set PYTHONHASHSEED to 0 in the interpreter to test as intended (comp26 comes before comp14)
        scene.load(['comp26', 'comp14'], generate=False)
        scene.resample(area_def, unload=True)
        new_scene_2 = scene.resample(area_def, unload=True)

        assert 'comp14' not in scene
        assert 'comp26' not in scene
        assert 'comp14' in new_scene_2
        assert 'comp26' in new_scene_2
        assert 'ds1' not in new_scene_2  # unloaded

    @mock.patch('satpy.scene.resample_dataset')
    def test_resample_reduce_data_toggle(self, rs):
        """Test that the Scene can be reduced or not reduced during resampling."""
        from pyresample.geometry import AreaDefinition

        rs.side_effect = self._fake_resample_dataset_force_20x20
        proj_str = ('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                    '+lon_0=-95. +lat_0=25 +lat_1=25 +units=m +no_defs')
        target_area = AreaDefinition('test', 'test', 'test', proj_str, 4, 4, (-1000., -1500., 1000., 1500.))
        area_def = AreaDefinition('test', 'test', 'test', proj_str, 5, 5, (-1000., -1500., 1000., 1500.))
        area_def.get_area_slices = mock.MagicMock()
        get_area_slices = area_def.get_area_slices
        get_area_slices.return_value = (slice(0, 3, None), slice(0, 3, None))
        area_def_big = AreaDefinition('test', 'test', 'test', proj_str, 10, 10, (-1000., -1500., 1000., 1500.))
        area_def_big.get_area_slices = mock.MagicMock()
        get_area_slices_big = area_def_big.get_area_slices
        get_area_slices_big.return_value = (slice(0, 6, None), slice(0, 6, None))

        # Test that data reduction can be disabled
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp19'])
        scene['comp19'].attrs['area'] = area_def
        scene['comp19_big'] = xr.DataArray(
            da.zeros((10, 10)), dims=('y', 'x'),
            attrs=scene['comp19'].attrs.copy())
        scene['comp19_big'].attrs['area'] = area_def_big
        scene['comp19_copy'] = scene['comp19'].copy()
        orig_slice_data = scene._slice_data
        # we force the below order of processing to test that success isn't
        # based on data of the same resolution being processed together
        test_order = [
            make_cid(**scene['comp19'].attrs),
            make_cid(**scene['comp19_big'].attrs),
            make_cid(**scene['comp19_copy'].attrs),
        ]
        with mock.patch('satpy.scene.Scene._slice_data') as slice_data, \
                mock.patch('satpy.dataset.dataset_walker') as ds_walker:
            ds_walker.return_value = test_order
            slice_data.side_effect = orig_slice_data
            scene.resample(target_area, reduce_data=False)
            assert not slice_data.called
            assert not get_area_slices.called
            scene.resample(target_area)
            assert slice_data.called_once
            assert get_area_slices.called_once
            scene.resample(target_area, reduce_data=True)
            # 2 times for each dataset
            # once for default (reduce_data=True)
            # once for kwarg forced to `True`
            assert slice_data.call_count == 2 * 3
            assert get_area_slices.called_once

    def test_resample_ancillary(self):
        """Test that the Scene reducing data does not affect final output."""
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        proj_dict = proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                                      '+lon_0=-95. +lat_0=25 +lat_1=25 '
                                      '+units=m +no_defs')
        area_def = AreaDefinition('test', 'test', 'test', proj_dict, 5, 5, (-1000., -1500., 1000., 1500.))
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')

        scene.load(['comp19', 'comp20'])
        scene['comp19'].attrs['area'] = area_def
        scene['comp19'].attrs['ancillary_variables'] = [scene['comp20']]
        scene['comp20'].attrs['area'] = area_def

        dst_area = AreaDefinition('dst', 'dst', 'dst',
                                  proj_dict,
                                  2, 2,
                                  (-1000., -1500., 0., 0.),
                                  )
        new_scene = scene.resample(dst_area)
        assert new_scene['comp20'] is new_scene['comp19'].attrs['ancillary_variables'][0]

    def test_resample_multi_ancillary(self):
        """Test that multiple ancillary variables are retained after resampling.

        This test corresponds to GH#2329
        """
        from pyresample import create_area_def
        sc = Scene()
        n = 5
        ar = create_area_def("a", 4087, resolution=1000, center=(0, 0), shape=(n, n))
        anc_vars = [xr.DataArray(
            np.arange(n*n).reshape(n, n)*i,
            dims=("y", "x"),
            attrs={"name": f"anc{i:d}", "area": ar}) for i in range(2)]
        sc["test"] = xr.DataArray(
            np.arange(n*n).reshape(n, n),
            dims=("y", "x"),
            attrs={
                "area": ar,
                "name": "test",
                "ancillary_variables": anc_vars})
        subset = create_area_def("b", 4087, resolution=800, center=(0, 0),
                                 shape=(n-1, n-1))
        ls = sc.resample(subset)
        assert ([av.attrs["name"] for av in sc["test"].attrs["ancillary_variables"]] ==
                [av.attrs["name"] for av in ls["test"].attrs["ancillary_variables"]])

    def test_resample_reduce_data(self):
        """Test that the Scene reducing data does not affect final output."""
        from pyresample.geometry import AreaDefinition
        proj_str = ('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                    '+lon_0=-95. +lat_0=25 +lat_1=25 +units=m +no_defs')
        area_def = AreaDefinition('test', 'test', 'test', proj_str, 20, 20, (-1000., -1500., 1000., 1500.))
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')

        scene.load(['comp19'])
        scene['comp19'].attrs['area'] = area_def
        dst_area = AreaDefinition('dst', 'dst', 'dst',
                                  proj_str,
                                  20, 20,
                                  (-1000., -1500., 0., 0.),
                                  )
        new_scene1 = scene.resample(dst_area, reduce_data=False)
        new_scene2 = scene.resample(dst_area)
        new_scene3 = scene.resample(dst_area, reduce_data=True)
        assert new_scene1['comp19'].shape == (20, 20, 3)
        assert new_scene2['comp19'].shape == (20, 20, 3)
        assert new_scene3['comp19'].shape == (20, 20, 3)

    @mock.patch('satpy.scene.resample_dataset')
    def test_no_generate_comp10(self, rs):
        """Test generating a composite after loading."""
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict

        rs.side_effect = self._fake_resample_dataset
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

        # it is fine that an optional prereq doesn't exist
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp10'], generate=False)
        assert any(ds_id['name'] == 'comp10' for ds_id in scene._wishlist)
        assert 'comp10' not in scene
        # two dependencies should have been loaded
        assert len(scene._datasets) == 2
        assert len(scene.missing_datasets) == 1

        new_scn = scene.resample(area_def, generate=False)
        assert 'comp10' not in scene
        # two dependencies should have been loaded
        assert len(scene._datasets) == 2
        assert len(scene.missing_datasets) == 1

        new_scn._generate_composites_from_loaded_datasets()
        assert any(ds_id['name'] == 'comp10' for ds_id in new_scn._wishlist)
        assert 'comp10' in new_scn
        assert not new_scn.missing_datasets

        # try generating them right away
        new_scn = scene.resample(area_def)
        assert any(ds_id['name'] == 'comp10' for ds_id in new_scn._wishlist)
        assert 'comp10' in new_scn
        assert not new_scn.missing_datasets

    def test_comp_loading_after_resampling_existing_sensor(self):
        """Test requesting a composite after resampling."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(["ds1", "ds2"])
        new_scn = scene.resample(resampler='native')

        # Can't load from readers after resampling
        with pytest.raises(KeyError):
            new_scn.load(["ds3"])

        # But we can load composites because the sensor composites were loaded
        # when the reader datasets were accessed
        new_scn.load(["comp2"])
        assert "comp2" in new_scn

    def test_comp_loading_after_resampling_new_sensor(self):
        """Test requesting a composite after resampling when the sensor composites weren't loaded before."""
        # this is our base Scene with sensor "fake_sensor2"
        scene1 = Scene(filenames=['fake2_3ds_1.txt'], reader='fake2_3ds')
        scene1.load(["ds2"])
        new_scn = scene1.resample(resampler='native')

        # Can't load from readers after resampling
        with pytest.raises(KeyError):
            new_scn.load(["ds3"])

        # Can't load the composite from fake_sensor composites yet
        # 'ds1' is missing
        with pytest.raises(KeyError):
            new_scn.load(["comp2"])

        # artificial DataArray "created by the user"
        # mimics a user adding their own data with the same sensor
        user_da = scene1["ds2"].copy()
        user_da.attrs["name"] = "ds1"
        user_da.attrs["sensor"] = {"fake_sensor2"}
        # Add 'ds1' that doesn't provide the 'fake_sensor' sensor
        new_scn["ds1"] = user_da
        with pytest.raises(KeyError):
            new_scn.load(["comp2"])
        assert "comp2" not in new_scn

        # artificial DataArray "created by the user"
        # mimics a user adding their own data with its own sensor to the Scene
        user_da = scene1["ds2"].copy()
        user_da.attrs["name"] = "ds1"
        user_da.attrs["sensor"] = {"fake_sensor"}
        # Now 'fake_sensor' composites have been loaded
        new_scn["ds1"] = user_da
        new_scn.load(["comp2"])
        assert "comp2" in new_scn

    def test_comp_loading_multisensor_composite_created_user(self):
        """Test that multisensor composite can be created manually.

        Test that if the user has created datasets "manually", that
        multi-sensor composites provided can still be read.
        """
        scene1 = Scene(filenames=["fake1_1.txt"], reader="fake1")
        scene1.load(["ds1"])
        scene2 = Scene(filenames=["fake4_1.txt"], reader="fake4")
        scene2.load(["ds4_b"])
        scene3 = Scene()
        scene3["ds1"] = scene1["ds1"]
        scene3["ds4_b"] = scene2["ds4_b"]
        scene3.load(["comp_multi"])
        assert "comp_multi" in scene3

    def test_comps_need_resampling_optional_mod_deps(self):
        """Test that a composite with complex dependencies.

        This is specifically testing the case where a compositor depends on
        multiple resolution prerequisites which themselves are composites.
        These sub-composites depend on data with a modifier that only has
        optional dependencies. This is a very specific use case and is the
        simplest way to present the problem (so far).

        The general issue is that the Scene loading creates the "ds13"
        dataset which already has one modifier on it. The "comp27"
        composite requires resampling so its 4 prerequisites + the
        requested "ds13" (from the reader which includes mod1 modifier)
        remain. If the DependencyTree is not copied properly in this
        situation then the new Scene object will have the composite
        dependencies without resolution in its dep tree, but have
        the DataIDs with the resolution in the dataset dictionary.
        This all results in the Scene trying to regenerate composite
        dependencies that aren't needed which fail.

        """
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        # should require resampling
        scene.load(['comp27', 'ds13'])
        assert 'comp27' not in scene
        assert 'ds13' in scene

        new_scene = scene.resample(resampler='native')
        assert len(list(new_scene.keys())) == 2
        assert 'comp27' in new_scene
        assert 'ds13' in new_scene


class TestSceneAggregation:
    """Test the scene's aggregate method."""

    def test_aggregate(self):
        """Test the aggregate method."""
        x_size = 3712
        y_size = 3712

        scene1 = self._create_test_data(x_size, y_size)

        scene2 = scene1.aggregate(func='sum', x=2, y=2)
        expected_aggregated_shape = (y_size / 2, x_size / 2)
        self._check_aggregation_results(expected_aggregated_shape, scene1, scene2, x_size, y_size)

    def test_custom_aggregate(self):
        """Test the aggregate method with custom function."""
        x_size = 3712
        y_size = 3712

        scene1 = self._create_test_data(x_size, y_size)

        scene2 = scene1.aggregate(func=np.sum, x=2, y=2)
        expected_aggregated_shape = (y_size / 2, x_size / 2)
        self._check_aggregation_results(expected_aggregated_shape, scene1, scene2, x_size, y_size)

    @staticmethod
    def _create_test_data(x_size, y_size):
        from pyresample.geometry import AreaDefinition
        scene1 = Scene()
        area_extent = (-5570248.477339745, -5561247.267842293, 5567248.074173927,
                       5570248.477339745)
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'h': 35785831.0,
                     'lon_0': 0.0, 'proj': 'geos', 'units': 'm'}
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            x_size,
            y_size,
            area_extent,
        )
        scene1["1"] = xr.DataArray(np.ones((y_size, x_size)),
                                   attrs={'_satpy_id_keys': default_id_keys_config})
        scene1["2"] = xr.DataArray(np.ones((y_size, x_size)),
                                   dims=('y', 'x'),
                                   attrs={'_satpy_id_keys': default_id_keys_config})
        scene1["3"] = xr.DataArray(np.ones((y_size, x_size)),
                                   dims=('y', 'x'),
                                   attrs={'area': area_def, '_satpy_id_keys': default_id_keys_config})
        scene1["4"] = xr.DataArray(np.ones((y_size, x_size)),
                                   dims=('y', 'x'),
                                   attrs={'area': area_def, 'standard_name': 'backscatter',
                                          '_satpy_id_keys': default_id_keys_config})
        return scene1

    def _check_aggregation_results(self, expected_aggregated_shape, scene1, scene2, x_size, y_size):
        assert scene1['1'] is scene2['1']
        assert scene1['2'] is scene2['2']
        np.testing.assert_allclose(scene2['3'].data, 4)
        assert scene2['1'].shape == (y_size, x_size)
        assert scene2['2'].shape == (y_size, x_size)
        assert scene2['3'].shape == expected_aggregated_shape
        assert 'standard_name' in scene2['4'].attrs
        assert scene2['4'].attrs['standard_name'] == 'backscatter'

    def test_aggregate_with_boundary(self):
        """Test aggregation with boundary argument."""
        x_size = 3711
        y_size = 3711

        scene1 = self._create_test_data(x_size, y_size)

        with pytest.raises(ValueError):
            scene1.aggregate(func='sum', x=2, y=2, boundary='exact')

        scene2 = scene1.aggregate(func='sum', x=2, y=2, boundary='trim')
        expected_aggregated_shape = (y_size // 2, x_size // 2)
        self._check_aggregation_results(expected_aggregated_shape, scene1, scene2, x_size, y_size)
