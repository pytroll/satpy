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
"""Unit tests for loading-related functionality in scene.py."""

from unittest import mock

import pytest
import xarray as xr
from dask import array as da

from satpy import Scene
from satpy.tests.utils import make_cid, make_dataid, make_dsq, spy_decorator

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - include_test_etc


@pytest.mark.usefixtures("include_test_etc")
class TestSceneAllAvailableDatasets:
    """Test the Scene's handling of various dependencies."""

    def test_all_datasets_no_readers(self):
        """Test all datasets with no reader."""
        scene = Scene()
        pytest.raises(KeyError, scene.all_dataset_ids, reader_name='fake')
        id_list = scene.all_dataset_ids()
        assert id_list == []
        # no sensors are loaded so we shouldn't get any comps either
        id_list = scene.all_dataset_ids(composites=True)
        assert id_list == []

    def test_all_dataset_names_no_readers(self):
        """Test all dataset names with no reader."""
        scene = Scene()
        pytest.raises(KeyError, scene.all_dataset_names, reader_name='fake')
        name_list = scene.all_dataset_names()
        assert name_list == []
        # no sensors are loaded so we shouldn't get any comps either
        name_list = scene.all_dataset_names(composites=True)
        assert name_list == []

    def test_available_dataset_no_readers(self):
        """Test the available datasets without a reader."""
        scene = Scene()
        pytest.raises(
            KeyError, scene.available_dataset_ids, reader_name='fake')
        name_list = scene.available_dataset_ids()
        assert name_list == []
        # no sensors are loaded so we shouldn't get any comps either
        name_list = scene.available_dataset_ids(composites=True)
        assert name_list == []

    def test_available_dataset_names_no_readers(self):
        """Test the available dataset names without a reader."""
        scene = Scene()
        pytest.raises(
            KeyError, scene.available_dataset_names, reader_name='fake')
        name_list = scene.available_dataset_names()
        assert name_list == []
        # no sensors are loaded so we shouldn't get any comps either
        name_list = scene.available_dataset_names(composites=True)
        assert name_list == []

    def test_all_datasets_one_reader(self):
        """Test all datasets for one reader."""
        scene = Scene(filenames=['fake1_1.txt'],
                      reader='fake1')
        id_list = scene.all_dataset_ids()
        # 20 data products + 6 lon/lat products
        num_reader_ds = 21 + 6
        assert len(id_list) == num_reader_ds
        id_list = scene.all_dataset_ids(composites=True)
        assert len(id_list) == num_reader_ds + 33

    def test_all_datasets_multiple_reader(self):
        """Test all datasets for multiple readers."""
        scene = Scene(filenames={'fake1_1ds': ['fake1_1ds_1.txt'],
                                 'fake2_1ds': ['fake2_1ds_1.txt']})
        id_list = scene.all_dataset_ids()
        assert len(id_list) == 2
        id_list = scene.all_dataset_ids(composites=True)
        # ds1 and ds2 => 2
        # composites that use these two datasets => 11
        assert len(id_list) == 2 + 11

    def test_available_datasets_one_reader(self):
        """Test the available datasets for one reader."""
        scene = Scene(filenames=['fake1_1ds_1.txt'],
                      reader='fake1_1ds')
        id_list = scene.available_dataset_ids()
        assert len(id_list) == 1
        id_list = scene.available_dataset_ids(composites=True)
        # ds1, comp1, comp14, comp16, static_image, comp26
        assert len(id_list) == 6

    def test_available_composite_ids_missing_available(self):
        """Test available_composite_ids when a composites dep is missing."""
        scene = Scene(filenames=['fake1_1ds_1.txt'],
                      reader='fake1_1ds')
        assert 'comp2' not in scene.available_composite_names()

    def test_available_composites_known_versus_all(self):
        """Test available_composite_ids when some datasets aren't available."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1',
                      reader_kwargs={"not_available": ["ds2", "ds3"]})
        all_comps = scene.all_composite_names()
        avail_comps = scene.available_composite_names()
        # there should always be more known composites than available composites
        assert len(all_comps) > len(avail_comps)
        for not_avail_comp in ("comp2", "comp3"):
            assert not_avail_comp in all_comps
            assert not_avail_comp not in avail_comps

    def test_available_comps_no_deps(self):
        """Test Scene available composites when composites don't have a dependency."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        all_comp_ids = scene.available_composite_ids()
        assert make_cid(name='static_image') in all_comp_ids
        available_comp_ids = scene.available_composite_ids()
        assert make_cid(name='static_image') in available_comp_ids

    def test_available_when_sensor_none_in_preloaded_dataarrays(self):
        """Test Scene available composites when existing loaded arrays have sensor set to None.

        Some readers or composites (ex. static images) don't have a sensor and
        developers choose to set it to `None`. This test makes sure this
        doesn't break available composite IDs.

        """
        scene = _scene_with_data_array_none_sensor()
        available_comp_ids = scene.available_composite_ids()
        assert make_cid(name='static_image') in available_comp_ids


@pytest.mark.usefixtures("include_test_etc")
class TestBadLoading:
    """Test the Scene object's `.load` method with bad inputs."""

    def test_load_str(self):
        """Test passing a string to Scene.load."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        pytest.raises(TypeError, scene.load, 'ds1')

    def test_load_no_exist(self):
        """Test loading a dataset that doesn't exist."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        pytest.raises(KeyError, scene.load, ['im_a_dataset_that_doesnt_exist'])


@pytest.mark.usefixtures("include_test_etc")
class TestLoadingReaderDatasets:
    """Test the Scene object's `.load` method for datasets coming from a reader."""

    def test_load_no_exist2(self):
        """Test loading a dataset that doesn't exist then another load."""
        from satpy.readers.yaml_reader import FileYAMLReader
        load_mock = spy_decorator(FileYAMLReader.load)
        with mock.patch.object(FileYAMLReader, 'load', load_mock):
            lmock = load_mock.mock
            scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
            scene.load(['ds9_fail_load'])
            loaded_ids = list(scene._datasets.keys())
            assert len(loaded_ids) == 0
            lmock.assert_called_once_with(
                {make_dataid(name='ds9_fail_load', wavelength=(1.0, 1.1, 1.2))})

            scene.load(['ds1'])
            loaded_ids = list(scene._datasets.keys())
            assert lmock.call_count == 2
            # most recent call should have only been ds1
            lmock.assert_called_with({
                make_dataid(name='ds1', resolution=250, calibration='reflectance', modifiers=tuple()),
            })
            assert len(loaded_ids) == 1

    def test_load_ds1_no_comps(self):
        """Test loading one dataset with no loaded compositors."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['ds1'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0] == make_dataid(name='ds1', resolution=250, calibration='reflectance', modifiers=tuple())

    def test_load_ds1_load_twice(self):
        """Test loading one dataset with no loaded compositors."""
        from satpy.readers.yaml_reader import FileYAMLReader
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['ds1'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0] == make_dataid(name='ds1', resolution=250, calibration='reflectance', modifiers=tuple())

        load_mock = spy_decorator(FileYAMLReader.load)
        with mock.patch.object(FileYAMLReader, 'load', load_mock):
            lmock = load_mock.mock
            scene.load(['ds1'])
            loaded_ids = list(scene._datasets.keys())
            assert len(loaded_ids) == 1
            assert loaded_ids[0] == make_dataid(name='ds1',
                                                resolution=250,
                                                calibration='reflectance',
                                                modifiers=tuple())
            assert not lmock.called, ("Reader.load was called again when "
                                      "loading something that's already "
                                      "loaded")

    def test_load_ds1_unknown_modifier(self):
        """Test loading one dataset with no loaded compositors."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        pytest.raises(KeyError, scene.load,
                      [make_dataid(name='ds1', modifiers=('_fake_bad_mod_',))])

    def test_load_ds4_cal(self):
        """Test loading a dataset that has two calibration variations."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['ds4'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0]['calibration'] == 'reflectance'

    @pytest.mark.parametrize(
        ("input_filenames", "load_kwargs", "exp_resolution"),
        [
            (["fake1_1.txt", "fake1_highres_1.txt"], {}, 250),
            (["fake1_1.txt"], {"resolution": [500, 1000]}, 500),
            (["fake1_1.txt"], {"modifiers": tuple()}, 500),
            (["fake1_1.txt"], {}, 500),
        ]
    )
    def test_load_ds5_variations(self, input_filenames, load_kwargs, exp_resolution):
        """Test loading a dataset has multiple resolutions available."""
        scene = Scene(filenames=input_filenames, reader='fake1')
        scene.load(['ds5'], **load_kwargs)
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0]['name'] == 'ds5'
        assert loaded_ids[0]['resolution'] == exp_resolution

    def test_load_ds5_multiple_resolution_loads(self):
        """Test loading a dataset with multiple resolutions available as separate loads."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['ds5'], resolution=1000)
        scene.load(['ds5'], resolution=500)
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 2
        assert loaded_ids[0]['name'] == 'ds5'
        assert loaded_ids[0]['resolution'] == 500
        assert loaded_ids[1]['name'] == 'ds5'
        assert loaded_ids[1]['resolution'] == 1000

    def test_load_ds6_wl(self):
        """Test loading a dataset by wavelength."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load([0.22])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0]['name'] == 'ds6'

    def test_load_ds9_fail_load(self):
        """Test loading a dataset that will fail during load."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['ds9_fail_load'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 0


@pytest.mark.usefixtures("include_test_etc")
class TestLoadingComposites:
    """Test the Scene object's `.load` method for composites."""

    @pytest.mark.parametrize(
        ("comp_name", "exp_id_or_name"),
        [
            pytest.param("comp1", make_cid(name="comp1"), id="composite with one required reader prereq"),
            pytest.param("comp4", make_cid(name="comp4"), id="composite with a required composite prereq"),
            pytest.param("comp5", make_cid(name="comp5"), id="composite with an optional reader prereq"),
            pytest.param("comp6", make_cid(name="comp6"), id="composite with an optional composite prereq"),
            pytest.param("comp9", make_cid(name="comp9"), id="composite with an unknown optional prereq"),
            pytest.param("comp10", make_cid(name="comp10"), id="composite with a modified required prereq"),
            pytest.param("comp11", make_cid(name="comp11"), id="composite with required prereqs as wavelength"),
            pytest.param("comp12", make_cid(name="comp12"),
                         id="composite with required prereqs as modified wavelengths"),
            pytest.param("comp13", make_cid(name="comp13"), id="composite with modified res-changed prereq"),
            pytest.param("comp14", make_cid(name="comp14", resolution=555),
                         id="composite that changes DataID resolution"),
            pytest.param("comp16", make_cid(name="comp16"), id="composite with unloadable optional prereq"),
            pytest.param("comp20", make_cid(name="comp20"), id="composite with prereq with modifier with opt prereq"),
            pytest.param("comp21", make_cid(name="comp21"),
                         id="composite with prereq with modifier with unloadable opt prereq"),
            pytest.param("comp22", make_cid(name="comp22"),
                         id="composite with prereq with modifier with only opt prereqs"),
            pytest.param("ahi_green", make_cid(name="ahi_green"), id="ahi_green composite"),
        ]
    )
    def test_single_composite_loading(self, comp_name, exp_id_or_name):
        """Test that certain composites can be loaded individually."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load([comp_name])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        if isinstance(exp_id_or_name, str):
            assert loaded_ids[0]["name"] == exp_id_or_name
        else:
            assert loaded_ids[0] == exp_id_or_name

    def test_load_multiple_resolutions(self):
        """Test loading a dataset has multiple resolutions available with different resolutions."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        comp25 = make_cid(name='comp25', resolution=1000)
        scene[comp25] = xr.DataArray([], attrs={'name': 'comp25', 'resolution': 1000})
        scene.load(['comp25'], resolution=500)

        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 2
        assert loaded_ids[0]['name'] == 'comp25'
        assert loaded_ids[0]['resolution'] == 500
        assert loaded_ids[1]['name'] == 'comp25'
        assert loaded_ids[1]['resolution'] == 1000

    def test_load_same_subcomposite(self):
        """Test loading a composite and one of it's subcomposites at the same time."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp24', 'comp25'], resolution=500)
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 2
        assert loaded_ids[0]['name'] == 'comp24'
        assert loaded_ids[0]['resolution'] == 500
        assert loaded_ids[1]['name'] == 'comp25'
        assert loaded_ids[1]['resolution'] == 500

    def test_load_comp8(self):
        """Test loading a composite that has a non-existent prereq."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        pytest.raises(KeyError, scene.load, ['comp8'])

    def test_load_comp15(self):
        """Test loading a composite whose prerequisites can't be loaded.

        Note that the prereq exists in the reader, but fails in loading.

        """
        # it is fine that an optional prereq doesn't exist
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp15'])
        loaded_ids = list(scene._datasets.keys())
        assert not loaded_ids

    def test_load_comp17(self):
        """Test loading a composite that depends on a composite that won't load."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp17'])
        loaded_ids = list(scene._datasets.keys())
        assert not loaded_ids

    def test_load_comp18(self):
        """Test loading a composite that depends on an incompatible area modified dataset."""
        # it is fine that an optional prereq doesn't exist
        scene = Scene(filenames=['fake1_1.txt', 'fake1_highres_1.txt'], reader='fake1')
        scene.load(['comp18'])
        loaded_ids = list(scene._datasets.keys())
        # depends on:
        #   ds3
        #   ds4 (mod1, mod3)
        #   ds5 (mod1, incomp_areas)
        # We should end up with ds3, ds4 (mod1, mod3), ds5 (mod1), and ds1
        # for the incomp_areas modifier
        assert len(loaded_ids) == 4  # the 1 dependencies
        assert 'ds3' in scene._datasets
        assert make_dataid(name='ds4', calibration='reflectance',
                           modifiers=('mod1', 'mod3')) in scene._datasets
        assert make_dataid(name='ds5', resolution=250,
                           modifiers=('mod1',)) in scene._datasets

    def test_load_comp18_2(self):
        """Test loading a composite that depends on an incompatible area modified dataset.

        Specifically a modified dataset where the modifier has optional
        dependencies.

        """
        # it is fine that an optional prereq doesn't exist
        scene = Scene(filenames=['fake1_1.txt', 'fake1_highres_1.txt'], reader='fake1')
        scene.load(['comp18_2'])
        loaded_ids = list(scene._datasets.keys())
        # depends on:
        #   ds3
        #   ds4 (mod1, mod3)
        #   ds5 (mod1, incomp_areas_opt)
        # We should end up with ds3, ds4 (mod1, mod3), ds5 (mod1), and ds1
        # and ds2 for the incomp_areas_opt modifier
        assert len(loaded_ids) == 5  # the 1 dependencies
        assert 'ds3' in scene._datasets
        assert 'ds2' in scene._datasets
        assert make_dataid(name='ds4', calibration='reflectance',
                           modifiers=('mod1', 'mod3')) in scene._datasets
        assert make_dataid(name='ds5', resolution=250,
                           modifiers=('mod1',)) in scene._datasets

    def test_load_comp19(self):
        """Test loading a composite that shares a dep with a dependency.

        More importantly test that loading a dependency that depends on
        the same dependency as this composite (a sibling dependency) and
        that sibling dependency includes a modifier. This test makes sure
        that the Node in the dependency tree is the exact same node.

        """
        # Check dependency tree nodes
        # initialize the dep tree without loading the data
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene._update_dependency_tree({'comp19'}, None)

        this_node = scene._dependency_tree['comp19']
        shared_dep_id = make_dataid(name='ds5', modifiers=('res_change',))
        shared_dep_expected_node = scene._dependency_tree[shared_dep_id]
        # get the node for the first dep in the prereqs list of the
        # comp13 node
        shared_dep_node = scene._dependency_tree['comp13'].data[1][0]
        shared_dep_node2 = this_node.data[1][0]
        assert shared_dep_expected_node is shared_dep_node
        assert shared_dep_expected_node is shared_dep_node2

        scene.load(['comp19'])

        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0] == make_cid(name='comp19')

    def test_load_multiple_comps(self):
        """Test loading multiple composites."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp1', 'comp2', 'comp3', 'comp4', 'comp5', 'comp6',
                    'comp7', 'comp9', 'comp10'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 9

    def test_load_multiple_comps_separate(self):
        """Test loading multiple composites, one at a time."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp10'])
        scene.load(['comp9'])
        scene.load(['comp7'])
        scene.load(['comp6'])
        scene.load(['comp5'])
        scene.load(['comp4'])
        scene.load(['comp3'])
        scene.load(['comp2'])
        scene.load(['comp1'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 9

    def test_load_modified(self):
        """Test loading a modified dataset."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load([make_dsq(name='ds1', modifiers=('mod1', 'mod2'))])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0]['modifiers'] == ('mod1', 'mod2')

    def test_load_modified_with_load_kwarg(self):
        """Test loading a modified dataset using the ``Scene.load`` keyword argument."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['ds1'], modifiers=('mod1', 'mod2'))
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0]['modifiers'] == ('mod1', 'mod2')

    def test_load_multiple_modified(self):
        """Test loading multiple modified datasets."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load([
            make_dataid(name='ds1', modifiers=('mod1', 'mod2')),
            make_dataid(name='ds2', modifiers=('mod2', 'mod1')),
        ])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 2
        for i in loaded_ids:
            if i['name'] == 'ds1':
                assert i['modifiers'] == ('mod1', 'mod2')
            else:
                assert i['name'] == 'ds2'
                assert i['modifiers'] == ('mod2', 'mod1')

    def test_load_dataset_after_composite(self):
        """Test load composite followed by other datasets."""
        from satpy.readers.yaml_reader import FileYAMLReader
        from satpy.tests.utils import FakeCompositor
        load_mock = spy_decorator(FileYAMLReader.load)
        comp_mock = spy_decorator(FakeCompositor.__call__)
        with mock.patch.object(FileYAMLReader, 'load', load_mock), \
                mock.patch.object(FakeCompositor, '__call__', comp_mock):
            lmock = load_mock.mock
            scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
            scene.load(['comp3'])
            assert lmock.call_count == 1
            scene.load(['ds1'])
            assert lmock.call_count == 2
            scene.load(['ds1'])
            # we should only load from the file twice
            assert lmock.call_count == 2
            # we should only generate the composite once
            assert comp_mock.mock.call_count == 1
            loaded_ids = list(scene._datasets.keys())
            assert len(loaded_ids) == 2

    def test_load_dataset_after_composite2(self):
        """Test load complex composite followed by other datasets."""
        from satpy.readers.yaml_reader import FileYAMLReader
        from satpy.tests.utils import FakeCompositor, FakeModifier
        load_mock = spy_decorator(FileYAMLReader.load)
        comp_mock = spy_decorator(FakeCompositor.__call__)
        mod_mock = spy_decorator(FakeModifier.__call__)
        with mock.patch.object(FileYAMLReader, 'load', load_mock), \
             mock.patch.object(FakeCompositor, '__call__', comp_mock), \
             mock.patch.object(FakeModifier, '__call__', mod_mock):
            lmock = load_mock.mock
            scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
            scene.load(['comp10'])
            assert lmock.call_count == 1
            loaded_ids = list(scene._datasets.keys())
            assert len(loaded_ids) == 1
            with mock.patch.object(scene, '_generate_composites_nodes_from_loaded_datasets',
                                   wraps=scene._generate_composites_nodes_from_loaded_datasets) as m:
                scene.load(['ds1'])
                assert lmock.call_count == 2
                loaded_ids = list(scene._datasets.keys())
                assert len(loaded_ids) == 2
                # this is the unmodified ds1
                assert make_dataid(
                    name='ds1', resolution=250, calibration='reflectance', modifiers=tuple()
                ) in loaded_ids
                # m.assert_called_once_with(set([scene._dependency_tree['ds1']]))
                m.assert_called_once_with(set())
            with mock.patch.object(scene, '_generate_composites_nodes_from_loaded_datasets',
                                   wraps=scene._generate_composites_nodes_from_loaded_datasets) as m:
                scene.load(['ds1'])
                assert lmock.call_count == 2
                loaded_ids = list(scene._datasets.keys())
                assert len(loaded_ids) == 2
                # this is the unmodified ds1
                assert make_dataid(
                    name='ds1', resolution=250, calibration='reflectance', modifiers=tuple()
                ) in loaded_ids
                m.assert_called_once_with(set())
            # we should only generate the comp10 composite once but comp2 was also generated
            assert comp_mock.mock.call_count == 1 + 1
            # Create the modded ds1 at comp10, then load the umodified version
            # again
            assert mod_mock.mock.call_count == 1
            loaded_ids = list(scene._datasets.keys())
            assert len(loaded_ids) == 2

    def test_no_generate_comp10(self):
        """Test generating a composite after loading."""
        # it is fine that an optional prereq doesn't exist
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp10'], generate=False)
        assert any(ds_id['name'] == 'comp10' for ds_id in scene._wishlist)
        assert 'comp10' not in scene._datasets
        # two dependencies should have been loaded
        assert len(scene._datasets) == 2
        assert len(scene.missing_datasets) == 1

        scene._generate_composites_from_loaded_datasets()
        assert any(ds_id['name'] == 'comp10' for ds_id in scene._wishlist)
        assert 'comp10' in scene._datasets
        assert not scene.missing_datasets

    def test_modified_with_wl_dep(self):
        """Test modifying a dataset with a modifier with modified deps.

        More importantly test that loading the modifiers dependency at the
        same time as the original modified dataset that the dependency tree
        nodes are unique and that DataIDs.

        """
        from satpy.dataset.dataid import WavelengthRange

        # Check dependency tree nodes
        # initialize the dep tree without loading the data
        ds1_mod_id = make_dsq(name='ds1', modifiers=('mod_wl',))
        ds3_mod_id = make_dsq(name='ds3', modifiers=('mod_wl',))

        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene._update_dependency_tree({ds1_mod_id, ds3_mod_id}, None)

        ds1_mod_node = scene._dependency_tree[ds1_mod_id]
        ds3_mod_node = scene._dependency_tree[ds3_mod_id]
        ds1_mod_dep_node = ds1_mod_node.data[1][1]
        ds3_mod_dep_node = ds3_mod_node.data[1][1]
        # mod_wl depends on the this node:
        ds6_modded_node = scene._dependency_tree[make_dataid(name='ds6', modifiers=('mod1',))]
        # this dep should be full qualified with name and wavelength
        assert ds6_modded_node.name['name'] is not None
        assert isinstance(ds6_modded_node.name['wavelength'], WavelengthRange)
        # the node should be shared between everything that uses it
        assert ds1_mod_dep_node is ds3_mod_dep_node
        assert ds1_mod_dep_node is ds6_modded_node

        # it is fine that an optional prereq doesn't exist
        scene.load([ds1_mod_id, ds3_mod_id])

        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 2
        assert ds1_mod_id in scene._datasets
        assert ds3_mod_id in scene._datasets

    def test_load_comp11_and_23(self):
        """Test loading two composites that depend on similar wavelengths."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        # mock the available comps/mods in the compositor loader
        avail_comps = scene.available_composite_ids()
        assert make_cid(name='comp11') in avail_comps
        assert make_cid(name='comp23') in avail_comps
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp11', 'comp23'])
        comp11_node = scene._dependency_tree['comp11']
        comp23_node = scene._dependency_tree['comp23']
        assert comp11_node.data[1][-1].name['name'] == 'ds10'
        assert comp23_node.data[1][0].name['name'] == 'ds8'
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 2
        assert 'comp11' in scene
        assert 'comp23' in scene

    def test_load_too_many(self):
        """Test dependency tree if too many reader keys match."""
        scene = Scene(filenames=['fake3_1.txt'], reader='fake3')
        avail_comps = scene.available_composite_ids()
        # static image => 1
        assert len(avail_comps) == 1
        pytest.raises(KeyError, scene.load, [0.21])

    def test_load_when_sensor_none_in_preloaded_dataarrays(self):
        """Test Scene loading when existing loaded arrays have sensor set to None.

        Some readers or composites (ex. static images) don't have a sensor and
        developers choose to set it to `None`. This test makes sure this
        doesn't break loading.

        """
        scene = _scene_with_data_array_none_sensor()
        scene.load(["static_image"])
        assert "static_image" in scene
        assert "my_data" in scene


def _scene_with_data_array_none_sensor():
    scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
    scene['my_data'] = _data_array_none_sensor("my_data")
    return scene


def _data_array_none_sensor(name: str) -> xr.DataArray:
    """Create a DataArray with sensor set to ``None``."""
    return xr.DataArray(
        da.zeros((2, 2)),
        attrs={
            "name": name,
            "sensor": None,
        })
