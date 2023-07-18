#!/usr/bin/python
# Copyright (c) 2015 Satpy developers
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
"""Test generic writer functions."""

from __future__ import annotations

import datetime
import os
import shutil
import unittest
import warnings
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from trollimage.colormap import greys


class TestWritersModule(unittest.TestCase):
    """Test the writers module."""

    def test_to_image_1d(self):
        """Conversion to image."""
        # 1D
        from satpy.writers import to_image
        p = xr.DataArray(np.arange(25), dims=['y'])
        self.assertRaises(ValueError, to_image, p)

    @mock.patch('satpy.writers.XRImage')
    def test_to_image_2d(self, mock_geoimage):
        """Conversion to image."""
        from satpy.writers import to_image

        # 2D
        data = np.arange(25).reshape((5, 5))
        p = xr.DataArray(data, attrs=dict(mode="L", fill_value=0,
                                          palette=[0, 1, 2, 3, 4, 5]),
                         dims=['y', 'x'])
        to_image(p)

        np.testing.assert_array_equal(
            data, mock_geoimage.call_args[0][0].values)
        mock_geoimage.reset_mock()

    @mock.patch('satpy.writers.XRImage')
    def test_to_image_3d(self, mock_geoimage):
        """Conversion to image."""
        # 3D
        from satpy.writers import to_image
        data = np.arange(75).reshape((3, 5, 5))
        p = xr.DataArray(data, dims=['bands', 'y', 'x'])
        p['bands'] = ['R', 'G', 'B']
        to_image(p)
        np.testing.assert_array_equal(data[0], mock_geoimage.call_args[0][0][0])
        np.testing.assert_array_equal(data[1], mock_geoimage.call_args[0][0][1])
        np.testing.assert_array_equal(data[2], mock_geoimage.call_args[0][0][2])

    @mock.patch('satpy.writers.get_enhanced_image')
    def test_show(self, mock_get_image):
        """Check showing."""
        from satpy.writers import show

        data = np.arange(25).reshape((5, 5))
        p = xr.DataArray(data, dims=['y', 'x'])
        show(p)
        self.assertTrue(mock_get_image.return_value.show.called)


class TestEnhancer(unittest.TestCase):
    """Test basic `Enhancer` functionality with builtin configs."""

    def test_basic_init_no_args(self):
        """Test Enhancer init with no arguments passed."""
        from satpy.writers import Enhancer
        e = Enhancer()
        self.assertIsNotNone(e.enhancement_tree)

    def test_basic_init_no_enh(self):
        """Test Enhancer init requesting no enhancements."""
        from satpy.writers import Enhancer
        e = Enhancer(enhancement_config_file=False)
        self.assertIsNone(e.enhancement_tree)

    def test_basic_init_provided_enh(self):
        """Test Enhancer init with string enhancement configs."""
        from satpy.writers import Enhancer
        e = Enhancer(enhancement_config_file=["""enhancements:
  enh1:
    standard_name: toa_bidirectional_reflectance
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.stretch
      kwargs: {stretch: linear}
"""])
        self.assertIsNotNone(e.enhancement_tree)

    def test_init_nonexistent_enh_file(self):
        """Test Enhancer init with a nonexistent enhancement configuration file."""
        from satpy.writers import Enhancer
        self.assertRaises(
            ValueError, Enhancer, enhancement_config_file="is_not_a_valid_filename_?.yaml")


class _BaseCustomEnhancementConfigTests:

    TEST_CONFIGS: dict[str, str] = {}

    @classmethod
    def setup_class(cls):
        """Create fake user configurations."""
        for fn, content in cls.TEST_CONFIGS.items():
            base_dir = os.path.dirname(fn)
            if base_dir:
                os.makedirs(base_dir, exist_ok=True)
            with open(fn, 'w') as f:
                f.write(content)

        # create fake test image writer
        from satpy.writers import ImageWriter

        class CustomImageWriter(ImageWriter):
            def __init__(self, **kwargs):
                super(CustomImageWriter, self).__init__(name='test', config_files=[], **kwargs)
                self.img = None

            def save_image(self, img, **kwargs):
                self.img = img
        cls.CustomImageWriter = CustomImageWriter

    @classmethod
    def teardown_class(cls):
        """Remove fake user configurations."""
        for fn, _content in cls.TEST_CONFIGS.items():
            base_dir = os.path.dirname(fn)
            if base_dir not in ['.', ''] and os.path.isdir(base_dir):
                shutil.rmtree(base_dir)
            elif os.path.isfile(fn):
                os.remove(fn)


class TestComplexSensorEnhancerConfigs(_BaseCustomEnhancementConfigTests):
    """Test enhancement configs that use or expect multiple sensors."""

    ENH_FN = 'test_sensor1.yaml'
    ENH_FN2 = 'test_sensor2.yaml'

    TEST_CONFIGS = {
        ENH_FN: """
enhancements:
  test1_sensor1_specific:
    name: test1
    sensor: test_sensor1
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.stretch
      kwargs: {stretch: crude, min_stretch: 0, max_stretch: 200}

        """,
        ENH_FN2: """
enhancements:
  default:
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.stretch
      kwargs: {stretch: crude, min_stretch: 0, max_stretch: 100}
  test1_sensor2_specific:
    name: test1
    sensor: test_sensor2
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.stretch
      kwargs: {stretch: crude, min_stretch: 0, max_stretch: 50}
  exact_multisensor_comp:
    name: my_comp
    sensor: [test_sensor1, test_sensor2]
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.stretch
      kwargs: {stretch: crude, min_stretch: 0, max_stretch: 20}
            """,
    }

    def test_multisensor_choice(self):
        """Test that a DataArray with two sensors works."""
        from xarray import DataArray

        from satpy.writers import Enhancer, get_enhanced_image
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs={
                           'name': 'test1',
                           'sensor': {'test_sensor2', 'test_sensor1'},
                           'mode': 'L'
                       },
                       dims=['y', 'x'])
        e = Enhancer()
        assert e.enhancement_tree is not None
        img = get_enhanced_image(ds, enhance=e)
        # make sure that both sensor configs were loaded
        assert (set(e.sensor_enhancement_configs) ==
                {os.path.abspath(self.ENH_FN),
                 os.path.abspath(self.ENH_FN2)})
        # test_sensor1 config should have been used because it is
        # alphabetically first
        np.testing.assert_allclose(img.data.values[0], ds.data / 200.0)

    def test_multisensor_exact(self):
        """Test that a DataArray with two sensors can match exactly."""
        from xarray import DataArray

        from satpy.writers import Enhancer, get_enhanced_image
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs={
                           'name': 'my_comp',
                           'sensor': {'test_sensor2', 'test_sensor1'},
                           'mode': 'L'
                       },
                       dims=['y', 'x'])
        e = Enhancer()
        assert e.enhancement_tree is not None
        img = get_enhanced_image(ds, enhance=e)
        # make sure that both sensor configs were loaded
        assert (set(e.sensor_enhancement_configs) ==
                {os.path.abspath(self.ENH_FN),
                 os.path.abspath(self.ENH_FN2)})
        # test_sensor1 config should have been used because it is
        # alphabetically first
        np.testing.assert_allclose(img.data.values[0], ds.data / 20.0)

    def test_enhance_bad_query_value(self):
        """Test Enhancer doesn't fail when query includes bad values."""
        from xarray import DataArray

        from satpy.writers import Enhancer, get_enhanced_image
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs=dict(name=["I", "am", "invalid"], sensor='test_sensor2', mode='L'),
                       dims=['y', 'x'])
        e = Enhancer()
        assert e.enhancement_tree is not None
        with pytest.raises(KeyError, match="No .* found for None"):
            get_enhanced_image(ds, enhance=e)


class TestEnhancerUserConfigs(_BaseCustomEnhancementConfigTests):
    """Test `Enhancer` functionality when user's custom configurations are present."""

    ENH_FN = 'test_sensor.yaml'
    ENH_ENH_FN = os.path.join('enhancements', ENH_FN)
    ENH_FN2 = 'test_sensor2.yaml'
    ENH_ENH_FN2 = os.path.join('enhancements', ENH_FN2)
    ENH_FN3 = 'test_empty.yaml'

    TEST_CONFIGS = {
        ENH_FN: """
enhancements:
  test1_default:
    name: test1
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.stretch
      kwargs: {stretch: linear, cutoffs: [0., 0.]}

        """,
        ENH_ENH_FN: """
enhancements:
  test1_kelvin:
    name: test1
    units: kelvin
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.stretch
      kwargs: {stretch: crude, min_stretch: 0, max_stretch: 20}

        """,
        ENH_FN2: """


        """,
        ENH_ENH_FN2: """

        """,
        ENH_FN3: """""",
    }

    def test_enhance_empty_config(self):
        """Test Enhancer doesn't fail with empty enhancement file."""
        from xarray import DataArray

        from satpy.writers import Enhancer, get_enhanced_image
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs=dict(sensor='test_empty', mode='L'),
                       dims=['y', 'x'])
        e = Enhancer()
        assert e.enhancement_tree is not None
        get_enhanced_image(ds, enhance=e)
        assert (set(e.sensor_enhancement_configs) ==
                {os.path.abspath(self.ENH_FN3)})

    def test_enhance_with_sensor_no_entry(self):
        """Test enhancing an image that has no configuration sections."""
        from xarray import DataArray

        from satpy.writers import Enhancer, get_enhanced_image
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs=dict(sensor='test_sensor2', mode='L'),
                       dims=['y', 'x'])
        e = Enhancer()
        assert e.enhancement_tree is not None
        get_enhanced_image(ds, enhance=e)
        assert (set(e.sensor_enhancement_configs) ==
                {os.path.abspath(self.ENH_FN2),
                 os.path.abspath(self.ENH_ENH_FN2)})

    def test_no_enhance(self):
        """Test turning off enhancements."""
        from xarray import DataArray

        from satpy.writers import get_enhanced_image
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs=dict(name='test1', sensor='test_sensor', mode='L'),
                       dims=['y', 'x'])
        img = get_enhanced_image(ds, enhance=False)
        np.testing.assert_allclose(img.data.data.compute().squeeze(), ds.data)

    def test_writer_no_enhance(self):
        """Test turning off enhancements with writer."""
        from xarray import DataArray
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs=dict(name='test1', sensor='test_sensor', mode='L'),
                       dims=['y', 'x'])
        writer = self.CustomImageWriter(enhance=False)
        writer.save_datasets((ds,), compute=False)
        img = writer.img
        np.testing.assert_allclose(img.data.data.compute().squeeze(), ds.data)

    def test_writer_custom_enhance(self):
        """Test using custom enhancements with writer."""
        from xarray import DataArray

        from satpy.writers import Enhancer
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs=dict(name='test1', sensor='test_sensor', mode='L'),
                       dims=['y', 'x'])
        enhance = Enhancer()
        writer = self.CustomImageWriter(enhance=enhance)
        writer.save_datasets((ds,), compute=False)
        img = writer.img
        np.testing.assert_almost_equal(img.data.isel(bands=0).max().values, 1.)

    def test_enhance_with_sensor_entry(self):
        """Test enhancing an image with a configuration section."""
        from xarray import DataArray

        from satpy.writers import Enhancer, get_enhanced_image
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs=dict(name='test1', sensor='test_sensor', mode='L'),
                       dims=['y', 'x'])
        e = Enhancer()
        assert e.enhancement_tree is not None
        img = get_enhanced_image(ds, enhance=e)
        assert (set(e.sensor_enhancement_configs) ==
                {os.path.abspath(self.ENH_FN),
                 os.path.abspath(self.ENH_ENH_FN)})
        np.testing.assert_almost_equal(img.data.isel(bands=0).max().values,
                                       1.)

        ds = DataArray(da.arange(1, 11., chunks=5).reshape((2, 5)),
                       attrs=dict(name='test1', sensor='test_sensor', mode='L'),
                       dims=['y', 'x'])
        e = Enhancer()
        assert e.enhancement_tree is not None
        img = get_enhanced_image(ds, enhance=e)
        assert (set(e.sensor_enhancement_configs) ==
                {os.path.abspath(self.ENH_FN),
                 os.path.abspath(self.ENH_ENH_FN)})
        np.testing.assert_almost_equal(img.data.isel(bands=0).max().values, 1.)

    def test_enhance_with_sensor_entry2(self):
        """Test enhancing an image with a more detailed configuration section."""
        from xarray import DataArray

        from satpy.writers import Enhancer, get_enhanced_image
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs=dict(name='test1', units='kelvin',
                                  sensor='test_sensor', mode='L'),
                       dims=['y', 'x'])
        e = Enhancer()
        assert e.enhancement_tree is not None
        img = get_enhanced_image(ds, enhance=e)
        assert (set(e.sensor_enhancement_configs) ==
                {os.path.abspath(self.ENH_FN),
                 os.path.abspath(self.ENH_ENH_FN)})
        np.testing.assert_almost_equal(img.data.isel(bands=0).max().values, 0.5)


class TestReaderEnhancerConfigs(_BaseCustomEnhancementConfigTests):
    """Test enhancement configs that use reader name."""

    ENH_FN = 'test_sensor1.yaml'

    # NOTE: The sections are ordered in a special way so that if 'reader' key
    #   isn't provided that we'll get the section we didn't want and all tests
    #   will fail. Otherwise the correct sections get chosen just by the order
    #   of how they are added to the decision tree.
    TEST_CONFIGS = {
        ENH_FN: """
enhancements:
  default_reader2:
    reader: reader2
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.stretch
      kwargs: {stretch: crude, min_stretch: 0, max_stretch: 75}
  default:
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.stretch
      kwargs: {stretch: crude, min_stretch: 0, max_stretch: 100}
  test1_reader2_specific:
    name: test1
    reader: reader2
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.stretch
      kwargs: {stretch: crude, min_stretch: 0, max_stretch: 50}
  test1_reader1_specific:
    name: test1
    reader: reader1
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.stretch
      kwargs: {stretch: crude, min_stretch: 0, max_stretch: 200}
            """,
    }

    def _get_test_data_array(self):
        from xarray import DataArray
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs={
                           'name': 'test1',
                           'sensor': 'test_sensor1',
                           'mode': 'L',
                       },
                       dims=['y', 'x'])
        return ds

    def _get_enhanced_image(self, data_arr):
        from satpy.writers import Enhancer, get_enhanced_image
        e = Enhancer()
        assert e.enhancement_tree is not None
        img = get_enhanced_image(data_arr, enhance=e)
        # make sure that both configs were loaded
        assert (set(e.sensor_enhancement_configs) ==
                {os.path.abspath(self.ENH_FN)})
        return img

    def test_no_reader(self):
        """Test that a DataArray with no 'reader' metadata works."""
        data_arr = self._get_test_data_array()
        img = self._get_enhanced_image(data_arr)
        # no reader available, should use default no specified reader
        np.testing.assert_allclose(img.data.values[0], data_arr.data / 100.0)

    def test_no_matching_reader(self):
        """Test that a DataArray with no matching 'reader' works."""
        data_arr = self._get_test_data_array()
        data_arr.attrs["reader"] = "reader3"
        img = self._get_enhanced_image(data_arr)
        # no reader available, should use default no specified reader
        np.testing.assert_allclose(img.data.values[0], data_arr.data / 100.0)

    def test_only_reader_matches(self):
        """Test that a DataArray with only a matching 'reader' works."""
        data_arr = self._get_test_data_array()
        data_arr.attrs["reader"] = "reader2"
        data_arr.attrs["name"] = "not_configured"
        img = self._get_enhanced_image(data_arr)
        # no reader available, should use default no specified reader
        np.testing.assert_allclose(img.data.values[0], data_arr.data / 75.0)

    def test_reader_and_name_match(self):
        """Test that a DataArray with a matching 'reader' and 'name' works."""
        data_arr = self._get_test_data_array()
        data_arr.attrs["reader"] = "reader2"
        img = self._get_enhanced_image(data_arr)
        # no reader available, should use default no specified reader
        np.testing.assert_allclose(img.data.values[0], data_arr.data / 50.0)


class TestYAMLFiles(unittest.TestCase):
    """Test and analyze the writer configuration files."""

    def test_filename_matches_writer_name(self):
        """Test that every writer filename matches the name in the YAML."""
        import yaml

        class IgnoreLoader(yaml.SafeLoader):

            def _ignore_all_tags(self, tag_suffix, node):
                return tag_suffix + ' ' + node.value
        IgnoreLoader.add_multi_constructor('', IgnoreLoader._ignore_all_tags)

        from satpy._config import glob_config
        from satpy.writers import read_writer_config
        for writer_config in glob_config('writers/*.yaml'):
            writer_fn = os.path.basename(writer_config)
            writer_fn_name = os.path.splitext(writer_fn)[0]
            writer_info = read_writer_config([writer_config],
                                             loader=IgnoreLoader)
            self.assertEqual(writer_fn_name, writer_info['name'],
                             "Writer YAML filename doesn't match writer "
                             "name in the YAML file.")

    def test_available_writers(self):
        """Test the 'available_writers' function."""
        from satpy import available_writers
        writer_names = available_writers()
        self.assertGreater(len(writer_names), 0)
        self.assertIsInstance(writer_names[0], str)
        self.assertIn('geotiff', writer_names)

        writer_infos = available_writers(as_dict=True)
        self.assertEqual(len(writer_names), len(writer_infos))
        self.assertIsInstance(writer_infos[0], dict)
        for writer_info in writer_infos:
            self.assertIn('name', writer_info)


class TestComputeWriterResults(unittest.TestCase):
    """Test compute_writer_results()."""

    def setUp(self):
        """Create temporary directory to save files to and a mock scene."""
        import tempfile
        from datetime import datetime

        from satpy.scene import Scene

        ds1 = xr.DataArray(
            da.zeros((100, 200), chunks=50),
            dims=('y', 'x'),
            attrs={'name': 'test',
                   'start_time': datetime(2018, 1, 1, 0, 0, 0)}
        )
        self.scn = Scene()
        self.scn['test'] = ds1

        # Temp dir
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary directory created for a test."""
        try:
            shutil.rmtree(self.base_dir, ignore_errors=True)
        except OSError:
            pass

    def test_empty(self):
        """Test empty result list."""
        from satpy.writers import compute_writer_results
        compute_writer_results([])

    def test_simple_image(self):
        """Test writing to PNG file."""
        from satpy.writers import compute_writer_results
        fname = os.path.join(self.base_dir, 'simple_image.png')
        res = self.scn.save_datasets(filename=fname,
                                     datasets=['test'],
                                     writer='simple_image',
                                     compute=False)
        compute_writer_results([res])
        self.assertTrue(os.path.isfile(fname))

    def test_geotiff(self):
        """Test writing to mitiff file."""
        from satpy.writers import compute_writer_results
        fname = os.path.join(self.base_dir, 'geotiff.tif')
        res = self.scn.save_datasets(filename=fname,
                                     datasets=['test'],
                                     writer='geotiff', compute=False)
        compute_writer_results([res])
        self.assertTrue(os.path.isfile(fname))

# FIXME: This reader needs more information than exist at the moment
#    def test_mitiff(self):
#        """Test writing to mitiff file"""
#        fname = os.path.join(self.base_dir, 'mitiff.tif')
#        res = self.scn.save_datasets(filename=fname,
#                                     datasets=['test'],
#                                     writer='mitiff')
#        compute_writer_results([res])
#        self.assertTrue(os.path.isfile(fname))

# FIXME: This reader needs more information than exist at the moment
#    def test_cf(self):
#        """Test writing to NetCDF4 file"""
#        fname = os.path.join(self.base_dir, 'cf.nc')
#        res = self.scn.save_datasets(filename=fname,
#                                     datasets=['test'],
#                                     writer='cf')
#        compute_writer_results([res])
#        self.assertTrue(os.path.isfile(fname))

    def test_multiple_geotiff(self):
        """Test writing to mitiff file."""
        from satpy.writers import compute_writer_results
        fname1 = os.path.join(self.base_dir, 'geotiff1.tif')
        res1 = self.scn.save_datasets(filename=fname1,
                                      datasets=['test'],
                                      writer='geotiff', compute=False)
        fname2 = os.path.join(self.base_dir, 'geotiff2.tif')
        res2 = self.scn.save_datasets(filename=fname2,
                                      datasets=['test'],
                                      writer='geotiff', compute=False)
        compute_writer_results([res1, res2])
        self.assertTrue(os.path.isfile(fname1))
        self.assertTrue(os.path.isfile(fname2))

    def test_multiple_simple(self):
        """Test writing to geotiff files."""
        from satpy.writers import compute_writer_results
        fname1 = os.path.join(self.base_dir, 'simple_image1.png')
        res1 = self.scn.save_datasets(filename=fname1,
                                      datasets=['test'],
                                      writer='simple_image', compute=False)
        fname2 = os.path.join(self.base_dir, 'simple_image2.png')
        res2 = self.scn.save_datasets(filename=fname2,
                                      datasets=['test'],
                                      writer='simple_image', compute=False)
        compute_writer_results([res1, res2])
        self.assertTrue(os.path.isfile(fname1))
        self.assertTrue(os.path.isfile(fname2))

    def test_mixed(self):
        """Test writing to multiple mixed-type files."""
        from satpy.writers import compute_writer_results
        fname1 = os.path.join(self.base_dir, 'simple_image3.png')
        res1 = self.scn.save_datasets(filename=fname1,
                                      datasets=['test'],
                                      writer='simple_image', compute=False)
        fname2 = os.path.join(self.base_dir, 'geotiff3.tif')
        res2 = self.scn.save_datasets(filename=fname2,
                                      datasets=['test'],
                                      writer='geotiff', compute=False)
        res3 = []
        compute_writer_results([res1, res2, res3])
        self.assertTrue(os.path.isfile(fname1))
        self.assertTrue(os.path.isfile(fname2))


class TestBaseWriter:
    """Test the base writer class."""

    def setup_method(self):
        """Set up tests."""
        import tempfile
        from datetime import datetime

        from satpy.scene import Scene

        ds1 = xr.DataArray(
            da.zeros((100, 200), chunks=50),
            dims=('y', 'x'),
            attrs={
                'name': 'test',
                'start_time': datetime(2018, 1, 1, 0, 0, 0),
                'sensor': 'fake_sensor',
            }
        )
        ds2 = ds1.copy()
        ds2.attrs['sensor'] = {'fake_sensor1', 'fake_sensor2'}
        self.scn = Scene()
        self.scn['test'] = ds1
        self.scn['test2'] = ds2

        # Temp dir
        self.base_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Remove the temporary directory created for a test."""
        try:
            shutil.rmtree(self.base_dir, ignore_errors=True)
        except OSError:
            pass

    def test_save_dataset_static_filename(self):
        """Test saving a dataset with a static filename specified."""
        self.scn.save_datasets(base_dir=self.base_dir, filename='geotiff.tif')
        assert os.path.isfile(os.path.join(self.base_dir, 'geotiff.tif'))

    @pytest.mark.parametrize(
        ('fmt_fn', 'exp_fns'),
        [
            ('geotiff_{name}_{start_time:%Y%m%d_%H%M%S}.tif',
             ['geotiff_test_20180101_000000.tif', 'geotiff_test2_20180101_000000.tif']),
            ('geotiff_{name}_{sensor}.tif',
             ['geotiff_test_fake_sensor.tif', 'geotiff_test2_fake_sensor1-fake_sensor2.tif']),
        ]
    )
    def test_save_dataset_dynamic_filename(self, fmt_fn, exp_fns):
        """Test saving a dataset with a format filename specified."""
        self.scn.save_datasets(base_dir=self.base_dir, filename=fmt_fn)
        for exp_fn in exp_fns:
            exp_path = os.path.join(self.base_dir, exp_fn)
            assert os.path.isfile(exp_path)

    def test_save_dataset_dynamic_filename_with_dir(self):
        """Test saving a dataset with a format filename that includes a directory."""
        fmt_fn = os.path.join('{start_time:%Y%m%d}', 'geotiff_{name}_{start_time:%Y%m%d_%H%M%S}.tif')
        exp_fn = os.path.join('20180101', 'geotiff_test_20180101_000000.tif')
        self.scn.save_datasets(base_dir=self.base_dir, filename=fmt_fn)
        assert os.path.isfile(os.path.join(self.base_dir, exp_fn))

        # change the filename pattern but keep the same directory
        fmt_fn2 = os.path.join('{start_time:%Y%m%d}', 'geotiff_{name}_{start_time:%Y%m%d_%H}.tif')
        exp_fn2 = os.path.join('20180101', 'geotiff_test_20180101_00.tif')
        self.scn.save_datasets(base_dir=self.base_dir, filename=fmt_fn2)
        assert os.path.isfile(os.path.join(self.base_dir, exp_fn2))
        # the original file should still exist
        assert os.path.isfile(os.path.join(self.base_dir, exp_fn))


class TestOverlays(unittest.TestCase):
    """Tests for add_overlay and add_decorate functions."""

    def setUp(self):
        """Create test data and mock pycoast/pydecorate."""
        from pyresample.geometry import AreaDefinition
        from trollimage.xrimage import XRImage

        proj_dict = {'proj': 'lcc', 'datum': 'WGS84', 'ellps': 'WGS84',
                     'lon_0': -95., 'lat_0': 25, 'lat_1': 25,
                     'units': 'm', 'no_defs': True}
        self.area_def = AreaDefinition(
            'test', 'test', 'test', proj_dict,
            200, 400, (-1000., -1500., 1000., 1500.),
        )
        self.orig_rgb_img = XRImage(
            xr.DataArray(da.arange(75., chunks=10).reshape(3, 5, 5) / 75.,
                         dims=('bands', 'y', 'x'),
                         coords={'bands': ['R', 'G', 'B']},
                         attrs={'name': 'test_ds', 'area': self.area_def})
        )
        self.orig_l_img = XRImage(
            xr.DataArray(da.arange(25., chunks=10).reshape(5, 5) / 75.,
                         dims=('y', 'x'),
                         attrs={'name': 'test_ds', 'area': self.area_def})
        )

        self.decorate = {
            'decorate': [
                {'logo': {'logo_path': '', 'height': 143, 'bg': 'white', 'bg_opacity': 255}},
                {'text': {
                    'txt': 'TEST',
                    'align': {'top_bottom': 'bottom', 'left_right': 'right'},
                    'font': '',
                    'font_size': 22,
                    'height': 30,
                    'bg': 'black',
                    'bg_opacity': 255,
                    'line': 'white'}},
                {'scale': {
                    'colormap': greys,
                    'extend': False,
                    'width': 1670, 'height': 110,
                    'tick_marks': 5, 'minor_tick_marks': 1,
                    'cursor': [0, 0], 'bg':'white',
                    'title':'TEST TITLE OF SCALE',
                    'fontsize': 110, 'align': 'cc'
                }}
            ]
        }

        import_mock = mock.MagicMock()
        modules = {'pycoast': import_mock.pycoast,
                   'pydecorate': import_mock.pydecorate}
        self.module_patcher = mock.patch.dict('sys.modules', modules)
        self.module_patcher.start()

    def tearDown(self):
        """Turn off pycoast/pydecorate mocking."""
        self.module_patcher.stop()

    def test_add_overlay_basic_rgb(self):
        """Test basic add_overlay usage with RGB data."""
        from pycoast import ContourWriterAGG

        from satpy.writers import _burn_overlay, add_overlay
        coast_dir = '/path/to/coast/data'
        with mock.patch.object(self.orig_rgb_img, "apply_pil") as apply_pil:
            apply_pil.return_value = self.orig_rgb_img
            new_img = add_overlay(self.orig_rgb_img, self.area_def, coast_dir, fill_value=0)
            self.assertEqual(self.orig_rgb_img.mode, new_img.mode)
            new_img = add_overlay(self.orig_rgb_img, self.area_def, coast_dir)
            self.assertEqual(self.orig_rgb_img.mode + 'A', new_img.mode)

            with mock.patch.object(self.orig_rgb_img, "convert") as convert:
                convert.return_value = self.orig_rgb_img
                overlays = {'coasts': {'outline': 'red'}}
                new_img = add_overlay(self.orig_rgb_img, self.area_def, coast_dir,
                                      overlays=overlays, fill_value=0)
                pil_args = None
                pil_kwargs = {'fill_value': 0}
                fun_args = (self.orig_rgb_img.data.area, ContourWriterAGG.return_value, overlays)
                fun_kwargs = None
                apply_pil.assert_called_with(_burn_overlay, self.orig_rgb_img.mode,
                                             pil_args, pil_kwargs, fun_args, fun_kwargs)
                ContourWriterAGG.assert_called_with(coast_dir)

                # test legacy call

                grid = {'minor_is_tick': True}
                color = 'red'
                expected_overlays = {'coasts': {'outline': color, 'width': 0.5, 'level': 1},
                                     'borders': {'outline': color, 'width': 0.5, 'level': 1},
                                     'grid': grid}
                with warnings.catch_warnings(record=True) as wns:
                    warnings.simplefilter("always")
                    new_img = add_overlay(self.orig_rgb_img, self.area_def, coast_dir,
                                          color=color, grid=grid, fill_value=0)
                    assert len(wns) == 1
                    assert issubclass(wns[0].category, DeprecationWarning)
                    assert "deprecated" in str(wns[0].message)

                pil_args = None
                pil_kwargs = {'fill_value': 0}
                fun_args = (self.orig_rgb_img.data.area, ContourWriterAGG.return_value, expected_overlays)
                fun_kwargs = None
                apply_pil.assert_called_with(_burn_overlay, self.orig_rgb_img.mode,
                                             pil_args, pil_kwargs, fun_args, fun_kwargs)
                ContourWriterAGG.assert_called_with(coast_dir)

    def test_add_overlay_basic_l(self):
        """Test basic add_overlay usage with L data."""
        from satpy.writers import add_overlay
        new_img = add_overlay(self.orig_l_img, self.area_def, '', fill_value=0)
        self.assertEqual('RGB', new_img.mode)
        new_img = add_overlay(self.orig_l_img, self.area_def, '')
        self.assertEqual('RGBA', new_img.mode)

    def test_add_decorate_basic_rgb(self):
        """Test basic add_decorate usage with RGB data."""
        from satpy.writers import add_decorate
        new_img = add_decorate(self.orig_rgb_img, **self.decorate)
        self.assertEqual('RGBA', new_img.mode)

    def test_add_decorate_basic_l(self):
        """Test basic add_decorate usage with L data."""
        from satpy.writers import add_decorate
        new_img = add_decorate(self.orig_l_img, **self.decorate)
        self.assertEqual('RGBA', new_img.mode)


def test_group_results_by_output_file(tmp_path):
    """Test grouping results by output file.

    Add a test for grouping the results from save_datasets(..., compute=False)
    by output file.  This is useful if for some reason we want to treat each
    output file as a seperate computation (that can still be computed together
    later).
    """
    from pyresample import create_area_def

    from satpy.writers import group_results_by_output_file

    from .utils import make_fake_scene
    x = 10
    fake_area = create_area_def("sargasso", 4326, resolution=1, width=x, height=x, center=(0, 0))
    fake_scene = make_fake_scene(
        {"dragon_top_height": (dat := xr.DataArray(
            dims=("y", "x"),
            data=da.arange(x*x).reshape((x, x)))),
         "penguin_bottom_height": dat,
         "kraken_depth": dat},
        daskify=True,
        area=fake_area,
        common_attrs={"start_time": datetime.datetime(2022, 11, 16, 13, 27)})
    # NB: even if compute=False, ``save_datasets`` creates (empty) files
    (sources, targets) = fake_scene.save_datasets(
            filename=os.fspath(tmp_path / "test-{name}.tif"),
            writer="ninjogeotiff",
            compress="NONE",
            fill_value=0,
            compute=False,
            ChannelID="x",
            DataType="x",
            PhysicUnit="K",
            PhysicValue="Temperature",
            SatelliteNameID="x")

    grouped = group_results_by_output_file(sources, targets)

    assert len(grouped) == 3
    assert len({x.rfile.path for x in grouped[0][1]}) == 1
    for x in grouped:
        assert len(x[0]) == len(x[1])
    assert sources[:5] == grouped[0][0]
    assert targets[:5] == grouped[0][1]
    assert sources[10:] == grouped[2][0]
    assert targets[10:] == grouped[2][1]
