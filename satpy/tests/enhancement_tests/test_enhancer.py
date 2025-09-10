# Copyright (c) 2025 Satpy developers
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
"""Test enhancer class."""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest
import xarray as xr
from dask import array as da
from trollimage.xrimage import XRImage

from satpy.enhancements.enhancer import Enhancer
from satpy.writers.core.image import ImageWriter


class TestEnhancer:
    """Test basic `Enhancer` functionality with builtin configs."""

    def test_basic_init_no_args(self):
        """Test Enhancer init with no arguments passed."""
        e = Enhancer()
        assert e.enhancement_tree is not None

    def test_basic_init_no_enh(self):
        """Test Enhancer init requesting no enhancements."""
        e = Enhancer(enhancement_config_file=False)
        assert e.enhancement_tree is None

    def test_basic_init_provided_enh(self):
        """Test Enhancer init with string enhancement configs."""
        e = Enhancer(enhancement_config_file=["""enhancements:
  enh1:
    standard_name: toa_bidirectional_reflectance
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.contrast.stretch
      kwargs: {stretch: linear}
"""])
        assert e.enhancement_tree is not None

    def test_init_nonexistent_enh_file(self):
        """Test Enhancer init with a nonexistent enhancement configuration file."""
        with pytest.raises(ValueError, match="YAML file doesn't exist or string is not YAML dict:.*"):
            Enhancer(enhancement_config_file="is_not_a_valid_filename_?.yaml")

    def test_print_tree(self, capsys):
        """Test enhancement decision tree printing."""
        enh = Enhancer()
        enh.enhancement_tree.print_tree()
        stdout = capsys.readouterr().out
        lines = stdout.splitlines()
        assert lines[0].startswith("name=<wildcard>")
        # make sure lines are indented
        assert lines[1].startswith("  reader=")
        assert lines[2].startswith("    platform_name=")


def test_xrimage_1d():
    """Conversion to image."""
    p = xr.DataArray(np.arange(25), dims=["y"])
    with pytest.raises(ValueError, match="Data must have a 'y' and 'x' dimension"):
        XRImage(p)


def test_xrimage_2d():
    """Conversion to image."""
    data = np.arange(25).reshape((5, 5))
    p = xr.DataArray(data, attrs=dict(mode="L", fill_value=0,
                                      palette=[0, 1, 2, 3, 4, 5]),
                     dims=["y", "x"])
    XRImage(p)


def test_xrimage_3d():
    """Conversion to image."""
    data = np.arange(75).reshape((3, 5, 5))
    p = xr.DataArray(data, dims=["bands", "y", "x"])
    p["bands"] = ["R", "G", "B"]
    XRImage(p)


class _CustomImageWriter(ImageWriter):
    def __init__(self, **kwargs):
        super().__init__(name="test", config_files=[], **kwargs)
        self.img = None

    def save_image(self, img, **kwargs):
        self.img = img


class _BaseCustomEnhancementConfigTests:

    TEST_CONFIGS: dict[str, str] = {}

    @pytest.fixture(scope="class", autouse=True)
    def test_configs_path(self, tmp_path_factory):
        """Create test enhancement configuration files in a temporary directory.

        The root temporary directory is changed to and returned.

        """
        prev_cwd = pathlib.Path.cwd()
        tmp_path = tmp_path_factory.mktemp("config")
        os.chdir(tmp_path)

        for fn, content in self.TEST_CONFIGS.items():
            config_rel_dir = os.path.dirname(fn)
            if config_rel_dir:
                os.makedirs(config_rel_dir, exist_ok=True)
            with open(fn, "w") as f:
                f.write(content)

        try:
            yield tmp_path
        finally:
            os.chdir(prev_cwd)


class TestComplexSensorEnhancerConfigs(_BaseCustomEnhancementConfigTests):
    """Test enhancement configs that use or expect multiple sensors."""

    ENH_FN = "test_sensor1.yaml"
    ENH_FN2 = "test_sensor2.yaml"

    TEST_CONFIGS = {
        ENH_FN: """
enhancements:
  test1_sensor1_specific:
    name: test1
    sensor: test_sensor1
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.contrast.stretch
      kwargs: {stretch: crude, min_stretch: 0, max_stretch: 200}

        """,
        ENH_FN2: """
enhancements:
  default:
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.contrast.stretch
      kwargs: {stretch: crude, min_stretch: 0, max_stretch: 100}
  test1_sensor2_specific:
    name: test1
    sensor: test_sensor2
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.contrast.stretch
      kwargs: {stretch: crude, min_stretch: 0, max_stretch: 50}
  exact_multisensor_comp:
    name: my_comp
    sensor: [test_sensor1, test_sensor2]
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.contrast.stretch
      kwargs: {stretch: crude, min_stretch: 0, max_stretch: 20}
            """,
    }

    def test_multisensor_choice(self, test_configs_path):
        """Test that a DataArray with two sensors works."""
        from xarray import DataArray

        from satpy.enhancements.enhancer import Enhancer, get_enhanced_image
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs={
                           "name": "test1",
                           "sensor": {"test_sensor2", "test_sensor1"},
                           "mode": "L"
                       },
                       dims=["y", "x"])
        e = Enhancer()
        assert e.enhancement_tree is not None
        img = get_enhanced_image(ds, enhance=e)
        # make sure that both sensor configs were loaded
        assert (set(pathlib.Path(config) for config in e.sensor_enhancement_configs) ==
                {test_configs_path / self.ENH_FN,
                 test_configs_path / self.ENH_FN2})
        # test_sensor1 config should have been used because it is
        # alphabetically first
        np.testing.assert_allclose(img.data.values[0], ds.data / 200.0)

    def test_multisensor_exact(self, test_configs_path):
        """Test that a DataArray with two sensors can match exactly."""
        from xarray import DataArray

        from satpy.enhancements.enhancer import Enhancer, get_enhanced_image
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs={
                           "name": "my_comp",
                           "sensor": {"test_sensor2", "test_sensor1"},
                           "mode": "L"
                       },
                       dims=["y", "x"])
        e = Enhancer()
        assert e.enhancement_tree is not None
        img = get_enhanced_image(ds, enhance=e)
        # make sure that both sensor configs were loaded
        assert (set(pathlib.Path(config) for config in e.sensor_enhancement_configs) ==
                {test_configs_path / self.ENH_FN,
                 test_configs_path / self.ENH_FN2})
        # test_sensor1 config should have been used because it is
        # alphabetically first
        np.testing.assert_allclose(img.data.values[0], ds.data / 20.0)

    def test_enhance_bad_query_value(self):
        """Test Enhancer doesn't fail when query includes bad values."""
        from xarray import DataArray

        from satpy.enhancements.enhancer import Enhancer, get_enhanced_image
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs=dict(name=["I", "am", "invalid"], sensor="test_sensor2", mode="L"),
                       dims=["y", "x"])
        e = Enhancer()
        assert e.enhancement_tree is not None
        with pytest.raises(KeyError, match="No .* found for None"):
            get_enhanced_image(ds, enhance=e)


class TestEnhancerUserConfigs(_BaseCustomEnhancementConfigTests):
    """Test `Enhancer` functionality when user's custom configurations are present."""

    ENH_FN = "test_sensor.yaml"
    ENH_ENH_FN = os.path.join("enhancements", ENH_FN)
    ENH_FN2 = "test_sensor2.yaml"
    ENH_ENH_FN2 = os.path.join("enhancements", ENH_FN2)
    ENH_FN3 = "test_empty.yaml"

    TEST_CONFIGS = {
        ENH_FN: """
enhancements:
  test1_default:
    name: test1
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.contrast.stretch
      kwargs: {stretch: linear, cutoffs: [0., 0.]}

        """,
        ENH_ENH_FN: """
enhancements:
  test1_kelvin:
    name: test1
    units: kelvin
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.contrast.stretch
      kwargs: {stretch: crude, min_stretch: 0, max_stretch: 20}

        """,
        ENH_FN2: """


        """,
        ENH_ENH_FN2: """

        """,
        ENH_FN3: """""",
    }

    def test_enhance_empty_config(self, test_configs_path):
        """Test Enhancer doesn't fail with empty enhancement file."""
        from xarray import DataArray

        from satpy.enhancements.enhancer import Enhancer, get_enhanced_image
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs=dict(sensor="test_empty", mode="L"),
                       dims=["y", "x"])
        e = Enhancer()
        assert e.enhancement_tree is not None
        get_enhanced_image(ds, enhance=e)
        assert (set(pathlib.Path(config) for config in e.sensor_enhancement_configs) ==
                {test_configs_path / self.ENH_FN3})

    def test_enhance_with_sensor_no_entry(self, test_configs_path):
        """Test enhancing an image that has no configuration sections."""
        from xarray import DataArray

        from satpy.enhancements.enhancer import Enhancer, get_enhanced_image
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs=dict(sensor="test_sensor2", mode="L"),
                       dims=["y", "x"])
        e = Enhancer()
        assert e.enhancement_tree is not None
        get_enhanced_image(ds, enhance=e)
        assert (set(pathlib.Path(config) for config in e.sensor_enhancement_configs) ==
                {test_configs_path / self.ENH_FN2,
                 test_configs_path / self.ENH_ENH_FN2})

    def test_no_enhance(self):
        """Test turning off enhancements."""
        from xarray import DataArray

        from satpy.enhancements.enhancer import get_enhanced_image
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs=dict(name="test1", sensor="test_sensor", mode="L"),
                       dims=["y", "x"])
        img = get_enhanced_image(ds, enhance=False)
        np.testing.assert_allclose(img.data.data.compute().squeeze(), ds.data)

    def test_writer_no_enhance(self):
        """Test turning off enhancements with writer."""
        from xarray import DataArray
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs=dict(name="test1", sensor="test_sensor", mode="L"),
                       dims=["y", "x"])
        writer = _CustomImageWriter(enhance=False)
        writer.save_datasets((ds,), compute=False)
        img = writer.img
        np.testing.assert_allclose(img.data.data.compute().squeeze(), ds.data)

    def test_writer_custom_enhance(self):
        """Test using custom enhancements with writer."""
        from xarray import DataArray

        from satpy.enhancements.enhancer import Enhancer
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs=dict(name="test1", sensor="test_sensor", mode="L"),
                       dims=["y", "x"])
        enhance = Enhancer()
        writer = _CustomImageWriter(enhance=enhance)
        writer.save_datasets((ds,), compute=False)
        img = writer.img
        np.testing.assert_almost_equal(img.data.isel(bands=0).max().values, 1.)

    def test_enhance_with_sensor_entry(self, test_configs_path):
        """Test enhancing an image with a configuration section."""
        from xarray import DataArray

        from satpy.enhancements.enhancer import Enhancer, get_enhanced_image
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs=dict(name="test1", sensor="test_sensor", mode="L"),
                       dims=["y", "x"])
        e = Enhancer()
        assert e.enhancement_tree is not None
        img = get_enhanced_image(ds, enhance=e)
        assert (set(pathlib.Path(config) for config in e.sensor_enhancement_configs) ==
                {test_configs_path / self.ENH_FN,
                 test_configs_path / self.ENH_ENH_FN})
        np.testing.assert_almost_equal(img.data.isel(bands=0).max().values,
                                       1.)

        ds = DataArray(da.arange(1, 11., chunks=5).reshape((2, 5)),
                       attrs=dict(name="test1", sensor="test_sensor", mode="L"),
                       dims=["y", "x"])
        e = Enhancer()
        assert e.enhancement_tree is not None
        img = get_enhanced_image(ds, enhance=e)
        assert (set(pathlib.Path(config) for config in e.sensor_enhancement_configs) ==
                {test_configs_path / self.ENH_FN,
                 test_configs_path / self.ENH_ENH_FN})
        np.testing.assert_almost_equal(img.data.isel(bands=0).max().values, 1.)

    def test_enhance_with_sensor_entry2(self, test_configs_path):
        """Test enhancing an image with a more detailed configuration section."""
        from xarray import DataArray

        from satpy.enhancements.enhancer import Enhancer, get_enhanced_image
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs=dict(name="test1", units="kelvin",
                                  sensor="test_sensor", mode="L"),
                       dims=["y", "x"])
        e = Enhancer()
        assert e.enhancement_tree is not None
        img = get_enhanced_image(ds, enhance=e)
        assert (set(pathlib.Path(config) for config in e.sensor_enhancement_configs) ==
                {test_configs_path / self.ENH_FN,
                 test_configs_path / self.ENH_ENH_FN})
        np.testing.assert_almost_equal(img.data.isel(bands=0).max().values, 0.5)


class TestReaderEnhancerConfigs(_BaseCustomEnhancementConfigTests):
    """Test enhancement configs that use reader name."""

    ENH_FN = "test_sensor1.yaml"

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
      method: !!python/name:satpy.enhancements.contrast.stretch
      kwargs: {stretch: crude, min_stretch: 0, max_stretch: 75}
  default:
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.contrast.stretch
      kwargs: {stretch: crude, min_stretch: 0, max_stretch: 100}
  test1_reader2_specific:
    name: test1
    reader: reader2
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.contrast.stretch
      kwargs: {stretch: crude, min_stretch: 0, max_stretch: 50}
  test1_reader1_specific:
    name: test1
    reader: reader1
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.contrast.stretch
      kwargs: {stretch: crude, min_stretch: 0, max_stretch: 200}
            """,
    }

    def _get_test_data_array(self):
        from xarray import DataArray
        ds = DataArray(np.arange(1, 11.).reshape((2, 5)),
                       attrs={
                           "name": "test1",
                           "sensor": "test_sensor1",
                           "mode": "L",
                       },
                       dims=["y", "x"])
        return ds

    def _get_enhanced_image(self, data_arr, test_configs_path):
        from satpy.enhancements.enhancer import Enhancer, get_enhanced_image
        e = Enhancer()
        assert e.enhancement_tree is not None
        img = get_enhanced_image(data_arr, enhance=e)
        # make sure that both configs were loaded
        assert (set(pathlib.Path(config) for config in e.sensor_enhancement_configs) ==
                {test_configs_path / self.ENH_FN})
        return img

    def test_no_reader(self, test_configs_path):
        """Test that a DataArray with no 'reader' metadata works."""
        data_arr = self._get_test_data_array()
        img = self._get_enhanced_image(data_arr, test_configs_path)
        # no reader available, should use default no specified reader
        np.testing.assert_allclose(img.data.values[0], data_arr.data / 100.0)

    def test_no_matching_reader(self, test_configs_path):
        """Test that a DataArray with no matching 'reader' works."""
        data_arr = self._get_test_data_array()
        data_arr.attrs["reader"] = "reader3"
        img = self._get_enhanced_image(data_arr, test_configs_path)
        # no reader available, should use default no specified reader
        np.testing.assert_allclose(img.data.values[0], data_arr.data / 100.0)

    def test_only_reader_matches(self, test_configs_path):
        """Test that a DataArray with only a matching 'reader' works."""
        data_arr = self._get_test_data_array()
        data_arr.attrs["reader"] = "reader2"
        data_arr.attrs["name"] = "not_configured"
        img = self._get_enhanced_image(data_arr, test_configs_path)
        # no reader available, should use default no specified reader
        np.testing.assert_allclose(img.data.values[0], data_arr.data / 75.0)

    def test_reader_and_name_match(self, test_configs_path):
        """Test that a DataArray with a matching 'reader' and 'name' works."""
        data_arr = self._get_test_data_array()
        data_arr.attrs["reader"] = "reader2"
        img = self._get_enhanced_image(data_arr, test_configs_path)
        # no reader available, should use default no specified reader
        np.testing.assert_allclose(img.data.values[0], data_arr.data / 50.0)
