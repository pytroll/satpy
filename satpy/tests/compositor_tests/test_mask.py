#!/usr/bin/env python
# Copyright (c) 2018-2025 Satpy developers
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

"""Tests for compositors creating masks."""

import unittest
from unittest import mock

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from satpy.tests.utils import CustomScheduler


class TestHighCloudCompositor:
    """Test HighCloudCompositor."""

    def setup_method(self):
        """Create test data."""
        from pyresample.geometry import create_area_def
        area = create_area_def(area_id="test", projection={"proj": "latlong"},
                               center=(0, 45), width=3, height=3, resolution=35)
        self.dtype = np.float32
        self.data = xr.DataArray(
            da.from_array(np.array([[200, 250, 300], [200, 250, 300], [200, 250, 300]], dtype=self.dtype)),
            dims=("y", "x"), coords={"y": [0, 1, 2], "x": [0, 1, 2]},
            attrs={"area": area}
        )

    def test_high_cloud_compositor(self):
        """Test general default functionality of compositor."""
        from satpy.composites.mask import HighCloudCompositor
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = HighCloudCompositor(name="test")
            res = comp([self.data])
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        expexted_alpha = np.array([[1.0, 0.7142857, 0.0], [1.0, 0.625, 0.0], [1.0, 0.5555555, 0.0]])
        expected = np.stack([self.data, expexted_alpha])
        np.testing.assert_almost_equal(res.values, expected)

    def test_high_cloud_compositor_multiple_calls(self):
        """Test that the modified init variables are reset properly when calling the compositor multiple times."""
        from satpy.composites.mask import HighCloudCompositor
        comp = HighCloudCompositor(name="test")
        res = comp([self.data])
        res2 = comp([self.data])
        np.testing.assert_equal(res.values, res2.values)

    def test_high_cloud_compositor_dtype(self):
        """Test that the datatype is not altered by the compositor."""
        from satpy.composites.mask import HighCloudCompositor
        comp = HighCloudCompositor(name="test")
        res = comp([self.data])
        assert res.data.dtype == self.dtype

    def test_high_cloud_compositor_validity_checks(self):
        """Test that errors are raised for invalid input data and settings."""
        from satpy.composites.mask import HighCloudCompositor

        with pytest.raises(ValueError, match="Expected 2 `transition_min_limits` values, got 1"):
            _ = HighCloudCompositor("test", transition_min_limits=(210., ))

        with pytest.raises(ValueError, match="Expected 2 `latitude_min_limits` values, got 3"):
            _ = HighCloudCompositor("test", latitude_min_limits=(20., 40., 60.))

        with pytest.raises(ValueError, match="Expected `transition_max` to be of type float, "
                                             "is of type <class 'tuple'>"):
            _ = HighCloudCompositor("test", transition_max=(250., 300.))

        comp = HighCloudCompositor("test")
        with pytest.raises(ValueError, match="Expected 1 dataset, got 2"):
            _ = comp([self.data, self.data])


class TestLowCloudCompositor:
    """Test LowCloudCompositor."""

    def setup_method(self):
        """Create test data."""
        self.dtype = np.float32
        self.btd = xr.DataArray(
            da.from_array(np.array([[0.0, 1.0, 10.0], [0.0, 1.0, 10.0], [0.0, 1.0, 10.0]], dtype=self.dtype)),
            dims=("y", "x"), coords={"y": [0, 1, 2], "x": [0, 1, 2]}
        )
        self.bt_win = xr.DataArray(
            da.from_array(np.array([[250, 250, 250], [250, 250, 250], [150, 150, 150]], dtype=self.dtype)),
            dims=("y", "x"), coords={"y": [0, 1, 2], "x": [0, 1, 2]}
        )
        self.lsm = xr.DataArray(
            da.from_array(np.array([[0., 0., 0.], [1., 1., 1.], [0., 1., 0.]], dtype=self.dtype)),
            dims=("y", "x"), coords={"y": [0, 1, 2], "x": [0, 1, 2]}
        )

    def test_low_cloud_compositor(self):
        """Test general default functionality of compositor."""
        from satpy.composites.mask import LowCloudCompositor
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = LowCloudCompositor(name="test")
            res = comp([self.btd, self.bt_win, self.lsm])
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        expexted_alpha = np.array([[0.0, 0.25, 1.0], [0.0, 0.25, 1.0], [0.0, 0.0, 0.0]])
        expected = np.stack([self.btd, expexted_alpha])
        np.testing.assert_equal(res.values, expected)

    def test_low_cloud_compositor_dtype(self):
        """Test that the datatype is not altered by the compositor."""
        from satpy.composites.mask import LowCloudCompositor
        comp = LowCloudCompositor(name="test")
        res = comp([self.btd, self.bt_win, self.lsm])
        assert res.data.dtype == self.dtype

    def test_low_cloud_compositor_validity_checks(self):
        """Test that errors are raised for invalid input data and settings."""
        from satpy.composites.mask import LowCloudCompositor

        with pytest.raises(ValueError, match="Expected 2 `range_land` values, got 1"):
            _ = LowCloudCompositor("test", range_land=(2.0, ))

        with pytest.raises(ValueError, match="Expected 2 `range_water` values, got 1"):
            _ = LowCloudCompositor("test", range_water=(2.0,))

        comp = LowCloudCompositor("test")
        with pytest.raises(ValueError, match="Expected 3 datasets, got 2"):
            _ = comp([self.btd, self.lsm])


class TestMaskingCompositor:
    """Test case for the simple masking compositor."""

    @pytest.fixture
    def conditions_v1(self):
        """Masking conditions with string values."""
        return [{"method": "equal",
                 "value": "Cloud-free_land",
                 "transparency": 100},
                {"method": "equal",
                 "value": "Cloud-free_sea",
                 "transparency": 50}]

    @pytest.fixture
    def conditions_v2(self):
        """Masking conditions with numerical values."""
        return [{"method": "equal",
                 "value": 1,
                 "transparency": 100},
                {"method": "equal",
                 "value": 2,
                 "transparency": 50}]

    @pytest.fixture
    def conditions_v3(self):
        """Masking conditions with other numerical values."""
        return [{"method": "equal",
                 "value": 0,
                 "transparency": 100},
                {"method": "equal",
                 "value": 1,
                 "transparency": 0}]

    @pytest.fixture
    def test_data(self):
        """Test data to use with masking compositors."""
        return xr.DataArray(da.random.random((3, 3)), dims=["y", "x"])

    @pytest.fixture
    def test_ct_data(self):
        """Test 2D CT data array."""
        flag_meanings = ["Cloud-free_land", "Cloud-free_sea"]
        flag_values = da.array([1, 2])
        ct_data = da.array([[1, 2, 2],
                            [2, 1, 2],
                            [2, 2, 1]])
        ct_data = xr.DataArray(ct_data, dims=["y", "x"])
        ct_data.attrs["flag_meanings"] = flag_meanings
        ct_data.attrs["flag_values"] = flag_values
        return ct_data

    @pytest.fixture
    def value_3d_data(self):
        """Test 3D data array."""
        value_3d_data = da.array([[[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]]])
        value_3d_data = xr.DataArray(value_3d_data, dims=["bands", "y", "x"])
        return value_3d_data

    @pytest.fixture
    def value_3d_data_bands(self):
        """Test 3D data array."""
        value_3d_data = da.array([[[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]],
                                  [[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]],
                                  [[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]]])
        value_3d_data = xr.DataArray(value_3d_data, dims=["bands", "y", "x"])
        return value_3d_data

    @pytest.fixture
    def test_ct_data_v3(self, test_ct_data):
        """Set ct data to NaN where it originally is 1."""
        return test_ct_data.where(test_ct_data == 1)

    @pytest.fixture
    def reference_data(self, test_data, test_ct_data):
        """Get reference data to use in masking compositor tests."""
        # The data are set to NaN where ct is `1`
        return test_data.where(test_ct_data > 1)

    @pytest.fixture
    def reference_alpha(self):
        """Get reference alpha to use in masking compositor tests."""
        ref_alpha = da.array([[0, 0.5, 0.5],
                              [0.5, 0, 0.5],
                              [0.5, 0.5, 0]])
        return xr.DataArray(ref_alpha, dims=["y", "x"])

    def test_init(self):
        """Test the initializiation of compositor."""
        from satpy.composites.mask import MaskingCompositor

        # No transparency or conditions given raises ValueError
        with pytest.raises(ValueError, match="Masking conditions not defined."):
            _ = MaskingCompositor("name")

        # transparency defined
        transparency = {0: 100, 1: 50}
        conditions = [{"method": "equal", "value": 0, "transparency": 100},
                      {"method": "equal", "value": 1, "transparency": 50}]
        comp = MaskingCompositor("name", transparency=transparency.copy())
        assert not hasattr(comp, "transparency")
        # Transparency should be converted to conditions
        assert comp.conditions == conditions

        # conditions defined
        comp = MaskingCompositor("name", conditions=conditions.copy())
        assert comp.conditions == conditions

    def test_get_flag_value(self):
        """Test reading flag value from attributes based on a name."""
        from satpy.composites.mask import _get_flag_value

        flag_values = da.array([1, 2])
        mask = da.array([[1, 2, 2],
                         [2, 1, 2],
                         [2, 2, 1]])
        mask = xr.DataArray(mask, dims=["y", "x"])
        flag_meanings = ["Cloud-free_land", "Cloud-free_sea"]
        mask.attrs["flag_meanings"] = flag_meanings
        mask.attrs["flag_values"] = flag_values

        assert _get_flag_value(mask, "Cloud-free_land") == 1
        assert _get_flag_value(mask, "Cloud-free_sea") == 2

        flag_meanings_str = "Cloud-free_land Cloud-free_sea"
        mask.attrs["flag_meanings"] = flag_meanings_str
        assert _get_flag_value(mask, "Cloud-free_land") == 1
        assert _get_flag_value(mask, "Cloud-free_sea") == 2

    @pytest.mark.parametrize("mode", ["LA", "RGBA"])
    def test_call_numerical_transparency_data(
            self, conditions_v1, test_data, test_ct_data, reference_data,
            reference_alpha, mode):
        """Test call the compositor with numerical transparency data.

        Use parameterisation to test different image modes.
        """
        from satpy.composites.mask import MaskingCompositor
        from satpy.tests.utils import CustomScheduler

        # Test with numerical transparency data
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = MaskingCompositor("name", conditions=conditions_v1,
                                     mode=mode)
            res = comp([test_data, test_ct_data])
        assert res.mode == mode
        for m in mode.rstrip("A"):
            np.testing.assert_allclose(res.sel(bands=m), reference_data)
        np.testing.assert_allclose(res.sel(bands="A"), reference_alpha)

    @pytest.mark.parametrize("mode", ["LA", "RGBA"])
    def test_call_numerical_transparency_data_with_3d_mask_data(
            self, test_data, value_3d_data, conditions_v3, mode):
        """Test call the compositor with numerical transparency data.

        Use parameterisation to test different image modes.
        """
        from satpy.composites.mask import MaskingCompositor

        reference_data_v3 = test_data.where(value_3d_data[0] > 0)
        reference_alpha_v3 = xr.DataArray([[1., 0., 0.],
                                           [0., 1., 0.],
                                           [0., 0., 1.]])

        # Test with numerical transparency data using 3d test mask data which can be squeezed
        comp = MaskingCompositor("name", conditions=conditions_v3,
                                 mode=mode)
        res = comp([test_data, value_3d_data])
        assert res.mode == mode
        for m in mode.rstrip("A"):
            np.testing.assert_allclose(res.sel(bands=m), reference_data_v3)
        np.testing.assert_allclose(res.sel(bands="A"), reference_alpha_v3)

    @pytest.mark.parametrize("mode", ["LA", "RGBA"])
    def test_call_numerical_transparency_data_with_3d_mask_data_exception(
            self, test_data, value_3d_data_bands, conditions_v3, mode):
        """Test call the compositor with numerical transparency data, too many dimensions to squeeze.

        Use parameterisation to test different image modes.
        """
        from satpy.composites.mask import MaskingCompositor

        # Test with numerical transparency data using 3d test mask data which can not be squeezed
        comp = MaskingCompositor("name", conditions=conditions_v3,
                                 mode=mode)
        with pytest.raises(ValueError, match=".*Received 3 dimension\\(s\\) but expected 2.*"):
            comp([test_data, value_3d_data_bands])

    def test_call_named_fields(self, conditions_v2, test_data, test_ct_data,
                               reference_data, reference_alpha):
        """Test with named fields."""
        from satpy.composites.mask import MaskingCompositor
        from satpy.tests.utils import CustomScheduler

        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = MaskingCompositor("name", conditions=conditions_v2)
            res = comp([test_data, test_ct_data])
        assert res.mode == "LA"
        np.testing.assert_allclose(res.sel(bands="L"), reference_data)
        np.testing.assert_allclose(res.sel(bands="A"), reference_alpha)

    def test_call_named_fields_string(
            self, conditions_v2, test_data, test_ct_data, reference_data,
            reference_alpha):
        """Test with named fields which are as a string in the mask attributes."""
        from satpy.composites.mask import MaskingCompositor
        from satpy.tests.utils import CustomScheduler

        flag_meanings_str = "Cloud-free_land Cloud-free_sea"
        test_ct_data.attrs["flag_meanings"] = flag_meanings_str
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = MaskingCompositor("name", conditions=conditions_v2)
            res = comp([test_data, test_ct_data])
        assert res.mode == "LA"
        np.testing.assert_allclose(res.sel(bands="L"), reference_data)
        np.testing.assert_allclose(res.sel(bands="A"), reference_alpha)

    def test_method_isnan(self, test_data,
                          test_ct_data, test_ct_data_v3):
        """Test "isnan" as method."""
        from satpy.composites.mask import MaskingCompositor
        from satpy.tests.utils import CustomScheduler

        conditions_v3 = [{"method": "isnan", "transparency": 100}]

        # The data are set to NaN where ct is NaN
        reference_data_v3 = test_data.where(test_ct_data == 1)
        reference_alpha_v3 = da.array([[1., 0., 0.],
                                       [0., 1., 0.],
                                       [0., 0., 1.]])
        reference_alpha_v3 = xr.DataArray(reference_alpha_v3, dims=["y", "x"])
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = MaskingCompositor("name", conditions=conditions_v3)
            res = comp([test_data, test_ct_data_v3])
        assert res.mode == "LA"
        np.testing.assert_allclose(res.sel(bands="L"), reference_data_v3)
        np.testing.assert_allclose(res.sel(bands="A"), reference_alpha_v3)

    def test_method_absolute_import(self, test_data, test_ct_data_v3):
        """Test "absolute_import" as method."""
        from satpy.composites.mask import MaskingCompositor
        from satpy.tests.utils import CustomScheduler

        conditions_v4 = [{"method": "absolute_import", "transparency": "satpy.resample"}]
        # This should raise AttributeError
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = MaskingCompositor("name", conditions=conditions_v4)
            with pytest.raises(AttributeError):
                comp([test_data, test_ct_data_v3])

    def test_rgb_dataset(self, conditions_v1, test_ct_data, reference_alpha):
        """Test RGB dataset."""
        from satpy.composites.mask import MaskingCompositor
        from satpy.tests.utils import CustomScheduler

        # 3D data array
        data = xr.DataArray(da.random.random((3, 3, 3)),
                            dims=["bands", "y", "x"],
                            coords={"bands": ["R", "G", "B"],
                                    "y": np.arange(3),
                                    "x": np.arange(3)})

        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = MaskingCompositor("name", conditions=conditions_v1)
            res = comp([data, test_ct_data])
        assert res.mode == "RGBA"
        np.testing.assert_allclose(res.sel(bands="R"),
                                   data.sel(bands="R").where(test_ct_data > 1))
        np.testing.assert_allclose(res.sel(bands="G"),
                                   data.sel(bands="G").where(test_ct_data > 1))
        np.testing.assert_allclose(res.sel(bands="B"),
                                   data.sel(bands="B").where(test_ct_data > 1))
        np.testing.assert_allclose(res.sel(bands="A"), reference_alpha)

    def test_rgba_dataset(self, conditions_v2, test_ct_data, reference_alpha):
        """Test RGBA dataset."""
        from satpy.composites.mask import MaskingCompositor
        from satpy.tests.utils import CustomScheduler
        data = xr.DataArray(da.random.random((4, 3, 3)),
                            dims=["bands", "y", "x"],
                            coords={"bands": ["R", "G", "B", "A"],
                                    "y": np.arange(3),
                                    "x": np.arange(3)})

        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = MaskingCompositor("name", conditions=conditions_v2)
            res = comp([data, test_ct_data])
        assert res.mode == "RGBA"
        np.testing.assert_allclose(res.sel(bands="R"),
                                   data.sel(bands="R").where(test_ct_data > 1))
        np.testing.assert_allclose(res.sel(bands="G"),
                                   data.sel(bands="G").where(test_ct_data > 1))
        np.testing.assert_allclose(res.sel(bands="B"),
                                   data.sel(bands="B").where(test_ct_data > 1))
        # The compositor should drop the original alpha band
        np.testing.assert_allclose(res.sel(bands="A"), reference_alpha)

    def test_incorrect_method(self, test_data, test_ct_data):
        """Test incorrect method."""
        from satpy.composites.mask import MaskingCompositor
        conditions = [{"method": "foo", "value": 0, "transparency": 100}]
        comp = MaskingCompositor("name", conditions=conditions)
        with pytest.raises(AttributeError):
            comp([test_data, test_ct_data])
        # Test with too few projectables.
        with pytest.raises(ValueError, match="Expected 2 datasets, got 1"):
            comp([test_data])

    def test_incorrect_mode(self, conditions_v1):
        """Test initiating with unsupported mode."""
        from satpy.composites.mask import MaskingCompositor

        # Incorrect mode raises ValueError
        with pytest.raises(ValueError, match="Invalid mode YCbCrA.  Supported modes: .*"):
            MaskingCompositor("name", conditions=conditions_v1,
                              mode="YCbCrA")


class TestLongitudeMaskingCompositor(unittest.TestCase):
    """Test case for the LongitudeMaskingCompositor compositor."""

    def test_masking(self):
        """Test longitude masking."""
        from satpy.composites.mask import LongitudeMaskingCompositor

        area = mock.MagicMock()
        lons = np.array([-180., -100., -50., 0., 50., 100., 180.])
        area.get_lonlats = mock.MagicMock(return_value=[lons, []])
        a = xr.DataArray(np.array([1, 2, 3, 4, 5, 6, 7]),
                         attrs={"area": area, "units": "K"})

        comp = LongitudeMaskingCompositor(name="test", lon_min=-40., lon_max=120.)
        expected = xr.DataArray(np.array([np.nan, np.nan, np.nan, 4, 5, 6, np.nan]))
        res = comp([a])
        np.testing.assert_allclose(res.data, expected.data)
        assert "units" in res.attrs
        assert res.attrs["units"] == "K"

        comp = LongitudeMaskingCompositor(name="test", lon_min=-40.)
        expected = xr.DataArray(np.array([np.nan, np.nan, np.nan, 4, 5, 6, 7]))
        res = comp([a])
        np.testing.assert_allclose(res.data, expected.data)

        comp = LongitudeMaskingCompositor(name="test", lon_max=120.)
        expected = xr.DataArray(np.array([1, 2, 3, 4, 5, 6, np.nan]))
        res = comp([a])
        np.testing.assert_allclose(res.data, expected.data)

        comp = LongitudeMaskingCompositor(name="test", lon_min=120., lon_max=-40.)
        expected = xr.DataArray(np.array([1, 2, 3, np.nan, np.nan, np.nan, 7]))
        res = comp([a])
        np.testing.assert_allclose(res.data, expected.data)


class TestFireMaskCompositor:
    """Test fire mask compositors."""

    def test_SimpleFireMaskCompositor(self):
        """Test the SimpleFireMaskCompositor class."""
        from satpy.composites.mask import SimpleFireMaskCompositor
        rows = 2
        cols = 2
        ir_105 = xr.DataArray(da.zeros((rows, cols), dtype=np.float32), dims=("y", "x"),
                              attrs={"name": "ir_105"})
        ir_105[0, 0] = 300
        ir_38 = xr.DataArray(da.zeros((rows, cols), dtype=np.float32), dims=("y", "x"),
                             attrs={"name": "ir_38"})
        ir_38[0, 0] = 400
        nir_22 = xr.DataArray(da.zeros((rows, cols), dtype=np.float32), dims=("y", "x"),
                              attrs={"name": "nir_22"})
        nir_22[0, 0] = 100
        vis_06 = xr.DataArray(da.zeros((rows, cols), dtype=np.float32), dims=("y", "x"),
                              attrs={"name": "vis_06"})
        vis_06[0, 0] = 5

        projectables = (ir_105, ir_38, nir_22, vis_06)

        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = SimpleFireMaskCompositor(
                "simple_fci_fire_mask",
                prerequisites=("ir_105", "ir_38", "nir_22", "vis_06"),
                standard_name="simple_fci_fire_mask",
                test_thresholds=[293, 20, 15, 340])
            res = comp(projectables)

        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs["name"] == "simple_fci_fire_mask"
        assert res.data.dtype == bool

        assert np.array_equal(res.data.compute(),
                              np.array([[True, False], [False, False]]))
