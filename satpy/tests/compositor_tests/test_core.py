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

"""Tests for compositor core functionality."""

import unittest
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyresample import AreaDefinition


class TestMatchDataArrays:
    """Test the utility method 'match_data_arrays'."""

    def _get_test_ds(self, shape=(50, 100), dims=("y", "x")):
        """Get a fake DataArray."""
        data = da.random.random(shape, chunks=25)
        area = AreaDefinition(
            "test", "test", "test",
            {"proj": "eqc", "lon_0": 0.0,
             "lat_0": 0.0},
            shape[dims.index("x")], shape[dims.index("y")],
            (-20037508.34, -10018754.17, 20037508.34, 10018754.17))
        attrs = {"area": area}
        return xr.DataArray(data, dims=dims, attrs=attrs)

    def test_single_ds(self):
        """Test a single dataset is returned unharmed."""
        from satpy.composites.core import CompositeBase
        ds1 = self._get_test_ds()
        comp = CompositeBase("test_comp")
        ret_datasets = comp.match_data_arrays((ds1,))
        assert ret_datasets[0].identical(ds1)

    def test_mult_ds_area(self):
        """Test multiple datasets successfully pass."""
        from satpy.composites.core import CompositeBase
        ds1 = self._get_test_ds()
        ds2 = self._get_test_ds()
        comp = CompositeBase("test_comp")
        ret_datasets = comp.match_data_arrays((ds1, ds2))
        assert ret_datasets[0].identical(ds1)
        assert ret_datasets[1].identical(ds2)

    def test_mult_ds_no_area(self):
        """Test that all datasets must have an area attribute."""
        from satpy.composites.core import CompositeBase
        ds1 = self._get_test_ds()
        ds2 = self._get_test_ds()
        del ds2.attrs["area"]
        comp = CompositeBase("test_comp")
        with pytest.raises(ValueError, match="Missing 'area' attribute"):
            comp.match_data_arrays((ds1, ds2))

    def test_mult_ds_diff_area(self):
        """Test that datasets with different areas fail."""
        from satpy.composites.core import CompositeBase, IncompatibleAreas
        ds1 = self._get_test_ds()
        ds2 = self._get_test_ds()
        ds2.attrs["area"] = AreaDefinition(
            "test", "test", "test",
            {"proj": "eqc", "lon_0": 0.0,
             "lat_0": 0.0},
            100, 50,
            (-30037508.34, -20018754.17, 10037508.34, 18754.17))
        comp = CompositeBase("test_comp")
        with pytest.raises(IncompatibleAreas):
            comp.match_data_arrays((ds1, ds2))

    def test_mult_ds_diff_dims(self):
        """Test that datasets with different dimensions still pass."""
        from satpy.composites.core import CompositeBase

        # x is still 50, y is still 100, even though they are in
        # different order
        ds1 = self._get_test_ds(shape=(50, 100), dims=("y", "x"))
        ds2 = self._get_test_ds(shape=(3, 100, 50), dims=("bands", "x", "y"))
        comp = CompositeBase("test_comp")
        ret_datasets = comp.match_data_arrays((ds1, ds2))
        assert ret_datasets[0].identical(ds1)
        assert ret_datasets[1].identical(ds2)

    def test_mult_ds_diff_size(self):
        """Test that datasets with different sizes fail."""
        from satpy.composites.core import CompositeBase, IncompatibleAreas

        # x is 50 in this one, 100 in ds2
        # y is 100 in this one, 50 in ds2
        ds1 = self._get_test_ds(shape=(50, 100), dims=("x", "y"))
        ds2 = self._get_test_ds(shape=(3, 50, 100), dims=("bands", "y", "x"))
        comp = CompositeBase("test_comp")
        with pytest.raises(IncompatibleAreas):
            comp.match_data_arrays((ds1, ds2))

    def test_nondimensional_coords(self):
        """Test the removal of non-dimensional coordinates when compositing."""
        from satpy.composites.core import CompositeBase
        ds = self._get_test_ds(shape=(2, 2))
        ds["acq_time"] = ("y", [0, 1])
        comp = CompositeBase("test_comp")
        ret_datasets = comp.match_data_arrays([ds, ds])
        assert "acq_time" not in ret_datasets[0].coords

    def test_almost_equal_geo_coordinates(self):
        """Test that coordinates that are almost-equal still match.

        See https://github.com/pytroll/satpy/issues/2668 for discussion.

        Various operations like cropping and resampling can cause
        geo-coordinates (y, x) to be very slightly unequal due to floating
        point precision. This test makes sure that even in those cases we
        can still generate composites from DataArrays with these coordinates.

        """
        from satpy.composites.core import CompositeBase
        from satpy.coords import add_crs_xy_coords

        comp = CompositeBase("test_comp")
        data_arr1 = self._get_test_ds(shape=(2, 2))
        data_arr1 = add_crs_xy_coords(data_arr1, data_arr1.attrs["area"])
        data_arr2 = self._get_test_ds(shape=(2, 2))
        data_arr2 = data_arr2.assign_coords(
            x=data_arr1.coords["x"] + 0.000001,
            y=data_arr1.coords["y"],
            crs=data_arr1.coords["crs"],
        )
        # data_arr2 = add_crs_xy_coords(data_arr2, data_arr2.attrs["area"])
        # data_arr2.assign_coords(x=data_arr2.coords["x"].copy() + 1.1)
        # default xarray alignment would fail and collapse one of our dims
        assert 0 in (data_arr2 - data_arr1).shape
        new_data_arr1, new_data_arr2 = comp.match_data_arrays([data_arr1, data_arr2])
        assert 0 not in new_data_arr1.shape
        assert 0 not in new_data_arr2.shape
        assert 0 not in (new_data_arr2 - new_data_arr1).shape


class TestInlineComposites(unittest.TestCase):
    """Test inline composites."""

    def test_inline_composites(self):
        """Test that inline composites are working."""
        from satpy.composites.config_loader import load_compositor_configs_for_sensors
        comps = load_compositor_configs_for_sensors(["visir"])[0]
        # Check that "fog" product has all its prerequisites defined
        keys = comps["visir"].keys()
        fog = [comps["visir"][dsid] for dsid in keys if "fog" == dsid["name"]][0]
        assert fog.attrs["prerequisites"][0]["name"] == "_fog_dep_0"
        assert fog.attrs["prerequisites"][1]["name"] == "_fog_dep_1"
        assert fog.attrs["prerequisites"][2] == 10.8

        # Check that the sub-composite dependencies use wavelengths
        # (numeric values)
        keys = comps["visir"].keys()
        fog_dep_ids = [dsid for dsid in keys if "fog_dep" in dsid["name"]]
        assert comps["visir"][fog_dep_ids[0]].attrs["prerequisites"] == [12.0, 10.8]
        assert comps["visir"][fog_dep_ids[1]].attrs["prerequisites"] == [10.8, 8.7]

        # Check the same for SEVIRI and verify channel names are used
        # in the sub-composite dependencies instead of wavelengths
        comps = load_compositor_configs_for_sensors(["seviri"])[0]
        keys = comps["seviri"].keys()
        fog_dep_ids = [dsid for dsid in keys if "fog_dep" in dsid["name"]]
        assert comps["seviri"][fog_dep_ids[0]].attrs["prerequisites"] == ["IR_120", "IR_108"]
        assert comps["seviri"][fog_dep_ids[1]].attrs["prerequisites"] == ["IR_108", "IR_087"]


class TestSingleBandCompositor(unittest.TestCase):
    """Test the single-band compositor."""

    def setUp(self):
        """Create test data."""
        from satpy.composites.core import SingleBandCompositor
        self.comp = SingleBandCompositor(name="test")

        all_valid = np.ones((2, 2))
        self.all_valid = xr.DataArray(all_valid, dims=["y", "x"])

    def test_call(self):
        """Test calling the compositor."""
        # Dataset with extra attributes
        all_valid = self.all_valid
        all_valid.attrs["sensor"] = "foo"
        attrs = {
            "foo": "bar",
            "resolution": 333,
            "units": "K",
            "sensor": {"fake_sensor1", "fake_sensor2"},
            "calibration": "BT",
            "wavelength": 10.8
        }
        self.comp.attrs["resolution"] = None
        res = self.comp([all_valid], **attrs)
        # Verify attributes
        assert res.attrs.get("sensor") == "foo"
        assert "foo" in res.attrs
        assert res.attrs.get("foo") == "bar"
        assert "units" in res.attrs
        assert "calibration" in res.attrs
        assert "modifiers" not in res.attrs
        assert res.attrs["wavelength"] == 10.8
        assert res.attrs["resolution"] == 333


class TestGenericCompositor(unittest.TestCase):
    """Test generic compositor."""

    def setUp(self):
        """Create test data."""
        from satpy.composites.core import GenericCompositor
        self.comp = GenericCompositor(name="test")
        self.comp2 = GenericCompositor(name="test2", common_channel_mask=False)

        all_valid = np.ones((1, 2, 2))
        self.all_valid = xr.DataArray(all_valid, dims=["bands", "y", "x"])
        first_invalid = np.reshape(np.array([np.nan, 1., 1., 1.]), (1, 2, 2))
        self.first_invalid = xr.DataArray(first_invalid,
                                          dims=["bands", "y", "x"])
        second_invalid = np.reshape(np.array([1., np.nan, 1., 1.]), (1, 2, 2))
        self.second_invalid = xr.DataArray(second_invalid,
                                           dims=["bands", "y", "x"])
        wrong_shape = np.reshape(np.array([1., 1., 1.]), (1, 3, 1))
        self.wrong_shape = xr.DataArray(wrong_shape, dims=["bands", "y", "x"])

    def test_masking(self):
        """Test masking in generic compositor."""
        # Single channel
        res = self.comp([self.all_valid])
        np.testing.assert_allclose(res.data, 1., atol=1e-9)
        # Three channels, one value invalid
        res = self.comp([self.all_valid, self.all_valid, self.first_invalid])
        correct = np.reshape(np.array([np.nan, 1., 1., 1.]), (2, 2))
        for i in range(3):
            np.testing.assert_almost_equal(res.data[i, :, :], correct)
        # Three channels, two values invalid
        res = self.comp([self.all_valid, self.first_invalid, self.second_invalid])
        correct = np.reshape(np.array([np.nan, np.nan, 1., 1.]), (2, 2))
        for i in range(3):
            np.testing.assert_almost_equal(res.data[i, :, :], correct)

    def test_concat_datasets(self):
        """Test concatenation of datasets."""
        from satpy.composites.core import IncompatibleAreas
        res = self.comp._concat_datasets([self.all_valid], "L")
        num_bands = len(res.bands)
        assert num_bands == 1
        assert res.shape[0] == num_bands
        assert res.bands[0] == "L"
        res = self.comp._concat_datasets([self.all_valid, self.all_valid], "LA")
        num_bands = len(res.bands)
        assert num_bands == 2
        assert res.shape[0] == num_bands
        assert res.bands[0] == "L"
        assert res.bands[1] == "A"
        with pytest.raises(IncompatibleAreas):
            self.comp._concat_datasets([self.all_valid, self.wrong_shape], "LA")

    def test_get_sensors(self):
        """Test getting sensors from the dataset attributes."""
        res = self.comp._get_sensors([self.all_valid])
        assert res is None
        dset1 = self.all_valid
        dset1.attrs["sensor"] = "foo"
        res = self.comp._get_sensors([dset1])
        assert res == "foo"
        dset2 = self.first_invalid
        dset2.attrs["sensor"] = "bar"
        res = self.comp._get_sensors([dset1, dset2])
        assert "foo" in res
        assert "bar" in res
        assert len(res) == 2
        assert isinstance(res, set)

    @mock.patch("satpy.composites.core.GenericCompositor._get_sensors")
    @mock.patch("satpy.composites.core.combine_metadata")
    @mock.patch("satpy.composites.core.check_times")
    @mock.patch("satpy.composites.core.GenericCompositor.match_data_arrays")
    def test_call_with_mock(self, match_data_arrays, check_times, combine_metadata, get_sensors):
        """Test calling generic compositor."""
        from satpy.composites.core import IncompatibleAreas
        combine_metadata.return_value = dict()
        get_sensors.return_value = "foo"
        # One dataset, no mode given
        res = self.comp([self.all_valid])
        assert res.shape[0] == 1
        assert res.attrs["mode"] == "L"
        match_data_arrays.assert_not_called()
        # This compositor has been initialized without common masking, so the
        # masking shouldn't have been called
        projectables = [self.all_valid, self.first_invalid, self.second_invalid]
        match_data_arrays.return_value = projectables
        res = self.comp2(projectables)
        match_data_arrays.assert_called_once()
        match_data_arrays.reset_mock()
        # Dataset for alpha given, so shouldn't be masked
        projectables = [self.all_valid, self.all_valid]
        match_data_arrays.return_value = projectables
        res = self.comp(projectables)
        match_data_arrays.assert_called_once()
        match_data_arrays.reset_mock()
        # When areas are incompatible, masking shouldn't happen
        match_data_arrays.side_effect = IncompatibleAreas()
        with pytest.raises(IncompatibleAreas):
            self.comp([self.all_valid, self.wrong_shape])
        match_data_arrays.assert_called_once()

    def test_call(self):
        """Test calling generic compositor."""
        # Multiple datasets with extra attributes
        all_valid = self.all_valid
        all_valid.attrs["sensor"] = "foo"
        attrs = {"foo": "bar", "resolution": 333}
        self.comp.attrs["resolution"] = None
        res = self.comp([self.all_valid, self.first_invalid], **attrs)
        # Verify attributes
        assert res.attrs.get("sensor") == "foo"
        assert "foo" in res.attrs
        assert res.attrs.get("foo") == "bar"
        assert "units" not in res.attrs
        assert "calibration" not in res.attrs
        assert "modifiers" not in res.attrs
        assert res.attrs["wavelength"] is None
        assert res.attrs["mode"] == "LA"
        assert res.attrs["resolution"] == 333

    def test_deprecation_warning(self):
        """Test deprecation warning for dcprecated composite recipes."""
        warning_message = "foo is a deprecated composite. Use composite bar instead."
        self.comp.attrs["deprecation_warning"] = warning_message
        with pytest.warns(UserWarning, match=warning_message):
            self.comp([self.all_valid])


class TestAddBands(unittest.TestCase):
    """Test case for the `add_bands` function."""

    def test_add_bands_l_rgb(self):
        """Test adding bands."""
        from satpy.composites.core import add_bands

        # L + RGB -> RGB
        data = xr.DataArray(da.ones((1, 3, 3), dtype="float32"), dims=("bands", "y", "x"),
                            coords={"bands": ["L"]})
        new_bands = xr.DataArray(da.array(["R", "G", "B"]), dims=("bands"),
                                 coords={"bands": ["R", "G", "B"]})
        res = add_bands(data, new_bands)
        res_bands = ["R", "G", "B"]
        _check_add_band_results(res, res_bands, np.float32)

    def test_add_bands_l_rgba(self):
        """Test adding bands."""
        from satpy.composites.core import add_bands

        # L + RGBA -> RGBA
        data = xr.DataArray(da.ones((1, 3, 3), dtype="float32"), dims=("bands", "y", "x"),
                            coords={"bands": ["L"]}, attrs={"mode": "L"})
        new_bands = xr.DataArray(da.array(["R", "G", "B", "A"]), dims=("bands"),
                                 coords={"bands": ["R", "G", "B", "A"]})
        res = add_bands(data, new_bands)
        res_bands = ["R", "G", "B", "A"]
        _check_add_band_results(res, res_bands, np.float32)

    def test_add_bands_la_rgb(self):
        """Test adding bands."""
        from satpy.composites.core import add_bands

        # LA + RGB -> RGBA
        data = xr.DataArray(da.ones((2, 3, 3), dtype="float32"), dims=("bands", "y", "x"),
                            coords={"bands": ["L", "A"]}, attrs={"mode": "LA"})
        new_bands = xr.DataArray(da.array(["R", "G", "B"]), dims=("bands"),
                                 coords={"bands": ["R", "G", "B"]})
        res = add_bands(data, new_bands)
        res_bands = ["R", "G", "B", "A"]
        _check_add_band_results(res, res_bands, np.float32)

    def test_add_bands_rgb_rbga(self):
        """Test adding bands."""
        from satpy.composites.core import add_bands

        # RGB + RGBA -> RGBA
        data = xr.DataArray(da.ones((3, 3, 3), dtype="float32"), dims=("bands", "y", "x"),
                            coords={"bands": ["R", "G", "B"]},
                            attrs={"mode": "RGB"})
        new_bands = xr.DataArray(da.array(["R", "G", "B", "A"]), dims=("bands"),
                                 coords={"bands": ["R", "G", "B", "A"]})
        res = add_bands(data, new_bands)
        res_bands = ["R", "G", "B", "A"]
        _check_add_band_results(res, res_bands, np.float32)

    def test_add_bands_p_l(self):
        """Test adding bands."""
        from satpy.composites.core import add_bands

        # P(RGBA) + L -> RGBA
        data = xr.DataArray(da.ones((1, 3, 3)), dims=("bands", "y", "x"),
                            coords={"bands": ["P"]},
                            attrs={"mode": "P"})
        new_bands = xr.DataArray(da.array(["L"]), dims=("bands"),
                                 coords={"bands": ["L"]})
        with pytest.raises(NotImplementedError):
            add_bands(data, new_bands)


def _check_add_band_results(res, res_bands, dtype):
    assert res.attrs["mode"] == "".join(res_bands)
    np.testing.assert_array_equal(res.bands, res_bands)
    np.testing.assert_array_equal(res.coords["bands"], res_bands)
    assert res.dtype == dtype


class TestEnhance2Dataset(unittest.TestCase):
    """Test the enhance2dataset utility."""

    @mock.patch("satpy.enhancements.enhancer.get_enhanced_image")
    def test_enhance_p_to_rgb(self, get_enhanced_image):
        """Test enhancing a paletted dataset in RGB mode."""
        from trollimage.xrimage import XRImage
        img = XRImage(xr.DataArray(np.ones((1, 20, 20)) * 2, dims=("bands", "y", "x"), coords={"bands": ["P"]}))
        img.palette = ((0, 0, 0), (4, 4, 4), (8, 8, 8))
        get_enhanced_image.return_value = img

        from satpy.composites.core import enhance2dataset
        dataset = xr.DataArray(np.ones((1, 20, 20)))
        res = enhance2dataset(dataset, convert_p=True)
        assert res.attrs["mode"] == "RGB"

    @mock.patch("satpy.enhancements.enhancer.get_enhanced_image")
    def test_enhance_p_to_rgba(self, get_enhanced_image):
        """Test enhancing a paletted dataset in RGBA mode."""
        from trollimage.xrimage import XRImage
        img = XRImage(xr.DataArray(np.ones((1, 20, 20)) * 2, dims=("bands", "y", "x"), coords={"bands": ["P"]}))
        img.palette = ((0, 0, 0, 255), (4, 4, 4, 255), (8, 8, 8, 255))
        get_enhanced_image.return_value = img

        from satpy.composites.core import enhance2dataset
        dataset = xr.DataArray(np.ones((1, 20, 20)))
        res = enhance2dataset(dataset, convert_p=True)
        assert res.attrs["mode"] == "RGBA"

    @mock.patch("satpy.enhancements.enhancer.get_enhanced_image")
    def test_enhance_p(self, get_enhanced_image):
        """Test enhancing a paletted dataset in P mode."""
        from trollimage.xrimage import XRImage
        img = XRImage(xr.DataArray(np.ones((1, 20, 20)) * 2, dims=("bands", "y", "x"), coords={"bands": ["P"]}))
        img.palette = ((0, 0, 0, 255), (4, 4, 4, 255), (8, 8, 8, 255))
        get_enhanced_image.return_value = img

        from satpy.composites.core import enhance2dataset
        dataset = xr.DataArray(np.ones((1, 20, 20)))
        res = enhance2dataset(dataset)
        assert res.attrs["mode"] == "P"
        assert res.max().values == 2

    @mock.patch("satpy.enhancements.enhancer.get_enhanced_image")
    def test_enhance_l(self, get_enhanced_image):
        """Test enhancing a paletted dataset in P mode."""
        from trollimage.xrimage import XRImage
        img = XRImage(xr.DataArray(np.ones((1, 20, 20)) * 2, dims=("bands", "y", "x"), coords={"bands": ["L"]}))
        get_enhanced_image.return_value = img

        from satpy.composites.core import enhance2dataset
        dataset = xr.DataArray(np.ones((1, 20, 20)))
        res = enhance2dataset(dataset)
        assert res.attrs["mode"] == "L"
        assert res.max().values == 1


class TestInferMode(unittest.TestCase):
    """Test the infer_mode utility."""

    def test_bands_coords_is_used(self):
        """Test that the `bands` coord is used."""
        from satpy.composites.core import GenericCompositor
        arr = xr.DataArray(np.ones((1, 5, 5)), dims=("bands", "x", "y"), coords={"bands": ["P"]})
        assert GenericCompositor.infer_mode(arr) == "P"

        arr = xr.DataArray(np.ones((3, 5, 5)), dims=("bands", "x", "y"), coords={"bands": ["Y", "Cb", "Cr"]})
        assert GenericCompositor.infer_mode(arr) == "YCbCr"

    def test_mode_is_used(self):
        """Test that the `mode` attribute is used."""
        from satpy.composites.core import GenericCompositor
        arr = xr.DataArray(np.ones((1, 5, 5)), dims=("bands", "x", "y"), attrs={"mode": "P"})
        assert GenericCompositor.infer_mode(arr) == "P"

    def test_band_size_is_used(self):
        """Test that the band size is used."""
        from satpy.composites.core import GenericCompositor
        arr = xr.DataArray(np.ones((2, 5, 5)), dims=("bands", "x", "y"))
        assert GenericCompositor.infer_mode(arr) == "LA"

    def test_no_bands_is_l(self):
        """Test that default (no band) is L."""
        from satpy.composites.core import GenericCompositor
        arr = xr.DataArray(np.ones((5, 5)), dims=("x", "y"))
        assert GenericCompositor.infer_mode(arr) == "L"
