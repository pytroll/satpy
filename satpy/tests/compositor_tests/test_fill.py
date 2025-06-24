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

"""Tests for compositors filling composites with other composites."""

import datetime as dt
import unittest
from unittest import mock

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyresample import AreaDefinition

from satpy.tests.utils import CustomScheduler


class TestDayNightCompositor(unittest.TestCase):
    """Test DayNightCompositor."""

    def setUp(self):
        """Create test data."""
        bands = ["R", "G", "B"]
        start_time = dt.datetime(2018, 1, 1, 18, 0, 0)

        # RGB
        a = np.zeros((3, 2, 2), dtype=np.float32)
        a[:, 0, 0] = 0.1
        a[:, 0, 1] = 0.2
        a[:, 1, 0] = 0.3
        a[:, 1, 1] = 0.4
        a = da.from_array(a, a.shape)
        self.data_a = xr.DataArray(a, attrs={"test": "a", "start_time": start_time},
                                   coords={"bands": bands}, dims=("bands", "y", "x"))
        b = np.zeros((3, 2, 2), dtype=np.float32)
        b[:, 0, 0] = np.nan
        b[:, 0, 1] = 0.25
        b[:, 1, 0] = 0.50
        b[:, 1, 1] = 0.75
        b = da.from_array(b, b.shape)
        self.data_b = xr.DataArray(b, attrs={"test": "b", "start_time": start_time},
                                   coords={"bands": bands}, dims=("bands", "y", "x"))

        sza = np.array([[80., 86.], [94., 100.]], dtype=np.float32)
        sza = da.from_array(sza, sza.shape)
        self.sza = xr.DataArray(sza, dims=("y", "x"))

        # fake area
        my_area = AreaDefinition(
            "test", "", "",
            "+proj=longlat",
            2, 2,
            (-95.0, 40.0, -92.0, 43.0),
        )
        self.data_a.attrs["area"] = my_area
        self.data_b.attrs["area"] = my_area
        # not used except to check that it matches the data arrays
        self.sza.attrs["area"] = my_area

    def test_daynight_sza(self):
        """Test compositor with both day and night portions when SZA data is included."""
        from satpy.composites.fill import DayNightCompositor

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = DayNightCompositor(name="dn_test", day_night="day_night")
            res = comp((self.data_a, self.data_b, self.sza))
            res = res.compute()
        expected = np.array([[0., 0.22122374], [0.5, 1.]], dtype=np.float32)
        assert res.dtype == np.float32
        np.testing.assert_allclose(res.values[0], expected, rtol=1e-6)

    def test_daynight_area(self):
        """Test compositor both day and night portions when SZA data is not provided."""
        from satpy.composites.fill import DayNightCompositor

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = DayNightCompositor(name="dn_test", day_night="day_night")
            res = comp((self.data_a, self.data_b))
            res = res.compute()
        expected_channel = np.array([[0., 0.33164983], [0.66835017, 1.]], dtype=np.float32)
        assert res.dtype == np.float32
        for i in range(3):
            np.testing.assert_allclose(res.values[i], expected_channel)

    def test_night_only_sza_with_alpha(self):
        """Test compositor with night portion with alpha band when SZA data is included."""
        from satpy.composites.fill import DayNightCompositor

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = DayNightCompositor(name="dn_test", day_night="night_only", include_alpha=True)
            res = comp((self.data_b, self.sza))
            res = res.compute()
        expected_red_channel = np.array([[np.nan, 0.], [0.5, 1.]], dtype=np.float32)
        expected_alpha = np.array([[0., 0.3329599], [1., 1.]], dtype=np.float32)
        assert res.dtype == np.float32
        np.testing.assert_allclose(res.values[0], expected_red_channel)
        np.testing.assert_allclose(res.values[-1], expected_alpha)

    def test_night_only_sza_without_alpha(self):
        """Test compositor with night portion without alpha band when SZA data is included."""
        from satpy.composites.fill import DayNightCompositor

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = DayNightCompositor(name="dn_test", day_night="night_only", include_alpha=False)
            res = comp((self.data_a, self.sza))
            res = res.compute()
        expected = np.array([[0., 0.11042609], [0.6683502, 1.]], dtype=np.float32)
        assert res.dtype == np.float32
        np.testing.assert_allclose(res.values[0], expected)
        assert "A" not in res.bands

    def test_night_only_area_with_alpha(self):
        """Test compositor with night portion with alpha band when SZA data is not provided."""
        from satpy.composites.fill import DayNightCompositor

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = DayNightCompositor(name="dn_test", day_night="night_only", include_alpha=True)
            res = comp((self.data_b,))
            res = res.compute()
        expected_l_channel = np.array([[np.nan, 0.], [0.5, 1.]], dtype=np.float32)
        expected_alpha = np.array([[np.nan, 0.], [0., 0.]], dtype=np.float32)
        assert res.dtype == np.float32
        np.testing.assert_allclose(res.values[0], expected_l_channel)
        np.testing.assert_allclose(res.values[-1], expected_alpha)

    def test_night_only_area_without_alpha(self):
        """Test compositor with night portion without alpha band when SZA data is not provided."""
        from satpy.composites.fill import DayNightCompositor

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = DayNightCompositor(name="dn_test", day_night="night_only", include_alpha=False)
            res = comp((self.data_b,))
            res = res.compute()
        expected = np.array([[np.nan, 0.], [0., 0.]], dtype=np.float32)
        assert res.dtype == np.float32
        np.testing.assert_allclose(res.values[0], expected)
        assert "A" not in res.bands

    def test_day_only_sza_with_alpha(self):
        """Test compositor with day portion with alpha band when SZA data is included."""
        from satpy.composites.fill import DayNightCompositor

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = DayNightCompositor(name="dn_test", day_night="day_only", include_alpha=True)
            res = comp((self.data_a, self.sza))
            res = res.compute()
        expected_red_channel = np.array([[0., 0.33164983], [0.66835017, 1.]], dtype=np.float32)
        expected_alpha = np.array([[1., 0.6670401], [0., 0.]], dtype=np.float32)
        assert res.dtype == np.float32
        np.testing.assert_allclose(res.values[0], expected_red_channel)
        np.testing.assert_allclose(res.values[-1], expected_alpha)

    def test_day_only_sza_without_alpha(self):
        """Test compositor with day portion without alpha band when SZA data is included."""
        from satpy.composites.fill import DayNightCompositor

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = DayNightCompositor(name="dn_test", day_night="day_only", include_alpha=False)
            res = comp((self.data_a, self.sza))
            res = res.compute()
        expected_channel_data = np.array([[0., 0.22122373], [0., 0.]], dtype=np.float32)
        assert res.dtype == np.float32
        for i in range(3):
            np.testing.assert_allclose(res.values[i], expected_channel_data)
        assert "A" not in res.bands

    def test_day_only_area_with_alpha(self):
        """Test compositor with day portion with alpha_band when SZA data is not provided."""
        from satpy.composites.fill import DayNightCompositor

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = DayNightCompositor(name="dn_test", day_night="day_only", include_alpha=True)
            res = comp((self.data_a,))
            res = res.compute()
        expected_l_channel = np.array([[0., 0.33164983], [0.66835017, 1.]], dtype=np.float32)
        expected_alpha = np.array([[1., 1.], [1., 1.]], dtype=np.float32)
        assert res.dtype == np.float32
        np.testing.assert_allclose(res.values[0], expected_l_channel)
        np.testing.assert_allclose(res.values[-1], expected_alpha)

    def test_day_only_area_with_alpha_and_missing_data(self):
        """Test compositor with day portion with alpha_band when SZA data is not provided and there is missing data."""
        from satpy.composites.fill import DayNightCompositor

        with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
            comp = DayNightCompositor(name="dn_test", day_night="day_only", include_alpha=True)
            res = comp((self.data_b,))
            res = res.compute()
        expected_l_channel = np.array([[np.nan, 0.], [0.5, 1.]], dtype=np.float32)
        expected_alpha = np.array([[np.nan, 1.], [1., 1.]], dtype=np.float32)
        assert res.dtype == np.float32
        np.testing.assert_allclose(res.values[0], expected_l_channel)
        np.testing.assert_allclose(res.values[-1], expected_alpha)

    def test_day_only_area_without_alpha(self):
        """Test compositor with day portion without alpha_band when SZA data is not provided."""
        from satpy.composites.fill import DayNightCompositor

        # with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
        comp = DayNightCompositor(name="dn_test", day_night="day_only", include_alpha=False)
        res_dask = comp((self.data_a,))
        res = res_dask.compute()
        expected = np.array([[0., 0.33164983], [0.66835017, 1.]], dtype=np.float32)
        assert res_dask.dtype == res.dtype
        assert res.dtype == np.float32
        np.testing.assert_allclose(res.values[0], expected)
        assert "A" not in res.bands


class TestFillingCompositor(unittest.TestCase):
    """Test case for the filling compositor."""

    def test_fill(self):
        """Test filling."""
        from satpy.composites.fill import FillingCompositor
        comp = FillingCompositor(name="fill_test")
        filler = xr.DataArray(np.array([1, 2, 3, 4, 3, 2, 1]))
        red = xr.DataArray(np.array([1, 2, 3, np.nan, 3, 2, 1]))
        green = xr.DataArray(np.array([np.nan, 2, 3, 4, 3, 2, np.nan]))
        blue = xr.DataArray(np.array([4, 3, 2, 1, 2, 3, 4]))
        res = comp([filler, red, green, blue])
        np.testing.assert_allclose(res.sel(bands="R").data, filler.data)
        np.testing.assert_allclose(res.sel(bands="G").data, filler.data)
        np.testing.assert_allclose(res.sel(bands="B").data, blue.data)


class TestMultiFiller(unittest.TestCase):
    """Test case for the MultiFiller compositor."""

    def test_fill(self):
        """Test filling."""
        from satpy.composites.fill import MultiFiller
        comp = MultiFiller(name="fill_test")
        attrs = {"units": "K"}
        a = xr.DataArray(np.array([1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]), attrs=attrs.copy())
        b = xr.DataArray(np.array([np.nan, 2, 3, np.nan, np.nan, np.nan, np.nan]), attrs=attrs.copy())
        c = xr.DataArray(np.array([np.nan, 22, 3, np.nan, np.nan, np.nan, 7]), attrs=attrs.copy())
        d = xr.DataArray(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 6, np.nan]), attrs=attrs.copy())
        e = xr.DataArray(np.array([np.nan, np.nan, np.nan, np.nan, 5, np.nan, np.nan]), attrs=attrs.copy())
        expected = xr.DataArray(np.array([1, 2, 3, np.nan, 5, 6, 7]))
        res = comp([a, b, c], optional_datasets=[d, e])
        np.testing.assert_allclose(res.data, expected.data)
        assert "units" in res.attrs
        assert res.attrs["units"] == "K"


def _enhance2dataset(dataset, convert_p=False):
    """Mock the enhance2dataset to return the original data."""
    return dataset


class TestBackgroundCompositor:
    """Test case for the background compositor."""

    @classmethod
    def setup_class(cls):
        """Create shared input data arrays."""
        foreground_data = {
            "L": np.array([[[1., 0.5], [0., np.nan]]]),
            "LA": np.array([[[1., 0.5], [0., np.nan]], [[0.5, 0.5], [0.5, 0.5]]]),
            "RGB": np.array([
                [[1., 0.5], [0., np.nan]],
                [[1., 0.5], [0., np.nan]],
                [[1., 0.5], [0., np.nan]]]),
            "RGBA": np.array([
                [[1., 0.5], [0., np.nan]],
                [[1., 0.5], [0., np.nan]],
                [[1., 0.5], [0., np.nan]],
                [[0.5, 0.5], [0., 0.5]]]),
        }
        cls.foreground_data = foreground_data

    @mock.patch("satpy.composites.fill.enhance2dataset", _enhance2dataset)
    @pytest.mark.parametrize(
        ("foreground_bands", "background_bands", "exp_bands", "exp_result"),
        [
            ("L", "L", "L", np.array([[1., 0.5], [0., 1.]])),
            ("L", "RGB", "RGB", np.array([
                [[1., 0.5], [0., 1.]],
                [[1., 0.5], [0., 1.]],
                [[1., 0.5], [0., 1.]]])),
            ("LA", "LA", "LA", np.array([
                [[1., 0.75], [0.5, 1.]],
                [[1., 1.], [1., 1.]]])),
            ("LA", "RGB", "RGB", np.array([
                [[1., 0.75], [0.5, 1.]],
                [[1., 0.75], [0.5, 1.]],
                [[1., 0.75], [0.5, 1.]]])),
            ("RGB", "RGB", "RGB", np.array([
                [[1., 0.5], [0., 1.]],
                [[1., 0.5], [0., 1.]],
                [[1., 0.5], [0., 1.]]])),
            ("RGB", "LA", "RGBA", np.array([
                [[1., 0.5], [0., 1.]],
                [[1., 0.5], [0., 1.]],
                [[1., 0.5], [0., 1.]],
                [[1., 1.], [1., 1.]]])),
            ("RGB", "RGBA", "RGBA", np.array([
                [[1., 0.5], [0., 1.]],
                [[1., 0.5], [0., 1.]],
                [[1., 0.5], [0., 1.]],
                [[1., 1.], [1., 1.]]])),
            ("RGBA", "RGBA", "RGBA", np.array([
                [[1., 0.75], [1., 1.]],
                [[1., 0.75], [1., 1.]],
                [[1., 0.75], [1., 1.]],
                [[1., 1.], [1., 1.]]])),
            ("RGBA", "RGB", "RGB", np.array([
                [[1., 0.75], [1., 1.]],
                [[1., 0.75], [1., 1.]],
                [[1., 0.75], [1., 1.]]])),
        ]
    )
    def test_call(self, foreground_bands, background_bands, exp_bands, exp_result):
        """Test the background compositing."""
        from satpy.composites.fill import BackgroundCompositor
        comp = BackgroundCompositor("name")

        # L mode images
        foreground_data = self.foreground_data[foreground_bands]

        attrs = {"mode": foreground_bands, "area": "foo"}
        foreground = xr.DataArray(da.from_array(foreground_data),
                                  dims=("bands", "y", "x"),
                                  coords={"bands": [c for c in attrs["mode"]]},
                                  attrs=attrs)
        attrs = {"mode": background_bands, "area": "foo"}
        background = xr.DataArray(da.ones((len(background_bands), 2, 2)), dims=("bands", "y", "x"),
                                  coords={"bands": [c for c in attrs["mode"]]},
                                  attrs=attrs)

        res = comp([foreground, background])

        assert res.attrs["area"] == "foo"
        np.testing.assert_allclose(res, exp_result)
        assert res.attrs["mode"] == exp_bands

    @mock.patch("satpy.composites.fill.enhance2dataset", _enhance2dataset)
    def test_multiple_sensors(self):
        """Test the background compositing from multiple sensor data."""
        from satpy.composites.fill import BackgroundCompositor
        comp = BackgroundCompositor("name")

        # L mode images
        attrs = {"mode": "L", "area": "foo"}
        foreground_data = self.foreground_data["L"]
        foreground = xr.DataArray(da.from_array(foreground_data),
                                  dims=("bands", "y", "x"),
                                  coords={"bands": [c for c in attrs["mode"]]},
                                  attrs=attrs.copy())
        foreground.attrs["sensor"] = "abi"
        background = xr.DataArray(da.ones((1, 2, 2)), dims=("bands", "y", "x"),
                                  coords={"bands": [c for c in attrs["mode"]]},
                                  attrs=attrs.copy())
        background.attrs["sensor"] = "glm"
        res = comp([foreground, background])
        assert res.attrs["area"] == "foo"
        np.testing.assert_allclose(res, np.array([[1., 0.5], [0., 1.]]))
        assert res.attrs["mode"] == "L"
        assert res.attrs["sensor"] == {"abi", "glm"}
