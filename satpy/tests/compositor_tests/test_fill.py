
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

from satpy.composites.fill import DayNightCompositor
from satpy.tests.utils import CustomScheduler

# Solar zenith angles used when the SZA is given as an input. With the default limits (85 and 88 degrees)
# these pixels are fully day, twilight, fully night, and fully night.
SZA = [[80., 86.], [94., 100.]]
# Blending weight (1 = day, 0 = night) of the twilight pixel
TWILIGHT_WEIGHT = 0.6670401
# When the SZA is computed instead, it is about 64 degrees over the whole area at START_TIME, so every
# pixel is fully day.
AREA = AreaDefinition("test", "", "", "+proj=longlat", 2, 2, (-95.0, 40.0, -92.0, 43.0))
START_TIME = dt.datetime(2018, 1, 1, 18, 0, 0)

DAY = [[0.1, 0.2], [0.3, 0.4]]
NIGHT = [[0.6, 0.7], [0.8, 0.9]]
# DAY and NIGHT blended at the twilight pixel
TWILIGHT_BLEND = 0.2 * TWILIGHT_WEIGHT + 0.7 * (1 - TWILIGHT_WEIGHT)
# Alpha band value of LA and RGBA inputs
INPUT_ALPHA = 0.5


def _make_image(mode, values):
    """Make an image with ``values`` in every color band and INPUT_ALPHA in the alpha band, if any."""
    values = np.array(values, dtype=np.float32)
    bands = [np.full_like(values, INPUT_ALPHA) if band == "A" else values for band in mode]
    return xr.DataArray(da.from_array(np.stack(bands)), dims=("bands", "y", "x"), coords={"bands": list(mode)},
                        attrs={"mode": mode, "area": AREA, "start_time": START_TIME})


def _make_sza():
    """Make the SZA input."""
    return xr.DataArray(da.from_array(np.array(SZA, dtype=np.float32)), dims=("y", "x"), attrs={"area": AREA})


def _enhance2dataset(dataset, convert_p=False):
    """Mock the enhance2dataset to return the original data."""
    return dataset


def test_day_night_compositor_enhances_inputs():
    """Test that both inputs are enhanced before they are blended."""
    with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
        comp = DayNightCompositor(name="dn_test", day_night="day_night")
        res_dask = comp((_make_image("RGB", DAY), _make_image("RGB", NIGHT), _make_sza()))
        res = res_dask.compute()
    # The default linear stretch maps both inputs onto the same values
    expected = np.array([[0., 0.33164983], [0.66835017, 1.]], dtype=np.float32)
    assert res_dask.dtype == res.dtype
    assert res.dtype == np.float32
    for band in "RGB":
        np.testing.assert_allclose(res.sel(bands=band).values, expected, rtol=1e-6)


@mock.patch("satpy.composites.fill.enhance2dataset", _enhance2dataset)
@pytest.mark.parametrize(
    ("include_alpha", "sza_given", "day", "night", "expected"),
    [
        pytest.param(True, True, DAY, [[np.nan, 0.7], [0.8, 0.9]], [[0.1, TWILIGHT_BLEND], [0.8, 0.9]],
                     id="sza-night-missing-on-day-side"),
        pytest.param(True, True, DAY, [[0.6, np.nan], [0.8, 0.9]], [[0.1, 0.2 * TWILIGHT_WEIGHT], [0.8, 0.9]],
                     id="sza-night-missing-in-twilight"),
        pytest.param(True, True, [[np.nan, 0.2], [0.3, 0.4]], NIGHT, [[np.nan, TWILIGHT_BLEND], [0.8, 0.9]],
                     id="sza-day-missing-on-day-side"),
        pytest.param(False, True, [[np.nan, 0.2], [0.3, 0.4]], NIGHT, [[0., TWILIGHT_BLEND], [0.8, 0.9]],
                     id="sza-day-missing-on-day-side-without-alpha"),
        pytest.param(True, False, [[np.nan, 0.2], [0.3, 0.4]], NIGHT, [[np.nan, 0.2], [0.3, 0.4]],
                     id="computed-day-missing"),
        pytest.param(False, False, [[np.nan, 0.2], [0.3, 0.4]], NIGHT, [[0., 0.2], [0.3, 0.4]],
                     id="computed-day-missing-without-alpha"),
    ],
)
def test_day_night_compositor_day_night(include_alpha, sza_given, day, night, expected):
    """Test blending a day and a night input, including pixels where one of them is missing."""
    inputs = (_make_image("RGB", day), _make_image("RGB", night))
    if sza_given:
        inputs += (_make_sza(),)
    with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
        comp = DayNightCompositor(name="dn_test", day_night="day_night", include_alpha=include_alpha)
        res_dask = comp(inputs)
        res = res_dask.compute()
    assert list(res["bands"].values) == ["R", "G", "B"]
    assert res_dask.dtype == res.dtype
    assert res.dtype == np.float32
    for band in "RGB":
        np.testing.assert_allclose(res.sel(bands=band).values, expected, rtol=1e-6)


@mock.patch("satpy.composites.fill.enhance2dataset", _enhance2dataset)
@pytest.mark.parametrize(
    ("day_night", "mode", "sza_given", "values", "exp_weight"),
    [
        pytest.param("day_only", "RGB", True, DAY, [[1., TWILIGHT_WEIGHT], [0., 0.]],
                     id="day_only-RGB-sza"),
        pytest.param("day_only", "RGB", True, [[np.nan, 0.2], [0.3, 0.4]], [[np.nan, TWILIGHT_WEIGHT], [0., 0.]],
                     id="day_only-RGB-sza-missing-on-day-side"),
        pytest.param("day_only", "RGB", False, [[np.nan, 0.2], [0.3, 0.4]], [[np.nan, 1.], [1., 1.]],
                     id="day_only-RGB-computed-missing-on-day-side"),
        pytest.param("night_only", "RGB", True, [[0.6, 0.7], [np.nan, 0.9]],
                     [[0., 1 - TWILIGHT_WEIGHT], [np.nan, 1.]],
                     id="night_only-RGB-sza-missing-on-night-side"),
        pytest.param("night_only", "RGB", False, [[np.nan, 0.7], [0.8, 0.9]], [[np.nan, 0.], [0., 0.]],
                     id="night_only-RGB-computed-missing-on-day-side"),
        pytest.param("day_only", "LA", True, DAY, [[1., TWILIGHT_WEIGHT], [0., 0.]],
                     id="day_only-LA-sza"),
        pytest.param("night_only", "RGBA", True, NIGHT, [[0., 1 - TWILIGHT_WEIGHT], [1., 1.]],
                     id="night_only-RGBA-sza"),
    ],
)
def test_day_night_compositor_single_side_with_alpha(day_night, mode, sza_given, values, exp_weight):
    """Test that the alpha band is scaled by the weight of the kept side and the color bands are untouched.

    The weight is NaN where the input is missing.
    """
    image = _make_image(mode, values)
    inputs = (image, _make_sza()) if sza_given else (image,)
    with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
        comp = DayNightCompositor(name="dn_test", day_night=day_night, include_alpha=True)
        res_dask = comp(inputs)
        res = res_dask.compute()
    exp_mode = mode if mode.endswith("A") else mode + "A"
    assert list(res["bands"].values) == list(exp_mode)
    assert res_dask.dtype == res.dtype
    assert res.dtype == np.float32
    input_alpha = image.sel(bands="A").values if mode.endswith("A") else 1.
    np.testing.assert_allclose(res.sel(bands="A").values, input_alpha * np.array(exp_weight), rtol=1e-6)
    for band in exp_mode[:-1]:
        np.testing.assert_allclose(res.sel(bands=band).values, values)


@mock.patch("satpy.composites.fill.enhance2dataset", _enhance2dataset)
@pytest.mark.parametrize(
    ("day_night", "mode", "sza_given", "values", "exp_weight"),
    [
        pytest.param("day_only", "RGB", True, DAY, [[1., TWILIGHT_WEIGHT], [np.nan, np.nan]],
                     id="day_only-RGB-sza"),
        pytest.param("day_only", "RGB", False, [[np.nan, 0.2], [0.3, 0.4]], [[1., 1.], [1., 1.]],
                     id="day_only-RGB-computed-missing-on-day-side"),
        pytest.param("night_only", "RGB", True, NIGHT, [[np.nan, 1 - TWILIGHT_WEIGHT], [1., 1.]],
                     id="night_only-RGB-sza"),
        pytest.param("night_only", "RGB", False, NIGHT, [[np.nan, np.nan], [np.nan, np.nan]],
                     id="night_only-RGB-computed"),
        pytest.param("day_only", "LA", True, DAY, [[1., TWILIGHT_WEIGHT], [np.nan, np.nan]],
                     id="day_only-LA-sza"),
        pytest.param("night_only", "RGBA", True, NIGHT, [[np.nan, 1 - TWILIGHT_WEIGHT], [1., 1.]],
                     id="night_only-RGBA-sza"),
    ],
)
def test_day_night_compositor_single_side_without_alpha(day_night, mode, sza_given, values, exp_weight):
    """Test that every band, including an input alpha band, is scaled by the weight of the kept side.

    The weight is NaN on the discarded side.
    """
    image = _make_image(mode, values)
    inputs = (image, _make_sza()) if sza_given else (image,)
    with dask.config.set(scheduler=CustomScheduler(max_computes=1)):
        comp = DayNightCompositor(name="dn_test", day_night=day_night, include_alpha=False)
        res_dask = comp(inputs)
        res = res_dask.compute()
    assert list(res["bands"].values) == list(mode)
    assert res_dask.dtype == res.dtype
    assert res.dtype == np.float32
    for band in mode:
        np.testing.assert_allclose(res.sel(bands=band).values,
                                   image.sel(bands=band).values * np.array(exp_weight), rtol=1e-6)


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
