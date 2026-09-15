
"""Tests for modifiers in modifiers/__init__.py."""
import datetime as dt

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition, StackedAreaDefinition
from pytest_lazy_fixtures import lf as lazy_fixture


def _sunz_area_def():
    """Get fake area for testing sunz generation."""
    area = AreaDefinition("test", "test", "test",
                          {"proj": "merc"}, 2, 2,
                          (-2000, -2000, 2000, 2000))
    return area


def _sunz_bigger_area_def():
    """Get area that is twice the size of 'sunz_area_def'."""
    bigger_area = AreaDefinition("test", "test", "test",
                                 {"proj": "merc"}, 4, 4,
                                 (-2000, -2000, 2000, 2000))
    return bigger_area


def _sunz_stacked_area_def():
    """Get fake stacked area for testing sunz generation."""
    area1 = AreaDefinition("test", "test", "test",
                           {"proj": "merc"}, 2, 1,
                           (-2000, 0, 2000, 2000))
    area2 = AreaDefinition("test", "test", "test",
                           {"proj": "merc"}, 2, 1,
                           (-2000, -2000, 2000, 0))
    return StackedAreaDefinition(area1, area2)


def _shared_sunz_attrs(area_def):
    attrs = {"area": area_def,
             "start_time": dt.datetime(2018, 1, 1, 18),
             "modifiers": tuple(),
             "name": "test_vis"}
    return attrs


def _get_ds1(attrs):
    ds1 = xr.DataArray(da.ones((2, 2), chunks=2, dtype=np.float64),
                       attrs=attrs, dims=("y", "x"),
                       coords={"y": [0, 1], "x": [0, 1]})
    return ds1


@pytest.fixture(scope="module")
def sunz_ds1():
    """Generate fake dataset for sunz tests."""
    attrs = _shared_sunz_attrs(_sunz_area_def())
    return _get_ds1(attrs)


@pytest.fixture(scope="module")
def sunz_ds1_stacked():
    """Generate fake dataset for sunz tests."""
    attrs = _shared_sunz_attrs(_sunz_stacked_area_def())
    return _get_ds1(attrs)


@pytest.fixture(scope="module")
def sunz_ds2():
    """Generate larger fake dataset for sunz tests."""
    attrs = _shared_sunz_attrs(_sunz_bigger_area_def())
    ds2 = xr.DataArray(da.ones((4, 4), chunks=2, dtype=np.float64),
                       attrs=attrs, dims=("y", "x"),
                       coords={"y": [0, 0.5, 1, 1.5], "x": [0, 0.5, 1, 1.5]})
    return ds2


@pytest.fixture(scope="module")
def sunz_sza():
    """Generate fake solar zenith angle data array for testing."""
    sza = xr.DataArray(
        np.rad2deg(np.arccos(da.from_array([[0.0149581333, 0.0146694376], [0.0150812684, 0.0147925727]],
                                           chunks=2))),
        attrs={"area": _sunz_area_def()},
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]},
    )
    return sza


class TestSunZenithCorrector:
    """Test case for the zenith corrector."""

    @pytest.mark.parametrize("as_32bit", [False, True])
    def test_basic_default_not_provided(self, sunz_ds1, as_32bit):
        """Test default limits when SZA isn't provided."""
        from satpy.modifiers.geometry import SunZenithCorrector

        if as_32bit:
            sunz_ds1 = sunz_ds1.astype(np.float32)
        comp = SunZenithCorrector(name="sza_test", modifiers=tuple())
        res = comp((sunz_ds1,), test_attr="test")
        np.testing.assert_allclose(res.values, np.array([[22.401667, 22.31777], [22.437503, 22.353533]]),
                                   rtol=1e-6)
        assert "y" in res.coords
        assert "x" in res.coords
        ds1 = sunz_ds1.copy().drop_vars(("y", "x"))
        res = comp((ds1,), test_attr="test")
        res_np = res.compute()
        np.testing.assert_allclose(res_np.values, np.array([[22.401667, 22.31777], [22.437503, 22.353533]]),
                                   rtol=1e-6)
        assert res.dtype == res_np.dtype
        assert "y" not in res.coords
        assert "x" not in res.coords
        if as_32bit:
            assert res.dtype == np.float32

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_basic_lims_not_provided(self, sunz_ds1, dtype):
        """Test custom limits when SZA isn't provided."""
        from satpy.modifiers.geometry import SunZenithCorrector
        comp = SunZenithCorrector(name="sza_test", modifiers=tuple(), correction_limit=90)
        res = comp((sunz_ds1.astype(dtype),), test_attr="test")
        expected = np.array([[66.853262, 68.168939], [66.30742, 67.601493]], dtype=dtype)
        values = res.values
        np.testing.assert_allclose(values, expected, rtol=1e-5)
        assert res.dtype == dtype
        assert values.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("data_arr", [lazy_fixture("sunz_ds1"), lazy_fixture("sunz_ds1_stacked")])
    def test_basic_default_provided(self, data_arr, sunz_sza, dtype):
        """Test default limits when SZA is provided."""
        from satpy.modifiers.geometry import SunZenithCorrector
        comp = SunZenithCorrector(name="sza_test", modifiers=tuple())
        res = comp((data_arr.astype(dtype), sunz_sza.astype(dtype)), test_attr="test")
        expected = np.array([[22.401667, 22.31777], [22.437503, 22.353533]], dtype=dtype)
        values = res.values
        np.testing.assert_allclose(values, expected)
        assert res.dtype == dtype
        assert values.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("data_arr", [lazy_fixture("sunz_ds1"), lazy_fixture("sunz_ds1_stacked")])
    def test_basic_lims_provided(self, data_arr, sunz_sza, dtype):
        """Test custom limits when SZA is provided."""
        from satpy.modifiers.geometry import SunZenithCorrector
        comp = SunZenithCorrector(name="sza_test", modifiers=tuple(), correction_limit=90)
        res = comp((data_arr.astype(dtype), sunz_sza.astype(dtype)), test_attr="test")
        expected = np.array([[66.853262, 68.168939], [66.30742, 67.601493]], dtype=dtype)
        values = res.values
        np.testing.assert_allclose(values, expected, rtol=1e-5)
        assert res.dtype == dtype
        assert values.dtype == dtype

    def test_imcompatible_areas(self, sunz_ds2, sunz_sza):
        """Test sunz correction on incompatible areas."""
        from satpy.composites.core import IncompatibleAreas
        from satpy.modifiers.geometry import SunZenithCorrector
        comp = SunZenithCorrector(name="sza_test", modifiers=tuple(), correction_limit=90)
        with pytest.raises(IncompatibleAreas):
            comp((sunz_ds2, sunz_sza), test_attr="test")


class TestSunZenithReducer:
    """Test case for the sun zenith reducer."""

    @classmethod
    def setup_class(cls):
        """Initialze SunZenithReducer classes that shall be tested."""
        from satpy.modifiers.geometry import SunZenithReducer
        cls.default = SunZenithReducer(name="sza_reduction_test_default", modifiers=tuple())
        cls.custom = SunZenithReducer(name="sza_reduction_test_custom", modifiers=tuple(),
                                      correction_limit=70, max_sza=95, strength=3.0)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_default_settings(self, sunz_ds1, sunz_sza, dtype):
        """Test default settings with sza data available."""
        res = self.default((sunz_ds1.astype(dtype), sunz_sza.astype(dtype)), test_attr="test")
        expected = np.array([[0.02916261, 0.02839063], [0.02949383, 0.02871911]], dtype=dtype)
        assert res.dtype == dtype
        values = res.values
        assert values.dtype == dtype
        np.testing.assert_allclose(values,
                                   expected,
                                   rtol=2e-5)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_custom_settings(self, sunz_ds1, sunz_sza, dtype):
        """Test custom settings with sza data available."""
        res = self.custom((sunz_ds1.astype(dtype), sunz_sza.astype(dtype)), test_attr="test")
        expected = np.array([[0.01041319, 0.01030033], [0.01046164, 0.01034834]], dtype=dtype)
        assert res.dtype == dtype
        values = res.values
        assert values.dtype == dtype
        np.testing.assert_allclose(values,
                                   expected,
                                   rtol=1e-5)

    def test_invalid_max_sza(self, sunz_ds1, sunz_sza):
        """Test invalid max_sza with sza data available."""
        from satpy.modifiers.geometry import SunZenithReducer
        with pytest.raises(ValueError, match="`max_sza` must be defined when using the SunZenithReducer."):
            SunZenithReducer(name="sza_reduction_test_invalid", modifiers=tuple(), max_sza=None)

