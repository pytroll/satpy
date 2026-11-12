
"""Tests for modifiers in modifiers/__init__.py."""
import datetime as dt

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition, StackedAreaDefinition
from pytest_lazy_fixtures import lf as lazy_fixture

import satpy


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
        da.from_array([[80.0, 87.0], [89.0, 93.0]], chunks=2),
        attrs={"area": _sunz_area_def()},
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [0, 1]},
    )
    return sza


def call_sunz_modifier(comp, data_arr, sunz_sza=None, dtype=None, dtype_sunz=None):
    """Helper function for running sunz modifiers."""
    if dtype is not None:
        data_arr = data_arr.astype(dtype)

    if sunz_sza is not None and dtype_sunz is not None:
        sunz_sza = sunz_sza.astype(dtype_sunz)

    info = {"test_attr": "test"}
    if sunz_sza is not None:
        info["optional_datasets"] = [sunz_sza]

    return comp((data_arr,), **info)


def assert_sunz_modifier_result(res, expected, data_arr, rtol=1e-5):
    """Helper function for asserting sunz modifier results."""
    np.testing.assert_allclose(res.values, expected, rtol=rtol)
    assert type(res.data) is type(data_arr.data)


def assert_sunz_modifier_dtype(res, dtype):
    """Helper function for asserting sunz modifier dtype."""
    assert res.dtype == dtype
    assert res.values.dtype == dtype


class TestSunZenithCorrectorBase:
    """Test Sun zenith angle corrector base class."""
    def test_missing_method(self):
        """Test that a ValueError is raised if the 'method' attribute is not defined."""
        from satpy.modifiers.geometry import SunZenithCorrectorBase

        class DummyCorrector(SunZenithCorrectorBase):
            pass

        with pytest.raises(ValueError, match="DummyCorrector must define a 'method'"):
            DummyCorrector(name="dummy_test")


class TestSunZenithCorrector:
    """Test case for the standard Sun zenith angle corrector."""

    @pytest.mark.parametrize("as_32bit", [False, True])
    def test_default_with_sza_not_provided(self, sunz_ds1, as_32bit):
        """Test default limits when SZA isn't provided."""
        from satpy.modifiers.geometry import SunZenithCorrector
        if as_32bit:
            sunz_ds1 = sunz_ds1.astype(np.float32)
        comp = SunZenithCorrector(name="sza_test", modifiers=tuple())
        expected = np.array([[22.401667, 22.31777], [22.437503, 22.353533]])
        with pytest.warns(UserWarning, match="removed in Satpy v1.0"):
            res = call_sunz_modifier(comp, sunz_ds1)
        assert_sunz_modifier_result(res, expected, sunz_ds1)
        assert "y" in res.coords
        assert "x" in res.coords

        ds1 = sunz_ds1.copy().drop_vars(("y", "x"))

        with pytest.warns(UserWarning, match="removed in Satpy v1.0"):
            res = call_sunz_modifier(comp, ds1)
        res_np = res.compute()
        np.testing.assert_allclose(res_np.values, expected, rtol=1e-5)
        assert res.dtype == res_np.dtype
        assert "y" not in res.coords
        assert "x" not in res.coords
        if as_32bit:
            assert res.dtype == np.float32

    @pytest.mark.parametrize("dtype_sunz", [np.float32, np.float64])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("data_arr", [lazy_fixture("sunz_ds1"), lazy_fixture("sunz_ds1_stacked")])
    def test_default_with_sza_provided(self, data_arr, sunz_sza, dtype, dtype_sunz):
        """Test default limits."""
        from satpy.modifiers.geometry import SunZenithCorrector
        comp = SunZenithCorrector(name="sza_test", modifiers=tuple())
        # Update expected to values of test_default_new_with_sza_provided in Satpy v1.0
        expected = np.array([[5.758770, 19.107323], [23.133712, 6.372368]], dtype=dtype)
        with pytest.warns(UserWarning, match="removed in Satpy v1.0"):
            res = call_sunz_modifier(comp, data_arr, sunz_sza, dtype, dtype_sunz)
        assert_sunz_modifier_result(res, expected, data_arr)
        assert_sunz_modifier_dtype(res, dtype)


    @pytest.mark.parametrize("dtype_sunz", [np.float32, np.float64])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("data_arr", [lazy_fixture("sunz_ds1"), lazy_fixture("sunz_ds1_stacked")])
    def test_default_legacy_with_sza_provided(self, data_arr, sunz_sza, dtype, dtype_sunz):
        """Test default limits using legacy configuration."""
        with satpy.config.set(use_legacy_sunz_correction=True):
            from satpy.modifiers.geometry import SunZenithCorrector
            comp = SunZenithCorrector(name="sza_test", modifiers=tuple())
            expected = np.array([[5.758770, 19.107323], [23.133712, 6.372368]], dtype=dtype)
            with pytest.warns(UserWarning, match="removed in Satpy v1.0"):
                res = call_sunz_modifier(comp, data_arr, sunz_sza, dtype, dtype_sunz)
            assert_sunz_modifier_result(res, expected, data_arr)
            assert_sunz_modifier_dtype(res, dtype)

    @pytest.mark.parametrize("dtype_sunz", [np.float32, np.float64])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("data_arr", [lazy_fixture("sunz_ds1"), lazy_fixture("sunz_ds1_stacked")])
    def test_default_new_with_sza_provided(self, data_arr, sunz_sza, dtype, dtype_sunz):
        """Test default limits using new configuration."""
        with satpy.config.set(use_legacy_sunz_correction=False):
            from satpy.modifiers.geometry import SunZenithCorrector
            comp = SunZenithCorrector(name="sza_test", modifiers=tuple())
            expected = np.array([[5.758770, 19.107323], [57.298689, 0.0]], dtype=dtype)
            res = call_sunz_modifier(comp, data_arr, sunz_sza, dtype, dtype_sunz)
            assert_sunz_modifier_result(res, expected, data_arr)
            assert_sunz_modifier_dtype(res, dtype)

    @pytest.mark.parametrize("dtype_sunz", [np.float32, np.float64])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("data_arr", [lazy_fixture("sunz_ds1"), lazy_fixture("sunz_ds1_stacked")])
    def test_standard_with_sza_provided(self, data_arr, sunz_sza, dtype, dtype_sunz):
        """Test correction_limit=None and max_sza=None."""
        from satpy.modifiers.geometry import SunZenithCorrector
        comp = SunZenithCorrector(name="sza_test", modifiers=tuple(), correction_limit=None, max_sza=None)
        expected = np.array([[5.758770, 19.107323], [57.298689, 0.0]], dtype=dtype)
        res = call_sunz_modifier(comp, data_arr, sunz_sza, dtype, dtype_sunz)
        assert_sunz_modifier_result(res, expected, data_arr)
        assert_sunz_modifier_dtype(res, dtype)

    @pytest.mark.parametrize("dtype_sunz", [np.float32, np.float64])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("data_arr", [lazy_fixture("sunz_ds1"), lazy_fixture("sunz_ds1_stacked")])
    def test_custom_max_sza_with_sza_provided(self, data_arr, sunz_sza, dtype, dtype_sunz):
        """Test custom max_sza (and correction_limit=None)."""
        from satpy.modifiers.geometry import SunZenithCorrector
        comp = SunZenithCorrector(name="sza_test", modifiers=tuple(), correction_limit=None, max_sza=88)
        expected = np.array([[5.758770, 19.107323], [0.0, 0.0]], dtype=dtype)
        res = call_sunz_modifier(comp, data_arr, sunz_sza, dtype, dtype_sunz)
        assert_sunz_modifier_result(res, expected, data_arr)
        assert_sunz_modifier_dtype(res, dtype)

    @pytest.mark.parametrize("dtype_sunz", [np.float32, np.float64])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("data_arr", [lazy_fixture("sunz_ds1"), lazy_fixture("sunz_ds1_stacked")])
    def test_custom_correction_limit_with_sza_provided(self, data_arr, sunz_sza, dtype, dtype_sunz):
        """Test custom correction_limit (and max_sza=None)."""
        from satpy.modifiers.geometry import SunZenithCorrector
        comp = SunZenithCorrector(name="sza_test", modifiers=tuple(), correction_limit=88, max_sza=None)
        expected = np.array([[5.758770, 19.107323], [28.653708, 28.653708]], dtype=dtype)
        res = call_sunz_modifier(comp, data_arr, sunz_sza, dtype, dtype_sunz)
        assert_sunz_modifier_result(res, expected, data_arr)
        assert_sunz_modifier_dtype(res, dtype)

    @pytest.mark.parametrize("dtype_sunz", [np.float32, np.float64])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("data_arr", [lazy_fixture("sunz_ds1"), lazy_fixture("sunz_ds1_stacked")])
    def test_custom_max_sza_and_correction_limit_with_sza_provided(self, data_arr, sunz_sza, dtype, dtype_sunz):
        """Test custom correction_limit and max_sza."""
        from satpy.modifiers.geometry import SunZenithCorrector
        comp = SunZenithCorrector(name="sza_test", modifiers=tuple(), correction_limit=88, max_sza=95)
        expected = np.array([[5.758770, 19.107323], [23.133712, 6.372368]], dtype=dtype)
        res = call_sunz_modifier(comp, data_arr, sunz_sza, dtype, dtype_sunz)
        assert_sunz_modifier_result(res, expected, data_arr)
        assert_sunz_modifier_dtype(res, dtype)

    @pytest.mark.parametrize("dtype_sunz", [np.float32, np.float64])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("data_arr", [lazy_fixture("sunz_ds1"), lazy_fixture("sunz_ds1_stacked")])
    def test_double_correction(self, data_arr, sunz_sza, dtype, dtype_sunz):
        """Test that no double sunz correction is applied."""
        from satpy.modifiers.geometry import SunZenithCorrector
        comp = SunZenithCorrector(name="sza_test", modifiers=tuple())
        data_arr = data_arr.copy()
        data_arr.attrs["modifiers"] = ("effective_solar_pathlength_corrected",)
        expected = data_arr.values
        res = call_sunz_modifier(comp, data_arr, sunz_sza, dtype, dtype_sunz)
        assert_sunz_modifier_result(res, expected, data_arr)
        assert_sunz_modifier_dtype(res, dtype)

    @pytest.mark.parametrize("dtype_sunz", [np.float32, np.float64])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("data_arr", [lazy_fixture("sunz_ds1"), lazy_fixture("sunz_ds1_stacked")])
    def test_invalid_max_sza_and_correction_limit_with_sza_provided(self, data_arr, sunz_sza, dtype, dtype_sunz):
        """Test with correction_limit > max_sza."""
        from satpy.modifiers.geometry import SunZenithCorrector
        comp = SunZenithCorrector(name="sza_test", modifiers=tuple(), correction_limit=88, max_sza=80)
        with pytest.raises(ValueError, match="`max_sza` must be larger than `correction_limit`"):
            call_sunz_modifier(comp, data_arr, sunz_sza, dtype, dtype_sunz)

    def test_incompatible_areas(self, sunz_ds2, sunz_sza):
        """Test sunz correction on incompatible areas."""
        from satpy.composites.core import IncompatibleAreas
        from satpy.modifiers.geometry import SunZenithCorrector
        comp = SunZenithCorrector(name="sza_test", modifiers=tuple())
        with pytest.raises(IncompatibleAreas):
            call_sunz_modifier(comp, sunz_ds2, sunz_sza)


class TestEffectiveSolarPathLengthCorrector:
    """Test case for the effective solar path length corrector."""

    @pytest.mark.parametrize("as_32bit", [False, True])
    def test_with_sza_not_provided(self, sunz_ds1, as_32bit):
        """Test when SZA isn't provided."""
        from satpy.modifiers.geometry import EffectiveSolarPathLengthCorrector
        if as_32bit:
            sunz_ds1 = sunz_ds1.astype(np.float32)
        comp = EffectiveSolarPathLengthCorrector(name="sza_test", modifiers=tuple())
        expected = np.array([[22.458680, 22.512698], [22.435495, 22.489718]])
        res = call_sunz_modifier(comp, sunz_ds1)
        assert_sunz_modifier_result(res, expected, sunz_ds1)
        assert "y" in res.coords
        assert "x" in res.coords

        ds1 = sunz_ds1.copy().drop_vars(("y", "x"))
        res = call_sunz_modifier(comp, ds1)
        res_np = res.compute()
        np.testing.assert_allclose(res_np.values, expected, rtol=1e-5)
        assert res.dtype == res_np.dtype
        assert "y" not in res.coords
        assert "x" not in res.coords
        if as_32bit:
            assert res.dtype == np.float32

    @pytest.mark.parametrize("dtype_sunz", [np.float32, np.float64])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("data_arr", [lazy_fixture("sunz_ds1"), lazy_fixture("sunz_ds1_stacked")])
    def test_with_sza_provided(self, data_arr, sunz_sza, dtype, dtype_sunz):
        """Test when SZA is provided."""
        from satpy.modifiers.geometry import EffectiveSolarPathLengthCorrector
        comp = EffectiveSolarPathLengthCorrector(name="sza_test", modifiers=tuple())
        expected = np.array([[5.595989, 14.823307], [21.973671, 16.988299]], dtype=dtype)
        res = call_sunz_modifier(comp, data_arr, sunz_sza, dtype, dtype_sunz)
        assert_sunz_modifier_result(res, expected, data_arr)
        assert_sunz_modifier_dtype(res, dtype)

    @pytest.mark.parametrize("dtype_sunz", [np.float32, np.float64])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("data_arr", [lazy_fixture("sunz_ds1"), lazy_fixture("sunz_ds1_stacked")])
    def test_double_correction(self, data_arr, sunz_sza, dtype, dtype_sunz):
        """Test that no double sunz correction is applied."""
        from satpy.modifiers.geometry import EffectiveSolarPathLengthCorrector
        comp = EffectiveSolarPathLengthCorrector(name="sza_test", modifiers=tuple())
        data_arr = data_arr.copy()
        data_arr.attrs["modifiers"] = ("sunz_corrected",)
        expected = data_arr.values
        res = call_sunz_modifier(comp, data_arr, sunz_sza, dtype, dtype_sunz)
        assert_sunz_modifier_result(res, expected, data_arr)
        assert_sunz_modifier_dtype(res, dtype)

    def test_with_deprecated_correction_limit(self):
        """Test with deprecated correction_limit."""
        from satpy.modifiers.geometry import EffectiveSolarPathLengthCorrector
        with pytest.warns(UserWarning, match="deprecated"):
            EffectiveSolarPathLengthCorrector(name="sza_test", modifiers=tuple(), correction_limit=88)

    def test_with_deprecated_sza_max(self):
        """Test with deprecated max_sza."""
        from satpy.modifiers.geometry import EffectiveSolarPathLengthCorrector
        with pytest.warns(UserWarning, match="deprecated"):
            EffectiveSolarPathLengthCorrector(name="sza_test", modifiers=tuple(), max_sza=95)

    def test_incompatible_areas(self, sunz_ds2, sunz_sza):
        """Test with incompatible areas."""
        from satpy.composites.core import IncompatibleAreas
        from satpy.modifiers.geometry import EffectiveSolarPathLengthCorrector
        comp = EffectiveSolarPathLengthCorrector(name="sza_test", modifiers=tuple())
        with pytest.raises(IncompatibleAreas):
            call_sunz_modifier(comp, sunz_ds2, sunz_sza)

    @pytest.mark.parametrize("dtype_sunz", [np.float32, np.float64])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("data_arr", [lazy_fixture("sunz_ds1"), lazy_fixture("sunz_ds1_stacked")])
    def test_old_import(self, data_arr, sunz_sza, dtype, dtype_sunz):
        """Test that the deprecated import of atmospheric_path_length_correction works."""
        from satpy.utils import atmospheric_path_length_correction
        data_arr = data_arr.astype(dtype)
        sunz_sza = sunz_sza.astype(dtype_sunz)
        cos_zen = np.cos(np.deg2rad(sunz_sza))
        expected = np.array([[5.595989, 14.823307], [21.973671, 16.988299]])
        with pytest.warns(UserWarning, match="has been moved to"):
            with pytest.warns(UserWarning, match="deprecated"):
                res = atmospheric_path_length_correction(data_arr, cos_zen)
        assert_sunz_modifier_result(res, expected, data_arr)
        assert_sunz_modifier_dtype(res, dtype)


class TestSunZenithReducer:
    """Test case for the sun zenith reducer."""

    @classmethod
    def setup_class(cls):
        """Initialze SunZenithReducer classes that shall be tested."""
        from satpy.modifiers.geometry import SunZenithReducer
        cls.default = SunZenithReducer(name="sza_reduction_test_default", modifiers=tuple())
        cls.custom = SunZenithReducer(name="sza_reduction_test_custom", modifiers=tuple(),
                                      correction_limit=70, max_sza=95, strength=3.0)

    @pytest.mark.parametrize("dtype_sunz", [np.float32, np.float64])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_default_settings(self, sunz_ds1, sunz_sza, dtype, dtype_sunz):
        """Test default settings with sza data available."""
        expected = np.array([[1.0, 0.176790], [0.036095, 0.0]], dtype=dtype)
        res = call_sunz_modifier(self.default, sunz_ds1, sunz_sza, dtype, dtype_sunz)
        assert_sunz_modifier_result(res, expected, sunz_ds1, rtol=2e-5)
        assert_sunz_modifier_dtype(res, dtype)

    @pytest.mark.parametrize("dtype_sunz", [np.float32, np.float64])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_custom_settings(self, sunz_ds1, sunz_sza, dtype, dtype_sunz):
        """Test custom settings with sza data available."""
        expected = np.array([[5.436207e-01, 3.657017e-02], [1.143065e-02, 2.450100e-04]], dtype=dtype)
        res = call_sunz_modifier(self.custom, sunz_ds1, sunz_sza, dtype, dtype_sunz)
        assert_sunz_modifier_result(res, expected, sunz_ds1, rtol=2e-5)
        assert_sunz_modifier_dtype(res, dtype)

    def test_invalid_max_sza(self):
        """Test invalid max_sza with sza data available."""
        from satpy.modifiers.geometry import SunZenithReducer
        with pytest.raises(ValueError, match="`max_sza` must be defined when using the SunZenithReducer."):
            SunZenithReducer(name="sza_reduction_test_invalid", modifiers=tuple(), max_sza=None)
