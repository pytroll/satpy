#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Satpy developers
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
"""Tests for modifiers in modifiers/__init__.py."""
import unittest
from datetime import datetime
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition, StackedAreaDefinition
from pytest_lazyfixture import lazy_fixture


def _sunz_area_def():
    """Get fake area for testing sunz generation."""
    area = AreaDefinition('test', 'test', 'test',
                          {'proj': 'merc'}, 2, 2,
                          (-2000, -2000, 2000, 2000))
    return area


def _sunz_bigger_area_def():
    """Get area that is twice the size of 'sunz_area_def'."""
    bigger_area = AreaDefinition('test', 'test', 'test',
                                 {'proj': 'merc'}, 4, 4,
                                 (-2000, -2000, 2000, 2000))
    return bigger_area


def _sunz_stacked_area_def():
    """Get fake stacked area for testing sunz generation."""
    area1 = AreaDefinition('test', 'test', 'test',
                           {'proj': 'merc'}, 2, 1,
                           (-2000, 0, 2000, 2000))
    area2 = AreaDefinition('test', 'test', 'test',
                           {'proj': 'merc'}, 2, 1,
                           (-2000, -2000, 2000, 0))
    return StackedAreaDefinition(area1, area2)


def _shared_sunz_attrs(area_def):
    attrs = {'area': area_def,
             'start_time': datetime(2018, 1, 1, 18),
             'modifiers': tuple(),
             'name': 'test_vis'}
    return attrs


def _get_ds1(attrs):
    ds1 = xr.DataArray(da.ones((2, 2), chunks=2, dtype=np.float64),
                       attrs=attrs, dims=('y', 'x'),
                       coords={'y': [0, 1], 'x': [0, 1]})
    return ds1


@pytest.fixture(scope="session")
def sunz_ds1():
    """Generate fake dataset for sunz tests."""
    attrs = _shared_sunz_attrs(_sunz_area_def())
    return _get_ds1(attrs)


@pytest.fixture(scope="session")
def sunz_ds1_stacked():
    """Generate fake dataset for sunz tests."""
    attrs = _shared_sunz_attrs(_sunz_stacked_area_def())
    return _get_ds1(attrs)


@pytest.fixture(scope="session")
def sunz_ds2():
    """Generate larger fake dataset for sunz tests."""
    attrs = _shared_sunz_attrs(_sunz_bigger_area_def())
    ds2 = xr.DataArray(da.ones((4, 4), chunks=2, dtype=np.float64),
                       attrs=attrs, dims=('y', 'x'),
                       coords={'y': [0, 0.5, 1, 1.5], 'x': [0, 0.5, 1, 1.5]})
    return ds2


@pytest.fixture(scope="session")
def sunz_sza():
    """Generate fake solar zenith angle data array for testing."""
    sza = xr.DataArray(
        np.rad2deg(np.arccos(da.from_array([[0.0149581333, 0.0146694376], [0.0150812684, 0.0147925727]],
                                           chunks=2))),
        attrs={'area': _sunz_area_def()},
        dims=('y', 'x'),
        coords={'y': [0, 1], 'x': [0, 1]},
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
        comp = SunZenithCorrector(name='sza_test', modifiers=tuple())
        res = comp((sunz_ds1,), test_attr='test')
        np.testing.assert_allclose(res.values, np.array([[22.401667, 22.31777], [22.437503, 22.353533]]))
        assert 'y' in res.coords
        assert 'x' in res.coords
        ds1 = sunz_ds1.copy().drop_vars(('y', 'x'))
        res = comp((ds1,), test_attr='test')
        res_np = res.compute()
        np.testing.assert_allclose(res_np.values, np.array([[22.401667, 22.31777], [22.437503, 22.353533]]))
        assert res.dtype == res_np.dtype
        assert 'y' not in res.coords
        assert 'x' not in res.coords

    def test_basic_lims_not_provided(self, sunz_ds1):
        """Test custom limits when SZA isn't provided."""
        from satpy.modifiers.geometry import SunZenithCorrector
        comp = SunZenithCorrector(name='sza_test', modifiers=tuple(), correction_limit=90)
        res = comp((sunz_ds1,), test_attr='test')
        np.testing.assert_allclose(res.values, np.array([[66.853262, 68.168939], [66.30742, 67.601493]]))

    @pytest.mark.parametrize("data_arr", [lazy_fixture("sunz_ds1"), lazy_fixture("sunz_ds1_stacked")])
    def test_basic_default_provided(self, data_arr, sunz_sza):
        """Test default limits when SZA is provided."""
        from satpy.modifiers.geometry import SunZenithCorrector
        comp = SunZenithCorrector(name='sza_test', modifiers=tuple())
        res = comp((data_arr, sunz_sza), test_attr='test')
        np.testing.assert_allclose(res.values, np.array([[22.401667, 22.31777], [22.437503, 22.353533]]))

    @pytest.mark.parametrize("data_arr", [lazy_fixture("sunz_ds1"), lazy_fixture("sunz_ds1_stacked")])
    def test_basic_lims_provided(self, data_arr, sunz_sza):
        """Test custom limits when SZA is provided."""
        from satpy.modifiers.geometry import SunZenithCorrector
        comp = SunZenithCorrector(name='sza_test', modifiers=tuple(), correction_limit=90)
        res = comp((data_arr, sunz_sza), test_attr='test')
        np.testing.assert_allclose(res.values, np.array([[66.853262, 68.168939], [66.30742, 67.601493]]))

    def test_imcompatible_areas(self, sunz_ds2, sunz_sza):
        """Test sunz correction on incompatible areas."""
        from satpy.composites import IncompatibleAreas
        from satpy.modifiers.geometry import SunZenithCorrector
        comp = SunZenithCorrector(name='sza_test', modifiers=tuple(), correction_limit=90)
        with pytest.raises(IncompatibleAreas):
            comp((sunz_ds2, sunz_sza), test_attr='test')


class TestSunZenithReducer:
    """Test case for the sun zenith reducer."""

    @classmethod
    def setup_class(cls):
        """Initialze SunZenithReducer classes that shall be tested."""
        from satpy.modifiers.geometry import SunZenithReducer
        cls.default = SunZenithReducer(name='sza_reduction_test_default', modifiers=tuple())
        cls.custom = SunZenithReducer(name='sza_reduction_test_custom', modifiers=tuple(),
                                      correction_limit=70, max_sza=95, strength=3.0)

    def test_default_settings(self, sunz_ds1, sunz_sza):
        """Test default settings with sza data available."""
        res = self.default((sunz_ds1, sunz_sza), test_attr='test')
        np.testing.assert_allclose(res.values,
                                   np.array([[0.00242814, 0.00235669], [0.00245885, 0.00238707]]),
                                   rtol=1e-5)

    def test_custom_settings(self, sunz_ds1, sunz_sza):
        """Test custom settings with sza data available."""
        res = self.custom((sunz_ds1, sunz_sza), test_attr='test')
        np.testing.assert_allclose(res.values,
                                   np.array([[0.01041319, 0.01030033], [0.01046164, 0.01034834]]),
                                   rtol=1e-5)

    def test_invalid_max_sza(self, sunz_ds1, sunz_sza):
        """Test invalid max_sza with sza data available."""
        from satpy.modifiers.geometry import SunZenithReducer
        with pytest.raises(ValueError):
            SunZenithReducer(name='sza_reduction_test_invalid', modifiers=tuple(), max_sza=None)


class TestNIRReflectance(unittest.TestCase):
    """Test NIR reflectance compositor."""

    def setUp(self):
        """Set up the test case for the NIRReflectance compositor."""
        self.get_lonlats = mock.MagicMock()
        self.lons, self.lats = 1, 2
        self.get_lonlats.return_value = (self.lons, self.lats)
        area = mock.MagicMock(get_lonlats=self.get_lonlats)

        self.start_time = 1
        self.metadata = {'platform_name': 'Meteosat-11',
                         'sensor': 'seviri',
                         'name': 'IR_039',
                         'area': area,
                         'start_time': self.start_time}

        nir_arr = np.random.random((2, 2))
        self.nir = xr.DataArray(da.from_array(nir_arr), dims=['y', 'x'])
        self.nir.attrs.update(self.metadata)

        ir_arr = 100 * np.random.random((2, 2))
        self.ir_ = xr.DataArray(da.from_array(ir_arr), dims=['y', 'x'])
        self.ir_.attrs['area'] = area

        self.sunz_arr = 100 * np.random.random((2, 2))
        self.sunz = xr.DataArray(da.from_array(self.sunz_arr), dims=['y', 'x'])
        self.sunz.attrs['standard_name'] = 'solar_zenith_angle'
        self.sunz.attrs['area'] = area
        self.da_sunz = da.from_array(self.sunz_arr)

        refl_arr = np.random.random((2, 2))
        self.refl = da.from_array(refl_arr)
        self.refl_with_co2 = da.from_array(np.random.random((2, 2)))
        self.refl_from_tbs = mock.MagicMock()
        self.refl_from_tbs.side_effect = self.fake_refl_from_tbs

    def fake_refl_from_tbs(self, sun_zenith, da_nir, da_tb11, tb_ir_co2=None):
        """Fake refl_from_tbs."""
        del sun_zenith, da_nir, da_tb11
        if tb_ir_co2 is not None:
            return self.refl_with_co2
        return self.refl

    @mock.patch('satpy.modifiers.spectral.sun_zenith_angle')
    @mock.patch('satpy.modifiers.NIRReflectance.apply_modifier_info')
    @mock.patch('satpy.modifiers.spectral.Calculator')
    def test_provide_sunz_no_co2(self, calculator, apply_modifier_info, sza):
        """Test NIR reflectance compositor provided only sunz."""
        calculator.return_value = mock.MagicMock(
            reflectance_from_tbs=self.refl_from_tbs)
        sza.return_value = self.da_sunz
        from satpy.modifiers.spectral import NIRReflectance

        comp = NIRReflectance(name='test')
        info = {'modifiers': None}
        res = comp([self.nir, self.ir_], optional_datasets=[self.sunz], **info)

        assert self.metadata.items() <= res.attrs.items()
        assert res.attrs['units'] == '%'
        assert res.attrs['sun_zenith_threshold'] is not None
        assert np.allclose(res.data, self.refl * 100).compute()

    @mock.patch('satpy.modifiers.spectral.sun_zenith_angle')
    @mock.patch('satpy.modifiers.NIRReflectance.apply_modifier_info')
    @mock.patch('satpy.modifiers.spectral.Calculator')
    def test_no_sunz_no_co2(self, calculator, apply_modifier_info, sza):
        """Test NIR reflectance compositor with minimal parameters."""
        calculator.return_value = mock.MagicMock(
            reflectance_from_tbs=self.refl_from_tbs)
        sza.return_value = self.da_sunz
        from satpy.modifiers.spectral import NIRReflectance

        comp = NIRReflectance(name='test')
        info = {'modifiers': None}
        res = comp([self.nir, self.ir_], optional_datasets=[], **info)

        # due to copying of DataArrays, self.get_lonlats is not the same as the one that was called
        # we must used the area from the final result DataArray
        res.attrs["area"].get_lonlats.assert_called()
        sza.assert_called_with(self.start_time, self.lons, self.lats)
        self.refl_from_tbs.assert_called_with(self.da_sunz, self.nir.data, self.ir_.data, tb_ir_co2=None)
        assert np.allclose(res.data, self.refl * 100).compute()

    @mock.patch('satpy.modifiers.spectral.sun_zenith_angle')
    @mock.patch('satpy.modifiers.NIRReflectance.apply_modifier_info')
    @mock.patch('satpy.modifiers.spectral.Calculator')
    def test_no_sunz_with_co2(self, calculator, apply_modifier_info, sza):
        """Test NIR reflectance compositor provided extra co2 info."""
        calculator.return_value = mock.MagicMock(
            reflectance_from_tbs=self.refl_from_tbs)
        from satpy.modifiers.spectral import NIRReflectance
        sza.return_value = self.da_sunz

        comp = NIRReflectance(name='test')
        info = {'modifiers': None}
        co2_arr = np.random.random((2, 2))
        co2 = xr.DataArray(da.from_array(co2_arr), dims=['y', 'x'])
        co2.attrs['wavelength'] = [12.0, 13.0, 14.0]
        co2.attrs['units'] = 'K'
        res = comp([self.nir, self.ir_], optional_datasets=[co2], **info)

        self.refl_from_tbs.assert_called_with(self.da_sunz, self.nir.data, self.ir_.data, tb_ir_co2=co2.data)
        assert np.allclose(res.data, self.refl_with_co2 * 100).compute()

    @mock.patch('satpy.modifiers.spectral.sun_zenith_angle')
    @mock.patch('satpy.modifiers.NIRReflectance.apply_modifier_info')
    @mock.patch('satpy.modifiers.spectral.Calculator')
    def test_provide_sunz_and_threshold(self, calculator, apply_modifier_info, sza):
        """Test NIR reflectance compositor provided sunz and a sunz threshold."""
        calculator.return_value = mock.MagicMock(
            reflectance_from_tbs=self.refl_from_tbs)
        from satpy.modifiers.spectral import NIRReflectance
        sza.return_value = self.da_sunz

        comp = NIRReflectance(name='test', sunz_threshold=84.0)
        info = {'modifiers': None}
        res = comp([self.nir, self.ir_], optional_datasets=[self.sunz], **info)

        self.assertEqual(res.attrs['sun_zenith_threshold'], 84.0)
        calculator.assert_called_with('Meteosat-11', 'seviri', 'IR_039',
                                      sunz_threshold=84.0, masking_limit=NIRReflectance.MASKING_LIMIT)

    @mock.patch('satpy.modifiers.spectral.sun_zenith_angle')
    @mock.patch('satpy.modifiers.NIRReflectance.apply_modifier_info')
    @mock.patch('satpy.modifiers.spectral.Calculator')
    def test_sunz_threshold_default_value_is_not_none(self, calculator, apply_modifier_info, sza):
        """Check that sun_zenith_threshold is not None."""
        from satpy.modifiers.spectral import NIRReflectance

        comp = NIRReflectance(name='test')
        info = {'modifiers': None}
        calculator.return_value = mock.MagicMock(
            reflectance_from_tbs=self.refl_from_tbs)
        comp([self.nir, self.ir_], optional_datasets=[self.sunz], **info)

        assert comp.sun_zenith_threshold is not None

    @mock.patch('satpy.modifiers.spectral.sun_zenith_angle')
    @mock.patch('satpy.modifiers.NIRReflectance.apply_modifier_info')
    @mock.patch('satpy.modifiers.spectral.Calculator')
    def test_provide_masking_limit(self, calculator, apply_modifier_info, sza):
        """Test NIR reflectance compositor provided sunz and a sunz threshold."""
        calculator.return_value = mock.MagicMock(
            reflectance_from_tbs=self.refl_from_tbs)
        from satpy.modifiers.spectral import NIRReflectance
        sza.return_value = self.da_sunz

        comp = NIRReflectance(name='test', masking_limit=None)
        info = {'modifiers': None}
        res = comp([self.nir, self.ir_], optional_datasets=[self.sunz], **info)

        self.assertIsNone(res.attrs['sun_zenith_masking_limit'])
        calculator.assert_called_with('Meteosat-11', 'seviri', 'IR_039',
                                      sunz_threshold=NIRReflectance.TERMINATOR_LIMIT, masking_limit=None)

    @mock.patch('satpy.modifiers.spectral.sun_zenith_angle')
    @mock.patch('satpy.modifiers.NIRReflectance.apply_modifier_info')
    @mock.patch('satpy.modifiers.spectral.Calculator')
    def test_masking_limit_default_value_is_not_none(self, calculator, apply_modifier_info, sza):
        """Check that sun_zenith_threshold is not None."""
        from satpy.modifiers.spectral import NIRReflectance

        comp = NIRReflectance(name='test')
        info = {'modifiers': None}
        calculator.return_value = mock.MagicMock(
            reflectance_from_tbs=self.refl_from_tbs)
        comp([self.nir, self.ir_], optional_datasets=[self.sunz], **info)

        assert comp.masking_limit is not None


class TestNIREmissivePartFromReflectance(unittest.TestCase):
    """Test the NIR Emissive part from reflectance compositor."""

    @mock.patch('satpy.modifiers.spectral.sun_zenith_angle')
    @mock.patch('satpy.modifiers.NIRReflectance.apply_modifier_info')
    @mock.patch('satpy.modifiers.spectral.Calculator')
    def test_compositor(self, calculator, apply_modifier_info, sza):
        """Test the NIR emissive part from reflectance compositor."""
        from satpy.modifiers.spectral import NIRReflectance

        refl_arr = np.random.random((2, 2))
        refl = da.from_array(refl_arr)

        refl_from_tbs = mock.MagicMock()
        refl_from_tbs.return_value = refl
        calculator.return_value = mock.MagicMock(reflectance_from_tbs=refl_from_tbs)

        emissive_arr = np.random.random((2, 2))
        emissive = da.from_array(emissive_arr)
        emissive_part = mock.MagicMock()
        emissive_part.return_value = emissive
        calculator.return_value = mock.MagicMock(emissive_part_3x=emissive_part)

        from satpy.modifiers.spectral import NIREmissivePartFromReflectance

        comp = NIREmissivePartFromReflectance(name='test', sunz_threshold=86.0)
        info = {'modifiers': None}

        platform = 'NOAA-20'
        sensor = 'viirs'
        chan_name = 'M12'

        get_lonlats = mock.MagicMock()
        lons, lats = 1, 2
        get_lonlats.return_value = (lons, lats)
        area = mock.MagicMock(get_lonlats=get_lonlats)

        nir_arr = np.random.random((2, 2))
        nir = xr.DataArray(da.from_array(nir_arr), dims=['y', 'x'])
        nir.attrs['platform_name'] = platform
        nir.attrs['sensor'] = sensor
        nir.attrs['name'] = chan_name
        nir.attrs['area'] = area
        ir_arr = np.random.random((2, 2))
        ir_ = xr.DataArray(da.from_array(ir_arr), dims=['y', 'x'])
        ir_.attrs['area'] = area

        sunz_arr = 100 * np.random.random((2, 2))
        sunz = xr.DataArray(da.from_array(sunz_arr), dims=['y', 'x'])
        sunz.attrs['standard_name'] = 'solar_zenith_angle'
        sunz.attrs['area'] = area
        sunz2 = da.from_array(sunz_arr)
        sza.return_value = sunz2

        res = comp([nir, ir_], optional_datasets=[sunz], **info)
        self.assertEqual(res.attrs['sun_zenith_threshold'], 86.0)
        self.assertEqual(res.attrs['units'], 'K')
        self.assertEqual(res.attrs['platform_name'], platform)
        self.assertEqual(res.attrs['sensor'], sensor)
        self.assertEqual(res.attrs['name'], chan_name)
        calculator.assert_called_with('NOAA-20', 'viirs', 'M12', sunz_threshold=86.0,
                                      masking_limit=NIRReflectance.MASKING_LIMIT)


class TestPSPRayleighReflectance:
    """Test the pyspectral-based Rayleigh correction modifier."""

    def _make_data_area(self):
        """Create test area definition and data."""
        rows = 3
        cols = 5
        area = AreaDefinition(
            'some_area_name', 'On-the-fly area', 'geosabii',
            {'a': '6378137.0', 'b': '6356752.31414', 'h': '35786023.0', 'lon_0': '-89.5', 'proj': 'geos', 'sweep': 'x',
             'units': 'm'},
            cols, rows,
            (-5434894.954752679, -5434894.964451744, 5434894.964451744, 5434894.954752679))

        data = np.zeros((rows, cols)) + 25
        data[1, :] += 25
        data[2, :] += 50
        data = da.from_array(data, chunks=2)
        return area, data

    def _create_test_data(self, name, wavelength, resolution):
        area, dnb = self._make_data_area()
        input_band = xr.DataArray(dnb,
                                  dims=('y', 'x'),
                                  attrs={
                                      'platform_name': 'Himawari-8',
                                      'calibration': 'reflectance', 'units': '%', 'wavelength': wavelength,
                                      'name': name, 'resolution': resolution, 'sensor': 'ahi',
                                      'start_time': '2017-09-20 17:30:40.800000',
                                      'end_time': '2017-09-20 17:41:17.500000',
                                      'area': area, 'ancillary_variables': [],
                                      'orbital_parameters': {
                                          'satellite_nominal_longitude': -89.5,
                                          'satellite_nominal_latitude': 0.0,
                                          'satellite_nominal_altitude': 35786023.4375,
                                      },
                                  })

        red_band = xr.DataArray(dnb,
                                dims=('y', 'x'),
                                attrs={
                                    'platform_name': 'Himawari-8',
                                    'calibration': 'reflectance', 'units': '%', 'wavelength': (0.62, 0.64, 0.66),
                                    'name': 'B03', 'resolution': 500, 'sensor': 'ahi',
                                    'start_time': '2017-09-20 17:30:40.800000',
                                    'end_time': '2017-09-20 17:41:17.500000',
                                    'area': area, 'ancillary_variables': [],
                                    'orbital_parameters': {
                                        'satellite_nominal_longitude': -89.5,
                                        'satellite_nominal_latitude': 0.0,
                                        'satellite_nominal_altitude': 35786023.4375,
                                    },
                                })
        fake_angle_data = da.ones_like(dnb, dtype=np.float32) * 90.0
        angle1 = xr.DataArray(fake_angle_data,
                              dims=('y', 'x'),
                              attrs={
                                  'platform_name': 'Himawari-8',
                                  'calibration': 'reflectance', 'units': '%', 'wavelength': wavelength,
                                  'name': "satellite_azimuth_angle", 'resolution': resolution, 'sensor': 'ahi',
                                  'start_time': '2017-09-20 17:30:40.800000',
                                  'end_time': '2017-09-20 17:41:17.500000',
                                  'area': area, 'ancillary_variables': [],
                              })
        return input_band, red_band, angle1, angle1, angle1, angle1

    @pytest.mark.parametrize(
        ("name", "wavelength", "resolution", "aerosol_type", "reduce_lim_low", "reduce_lim_high", "reduce_strength",
         "exp_mean", "exp_unique"),
        [
            ("B01", (0.45, 0.47, 0.49), 1000, "rayleigh_only", 70, 95, 1, 41.540239,
             np.array([9.22630464, 10.67844368, 13.58057226, 37.92186549, 40.13822472, 44.66259518,
                       44.92748445, 45.03917091, 69.5821722, 70.11226943, 71.07352559])),
            ("B02", (0.49, 0.51, 0.53), 1000, "rayleigh_only", 70, 95, 1, 43.663805,
             np.array([13.15770104, 14.26526104, 16.49084485, 40.88633902, 42.60682921, 46.04288,
                       46.2356062, 46.28276282, 70.92799823, 71.33561614, 72.07001693])),
            ("B03", (0.62, 0.64, 0.66), 500, "rayleigh_only", 70, 95, 1, 46.916187,
             np.array([19.22922328, 19.76884762, 20.91027446, 45.51075967, 46.39925968, 48.10221156,
                       48.15715058, 48.18698356, 73.01115816, 73.21552816, 73.58666477])),
            ("B01", (0.45, 0.47, 0.49), 1000, "rayleigh_only", -95, -70, -1, 41.540239,
             np.array([9.22630464, 10.67844368, 13.58057226, 37.92186549, 40.13822472, 44.66259518,
                       44.92748445, 45.03917091, 69.5821722, 70.11226943, 71.07352559])),
        ]
    )
    def test_rayleigh_corrector(self, name, wavelength, resolution, aerosol_type, reduce_lim_low, reduce_lim_high,
                                reduce_strength, exp_mean, exp_unique):
        """Test PSPRayleighReflectance with fake data."""
        from satpy.modifiers.atmosphere import PSPRayleighReflectance
        ray_cor = PSPRayleighReflectance(name=name, atmosphere='us-standard', aerosol_types=aerosol_type,
                                         reduce_lim_low=reduce_lim_low, reduce_lim_high=reduce_lim_high,
                                         reduce_strength=reduce_strength)
        assert ray_cor.attrs['name'] == name
        assert ray_cor.attrs['atmosphere'] == 'us-standard'
        assert ray_cor.attrs['aerosol_types'] == aerosol_type
        assert ray_cor.attrs['reduce_lim_low'] == reduce_lim_low
        assert ray_cor.attrs['reduce_lim_high'] == reduce_lim_high
        assert ray_cor.attrs['reduce_strength'] == reduce_strength

        input_band, red_band, *_ = self._create_test_data(name, wavelength, resolution)
        res = ray_cor([input_band, red_band])

        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)

        data = res.values
        unique = np.unique(data[~np.isnan(data)])
        np.testing.assert_allclose(np.nanmean(data), exp_mean, rtol=1e-5)
        assert data.shape == (3, 5)
        np.testing.assert_allclose(unique, exp_unique, rtol=1e-5)

    @pytest.mark.parametrize("as_optionals", [False, True])
    def test_rayleigh_with_angles(self, as_optionals):
        """Test PSPRayleighReflectance with angles provided."""
        from satpy.modifiers.atmosphere import PSPRayleighReflectance
        aerosol_type = "rayleigh_only"
        ray_cor = PSPRayleighReflectance(name="B01", atmosphere='us-standard', aerosol_types=aerosol_type)
        prereqs, opt_prereqs = self._get_angles_prereqs_and_opts(as_optionals)
        with mock.patch("satpy.modifiers.atmosphere.get_angles") as get_angles:
            res = ray_cor(prereqs, opt_prereqs)
        get_angles.assert_not_called()

        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)

        data = res.values
        unique = np.unique(data[~np.isnan(data)])
        np.testing.assert_allclose(unique, np.array([-75.0, -37.71298492, 31.14350754]), rtol=1e-5)
        assert data.shape == (3, 5)

    def _get_angles_prereqs_and_opts(self, as_optionals):
        wavelength = (0.45, 0.47, 0.49)
        resolution = 1000
        input_band, red_band, *angles = self._create_test_data("B01", wavelength, resolution)
        prereqs = [input_band, red_band]
        opt_prereqs = []
        if as_optionals:
            opt_prereqs = angles
        else:
            prereqs += angles
        return prereqs, opt_prereqs


class TestPSPAtmosphericalCorrection(unittest.TestCase):
    """Test the pyspectral-based atmospheric correction modifier."""

    def test_call(self):
        """Test atmospherical correction."""
        from pyresample.geometry import SwathDefinition

        from satpy.modifiers import PSPAtmosphericalCorrection

        # Patch methods
        lons = np.zeros((5, 5))
        lons[1, 1] = np.inf
        lons = da.from_array(lons, chunks=5)
        lats = np.zeros((5, 5))
        lats[1, 1] = np.inf
        lats = da.from_array(lats, chunks=5)
        area = SwathDefinition(lons, lats)
        stime = datetime(2020, 1, 1, 12, 0, 0)
        orb_params = {
            "satellite_actual_altitude": 12345678,
            "nadir_longitude": 0.0,
            "nadir_latitude": 0.0,
        }
        band = xr.DataArray(da.zeros((5, 5)),
                            attrs={'area': area,
                                   'start_time': stime,
                                   'name': 'name',
                                   'platform_name': 'platform',
                                   'sensor': 'sensor',
                                   'orbital_parameters': orb_params},
                            dims=('y', 'x'))

        # Perform atmospherical correction
        psp = PSPAtmosphericalCorrection(name='dummy')
        res = psp(projectables=[band])
        res.compute()
