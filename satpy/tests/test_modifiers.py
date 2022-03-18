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
import contextlib
import unittest
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from glob import glob
from typing import Optional, Union
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition, StackedAreaDefinition
from pytest_lazyfixture import lazy_fixture

import satpy
from satpy.utils import PerformanceWarning


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

    def test_basic_default_not_provided(self, sunz_ds1):
        """Test default limits when SZA isn't provided."""
        from satpy.modifiers.geometry import SunZenithCorrector
        comp = SunZenithCorrector(name='sza_test', modifiers=tuple())
        res = comp((sunz_ds1,), test_attr='test')
        np.testing.assert_allclose(res.values, np.array([[22.401667, 22.31777], [22.437503, 22.353533]]))
        assert 'y' in res.coords
        assert 'x' in res.coords
        ds1 = sunz_ds1.copy().drop_vars(('y', 'x'))
        res = comp((ds1,), test_attr='test')
        np.testing.assert_allclose(res.values, np.array([[22.401667, 22.31777], [22.437503, 22.353533]]))
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

        self.get_lonlats.assert_called()
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


def _angle_cache_area_def():
    area = AreaDefinition(
        "test", "", "",
        {"proj": "merc"},
        5, 5,
        (-2500, -2500, 2500, 2500),
    )
    return area


def _angle_cache_stacked_area_def():
    area1 = AreaDefinition(
        "test", "", "",
        {"proj": "merc"},
        5, 2,
        (2500, 500, 7500, 2500),
    )
    area2 = AreaDefinition(
        "test", "", "",
        {"proj": "merc"},
        5, 3,
        (2500, -2500, 7500, 500),
    )
    return StackedAreaDefinition(area1, area2)


def _get_angle_test_data(area_def: Optional[Union[AreaDefinition, StackedAreaDefinition]] = None,
                         chunks: Optional[Union[int, tuple]] = 2) -> xr.DataArray:
    if area_def is None:
        area_def = _angle_cache_area_def()
    orb_params = {
        "satellite_nominal_altitude": 12345678,
        "satellite_nominal_longitude": 10.0,
        "satellite_nominal_latitude": 0.0,
    }
    stime = datetime(2020, 1, 1, 12, 0, 0)
    data = da.zeros((5, 5), chunks=chunks)
    vis = xr.DataArray(data,
                       attrs={
                           'area': area_def,
                           'start_time': stime,
                           'orbital_parameters': orb_params,
                       })
    return vis


def _get_stacked_angle_test_data():
    return _get_angle_test_data(area_def=_angle_cache_stacked_area_def(),
                                chunks=(5, (2, 2, 1)))


def _get_angle_test_data_odd_chunks():
    return _get_angle_test_data(chunks=((2, 1, 2), (1, 1, 2, 1)))


def _similar_sat_pos_datetime(orig_data, lon_offset=0.04):
    # change data slightly
    new_data = orig_data.copy()
    old_lon = new_data.attrs["orbital_parameters"]["satellite_nominal_longitude"]
    new_data.attrs["orbital_parameters"]["satellite_nominal_longitude"] = old_lon + lon_offset
    new_data.attrs["start_time"] = new_data.attrs["start_time"] + timedelta(hours=36)
    return new_data


def _diff_sat_pos_datetime(orig_data):
    return _similar_sat_pos_datetime(orig_data, lon_offset=0.05)


def _glob_reversed(pat):
    """Behave like glob but force results to be in the wrong order."""
    return sorted(glob(pat), reverse=True)


@contextlib.contextmanager
def _mock_glob_if(mock_glob):
    if mock_glob:
        with mock.patch("satpy.modifiers.angles.glob", _glob_reversed):
            yield
    else:
        yield


def _assert_allclose_if(expect_equal, arr1, arr2):
    if not expect_equal:
        pytest.raises(AssertionError, np.testing.assert_allclose, arr1, arr2)
    else:
        np.testing.assert_allclose(arr1, arr2)


class TestAngleGeneration:
    """Test the angle generation utility functions."""

    @pytest.mark.parametrize("input_func", [_get_angle_test_data, _get_stacked_angle_test_data])
    def test_get_angles(self, input_func):
        """Test sun and satellite angle calculation."""
        from satpy.modifiers.angles import get_angles
        data = input_func()

        from pyorbital.orbital import get_observer_look
        with mock.patch("satpy.modifiers.angles.get_observer_look", wraps=get_observer_look) as gol:
            angles = get_angles(data)
            assert all(isinstance(x, xr.DataArray) for x in angles)
            da.compute(angles)

        # get_observer_look should have been called once per array chunk
        assert gol.call_count == data.data.blocks.size
        # Check arguments of get_orbserver_look() call, especially the altitude
        # unit conversion from meters to kilometers
        args = gol.call_args[0]
        assert args[:4] == (10.0, 0.0, 12345.678, data.attrs["start_time"])

    @pytest.mark.parametrize("forced_preference", ["actual", "nadir"])
    def test_get_angles_satpos_preference(self, forced_preference):
        """Test that 'actual' satellite position is used for generating sensor angles."""
        from satpy.modifiers.angles import get_angles

        input_data1 = _get_angle_test_data()
        # add additional satellite position metadata
        input_data1.attrs["orbital_parameters"]["nadir_longitude"] = 9.0
        input_data1.attrs["orbital_parameters"]["nadir_latitude"] = 0.01
        input_data1.attrs["orbital_parameters"]["satellite_actual_longitude"] = 9.5
        input_data1.attrs["orbital_parameters"]["satellite_actual_latitude"] = 0.005
        input_data1.attrs["orbital_parameters"]["satellite_actual_altitude"] = 12345679
        input_data2 = input_data1.copy(deep=True)
        input_data2.attrs = deepcopy(input_data1.attrs)
        input_data2.attrs["orbital_parameters"]["nadir_longitude"] = 9.1
        input_data2.attrs["orbital_parameters"]["nadir_latitude"] = 0.02
        input_data2.attrs["orbital_parameters"]["satellite_actual_longitude"] = 9.5
        input_data2.attrs["orbital_parameters"]["satellite_actual_latitude"] = 0.005
        input_data2.attrs["orbital_parameters"]["satellite_actual_altitude"] = 12345679

        from pyorbital.orbital import get_observer_look
        with mock.patch("satpy.modifiers.angles.get_observer_look", wraps=get_observer_look) as gol, \
                satpy.config.set(sensor_angles_position_preference=forced_preference):
            angles1 = get_angles(input_data1)
            da.compute(angles1)
            angles2 = get_angles(input_data2)
            da.compute(angles2)

        # get_observer_look should have been called once per array chunk
        assert gol.call_count == input_data1.data.blocks.size * 2
        if forced_preference == "actual":
            exp_call = mock.call(9.5, 0.005, 12345.679, input_data1.attrs["start_time"], mock.ANY, mock.ANY, 0)
            all_same_calls = [exp_call] * gol.call_count
            gol.assert_has_calls(all_same_calls)
            # the dask arrays should have the same name to prove they are the same computation
            for angle_arr1, angle_arr2 in zip(angles1, angles2):
                assert angle_arr1.data.name == angle_arr2.data.name
        else:
            # nadir 1
            gol.assert_any_call(9.0, 0.01, 12345.679, input_data1.attrs["start_time"], mock.ANY, mock.ANY, 0)
            # nadir 2
            gol.assert_any_call(9.1, 0.02, 12345.679, input_data1.attrs["start_time"], mock.ANY, mock.ANY, 0)

    @pytest.mark.parametrize("force_bad_glob", [False, True])
    @pytest.mark.parametrize(
        ("input2_func", "exp_equal_sun", "exp_num_zarr"),
        [
            (lambda x: x, True, 4),
            (_similar_sat_pos_datetime, False, 4),
            (_diff_sat_pos_datetime, False, 6),
        ]
    )
    @pytest.mark.parametrize(
        ("input_func", "num_normalized_chunks"),
        [
            (_get_angle_test_data, 9),
            (_get_stacked_angle_test_data, 3),
            (_get_angle_test_data_odd_chunks, 9),
        ])
    def test_cache_get_angles(
            self,
            input_func, num_normalized_chunks,
            input2_func, exp_equal_sun, exp_num_zarr,
            force_bad_glob, tmp_path):
        """Test get_angles when caching is enabled."""
        from satpy.modifiers.angles import STATIC_EARTH_INERTIAL_DATETIME, get_angles

        # Patch methods
        data = input_func()
        additional_cache = exp_num_zarr > 4

        # Compute angles
        from pyorbital.orbital import get_observer_look
        with mock.patch("satpy.modifiers.angles.get_observer_look", wraps=get_observer_look) as gol, \
                satpy.config.set(cache_lonlats=True, cache_sensor_angles=True, cache_dir=str(tmp_path)), \
                warnings.catch_warnings(record=True) as caught_warnings:
            res = get_angles(data)
            self._check_cached_result(res, data)

            # call again, should be cached
            new_data = input2_func(data)
            with _mock_glob_if(force_bad_glob):
                res2 = get_angles(new_data)
            self._check_cached_result(res2, data)

            res_numpy, res2_numpy = da.compute(res, res2)
            for r1, r2 in zip(res_numpy[:2], res2_numpy[:2]):
                _assert_allclose_if(not additional_cache, r1, r2)
            for r1, r2 in zip(res_numpy[2:], res2_numpy[2:]):
                _assert_allclose_if(exp_equal_sun, r1, r2)

            self._check_cache_and_clear(tmp_path, exp_num_zarr)

        if "odd_chunks" in input_func.__name__:
            assert any(w.category is PerformanceWarning for w in caught_warnings)
        else:
            assert not any(w.category is PerformanceWarning for w in caught_warnings)
        assert gol.call_count == num_normalized_chunks * (int(additional_cache) + 1)
        args = gol.call_args_list[0][0]
        assert args[:4] == (10.0, 0.0, 12345.678, STATIC_EARTH_INERTIAL_DATETIME)
        exp_sat_lon = 10.1 if additional_cache else 10.0
        args = gol.call_args_list[-1][0]
        assert args[:4] == (exp_sat_lon, 0.0, 12345.678, STATIC_EARTH_INERTIAL_DATETIME)

    @staticmethod
    def _check_cached_result(results, input_data):
        assert all(isinstance(x, xr.DataArray) for x in results)
        # output chunks should be consistent
        for angle_data_arr in results:
            assert angle_data_arr.chunks == input_data.chunks

    @staticmethod
    def _check_cache_and_clear(tmp_path, exp_num_zarr):
        from satpy.modifiers.angles import _get_sensor_angles_from_sat_pos, _get_valid_lonlats
        zarr_dirs = glob(str(tmp_path / "*.zarr"))
        assert len(zarr_dirs) == exp_num_zarr  # two for lon/lat, one for sata, one for satz

        _get_valid_lonlats.cache_clear()
        _get_sensor_angles_from_sat_pos.cache_clear()
        zarr_dirs = glob(str(tmp_path / "*.zarr"))
        assert len(zarr_dirs) == 0

    def test_cached_no_chunks_fails(self, tmp_path):
        """Test that trying to pass non-dask arrays and no chunks fails."""
        from satpy.modifiers.angles import _sanitize_args_with_chunks, cache_to_zarr_if

        @cache_to_zarr_if("cache_lonlats", sanitize_args_func=_sanitize_args_with_chunks)
        def _fake_func(data, tuple_arg, chunks):
            return da.from_array(data)

        data = list(range(5))
        with pytest.raises(RuntimeError), \
                satpy.config.set(cache_lonlats=True, cache_dir=str(tmp_path)):
            _fake_func(data, (1, 2, 3), 5)

    def test_cached_result_numpy_fails(self, tmp_path):
        """Test that trying to cache with non-dask arrays fails."""
        from satpy.modifiers.angles import _sanitize_args_with_chunks, cache_to_zarr_if

        @cache_to_zarr_if("cache_lonlats", sanitize_args_func=_sanitize_args_with_chunks)
        def _fake_func(shape, chunks):
            return np.zeros(shape)

        with pytest.raises(ValueError), \
                satpy.config.set(cache_lonlats=True, cache_dir=str(tmp_path)):
            _fake_func((5, 5), ((5,), (5,)))

    def test_no_cache_dir_fails(self, tmp_path):
        """Test that 'cache_dir' not being set fails."""
        from satpy.modifiers.angles import _get_sensor_angles_from_sat_pos, get_angles
        data = _get_angle_test_data()
        with pytest.raises(RuntimeError), \
                satpy.config.set(cache_lonlats=True, cache_sensor_angles=True, cache_dir=None):
            get_angles(data)
        with pytest.raises(RuntimeError), \
                satpy.config.set(cache_lonlats=True, cache_sensor_angles=True, cache_dir=None):
            _get_sensor_angles_from_sat_pos.cache_clear()
