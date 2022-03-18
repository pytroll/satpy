#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 - 2020 Satpy developers
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
"""Tests for the CREFL ReflectanceCorrector modifier."""
from contextlib import contextmanager
from datetime import datetime
from unittest import mock

import numpy as np
import pytest
import xarray as xr
from dask import array as da
from pyresample.geometry import AreaDefinition

from ..utils import assert_maximum_dask_computes


@contextmanager
def mock_cmgdem(tmpdir, url):
    """Create fake file representing CMGDEM.hdf."""
    yield from _mock_and_create_dem_file(tmpdir, url, "averaged elevation", fill_value=-9999)


@contextmanager
def mock_tbase(tmpdir, url):
    """Create fake file representing tbase.hdf."""
    yield from _mock_and_create_dem_file(tmpdir, url, "Elevation")


def _mock_and_create_dem_file(tmpdir, url, var_name, fill_value=None):
    if not url:
        yield None
        return

    rmock_obj, dem_fn = _mock_dem_retrieve(tmpdir, url)
    _create_fake_dem_file(dem_fn, var_name, fill_value)

    try:
        yield rmock_obj
    finally:
        rmock_obj.stop()


def _mock_dem_retrieve(tmpdir, url):
    rmock_obj = mock.patch('satpy.modifiers._crefl.retrieve')
    rmock = rmock_obj.start()
    dem_fn = str(tmpdir.join(url))
    rmock.return_value = dem_fn
    return rmock_obj, dem_fn


def _create_fake_dem_file(dem_fn, var_name, fill_value):
    from pyhdf.SD import SD, SDC
    h = SD(dem_fn, SDC.WRITE | SDC.CREATE)
    dem_var = h.create(var_name, SDC.INT16, (10, 10))
    dem_var[:] = np.zeros((10, 10), dtype=np.int16)
    if fill_value is not None:
        dem_var.setfillvalue(fill_value)
    h.end()


def _make_viirs_xarray(data, area, name, standard_name, wavelength=None, units='degrees', calibration=None):
    return xr.DataArray(data, dims=('y', 'x'),
                        attrs={
                            'start_orbit': 1708, 'end_orbit': 1708, 'wavelength': wavelength,
                            'modifiers': None, 'calibration': calibration,
                            'resolution': 371, 'name': name,
                            'standard_name': standard_name, 'platform_name': 'Suomi-NPP',
                            'polarization': None, 'sensor': 'viirs', 'units': units,
                            'start_time': datetime(2012, 2, 25, 18, 1, 24, 570942),
                            'end_time': datetime(2012, 2, 25, 18, 11, 21, 175760), 'area': area,
                            'ancillary_variables': []
                        })


class TestReflectanceCorrectorModifier:
    """Test the CREFL modifier."""

    @staticmethod
    def data_area_ref_corrector():
        """Create test area definition and data."""
        rows = 5
        cols = 10
        area = AreaDefinition(
            'some_area_name', 'On-the-fly area', 'geosabii',
            {'a': '6378137.0', 'b': '6356752.31414', 'h': '35786023.0', 'lon_0': '-89.5', 'proj': 'geos', 'sweep': 'x',
             'units': 'm'},
            cols, rows,
            (-5434894.954752679, -5434894.964451744, 5434894.964451744, 5434894.954752679))

        data = np.zeros((rows, cols)) + 25
        data[3, :] += 25
        data[4:, :] += 50
        data = da.from_array(data, chunks=2)
        return area, data

    def test_reflectance_corrector_abi(self):
        """Test ReflectanceCorrector modifier with ABI data."""
        from satpy.modifiers._crefl import ReflectanceCorrector
        from satpy.tests.utils import make_dsq
        ref_cor = ReflectanceCorrector(optional_prerequisites=[
            make_dsq(name='satellite_azimuth_angle'),
            make_dsq(name='satellite_zenith_angle'),
            make_dsq(name='solar_azimuth_angle'),
            make_dsq(name='solar_zenith_angle')], name='C01', prerequisites=[],
                                       wavelength=(0.45, 0.47, 0.49), resolution=1000, calibration='reflectance',
                                       modifiers=('sunz_corrected', 'rayleigh_corrected_crefl',), sensor='abi')

        assert ref_cor.attrs['modifiers'] == ('sunz_corrected', 'rayleigh_corrected_crefl')
        assert ref_cor.attrs['calibration'] == 'reflectance'
        assert ref_cor.attrs['wavelength'] == (0.45, 0.47, 0.49)
        assert ref_cor.attrs['name'] == 'C01'
        assert ref_cor.attrs['resolution'] == 1000
        assert ref_cor.attrs['sensor'] == 'abi'
        assert ref_cor.attrs['prerequisites'] == []
        assert ref_cor.attrs['optional_prerequisites'] == [
            make_dsq(name='satellite_azimuth_angle'),
            make_dsq(name='satellite_zenith_angle'),
            make_dsq(name='solar_azimuth_angle'),
            make_dsq(name='solar_zenith_angle')]

        area, dnb = self.data_area_ref_corrector()
        c01 = xr.DataArray(dnb,
                           dims=('y', 'x'),
                           attrs={
                               'platform_name': 'GOES-16',
                               'calibration': 'reflectance', 'units': '%', 'wavelength': (0.45, 0.47, 0.49),
                               'name': 'C01', 'resolution': 1000, 'sensor': 'abi',
                               'start_time': '2017-09-20 17:30:40.800000', 'end_time': '2017-09-20 17:41:17.500000',
                               'area': area, 'ancillary_variables': [],
                               'orbital_parameters': {
                                   'satellite_nominal_longitude': -89.5,
                                   'satellite_nominal_latitude': 0.0,
                                   'satellite_nominal_altitude': 35786023.4375,
                               },
                           })
        with assert_maximum_dask_computes(0):
            res = ref_cor([c01], [])

        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs['modifiers'] == ('sunz_corrected', 'rayleigh_corrected_crefl')
        assert res.attrs['platform_name'] == 'GOES-16'
        assert res.attrs['calibration'] == 'reflectance'
        assert res.attrs['units'] == '%'
        assert res.attrs['wavelength'] == (0.45, 0.47, 0.49)
        assert res.attrs['name'] == 'C01'
        assert res.attrs['resolution'] == 1000
        assert res.attrs['sensor'] == 'abi'
        assert res.attrs['start_time'] == '2017-09-20 17:30:40.800000'
        assert res.attrs['end_time'] == '2017-09-20 17:41:17.500000'
        assert res.attrs['area'] == area
        assert res.attrs['ancillary_variables'] == []
        data = res.values
        assert abs(np.nanmean(data) - 26.00760944144745) < 1e-10
        assert data.shape == (5, 10)
        unique = np.unique(data[~np.isnan(data)])
        np.testing.assert_allclose(unique, [-1.0, 4.210745457958135, 6.7833906076177595, 8.730371329824473,
                                            10.286627569545209, 11.744159436709374, 12.20226097829902,
                                            13.501444598985305, 15.344399223932212, 17.173329483996515,
                                            17.28798660754271, 18.29594550575925, 19.076835059905125,
                                            19.288331720959864, 19.77043407084455, 19.887082168377006,
                                            20.091028778326375, 20.230341149334617, 20.457671064690196,
                                            20.82686905639114, 21.021094816441195, 21.129963777952124,
                                            41.601857910095575, 43.963919057675504,
                                            46.21672174361075, 46.972099490462085, 47.497072794632835,
                                            47.80393007974336, 47.956765988770385, 48.043025685032106,
                                            51.909142813383916, 58.8234273736508, 68.84706145641482, 69.91085190887961,
                                            71.10179768327806, 71.33161009169649])

    @pytest.mark.parametrize(
        'url,dem_mock_cm,dem_sds',
        [
            (None, mock_cmgdem, "average elevation"),
            ("CMGDEM.hdf", mock_cmgdem, "averaged elevation"),
            ("tbase.hdf", mock_tbase, "Elevation"),
        ])
    def test_reflectance_corrector_viirs(self, tmpdir, url, dem_mock_cm, dem_sds):
        """Test ReflectanceCorrector modifier with VIIRS data."""
        from satpy.modifiers._crefl import ReflectanceCorrector
        from satpy.tests.utils import make_dsq

        ref_cor = ReflectanceCorrector(
            optional_prerequisites=[
                make_dsq(name='satellite_azimuth_angle'),
                make_dsq(name='satellite_zenith_angle'),
                make_dsq(name='solar_azimuth_angle'),
                make_dsq(name='solar_zenith_angle')
            ],
            name='I01',
            prerequisites=[],
            wavelength=(0.6, 0.64, 0.68),
            resolution=371,
            calibration='reflectance',
            modifiers=('sunz_corrected_iband', 'rayleigh_corrected_crefl_iband'),
            sensor='viirs',
            url=url,
            dem_sds=dem_sds,
        )

        assert ref_cor.attrs['modifiers'] == ('sunz_corrected_iband', 'rayleigh_corrected_crefl_iband')
        assert ref_cor.attrs['calibration'] == 'reflectance'
        assert ref_cor.attrs['wavelength'] == (0.6, 0.64, 0.68)
        assert ref_cor.attrs['name'] == 'I01'
        assert ref_cor.attrs['resolution'] == 371
        assert ref_cor.attrs['sensor'] == 'viirs'
        assert ref_cor.attrs['prerequisites'] == []
        assert ref_cor.attrs['optional_prerequisites'] == [
            make_dsq(name='satellite_azimuth_angle'),
            make_dsq(name='satellite_zenith_angle'),
            make_dsq(name='solar_azimuth_angle'),
            make_dsq(name='solar_zenith_angle')]

        area, data = self.data_area_ref_corrector()
        c01 = _make_viirs_xarray(data, area, 'I01', 'toa_bidirectional_reflectance',
                                 wavelength=(0.6, 0.64, 0.68), units='%',
                                 calibration='reflectance')
        c02 = _make_viirs_xarray(data, area, 'satellite_azimuth_angle', 'sensor_azimuth_angle')
        c03 = _make_viirs_xarray(data, area, 'satellite_zenith_angle', 'sensor_zenith_angle')
        c04 = _make_viirs_xarray(data, area, 'solar_azimuth_angle', 'solar_azimuth_angle')
        c05 = _make_viirs_xarray(data, area, 'solar_zenith_angle', 'solar_zenith_angle')

        with dem_mock_cm(tmpdir, url), assert_maximum_dask_computes(0):
            res = ref_cor([c01], [c02, c03, c04, c05])

        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs['wavelength'] == (0.6, 0.64, 0.68)
        assert res.attrs['modifiers'] == ('sunz_corrected_iband', 'rayleigh_corrected_crefl_iband')
        assert res.attrs['calibration'] == 'reflectance'
        assert res.attrs['resolution'] == 371
        assert res.attrs['name'] == 'I01'
        assert res.attrs['standard_name'] == 'toa_bidirectional_reflectance'
        assert res.attrs['platform_name'] == 'Suomi-NPP'
        assert res.attrs['sensor'] == 'viirs'
        assert res.attrs['units'] == '%'
        assert res.attrs['start_time'] == datetime(2012, 2, 25, 18, 1, 24, 570942)
        assert res.attrs['end_time'] == datetime(2012, 2, 25, 18, 11, 21, 175760)
        assert res.attrs['area'] == area
        assert res.attrs['ancillary_variables'] == []
        data = res.values
        assert abs(np.mean(data) - 40.7578684169142) < 1e-10
        assert data.shape == (5, 10)
        unique = np.unique(data)
        np.testing.assert_allclose(unique, [25.20341702519979, 52.38819447051263, 75.79089653845898])

    def test_reflectance_corrector_modis(self):
        """Test ReflectanceCorrector modifier with MODIS data."""
        from satpy.modifiers._crefl import ReflectanceCorrector
        from satpy.tests.utils import make_dsq
        sataa_did = make_dsq(name='satellite_azimuth_angle')
        satza_did = make_dsq(name='satellite_zenith_angle')
        solaa_did = make_dsq(name='solar_azimuth_angle')
        solza_did = make_dsq(name='solar_zenith_angle')
        ref_cor = ReflectanceCorrector(
            optional_prerequisites=[sataa_did, satza_did, solaa_did, solza_did], name='1',
            prerequisites=[], wavelength=(0.62, 0.645, 0.67), resolution=250, calibration='reflectance',
            modifiers=('sunz_corrected', 'rayleigh_corrected_crefl'), sensor='modis')
        assert ref_cor.attrs['modifiers'] == ('sunz_corrected', 'rayleigh_corrected_crefl')
        assert ref_cor.attrs['calibration'] == 'reflectance'
        assert ref_cor.attrs['wavelength'] == (0.62, 0.645, 0.67)
        assert ref_cor.attrs['name'] == '1'
        assert ref_cor.attrs['resolution'] == 250
        assert ref_cor.attrs['sensor'] == 'modis'
        assert ref_cor.attrs['prerequisites'] == []
        assert ref_cor.attrs['optional_prerequisites'] == [
            make_dsq(name='satellite_azimuth_angle'),
            make_dsq(name='satellite_zenith_angle'),
            make_dsq(name='solar_azimuth_angle'),
            make_dsq(name='solar_zenith_angle')]

        area, dnb = self.data_area_ref_corrector()

        def make_xarray(name, calibration, wavelength=None, modifiers=None, resolution=1000):
            return xr.DataArray(dnb,
                                dims=('y', 'x'),
                                attrs={
                                    'wavelength': wavelength, 'level': None, 'modifiers': modifiers,
                                    'calibration': calibration, 'resolution': resolution,
                                    'name': name, 'coordinates': ['longitude', 'latitude'],
                                    'platform_name': 'EOS-Aqua', 'polarization': None, 'sensor': 'modis',
                                    'units': '%', 'start_time': datetime(2012, 8, 13, 18, 46, 1, 439838),
                                    'end_time': datetime(2012, 8, 13, 18, 57, 47, 746296), 'area': area,
                                    'ancillary_variables': []
                                })

        c01 = make_xarray('1', 'reflectance', wavelength=(0.62, 0.645, 0.67), modifiers='sunz_corrected',
                          resolution=500)
        c02 = make_xarray('satellite_azimuth_angle', None)
        c03 = make_xarray('satellite_zenith_angle', None)
        c04 = make_xarray('solar_azimuth_angle', None)
        c05 = make_xarray('solar_zenith_angle', None)
        res = ref_cor([c01], [c02, c03, c04, c05])

        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs['wavelength'] == (0.62, 0.645, 0.67)
        assert res.attrs['modifiers'] == ('sunz_corrected', 'rayleigh_corrected_crefl',)
        assert res.attrs['calibration'] == 'reflectance'
        assert res.attrs['resolution'] == 500
        assert res.attrs['name'] == '1'
        assert res.attrs['platform_name'] == 'EOS-Aqua'
        assert res.attrs['sensor'] == 'modis'
        assert res.attrs['units'] == '%'
        assert res.attrs['start_time'] == datetime(2012, 8, 13, 18, 46, 1, 439838)
        assert res.attrs['end_time'] == datetime(2012, 8, 13, 18, 57, 47, 746296)
        assert res.attrs['area'] == area
        assert res.attrs['ancillary_variables'] == []
        data = res.values
        if abs(np.mean(data) - 38.734365117099145) >= 1e-10:
            raise AssertionError('{} is not within {} of {}'.format(np.mean(data), 1e-10, 38.734365117099145))
        assert data.shape == (5, 10)
        unique = np.unique(data)
        np.testing.assert_allclose(unique, [24.641586, 50.431692, 69.315375])

    def test_reflectance_corrector_bad_prereqs(self):
        """Test ReflectanceCorrector modifier with wrong number of inputs."""
        from satpy.modifiers._crefl import ReflectanceCorrector
        ref_cor = ReflectanceCorrector("test")
        pytest.raises(ValueError, ref_cor, [1], [2, 3, 4])
        pytest.raises(ValueError, ref_cor, [1, 2, 3, 4], [])
        pytest.raises(ValueError, ref_cor, [], [1, 2, 3, 4])

    @pytest.mark.parametrize(
        'url,dem_mock_cm,dem_sds',
        [
            (None, mock_cmgdem, "average elevation"),
            ("CMGDEM.hdf", mock_cmgdem, "averaged elevation"),
            ("tbase.hdf", mock_tbase, "Elevation"),
        ])
    def test_reflectance_corrector_different_chunks(self, tmpdir, url, dem_mock_cm, dem_sds):
        """Test that the modifier works with different chunk sizes for inputs.

        The modifier uses dask's "map_blocks". If the input chunks aren't the
        same an error is raised.

        """
        from satpy.modifiers._crefl import ReflectanceCorrector
        from satpy.tests.utils import make_dsq

        ref_cor = ReflectanceCorrector(
            optional_prerequisites=[
                make_dsq(name='satellite_azimuth_angle'),
                make_dsq(name='satellite_zenith_angle'),
                make_dsq(name='solar_azimuth_angle'),
                make_dsq(name='solar_zenith_angle')
            ],
            name='I01',
            prerequisites=[],
            wavelength=(0.6, 0.64, 0.68),
            resolution=371,
            calibration='reflectance',
            modifiers=('sunz_corrected_iband', 'rayleigh_corrected_crefl_iband'),
            sensor='viirs',
            url=url,
            dem_sds=dem_sds,
        )

        area, data = self.data_area_ref_corrector()
        c01 = _make_viirs_xarray(data, area, 'I01', 'toa_bidirectional_reflectance',
                                 wavelength=(0.6, 0.64, 0.68), units='%',
                                 calibration='reflectance')
        c02 = _make_viirs_xarray(data, area, 'satellite_azimuth_angle', 'sensor_azimuth_angle')
        c02.data = c02.data.rechunk((1, -1))
        c03 = _make_viirs_xarray(data, area, 'satellite_zenith_angle', 'sensor_zenith_angle')
        c04 = _make_viirs_xarray(data, area, 'solar_azimuth_angle', 'solar_azimuth_angle')
        c05 = _make_viirs_xarray(data, area, 'solar_zenith_angle', 'solar_zenith_angle')

        with dem_mock_cm(tmpdir, url):
            res = ref_cor([c01], [c02, c03, c04, c05])

        # make sure it can actually compute
        res.compute()
