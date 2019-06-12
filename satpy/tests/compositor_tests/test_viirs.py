#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Satpy developers
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
"""Tests for VIIRS compositors.
"""

import sys

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class TestVIIRSComposites(unittest.TestCase):
    """Test VIIRS-specific composites."""

    def data_area_ref_corrector(self):
        import dask.array as da
        import numpy as np
        from pyresample.geometry import AreaDefinition
        rows = 5
        cols = 10
        area = AreaDefinition(
            'some_area_name', 'On-the-fly area', 'geosabii',
            {'a': '6378137.0', 'b': '6356752.31414', 'h': '35786023.0', 'lon_0': '-89.5', 'proj': 'geos', 'sweep': 'x',
             'units': 'm'},
            cols, rows,
            (-5434894.954752679, -5434894.964451744, 5434894.964451744, 5434894.954752679))

        dnb = np.zeros((rows, cols)) + 25
        dnb[3, :] += 25
        dnb[4:, :] += 50
        dnb = da.from_array(dnb, chunks=100)
        return area, dnb

    def test_load_composite_yaml(self):
        """Test loading the yaml for this sensor."""
        from satpy.composites import CompositorLoader
        cl = CompositorLoader()
        cl.load_sensor_composites('viirs')

    def test_histogram_dnb(self):
        """Test the 'histogram_dnb' compositor."""
        import xarray as xr
        import dask.array as da
        import numpy as np
        from satpy.composites.viirs import HistogramDNB
        from pyresample.geometry import AreaDefinition
        rows = 5
        cols = 10
        area = AreaDefinition(
            'test', 'test', 'test',
            {'proj': 'eqc', 'lon_0': 0.0,
             'lat_0': 0.0},
            cols, rows,
            (-20037508.34, -10018754.17, 20037508.34, 10018754.17))

        comp = HistogramDNB('histogram_dnb', prerequisites=('dnb',),
                            standard_name='toa_outgoing_radiance_per_'
                                          'unit_wavelength')
        dnb = np.zeros((rows, cols)) + 0.25
        dnb[3, :] += 0.25
        dnb[4:, :] += 0.5
        dnb = da.from_array(dnb, chunks=25)
        c01 = xr.DataArray(dnb,
                           dims=('y', 'x'),
                           attrs={'name': 'DNB', 'area': area})
        # data changes by row, sza changes by col for testing
        sza = np.zeros((rows, cols)) + 70.0
        sza[:, 3] += 20.0
        sza[:, 4:] += 45.0
        sza = da.from_array(sza, chunks=25)
        c02 = xr.DataArray(sza,
                           dims=('y', 'x'),
                           attrs={'name': 'solar_zenith_angle', 'area': area})
        res = comp((c01, c02))
        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res.data, da.Array)
        self.assertEqual(res.attrs['name'], 'histogram_dnb')
        self.assertEqual(res.attrs['standard_name'],
                         'equalized_radiance')
        data = res.compute()
        unique_values = np.unique(data)
        np.testing.assert_allclose(unique_values, [0.5994, 0.7992, 0.999], rtol=1e-3)

    def test_adaptive_dnb(self):
        """Test the 'adaptive_dnb' compositor."""
        import xarray as xr
        import dask.array as da
        import numpy as np
        from satpy.composites.viirs import AdaptiveDNB
        from pyresample.geometry import AreaDefinition
        rows = 5
        cols = 10
        area = AreaDefinition(
            'test', 'test', 'test',
            {'proj': 'eqc', 'lon_0': 0.0,
             'lat_0': 0.0},
            cols, rows,
            (-20037508.34, -10018754.17, 20037508.34, 10018754.17))

        comp = AdaptiveDNB('adaptive_dnb', prerequisites=('dnb',),
                           standard_name='toa_outgoing_radiance_per_'
                                         'unit_wavelength')
        dnb = np.zeros((rows, cols)) + 0.25
        dnb[3, :] += 0.25
        dnb[4:, :] += 0.5
        dnb = da.from_array(dnb, chunks=25)
        c01 = xr.DataArray(dnb,
                           dims=('y', 'x'),
                           attrs={'name': 'DNB', 'area': area})
        sza = np.zeros((rows, cols)) + 70.0
        sza[:, 3] += 20.0
        sza[:, 4:] += 45.0
        sza = da.from_array(sza, chunks=25)
        c02 = xr.DataArray(sza,
                           dims=('y', 'x'),
                           attrs={'name': 'solar_zenith_angle', 'area': area})
        res = comp((c01, c02))
        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res.data, da.Array)
        self.assertEqual(res.attrs['name'], 'adaptive_dnb')
        self.assertEqual(res.attrs['standard_name'],
                         'equalized_radiance')
        data = res.compute()
        np.testing.assert_allclose(data.data, 0.999, rtol=1e-4)

    def test_erf_dnb(self):
        """Test the 'dynamic_dnb' or ERF DNB compositor."""
        import xarray as xr
        import dask.array as da
        import numpy as np
        from satpy.composites.viirs import ERFDNB
        from pyresample.geometry import AreaDefinition
        rows = 5
        cols = 10
        area = AreaDefinition(
            'test', 'test', 'test',
            {'proj': 'eqc', 'lon_0': 0.0,
             'lat_0': 0.0},
            cols, rows,
            (-20037508.34, -10018754.17, 20037508.34, 10018754.17))

        comp = ERFDNB('dynamic_dnb', prerequisites=('dnb',),
                      standard_name='toa_outgoing_radiance_per_'
                                    'unit_wavelength')
        dnb = np.zeros((rows, cols)) + 0.25
        dnb[3, :] += 0.25
        dnb[4:, :] += 0.5
        dnb = da.from_array(dnb, chunks=25)
        c01 = xr.DataArray(dnb,
                           dims=('y', 'x'),
                           attrs={'name': 'DNB', 'area': area})
        sza = np.zeros((rows, cols)) + 70.0
        sza[:, 3] += 20.0
        sza[:, 4:] += 45.0
        sza = da.from_array(sza, chunks=25)
        c02 = xr.DataArray(sza,
                           dims=('y', 'x'),
                           attrs={'name': 'solar_zenith_angle', 'area': area})
        lza = da.from_array(sza, chunks=25)
        c03 = xr.DataArray(lza,
                           dims=('y', 'x'),
                           attrs={'name': 'lunar_zenith_angle', 'area': area})
        mif = xr.DataArray(da.zeros((5,), chunks=5) + 0.1,
                           dims=('y',),
                           attrs={'name': 'moon_illumination_fraction', 'area': area})
        res = comp((c01, c02, c03, mif))
        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res.data, da.Array)
        self.assertEqual(res.attrs['name'], 'dynamic_dnb')
        self.assertEqual(res.attrs['standard_name'],
                         'equalized_radiance')
        data = res.compute()
        unique = np.unique(data)
        np.testing.assert_allclose(unique, [0.00000000e+00, 1.00446703e-01, 1.64116082e-01, 2.09233451e-01,
                                            1.43916324e+02, 2.03528498e+02, 2.49270516e+02])

    def test_hncc_dnb(self):
        """Test the 'hncc_dnb' compositor."""
        import xarray as xr
        import dask.array as da
        import numpy as np
        from satpy.composites.viirs import NCCZinke
        from pyresample.geometry import AreaDefinition
        rows = 5
        cols = 10
        area = AreaDefinition(
            'test', 'test', 'test',
            {'proj': 'eqc', 'lon_0': 0.0,
             'lat_0': 0.0},
            cols, rows,
            (-20037508.34, -10018754.17, 20037508.34, 10018754.17))

        comp = NCCZinke('hncc_dnb', prerequisites=('dnb',),
                        standard_name='toa_outgoing_radiance_per_'
                                      'unit_wavelength')
        dnb = np.zeros((rows, cols)) + 0.25
        dnb[3, :] += 0.25
        dnb[4:, :] += 0.5
        dnb = da.from_array(dnb, chunks=25)
        c01 = xr.DataArray(dnb,
                           dims=('y', 'x'),
                           attrs={'name': 'DNB', 'area': area})
        sza = np.zeros((rows, cols)) + 70.0
        sza[:, 3] += 20.0
        sza[:, 4:] += 45.0
        sza = da.from_array(sza, chunks=25)
        c02 = xr.DataArray(sza,
                           dims=('y', 'x'),
                           attrs={'name': 'solar_zenith_angle', 'area': area})
        lza = da.from_array(sza, chunks=25)
        c03 = xr.DataArray(lza,
                           dims=('y', 'x'),
                           attrs={'name': 'lunar_zenith_angle', 'area': area})
        mif = xr.DataArray(da.zeros((5,), chunks=5) + 0.1,
                           dims=('y',),
                           attrs={'name': 'moon_illumination_fraction', 'area': area})
        res = comp((c01, c02, c03, mif))
        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res.data, da.Array)
        self.assertEqual(res.attrs['name'], 'hncc_dnb')
        self.assertEqual(res.attrs['standard_name'],
                         'ncc_radiance')
        data = res.compute()
        unique = np.unique(data)
        np.testing.assert_allclose(
            unique, [3.48479712e-04, 6.96955799e-04, 1.04543189e-03, 4.75394738e-03,
                     9.50784532e-03, 1.42617433e-02, 1.50001560e+03, 3.00001560e+03,
                     4.50001560e+03])

    def test_reflectance_corrector_abi(self):
        import xarray as xr
        import dask.array as da
        import numpy as np
        from satpy.composites.viirs import ReflectanceCorrector
        from satpy import DatasetID
        ref_cor = ReflectanceCorrector(dem_filename='_fake.hdf', optional_prerequisites=[
            DatasetID(name='satellite_azimuth_angle'),
            DatasetID(name='satellite_zenith_angle'),
            DatasetID(name='solar_azimuth_angle'),
            DatasetID(name='solar_zenith_angle')], name='C01', prerequisites=[],
                                       wavelength=(0.45, 0.47, 0.49), resolution=1000, calibration='reflectance',
                                       modifiers=('sunz_corrected', 'rayleigh_corrected_crefl',), sensor='abi')

        self.assertEqual(ref_cor.attrs['modifiers'], ('sunz_corrected', 'rayleigh_corrected_crefl',))
        self.assertEqual(ref_cor.attrs['calibration'], 'reflectance')
        self.assertEqual(ref_cor.attrs['wavelength'], (0.45, 0.47, 0.49))
        self.assertEqual(ref_cor.attrs['name'], 'C01')
        self.assertEqual(ref_cor.attrs['resolution'], 1000)
        self.assertEqual(ref_cor.attrs['sensor'], 'abi')
        self.assertEqual(ref_cor.attrs['prerequisites'], [])
        self.assertEqual(ref_cor.attrs['optional_prerequisites'], [
            DatasetID(name='satellite_azimuth_angle'),
            DatasetID(name='satellite_zenith_angle'),
            DatasetID(name='solar_azimuth_angle'),
            DatasetID(name='solar_zenith_angle')])

        area, dnb = self.data_area_ref_corrector()
        c01 = xr.DataArray(dnb,
                           dims=('y', 'x'),
                           attrs={'satellite_longitude': -89.5, 'satellite_latitude': 0.0,
                                  'satellite_altitude': 35786.0234375, 'platform_name': 'GOES-16',
                                  'calibration': 'reflectance', 'units': '%', 'wavelength': (0.45, 0.47, 0.49),
                                  'name': 'C01', 'resolution': 1000, 'sensor': 'abi',
                                  'start_time': '2017-09-20 17:30:40.800000', 'end_time': '2017-09-20 17:41:17.500000',
                                  'area': area, 'ancillary_variables': []})
        res = ref_cor([c01], [])

        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res.data, da.Array)
        self.assertEqual(res.attrs['satellite_longitude'], -89.5)
        self.assertEqual(res.attrs['satellite_latitude'], 0.0)
        self.assertEqual(res.attrs['satellite_altitude'], 35786.0234375)
        self.assertEqual(res.attrs['modifiers'], ('sunz_corrected', 'rayleigh_corrected_crefl',))
        self.assertEqual(res.attrs['platform_name'], 'GOES-16')
        self.assertEqual(res.attrs['calibration'], 'reflectance')
        self.assertEqual(res.attrs['units'], '%')
        self.assertEqual(res.attrs['wavelength'], (0.45, 0.47, 0.49))
        self.assertEqual(res.attrs['name'], 'C01')
        self.assertEqual(res.attrs['resolution'], 1000)
        self.assertEqual(res.attrs['sensor'], 'abi')
        self.assertEqual(res.attrs['start_time'], '2017-09-20 17:30:40.800000')
        self.assertEqual(res.attrs['end_time'], '2017-09-20 17:41:17.500000')
        self.assertEqual(res.attrs['area'], area)
        self.assertEqual(res.attrs['ancillary_variables'], [])
        data = res.values
        self.assertLess(abs(np.mean(data) - 29.907390988422513), 1e-10)
        self.assertEqual(data.shape, (5, 10))
        unique = np.unique(data)
        np.testing.assert_allclose(unique, [-1.0, 4.210745457958135, 6.7833906076177595, 8.730371329824473,
                                            10.286627569545209, 11.744159436709374, 12.20226097829902,
                                            13.501444598985305, 15.344399223932212, 17.173329483996515,
                                            17.28798660754271, 18.29594550575925, 19.076835059905125,
                                            19.288331720959864, 19.77043407084455, 19.887082168377006,
                                            20.091028778326375, 20.230341149334617, 20.457671064690196,
                                            20.82686905639114, 21.021094816441195, 21.129963777952124,
                                            21.94957397026227, 41.601857910095575, 43.963919057675504,
                                            46.21672174361075, 46.972099490462085, 47.497072794632835,
                                            47.80393007974336, 47.956765988770385, 48.043025685032106,
                                            51.909142813383916, 58.8234273736508, 68.84706145641482, 69.91085190887961,
                                            71.10179768327806, 71.33161009169649, 78.81291424983952])

    def test_reflectance_corrector_viirs(self):
        import xarray as xr
        import dask.array as da
        import numpy as np
        import datetime
        from satpy.composites.viirs import ReflectanceCorrector
        from satpy import DatasetID
        ref_cor = ReflectanceCorrector(dem_filename='_fake.hdf', optional_prerequisites=[
         DatasetID(name='satellite_azimuth_angle'),
         DatasetID(name='satellite_zenith_angle'),
         DatasetID(name='solar_azimuth_angle'),
         DatasetID(name='solar_zenith_angle')],
                                       name='I01', prerequisites=[], wavelength=(0.6, 0.64, 0.68), resolution=371,
                                       calibration='reflectance', modifiers=('sunz_corrected_iband',
                                                                             'rayleigh_corrected_crefl_iband'),
                                       sensor='viirs')

        self.assertEqual(ref_cor.attrs['modifiers'], ('sunz_corrected_iband', 'rayleigh_corrected_crefl_iband'))
        self.assertEqual(ref_cor.attrs['calibration'], 'reflectance')
        self.assertEqual(ref_cor.attrs['wavelength'], (0.6, 0.64, 0.68))
        self.assertEqual(ref_cor.attrs['name'], 'I01')
        self.assertEqual(ref_cor.attrs['resolution'], 371)
        self.assertEqual(ref_cor.attrs['sensor'], 'viirs')
        self.assertEqual(ref_cor.attrs['prerequisites'], [])
        self.assertEqual(ref_cor.attrs['optional_prerequisites'], [
            DatasetID(name='satellite_azimuth_angle'),
            DatasetID(name='satellite_zenith_angle'),
            DatasetID(name='solar_azimuth_angle'),
            DatasetID(name='solar_zenith_angle')])

        area, dnb = self.data_area_ref_corrector()

        def make_xarray(self, file_key, name, standard_name, wavelength=None, units='degrees', calibration=None,
                        file_type=['gitco', 'gimgo']):
            return xr.DataArray(dnb, dims=('y', 'x'),
                                attrs={'start_orbit': 1708, 'end_orbit': 1708, 'wavelength': wavelength, 'level': None,
                                       'modifiers': None, 'calibration': calibration, 'file_key': file_key,
                                       'resolution': 371, 'file_type': file_type, 'name': name,
                                       'standard_name': standard_name, 'platform_name': 'Suomi-NPP',
                                       'polarization': None, 'sensor': 'viirs', 'units': units,
                                       'start_time': datetime.datetime(2012, 2, 25, 18, 1, 24, 570942),
                                       'end_time': datetime.datetime(2012, 2, 25, 18, 11, 21, 175760), 'area': area,
                                       'ancillary_variables': []})
        c01 = make_xarray(self, None, 'I01', 'toa_bidirectional_reflectance', wavelength=(0.6, 0.64, 0.68), units='%',
                          calibration='reflectance', file_type='svi01')
        c02 = make_xarray(self, 'All_Data/{file_group}_All/SatelliteAzimuthAngle', 'satellite_azimuth_angle',
                          'sensor_azimuth_angle')
        c03 = make_xarray(self, 'All_Data/{file_group}_All/SatelliteZenithAngle', 'satellite_zenith_angle',
                          'sensor_zenith_angle')
        c04 = make_xarray(self, 'All_Data/{file_group}_All/SolarAzimuthAngle', 'solar_azimuth_angle',
                          'solar_azimuth_angle')
        c05 = make_xarray(self, 'All_Data/{file_group}_All/SolarZenithAngle', 'solar_zenith_angle',
                          'solar_zenith_angle')
        res = ref_cor([c01], [c02, c03, c04, c05])

        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res.data, da.Array)
        self.assertEqual(res.attrs['wavelength'], (0.6, 0.64, 0.68))
        self.assertEqual(res.attrs['modifiers'], ('sunz_corrected_iband', 'rayleigh_corrected_crefl_iband'))
        self.assertEqual(res.attrs['calibration'], 'reflectance')
        self.assertEqual(res.attrs['resolution'], 371)
        self.assertEqual(res.attrs['file_type'], 'svi01')
        self.assertEqual(res.attrs['name'], 'I01')
        self.assertEqual(res.attrs['standard_name'], 'toa_bidirectional_reflectance')
        self.assertEqual(res.attrs['platform_name'], 'Suomi-NPP')
        self.assertEqual(res.attrs['sensor'], 'viirs')
        self.assertEqual(res.attrs['units'], '%')
        self.assertEqual(res.attrs['start_time'], datetime.datetime(2012, 2, 25, 18, 1, 24, 570942))
        self.assertEqual(res.attrs['end_time'], datetime.datetime(2012, 2, 25, 18, 11, 21, 175760))
        self.assertEqual(res.attrs['area'], area)
        self.assertEqual(res.attrs['ancillary_variables'], [])
        data = res.values
        self.assertLess(abs(np.mean(data) - 40.7578684169142), 1e-10)
        self.assertEqual(data.shape, (5, 10))
        unique = np.unique(data)
        np.testing.assert_allclose(unique, [25.20341702519979, 52.38819447051263, 75.79089653845898])

    def test_reflectance_corrector_modis(self):
        import xarray as xr
        import dask.array as da
        import numpy as np
        import datetime
        from satpy.composites.viirs import ReflectanceCorrector
        from satpy import DatasetID
        sataa_did = DatasetID(name='satellite_azimuth_angle')
        satza_did = DatasetID(name='satellite_zenith_angle')
        solaa_did = DatasetID(name='solar_azimuth_angle')
        solza_did = DatasetID(name='solar_zenith_angle')
        ref_cor = ReflectanceCorrector(
            dem_filename='_fake.hdf', optional_prerequisites=[sataa_did, satza_did, solaa_did, solza_did], name='1',
            prerequisites=[], wavelength=(0.62, 0.645, 0.67), resolution=250, calibration='reflectance',
            modifiers=('sunz_corrected', 'rayleigh_corrected_crefl'), sensor='modis')
        self.assertEqual(ref_cor.attrs['modifiers'], ('sunz_corrected', 'rayleigh_corrected_crefl'))
        self.assertEqual(ref_cor.attrs['calibration'], 'reflectance')
        self.assertEqual(ref_cor.attrs['wavelength'], (0.62, 0.645, 0.67))
        self.assertEqual(ref_cor.attrs['name'], '1')
        self.assertEqual(ref_cor.attrs['resolution'], 250)
        self.assertEqual(ref_cor.attrs['sensor'], 'modis')
        self.assertEqual(ref_cor.attrs['prerequisites'], [])
        self.assertEqual(ref_cor.attrs['optional_prerequisites'], [
            DatasetID(name='satellite_azimuth_angle'),
            DatasetID(name='satellite_zenith_angle'),
            DatasetID(name='solar_azimuth_angle'),
            DatasetID(name='solar_zenith_angle')])

        area, dnb = self.data_area_ref_corrector()

        def make_xarray(self, name, calibration, wavelength=None, modifiers=None, resolution=1000,
                        file_type='hdf_eos_geo'):
            return xr.DataArray(dnb,
                                dims=('y', 'x'),
                                attrs={'wavelength': wavelength, 'level': None, 'modifiers': modifiers,
                                       'calibration': calibration, 'resolution': resolution, 'file_type': file_type,
                                       'name': name, 'coordinates': ['longitude', 'latitude'],
                                       'platform_name': 'EOS-Aqua', 'polarization': None, 'sensor': 'modis',
                                       'units': '%', 'start_time': datetime.datetime(2012, 8, 13, 18, 46, 1, 439838),
                                       'end_time': datetime.datetime(2012, 8, 13, 18, 57, 47, 746296), 'area': area,
                                       'ancillary_variables': []})
        c01 = make_xarray(self, '1', 'reflectance', wavelength=(0.62, 0.645, 0.67), modifiers='sunz_corrected',
                          resolution=500, file_type='hdf_eos_data_500m')
        c02 = make_xarray(self, 'satellite_azimuth_angle', None)
        c03 = make_xarray(self, 'satellite_zenith_angle', None)
        c04 = make_xarray(self, 'solar_azimuth_angle', None)
        c05 = make_xarray(self, 'solar_zenith_angle', None)
        res = ref_cor([c01], [c02, c03, c04, c05])

        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res.data, da.Array)
        self.assertEqual(res.attrs['wavelength'], (0.62, 0.645, 0.67))
        self.assertEqual(res.attrs['modifiers'], ('sunz_corrected', 'rayleigh_corrected_crefl',))
        self.assertEqual(res.attrs['calibration'], 'reflectance')
        self.assertEqual(res.attrs['resolution'], 500)
        self.assertEqual(res.attrs['file_type'], 'hdf_eos_data_500m')
        self.assertEqual(res.attrs['name'], '1')
        self.assertEqual(res.attrs['platform_name'], 'EOS-Aqua')
        self.assertEqual(res.attrs['sensor'], 'modis')
        self.assertEqual(res.attrs['units'], '%')
        self.assertEqual(res.attrs['start_time'], datetime.datetime(2012, 8, 13, 18, 46, 1, 439838))
        self.assertEqual(res.attrs['end_time'], datetime.datetime(2012, 8, 13, 18, 57, 47, 746296))
        self.assertEqual(res.attrs['area'], area)
        self.assertEqual(res.attrs['ancillary_variables'], [])
        data = res.values
        if abs(np.mean(data) - 38.734365117099145) >= 1e-10:
            raise AssertionError('{} is not within {} of {}'.format(np.mean(data), 1e-10, 38.734365117099145))
        self.assertEqual(data.shape, (5, 10))
        unique = np.unique(data)
        np.testing.assert_allclose(unique, [24.641586, 50.431692, 69.315375])


def suite():
    """The test suite for test_ahi.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestVIIRSComposites))
    return mysuite
