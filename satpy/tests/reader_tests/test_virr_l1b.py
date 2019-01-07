from satpy.tests.reader_tests.test_hdf5_utils import FakeHDF5FileHandler
import sys
import numpy as np
import dask.array as da
import xarray as xr
import os

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


class FakeHDF5FileHandler2(FakeHDF5FileHandler):
    """Swap-in HDF5 File Handler."""

    def make_test_data(self, dims):
        return xr.DataArray(da.from_array(np.ones([dim for dim in dims], dtype=np.float32) * 10, [dim for dim in dims]))

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        dim_0 = 2047
        dim_1 = 2048
        if filename_info['platform_id'] == 'FY3B':
            return {
                # Satellite data.
                '/attr/Day Or Night Flag': 'D', '/attr/Observing Beginning Date': '2018-12-25',
                '/attr/Observing Beginning Time': '21:41:47.090', '/attr/Observing Ending Date': '2018-12-25',
                '/attr/Observing Ending Time': '21:47:28.254', '/attr/Satellite Name': filename_info['platform_id'],
                '/attr/Sensor Identification Code': 'VIRR', 'Latitude': self.make_test_data([dim_0, dim_1]),
                'Latitude/attr/Intercept': np.array(0), 'Latitude/attr/Slope': np.array(1),
                'Latitude/attr/valid_range': [-90., 90.], 'Latitude/attr/units': 'degrees',
                'Longitude': self.make_test_data([dim_0, dim_1]), 'Longitude/attr/Intercept': 0,
                'Longitude/attr/Slope': 1, 'Longitude/attr/units': 'degrees',
                'Longitude/attr/valid_range': [-180., 180.],
                # Solar zenith data.
                'SolarZenith': self.make_test_data([dim_0, dim_1]), 'SolarZenith/attr/valid_range': [0, 18000],
                'SolarZenith/attr/Slope': .01, 'SolarZenith/attr/Intercept': 0.,
                # Sensor zenith data.
                'SensorZenith': self.make_test_data([dim_0, dim_1]), 'SensorZenith/attr/valid_range': [0, 18000],
                'SensorZenith/attr/Slope': .01, 'SensorZenith/attr/Intercept': 0.,
                # Solar azimuth data.
                'SolarAzimuth': self.make_test_data([dim_0, dim_1]), 'SolarAzimuth/attr/valid_range': [0, 18000],
                'SolarAzimuth/attr/Slope': .01, 'SolarAzimuth/attr/Intercept': 0.,
                # Sensor azimuth data.
                'SensorAzimuth': self.make_test_data([dim_0, dim_1]), 'SensorAzimuth/attr/valid_range': [0, 18000],
                'SensorAzimuth/attr/Slope': .01, 'SensorAzimuth/attr/Intercept': 0.,
                # Emissive data.
                'EV_Emissive': self.make_test_data([3, dim_0, dim_1]), 'EV_Emissive/attr/valid_range': [0, 50000],
                'Emissive_Radiance_Scales': self.make_test_data([dim_0, dim_1]),
                'EV_Emissive/attr/units': 'milliWstts/m^2/cm^(-1)/steradian',
                'Emissive_Radiance_Offsets': self.make_test_data([dim_0, dim_1]),
                '/attr/Emmisive_Centroid_Wave_Number': [2610.31, 917.6268, 836.2546],
                # Reflectance data.
                'EV_RefSB': self.make_test_data([7, dim_0, dim_1]), 'EV_RefSB/attr/valid_range': [0, 32767],
                '/attr/RefSB_Cal_Coefficients': np.ones([dim for dim in [dim_0, dim_1]], dtype=np.float32) * 2,
                'EV_RefSB/attr/units': 'none'
            }
        return {
            # Satellite data.
            '/attr/Day Or Night Flag': 'D', '/attr/Observing Beginning Date': '2018-12-25',
            '/attr/Observing Beginning Time': '21:41:47.090', '/attr/Observing Ending Date': '2018-12-25',
            '/attr/Observing Ending Time': '21:47:28.254', '/attr/Satellite Name': filename_info['platform_id'],
            '/attr/Sensor Identification Code': 'VIRR', 'Latitude': self.make_test_data([dim_0, dim_1]),
            'Latitude/attr/Intercept': np.array(0), 'Latitude/attr/Slope': np.array(1),
            'Latitude/attr/valid_range': [-90., 90.], 'Latitude/attr/units': 'degrees',
            'Longitude': self.make_test_data([dim_0, dim_1]), 'Longitude/attr/Intercept': 0,
            'Longitude/attr/Slope': 1, 'Longitude/attr/units': 'degrees',
            'Longitude/attr/valid_range': [-180., 180.],
            # Solar zenith data.
            'Geolocation/SolarZenith': self.make_test_data([dim_0, dim_1]),
            'Geolocation/SolarZenith/attr/valid_range': [-18000, 18000],
            'Geolocation/SolarZenith/attr/Slope': .01, 'Geolocation/SolarZenith/attr/Intercept': 0.,
            # Sensor zenith data.
            'Geolocation/SensorZenith': self.make_test_data([dim_0, dim_1]),
            'Geolocation/SensorZenith/attr/valid_range': [-18000, 18000], 'Geolocation/SensorZenith/attr/Slope': .01,
            'Geolocation/SensorZenith/attr/Intercept': 0.,
            # Solar azimuth data.
            'Geolocation/SolarAzimuth': self.make_test_data([dim_0, dim_1]),
            'Geolocation/SolarAzimuth/attr/valid_range': [0, 18000], 'Geolocation/SolarAzimuth/attr/Slope': .01,
            'Geolocation/SolarAzimuth/attr/Intercept': 0.,
            # Sensor azimuth data.
            'Geolocation/SensorAzimuth': self.make_test_data([dim_0, dim_1]),
            'Geolocation/SensorAzimuth/attr/valid_range': [0, 18000], 'Geolocation/SensorAzimuth/attr/Slope': .01,
            'Geolocation/SensorAzimuth/attr/Intercept': 0.,
            # Emissive data.
            'Data/EV_Emissive': self.make_test_data([3, dim_0, dim_1]), 'Data/EV_Emissive/attr/valid_range': [0, 50000],
            'Data/Emissive_Radiance_Scales': self.make_test_data([dim_0, dim_1]),
            'Data/EV_Emissive/attr/units': 'milliWstts/m^2/cm^(-1)/steradian',
            'Data/Emissive_Radiance_Offsets': self.make_test_data([dim_0, dim_1]),
            '/attr/Emissive_Centroid_Wave_Number': [2610.31, 917.6268, 836.2546],
            # Reflectance data.
            'Data/EV_RefSB': self.make_test_data([7, dim_0, dim_1]), 'Data/EV_RefSB/attr/valid_range': [0, 32767],
            '/attr/RefSB_Cal_Coefficients': np.ones([dim for dim in [dim_0, dim_1]], dtype=np.float32) * 2,
            'Data/EV_RefSB/attr/units': 'none'
        }


class TestVIRRL1BReader(unittest.TestCase):
    """Test VIRR L1B Reader."""
    yaml_file = "virr_l1b.yaml"

    def setUp(self):
        """Wrap HDF5 file handler with our own fake handler."""
        from satpy.readers import load_reader
        from satpy.readers.virr_l1b import VIRR_L1B
        from satpy.config import config_search_paths

        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(VIRR_L1B, '__bases__', (FakeHDF5FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

        self.FY3B_reader = load_reader(self.reader_configs)
        self.FY3C_reader = load_reader(self.reader_configs)

        FY3B_file = self.FY3B_reader.select_files_from_pathnames(['tf2018359214943.FY3B-L_VIRRX_L1B.HDF'])
        FY3C_files = self.FY3C_reader.select_files_from_pathnames(['tf2018359143912.FY3C-L_VIRRX_GEOXX.HDF',
                                                                   'tf2018359143912.FY3C-L_VIRRX_L1B.HDF'])
        self.assertTrue(len(FY3B_file), 1)
        self.assertTrue(len(FY3C_files), 2)
        self.FY3B_reader.create_filehandlers(FY3B_file)
        self.FY3C_reader.create_filehandlers(FY3C_files)
        # Make sure we have some files
        self.assertTrue(self.FY3B_reader.file_handlers)
        self.assertTrue(self.FY3C_reader.file_handlers)

    def tearDown(self):
        """Stop wrapping the HDF5 file handler."""
        self.p.stop()

    def band_tester(self, attributes, units, calibration, standard_name, file_type, band_index_size, resolution, level):
        self.assertEqual(units, attributes['units'])
        self.assertEqual(calibration, attributes['calibration'])
        self.assertEqual(standard_name, attributes['standard_name'])
        self.assertEqual(file_type, attributes['file_type'])
        self.assertTrue(attributes['band_index'] in range(band_index_size))
        self.assertEqual(resolution, attributes['resolution'])
        self.assertEqual(level, attributes['level'])
        self.assertEqual(('longitude', 'latitude'), attributes['coordinates'])

    def test_FY3B_file(self):
        import datetime
        band_values = {'R1': 22.0, 'R2': 22.0, 'R3': 22.0, 'R4': 22.0, 'R5': 22.0, 'R6': 22.0, 'R7': 22.0,
                       'E1': 496.542155, 'E2': 297.444511, 'E3': 288.956557, 'solar_zenith_angle': .1,
                       'satellite_zenith_angle': .1, 'solar_azimuth_angle': .1, 'satellite_azimuth_angle': .1,
                       'longitude': 10}
        datasets = self.FY3B_reader.load([band for band, val in band_values.items()])
        for dataset in datasets:
            ds = datasets[dataset.name]
            attributes = ds.attrs
            if 'R' in dataset.name:
                self.band_tester(attributes, '%', 'reflectance', 'toa_bidirectional_reflectance',
                                 'virr_l1b', 7,  1000, 1)
            elif 'E' in dataset.name:
                self.band_tester(attributes, 'milliWstts/m^2/cm^(-1)/steradian', 'brightness_temperature',
                                 'toa_brightness_temperature', 'virr_l1b', 3, 1000, 1)
            elif dataset.name in ['longitude', 'latitude']:
                self.assertEqual('degrees', attributes['units'])
                self.assertTrue(attributes['standard_name'] in ['longitude', 'latitude'])
                self.assertEqual(['virr_l1b', 'virr_geoxx'], attributes['file_type'])
                self.assertEqual(1000, attributes['resolution'])
            else:
                self.assertEqual(1, attributes['units'])
                self.assertTrue(attributes['standard_name'] in ['solar_zenith_angle', 'sensor_zenith_angle',
                                                                'solar_azimuth_angle', 'sensor_azimuth_angle'])
                self.assertEqual(['virr_geoxx', 'virr_l1b'], attributes['file_type'])
                self.assertEqual(('longitude', 'latitude'), attributes['coordinates'])
            self.assertEqual(band_values[dataset.name],
                             round(float(np.array(ds[ds.shape[0] // 2][ds.shape[1] // 2])), 6))
            self.assertEqual('VIRR', attributes['sensor'])
            self.assertEqual('FY3B', attributes['platform_name'])
            self.assertEqual(datetime.datetime(2018, 12, 25, 21, 41, 47, 90000), attributes['start_time'])
            self.assertEqual(datetime.datetime(2018, 12, 25, 21, 47, 28, 254000), attributes['end_time'])
            self.assertEqual((2047, 2048), datasets[dataset.name].shape)
            self.assertEqual(('y', 'x'), datasets[dataset.name].dims)

    def test_FY3C_file(self):
        import datetime
        band_values = {'R1': 22.0, 'R2': 22.0, 'R3': 22.0, 'R4': 22.0, 'R5': 22.0, 'R6': 22.0, 'R7': 22.0,
                       'E1': 496.542155, 'E2': 297.444511, 'E3': 288.956557, 'solar_zenith_angle': .1,
                       'satellite_zenith_angle': .1, 'solar_azimuth_angle': .1, 'satellite_azimuth_angle': .1}
        datasets = self.FY3C_reader.load([band for band, val in band_values.items()])
        for dataset in datasets:
            ds = datasets[dataset.name]
            attributes = ds.attrs
            if 'R' in dataset.name:
                self.band_tester(attributes, '%', 'reflectance', 'toa_bidirectional_reflectance',
                                 'virr_l1b', 7,  1000, 1)
            elif 'E' in dataset.name:
                self.band_tester(attributes, 'milliWstts/m^2/cm^(-1)/steradian', 'brightness_temperature',
                                 'toa_brightness_temperature', 'virr_l1b', 3, 1000, 1)
            else:
                self.assertEqual(1, attributes['units'])
                self.assertTrue(attributes['standard_name'] in ['solar_zenith_angle', 'sensor_zenith_angle',
                                                                'solar_azimuth_angle', 'sensor_azimuth_angle'])
                self.assertEqual(['virr_geoxx', 'virr_l1b'], attributes['file_type'])
            self.assertEqual(band_values[dataset.name],
                             round(float(np.array(ds[ds.shape[0] // 2][ds.shape[1] // 2])), 6))
            self.assertEqual('VIRR', attributes['sensor'])
            self.assertEqual('FY3C', attributes['platform_name'])
            self.assertEqual(('longitude', 'latitude'), attributes['coordinates'])
            self.assertEqual(datetime.datetime(2018, 12, 25, 21, 41, 47, 90000), attributes['start_time'])
            self.assertEqual(datetime.datetime(2018, 12, 25, 21, 47, 28, 254000), attributes['end_time'])
            self.assertEqual((2047, 2048), datasets[dataset.name].shape)
            self.assertEqual(('y', 'x'), datasets[dataset.name].dims)


def suite():
    """The test suite for test_virr_l1b."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestVIRRL1BReader))
    return mysuite
