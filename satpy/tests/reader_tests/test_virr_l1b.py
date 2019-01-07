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
    """Swap-in HDF5 File Handler"""

    def make_test_data(self, dimension):
        if len(dimension) == 2:
            dims = ['y', 'x']
        else:
            dims = ['phony_dim', 'y', 'x']
        return xr.DataArray(da.from_array(np.ones([dim for dim in dimension], dtype=np.float32) * 10, [dim for dim in
                                                                                                       dimension]),
                            dims=dims)

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content"""
        dim_0 = 2047
        dim_1 = 2048
        file_content = {
            # General satellite data.
            '/attr/Day Or Night Flag': 'D', '/attr/Observing Beginning Date': '2018-12-25',
            '/attr/Observing Beginning Time': '21:41:47.090', '/attr/Observing Ending Date': '2018-12-25',
            '/attr/Observing Ending Time': '21:47:28.254', '/attr/Satellite Name': filename_info['platform_id'],
            '/attr/Sensor Identification Code': 'VIRR', 'Latitude': self.make_test_data([dim_0, dim_1]),
            'Latitude/attr/Intercept': np.array(0), 'Latitude/attr/Slope': np.array(1),
            'Latitude/attr/valid_range': [-90., 90.], 'Latitude/attr/units': 'degrees',
            'Longitude': self.make_test_data([dim_0, dim_1]), 'Longitude/attr/Intercept': 0,
            'Longitude/attr/Slope': 1, 'Longitude/attr/units': 'degrees',
            'Longitude/attr/valid_range': [-180., 180.],
            # Emissive data.
            'EV_Emissive': self.make_test_data([3, dim_0, dim_1]), 'EV_Emissive/attr/valid_range': [0, 50000],
            'Emissive_Radiance_Scales': self.make_test_data([dim_0, dim_1]),
            'EV_Emissive/attr/units': 'milliWstts/m^2/cm^(-1)/steradian',
            'Emissive_Radiance_Offsets': self.make_test_data([dim_0, dim_1]),
            '/attr/Emmisive_Centroid_Wave_Number': [2610.31, 917.6268, 836.2546],
            # Reflectance data.
            'EV_RefSB': self.make_test_data([7, dim_0, dim_1]), 'EV_RefSB/attr/valid_range': [0, 32767],
            '/attr/RefSB_Cal_Coefficients': np.ones([dim for dim in [dim_0, dim_1]], dtype=np.float32) * 2,
            'EV_RefSB/attr/units': 'none',
        }
        return file_content


class TestVIRRL1BReader(unittest.TestCase):
    """Test VIRR L1B Reader"""
    yaml_file = "virr_l1b.yaml"

    def setUp(self):
        """Wrap HDF5 file handler with our own fake handler"""
        from satpy.readers.virr_l1b import VIRR_L1B
        from satpy.config import config_search_paths
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(VIRR_L1B, '__bases__', (FakeHDF5FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the HDF5 file handler"""
        self.p.stop()

    def test_get_dataset(self):
        from satpy.readers import load_reader
        import datetime

        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames(['tf2018359214943.FY3B-L_VIRRX/tf2018359214943.FY3B-L_VIRRX_L1B.HDF'])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

        data_values = {'R1': 22.0, 'R2': 22.0, 'R3': 22.0, 'R4': 22.0, 'R5': 22.0, 'R6': 22.0, 'R7': 22.0,
                       'E1': 496.542155, 'E2': 297.444511, 'E3': 288.956557, }
        datasets = r.load(['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'E1', 'E2', 'E3'])
        for dataset in datasets:
            ds = datasets[dataset.name]
            attributes = ds.attrs
            if 'R' in dataset.name:
                self.assertEqual('%', attributes['units'])
                self.assertEqual('reflectance', attributes['calibration'])
                self.assertEqual('toa_bidirectional_reflectance', attributes['standard_name'])
                self.assertTrue(attributes['band_index'] in range(7))
            else:
                self.assertEqual('milliWstts/m^2/cm^(-1)/steradian', attributes['units'])
                self.assertEqual('brightness_temperature', attributes['calibration'])
                self.assertEqual('toa_brightness_temperature', attributes['standard_name'])
                self.assertTrue(attributes['band_index'] in range(3))
            self.assertEqual(data_values[dataset.name],
                             round(float(np.array(ds[ds.shape[0] // 2][ds.shape[1] // 2])), 6))
            self.assertEqual('VIRR', attributes['sensor'])
            self.assertEqual('FY3B', attributes['platform_name'])
            self.assertEqual(1000, attributes['resolution'])
            self.assertEqual('virr_l1b', attributes['file_type'])
            self.assertEqual(1, attributes['level'])
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
