import sys
import numpy as np
import xarray as xr

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


class FakeDataset(object):
    def __init__(self, info, attrs):
        for var_name, var_data in list(info.items()):
            if isinstance(var_data, np.ndarray):
                info[var_name] = xr.DataArray(var_data)
        self.info = info
        self.attrs = attrs

    def __getitem__(self, key):
        return self.info[key]

    def rename(self, *args, **kwargs):
        return self

    def close(self):
        return


class TestVIRRL1BReader(unittest.TestCase):

    def setUp(self):
        from satpy.readers.virr_l1b import VIRR_L1B
        import datetime
        self.reader = VIRR_L1B('/Users/wroberts/Documents/VIRR_data/FY3B/' +
                               'tf2018359214943.FY3B-L_VIRRX/tf2018359214943.FY3B-L_VIRRX_L1B.HDF',
                               {'start_time': datetime.datetime(2018, 12, 25, 21, 49, 43), 'platform_id': 'FY3B'},
                               {'file_reader': VIRR_L1B,
                                'file_patterns': ['tf{start_time:%Y%j%H%M%S}.{platform_id}-L_VIRRX_L1B.HDF'],
                                'geolocation_extension': '', 'file_type': 'virr_l1b'})

    def test_get_dataset(self):
        from satpy import DatasetID
        ds = self.reader.get_dataset(DatasetID(name='latitude', wavelength=None, resolution=1000, polarization=None,
                                               calibration=None, level=None, modifiers=()),
                                     {'name': 'latitude', 'resolution': 1000, 'file_type': ['virr_l1b', 'virr_geoxx'],
                                      'file_key': 'Latitude', 'standard_name': 'latitude', 'wavelength': None,
                                      'polarization': None, 'calibration': None, 'level': None, 'modifiers': ()})
        print(ds)
        self.assertEqual(ds.shape, (2047, 2048))
        self.assertEqual(ds.attrs['standard_name'], 'latitude')
        self.assertRaises(AttributeError, self.reader.get_dataset, DatasetID(name='C01'), {})

    def test_time(self):
        import datetime
        self.assertEqual(self.reader.start_time, datetime.datetime(2018, 12, 25, 21, 41, 47, 90000))
        self.assertEqual(self.reader.end_time, datetime.datetime(2018, 12, 25, 21, 47, 28, 254000))
        pass


def suite():
    """The test suite for test_virr_l1b."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestVIRRL1BReader))
    return mysuite
