import sys
import os
import numpy as np
from satpy.tests.reader_tests.test_hdf4_utils import FakeHDF4FileHandler
if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


DEFAULT_FILE_DTYPE = np.uint16
DEFAULT_FILE_SHAPE = (10, 300)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)


class FakeHDF4FileHandler2(FakeHDF4FileHandler):
    """Swap in HDF4 file handler"""
    def get_test_content(self, filename, filename_info, filename_type):
        """Mimic reader input file content"""
        file_content = {}
        file_content['/attr/Satellitename'] = filename_info['platform_shortname']
        file_content['/attr/SensorIdentifyCode'] = 'VIIRS'

        # only one dataset for the flood reader
        file_content['WaterDetection'] = DEFAULT_FILE_DATA
        file_content['WaterDetection/attr/_Fillvalue'] = 1
        file_content['WaterDetection/attr/scale_factor'] = 1.
        file_content['WaterDetection/attr/add_offset'] = 0.
        file_content['WaterDetection/attr/units'] = 'none'
        file_content['WaterDetection/shape'] = DEFAULT_FILE_SHAPE
        file_content['WaterDetection/attr/ProjectionMinLatitude'] = 15.
        file_content['WaterDetection/attr/ProjectionMaxLatitude'] = 68.
        file_content['WaterDetection/attr/ProjectionMinLongitude'] = -124.
        file_content['WaterDetection/attr/ProjectionMaxLongitude'] = -61.

        # convert tp xarrays
        from xarray import DataArray
        for key, val in file_content.items():
            if isinstance(val, np.ndarray):
                attrs = {}
                for a in ['_Fillvalue', 'units', 'ProjectionMinLatitude', 'ProjectionMaxLongitude',
                          'ProjectionMinLongitude', 'ProjectionMaxLatitude']:
                    if key + '/attr/' + a in file_content:
                        attrs[a] = file_content[key + '/attr/' + a]
                if val.ndim > 1:
                    file_content[key] = DataArray(val, dims=('fakeDim0', 'fakeDim1'), attrs=attrs)
                else:
                    file_content[key] = DataArray(val, attrs=attrs)

        if 'y' not in file_content['WaterDetection'].dims:
            file_content['WaterDetection'] = file_content['WaterDetection'].rename({'fakeDim0': 'x', 'fakeDim1': 'y'})
        return file_content


class TestVIIRSEDRFloodReader(unittest.TestCase):
    """Test VIIRS EDR Flood Reader"""
    yaml_file = 'viirs_edr_flood.yaml'

    def setUp(self):
        """Wrap HDF4 file handler with own fake file handler"""
        from satpy.config import config_search_paths
        from satpy.readers.viirs_edr_flood import VIIRSEDRFlood
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        self.p = mock.patch.object(VIIRSEDRFlood, '__bases__', (FakeHDF4FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the HDF4 file handler"""
        self.p.stop()

    def test_init(self):
        """Test basic init with no extra parameters"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'WATER_VIIRS_Prj_SVI_npp_d20180824_t1828213_e1839433_b35361_cspp_dev_10_300_01.hdf'
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        self.assertTrue(r.file_handlers)

    def test_load_dataset(self):
        """Test loading all datasets"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'WATER_VIIRS_Prj_SVI_npp_d20180824_t1828213_e1839433_b35361_cspp_dev_10_300_01.hdf'
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['WaterDetection'])
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            self.assertEqual(v.attrs['units'], 'none')


def suite():
    """The test suite for test_viirs_flood
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestVIIRSEDRFloodReader))

    return mysuite
