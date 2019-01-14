#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module for testing the satpy.readers.nc_slstr module.
"""
import os
import sys

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    import unittest.mock as mock
except ImportError:
    import mock


class TestSLSTRReader(unittest.TestCase):
    """Test various nc_slstr file handlers."""

    @mock.patch('xarray.open_dataset')
    def test_instantiate(self, mocked_dataset):
        """Test initialization of file handlers."""
        from satpy.readers.slstr_l1b import NCSLSTR1B, NCSLSTRGeo, NCSLSTRAngles, NCSLSTRFlag
        from satpy import DatasetID

        ds_id = DatasetID(name='foo')
        filename_info = {'mission_id': 'S3A', 'dataset_name': 'foo', 'start_time': 0, 'end_time': 0,
                         'stripe': 'a', 'view': 'n'}

        test = NCSLSTR1B('somedir/S1_radiance_an.nc', filename_info, 'c')
        assert(test.view == 'n')
        assert(test.stripe == 'a')
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCSLSTRFlag('somedir/S1_radiance_an.nc', filename_info, 'c')
        assert(test.view == 'n')
        assert(test.stripe == 'a')
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        filename_info = {'mission_id': 'S3A', 'dataset_name': 'foo', 'start_time': 0, 'end_time': 0,
                         'stripe': 'c', 'view': 'o'}
        test = NCSLSTR1B('somedir/S1_radiance_co.nc', filename_info, 'c')
        assert(test.view == 'o')
        assert(test.stripe == 'c')
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCSLSTRGeo('somedir/S1_radiance_an.nc', filename_info, 'c')
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCSLSTRAngles('somedir/S1_radiance_an.nc', filename_info, 'c')
        # TODO: Make this test work
        # DatasetID(name='solar_zenith_angle_ao')
        # test.get_dataset(ds_id, {'stripe': 'a', 'view': 'o'})
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

    @mock.patch('xarray.open_dataset')
    def test_angle_datasets(self, mocked_dataset):
        from satpy import Scene
        import xarray as xr
        import numpy as np

        dummy_dataarr = xr.DataArray(np.random.rand(1000, 1000), dims=('rows', 'columns'))
        dummy_dataset = xr.Dataset({'solar_zenith_tn': dummy_dataarr})
        dummy_dataset.attrs = {'start_time': '2018-10-08T09:35:37.0Z',
                               'stop_time': '2018-10-08T09:38:37.0Z',
                               'ac_subsampling_factor': 1}
        mocked_dataset.return_value = dummy_dataset

        scn = Scene(filenames=[os.path.abspath(os.path.join('data', 'S3A_SL_1_RBT____20181008T093537_20181008T093837_'
                                                            '20181008T114117_0179_036_307_2160_MAR_O_NR_003.SEN3',
                                                            'geometry_tn.nc'))],
                    reader='nc_slstr')

        scn.load(['solar_zenith_angle_an'])
        self.assertEqual(list(scn.keys())[0].name, 'solar_zenith_angle_an')


def suite():
    """The test suite for test_nc_slstr."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestSLSTRReader))
    return mysuite


if __name__ == '__main__':
    unittest.main()
