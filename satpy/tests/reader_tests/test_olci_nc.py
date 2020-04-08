#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2018 Satpy developers
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
"""Module for testing the satpy.readers.olci_nc module."""
import unittest
import unittest.mock as mock


class TestOLCIReader(unittest.TestCase):
    """Test various olci_nc filehandlers."""

    @mock.patch('xarray.open_dataset')
    def test_instantiate(self, mocked_dataset):
        """Test initialization of file handlers."""
        from satpy.readers.olci_nc import (NCOLCIBase, NCOLCICal, NCOLCIGeo,
                                           NCOLCIChannelBase, NCOLCI1B, NCOLCI2)
        from satpy import DatasetID
        import xarray as xr

        cal_data = xr.Dataset(
            {
                'solar_flux': (('bands'), [0, 1, 2]),
                'detector_index': (('bands'), [0, 1, 2]),
            },
            {'bands': [0, 1, 2], },
        )

        ds_id = DatasetID(name='Oa01', calibration='reflectance')
        ds_id2 = DatasetID(name='wsqf', calibration='reflectance')
        filename_info = {'mission_id': 'S3A', 'dataset_name': 'Oa01', 'start_time': 0, 'end_time': 0}

        test = NCOLCIBase('somedir/somefile.nc', filename_info, 'c')
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCOLCICal('somedir/somefile.nc', filename_info, 'c')
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCOLCIGeo('somedir/somefile.nc', filename_info, 'c')
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCOLCIChannelBase('somedir/somefile.nc', filename_info, 'c')
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        cal = mock.Mock()
        cal.nc = cal_data
        test = NCOLCI1B('somedir/somefile.nc', filename_info, 'c', cal)
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCOLCI2('somedir/somefile.nc', filename_info, 'c')
        test.get_dataset(ds_id, {'nc_key': 'the_key'})
        test.get_dataset(ds_id2, {'nc_key': 'the_key'})
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

    @mock.patch('xarray.open_dataset')
    def test_get_dataset(self, mocked_dataset):
        """Test reading datasets."""
        from satpy.readers.olci_nc import NCOLCI2
        from satpy import DatasetID
        import numpy as np
        import xarray as xr
        mocked_dataset.return_value = xr.Dataset({'mask': (['rows', 'columns'],
                                                           np.array([1 << x for x in range(30)]).reshape(5, 6))},
                                                 coords={'rows': np.arange(5),
                                                         'columns': np.arange(6)})
        ds_id = DatasetID(name='mask')
        filename_info = {'mission_id': 'S3A', 'dataset_name': 'mask', 'start_time': 0, 'end_time': 0}
        test = NCOLCI2('somedir/somefile.nc', filename_info, 'c')
        res = test.get_dataset(ds_id, {'nc_key': 'mask'})
        self.assertEqual(res.dtype, np.dtype('bool'))

    @mock.patch('xarray.open_dataset')
    def test_olci_angles(self, mocked_dataset):
        """Test reading datasets."""
        from satpy.readers.olci_nc import NCOLCIAngles
        from satpy import DatasetID
        import numpy as np
        import xarray as xr
        attr_dict = {
            'ac_subsampling_factor': 1,
            'al_subsampling_factor': 2,
        }
        mocked_dataset.return_value = xr.Dataset({'SAA': (['tie_rows', 'tie_columns'],
                                                          np.array([1 << x for x in range(30)]).reshape(5, 6)),
                                                  'SZA': (['tie_rows', 'tie_columns'],
                                                          np.array([1 << x for x in range(30)]).reshape(5, 6)),
                                                  'OAA': (['tie_rows', 'tie_columns'],
                                                          np.array([1 << x for x in range(30)]).reshape(5, 6)),
                                                  'OZA': (['tie_rows', 'tie_columns'],
                                                          np.array([1 << x for x in range(30)]).reshape(5, 6))},
                                                 coords={'rows': np.arange(5),
                                                         'columns': np.arange(6)},
                                                 attrs=attr_dict)
        filename_info = {'mission_id': 'S3A', 'dataset_name': 'Oa01', 'start_time': 0, 'end_time': 0}

        ds_id = DatasetID(name='solar_azimuth_angle')
        ds_id2 = DatasetID(name='satellite_zenith_angle')
        test = NCOLCIAngles('somedir/somefile.nc', filename_info, 'c')
        test.get_dataset(ds_id, filename_info)
        test.get_dataset(ds_id2, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

    @mock.patch('xarray.open_dataset')
    def test_olci_meteo(self, mocked_dataset):
        """Test reading datasets."""
        from satpy.readers.olci_nc import NCOLCIMeteo
        from satpy import DatasetID
        import numpy as np
        import xarray as xr
        attr_dict = {
            'ac_subsampling_factor': 1,
            'al_subsampling_factor': 2,
        }
        data = {'humidity': (['tie_rows', 'tie_columns'],
                             np.array([1 << x for x in range(30)]).reshape(5, 6)),
                'total_ozone': (['tie_rows', 'tie_columns'],
                                np.array([1 << x for x in range(30)]).reshape(5, 6)),
                'sea_level_pressure': (['tie_rows', 'tie_columns'],
                                       np.array([1 << x for x in range(30)]).reshape(5, 6)),
                'total_columnar_water_vapour': (['tie_rows', 'tie_columns'],
                                                np.array([1 << x for x in range(30)]).reshape(5, 6))}
        mocked_dataset.return_value = xr.Dataset(data,
                                                 coords={'rows': np.arange(5),
                                                         'columns': np.arange(6)},
                                                 attrs=attr_dict)
        filename_info = {'mission_id': 'S3A', 'dataset_name': 'humidity', 'start_time': 0, 'end_time': 0}

        ds_id = DatasetID(name='humidity')
        ds_id2 = DatasetID(name='total_ozone')
        test = NCOLCIMeteo('somedir/somefile.nc', filename_info, 'c')
        test.get_dataset(ds_id, filename_info)
        test.get_dataset(ds_id2, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()


class TestBitFlags(unittest.TestCase):
    """Test the bitflag reading."""

    def test_bitflags(self):
        """Test the BitFlags class."""
        import numpy as np
        from functools import reduce
        from satpy.readers.olci_nc import BitFlags
        flag_list = ['INVALID', 'WATER', 'LAND', 'CLOUD', 'SNOW_ICE',
                     'INLAND_WATER', 'TIDAL', 'COSMETIC', 'SUSPECT', 'HISOLZEN',
                     'SATURATED', 'MEGLINT', 'HIGHGLINT', 'WHITECAPS',
                     'ADJAC', 'WV_FAIL', 'PAR_FAIL', 'AC_FAIL', 'OC4ME_FAIL',
                     'OCNN_FAIL', 'Extra_1', 'KDM_FAIL', 'Extra_2',
                     'CLOUD_AMBIGUOUS', 'CLOUD_MARGIN', 'BPAC_ON',
                     'WHITE_SCATT', 'LOWRW', 'HIGHRW']

        bits = np.array([1 << x for x in range(len(flag_list))])

        bflags = BitFlags(bits)

        items = ["INVALID", "SNOW_ICE", "INLAND_WATER", "SUSPECT",
                 "AC_FAIL", "CLOUD", "HISOLZEN", "OCNN_FAIL",
                 "CLOUD_MARGIN", "CLOUD_AMBIGUOUS", "LOWRW", "LAND"]

        mask = reduce(np.logical_or, [bflags[item] for item in items])
        expected = np.array([True, False,  True,  True,  True,  True, False,
                             False,  True, True, False, False, False, False,
                             False, False, False,  True, False,  True, False,
                             False, False,  True,  True, False, False, True,
                             False])
        self.assertTrue(all(mask == expected))
