# Copyright (c) 2016-2021 Satpy developers
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
"""Module for testing the satpy.readers.meris_nc_sen3 module."""
import unittest
import unittest.mock as mock


class TestMERISReader(unittest.TestCase):
    """Test various meris_nc_sen3 filehandlers."""

    @mock.patch('xarray.open_dataset')
    def test_instantiate(self, mocked_dataset):
        """Test initialization of file handlers."""
        from satpy.readers.meris_nc_sen3 import NCMERIS2, NCMERISCal, NCMERISGeo
        from satpy.tests.utils import make_dataid

        ds_id = make_dataid(name='M01', calibration='reflectance')
        ds_id2 = make_dataid(name='wsqf', calibration='reflectance')
        filename_info = {'mission_id': 'ENV', 'dataset_name': 'M01', 'start_time': 0, 'end_time': 0}

        test = NCMERISCal('somedir/somefile.nc', filename_info, 'c')
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCMERISGeo('somedir/somefile.nc', filename_info, 'c')
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCMERIS2('somedir/somefile.nc', filename_info, 'c')
        test.get_dataset(ds_id, {'nc_key': 'the_key'})
        test.get_dataset(ds_id2, {'nc_key': 'the_key'})
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

    @mock.patch('xarray.open_dataset')
    def test_open_file_objects(self, mocked_open_dataset):
        """Test initialization of file handlers."""
        from satpy.readers.olci_nc import NCOLCIBase
        filename_info = {'mission_id': 'ENV', 'dataset_name': 'M01', 'start_time': 0, 'end_time': 0}

        open_file = mock.MagicMock()

        file_handler = NCOLCIBase(open_file, filename_info, 'c')
        #  deepcode ignore W0104: This is a property that is actually a function call.
        file_handler.nc  # pylint: disable=W0104
        mocked_open_dataset.assert_called()
        open_file.open.assert_called()
        assert (open_file.open.return_value in mocked_open_dataset.call_args[0] or
                open_file.open.return_value == mocked_open_dataset.call_args[1].get('filename_or_obj'))

    @mock.patch('xarray.open_dataset')
    def test_get_dataset(self, mocked_dataset):
        """Test reading datasets."""
        import numpy as np
        import xarray as xr

        from satpy.readers.meris_nc_sen3 import NCMERIS2
        from satpy.tests.utils import make_dataid
        mocked_dataset.return_value = xr.Dataset({'mask': (['rows', 'columns'],
                                                           np.array([1 << x for x in range(30)]).reshape(5, 6))},
                                                 coords={'rows': np.arange(5),
                                                         'columns': np.arange(6)})
        ds_id = make_dataid(name='mask')
        filename_info = {'mission_id': 'ENV', 'dataset_name': 'mask', 'start_time': 0, 'end_time': 0}
        test = NCMERIS2('somedir/somefile.nc', filename_info, 'c')
        res = test.get_dataset(ds_id, {'nc_key': 'mask'})
        self.assertEqual(res.dtype, np.dtype('bool'))

    @mock.patch('xarray.open_dataset')
    def test_meris_angles(self, mocked_dataset):
        """Test reading datasets."""
        import numpy as np
        import xarray as xr

        from satpy.readers.meris_nc_sen3 import NCMERISAngles
        from satpy.tests.utils import make_dataid
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
        filename_info = {'mission_id': 'ENV', 'dataset_name': 'M01', 'start_time': 0, 'end_time': 0}

        ds_id = make_dataid(name='solar_azimuth_angle')
        ds_id2 = make_dataid(name='satellite_zenith_angle')
        test = NCMERISAngles('somedir/somefile.nc', filename_info, 'c')
        test.get_dataset(ds_id, filename_info)
        test.get_dataset(ds_id2, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

    @mock.patch('xarray.open_dataset')
    def test_meris_meteo(self, mocked_dataset):
        """Test reading datasets."""
        import numpy as np
        import xarray as xr

        from satpy.readers.meris_nc_sen3 import NCMERISMeteo
        from satpy.tests.utils import make_dataid
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
        filename_info = {'mission_id': 'ENV', 'dataset_name': 'humidity', 'start_time': 0, 'end_time': 0}

        ds_id = make_dataid(name='humidity')
        ds_id2 = make_dataid(name='total_ozone')
        test = NCMERISMeteo('somedir/somefile.nc', filename_info, 'c')
        test.get_dataset(ds_id, filename_info)
        test.get_dataset(ds_id2, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()


class TestBitFlags(unittest.TestCase):
    """Test the bitflag reading."""

    def test_bitflags(self):
        """Test the BitFlags class."""
        from functools import reduce

        import numpy as np

        from satpy.readers.olci_nc import BitFlags

        flag_list = ['SEA_ICE', 'MEGLINT', 'HIGHGLINT', 'CASE2_S', 'CASE2_ANOM',
                     'HAZE_OVER_WATER', 'WHITECAPS', 'AC_FAIL', 'BPAC_ON', 'WHITE_SCATT',
                     'LOWRW', 'HIGHRW', 'OUT_OF_RANGE_AAC', 'OUT_OF_SCOPE_AAC',
                     'OUT_OF_RANGE_OC_NN', 'OUT_OF_SCOPE_OC_NN',
                     'OUT_OF_RANGE_CHL_OC4ME_INPUT', 'OUT_OF_RANGE_CHL_OC4ME']

        bits = np.array([1 << x for x in range(len(flag_list))])

        bflags = BitFlags(bits, flag_list=flag_list)

        items = ['SEA_ICE', 'MEGLINT', 'HIGHGLINT',
                 'HAZE_OVER_WATER', 'WHITECAPS', 'AC_FAIL', 'WHITE_SCATT',
                 'LOWRW', 'HIGHRW', 'OUT_OF_RANGE_AAC', 'OUT_OF_SCOPE_AAC',
                 'OUT_OF_RANGE_OC_NN', 'OUT_OF_SCOPE_OC_NN',
                 'OUT_OF_RANGE_CHL_OC4ME_INPUT', 'OUT_OF_RANGE_CHL_OC4ME']

        mask = reduce(np.logical_or, [bflags[item] for item in items])
        expected = np.array([True, True, True, False, False, True, True,
                             True, False, True, True, True, True, True, True,
                             True, True, True])
        self.assertTrue(all(mask == expected))
