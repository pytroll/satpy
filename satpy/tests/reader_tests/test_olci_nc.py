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
import datetime
import unittest
import unittest.mock as mock


class TestOLCIReader(unittest.TestCase):
    """Test various olci_nc filehandlers."""

    @mock.patch('xarray.open_dataset')
    def test_instantiate(self, mocked_dataset):
        """Test initialization of file handlers."""
        import xarray as xr

        from satpy.readers.olci_nc import NCOLCI1B, NCOLCI2, NCOLCIBase, NCOLCICal, NCOLCIChannelBase, NCOLCIGeo
        from satpy.tests.utils import make_dataid

        cal_data = xr.Dataset(
            {
                'solar_flux': (('bands'), [0, 1, 2]),
                'detector_index': (('bands'), [0, 1, 2]),
            },
            {'bands': [0, 1, 2], },
        )

        ds_id = make_dataid(name='Oa01', calibration='reflectance')
        ds_id2 = make_dataid(name='wsqf', calibration='reflectance')
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
    def test_open_file_objects(self, mocked_open_dataset):
        """Test initialization of file handlers."""
        from satpy.readers.olci_nc import NCOLCIBase
        filename_info = {'mission_id': 'S3A', 'dataset_name': 'Oa01', 'start_time': 0, 'end_time': 0}

        open_file = mock.MagicMock()

        file_handler = NCOLCIBase(open_file, filename_info, 'c')
        #  deepcode ignore W0104: This is a property that is actually a function call.
        file_handler.nc  # pylint: disable=W0104
        mocked_open_dataset.assert_called()
        open_file.open.assert_called()
        assert (open_file.open.return_value in mocked_open_dataset.call_args[0] or
                open_file.open.return_value == mocked_open_dataset.call_args[1].get('filename_or_obj'))

    @mock.patch('xarray.open_dataset')
    def test_get_mask(self, mocked_dataset):
        """Test reading datasets."""
        import numpy as np
        import xarray as xr

        from satpy.readers.olci_nc import NCOLCI2
        from satpy.tests.utils import make_dataid
        mocked_dataset.return_value = xr.Dataset({'mask': (['rows', 'columns'],
                                                           np.array([1 << x for x in range(30)]).reshape(5, 6))},
                                                 coords={'rows': np.arange(5),
                                                         'columns': np.arange(6)})
        ds_id = make_dataid(name='mask')
        filename_info = {'mission_id': 'S3A', 'dataset_name': 'mask', 'start_time': 0, 'end_time': 0}
        test = NCOLCI2('somedir/somefile.nc', filename_info, 'c')
        res = test.get_dataset(ds_id, {'nc_key': 'mask'})
        self.assertEqual(res.dtype, np.dtype('bool'))
        expected = np.array([[True, False, True, True, True, True],
                             [False, False, True, True, False, False],
                             [False, False, False, False, False, True],
                             [False, True, False, False, False, True],
                             [True, False, False, True, False, False]])
        np.testing.assert_array_equal(res.values, expected)

    @mock.patch('xarray.open_dataset')
    def test_get_mask_with_alternative_items(self, mocked_dataset):
        """Test reading datasets."""
        import numpy as np
        import xarray as xr

        from satpy.readers.olci_nc import NCOLCI2
        from satpy.tests.utils import make_dataid
        mocked_dataset.return_value = xr.Dataset({'mask': (['rows', 'columns'],
                                                           np.array([1 << x for x in range(30)]).reshape(5, 6))},
                                                 coords={'rows': np.arange(5),
                                                         'columns': np.arange(6)})
        ds_id = make_dataid(name='mask')
        filename_info = {'mission_id': 'S3A', 'dataset_name': 'mask', 'start_time': 0, 'end_time': 0}
        test = NCOLCI2('somedir/somefile.nc', filename_info, 'c', mask_items=["INVALID"])
        res = test.get_dataset(ds_id, {'nc_key': 'mask'})
        self.assertEqual(res.dtype, np.dtype('bool'))
        expected = np.array([True] + [False] * 29).reshape(5, 6)
        np.testing.assert_array_equal(res.values, expected)

    @mock.patch('xarray.open_dataset')
    def test_olci_angles(self, mocked_dataset):
        """Test reading datasets."""
        import numpy as np
        import xarray as xr

        from satpy.readers.olci_nc import NCOLCIAngles
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
        filename_info = {'mission_id': 'S3A', 'dataset_name': 'Oa01', 'start_time': 0, 'end_time': 0}

        ds_id = make_dataid(name='solar_azimuth_angle')
        ds_id2 = make_dataid(name='satellite_zenith_angle')
        test = NCOLCIAngles('somedir/somefile.nc', filename_info, 'c')
        test.get_dataset(ds_id, filename_info)
        test.get_dataset(ds_id2, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

    @mock.patch('xarray.open_dataset')
    def test_olci_meteo(self, mocked_dataset):
        """Test reading datasets."""
        import numpy as np
        import xarray as xr

        from satpy.readers.olci_nc import NCOLCIMeteo
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
        filename_info = {'mission_id': 'S3A', 'dataset_name': 'humidity', 'start_time': 0, 'end_time': 0}

        ds_id = make_dataid(name='humidity')
        ds_id2 = make_dataid(name='total_ozone')
        test = NCOLCIMeteo('somedir/somefile.nc', filename_info, 'c')
        test.get_dataset(ds_id, filename_info)
        test.get_dataset(ds_id2, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

    @mock.patch("xarray.open_dataset")
    def test_chl_nn(self, mocked_dataset):
        """Test unlogging the chl_nn product."""
        import numpy as np
        import xarray as xr

        from satpy.readers.olci_nc import NCOLCI2
        from satpy.tests.utils import make_dataid
        attr_dict = {
            'ac_subsampling_factor': 64,
            'al_subsampling_factor': 1,
        }
        data = {'CHL_NN': (['rows', 'columns'],
                           np.arange(30).reshape(5, 6).astype(float),
                           {"units": "lg(re mg.m-3)"})}
        mocked_dataset.return_value = xr.Dataset(data,
                                                 coords={'rows': np.arange(5),
                                                         'columns': np.arange(6)},
                                                 attrs=attr_dict)
        ds_info = {'name': 'chl_nn', 'sensor': 'olci', 'resolution': 300,
                   'standard_name': 'algal_pigment_concentration', 'units': 'lg(re mg.m-3)',
                   'coordinates': ('longitude', 'latitude'), 'file_type': 'esa_l2_chl_nn', 'nc_key': 'CHL_NN',
                   'modifiers': ()}
        filename_info = {'mission_id': 'S3A', 'datatype_id': 'WFR',
                         'start_time': datetime.datetime(2019, 9, 24, 9, 29, 39),
                         'end_time': datetime.datetime(2019, 9, 24, 9, 32, 39),
                         'creation_time': datetime.datetime(2019, 9, 24, 11, 40, 26), 'duration': 179, 'cycle': 49,
                         'relative_orbit': 307, 'frame': 1800, 'centre': 'MAR', 'mode': 'O', 'timeliness': 'NR',
                         'collection': '002'}
        ds_id = make_dataid(name='chl_nn')
        file_handler = NCOLCI2('somedir/somefile.nc', filename_info, None, unlog=True)
        res = file_handler.get_dataset(ds_id, ds_info)

        assert res.attrs["units"] == "mg.m-3"
        assert res.values[-1, -1] == 1e29


class TestBitFlags(unittest.TestCase):
    """Test the bitflag reading."""

    def test_bitflags(self):
        """Test the BitFlags class."""
        from functools import reduce

        import numpy as np

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
