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

import numpy as np
from satpy.readers.olci_nc import BitFlags

flag_list = ['INVALID', 'WATER', 'LAND', 'CLOUD', 'SNOW_ICE',
             'INLAND_WATER', 'TIDAL', 'COSMETIC', 'SUSPECT', 'HISOLZEN',
             'SATURATED', 'MEGLINT', 'HIGHGLINT', 'WHITECAPS',
             'ADJAC', 'WV_FAIL', 'PAR_FAIL', 'AC_FAIL', 'OC4ME_FAIL',
             'OCNN_FAIL', 'Extra_1', 'KDM_FAIL', 'Extra_2',
             'CLOUD_AMBIGUOUS', 'CLOUD_MARGIN', 'BPAC_ON',
             'WHITE_SCATT', 'LOWRW', 'HIGHRW']


@mock.patch('xarray.open_dataset')
class TestOLCIReader(unittest.TestCase):
    """Test various olci_nc filehandlers."""

    def test_instantiate(self, mocked_dataset):
        """Test initialization of file handlers."""
        from satpy.readers.olci_nc import (NCOLCIBase, NCOLCICal, NCOLCIGeo,
                                           NCOLCIChannelBase, NCOLCI1B, NCOLCI2)
        from satpy.tests.utils import make_dataid
        import xarray as xr

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

    def test_get_dataset(self, mocked_dataset):
        """Test reading datasets."""
        from satpy.tests.utils import make_dataid
        fh, _ = self._create_wqsf_filehandler(mocked_dataset)
        ds_id = make_dataid(name='wqsf')
        res = fh.get_dataset(ds_id, {'nc_key': 'WQSF'})
        self.assertEqual(res.dtype, np.uint64)

    def _create_wqsf_filehandler(self, mocked_dataset, meanings="INVALID WATER LAND CLOUD"):
        """Create a filehandle for the wqsf quality flags."""
        from satpy.readers.olci_nc import NCOLCI2Flags
        import xarray as xr
        nb_flags = len(meanings.split())
        wqsf_data = xr.DataArray((2 ** (np.arange(30) % nb_flags)).astype(np.uint64).reshape(5, 6),
                                 dims=["rows", "columns"],
                                 coords={'rows': np.arange(5),
                                         'columns': np.arange(6)},
                                 attrs={"flag_masks": 2 ** np.arange(nb_flags),
                                        "flag_meanings": meanings})
        mocked_dataset.return_value = xr.Dataset({'WQSF': wqsf_data})
        filename_info = {'mission_id': 'S3A', 'dataset_name': None, 'start_time': 0, 'end_time': 0}
        fh = NCOLCI2Flags('somedir/somefile.nc', filename_info, 'c')
        return fh, wqsf_data

    def test_meanings_are_read_from_file(self, mocked_dataset):
        """Test that the flag meanings are read from the file."""
        fh, wqsf_data = self._create_wqsf_filehandler(mocked_dataset)
        fh.create_bitflags(wqsf_data)
        res = fh.getbitmask(wqsf_data, ["CLOUD"])
        np.testing.assert_allclose(res, (np.arange(30) % 4).reshape(5, 6) == 3)

        fh, wqsf_data = self._create_wqsf_filehandler(mocked_dataset, "NOTHING FISH SHRIMP TURTLES")
        fh.create_bitflags(wqsf_data)
        res = fh.getbitmask(wqsf_data, ["TURTLES"])
        np.testing.assert_allclose(res, (np.arange(30) % 4).reshape(5, 6) == 3)

    def test_get_mask(self, mocked_dataset):
        """Test reading mask datasets."""
        from satpy.tests.utils import make_dataid
        fh, wqsf_data = self._create_wqsf_filehandler(mocked_dataset, " ".join(flag_list))

        masked_items = ["INVALID", "SNOW_ICE", "INLAND_WATER", "SUSPECT",
                        "AC_FAIL", "CLOUD", "HISOLZEN", "OCNN_FAIL",
                        "CLOUD_MARGIN", "CLOUD_AMBIGUOUS", "LOWRW", "LAND"]

        ds_id = make_dataid(name='mask')
        res = fh.get_dataset(ds_id, {'nc_key': 'WQSF', "masked_items": masked_items})
        self.assertEqual(res.dtype, np.dtype("bool"))

        expected = np.array([True, False,  True,  True,  True,  True, False,
                             False,  True, True, False, False, False, False,
                             False, False, False,  True, False,  True, False,
                             False, False,  True,  True, False, False, True,
                             False, True]).reshape(5, 6)
        np.testing.assert_array_equal(res, expected)

    def test_wqsf_has_bitflags_attribute(self, mocked_dataset):
        """Test wqsf has a bitflags attribute."""
        from satpy.tests.utils import make_dataid
        fh, wqsf_data = self._create_wqsf_filehandler(mocked_dataset, " ".join(flag_list))

        ds_id = make_dataid(name='wqsf')
        res = fh.get_dataset(ds_id, {'nc_key': 'WQSF'})
        assert isinstance(res.attrs["bitflags"], BitFlags)

    def test_get_cloud_mask(self, mocked_dataset):
        """Test reading the cloud_mask dataset."""
        from satpy.tests.utils import make_dataid
        fh, wqsf_data = self._create_wqsf_filehandler(mocked_dataset, " ".join(flag_list))

        ds_id = make_dataid(name='cloud_mask')
        res = fh.get_dataset(ds_id, {'nc_key': 'WQSF', "masked_items": ["CLOUD"]})
        self.assertEqual(res.dtype, np.dtype("bool"))

        expected = np.array([False, False,  False,  True,  False, False, False,
                             False,  False, False, False, False, False, False,
                             False, False, False,  False, False,  False, False,
                             False, False,  False,  False, False, False, False,
                             False, False]).reshape(5, 6)
        np.testing.assert_array_equal(res, expected)

    def test_get_ocnn_mask(self, mocked_dataset):
        """Test reading the ocnn_mask dataset."""
        from satpy.tests.utils import make_dataid
        fh, wqsf_data = self._create_wqsf_filehandler(mocked_dataset, " ".join(flag_list))

        ds_id = make_dataid(name='ocnn_mask')
        res = fh.get_dataset(ds_id, {'nc_key': 'WQSF', "masked_items": ["OCNN_FAIL"]})
        self.assertEqual(res.dtype, np.dtype("bool"))

        expected = np.array([False, False,  False,  False,  False, False, False,
                             False,  False, False, False, False, False, False,
                             False, False, False,  False, False,  True, False,
                             False, False,  False,  False, False, False, False,
                             False, False]).reshape(5, 6)
        np.testing.assert_array_equal(res, expected)


class TestOLCIAngles(unittest.TestCase):
    """Test the angles olci_nc filehandler."""

    def setUp(self):
        """Set up the test case."""
        from satpy.readers.olci_nc import NCOLCIAngles
        import xarray as xr
        attr_dict = {
            'ac_subsampling_factor': 1,
            'al_subsampling_factor': 2,
        }

        self.patcher = mock.patch("xarray.open_dataset")
        mocked_dataset = self.patcher.start()

        mocked_dataset.return_value = xr.Dataset({'SAA': (['tie_rows', 'tie_columns'],
                                                          np.arange(30).reshape(5, 6)),
                                                  'SZA': (['tie_rows', 'tie_columns'],
                                                          np.arange(30).reshape(5, 6) + 30),
                                                  'OAA': (['tie_rows', 'tie_columns'],
                                                          np.arange(30).reshape(5, 6)),
                                                  'OZA': (['tie_rows', 'tie_columns'],
                                                          np.arange(30).reshape(5, 6) + 30)},
                                                 coords={'tie_rows': np.arange(5),
                                                         'tie_columns': np.arange(6)},
                                                 attrs=attr_dict)
        self.filename_info = {'mission_id': 'S3A', 'dataset_name': 'Oa01', 'start_time': 0, 'end_time': 0}
        self.file_handler = NCOLCIAngles('somedir/somefile.nc', self.filename_info, 'c')

        self.expected_data = np.array([[0, 1, 2, 3, 4, 5],
                                       [3, 4, 5, 6, 7, 8],
                                       [6, 7, 8, 9, 10, 11],
                                       [9, 10, 11, 12, 13, 14],
                                       [12, 13, 14, 15, 16, 17],
                                       [15, 16, 17, 18, 19, 20],
                                       [18, 19, 20, 21, 22, 23],
                                       [21, 22, 23, 24, 25, 26],
                                       [24, 25, 26, 27, 28, 29]]
                                      )

    def test_olci_angles(self):
        """Test reading angles datasets."""
        from satpy.tests.utils import make_dataid
        ds_id_sun_azimuth = make_dataid(name='solar_azimuth_angle')
        ds_id_sat_zenith = make_dataid(name='satellite_zenith_angle')

        azi = self.file_handler.get_dataset(ds_id_sun_azimuth, self.filename_info)
        zen = self.file_handler.get_dataset(ds_id_sat_zenith, self.filename_info)
        np.testing.assert_allclose(azi, self.expected_data, atol=0.5)
        np.testing.assert_allclose(zen, self.expected_data + 30, atol=0.5)

    def test_olci_angles_caches_interpolation(self):
        """Test reading angles datasets caches interpolation."""
        from satpy.tests.utils import make_dataid

        ds_id = make_dataid(name='solar_zenith_angle')
        self._check_interpolator_is_called_only_once(ds_id, ds_id)

    def test_olci_different_angles_caches_interpolation(self):
        """Test reading different angles datasets caches interpolation."""
        from satpy.tests.utils import make_dataid

        ds_id_zenith = make_dataid(name='solar_zenith_angle')
        ds_id_azimuth = make_dataid(name='solar_azimuth_angle')
        self._check_interpolator_is_called_only_once(ds_id_azimuth, ds_id_zenith)

    def _check_interpolator_is_called_only_once(self, ds_id_1, ds_id_2):
        """Check that the interpolation is used only once."""
        with mock.patch("geotiepoints.interpolator.Interpolator") as interpolator:
            interpolator.return_value.interpolate.return_value = (
                self.expected_data, self.expected_data, self.expected_data)

            self.file_handler.get_dataset(ds_id_2, self.filename_info)
            self.file_handler.get_dataset(ds_id_1, self.filename_info)
            assert (interpolator.call_count == 1)

    def tearDown(self):
        """Tear down the test case."""
        self.patcher.stop()


class TestOLCIMeteo(unittest.TestCase):
    """Test the meteo olci_nc filehandler."""

    def setUp(self):
        """Set up the test case."""
        from satpy.readers.olci_nc import NCOLCIMeteo
        import xarray as xr
        attr_dict = {
            'ac_subsampling_factor': 1,
            'al_subsampling_factor': 2,
        }
        data = {'humidity': (['tie_rows', 'tie_columns'],
                             np.arange(30).reshape(5, 6)),
                'total_ozone': (['tie_rows', 'tie_columns'],
                                np.arange(30).reshape(5, 6)),
                'sea_level_pressure': (['tie_rows', 'tie_columns'],
                                       np.arange(30).reshape(5, 6)),
                'total_columnar_water_vapour': (['tie_rows', 'tie_columns'],
                                                np.arange(30).reshape(5, 6))}

        self.patcher = mock.patch("xarray.open_dataset")
        mocked_dataset = self.patcher.start()

        mocked_dataset.return_value = xr.Dataset(data,
                                                 coords={'rows': np.arange(5),
                                                         'columns': np.arange(6)},
                                                 attrs=attr_dict)
        self.filename_info = {'mission_id': 'S3A', 'dataset_name': 'humidity', 'start_time': 0, 'end_time': 0}
        self.file_handler = NCOLCIMeteo('somedir/somefile.nc', self.filename_info, 'c')

        self.expected_data = np.array([[0, 1, 2, 3, 4, 5],
                                       [3, 4, 5, 6, 7, 8],
                                       [6, 7, 8, 9, 10, 11],
                                       [9, 10, 11, 12, 13, 14],
                                       [12, 13, 14, 15, 16, 17],
                                       [15, 16, 17, 18, 19, 20],
                                       [18, 19, 20, 21, 22, 23],
                                       [21, 22, 23, 24, 25, 26],
                                       [24, 25, 26, 27, 28, 29]]
                                      )

    def tearDown(self):
        """Tear down the test case."""
        self.patcher.stop()

    def test_olci_meteo_reading(self):
        """Test reading meteo datasets."""
        from satpy.tests.utils import make_dataid

        ds_id_humidity = make_dataid(name='humidity')
        ds_id_total_ozone = make_dataid(name='total_ozone')

        humidity = self.file_handler.get_dataset(ds_id_humidity, self.filename_info)
        total_ozone = self.file_handler.get_dataset(ds_id_total_ozone, self.filename_info)

        np.testing.assert_allclose(humidity, self.expected_data, atol=1e-10)
        np.testing.assert_allclose(total_ozone, self.expected_data, atol=1e-10)

    def test_olci_meteo_caches_interpolation(self):
        """Test reading meteo datasets caches interpolation."""
        from satpy.tests.utils import make_dataid

        ds_id = make_dataid(name='humidity')
        with mock.patch("geotiepoints.interpolator.Interpolator") as interpolator:
            interpolator.return_value.interpolate.return_value = (self.expected_data, )

            self.file_handler.get_dataset(ds_id, self.filename_info)
            self.file_handler.get_dataset(ds_id, self.filename_info)
            assert(interpolator.call_count == 1)


class TestBitFlags(unittest.TestCase):
    """Test the bitflag reading."""

    def test_bitflags(self):
        """Test the BitFlags class."""
        nb_flags = len(flag_list)

        # As a test, the data is just an array with the possible masks
        data = 2 ** np.arange(nb_flags)
        masks = 2 ** np.arange(nb_flags)

        bflags = BitFlags(masks, flag_list)

        items = ["INVALID", "SNOW_ICE", "INLAND_WATER", "SUSPECT",
                 "AC_FAIL", "CLOUD", "HISOLZEN", "OCNN_FAIL",
                 "CLOUD_MARGIN", "CLOUD_AMBIGUOUS", "LOWRW", "LAND"]

        mask = bflags.match_any(items, data)
        expected = np.array([True, False,  True,  True,  True,  True, False,
                             False,  True, True, False, False, False, False,
                             False, False, False,  True, False,  True, False,
                             False, False,  True,  True, False, False, True,
                             False])
        np.testing.assert_array_equal(mask, expected)

    def test_match_item(self):
        """Test matching one item."""
        nb_flags = len(flag_list)

        # As a test, the data is just an array with the possible masks
        data = 2 ** np.arange(nb_flags)
        masks = 2 ** np.arange(nb_flags)

        bflags = BitFlags(masks, flag_list)
        mask = bflags.match_item("INVALID", data)
        expected = np.array([True, False, False, False, False, False, False,
                             False, False, False, False, False, False, False,
                             False, False, False, False, False, False, False,
                             False, False, False, False, False, False, False,
                             False])
        np.testing.assert_array_equal(mask, expected)

    def test_equality(self):
        """Test equality."""
        nb_flags = len(flag_list)

        # As a test, the data is just an array with the possible masks
        masks = 2 ** np.arange(nb_flags)

        one = BitFlags(masks, flag_list)

        two = BitFlags(masks, flag_list)

        assert one == two
