#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2019 Satpy developers
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
"""Unittesting the SEVIRI L2 BUFR reader."""

import os
import sys
import unittest
from datetime import datetime
from unittest import mock

import dask.array as da
import numpy as np
import pytest
from pyresample import geometry

from satpy.tests.utils import make_dataid

AREA_DEF_MSG_IODC = geometry.AreaDefinition(
    'msg_seviri_iodc_48km',
    'MSG SEVIRI Indian Ocean Data Coverage service area definition with 48 km resolution',
    "",
    {'a': 6378169., 'b': 6356583.8, 'lon_0': 45.5,
     'h': 35785831., 'proj': 'geos', 'units': 'm'},
    232,
    232,
    (-5570248.686685662, -5567248.28340708, 5567248.28340708, 5570248.686685662)
)

AREA_DEF_MSG_FES = geometry.AreaDefinition(
    'msg_seviri_fes_48km',
    'MSG SEVIRI Full Earth Scanning service area definition with 48 km resolution',
    "",
    {'a': 6378169., 'b': 6356583.8, 'lon_0': 0.0,
     'h': 35785831., 'proj': 'geos', 'units': 'm'},
    232,
    232,
    (-5570248.6867, -5567248.2834, 5567248.2834, 5570248.6867)
)

AREA_DEF_MSG_IODC_EXT = geometry.AreaDefinition(
    'msg_seviri_iodc_9km_ext',
    'MSG SEVIRI Indian Ocean Data Coverage service area definition with 9 km resolution '
    '(extended outside original 3km grid)',
    "",
    {'a': 6378169., 'b': 6356583.8, 'lon_0': 45.5,
     'h': 35785831., 'proj': 'geos', 'units': 'm'},
    1238,
    1238,
    (-5571748.8883, -5571748.8882, 5571748.8882, 5571748.8883)
)

AREA_DEF_FCI_FES = geometry.AreaDefinition(
    'mtg_fci_fdss_32km',
    'MTG FCI Full Disk Scanning Service area definition with 32 km SSP resolution',
    "",
    {'x_0': 0, 'y_0': 0, 'ellps': 'WGS84', 'lon_0': 0.0,
     'h': 35786400., 'proj': 'geos', 'units': 'm'},
    348,
    348,
    (-5567999.998550739, -5567999.998550739, 5567999.994203017,  5567999.994203017)
)


AREA_DEF_MSG_FES_3km_ext = geometry.AreaDefinition(
    'msg_seviri_fes_9km_ext',
    'MSG SEVIRI Full Earth Scanning service area definition with 9 km resolution',
    "",
    {'a': 6378169., 'b': 6356583.8, 'lon_0': 0.0,
     'h': 35785831., 'proj': 'geos', 'units': 'm'},
    1238,
    1238,
    (-5571748.888268564, -5571748.888155806, 5571748.888155806, 5571748.888268564)
)

# Test data for mock file
DATA = np.random.uniform(low=250, high=350, size=(128,))
LAT = np.random.uniform(low=-80, high=80, size=(128,))
LON = np.random.uniform(low=-38.5, high=121.5, size=(128,))

os.environ['ECCODES_BUFR_MULTI_ELEMENT_CONSTANT_ARRAYS'] = "1"

# Test cases dictionaries
TEST_DATA = {'GIIBUFRProduct_20231027140000Z_00_OMPEFS03_MET10_FES_E0000': {
                'platform_name': 'MSG3',
                'spacecraft_number': '10',
                'RectificationLongitude': 'E0000',
                'ssp_lon': 0.0,
                'area': AREA_DEF_MSG_FES_3km_ext,
                'seg_size': 3,
                'file_type': 'seviri_l2_bufr_gii',
                'key': '#1#brightnessTemperature',
                'resolution': 9000},
             'ASRBUFRProd_20231022224500Z_00_OMPEFS03_MET10_FES_E0000': {
                'platform_name': 'MSG3',
                'spacecraft_number': '10',
                'RectificationLongitude': 'E0000',
                'ssp_lon': 0.0,
                'area': AREA_DEF_MSG_FES,
                'seg_size': 16,
                'file_type': 'seviri_l2_bufr_asr',
                'key': '#1#brightnessTemperature',
                'resolution': 48000},
             'ASRBUFRProd_20231023044500Z_00_OMPEFS02_MET09_FES_E0455': {
                'platform_name': 'MSG2',
                'spacecraft_number': '9',
                'RectificationLongitude': 'E0455',
                'area': AREA_DEF_MSG_IODC,
                'ssp_lon': 45.5,
                'seg_size': 16,
                'file_type': 'seviri_l2_bufr_asr',
                'key': '#1#brightnessTemperature',
                'resolution': 48000},
             'MSG2-SEVI-MSGASRE-0101-0101-20191106130000.000000000Z-20191106131702-1362128.bfr': {
                'platform_name': 'MSG2',
                'spacecraft_number': '9',
                'RectificationLongitude': 'E0455',
                'area': AREA_DEF_MSG_IODC,
                'ssp_lon': 45.5,
                'seg_size': 16,
                'file_type': 'seviri_l2_bufr_asr',
                'key': '#1#brightnessTemperature',
                'resolution': 48000},
             'W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-2-ASR--FD------BUFR_C_EUMT_20230623092246_L2PF_IV_20170410170000_20170410171000_V__C_0103_0000.bin': {
                'platform_name': 'MTGi1',
                'spacecraft_number': '24',
                'RectificationLongitude': 'E0000',
                'area': AREA_DEF_FCI_FES,
                'ssp_lon': 0.0,
                'seg_size': 32,
                'file_type': 'fci_l2_bufr_asr',
                'key': '#1#brightnessTemperature',
                'resolution': 32000}
          }

TEST_FILES = []
for name, _dict_ in TEST_DATA.items():
    TEST_FILES.append(name)


class L2BufrData:
    """Mock SEVIRI L2 BUFR data."""

    @unittest.skipIf(sys.platform.startswith('win'), "'eccodes' not supported on Windows")
    def __init__(self, filename, with_adef=False, rect_lon='default'):
        """Initialize by mocking test data for testing the SEVIRI L2 BUFR reader."""
        import eccodes as ec

        from satpy.readers.eum_l2_bufr import EumetsatL2BufrFileHandler
        self.buf1 = ec.codes_bufr_new_from_samples('BUFR4_local_satellite')
        ec.codes_set(self.buf1, 'unpack', 1)
        # write the bufr test data twice as we want to read in and then concatenate the data in the reader
        # 55 id corresponds to METEOSAT 8`
        ec.codes_set(self.buf1, 'satelliteIdentifier', 47 + int(TEST_DATA[filename]['spacecraft_number']))
        ec.codes_set_array(self.buf1, '#1#latitude', LAT)
        ec.codes_set_array(self.buf1, '#1#latitude', LAT)
        ec.codes_set_array(self.buf1, '#1#longitude', LON)
        ec.codes_set_array(self.buf1, '#1#longitude', LON)
        ec.codes_set_array(self.buf1, '#1#brightnessTemperature', DATA)

        self.m = mock.mock_open()

        FILETYPE_INFO = {'file_type':  TEST_DATA[filename]['file_type']}

        # only our offline product contain MPEF product headers so we get the metadata from there
        if ('BUFRProd' in filename):
            with mock.patch('satpy.readers.eum_l2_bufr.np.fromfile') as fromfile:

                MPEF_PRODUCT_HEADER = {
                    'NominalTime': datetime(2019, 11, 6, 18, 0),
                    'SpacecraftName': TEST_DATA[filename]['spacecraft_number'],
                    'RectificationLongitude': TEST_DATA[filename]['RectificationLongitude']
                    }
                fromfile.return_value = MPEF_PRODUCT_HEADER
                with mock.patch('satpy.readers.eum_l2_bufr.recarray2dict') as recarray2dict:
                    recarray2dict.side_effect = (lambda x: x)

                    FILENAME_INFO = {'start_time': '20231022224500', 'spacecraft': TEST_DATA[filename]['platform_name'],
                                     'server': 'TESTSERVER'}
                    self.fh = EumetsatL2BufrFileHandler(filename, FILENAME_INFO, FILETYPE_INFO,
                                        with_area_definition=with_adef,
                                        rectification_longitude=int(TEST_DATA[filename]['RectificationLongitude'][1:])/10)
                    self.fh.mpef_header = MPEF_PRODUCT_HEADER

        else:
            # No Mpef Header  so we get the metadata from the BUFR messages
            with mock.patch('satpy.readers.eum_l2_bufr.open', self.m, create=True):
                with mock.patch('eccodes.codes_bufr_new_from_file',
                                side_effect=[self.buf1, None, self.buf1, None, self.buf1, None]) as ec1:
                    ec1.return_value = ec1.side_effect
                    with mock.patch('eccodes.codes_set') as ec2:
                        ec2.return_value = 1
                        with mock.patch('eccodes.codes_release') as ec5:
                            ec5.return_value = 1

                            FILENAME_INFO = {'start_time': '20191112000000', 'spacecraft': TEST_DATA[filename]['platform_name']}
                            self.fh = EumetsatL2BufrFileHandler(filename, FILENAME_INFO, FILETYPE_INFO,
                                                    with_area_definition=with_adef,
                                                    rectification_longitude=int(TEST_DATA[filename]['RectificationLongitude'][1:])/10)

        # Force resolution propertie in the file handler because the mock template doesn't have the
        # segmentSizeAtNadirInXDirection key so it can't be initialized the normal way
        self.fh.resolution = TEST_DATA[filename]['resolution']

    def get_data(self, dataset_name, key, coordinates):
        """Read data from mock file."""
        DATASET_INFO = {
            'name': dataset_name,
            'key': key,
            'fill_value': -1.e+100
        }
        if coordinates:
            DATASET_INFO.update({'coordinates': ('longitude', 'latitude')})

        with mock.patch('satpy.readers.eum_l2_bufr.open', self.m, create=True):
            with mock.patch('eccodes.codes_bufr_new_from_file',
                            side_effect=[self.buf1, self.buf1, None]) as ec1:
                ec1.return_value = ec1.side_effect
                with mock.patch('eccodes.codes_set') as ec2:
                    ec2.return_value = 1
                    with mock.patch('eccodes.codes_release') as ec5:
                        ec5.return_value = 1
                        z = self.fh.get_dataset(make_dataid(name = dataset_name, resolution = self.fh.resolution), DATASET_INFO)

        return z


@pytest.mark.parametrize("input_file", TEST_FILES)
class TestL2BufrReader:
    """Test SEVIRI L2 BUFR Reader."""

    @staticmethod
    def test_lonslats(input_file):
        print(input_file)
        """Test reading of longitude and latitude data with SEVIRI L2 BUFR reader."""
        bufr_obj = L2BufrData(input_file)
        print('get zlat')
        zlat = bufr_obj.get_data('latitude', '#1#latitude', coordinates=False)
        print(zlat)
        zlon = bufr_obj.get_data('longitude', '#1#longitude', coordinates=False)
        np.testing.assert_array_equal(zlat.values, np.concatenate((LAT, LAT), axis=0))
        np.testing.assert_array_equal(zlon.values, np.concatenate((LON, LON), axis=0))

    @staticmethod
    def test_attributes_with_swath_definition(input_file):
        """Test correctness of dataset attributes with data loaded with a SwathDefinition (default behaviour)."""
        bufr_obj = L2BufrData(input_file)
        z = bufr_obj.get_data(dataset_name='TestData', key=TEST_DATA[input_file]['key'], coordinates=True)
        assert z.attrs['platform_name'] == TEST_DATA[input_file]['platform_name']
        assert z.attrs['ssp_lon'] == TEST_DATA[input_file]['ssp_lon']
        assert z.attrs['seg_size'] == TEST_DATA[input_file]['seg_size']

    @staticmethod
    def test_attributes_with_area_definition(input_file):
        """Test correctness of dataset attributes with data loaded with a AreaDefinition."""
        bufr_obj = L2BufrData(input_file, with_adef=True)
        _ = bufr_obj.get_data('latitude', '#1#latitude', coordinates=False)  # We need to load the lat/lon data in order to
        _ = bufr_obj.get_data('longitude', '#1#longitude', coordinates=False)  # populate the file handler with these data

        z = bufr_obj.get_data(dataset_name='TestData', key=TEST_DATA[input_file]['key'], coordinates=True)
        assert z.attrs['platform_name'] == TEST_DATA[input_file]['platform_name']
        assert z.attrs['ssp_lon'] == TEST_DATA[input_file]['ssp_lon']
        assert z.attrs['seg_size'] == TEST_DATA[input_file]['seg_size']

    @staticmethod
    def test_data_with_swath_definition(input_file):
        """Test data loaded with SwathDefinition (default behaviour)."""
        bufr_obj = L2BufrData(input_file)
        with pytest.raises(NotImplementedError):
            bufr_obj.fh.get_area_def(None)

        # concatenate original test arrays as get_dataset will have read and concatented the data
        x1 = np.concatenate((DATA, DATA), axis=0)
        z = bufr_obj.get_data(dataset_name='TestData', key=TEST_DATA[input_file]['key'], coordinates=True)
        np.testing.assert_array_equal(z.values, x1)

    def test_data_with_area_definition(self, input_file):
        """Test data loaded with AreaDefinition."""
        bufr_obj = L2BufrData(input_file, with_adef=True)
        _ = bufr_obj.get_data('latitude', '#1#latitude', coordinates=False)  # We need to load the lat/lon data in order to
        _ = bufr_obj.get_data('longitude', '#1#longitude', coordinates=False)  # populate the file handler with these data

        z = bufr_obj.get_data(dataset_name='TestData', key=TEST_DATA[input_file]['key'], coordinates=True)

        ad = bufr_obj.fh.get_area_def(None)
        assert ad == TEST_DATA[input_file]['area']
        data_1d = np.concatenate((DATA, DATA), axis=0)

        # Put BUFR data on 2D grid that the 2D array returned by get_dataset should correspond to
        lons_1d, lats_1d = da.compute(bufr_obj.fh.longitude, bufr_obj.fh.latitude)
        icol, irow = ad.get_array_indices_from_lonlat(lons_1d, lats_1d)

        data_2d = np.empty(ad.shape)
        data_2d[:] = np.nan
        data_2d[irow.compressed(), icol.compressed()] = data_1d[~irow.mask]
        np.testing.assert_array_equal(z.values, data_2d)

        # Removed assert dedicated to products with seg_size=3 (covered by GII test case)

    def test_data_with_rect_lon(self, input_file):
        """Test data loaded with AreaDefinition and user defined rectification longitude."""
        bufr_obj = L2BufrData(input_file, with_adef=True)
        np.testing.assert_equal(bufr_obj.fh.ssp_lon, int(TEST_DATA[input_file]['RectificationLongitude'][1:])/10)
        _ = bufr_obj.get_data('latitude', '#1#latitude', coordinates=False)  # We need to load the lat/lon data in order to
        _ = bufr_obj.get_data('longitude', '#1#longitude', coordinates=False)  # populate the file handler with these data
        _ = bufr_obj.get_data(dataset_name='TestData', key=TEST_DATA[input_file]['key'], coordinates=True)
        # We need to lead the data in order to create the AreaDefinition

        ad = bufr_obj.fh.get_area_def(None)
        assert ad == TEST_DATA[input_file]['area']


class AMVBufrData:
    """Mock SEVIRI L2 AMV BUFR data."""

    @unittest.skipIf(sys.platform.startswith('win'), "'eccodes' not supported on Windows")
    def __init__(self, filename):
        """Initialize by mocking test data for testing the SEVIRI L2 BUFR reader."""
        from satpy.readers.eum_l2_bufr import EumetsatL2BufrFileHandler

        with mock.patch('satpy.readers.eum_l2_bufr.np.fromfile'):
            FILENAME_INFO = {'start_time': '20191112000000',
                             'spacecraft': 'MSG3',
                             'server': 'TESTSERVER'}
            self.fh = EumetsatL2BufrFileHandler(filename, FILENAME_INFO,
                                filetype_info={'file_type': 'seviri_l2_bufr_amv'},
                                with_area_definition=True)


class TestAMVBufrReader:
    """Test SEVIRI L2 BUFR Reader for AMV data."""

    @staticmethod
    def test_amv_with_area_def():
        """Test that AMV data can not be loaded with an area definition."""
        bufr_obj = AMVBufrData('AMVBUFRProd_20231023044500Z_00_OMPEFS03_MET10_FES_E0000')
        assert bufr_obj.fh.with_adef is False
