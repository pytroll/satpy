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

import sys
import unittest
from datetime import datetime
from unittest import mock

import dask.array as da
import numpy as np
import pytest
from pyresample import geometry

from satpy.tests.utils import make_dataid

FILETYPE_INFO = {'file_type':  'seviri_l2_bufr_asr'}

FILENAME_INFO = {'start_time': '20191112000000',
                 'spacecraft': 'MSG2'}
FILENAME_INFO2 = {'start_time': '20191112000000',
                  'spacecraft': 'MSG2',
                  'server': 'TESTSERVER'}
MPEF_PRODUCT_HEADER = {
    'NominalTime': datetime(2019, 11, 6, 18, 0),
    'SpacecraftName': '09',
    'RectificationLongitude': 'E0455'
}

DATASET_INFO = {
    'name': 'testdata',
    'key': '#1#brightnessTemperature',
    'coordinates': ('longitude', 'latitude'),
    'fill_value': 0
}

DATASET_INFO_LAT = {
    'name': 'latitude',
    'key': 'latitude',
    'fill_value': -1.e+100
}

DATASET_INFO_LON = {
    'name': 'longitude',
    'key': 'longitude',
    'fill_value': -1.e+100
}


DATASET_ATTRS = {
    'platform_name': 'MET09',
    'ssp_lon': 45.5,
    'seg_size': 16
}

AREA_DEF = geometry.AreaDefinition(
    'msg_seviri_iodc_48km',
    'MSG SEVIRI Indian Ocean Data Coverage service area definition with 48 km resolution',
    "",
    {'a': 6378169., 'b': 6356583.8, 'lon_0': DATASET_ATTRS['ssp_lon'],
     'h': 35785831., 'proj': 'geos', 'units': 'm'},
    232,
    232,
    (-5570248.6867, -5567248.2834, 5567248.2834, 5570248.6867)
)

AREA_DEF_FES = geometry.AreaDefinition(
    'msg_seviri_res_48km',
    'MSG SEVIRI Full Earth Scanning service area definition with 48 km resolution',
    "",
    {'a': 6378169., 'b': 6356583.8, 'lon_0': 0.0,
     'h': 35785831., 'proj': 'geos', 'units': 'm'},
    232,
    232,
    (-5570248.6867, -5567248.2834, 5567248.2834, 5570248.6867)
)

AREA_DEF_EXT = geometry.AreaDefinition(
    'msg_seviri_iodc_9km_ext',
    'MSG SEVIRI Indian Ocean Data Coverage service area definition with 9 km resolution '
    '(extended outside original 3km grid)',
    "",
    {'a': 6378169., 'b': 6356583.8, 'lon_0': DATASET_ATTRS['ssp_lon'],
     'h': 35785831., 'proj': 'geos', 'units': 'm'},
    1238,
    1238,
    (-5571748.8883, -5571748.8882, 5571748.8882, 5571748.8883)
)

TEST_FILES = [
    'ASRBUFRProd_20191106130000Z_00_OMPEFS02_MET09_FES_E0000',
    'MSG2-SEVI-MSGASRE-0101-0101-20191106130000.000000000Z-20191106131702-1362128.bfr',
    'MSG2-SEVI-MSGASRE-0101-0101-20191106101500.000000000Z-20191106103218-1362148'
]

# Test data
DATA = np.random.uniform(low=250, high=350, size=(128,))
LAT = np.random.uniform(low=-80, high=80, size=(128,))
LON = np.random.uniform(low=-38.5, high=121.5, size=(128,))


class SeviriL2BufrData:
    """Mock SEVIRI L2 BUFR data."""

    @unittest.skipIf(sys.platform.startswith('win'), "'eccodes' not supported on Windows")
    def __init__(self, filename, with_adef=False, rect_lon='default'):
        """Initialize by mocking test data for testing the SEVIRI L2 BUFR reader."""
        import eccodes as ec

        from satpy.readers.seviri_l2_bufr import SeviriL2BufrFileHandler
        self.buf1 = ec.codes_bufr_new_from_samples('BUFR4_local_satellite')
        ec.codes_set(self.buf1, 'unpack', 1)
        # write the bufr test data twice as we want to read in and then concatenate the data in the reader
        # 55 id corresponds to METEOSAT 8`
        ec.codes_set(self.buf1, 'satelliteIdentifier', 56)
        ec.codes_set_array(self.buf1, 'latitude', LAT)
        ec.codes_set_array(self.buf1, 'latitude', LAT)
        ec.codes_set_array(self.buf1, 'longitude', LON)
        ec.codes_set_array(self.buf1, 'longitude', LON)
        ec.codes_set_array(self.buf1, '#1#brightnessTemperature', DATA)
        ec.codes_set_array(self.buf1, '#1#brightnessTemperature', DATA)

        self.m = mock.mock_open()
        # only our offline product contain MPEF product headers so we get the metadata from there
        if ('BUFRProd' in filename):
            with mock.patch('satpy.readers.seviri_l2_bufr.np.fromfile') as fromfile:
                fromfile.return_value = MPEF_PRODUCT_HEADER
                with mock.patch('satpy.readers.seviri_l2_bufr.recarray2dict') as recarray2dict:
                    recarray2dict.side_effect = (lambda x: x)
                    self.fh = SeviriL2BufrFileHandler(filename, FILENAME_INFO2, FILETYPE_INFO,
                                                      with_area_definition=with_adef, rectification_longitude=rect_lon)
                    self.fh.mpef_header = MPEF_PRODUCT_HEADER

        else:
            # No Mpef Header  so we get the metadata from the BUFR messages
            with mock.patch('satpy.readers.seviri_l2_bufr.open', self.m, create=True):
                with mock.patch('eccodes.codes_bufr_new_from_file',
                                side_effect=[self.buf1, None, self.buf1, None, self.buf1, None]) as ec1:
                    ec1.return_value = ec1.side_effect
                    with mock.patch('eccodes.codes_set') as ec2:
                        ec2.return_value = 1
                        with mock.patch('eccodes.codes_release') as ec5:
                            ec5.return_value = 1
                            self.fh = SeviriL2BufrFileHandler(filename, FILENAME_INFO, FILETYPE_INFO,
                                                              with_area_definition=with_adef,
                                                              rectification_longitude=rect_lon)

    def get_data(self, dataset_info):
        """Read data from mock file."""
        with mock.patch('satpy.readers.seviri_l2_bufr.open', self.m, create=True):
            with mock.patch('eccodes.codes_bufr_new_from_file',
                            side_effect=[self.buf1, self.buf1, None]) as ec1:
                ec1.return_value = ec1.side_effect
                with mock.patch('eccodes.codes_set') as ec2:
                    ec2.return_value = 1
                    with mock.patch('eccodes.codes_release') as ec5:
                        ec5.return_value = 1
                        z = self.fh.get_dataset(make_dataid(name=dataset_info['name'], resolution=48000), dataset_info)

        return z


@pytest.mark.parametrize("input_file", TEST_FILES)
class TestSeviriL2BufrReader:
    """Test SEVIRI L2 BUFR Reader."""

    @staticmethod
    def test_lonslats(input_file):
        """Test reading of longitude and latitude data with SEVIRI L2 BUFR reader."""
        bufr_obj = SeviriL2BufrData(input_file)
        zlat = bufr_obj.get_data(DATASET_INFO_LAT)
        zlon = bufr_obj.get_data(DATASET_INFO_LON)
        np.testing.assert_array_equal(zlat.values, np.concatenate((LAT, LAT), axis=0))
        np.testing.assert_array_equal(zlon.values, np.concatenate((LON, LON), axis=0))

    @staticmethod
    def test_attributes_with_swath_definition(input_file):
        """Test correctness of dataset attributes with data loaded with a SwathDefinition (default behaviour)."""
        bufr_obj = SeviriL2BufrData(input_file)
        z = bufr_obj.get_data(DATASET_INFO)
        assert z.attrs['platform_name'] == DATASET_ATTRS['platform_name']
        assert z.attrs['ssp_lon'] == DATASET_ATTRS['ssp_lon']
        assert z.attrs['seg_size'] == DATASET_ATTRS['seg_size']

    @staticmethod
    def test_attributes_with_area_definition(input_file):
        """Test correctness of dataset attributes with data loaded with a AreaDefinition."""
        bufr_obj = SeviriL2BufrData(input_file, with_adef=True)
        _ = bufr_obj.get_data(DATASET_INFO_LAT)  # We need to load the lat/lon data in order to
        _ = bufr_obj.get_data(DATASET_INFO_LON)  # populate the file handler with these data
        z = bufr_obj.get_data(DATASET_INFO)
        assert z.attrs['platform_name'] == DATASET_ATTRS['platform_name']
        assert z.attrs['ssp_lon'] == DATASET_ATTRS['ssp_lon']
        assert z.attrs['seg_size'] == DATASET_ATTRS['seg_size']

    @staticmethod
    def test_data_with_swath_definition(input_file):
        """Test data loaded with SwathDefinition (default behaviour)."""
        bufr_obj = SeviriL2BufrData(input_file)
        with pytest.raises(NotImplementedError):
            bufr_obj.fh.get_area_def(None)

        # concatenate original test arrays as get_dataset will have read and concatented the data
        x1 = np.concatenate((DATA, DATA), axis=0)
        z = bufr_obj.get_data(DATASET_INFO)
        np.testing.assert_array_equal(z.values, x1)

    def test_data_with_area_definition(self, input_file):
        """Test data loaded with AreaDefinition."""
        bufr_obj = SeviriL2BufrData(input_file, with_adef=True)
        _ = bufr_obj.get_data(DATASET_INFO_LAT)  # We need to load the lat/lon data in order to
        _ = bufr_obj.get_data(DATASET_INFO_LON)  # populate the file handler with these data
        z = bufr_obj.get_data(DATASET_INFO)

        ad = bufr_obj.fh.get_area_def(None)
        assert ad == AREA_DEF
        data_1d = np.concatenate((DATA, DATA), axis=0)

        # Put BUFR data on 2D grid that the 2D array returned by get_dataset should correspond to
        lons_1d, lats_1d = da.compute(bufr_obj.fh.longitude, bufr_obj.fh.latitude)
        icol, irow = ad.get_array_indices_from_lonlat(lons_1d, lats_1d)

        data_2d = np.empty(ad.shape)
        data_2d[:] = np.nan
        data_2d[irow.compressed(), icol.compressed()] = data_1d[~irow.mask]
        np.testing.assert_array_equal(z.values, data_2d)

        # Test that the correct AreaDefinition is identified for products with 3 pixel segements
        bufr_obj.fh.seg_size = 3
        ad_ext = bufr_obj.fh._construct_area_def(make_dataid(name='dummmy', resolution=9000))
        assert ad_ext == AREA_DEF_EXT

    def test_data_with_rect_lon(self, input_file):
        """Test data loaded with AreaDefinition and user defined rectification longitude."""
        bufr_obj = SeviriL2BufrData(input_file, with_adef=True, rect_lon=0.0)
        np.testing.assert_equal(bufr_obj.fh.ssp_lon, 0.0)
        _ = bufr_obj.get_data(DATASET_INFO_LAT)  # We need to load the lat/lon data in order to
        _ = bufr_obj.get_data(DATASET_INFO_LON)  # populate the file handler with these data
        _ = bufr_obj.get_data(DATASET_INFO)  # We need to lead the data in order to create the AreaDefinition

        ad = bufr_obj.fh.get_area_def(None)
        assert ad == AREA_DEF_FES


class SeviriL2AMVBufrData:
    """Mock SEVIRI L2 AMV BUFR data."""

    @unittest.skipIf(sys.platform.startswith('win'), "'eccodes' not supported on Windows")
    def __init__(self, filename):
        """Initialize by mocking test data for testing the SEVIRI L2 BUFR reader."""
        from satpy.readers.seviri_l2_bufr import SeviriL2BufrFileHandler

        with mock.patch('satpy.readers.seviri_l2_bufr.np.fromfile'):
            self.fh = SeviriL2BufrFileHandler(filename, FILENAME_INFO2,
                                              filetype_info={'file_type': 'seviri_l2_bufr_amv'},
                                              with_area_definition=True)


class TestSeviriL2AMVBufrReader:
    """Test SEVIRI L2 BUFR Reader for AMV data."""

    @staticmethod
    def test_amv_with_area_def():
        """Test that AMV data can not be loaded with an area definition."""
        bufr_obj = SeviriL2AMVBufrData('AMVBUFRProd_20201110124500Z_00_OMPEFS04_MET11_FES_E0000')
        assert bufr_obj.fh.with_adef is False
