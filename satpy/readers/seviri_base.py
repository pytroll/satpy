#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2018 Satpy developers
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
"""Utilities and helper classes for MSG HRIT/Native data reading.

References:
    MSG Level 1.5 Image Data Format Description
    https://www.eumetsat.int/website/wcm/idc/idcplg?IdcService=GET_FILE&dDocName=PDF_TEN_05105_MSG_IMG_DATA&RevisionSelectionMethod=LatestReleased&Rendition=Web

"""

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
import dask.array as da

from satpy.readers.eum_base import (time_cds_short,
                                    issue_revision)

PLATFORM_DICT = {
    'MET08': 'Meteosat-8',
    'MET09': 'Meteosat-9',
    'MET10': 'Meteosat-10',
    'MET11': 'Meteosat-11',
    'MSG1': 'Meteosat-8',
    'MSG2': 'Meteosat-9',
    'MSG3': 'Meteosat-10',
    'MSG4': 'Meteosat-11',
}

REPEAT_CYCLE_DURATION = 15

C1 = 1.19104273e-5
C2 = 1.43877523

VISIR_NUM_COLUMNS = 3712
VISIR_NUM_LINES = 3712
HRV_NUM_COLUMNS = 11136

CHANNEL_NAMES = {1: "VIS006",
                 2: "VIS008",
                 3: "IR_016",
                 4: "IR_039",
                 5: "WV_062",
                 6: "WV_073",
                 7: "IR_087",
                 8: "IR_097",
                 9: "IR_108",
                 10: "IR_120",
                 11: "IR_134",
                 12: "HRV"}

VIS_CHANNELS = ['HRV', 'VIS006', 'VIS008', 'IR_016']

# Polynomial coefficients for spectral-effective BT fits
BTFIT = {}
# [A, B, C]
BTFIT['IR_039'] = [0.0, 1.011751900, -3.550400]
BTFIT['WV_062'] = [0.00001805700, 1.000255533, -1.790930]
BTFIT['WV_073'] = [0.00000231818, 1.000668281, -0.456166]
BTFIT['IR_087'] = [-0.00002332000, 1.011803400, -1.507390]
BTFIT['IR_097'] = [-0.00002055330, 1.009370670, -1.030600]
BTFIT['IR_108'] = [-0.00007392770, 1.032889800, -3.296740]
BTFIT['IR_120'] = [-0.00007009840, 1.031314600, -3.181090]
BTFIT['IR_134'] = [-0.00007293450, 1.030424800, -2.645950]

SATNUM = {321: "8",
          322: "9",
          323: "10",
          324: "11"}

CALIB = {}

# Meteosat 8
CALIB[321] = {'HRV': {'F': 78.7599 / np.pi},
              'VIS006': {'F': 65.2296 / np.pi},
              'VIS008': {'F': 73.0127 / np.pi},
              'IR_016': {'F': 62.3715 / np.pi},
              'IR_039': {'VC': 2567.33,
                         'ALPHA': 0.9956,
                         'BETA': 3.41},
              'WV_062': {'VC': 1598.103,
                         'ALPHA': 0.9962,
                         'BETA': 2.218},
              'WV_073': {'VC': 1362.081,
                         'ALPHA': 0.9991,
                         'BETA': 0.478},
              'IR_087': {'VC': 1149.069,
                         'ALPHA': 0.9996,
                         'BETA': 0.179},
              'IR_097': {'VC': 1034.343,
                         'ALPHA': 0.9999,
                         'BETA': 0.06},
              'IR_108': {'VC': 930.647,
                         'ALPHA': 0.9983,
                         'BETA': 0.625},
              'IR_120': {'VC': 839.66,
                         'ALPHA': 0.9988,
                         'BETA': 0.397},
              'IR_134': {'VC': 752.387,
                         'ALPHA': 0.9981,
                         'BETA': 0.578}}

# Meteosat 9
CALIB[322] = {'HRV': {'F': 79.0113 / np.pi},
              'VIS006': {'F': 65.2065 / np.pi},
              'VIS008': {'F': 73.1869 / np.pi},
              'IR_016': {'F': 61.9923 / np.pi},
              'IR_039': {'VC': 2568.832,
                         'ALPHA': 0.9954,
                         'BETA': 3.438},
              'WV_062': {'VC': 1600.548,
                         'ALPHA': 0.9963,
                         'BETA': 2.185},
              'WV_073': {'VC': 1360.330,
                         'ALPHA': 0.9991,
                         'BETA': 0.47},
              'IR_087': {'VC': 1148.620,
                         'ALPHA': 0.9996,
                         'BETA': 0.179},
              'IR_097': {'VC': 1035.289,
                         'ALPHA': 0.9999,
                         'BETA': 0.056},
              'IR_108': {'VC': 931.7,
                         'ALPHA': 0.9983,
                         'BETA': 0.64},
              'IR_120': {'VC': 836.445,
                         'ALPHA': 0.9988,
                         'BETA': 0.408},
              'IR_134': {'VC': 751.792,
                         'ALPHA': 0.9981,
                         'BETA': 0.561}}

# Meteosat 10
CALIB[323] = {'HRV': {'F': 78.9416 / np.pi},
              'VIS006': {'F': 65.5148 / np.pi},
              'VIS008': {'F': 73.1807 / np.pi},
              'IR_016': {'F': 62.0208 / np.pi},
              'IR_039': {'VC': 2547.771,
                         'ALPHA': 0.9915,
                         'BETA': 2.9002},
              'WV_062': {'VC': 1595.621,
                         'ALPHA': 0.9960,
                         'BETA': 2.0337},
              'WV_073': {'VC': 1360.337,
                         'ALPHA': 0.9991,
                         'BETA': 0.4340},
              'IR_087': {'VC': 1148.130,
                         'ALPHA': 0.9996,
                         'BETA': 0.1714},
              'IR_097': {'VC': 1034.715,
                         'ALPHA': 0.9999,
                         'BETA': 0.0527},
              'IR_108': {'VC': 929.842,
                         'ALPHA': 0.9983,
                         'BETA': 0.6084},
              'IR_120': {'VC': 838.659,
                         'ALPHA': 0.9988,
                         'BETA': 0.3882},
              'IR_134': {'VC': 750.653,
                         'ALPHA': 0.9982,
                         'BETA': 0.5390}}

# Meteosat 11
CALIB[324] = {'HRV': {'F': 79.0035 / np.pi},
              'VIS006': {'F': 65.2656 / np.pi},
              'VIS008': {'F': 73.1692 / np.pi},
              'IR_016': {'F': 61.9416 / np.pi},
              'IR_039': {'VC': 2555.280,
                         'ALPHA': 0.9916,
                         'BETA': 2.9438},
              'WV_062': {'VC': 1596.080,
                         'ALPHA': 0.9959,
                         'BETA': 2.0780},
              'WV_073': {'VC': 1361.748,
                         'ALPHA': 0.9990,
                         'BETA': 0.4929},
              'IR_087': {'VC': 1147.433,
                         'ALPHA': 0.9996,
                         'BETA': 0.1731},
              'IR_097': {'VC': 1034.851,
                         'ALPHA': 0.9998,
                         'BETA': 0.0597},
              'IR_108': {'VC': 931.122,
                         'ALPHA': 0.9983,
                         'BETA': 0.6256},
              'IR_120': {'VC': 839.113,
                         'ALPHA': 0.9988,
                         'BETA': 0.4002},
              'IR_134': {'VC': 748.585,
                         'ALPHA': 0.9981,
                         'BETA': 0.5635}}


def get_cds_time(days, msecs):
    """Compute timestamp given the days since epoch and milliseconds of the day.

    1958-01-01 00:00 is interpreted as fill value and will be replaced by NaT (Not a Time).

    Args:
        days (int, either scalar or numpy.ndarray):
            Days since 1958-01-01
        msecs (int, either scalar or numpy.ndarray):
            Milliseconds of the day

    Returns:
        numpy.datetime64: Timestamp(s)

    """
    if np.isscalar(days):
        days = np.array([days], dtype='int64')
        msecs = np.array([msecs], dtype='int64')

    time = np.datetime64('1958-01-01').astype('datetime64[ms]') + \
        days.astype('timedelta64[D]') + msecs.astype('timedelta64[ms]')
    time[time == np.datetime64('1958-01-01 00:00')] = np.datetime64("NaT")

    if len(time) == 1:
        return time[0]
    return time


def dec10216(inbuf):
    """Decode 10 bits data into 16 bits words.

    ::

        /*
         * pack 4 10-bit words in 5 bytes into 4 16-bit words
         *
         * 0       1       2       3       4       5
         * 01234567890123456789012345678901234567890
         * 0         1         2         3         4
         */
        ip = &in_buffer[i];
        op = &out_buffer[j];
        op[0] = ip[0]*4 + ip[1]/64;
        op[1] = (ip[1] & 0x3F)*16 + ip[2]/16;
        op[2] = (ip[2] & 0x0F)*64 + ip[3]/4;
        op[3] = (ip[3] & 0x03)*256 +ip[4];

    """
    arr10 = inbuf.astype(np.uint16)
    arr16_len = int(len(arr10) * 4 / 5)
    arr10_len = int((arr16_len * 5) / 4)
    arr10 = arr10[:arr10_len]  # adjust size

    # dask is slow with indexing
    arr10_0 = arr10[::5]
    arr10_1 = arr10[1::5]
    arr10_2 = arr10[2::5]
    arr10_3 = arr10[3::5]
    arr10_4 = arr10[4::5]

    arr16_0 = (arr10_0 << 2) + (arr10_1 >> 6)
    arr16_1 = ((arr10_1 & 63) << 4) + (arr10_2 >> 4)
    arr16_2 = ((arr10_2 & 15) << 6) + (arr10_3 >> 2)
    arr16_3 = ((arr10_3 & 3) << 8) + arr10_4
    arr16 = da.stack([arr16_0, arr16_1, arr16_2, arr16_3], axis=-1).ravel()
    arr16 = da.rechunk(arr16, arr16.shape[0])

    return arr16


class MpefProductHeader(object):
    """MPEF product header class."""

    def get(self):
        """Return numpy record_array for MPEF product header."""
        record = [
            ('MPEF_File_Id', np.int16),
            ('MPEF_Header_Version', np.uint8),
            ('ManualDissAuthRequest', np.bool),
            ('ManualDisseminationAuth', np.bool),
            ('DisseminationAuth', np.bool),
            ('NominalTime', time_cds_short),
            ('ProductQuality', np.uint8),
            ('ProductCompleteness', np.uint8),
            ('ProductTimeliness', np.uint8),
            ('ProcessingInstanceId', np.int8),
            ('ImagesUsed', self.images_used, (4,)),
            ('BaseAlgorithmVersion',
             issue_revision),
            ('ProductAlgorithmVersion',
             issue_revision),
            ('InstanceServerName', 'S2'),
            ('SpacecraftName', 'S2'),
            ('Mission', 'S3'),
            ('RectificationLongitude', 'S5'),
            ('Encoding', 'S1'),
            ('TerminationSpace', 'S1'),
            ('EncodingVersion', np.uint16),
            ('Channel', np.uint8),
            ('Filler', 'S20'),
            ('RepeatCycle', 'S15'),
        ]

        return np.dtype(record).newbyteorder('>')

    @property
    def images_used(self):
        """Return structure for images_used."""
        record = [
            ('Padding1', 'S2'),
            ('ExpectedImage', time_cds_short),
            ('ImageReceived', np.bool),
            ('Padding2', 'S1'),
            ('UsedImageStart_Day', np.uint16),
            ('UsedImageStart_Millsec', np.uint32),
            ('Padding3', 'S2'),
            ('UsedImageEnd_Day', np.uint16),
            ('UsedImageEndt_Millsec', np.uint32),
        ]

        return record


mpef_product_header = MpefProductHeader().get()


class SEVIRICalibrationHandler(object):
    """Calibration handler for SEVIRI HRIT- and native-formats."""

    def _convert_to_radiance(self, data, gain, offset):
        """Calibrate to radiance."""
        return (data * gain + offset).clip(0.0, None)

    def _erads2bt(self, data, channel_name):
        """Convert effective radiance to brightness temperature."""
        cal_info = CALIB[self.platform_id][channel_name]
        alpha = cal_info["ALPHA"]
        beta = cal_info["BETA"]
        wavenumber = CALIB[self.platform_id][channel_name]["VC"]

        return (self._tl15(data, wavenumber) - beta) / alpha

    def _ir_calibrate(self, data, channel_name, cal_type):
        """Calibrate to brightness temperature."""
        if cal_type == 1:
            # spectral radiances
            return self._srads2bt(data, channel_name)
        elif cal_type == 2:
            # effective radiances
            return self._erads2bt(data, channel_name)
        else:
            raise NotImplementedError('Unknown calibration type')

    def _srads2bt(self, data, channel_name):
        """Convert spectral radiance to brightness temperature."""
        a__, b__, c__ = BTFIT[channel_name]
        wavenumber = CALIB[self.platform_id][channel_name]["VC"]
        temp = self._tl15(data, wavenumber)

        return a__ * temp * temp + b__ * temp + c__

    def _tl15(self, data, wavenumber):
        """Compute the L15 temperature."""
        return ((C2 * wavenumber) /
                np.log((1.0 / data) * C1 * wavenumber ** 3 + 1.0))

    def _vis_calibrate(self, data, solar_irradiance):
        """Calibrate to reflectance."""
        return data * 100.0 / solar_irradiance


def chebyshev(coefs, time, domain):
    """Evaluate a Chebyshev Polynomial.

    Args:
        coefs (list, np.array): Coefficients defining the polynomial
        time (int, float): Time where to evaluate the polynomial
        domain (list, tuple): Domain (or time interval) for which the polynomial is defined: [left, right]
    Reference: Appendix A in the MSG Level 1.5 Image Data Format Description.

    """
    return Chebyshev(coefs, domain=domain)(time) - 0.5 * coefs[0]


def calculate_area_extent(area_dict):
    """Calculate the area extent seen by a geostationary satellite.

    Args:
        area_dict: A dictionary containing the required parameters
            center_point: Center point for the projection
            resolution: Pixel resulution in meters
            north: Northmost row number
            east: Eastmost column number
            west: Westmost column number
            south: Southmost row number
            [column_offset: Column offset, defaults to 0 if not given]
            [row_offset: Row offset, defaults to 0 if not given]
    Returns:
        tuple: An area extent for the scene defined by the lower left and
               upper right corners

    """
    # For Earth model 2 and full disk resolution center point
    # column and row is (1856.5, 1856.5)
    # See: MSG Level 1.5 Image Data Format Description, Figure 7
    cp_c = area_dict['center_point'] + area_dict.get('column_offset', 0)
    cp_r = area_dict['center_point'] + area_dict.get('row_offset', 0)

    # Calculate column and row for lower left and upper right corners.
    ll_c = (area_dict['west'] - cp_c)
    ll_r = (area_dict['north'] - cp_r + 1)
    ur_c = (area_dict['east'] - cp_c - 1)
    ur_r = (area_dict['south'] - cp_r)

    aex = np.array([ll_c, ll_r, ur_c, ur_r]) * area_dict['resolution']

    return tuple(aex)
