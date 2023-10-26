#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Satpy developers
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
"""EUMETSAT base reader tests package."""

import unittest
from datetime import datetime

import numpy as np

from satpy.readers.eum_base import (
    get_service_mode,
    recarray2dict,
    time_cds,
    time_cds_expanded,
    time_cds_short,
    timecds2datetime,
)
from satpy.readers.seviri_base import mpef_product_header


class TestMakeTimeCdsDictionary(unittest.TestCase):
    """Test TestMakeTimeCdsDictionary."""

    def test_fun(self):
        """Test function for TestMakeTimeCdsDictionary."""
        # time_cds_short
        tcds = {'Days': 1, 'Milliseconds': 2}
        expected = datetime(1958, 1, 2, 0, 0, 0, 2000)
        self.assertEqual(timecds2datetime(tcds), expected)

        # time_cds
        tcds = {'Days': 1, 'Milliseconds': 2, 'Microseconds': 3}
        expected = datetime(1958, 1, 2, 0, 0, 0, 2003)
        self.assertEqual(timecds2datetime(tcds), expected)

        # time_cds_expanded
        tcds = {'Days': 1, 'Milliseconds': 2, 'Microseconds': 3, 'Nanoseconds': 4}
        expected = datetime(1958, 1, 2, 0, 0, 0, 2003)
        self.assertEqual(timecds2datetime(tcds), expected)


class TestMakeTimeCdsRecarray(unittest.TestCase):
    """Test TestMakeTimeCdsRecarray."""

    def test_fun(self):
        """Test function for TestMakeTimeCdsRecarray."""
        # time_cds_short
        tcds = np.array([(1, 2)], dtype=np.dtype(time_cds_short))
        expected = datetime(1958, 1, 2, 0, 0, 0, 2000)
        self.assertEqual(timecds2datetime(tcds), expected)

        # time_cds
        tcds = np.array([(1, 2, 3)], dtype=np.dtype(time_cds))
        expected = datetime(1958, 1, 2, 0, 0, 0, 2003)
        self.assertEqual(timecds2datetime(tcds), expected)

        # time_cds_expanded
        tcds = np.array([(1, 2, 3, 4)], dtype=np.dtype(time_cds_expanded))
        expected = datetime(1958, 1, 2, 0, 0, 0, 2003)
        self.assertEqual(timecds2datetime(tcds), expected)


class TestRecarray2Dict(unittest.TestCase):
    """Test TestRecarray2Dict."""

    def test_timestamps(self):
        """Test function for TestRecarray2Dict."""
        # datatype definition

        pat_dt = np.dtype([
            ('TrueRepeatCycleStart', time_cds_expanded),
            ('PlanForwardScanEnd', time_cds_expanded),
            ('PlannedRepeatCycleEnd', time_cds_expanded)
        ])

        # planned acquisition time, add extra dimensions
        # these should be removed by recarray2dict
        pat = np.array([[[(
            (21916, 41409544, 305, 262),
            (21916, 42160340, 659, 856),
            (21916, 42309417, 918, 443))]]], dtype=pat_dt)

        expected = {
            'TrueRepeatCycleStart': datetime(2018, 1, 2, 11, 30, 9, 544305),
            'PlanForwardScanEnd': datetime(2018, 1, 2, 11, 42, 40, 340660),
            'PlannedRepeatCycleEnd': datetime(2018, 1, 2, 11, 45, 9, 417918)
        }

        self.assertEqual(recarray2dict(pat), expected)

    def test_mpef_product_header(self):
        """Test function for TestRecarray2Dict and mpef product header."""
        names = ['ImageLocation', 'GsicsCalMode', 'GsicsCalValidity',
                 'Padding', 'OffsetToData', 'Padding2']
        mpef_header = np.dtype([(name, mpef_product_header.fields[name][0])
                                for name in names])
        mph_struct = np.array([('OPE', True, False, 'XX', 1000, '12345678')], dtype=mpef_header)
        test_mph = {'ImageLocation': "OPE",
                    'GsicsCalMode': True,
                    'GsicsCalValidity': False,
                    'Padding': 'XX',
                    'OffsetToData': 1000,
                    'Padding2': '12345678'
                    }
        self.assertEqual(recarray2dict(mph_struct), test_mph)


class TestGetServiceMode(unittest.TestCase):
    """Test the get_service_mode function."""

    def test_get_seviri_service_mode_fes(self):
        """Test fetching of SEVIRI service mode information for FES."""
        ssp_lon = 0.0
        name = 'fes'
        desc = 'Full Earth Scanning service'
        res = get_service_mode('seviri', ssp_lon)
        self.assertEqual(res['service_name'], name)
        self.assertEqual(res['service_desc'], desc)

    def test_get_seviri_service_mode_rss(self):
        """Test fetching of SEVIRI service mode information for RSS."""
        ssp_lon = 9.5
        name = 'rss'
        desc = 'Rapid Scanning Service'
        res = get_service_mode('seviri', ssp_lon)
        self.assertEqual(res['service_name'], name)
        self.assertEqual(res['service_desc'], desc)

    def test_get_seviri_service_mode_iodc_E0415(self):
        """Test fetching of SEVIRI service mode information for IODC at 41.5 degrees East."""
        ssp_lon = 41.5
        name = 'iodc'
        desc = 'Indian Ocean Data Coverage service'
        res = get_service_mode('seviri', ssp_lon)
        self.assertEqual(res['service_name'], name)
        self.assertEqual(res['service_desc'], desc)

    def test_get_seviri_service_mode_iodc_E0455(self):
        """Test fetching of SEVIRI service mode information for IODC at 45.5 degrees East."""
        ssp_lon = 45.5
        name = 'iodc'
        desc = 'Indian Ocean Data Coverage service'
        res = get_service_mode('seviri', ssp_lon)
        self.assertEqual(res['service_name'], name)
        self.assertEqual(res['service_desc'], desc)

    def test_get_fci_service_mode_fdss(self):
        """Test fetching of FCI service mode information for FDSS."""
        ssp_lon = 0.0
        name = 'fdss'
        desc = 'Full Disk Scanning Service'
        res = get_service_mode('fci', ssp_lon)
        self.assertEqual(res['service_name'], name)
        self.assertEqual(res['service_desc'], desc)

    def test_get_fci_service_mode_rss(self):
        """Test fetching of FCI service mode information for RSS."""
        ssp_lon = 9.5
        name = 'rss'
        desc = 'Rapid Scanning Service'
        res = get_service_mode('fci', ssp_lon)
        self.assertEqual(res['service_name'], name)
        self.assertEqual(res['service_desc'], desc)

    def test_get_unknown_lon_service_mode(self):
        """Test fetching of service mode information for unknown input longitude."""
        ssp_lon = 13
        name = 'unknown'
        desc = 'unknown'
        res = get_service_mode('fci', ssp_lon)
        self.assertEqual(res['service_name'], name)
        self.assertEqual(res['service_desc'], desc)

    def test_get_unknown_instrument_service_mode(self):
        """Test fetching of service mode information for unknown input instrument."""
        ssp_lon = 0
        name = 'unknown'
        desc = 'unknown'
        res = get_service_mode('test', ssp_lon)
        self.assertEqual(res['service_name'], name)
        self.assertEqual(res['service_desc'], desc)
