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
"""EUMETSAT base reader tests package.
"""

import sys
from datetime import datetime

import numpy as np

from satpy.readers.eum_base import (timecds2datetime, time_cds_short,
                                    time_cds, time_cds_expanded,
                                    recarray2dict)


if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class TestMakeTimeCdsDictionary(unittest.TestCase):

    def test_fun(self):

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

    def test_fun(self):

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

    def test_fun(self):

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


def suite():
    """The test suite for EUMETSAT base reader.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestMakeTimeCdsDictionary))
    mysuite.addTest(loader.loadTestsFromTestCase(TestMakeTimeCdsRecarray))
    mysuite.addTest(loader.loadTestsFromTestCase(TestRecarray2Dict))
    return mysuite


if __name__ == '__main__':
    unittest.main()
