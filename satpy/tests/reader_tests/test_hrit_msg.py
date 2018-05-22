# -*- coding: utf-8 -*-

# Copyright (c) 2017 Martin Raspaud

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""The hrit msg reader tests package.
"""

import sys
from datetime import datetime

from satpy.readers.hrit_msg import (HRITMSGPrologueFileHandler,
                                    make_time_cds_expanded)

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


class TestMakeTimeCDSExpanded(unittest.TestCase):
    def test_fun(self):
        tcds = {'days': 1, 'milliseconds': 2, 'microseconds': 3, 'nanoseconds': 4}
        expected = datetime(1958, 1, 2, 0, 0, 0, 2003)
        self.assertEqual(make_time_cds_expanded(tcds), expected)


class TestHRITMSGPrologueFileHandler(unittest.TestCase):
    """Test the HRITFileHandler."""

    @mock.patch('satpy.readers.hrit_msg.make_time_cds_expanded')
    @mock.patch('satpy.readers.hrit_msg.recarray2dict')
    @mock.patch('satpy.readers.hrit_msg.np.fromfile')
    @mock.patch('satpy.readers.hrit_msg.HRITFileHandler.__init__')
    def test_init(self, new_fh_init, fromfile, recarray2dict, make_time):
        """Setup the hrit file handler for testing."""
        recarray2dict.side_effect = lambda x: x
        make_time.side_effect = lambda x: 'trans' + x
        new_fh_init.return_value.filename = 'filename'
        HRITMSGPrologueFileHandler.filename = 'filename'
        HRITMSGPrologueFileHandler.mda = {'total_header_length': 1}
        ret = {'ImageAcquisition': {'PlannedAcquisitionTime': {}}}
        ret["ImageAcquisition"]['PlannedAcquisitionTime']['TrueRepeatCycleStart'] = 'start'
        ret["ImageAcquisition"]['PlannedAcquisitionTime']['PlannedForwardScanEnd'] = 'endscan'
        ret["ImageAcquisition"]['PlannedAcquisitionTime']['PlannedRepeatCycleEnd'] = 'endcyle'
        fromfile.return_value = [ret]
        m = mock.mock_open()
        with mock.patch('satpy.readers.hrit_msg.open', m, create=True) as newopen:
            newopen.return_value.__enter__.return_value.seek.return_value = 1
            self.reader = HRITMSGPrologueFileHandler(
                'filename', {'platform_shortname': 'MSG3',
                             'start_time': datetime(2016, 3, 3, 0, 0),
                             'service': 'test_service'},
                {'filetype': 'info'})

        ret = {'ImageAcquisition': {'PlannedAcquisitionTime': {}}}
        ret["ImageAcquisition"]['PlannedAcquisitionTime']['TrueRepeatCycleStart'] = 'transstart'
        ret["ImageAcquisition"]['PlannedAcquisitionTime']['PlannedForwardScanEnd'] = 'transendscan'
        ret["ImageAcquisition"]['PlannedAcquisitionTime']['PlannedRepeatCycleEnd'] = 'transendcyle'
        self.assertEqual(ret, self.reader.prologue)


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestHRITMSGPrologueFileHandler))
    mysuite.addTest(loader.loadTestsFromTestCase(TestMakeTimeCDSExpanded))
    return mysuite


if __name__ == '__main__':
    unittest.main()
