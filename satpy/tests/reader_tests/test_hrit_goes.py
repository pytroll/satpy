# -*- coding: utf-8 -*-

# Copyright (c) 2018 Martin Raspaud

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
import datetime
import numpy as np
from satpy.readers.hrit_goes import (make_gvar_float, make_sgs_time,
                                     HRITGOESPrologueFileHandler, sgs_time)

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


class TestGVARFloat(unittest.TestCase):
    def test_fun(self):
        test_data = [(-1.0, b"\xbe\xf0\x00\x00"),
                     (-0.1640625, b"\xbf\xd6\x00\x00"),
                     (0.0, b"\x00\x00\x00\x00"),
                     (0.1640625, b"\x40\x2a\x00\x00"),
                     (1.0, b"\x41\x10\x00\x00"),
                     (100.1640625, b"\x42\x64\x2a\x00")]

        for expected, str_val in test_data:
            val = np.frombuffer(str_val, dtype='>i4')
            self.assertEqual(expected, make_gvar_float(val))


class TestMakeSGSTime(unittest.TestCase):
    def test_fun(self):
        # 2018-129 (may 9th), 21:33:27.999
        tcds = np.array([(32, 24, 18, 146, 19, 50, 121, 153)], dtype=sgs_time)
        expected = datetime.datetime(2018, 5, 9, 21, 33, 27, 999000)
        self.assertEqual(make_sgs_time(tcds[0]), expected)


class TestHRITGOESPrologueFileHandler(unittest.TestCase):
    """Test the HRITFileHandler."""

    @mock.patch('satpy.readers.hrit_goes.recarray2dict')
    @mock.patch('satpy.readers.hrit_goes.np.fromfile')
    @mock.patch('satpy.readers.hrit_goes.HRITFileHandler.__init__')
    def test_init(self, new_fh_init, fromfile, recarray2dict):
        """Setup the hrit file handler for testing."""
        recarray2dict.side_effect = lambda x: x
        new_fh_init.return_value.filename = 'filename'
        HRITGOESPrologueFileHandler.filename = 'filename'
        HRITGOESPrologueFileHandler.mda = {'total_header_length': 1}
        ret = {}
        the_time = np.array([(32, 24, 18, 146, 19, 50, 121, 153)], dtype=sgs_time)[0]
        for key in ['TCurr', 'TCHED', 'TCTRL', 'TLHED', 'TLTRL', 'TIPFS',
                    'TINFS', 'TISPC', 'TIECL', 'TIBBC', 'TISTR', 'TLRAN',
                    'TIIRT', 'TIVIT', 'TCLMT', 'TIONA']:
            ret[key] = the_time
        ret['SubSatLatitude'] = np.frombuffer(b"\x00\x00\x00\x00", dtype='>i4')[0]
        ret['ReferenceLatitude'] = np.frombuffer(b"\x00\x00\x00\x00", dtype='>i4')[0]
        ret['SubSatLongitude'] = np.frombuffer(b"\x42\x64\x2a\x00", dtype='>i4')[0]
        ret['ReferenceLongitude'] = np.frombuffer(b"\x42\x64\x2a\x00", dtype='>i4')[0]
        ret['ReferenceDistance'] = np.frombuffer(b"\x42\x64\x2a\x00", dtype='>i4')[0]
        fromfile.return_value = [ret]
        m = mock.mock_open()
        with mock.patch('satpy.readers.hrit_goes.open', m, create=True) as newopen:
            newopen.return_value.__enter__.return_value.seek.return_value = 1
            self.reader = HRITGOESPrologueFileHandler(
                'filename', {'platform_shortname': 'GOES15',
                             'start_time': datetime.datetime(2016, 3, 3, 0, 0),
                             'service': 'test_service'},
                {'filetype': 'info'})

        expected = {'TISTR': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
                    'TCurr': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
                    'TCLMT': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
                    'SubSatLongitude': np.array([100.1640625]),
                    'TCHED': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
                    'TLTRL': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
                    'TIPFS': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
                    'TISPC': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
                    'ReferenceLatitude': 0.0,
                    'TIIRT': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
                    'TLHED': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
                    'TIVIT': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
                    'SubSatLatitude': 0.0,
                    'TIECL': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
                    'ReferenceLongitude': np.array([100.1640625]),
                    'TCTRL': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
                    'TLRAN': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
                    'TINFS': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
                    'TIBBC': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
                    'TIONA': datetime.datetime(2018, 5, 9, 21, 33, 27, 999000),
                    'ReferenceDistance': np.array([100.1640625])}

        self.assertEqual(expected, self.reader.prologue)


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestHRITGOESPrologueFileHandler))
    mysuite.addTest(loader.loadTestsFromTestCase(TestGVARFloat))
    mysuite.addTest(loader.loadTestsFromTestCase(TestMakeSGSTime))
    return mysuite


if __name__ == '__main__':
    unittest.main()
