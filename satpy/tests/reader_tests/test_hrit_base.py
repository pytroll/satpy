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
"""The HRIT base reader tests package.
"""

import sys
from datetime import datetime
import os
from tempfile import gettempdir, NamedTemporaryFile

import numpy as np

from satpy.readers.hrit_base import HRITFileHandler, get_xritdecompress_cmd, get_xritdecompress_outfile, decompress

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


class TestHRITDecompress(unittest.TestCase):
    """Test the on-the-fly decompression."""

    def test_xrit_cmd(self):
        old_env = os.environ.get('XRIT_DECOMPRESS_PATH', None)

        os.environ['XRIT_DECOMPRESS_PATH'] = '/path/to/my/bin'
        self.assertRaises(IOError, get_xritdecompress_cmd)

        os.environ['XRIT_DECOMPRESS_PATH'] = gettempdir()
        self.assertRaises(IOError, get_xritdecompress_cmd)

        with NamedTemporaryFile() as fd:
            os.environ['XRIT_DECOMPRESS_PATH'] = fd.name
            fname = fd.name
            res = get_xritdecompress_cmd()

        if old_env is not None:
            os.environ['XRIT_DECOMPRESS_PATH'] = old_env
        else:
            os.environ.pop('XRIT_DECOMPRESS_PATH')

        self.assertEqual(fname, res)

    def test_xrit_outfile(self):
        stdout = [b"Decompressed file: bla.__\n"]
        outfile = get_xritdecompress_outfile(stdout)
        self.assertEqual(outfile, b'bla.__')

    @mock.patch('satpy.readers.hrit_base.Popen')
    def test_decompress(self, popen):

        popen.return_value.returncode = 0
        popen.return_value.communicate.return_value = [b"Decompressed file: bla.__\n"]

        old_env = os.environ.get('XRIT_DECOMPRESS_PATH', None)

        with NamedTemporaryFile() as fd:
            os.environ['XRIT_DECOMPRESS_PATH'] = fd.name
            res = decompress('bla.C_')

        if old_env is not None:
            os.environ['XRIT_DECOMPRESS_PATH'] = old_env
        else:
            os.environ.pop('XRIT_DECOMPRESS_PATH')

        self.assertEqual(res, os.path.join('.', 'bla.__'))


class TestHRITFileHandler(unittest.TestCase):
    """Test the HRITFileHandler."""

    @mock.patch('satpy.readers.hrit_base.np.fromfile')
    def setUp(self, fromfile):
        """Setup the hrit file handler for testing."""
        m = mock.mock_open()
        fromfile.return_value = np.array([(1, 2)], dtype=[('total_header_length', int),
                                                          ('hdr_id', int)])

        with mock.patch('satpy.readers.hrit_base.open', m, create=True) as newopen:
            newopen.return_value.__enter__.return_value.tell.return_value = 1
            self.reader = HRITFileHandler('filename',
                                          {'platform_shortname': 'MSG3',
                                           'start_time': datetime(2016, 3, 3, 0, 0)},
                                          {'filetype': 'info'},
                                          [mock.MagicMock(), mock.MagicMock(),
                                           mock.MagicMock()])
            ncols = 3712
            nlines = 464
            nbits = 10
            self.reader.mda['number_of_bits_per_pixel'] = nbits
            self.reader.mda['number_of_lines'] = nlines
            self.reader.mda['number_of_columns'] = ncols
            self.reader.mda['data_field_length'] = nlines * ncols * nbits
            self.reader.mda['cfac'] = 5
            self.reader.mda['lfac'] = 5
            self.reader.mda['coff'] = 10
            self.reader.mda['loff'] = 10
            self.reader.mda['projection_parameters'] = {}
            self.reader.mda['projection_parameters']['a'] = 6378169.0
            self.reader.mda['projection_parameters']['b'] = 6356583.8
            self.reader.mda['projection_parameters']['h'] = 35785831.0
            self.reader.mda['projection_parameters']['SSP_longitude'] = 44

    def test_get_xy_from_linecol(self):
        """Test get_xy_from_linecol."""
        x__, y__ = self.reader.get_xy_from_linecol(0, 0, (10, 10), (5, 5))
        self.assertEqual(-131072, x__)
        self.assertEqual(-131072, y__)
        x__, y__ = self.reader.get_xy_from_linecol(10, 10, (10, 10), (5, 5))
        self.assertEqual(0, x__)
        self.assertEqual(0, y__)
        x__, y__ = self.reader.get_xy_from_linecol(20, 20, (10, 10), (5, 5))
        self.assertEqual(131072, x__)
        self.assertEqual(131072, y__)

    def test_get_area_extent(self):
        res = self.reader.get_area_extent((20, 20), (10, 10), (5, 5), 33)
        exp = (-71717.44995740513, -71717.44995740513,
               79266.655216079365, 79266.655216079365)
        self.assertTupleEqual(res, exp)

    def test_get_area_def(self):
        area = self.reader.get_area_def('VIS06')
        self.assertEqual(area.proj_dict, {'a': 6378169.0,
                                          'b': 6356583.8,
                                          'h': 35785831.0,
                                          'lon_0': 44.0,
                                          'proj': 'geos',
                                          'units': 'm'})
        self.assertEqual(area.area_extent,
                         (-77771774058.38356, -77771774058.38356,
                          30310525626438.438, 3720765401003.719))

    @mock.patch('satpy.readers.hrit_base.np.memmap')
    def test_read_band(self, memmap):
        nbits = self.reader.mda['number_of_bits_per_pixel']
        memmap.return_value = np.random.randint(0, 256,
                                                size=int((464 * 3712 * nbits) / 8),
                                                dtype=np.uint8)
        res = self.reader.read_band('VIS006', None)
        self.assertEqual(res.compute().shape, (464, 3712))


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestHRITFileHandler))
    mysuite.addTest(loader.loadTestsFromTestCase(TestHRITDecompress))

    return mysuite


if __name__ == '__main__':
    unittest.main()
