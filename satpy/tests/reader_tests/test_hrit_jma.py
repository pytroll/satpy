# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 PyTroll developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""The hrit msg reader tests package.
"""

import sys
import numpy as np
import dask.array as da
from xarray import DataArray

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


class TestHRITJMAFileHandler(unittest.TestCase):
    """Test the HRITJMAFileHandler."""

    @mock.patch('satpy.readers.hrit_jma.HRITFileHandler.__init__')
    def setUp(self, new_fh_init):
        """Setup the hrit file handler for testing."""
        from satpy.readers.hrit_jma import HRITJMAFileHandler
        mda = {
            'projection_parameters': {
                'a': 6378169.00,
                'b': 6356583.80,
                'h': 35785831.00,
            },
            'image_data_function': b'$HALFTONE:=16\r_NAME:=VISIBLE\r_UNIT:=ALBEDO(%)\r'
                                   b'0:=-0.10\r1023:=100.00\r65535:=100.00\r',
            'image_segm_seq_no': 0,
            'total_no_image_segm': 1,
            'projection_name': b'GEOS(140.70)                    ',
            'cfac': 40932549,
            'lfac': 40932549,
            'coff': 5500,
            'loff': 5500,
            'number_of_columns': 11000,
            'number_of_lines': 11000,
        }
        HRITJMAFileHandler.filename = 'filename'
        HRITJMAFileHandler.mda = mda
        self.reader = HRITJMAFileHandler('filename', {}, {})

    def test_init(self):
        """Test creating the file handler."""
        mda = {
            'image_segm_seq_no': 0,
            'planned_end_segment_number': 1,
            'planned_start_segment_number': 1,
            'segment_sequence_number': 0,
            'total_no_image_segm': 1,
            'unit': 'ALBEDO(%)',
            'projection_name': b'GEOS(140.70)                    ',
            'projection_parameters': {
                'a': 6378169.00,
                'b': 6356583.80,
                'h': 35785831.00,
                'SSP_longitude': 140.7
            },
            'cfac': 40932549,
            'lfac': 40932549,
            'coff': 5500,
            'loff': 5500,
            'number_of_columns': 11000,
            'number_of_lines': 11000,
            'image_data_function': b'$HALFTONE:=16\r_NAME:=VISIBLE\r_UNIT:=ALBEDO(%)\r'
                                   b'0:=-0.10\r1023:=100.00\r65535:=100.00\r',
        }
        self.assertEqual(self.reader.mda, mda)

    @mock.patch('satpy.readers.hrit_jma.HRITFileHandler.get_dataset')
    def test_get_dataset(self, base_get_dataset):
        """Test getting a reflectance DataArray."""
        key = mock.MagicMock()
        key.calibration = 'reflectance'
        base_get_dataset.return_value = DataArray(da.arange(25, chunks=5).reshape(5, 5))
        res = self.reader.get_dataset(key, {'units': '%'})
        expected = np.array([
            [np.nan, -2.15053763e-03,  9.56989247e-02, 1.93548387e-01,  2.91397849e-01],
            [3.89247312e-01,  4.87096774e-01,  5.84946237e-01, 6.82795699e-01,  7.80645161e-01],
            [8.78494624e-01,  9.76344086e-01,  1.07419355e+00, 1.17204301e+00,  1.26989247e+00],
            [1.36774194e+00,  1.46559140e+00,  1.56344086e+00, 1.66129032e+00,  1.75913978e+00],
            [1.85698925e+00,  1.95483871e+00,  2.05268817e+00, 2.15053763e+00,  2.24838710e+00]])
        np.testing.assert_allclose(res.values, expected)
        self.assertEqual(res.attrs['units'], '%')
        self.assertEqual(res.attrs['satellite_longitude'], 140.7)

    def test_get_area_def(self):
        """Test getting an AreaDefinition."""
        from satpy import DatasetID
        area_def = self.reader.get_area_def(DatasetID(name='B03', calibration='reflectance'))
        self.assertTupleEqual(area_def.area_extent,
                              (-5499495.117842725, -16499485.352640428, 5500495.116954979, -5499495.117842725))


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestHRITJMAFileHandler))
    return mysuite


if __name__ == '__main__':
    unittest.main()
