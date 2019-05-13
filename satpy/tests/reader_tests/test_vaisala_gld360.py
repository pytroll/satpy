#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017-2018 PyTroll Community

# Author(s):

#   Adam.Dybbroe <adam.dybbroe@smhi.se>
#   Sauli Joro <sauli.joro@eumetsat.int>

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

"""Unittesting the Vaisala GLD360 reader.
"""

import sys
from io import StringIO

import numpy as np

from satpy.readers.vaisala_gld360 import (
    VaisalaGLD360TextFileHandler
)
from satpy.dataset import DatasetID

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


EXPECTED_POWER = np.array([12.3,  13.2, -31.])


class TestVaisalaGLD360TextFileHandler(unittest.TestCase):

    """Test the VaisalaGLD360TextFileHandler."""

    def test_vaisala_gld360(self):

        expected = EXPECTED_POWER

        filename = StringIO(
            u'2017-06-20 00:00:00.007178  30.5342  -90.1152    12.3 kA\n'
            '2017-06-20 00:00:00.020162  -0.5727  104.0688    13.2 kA\n'
            '2017-06-20 00:00:00.023183  12.1529  -10.8756   -31.0 kA'
            )
        filename_info = {}
        filetype_info = {}

        self.handler = VaisalaGLD360TextFileHandler(
            filename, filename_info, filetype_info
        )

        filename.close()
        dataset_id = DatasetID('power')
        dataset_info = {}
        result = self.handler.get_dataset(dataset_id, dataset_info).values

        np.testing.assert_allclose(result, expected, rtol=1e-05)


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestVaisalaGLD360TextFileHandler))
    return mysuite


if __name__ == "__main__":
    # So you can run tests from this module individually.
    unittest.main()
