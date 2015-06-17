#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015

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

"""test projectable objects.
"""

import unittest
from mpop import projectable
import numpy as np
from mock import MagicMock, patch

class TestDataset(unittest.TestCase):
    """
    Test the dataset class
    """

    def test_copy(self):
        """
        Test copying a dataset
        """
        ds = projectable.Dataset(np.arange(8), foo="bar")
        ds_copy = ds.copy(False)
        self.assert_(ds_copy.data is ds.data)
        self.assertDictEqual(ds.info, ds_copy.info)
        ds_copy = ds.copy(True)
        self.assert_(ds_copy.data is not ds.data
                     and all(ds.data == ds_copy.data))
        self.assertDictEqual(ds.info, ds_copy.info)
        ds_copy = ds.copy()
        self.assert_(ds_copy.data is not ds.data
                     and all(ds.data == ds_copy.data))
        self.assertDictEqual(ds.info, ds_copy.info)

    def test_arithmetics(self):
        """
        Test the arithmetic functions
        """
        ds = projectable.Dataset(np.arange(1, 25), foo="bar")
        ref = np.arange(1, 25)
        ds2 = ds + 1
        self.assert_(all((ds + 1).data == ref + 1))
        self.assert_(all((ds * 2).data == ref * 2))
        self.assert_(all((2 * ds).data == ref * 2))
        self.assert_(all((1 + ds).data == ref + 1))
        self.assert_(all((ds - 1).data == ref - 1))
        self.assert_(all((1 - ds).data == 1 - ref))
        self.assert_(all((ds / 2).data == ref / 2))
        self.assert_(all((2 / ds).data == 2 / ref))
        self.assert_(all((-ds).data == -ref))
        self.assert_(all((abs(ds)).data == abs(ref)))

class TestProjectable(unittest.TestCase):
    """
    Test the projectable class
    """
    def test_init(self):
        """
        Test initialization
        """
        self.assert_('uid' in projectable.Projectable().info)

    def test_isloaded(self):
        """
        Test isloaded method
        """
        self.assertFalse(projectable.Projectable().is_loaded())
        self.assertTrue(projectable.Projectable(data=1).is_loaded())

    def test_to_image(self):
        """
        Conversion to image
        """
        p = projectable.Projectable(np.arange(25))
        self.assertRaises(ValueError, p.to_image)
        p = projectable.Projectable(np.arange(25).reshape((5,5)))
        #self.assertRaises(ValueError, p.to_image)
        p = projectable.Projectable(np.arange(75).reshape(3, 5, 5))
        #self.assertRaises(ValueError, p.to_image)



if __name__ == "__main__":
    unittest.main()