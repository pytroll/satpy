#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009, 2012, 2013, 2014.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# satpy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Test module for satpy.projector.
"""
import unittest

import numpy as np


from mock import MagicMock, patch
import sys
sys.modules['pyresample'] = MagicMock()

from pyresample import geometry, utils

from satpy.projector import Projector
import satpy.projector



class TestProjector(unittest.TestCase):
    """Class for testing the Projector class.
    """

    proj = None
    @patch.object(utils, 'generate_quick_linesample_arrays')
    @patch.object(satpy.projector.kd_tree, 'get_neighbour_info')
    @patch.object(satpy.projector, '_get_area_hash')
    def test_init(self, gah, gni, gqla):
        """Creation of coverage.
        """

        # in case of wrong number of arguments

        self.assertRaises(TypeError, Projector)
        self.assertRaises(TypeError, Projector, random_string(20))


        # in case of string arguments

        in_area_id = random_string(20)
        out_area_id = random_string(20)

        area_type = utils.parse_area_file.return_value.__getitem__.return_value

        gni.side_effect = [("a", "b", "c", "d")] * 10

        self.proj = Projector(in_area_id, out_area_id)
        self.assertEquals(utils.parse_area_file.call_count, 2)
        utils.parse_area_file.assert_any_call('', in_area_id)
        utils.parse_area_file.assert_any_call('', out_area_id)



        self.assertEquals(self.proj.in_area, area_type)
        self.assertEquals(self.proj.out_area, area_type)


        # in case of undefined areas

        mock = MagicMock(side_effect=Exception("raise"))
        with patch.object(utils, 'parse_area_file', mock):
            self.assertRaises(Exception,
                              Projector,
                              "raise",
                              random_string(20))
            self.assertRaises(Exception,
                              Projector,
                              random_string(20),
                              "raise")

        # in case of geometry objects as input

        with patch.object(utils, 'AreaNotFound', Exception):
            mock = MagicMock(side_effect=[utils.AreaNotFound("raise"),
                                          MagicMock()])
            with patch.object(utils, 'parse_area_file', mock):
                in_area = geometry.AreaDefinition()
                self.proj = Projector(in_area, out_area_id)
                print self.proj.in_area
                self.assertEquals(self.proj.in_area, in_area)

        in_area = geometry.SwathDefinition()
        utils.parse_area_file.return_value.__getitem__.side_effect = [AttributeError, out_area_id]
        self.proj = Projector(in_area, out_area_id)
        self.assertEquals(self.proj.in_area, in_area)

        out_area = geometry.AreaDefinition()
        utils.parse_area_file.return_value.__getitem__.side_effect = [in_area_id, AttributeError]
        self.proj = Projector(in_area_id, out_area)
        self.assertEquals(self.proj.out_area, out_area)

        # in case of lon/lat is input

        utils.parse_area_file.return_value.__getitem__.side_effect = [AttributeError, out_area_id]
        lonlats = ("great_lons", "even_greater_lats")

        self.proj = Projector("raise", out_area_id, lonlats)
        geometry.SwathDefinition.assert_called_with(lons=lonlats[0],
                                                    lats=lonlats[1])

        utils.parse_area_file.return_value.__getitem__.side_effect = None
        # in case of wrong mode

        self.assertRaises(ValueError,
                          Projector,
                          random_string(20),
                          random_string(20),
                          mode=random_string(20))

        utils.parse_area_file.return_value.__getitem__.side_effect = ["a", "b",
                                                                      "c", "d"]
        gqla.side_effect = [("ridx", "cidx")]
        # quick mode cache
        self.proj = Projector(in_area_id, out_area_id, mode="quick")
        cache = getattr(self.proj, "_cache")
        self.assertTrue(cache['row_idx'] is not None)
        self.assertTrue(cache['col_idx'] is not None)

        # nearest mode cache

        self.proj = Projector(in_area_id, out_area_id, mode="nearest")
        cache = getattr(self.proj, "_cache")
        self.assertTrue(cache['valid_index'] is not None)
        self.assertTrue(cache['valid_output_index'] is not None)
        self.assertTrue(cache['index_array'] is not None)


    @patch.object(np.ma, "array")
    @patch.object(satpy.projector.kd_tree, 'get_sample_from_neighbour_info')
    @patch.object(np, "load")
    def test_project_array(self, npload, gsfni, marray):
        """Test the project_array function.
        """
        in_area_id = random_string(20)
        out_area_id = random_string(20)
        data = np.random.standard_normal((3, 1))

        utils.parse_area_file.return_value.__getitem__.side_effect = ["a", "b", "c", "d"]
        # test quick
        self.proj = Projector(in_area_id, out_area_id, mode="quick")
        self.proj.project_array(data)
        satpy.projector.image.ImageContainer.assert_called_with(\
            data, "a", fill_value=None)
        satpy.projector.image.ImageContainer.return_value.\
            get_array_from_linesample.assert_called_with(\
            self.proj._cache["row_idx"], self.proj._cache["col_idx"])
        marray.assert_called_once_with(\
            satpy.projector.image.ImageContainer.return_value.\
            get_array_from_linesample.return_value,
            dtype=np.dtype('float64'))

        # test nearest
        in_area = MagicMock()
        out_area = MagicMock()
        utils.parse_area_file.return_value.__getitem__.side_effect = \
                        [in_area, out_area]
        self.proj = Projector(in_area_id, out_area_id, mode="nearest")
        self.proj.project_array(data)
        satpy.projector.kd_tree.get_sample_from_neighbour_info.\
             assert_called_with('nn',
                                out_area.shape,
                                data,
                                npload.return_value.__getitem__.return_value,
                                npload.return_value.__getitem__.return_value,
                                npload.return_value.__getitem__.return_value,
                                fill_value=None)


def random_string(length,
                  choices="abcdefghijklmnopqrstuvwxyz"
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """Generates a random string with elements from *set* of the specified
    *length*.
    """
    import random
    return "".join([random.choice(choices)
                    for dummy in range(length)])


def suite():
    """The test suite for test_projector.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestProjector))

    return mysuite

