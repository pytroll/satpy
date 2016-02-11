#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009, 2013, 2014.

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
"""Module for testing the imageo.image module.
"""
import random
import unittest

import numpy as np

import satpy.imageo.image as image


EPSILON = 0.0001

# Support for python <2.5
try:
    all
except NameError:
    def all(iterable):
        for element in iterable:
            if not element:
                return False
        return True

class TestEmptyImage(unittest.TestCase):
    """Class for testing the satpy.imageo.image module
    """
    def setUp(self):
        """Setup the test.
        """
        self.img = image.Image()
        self.modes = ["L", "LA", "RGB", "RGBA", "YCbCr", "YCbCrA", "P", "PA"]


    def test_shape(self):
        """Shape of an empty image.
        """
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            self.assertEqual(self.img.shape, (0, 0))
        self.img.convert(oldmode)
        
    def test_is_empty(self):
        """Test if an image is empty.
        """
        self.assertEqual(self.img.is_empty(), True)

    def test_clip(self):
        """Clip an empty image.
        """
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            self.assertEqual(self.img.channels, [])
        self.img.convert(oldmode)
        
    def test_convert(self):
        """Convert an empty image.
        """
        for mode1 in self.modes:
            for mode2 in self.modes:
                self.img.convert(mode1)
                self.assertEqual(self.img.mode, mode1)
                self.assertEqual(self.img.channels, [])
                self.img.convert(mode2)
                self.assertEqual(self.img.mode, mode2)
                self.assertEqual(self.img.channels, [])
        while True:
            randstr = random_string(random.choice(range(1, 7)))
            if randstr not in self.modes:
                break
        self.assertRaises(ValueError, self.img.convert, randstr)

    def test_stretch(self):
        """Stretch an empty image
        """
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            self.img.stretch()
            self.assertEqual(self.img.channels, [])
            self.img.stretch("linear")
            self.assertEqual(self.img.channels, [])
            self.img.stretch("histogram")
            self.assertEqual(self.img.channels, [])
            self.img.stretch("crude")
            self.assertEqual(self.img.channels, [])
            self.img.stretch((0.05, 0.05))
            self.assertEqual(self.img.channels, [])
            self.assertRaises(ValueError, self.img.stretch, (0.05, 0.05, 0.05))

            # Generate a random string
            while True:
                testmode = random_string(random.choice(range(1, 7)))
                if testmode not in self.modes:
                    break
            
            self.assertRaises(ValueError, self.img.stretch, testmode)
            self.assertRaises(TypeError, self.img.stretch, 1)
        self.img.convert(oldmode)
        
    def test_gamma(self):
        """Gamma correction on an empty image.
        """
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            # input a single value
            self.img.gamma()
            self.assertEqual(self.img.channels, [])
            self.img.gamma(0.5)
            self.assertEqual(self.img.channels, [])
            self.img.gamma(1)
            self.assertEqual(self.img.channels, [])
            self.img.gamma(1.5)
            self.assertEqual(self.img.channels, [])

            # input a tuple
            self.assertRaises(ValueError, self.img.gamma, range(10))
            self.assertRaises(ValueError, self.img.gamma, (0.2, 3.5))
            self.assertRaises(TypeError, self.img.gamma, ("blue", "white"))

            # input a negative value
            self.assertRaises(ValueError, self.img.gamma, -0.5)
            self.assertRaises(ValueError, self.img.gamma, -1)
            self.assertRaises(ValueError, self.img.gamma, -3.8)
            self.assertRaises(TypeError, self.img.gamma, "blue")
        self.img.convert(oldmode)
        
    def test_invert(self):
        """Invert an empty image.
        """
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            self.img.invert()
            self.assertEqual(self.img.channels, [])
            self.img.invert(True)
            self.assertEqual(self.img.channels, [])
            self.assertRaises(ValueError, self.img.invert, [True, False])
            self.assertRaises(ValueError, self.img.invert,
                              [True, False, True, False,
                               True, False, True, False])
        self.img.convert(oldmode)
        
    def test_pil_image(self):
        """Return an empty PIL image.
        """
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            if mode == "YCbCrA":
                self.assertRaises(ValueError, self.img.pil_image)
            elif mode == "YCbCr":
                continue
            else:
                pilimg = self.img.pil_image()
                self.assertEqual(pilimg.size, (0, 0))
        self.img.convert(oldmode)
        
    def test_putalpha(self):
        """Add an alpha channel to en empty image
        """
        # Putting alpha channel to an empty image should not do anything except
        # change the mode if necessary.
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            self.img.putalpha(np.array([]))
            self.assertEqual(self.img.channels, [])
            if mode.endswith("A"):
                self.assertEqual(self.img.mode, mode)
            else:
                self.assertEqual(self.img.mode, mode + "A")

            self.img.convert(oldmode)

            self.img.convert(mode)
            self.assertRaises(ValueError, self.img.putalpha,
                              np.random.rand(3, 2))


        self.img.convert(oldmode)

    def test_save(self):
        """Save an empty image.
        """
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            self.assertRaises(IOError, self.img.save, "test.png")

        self.img.convert(oldmode)

    def test_replace_luminance(self):
        """Replace luminance in an empty image.
        """
        oldmode = self.img.mode
        for mode in self.modes:
            self.img.convert(mode)
            self.img.replace_luminance([])
            self.assertEqual(self.img.mode, mode)
            self.assertEqual(self.img.channels, [])
            self.assertEqual(self.img.shape, (0, 0))
        self.img.convert(oldmode)

    def test_resize(self):
        """Resize an empty image.
        """
        self.assertRaises(ValueError, self.img.resize, (10, 10))
        
    def test_merge(self):
        """Merging of an empty image with another.
        """
        newimg = image.Image()
        self.assertRaises(ValueError, self.img.merge, newimg)
        newimg = image.Image(np.array([[1, 2], [3, 4]]))
        self.assertRaises(ValueError, self.img.merge, newimg)
        newimg = image.Image(np.array([[1, 2, 3, 4]]))
        self.assertRaises(ValueError, self.img.merge, newimg)


class TestImageCreation(unittest.TestCase):
    """Class for testing the satpy.imageo.image module
    """
    def setUp(self):
        """Setup the test.
        """
        self.img = {}
        self.modes = ["L", "LA", "RGB", "RGBA", "YCbCr", "YCbCrA", "P", "PA"]
        self.modes_len = [1, 2, 3, 4, 3, 4, 1, 2]

    def test_creation(self):
        """Creation of an image.
        """

        self.assertRaises(TypeError, image.Image,
                          channels = random.randint(1,1000))
        self.assertRaises(TypeError, image.Image,
                          channels = random.random())
        self.assertRaises(TypeError, image.Image,
                          channels = random_string(random.randint(1,10)))
        
        chs = [np.random.rand(random.randint(1, 10), random.randint(1, 10)),
               np.random.rand(random.randint(1, 10), random.randint(1, 10)),
               np.random.rand(random.randint(1, 10), random.randint(1, 10)),
               np.random.rand(random.randint(1, 10), random.randint(1, 10))]

        self.assertRaises(ValueError, image.Image, channels = chs)

        one_channel = np.random.rand(random.randint(1, 10),
                                     random.randint(1, 10))

        i = 0

        for mode in self.modes:
            # Empty image, no channels
            self.img[mode] = image.Image(mode = mode)
            self.assertEqual(self.img[mode].channels, [])

            # Empty image, no channels, fill value
            
            self.img[mode] = image.Image(mode = mode, fill_value = 0)
            self.assertEqual(self.img[mode].channels, [])



            # Empty image, no channels, fill value, wrong color_range

            self.assertRaises(ValueError,
                              image.Image,
                              mode = mode,
                              fill_value = 0,
                              color_range = ((0, (1, 2))))

            self.assertRaises(ValueError,
                              image.Image,
                              mode = mode,
                              fill_value = 0,
                              color_range = ((0, 0), (1, 2), (0, 0),
                                             (1, 2), (0, 0), (1, 2)))


            # Regular image, too many channels

            self.assertRaises(ValueError, image.Image,
                              channels = ([one_channel] *
                                          (self.modes_len[i] + 1)),
                              mode = mode)

            # Regular image, not enough channels

            self.assertRaises(ValueError, image.Image,
                              channels = ([one_channel] *
                                          (self.modes_len[i] - 1)),
                              mode = mode)

            # Regular image, channels
            
            self.img[mode] = image.Image(channels = ([one_channel] *
                                                     (self.modes_len[i])),
                                         mode = mode)

            for nb_chan in range(self.modes_len[i]):
                self.assert_(np.all(self.img[mode].channels[nb_chan] ==
                                    one_channel))
                self.assert_(isinstance(self.img[mode].channels[nb_chan],
                                        np.ma.core.MaskedArray))
            
            i = i + 1

        
class TestRegularImage(unittest.TestCase):
    """Class for testing the satpy.imageo.image module
    """
    def setUp(self):
        """Setup the test.
        """
        import os
        import tempfile
        one_channel = np.random.rand(random.randint(1, 10),
                                     random.randint(1, 10))
        self.rand_img = image.Image(channels = [one_channel] * 3,
                                    mode = "RGB")
        self.rand_img2 = image.Image(channels = [one_channel] * 3,
                                    mode = "RGB",
                                    fill_value = (0, 0, 0))

        two_channel = np.array([[0, 0.5, 0.5], [0.5, 0.25, 0.25]])
        self.img = image.Image(channels = [two_channel] * 3,
                               mode = "RGB")

        self.flat_channel = [[1, 1, 1], [1, 1, 1]]
        self.flat_img = image.Image(channels = [self.flat_channel],
                                    mode = "L",
                                    fill_value = 0)


        
        self.modes = ["L", "LA", "RGB", "RGBA", "YCbCr", "YCbCrA", "P", "PA"]
        self.modes_len = [1, 2, 3, 4, 3, 4, 1, 2]

        # create an unusable directory for permission error checking

        self.tempdir = tempfile.mkdtemp()
        os.chmod(self.tempdir, 0000)
        

    def test_shape(self):
        """Shape of an image.
        """
        oldmode = self.img.mode
        for mode in self.modes:
            if mode == "P" or mode == "PA":
                continue
            self.img.convert(mode)
            self.assertEqual(self.img.shape, (2, 3))
        self.img.convert(oldmode)
        
    def test_is_empty(self):
        """Test if an image is empty.
        """
        self.assertEqual(self.img.is_empty(), False)

    def test_clip(self):
        """Clip an image.
        """
        oldmode = self.img.mode
        for mode in self.modes:
            if mode == "P" or mode == "PA":
                continue
            self.img.convert(mode)
            for chn in self.img.channels:
                self.assert_(chn.max() <= 1.0)
                self.assert_(chn.max() >= 0.0)
        self.img.convert(oldmode)
        
    def test_convert(self):
        """Convert an image.
        """
        i = 0
        for mode1 in self.modes:
            j = 0
            for mode2 in self.modes:
                self.img.convert(mode1)
                self.assertEqual(self.img.mode, mode1)
                self.assertEqual(len(self.img.channels),
                                 self.modes_len[i])
                self.img.convert(mode2)
                self.assertEqual(self.img.mode, mode2)
                self.assertEqual(len(self.img.channels),
                                 self.modes_len[j])

                self.rand_img2.convert(mode1)
                self.assertEqual(self.rand_img2.mode, mode1)
                self.assertEqual(len(self.rand_img2.channels),
                                 self.modes_len[i])
                if mode1 not in ["P", "PA"]:
                    self.assertEqual(len(self.rand_img2.fill_value),
                                     self.modes_len[i])
                self.rand_img2.convert(mode2)
                self.assertEqual(self.rand_img2.mode, mode2)
                self.assertEqual(len(self.rand_img2.channels),
                                 self.modes_len[j])
                if mode2 not in ["P", "PA"]:
                    self.assertEqual(len(self.rand_img2.fill_value),
                                     self.modes_len[j])
                j = j + 1
            i = i + 1
        while True:
            randstr = random_string(random.choice(range(1, 7)))
            if randstr not in self.modes:
                break
        self.assertRaises(ValueError, self.img.convert, randstr)

    def test_stretch(self):
        """Stretch an image.
        """
        oldmode = self.img.mode

        for mode in "L":
            self.img.convert(mode)
            old_channels = []
            for chn in self.img.channels:
                old_channels.append(chn)

            linear = np.array([[ 0., 1.00048852, 1.00048852],
                               [ 1.00048852, 0.50024426, 0.50024426]])
            crude = np.array([[0, 1, 1], [1, 0.5, 0.5]])
            histo = np.array([[0.0, 0.99951171875, 0.99951171875], 
                              [0.99951171875, 0.39990234375, 0.39990234375]])
            self.img.stretch()
            self.assert_(all(np.all(self.img.channels[i] == old_channels[i])
                         for i in range(len(self.img.channels))))
            self.img.stretch("linear")
            self.assert_(np.all((self.img.channels[0] - linear) < EPSILON))
            self.img.stretch("crude")
            self.assert_(np.all((self.img.channels[0] - crude) < EPSILON))
            self.img.stretch("histogram")
            self.assert_(np.all(np.abs(self.img.channels[0] - histo) < EPSILON))
            self.img.stretch((0.05, 0.05))
            self.assert_(np.all((self.img.channels[0] - linear) < EPSILON))
            self.assertRaises(ValueError, self.img.stretch, (0.05, 0.05, 0.05))

            # Generate a random string
            while True:
                testmode = random_string(random.choice(range(1, 7)))
                if testmode not in self.modes:
                    break
            
            self.assertRaises(ValueError, self.img.stretch, testmode)
            self.assertRaises(TypeError, self.img.stretch, 1)

            self.img.channels = old_channels

        self.img.convert(oldmode)

    def test_gamma(self):
        """Gamma correction on an image.
        """
        oldmode = self.img.mode
        for mode in self.modes:
            if mode == "P" or mode == "PA":
                continue
            self.img.convert(mode)

            old_channels = []
            for chn in self.img.channels:
                old_channels.append(chn)

            # input a single value
            self.img.gamma()
            for i in range(len(self.img.channels)):
                self.assert_(np.all(self.img.channels[i] == old_channels[i]))
            self.img.gamma(0.5)
            for i in range(len(self.img.channels)):
                self.assert_(np.all(self.img.channels[i] -
                                    old_channels[i] ** 2 < EPSILON))
            self.img.gamma(1)
            for i in range(len(self.img.channels)):
                self.assert_(np.all(self.img.channels[i] -
                                    old_channels[i] ** 2 < EPSILON))

            # self.img.gamma(2)
            # for i in range(len(self.img.channels)):
            #     print self.img.channels[i]
            #     print old_channels[i]
            #     self.assert_(np.all(np.abs(self.img.channels[i] -
            #                                old_channels[i]) < EPSILON))


            # input a tuple
            self.assertRaises(ValueError, self.img.gamma, range(10))
            self.assertRaises(ValueError, self.img.gamma, (0.2, 3., 8., 1., 9.))
            self.assertRaises(TypeError, self.img.gamma, ("blue", "white"))

            # input a negative value
            self.assertRaises(ValueError, self.img.gamma, -0.5)
            self.assertRaises(ValueError, self.img.gamma, -1)
            self.assertRaises(ValueError, self.img.gamma, -3.8)
            self.assertRaises(TypeError, self.img.gamma, "blue")
        self.img.convert(oldmode)
        
    def test_invert(self):
        """Invert an image.
        """
        oldmode = self.img.mode
        for mode in self.modes:
            if mode == "P" or mode == "PA":
                continue
            self.img.convert(mode)
            old_channels = []
            for chn in self.img.channels:
                old_channels.append(chn)
            self.img.invert()
            for i in range(len(self.img.channels)):
                self.assert_(np.all(self.img.channels[i] ==
                                    1 - old_channels[i]))
            self.img.invert(True)
            for i in range(len(self.img.channels)):
                self.assert_(np.all(self.img.channels[i] -
                                    old_channels[i] < EPSILON))
            self.assertRaises(ValueError, self.img.invert,
                              [True, False, True, False,
                               True, False, True, False])
        self.img.convert(oldmode)
        
    def test_pil_image(self):
        """Return an PIL image.
        """

        # FIXME: Should test on palette images
        
        oldmode = self.img.mode
        for mode in self.modes:
            if (mode == "YCbCr" or
                mode == "YCbCrA" or
                mode == "P" or
                mode == "PA"):
                continue
            self.img.convert(mode)
            if mode == "YCbCrA":
                self.assertRaises(ValueError, self.img.pil_image)
            else:
                pilimg = self.img.pil_image()
                self.assertEqual(pilimg.size, (3, 2))
        self.img.convert(oldmode)
        
    def test_putalpha(self):
        """Add an alpha channel.
        """
        # Putting alpha channel to an image should not do anything except
        # change the mode if necessary.
        oldmode = self.img.mode
        alpha = np.array(np.random.rand(2, 3))
        for mode in self.modes:
            if mode == "P" or mode == "PA":
                continue
            self.img.convert(mode)
            self.img.putalpha(alpha)
            self.assert_(np.all(self.img.channels[-1] == alpha))
            if mode.endswith("A"):
                self.assertEqual(self.img.mode, mode)
            else:
                self.assertEqual(self.img.mode, mode + "A")

            self.img.convert(oldmode)

            self.img.convert(mode)
            self.assertRaises(ValueError,
                              self.img.putalpha,
                              np.random.rand(4, 5))

        self.img.convert(oldmode)

    def test_save(self):
        """Save an image.
        """
        import os, os.path

        oldmode = self.img.mode
        for mode in self.modes:
            if (mode == "YCbCr" or
                mode == "YCbCrA" or
                mode == "P" or
                mode == "PA"):
                continue
            self.img.convert(mode)
            self.img.save("test.png")
            self.assert_(os.path.exists("test.png"))
            os.remove("test.png")

            # permissions
            self.assertRaises(IOError,
                              self.img.save,
                              os.path.join(self.tempdir, "test.png"))

        self.img.convert(oldmode)

    def test_replace_luminance(self):
        """Replace luminance in an image.
        """
        oldmode = self.img.mode
        for mode in self.modes:
            if (mode == "P" or
                mode == "PA"):
                continue
            self.img.convert(mode)
            luma = np.ma.array([[0, 0.5, 0.5],
                                [0.5, 0.25, 0.25]])
            self.img.replace_luminance(luma)
            self.assertEqual(self.img.mode, mode)
            if(self.img.mode.endswith("A")):
                chans = self.img.channels[:-1]
            else:
                chans = self.img.channels
            for chn in chans:
                self.assert_(np.all(chn - luma < EPSILON))
        self.img.convert(oldmode)

    def test_resize(self):
        """Resize an image.
        """
        self.img.resize((6, 6))
        res = np.array([[0, 0, 0.5, 0.5, 0.5, 0.5],
                        [0, 0, 0.5, 0.5, 0.5, 0.5],
                        [0, 0, 0.5, 0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.25, 0.25, 0.25, 0.25],
                        [0.5, 0.5, 0.25, 0.25, 0.25, 0.25],
                        [0.5, 0.5, 0.25, 0.25, 0.25, 0.25]])
        self.assert_(np.all(res == self.img.channels[0]))
        self.img.resize((2, 3))
        res = np.array([[0, 0.5, 0.5],
                        [0.5, 0.25, 0.25]])
        self.assert_(np.all(res == self.img.channels[0]))

        
    def test_merge(self):
        """Merging of an image with another.
        """
        newimg = image.Image()
        self.assertRaises(ValueError, self.img.merge, newimg)
        newimg = image.Image(np.array([[1, 2], [3, 4]]))
        self.assertRaises(ValueError, self.img.merge, newimg)
        newimg = image.Image(np.array([[1, 2, 3, 4]]))
        self.assertRaises(ValueError, self.img.merge, newimg)
        newimg = image.Image(np.ma.array([[1, 2, 3], [4, 5, 6]],
                                         mask = [[1, 0, 0], [1, 1, 0]]),
                             mode = "L")
        self.img.convert("L")
        newimg.merge(self.img)
        self.assert_(np.all(np.abs(newimg.channels[0] -
                                   np.array([[0, 2, 3], [0.5, 0.25, 6]])) <
                            EPSILON))


    def tearDown(self):
        """Clean up the mess.
        """
        import os
        os.rmdir(self.tempdir)


class TestFlatImage(unittest.TestCase):
    """Test a flat image, ie an image where min == max.
    """
    def setUp(self):
        channel = np.ma.array([[0, 0.5, 0.5], [0.5, 0.25, 0.25]],
                              mask = [[1, 1, 1], [1, 1, 0]])
        self.img = image.Image(channels = [channel] * 3,
                               mode = "RGB")
        self.modes = ["L", "LA", "RGB", "RGBA", "YCbCr", "YCbCrA", "P", "PA"]

    def test_stretch(self):
        """Stretch a flat image.
        """
        self.img.stretch()
        self.assert_(self.img.channels[0].shape == (2, 3) and
                     np.ma.count_masked(self.img.channels[0]) == 5)
        self.img.stretch("crude")
        self.assert_(self.img.channels[0].shape == (2, 3) and
                     np.ma.count_masked(self.img.channels[0]) == 5)
        self.img.crude_stretch(1, 2)
        self.assert_(self.img.channels[0].shape == (2, 3) and
                     np.ma.count_masked(self.img.channels[0]) == 5)
        self.img.stretch("linear")
        self.assert_(self.img.channels[0].shape == (2, 3) and
                     np.ma.count_masked(self.img.channels[0]) == 5)
        self.img.stretch("histogram")
        self.assert_(self.img.channels[0].shape == (2, 3) and
                     np.ma.count_masked(self.img.channels[0]) == 5)

class TestNoDataImage(unittest.TestCase):
    """Test an image filled with no data.
    """
    def setUp(self):
        channel = np.ma.array([[0, 0.5, 0.5], [0.5, 0.25, 0.25]],
                              mask = [[1, 1, 1], [1, 1, 1]])
        self.img = image.Image(channels = [channel] * 3,
                               mode = "RGB")
        self.modes = ["L", "LA", "RGB", "RGBA", "YCbCr", "YCbCrA", "P", "PA"]

    def test_stretch(self):
        """Stretch a no data image.
        """
        self.img.stretch()
        self.assert_(self.img.channels[0].shape == (2, 3))
        self.img.stretch("crude")
        self.assert_(self.img.channels[0].shape == (2, 3))
        self.img.crude_stretch(1, 2)
        self.assert_(self.img.channels[0].shape == (2, 3))
        self.img.stretch("linear")
        self.assert_(self.img.channels[0].shape == (2, 3))
        self.img.stretch("histogram")
        self.assert_(self.img.channels[0].shape == (2, 3))

def random_string(length,
                  choices="abcdefghijklmnopqrstuvwxyz"
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """Generates a random string with elements from *set* of the specified
    *length*.
    """
    return "".join([random.choice(choices)
                    for dummy in range(length)])


def suite():
    """The test suite for test_image.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestImageCreation))
    mysuite.addTest(loader.loadTestsFromTestCase(TestRegularImage))
    mysuite.addTest(loader.loadTestsFromTestCase(TestFlatImage))
    mysuite.addTest(loader.loadTestsFromTestCase(TestEmptyImage))
    mysuite.addTest(loader.loadTestsFromTestCase(TestNoDataImage))
    
    return mysuite

