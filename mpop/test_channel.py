#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module for testing the mpop.channel module.
"""

import unittest
from mpop.channel import GenericChannel, Channel
import string
import numpy as np

# epsilon
E = 0.0001

class TestGenericChannel(unittest.TestCase):
    """Class for testing the GenericChannel class.
    """
    chan = None
    chan2 = None
    
    def test_init(self):
        """Creation of a generic channel.
        """
        self.chan = GenericChannel(name="newchan")
        self.assertEqual(self.chan.name, "newchan")
        
        numb = int(np.random.uniform(100000))
        self.assertRaises(TypeError, GenericChannel, name=numb)
        
        self.chan = GenericChannel()
        self.assertTrue(self.chan.name is None)
        
    def test_cmp(self):
        """Comparison of generic channels.
        """
        
        self.chan = GenericChannel(name = "newchan")
        self.chan2 = GenericChannel(name = "mychan")

        self.assertTrue(self.chan > self.chan2)

        self.chan = GenericChannel(name = "newchan")
        self.chan2 = "mychan"
        
        self.assertTrue(self.chan > self.chan2)

        self.chan = GenericChannel(name = "newchan")
        self.chan2 = GenericChannel(name = "newchan")

        self.assert_(self.chan == self.chan2)

        self.chan = GenericChannel()
        self.chan2 = GenericChannel(name = "newchan")

        self.assert_(self.chan < self.chan2)

        self.chan = GenericChannel(name = "newchan")
        self.chan2 = GenericChannel(name = "_mychan")

        self.assert_(self.chan < self.chan2)

        self.chan = GenericChannel(name = "_newchan")
        self.chan2 = GenericChannel(name = "mychan")

        self.assert_(self.chan > self.chan2)


class TestChannel(unittest.TestCase):
    """Class for testing the Channel class.
    """
    chan = None
    chan2 = None
    
    def test_init(self):
        """Creation of a channel.
        """
        self.assertRaises(ValueError, Channel)

        # Name
        self.chan = Channel(name = "newchan")
        self.assertEqual(self.chan.name, "newchan")
        self.assertEqual(self.chan.wavelength_range,
                         (-np.inf, -np.inf, -np.inf))
        self.assertEqual(self.chan.resolution, 0)
        self.assert_(self.chan.data is None)

        numb = int(np.random.uniform(100000))
        self.assertRaises(TypeError, Channel, name = numb)
        numb = np.random.uniform() * 100000
        self.assertRaises(TypeError, Channel, name = numb)

        # Resolution
        numb = int(np.random.uniform(100000))
        self.assertRaises(ValueError, Channel, resolution = numb)

        numb = int(np.random.uniform(100000))
        self.chan = Channel(name = "newchan", resolution = numb)
        self.assertEqual(self.chan.name, "newchan")
        self.assertEqual(self.chan.wavelength_range,
                         (-np.inf, -np.inf, -np.inf))
        self.assertEqual(self.chan.resolution, numb)
        self.assert_(self.chan.data is None)

        self.assertRaises(TypeError, Channel,
                          name = "newchan",
                          resolution = "a")
        
        # Wavelength

        numbs = [np.random.uniform(100),
                 np.random.uniform(100),
                 np.random.uniform(100)]
        numbs.sort()

        self.chan = Channel(wavelength_range = numbs)
        self.assertEqual(self.chan.name, None)
        self.assertEqual(self.chan.wavelength_range, numbs)
        self.assertEqual(self.chan.resolution, 0)
        self.assert_(self.chan.data is None)

        self.assertRaises(TypeError, Channel,
                          wavelength_range = numbs[0:1])

        numbs.sort(reverse = True)
        self.assertRaises(ValueError, Channel,
                          wavelength_range = numbs)

        numbs = [int(np.random.uniform(100)),
                 int(np.random.uniform(100)),
                 int(np.random.uniform(100))]
        numbs.sort()        

        self.assertRaises(TypeError, Channel,
                          wavelength_range = numbs)

        self.assertRaises(TypeError, Channel,
                          wavelength_range = random_string(4))

        numb = np.random.uniform(100000)
        self.assertRaises(TypeError, Channel,
                          wavelength_range = numb)
        
        numb = int(np.random.uniform(100000))
        self.assertRaises(TypeError, Channel,
                          wavelength_range = numb)


        # Data

        data = np.random.rand(3, 3)
        
        self.assertRaises(ValueError, Channel, data = data)

        self.chan = Channel(name = "newchan", data = data)
        self.assertEqual(self.chan.name, "newchan")
        self.assertEqual(self.chan.wavelength_range,
                         (-np.inf, -np.inf, -np.inf))
        self.assertEqual(self.chan.resolution, 0)
        self.assert_(np.all(self.chan.data == data))

        mask = np.array(np.random.rand(3, 3) * 2, dtype = int)
        data = np.ma.array(data, mask = mask)
        
        self.chan = Channel(name = "newchan", data = data)
        self.assertEqual(self.chan.name, "newchan")
        self.assertEqual(self.chan.wavelength_range,
                         (-np.inf, -np.inf, -np.inf))
        self.assertEqual(self.chan.resolution, 0)
        self.assert_(np.all(self.chan.data == data))

        self.assertRaises(TypeError,
                          Channel,
                          name = "newchan",
                          data = random_string(4))

        numb = np.random.uniform(100000)
        self.assertRaises(TypeError,
                          Channel,
                          name = "newchan",
                          data = numb)
        
        numb = int(np.random.uniform(100000))
        self.assertRaises(TypeError,
                          Channel,
                          name = "newchan",
                          data = numb)
        
        numbs = [np.random.uniform(100),
                 np.random.uniform(100),
                 np.random.uniform(100)]
        self.assertRaises(TypeError,
                          Channel,
                          name = "newchan",
                          data = numbs)

    def test_cmp(self):
        """Comparison of channels.
        """
        
        self.chan = Channel(name = "newchan")
        self.chan2 = Channel(name = "mychan")

        self.assertTrue(self.chan > self.chan2)

        self.chan = Channel(name = "newchan")
        self.chan2 = "mychan"
        
        self.assertTrue(self.chan > self.chan2)

        self.chan = Channel(name = "newchan")
        self.chan2 = Channel(name = "newchan")

        self.assert_(self.chan == self.chan2)

        self.chan = Channel(wavelength_range=(1., 2., 3.))
        self.chan2 = Channel(name = "newchan")

        self.assert_(self.chan < self.chan2)

        self.chan = Channel(name = "newchan")
        self.chan2 = Channel(name = "_mychan")

        self.assert_(self.chan < self.chan2)

        self.chan = Channel(name = "_newchan")
        self.chan2 = Channel(name = "mychan")

        self.assert_(self.chan > self.chan2)

        self.chan = Channel(name = random_string(4),
                            wavelength_range = (1., 2., 3.))
        self.chan2 = Channel(name = random_string(4),
                             wavelength_range = (4., 5., 6.))

        self.assert_(self.chan < self.chan2)

        self.chan = Channel(name = "_" + random_string(4),
                            wavelength_range = (1., 2., 3.))
        self.chan2 = Channel(name = random_string(4),
                             wavelength_range = (4., 5., 6.))

        self.assert_(self.chan > self.chan2)


    def test_str(self):
        """String output for a channel.
        """
        self.chan = Channel(name="newchan",
                            wavelength_range=(1., 2., 3.),
                            resolution=1000)
        self.assertEqual(str(self.chan),
                         "'newchan: (1.000,2.000,3.000)μm, resolution 1000m,"
                         " not loaded'")

        self.chan.data = np.random.rand(3, 3)

        
        self.assertEqual(str(self.chan),
                         "'newchan: (1.000,2.000,3.000)μm, "
                         "shape (3, 3), "
                         "resolution 1000m'")
        

    def test_is_loaded(self):
        """Check load status of a channel.
        """
        data = np.random.rand(3, 3)
        
        self.chan = Channel(name = "newchan")
        self.assert_(not self.chan.is_loaded())

        self.chan = Channel(name = "newchan", data = data)
        self.assert_(self.chan.is_loaded())

    def test_as_image(self):
        """Check the geo_image version of the channel.
        """
        data = np.random.rand(3, 3)
        
        self.chan = Channel(name="newchan", data=data)
        img = self.chan.as_image(False)
        self.assert_(np.allclose(img.channels[0], data))
        self.assertEqual(img.mode, "L")
        img = self.chan.as_image(True)
        self.assertEqual(img.channels[0].max(), 1)
        self.assertEqual(img.channels[0].min(), 0)


    def test_check_range(self):
        """Check the range of a channel.
        """

        self.chan = Channel(name = "newchan")
        self.assertRaises(ValueError, self.chan.check_range)

        numb = np.random.uniform(10) 
        self.assertRaises(ValueError, self.chan.check_range, numb)

        # ndarray

        data = np.random.rand(3, 3)
        self.chan = Channel(name = "newchan", data = data)

        min_range = (data.max() - data.min()) / 2
        self.assert_(np.all(data == self.chan.check_range(min_range)))

        zeros = np.zeros_like(data)
        min_range = (data.max() - data.min()) + E
        self.assert_(np.all(zeros == self.chan.check_range(min_range)))

        # masked array

        mask = np.array(np.random.rand(3, 3) * 2, dtype = int)
        mask[1, 1] = False
        data = np.ma.array(data, mask = mask)
        self.chan = Channel(name = "newchan", data = data)

        min_range = (data.max() - data.min()) / 2
        self.assert_(np.all(data == self.chan.check_range(min_range)))
        self.assertEquals(data.count(),
                          self.chan.check_range(min_range).count())
        
        zeros = np.zeros_like(data)
        min_range = (data.max() - data.min()) + E
        self.assert_(np.all(zeros == self.chan.check_range(min_range)))

        data = np.ma.array(data, mask = True)
        self.chan = Channel(name = "newchan", data = data)
        self.assertEquals(0,
                          self.chan.check_range(min_range).count())
        self.assertEquals(data.count(),
                          self.chan.check_range(min_range).count())

        # Wrong type arguments

        self.assertRaises(TypeError, self.chan.check_range, random_string(4))

        self.assertRaises(TypeError,
                          self.chan.check_range,
                          [np.random.uniform()])

#    def test_project(self):
#        """Project a channel.
#        """
        # from pp.coverage import SatProjCov
#         from pp.scene import SatelliteInstrumentScene
        
#         cov = SatProjCov(SatelliteInstrumentScene(area = "euro"),
#                          "scan", 1000)
#         data = np.tile(np.array([[1, 2],[3, 4]]), (256, 256))
#         self.chan = Channel(name = "newchan", data = data)
#         self.chan.project(cov)

def random_string(length, choices=string.letters):
    """Generates a random string with elements from *set* of the specified
    *length*.
    """
    import random
    return "".join([random.choice(choices)
                    for i in range(length)])


if __name__ == '__main__':
    unittest.main()
