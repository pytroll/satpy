#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module for testing the satpy.channel module.
"""

import unittest
from satpy.channel import GenericChannel, Channel
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

    def test_area(self):
        """Area setting and retrieving.
        """
        self.chan = GenericChannel(name = "newchan")
        self.chan.area = "bla"
        self.assert_(self.chan.area == "bla")

        self.chan.area = None
        self.assert_(self.chan.area is None)

        class DummyArea(object):
            def __init__(self, area_extent, x_size, y_size, proj_id, proj_dict):
                self.area_extent = area_extent
                self.x_size = x_size
                self.y_size = y_size
                self.proj_id = proj_id
                self.proj_dict = proj_dict

        self.chan.area = DummyArea(1, 2, 3, 4, 5)
        self.assert_(self.chan.area.area_extent == 1)

        class DummyArea(object):
            def __init__(self, lons, lats):
                self.lons = lons
                self.lats = lats

        self.chan.area = DummyArea(1, 2)
        self.assert_(self.chan.area.lats == 2)

        self.assertRaises(TypeError, setattr, self.chan, "area", 1)


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
                         [-np.inf, -np.inf, -np.inf])
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
                         [-np.inf, -np.inf, -np.inf])
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
                         [-np.inf, -np.inf, -np.inf])
        self.assertEqual(self.chan.resolution, 0)
        self.assert_(np.all(self.chan.data == data))

        mask = np.array(np.random.rand(3, 3) * 2, dtype = int)
        data = np.ma.array(data, mask = mask)
        
        self.chan = Channel(name = "newchan", data = data)
        self.assertEqual(self.chan.name, "newchan")
        self.assertEqual(self.chan.wavelength_range,
                         [-np.inf, -np.inf, -np.inf])
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


    def test_sunzen_corr(self):
        '''Test Sun zenith angle correction.
        '''

        import datetime as dt

        chan = Channel(name='test')
        
        original_value = 10.

        chan.data = original_value * np.ones((2,11))
        lats = np.zeros((2,11)) # equator
        lons = np.array([np.linspace(-90, 90, 11), np.linspace(-90, 90, 11)])

        # Equinox, so the Sun is at the equator
        time_slot = dt.datetime(2014,3,20,16,57)

        new_ch = chan.sunzen_corr(time_slot, lonlats=(lons, lats), limit=80.)

        # Test minimum after correction, accuracy of three decimals is enough
        #self.assertTrue(np.abs(10.000 - np.min(new_ch.data)) < 10**-3)
        self.assertAlmostEqual(10.000, np.min(new_ch.data), places=3)
        # Test maximum after correction
        self.assertAlmostEqual(57.588, np.max(new_ch.data), places=3)

        # There should be ten values at zenith angle >= 80 deg, and
        # these are all equal
        self.assertTrue(np.where(new_ch.data == \
                                     np.max(new_ch.data))[0].shape[0] == 10)

        # All values should be larger than the starting values
        self.assertTrue(np.all(new_ch.data > original_value))

        # Channel name
        self.assertEqual(new_ch.name, chan.name+'_SZC')
        
        # Test channel name in the info dict
        self.assertEqual(new_ch.name, chan.info['sun_zen_corrected'])

        # Test with several locations and arbitrary data
        chan = Channel(name='test2')
        chan.data = np.array([[0., 67.31614275, 49.96271995,
                               99.41046645, 29.08660989],
                              [87.61007584, 79.6683524, 53.20397351,
                               29.88260374, 62.33623915],
                              [60.49283004, 54.04267222, 32.72365906,
                               91.44995651, 32.27232955],
                              [63.71580638, 69.57673795, 7.63064373,
                               32.15683105, 9.05786335],
                              [65.61434337, 33.2317155, 18.77672384,
                               30.13527574, 23.22572904]])
        lons = np.array([[116.28695847, 164.1125604, 40.77223701,
                          -113.54699788, 133.15558442],
                         [-17.18990601, 75.17472034, 12.81618371,
                           -40.75524952, 40.70898002],
                         [42.74662341, 164.05671859, -166.58469404,
                          -58.16684483, -144.97963063],
                         [46.26303645, -167.48682034, 170.28131412,
                          -17.80502488, -63.9031154],
                         [-107.14829679, -147.66665952, -0.75970554,
                           77.701768, -130.48677807]])
        lats = np.array([[-51.53681682, -83.21762788, 5.91008672, 
                           22.51730385, 66.83356427],
                         [82.78543163,  23.1529456 ,  -7.16337152,
                          -68.23118425, 28.72194953],
                         [31.03440852, 70.55322517, -83.61780288,
                          29.88413938, 25.7214828],
                         [-19.02517922, -19.20958728, -14.7825735,
                           22.66967876, 67.6089238],
                         [45.12202477, 61.79674149, 58.71037615,
                          -62.04350423, 13.06405864]])
        time_slot = dt.datetime(1998, 8, 1, 10, 0)

        # These are the expected results
        results = np.array([[0., 387.65821593, 51.74080022,
                             572.48205988, 138.96586013],
                            [227.24857818, 105.53045776, 62.24134162,
                             172.0870564, 64.12902666],
                            [63.08646652, 311.21934562, 188.44804188,
                             526.63931022, 185.84893885],
                            [82.86856236, 400.6764648, 43.9431259,
                             46.58056343, 36.04457644],
                            [377.85794388, 191.3738223, 27.55002934,
                             173.54213642, 133.75164285]])

        new_ch = chan.sunzen_corr(time_slot, lonlats=(lons, lats), limit=80.)
        self.assertAlmostEqual(np.max(results-new_ch.data), 0.000, places=3)
        
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


def suite():
    """The test suite for test_channel.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestGenericChannel))
    mysuite.addTest(loader.loadTestsFromTestCase(TestChannel))
    
    return mysuite
