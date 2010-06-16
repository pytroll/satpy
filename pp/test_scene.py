"""Unit tests for scene.py.
"""
import unittest
from pp.scene import SatelliteScene, SatelliteInstrumentScene
from pp.channel import NotLoadedError
import datetime
import string
import numpy as np

class TestSatelliteScene(unittest.TestCase):
    """Class for testing the SatelliteScene class.
    """

    scene = None

    def test_init(self):
        """Creation of a satellite scene.
        """

        self.scene = SatelliteScene()
        self.assertEquals(self.scene.satname, "")
        self.assertEquals(self.scene.number, "")
        self.assert_(self.scene.time_slot is None)
        self.assert_(self.scene.orbit is None)
        self.assert_(self.scene.area_id is None)
        self.assert_(self.scene.lat is None)
        self.assert_(self.scene.lon is None)

        # time_slot

        time_slot = datetime.datetime.now()

        self.scene = SatelliteScene(time_slot = time_slot)
        self.assertEquals(self.scene.satname, "")
        self.assertEquals(self.scene.number, "")
        self.assertEquals(self.scene.time_slot, time_slot)
        self.assert_(self.scene.orbit is None)
        self.assert_(self.scene.area_id is None)
        self.assert_(self.scene.lat is None)
        self.assert_(self.scene.lon is None)

        self.assertRaises(TypeError,
                          SatelliteScene,
                          time_slot = random_string(4))
        
        self.assertRaises(TypeError,
                          SatelliteScene,
                          time_slot = np.random.uniform(1000))
        
        self.assertRaises(TypeError,
                          SatelliteScene,
                          time_slot = int(np.random.uniform(1000)))

        self.assertRaises(TypeError,
                          SatelliteScene,
                          time_slot = [])

        # area_id

        area_id = random_string(int(np.random.uniform(9)) + 1)

        self.scene = SatelliteScene(area_id = area_id)
        self.assertEquals(self.scene.satname, "")
        self.assertEquals(self.scene.number, "")
        self.assertEquals(self.scene.area_id, area_id)
        self.assert_(self.scene.orbit is None)
        self.assert_(self.scene.time_slot is None)
        self.assert_(self.scene.lat is None)
        self.assert_(self.scene.lon is None)

        self.assertRaises(TypeError,
                          SatelliteScene,
                          area_id = np.random.uniform(1000))
        
        self.assertRaises(TypeError,
                          SatelliteScene,
                          area_id = int(np.random.uniform(1000)))

        self.assertRaises(TypeError,
                          SatelliteScene,
                          area_id = [])


        
        # orbit

        orbit = random_string(int(np.random.uniform(9)) + 1)

        self.scene = SatelliteScene(orbit = orbit)
        self.assertEquals(self.scene.satname, "")
        self.assertEquals(self.scene.number, "")
        self.assertEquals(self.scene.orbit, orbit)
        self.assert_(self.scene.area_id is None)
        self.assert_(self.scene.time_slot is None)
        self.assert_(self.scene.lat is None)
        self.assert_(self.scene.lon is None)

        self.assertRaises(TypeError,
                          SatelliteScene,
                          orbit = np.random.uniform(1000))
        
        self.assertRaises(TypeError,
                          SatelliteScene,
                          orbit = int(np.random.uniform(1000)))

        self.assertRaises(TypeError,
                          SatelliteScene,
                          orbit = [])


    def test_fullname(self):
        """Fullname of a sat scene.
        """
        self.scene = SatelliteScene()

        self.scene.satname = random_string(int(np.random.uniform(9)) + 1)
        self.scene.number = random_string(int(np.random.uniform(9)) + 1)
        self.assertEquals(self.scene.fullname,
                          self.scene.satname + self.scene.number)
        
class TestSatelliteInstrumentScene(unittest.TestCase):
    """Class for testing the SatelliteInstrumentScene class.
    """

    scene = None
    def test_init(self):
        """Creation of a satellite instrument scene.
        """

        channels = [["00_7", (0.5, 0.7, 0.9), 2500],
                    ["06_4", (5.7, 6.4, 7.1), 5000],
                    ["11_5", (10.5, 11.5, 12.5), 5000]]

        class SatelliteInstrumentScene2(SatelliteInstrumentScene):
            channel_list = channels

        self.scene = SatelliteInstrumentScene2()
        self.assertEquals(self.scene.satname, "")
        self.assertEquals(self.scene.number, "")
        self.assert_(self.scene.time_slot is None)
        self.assert_(self.scene.orbit is None)
        self.assert_(self.scene.area_id is None)
        self.assert_(self.scene.lat is None)
        self.assert_(self.scene.lon is None)
        self.assert_(self.scene.instrument_name is None)
        self.assertEquals(self.scene.channels_to_load, set([]))
        for i, chn in enumerate(self.scene.channels):
            self.assertEquals(chn.name, channels[i][0])
            self.assertEquals(chn.wavelength_range, channels[i][1])
            self.assertEquals(chn.resolution, channels[i][2])

        # time_slot

        time_slot = datetime.datetime.now()

        self.scene = SatelliteInstrumentScene2(time_slot = time_slot)
        self.assertEquals(self.scene.satname, "")
        self.assertEquals(self.scene.number, "")
        self.assertEquals(self.scene.time_slot, time_slot)
        self.assert_(self.scene.orbit is None)
        self.assert_(self.scene.area_id is None)
        self.assert_(self.scene.lat is None)
        self.assert_(self.scene.lon is None)
        self.assert_(self.scene.instrument_name is None)
        self.assertEquals(self.scene.channels_to_load, set([]))
        for i, chn in enumerate(self.scene.channels):
            self.assertEquals(chn.name, channels[i][0])
            self.assertEquals(chn.wavelength_range, channels[i][1])
            self.assertEquals(chn.resolution, channels[i][2])

        self.assertRaises(TypeError,
                          SatelliteInstrumentScene2,
                          time_slot = random_string(4))
        
        self.assertRaises(TypeError,
                          SatelliteInstrumentScene2,
                          time_slot = np.random.uniform(1000))
        
        self.assertRaises(TypeError,
                          SatelliteInstrumentScene2,
                          time_slot = int(np.random.uniform(1000)))

        self.assertRaises(TypeError,
                          SatelliteInstrumentScene2,
                          time_slot = [])

        # area_id

        area_id = random_string(int(np.random.uniform(9)) + 1)

        self.scene = SatelliteInstrumentScene2(area_id = area_id)
        self.assertEquals(self.scene.satname, "")
        self.assertEquals(self.scene.number, "")
        self.assertEquals(self.scene.area_id, area_id)
        self.assert_(self.scene.orbit is None)
        self.assert_(self.scene.time_slot is None)
        self.assert_(self.scene.lat is None)
        self.assert_(self.scene.lon is None)
        self.assert_(self.scene.instrument_name is None)
        self.assertEquals(self.scene.channels_to_load, set([]))
        for i, chn in enumerate(self.scene.channels):
            self.assertEquals(chn.name, channels[i][0])
            self.assertEquals(chn.wavelength_range, channels[i][1])
            self.assertEquals(chn.resolution, channels[i][2])

        self.assertRaises(TypeError,
                          SatelliteInstrumentScene2,
                          area_id = np.random.uniform(1000))
        
        self.assertRaises(TypeError,
                          SatelliteInstrumentScene2,
                          area_id = int(np.random.uniform(1000)))

        self.assertRaises(TypeError,
                          SatelliteInstrumentScene2,
                          area_id = [])


        
        # orbit

        orbit = random_string(int(np.random.uniform(9)) + 1)

        self.scene = SatelliteInstrumentScene2(orbit = orbit)
        self.assertEquals(self.scene.satname, "")
        self.assertEquals(self.scene.number, "")
        self.assertEquals(self.scene.orbit, orbit)
        self.assert_(self.scene.area_id is None)
        self.assert_(self.scene.time_slot is None)
        self.assert_(self.scene.lat is None)
        self.assert_(self.scene.lon is None)
        self.assert_(self.scene.instrument_name is None)
        self.assertEquals(self.scene.channels_to_load, set([]))
        for i, chn in enumerate(self.scene.channels):
            self.assertEquals(chn.name, channels[i][0])
            self.assertEquals(chn.wavelength_range, channels[i][1])
            self.assertEquals(chn.resolution, channels[i][2])

        self.assertRaises(TypeError,
                          SatelliteInstrumentScene2,
                          orbit = np.random.uniform(1000))
        
        self.assertRaises(TypeError,
                          SatelliteInstrumentScene2,
                          orbit = int(np.random.uniform(1000)))

        self.assertRaises(TypeError,
                          SatelliteInstrumentScene2,
                          orbit = [])

    def test_getitem(self):
        """__getitem__ for sat scenes.
        """

        # empty scene
        self.scene = SatelliteInstrumentScene()        

        self.assertRaises(KeyError, self.scene.__getitem__,
                          np.random.uniform(100))
        self.assertRaises(KeyError, self.scene.__getitem__,
                          int(np.random.uniform(10000)))
        self.assertRaises(KeyError, self.scene.__getitem__,
                          random_string(4))

        # scene with 3 channels

        channels = [["00_7", (0.5, 0.7, 0.9), 2500],
                    ["06_4", (5.7, 6.4, 7.1), 5000],
                    ["11_5", (10.5, 11.5, 12.5), 5000]]

        class SatelliteInstrumentScene2(SatelliteInstrumentScene):
            channel_list = channels

        self.scene = SatelliteInstrumentScene2()
        for chn in channels:
            self.assertEquals(self.scene[chn[0]].name, chn[0])
            for i in range(3):
                self.assertEquals(self.scene[chn[1][i]].wavelength_range[i],
                                  chn[1][i])
            self.assertEquals(self.scene[chn[2]].resolution, chn[2])
            self.assertEquals(self.scene[(chn[0], chn[2])].name, chn[0])

        self.assertRaises(KeyError, self.scene.__getitem__, [])
        self.assertRaises(KeyError, self.scene.__getitem__, random_string(5))
        self.assertRaises(TypeError, self.scene.__getitem__, set([]))
        self.assertRaises(KeyError, self.scene.__getitem__, 5.0)

        self.assertEquals(len(self.scene.__getitem__(5000, aslist = True)), 2)

        chans = self.scene.__getitem__(5000, aslist = True)
        self.assertEquals(self.scene[chans[0].name].name, channels[1][0])
        self.assertEquals(self.scene[chans[1].name].name, channels[2][0])

    def test_check_channels(self):
        """Check loaded channels.
        """

        # No data loaded

        channels = [["00_7", (0.5, 0.7, 0.9), 2500],
                    ["06_4", (5.7, 6.4, 7.1), 5000],
                    ["11_5", (10.5, 11.5, 12.5), 5000]]

        class SatelliteInstrumentScene2(SatelliteInstrumentScene):
            channel_list = channels

        self.scene = SatelliteInstrumentScene2()
        for chn in channels:
            self.assertRaises(NotLoadedError, self.scene.check_channels, chn[0])
            self.assertRaises(NotLoadedError, self.scene.check_channels, chn[2])
            for i in range(3):
                self.assertRaises(NotLoadedError,
                                  self.scene.check_channels,
                                  chn[1][i])

        # With data
        
        self.scene[0.7] = np.ma.array(np.random.rand(3, 3),
                                      mask = np.array(np.random.rand(3, 3) * 2,
                                                      dtype = int))
        self.scene[6.4] = np.ma.array(np.random.rand(3, 3),
                                      mask = np.array(np.random.rand(3, 3) * 2,
                                                      dtype = int))
        self.scene[11.5] = np.ma.array(np.random.rand(3, 3),
                                       mask = np.array(np.random.rand(3, 3) * 2,
                                                       dtype = int))
        self.assertTrue(self.scene.check_channels(0.7, 6.4, 11.5))
        self.assertRaises(KeyError, self.scene.check_channels, random_string(5))

    def test_loaded_channels(self):
        """Loaded channels list.
        """
        # No data loaded

        channels = [["00_7", (0.5, 0.7, 0.9), 2500],
                    ["06_4", (5.7, 6.4, 7.1), 5000],
                    ["11_5", (10.5, 11.5, 12.5), 5000]]

        class SatelliteInstrumentScene2(SatelliteInstrumentScene):
            channel_list = channels

        self.scene = SatelliteInstrumentScene2()
        self.assertEquals(self.scene.loaded_channels(), set([]))
        
        # With data
        
        self.scene[0.7] = np.ma.array(np.random.rand(3, 3),
                                      mask = np.array(np.random.rand(3, 3) * 2,
                                                      dtype = int))
        self.scene[6.4] = np.ma.array(np.random.rand(3, 3),
                                      mask = np.array(np.random.rand(3, 3) * 2,
                                                      dtype = int))
        self.scene[11.5] = np.ma.array(np.random.rand(3, 3),
                                       mask = np.array(np.random.rand(3, 3) * 2,
                                                       dtype = int))

    def test_project(self):
        """Projecting a scene.
        """
        pass
    
def random_string(length, choices = string.letters):
    """Generates a random string with elements from *set* of the specified
    *length*.
    """
    import random
    return "".join([random.choice(choices)
                    for i in range(length)])

if __name__ == '__main__':
    unittest.main()
