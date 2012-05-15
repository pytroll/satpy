#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2011, 2012.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# mpop is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# mpop.  If not, see <http://www.gnu.org/licenses/>.

"""Classes for gathering segmented data.
"""
import datetime
import glob
import os.path
from ConfigParser import ConfigParser, NoSectionError, NoOptionError
from fnmatch import fnmatch
import numpy as np
import sys

import mpop.utils
from mpop import CONFIG_PATH
from mpop.scene import Satellite, SatelliteInstrumentScene
from mpop.projector import get_area_def
from mpop.satellites import GenericFactory

DEFAULT_TIMELINESS = datetime.timedelta(minutes=5)
DEFAULT_GRANULARITY = datetime.timedelta(minutes=5)


LOG = mpop.utils.get_logger("gatherer")

if sys.version_info < (2, 5):
    import time
    def strptime(string, fmt=None):
        """This function is available in the datetime module only
        from Python >= 2.5.
        """

        return datetime.datetime(*time.strptime(string, fmt)[:6])
else:
    strptime = datetime.datetime.strptime

    
def globify(filename):
    """Replace datetime string variable with ?'s.
    """
    filename = filename.replace("%Y", "????")
    filename = filename.replace("%j", "???")
    filename = filename.replace("%m", "??")
    filename = filename.replace("%d", "??")
    filename = filename.replace("%H", "??")
    filename = filename.replace("%M", "??")
    filename = filename.replace("%S", "??")
    return filename

def beginning(filename):
    """Find the beginning of *filename* not having any wildcards (? or *).
    Returns a duplet containing the position of the first wildcard, and the
    same position if the datetime variables where expanded.
    """
    posqm = filename.find("?")
    posst = filename.find("*")

    if posqm == -1 and posst == -1:
        pos = len(filename)
    elif posqm == -1:
        pos = posst
    elif posst == -1:
        pos = posqm
    else:
        pos = min(posst, posqm)

    secondpos = pos
    if filename[:pos].find("%Y") > -1:
        secondpos += 2
    if filename[:pos].find("%j") > -1:
        secondpos += 1
    return (pos, secondpos)

class Granule(SatelliteInstrumentScene):
    """The granule object.
    """

    def __init__(self, filename=None, time_slot=None,
                 satellite=None, instrument=None):

        # Setting up a granule from metadata

        if filename is None:
            SatelliteInstrumentScene.__init__(self, time_slot=time_slot,
                                              satellite=(satellite.satname,
                                                         satellite.number,
                                                         satellite.variant),
                                              instrument=instrument)
            conf = ConfigParser()
            conf.read(os.path.join(CONFIG_PATH, self.fullname + ".cfg"))
            self.file_template = str(conf.get(instrument+"-granules",
                                              "filename", raw=True))
            self.directory = str(conf.get(instrument+"-granules",
                                          "dir", raw=True))
            
            self.file_name = time_slot.strftime(self.file_template)
            self.directory = time_slot.strftime(self.directory)
            self.file_type = conf.get(instrument + "-granules", "type")
            self.granularity = datetime.timedelta(
                seconds=int(conf.get(instrument + "-granules", "granularity")))
            self.span = float(conf.get(instrument + "-granules",
                                       "full_scan_period"))
            self.scan_width = int(conf.get(instrument + "-granules",
                                           "scan_width"))
            return

        # Setting up a granule from a filename
        
        filelist = glob.glob(os.path.join(CONFIG_PATH, "*.cfg"))

        the_directory, the_name = os.path.split(filename)

        self.satname = None
        
        for fil in filelist:
            conf = ConfigParser()
            conf.read(fil)
            try:
                instruments = eval(conf.get("satellite", "instruments"))
                for instrument in instruments:
                    directory = str(conf.get(instrument+"-granules", "dir"))
                    file_template = str(conf.get(instrument+"-granules",
                                                 "filename", raw=True))
                    file_glob = globify(file_template)

                    if(os.path.samefile(the_directory, directory) and
                       fnmatch(the_name, file_glob)):
                        try:
                            self.file_template = file_template
                            self.file_name = the_name

                            pos1, pos2 = beginning(self.file_template)
                            time_slot = strptime(self.file_name[:pos2],
                                                 self.file_template[:pos1])
        
                            SatelliteInstrumentScene.__init__(
                                self,
                                time_slot=time_slot,
                                satellite=(conf.get("satellite", "satname"),
                                           conf.get("satellite", "number"),
                                           conf.get("satellite", "variant")),
                                instrument=instrument)

                            self.file_type = conf.get(instrument +
                                                      "-granules", "type")
                            self.directory = the_directory
                            self.granularity = datetime.timedelta(
                                seconds=int(conf.get(instrument + "-granules",
                                                     "granularity")))
                            self.span = float(conf.get(instrument +
                                                       "-granules",
                                                       "full_scan_period"))
                            self.scan_width = int(conf.get(instrument +
                                                           "-granules",
                                                           "scan_width"))
                        except (NoSectionError, NoOptionError):
                            raise IOError("Inconsistency detected in " + fil)
                        break

                if self.satname is not None:
                    break
                    
            except (NoSectionError, NoOptionError):
                pass

            
        if not self.satname:
            raise ValueError("Can't find any matching satellite for "+filename)
        

    def __cmp__(self, obj):
        return (cmp(self.satname, obj.satname) or
                cmp(self.number, obj.number) or
                cmp(self.variant, obj.variant) or
                cmp(self.time_slot, obj.time_slot))

    def __str__(self):
        return "G:" + os.path.join(self.directory, self.file_name)

    def get_lonlat(self, row, col):
        """Get the longitude and latitude for the current scene at the given
        row and col.
        """
        
        conf = ConfigParser()
        conf.read(os.path.join(CONFIG_PATH, self.fullname + ".cfg"))

        try:
            reader_name = conf.get(self.instrument_name + "-level1", 'format')
        except NoSectionError:
            reader_name = conf.get(self.instrument_name + "-level2", 'format')
        reader = "mpop.satin." + reader_name
        try:
            reader_module = __import__(reader,
                                       globals(), locals(),
                                       ['get_lonlat'])
        except ImportError:
            LOG.exception("Problem finding a reader...")
            raise ImportError("No "+reader+" reader found.")
        return reader_module.get_lonlat(self, row, col)

    @property
    def gross_area(self):
        """Build a gross area of the segment based on its corners.
        """
        from pyresample.geometry import SwathDefinition

        nlines = self.granularity.seconds / self.span
        top_left = self.get_lonlat(0, 0)
        top_right = self.get_lonlat(0, self.scan_width - 1)
        bottom_left = self.get_lonlat(nlines - 1, 0)
        bottom_right = self.get_lonlat(nlines - 1, self.scan_width - 1)
        lons = np.array([[top_left[0], top_right[0]],
                         [bottom_left[0], bottom_right[0]]])
        lats = np.array([[top_left[1], top_right[1]],
                         [bottom_left[1], bottom_right[1]]])
        return SwathDefinition(lons, lats)
        
class SegmentedSwath(Satellite):

    def __init__(self, area, (satname, number, variant)):
        Satellite.__init__(self, (satname, number, variant))
        self.area = get_area_def(area)
        self.granules = []
        self.planned_granules = []
        self.timeout = None
    

    def add(self, granule):
        """Add a granule to the swath
        """
        if not self.granules:
            self._compute_plan(granule.time_slot,
                               granule.granularity,
                               granule.instrument_name)
        try:
            self.planned_granules.remove(granule)
        except ValueError:
            LOG.warning(str(granule) + " not in " +
                        str([str(gran) for gran in self.planned_granules]))
        self.granules.append(granule)
        if self.planned_granules:
            self.timeout = self.planned_granules[-1].time_slot
        else:
            self.timeout = datetime.datetime.now()

    def _compute_plan(self, utc_time, granularity, instrument):
        """Compute the planned granules for the current area.
        """
        nts = utc_time
        
        new_granule = Granule(time_slot=nts, satellite=self,
                              instrument=instrument)
        if new_granule.gross_area.overlaps(self.area):
            self.planned_granules.append(new_granule)

        while True:
            nts = nts - granularity
            new_granule = Granule(time_slot=nts, satellite=self,
                                  instrument=instrument)
            if new_granule.gross_area.overlaps(self.area):
                self.planned_granules.append(new_granule)
            else:
                break

        nts = utc_time
        while True:
            nts = nts + granularity
            new_granule = Granule(time_slot=nts, satellite=self,
                                  instrument=instrument)
            if new_granule.gross_area.overlaps(self.area):
                self.planned_granules.append(new_granule)
            else:
                break

        self.planned_granules.sort()
        self.timeout = self.planned_granules[-1].time_slot

    def __repr__(self):
        granules = [str(granule) for granule in self.granules]
        return "swath " + str(granules) + " on area " + self.area.area_id

    def concatenate(self, channels=None):
        """Concatenate the granules into a swath. Returns a loaded satellite
        scene.
        """
        self.granules.sort()
        conffile = (self.granules[0].variant +
                    self.granules[0].satname +
                    self.granules[0].number +
                    ".cfg")
        conf = ConfigParser()
        conf.read(os.path.join(CONFIG_PATH, conffile))

        try:
            
            # call concat

            reader_name = conf.get(self.granules[0].instrument_name +
                                   "-level1", 'format')

            str_granules = [str(granule) for granule in self.granules]
            LOG.debug("Concatenating "+str(str_granules))

            try:
                reader_name = eval(reader_name)
            except NameError:
                reader_name = str(reader_name)

            # read the data
            reader = "mpop.satin."+reader_name

            try:
                reader_module = __import__(reader, globals(), locals(),
                                           ['concatenate'])
            except ImportError:
                LOG.exception("ImportError when loading plugin for format " +
                              str(reader_name))
                raise ImportError("No "+reader+" reader found.")

            scene = reader_module.concatenate(self.granules, channels)
            
        except NoSectionError:
            #concatenate loaded granules.
            scenes = [GenericFactory.create_scene(granule.satname,
                                                  granule.number,
                                                  granule.instrument_name,
                                                  granule.time_slot,
                                                  None,
                                                  None,
                                                  granule.variant)
                      for granule in self.granules]
            for granule in scenes:
                granule.load(channels)
            scene = mpop.scene.assemble_segments(scenes)

        return scene
        
            
class Gatherer(Satellite):
    """The mighty gatherer class.
    """

    def __init__(self, areas_of_interest,
                 timeliness=DEFAULT_TIMELINESS,
                 satname=None, number=None, variant=None):
        Satellite.__init__(self, (satname, number, variant))
        self.timeliness = timeliness
        self.swaths = {}
        self.finished_swaths = []


        for area in areas_of_interest:
            self.swaths[area] = SegmentedSwath(area, (satname, number, variant))

    def add(self, granule):
        """Add a *granule* to the gatherer.
        """
        interesting = False
        for area_name in self.swaths:
            if granule.gross_area.overlaps(self.swaths[area_name].area):
                interesting = True
                if (len(self.swaths[area_name].planned_granules) > 0 and
                    granule not in self.swaths[area_name].planned_granules):
                    LOG.debug("Following swath is starting, closing the current"
                              " swath")
                    LOG.debug("Swath was " + str(self.swaths[area_name]) +
                              " while " +
                              str(self.swaths[area_name].planned_granules) +
                              " remains.")
                    self.finished_swaths.append(self.swaths[area_name])
                    self.swaths[area_name] = SegmentedSwath(area_name,
                                                            (granule.satname,
                                                             granule.number,
                                                             granule.variant))
                LOG.debug("Add " + str(granule) + " to " +
                          str(self.swaths[area_name]))
                self.swaths[area_name].add(granule)
                if len(self.swaths[area_name].planned_granules) == 0:
                    self.finished_swaths.append(self.swaths[area_name])
                    self.swaths[area_name] = SegmentedSwath(area_name,
                                                            (granule.satname,
                                                             granule.number,
                                                             granule.variant))
        return interesting
    
    def timeout(self):
        """Finishes swaths that are timed out.
        """
        now = datetime.datetime.utcnow()
        
        for area_name, swath in self.swaths.items():
            if swath.timeout and swath.timeout + self.timeliness < now:
                self.finished_swaths.append(swath)
                self.swaths[area_name] = SegmentedSwath(area_name,
                                                        (self.satname,
                                                         self.number,
                                                         self.variant))
    def clear(self):
        """Clean up the finished swaths.
        """
        self.finished_swaths = []
        
    def __str__(self):
        return self.fullname
        
# Typical example:

# when new_file:
#     gatherer.add(Granule(new_file))
#     for swath in gatherer.finished:
#         scene = swath.assemble()
#         do_something_with(scene)
#     gatherer.finished.clear()

    
# when its time (gatherer.next_timeout):
#     gatherer.finish_timeouts()
#     for swath in gatherer.finished:
#         scene = swath.assemble()
#         do_something_with(scene)
#     gatherer.finished.clear()


import unittest

OldConfigParser = None

def patch_glob():
    """Patch fnmatch.
    """
    def fake_glob(*args):
        """Fake glob.
        """
        del args
        return ["test.cfg"]

    glob.old_glob = glob.glob
    glob.glob = fake_glob

def unpatch_glob():
    """Unpatch glob.
    """
    glob.glob = glob.old_glob
    delattr(glob, "old_glob")

def patch_configparser():
    """Patch to fake ConfigParser.
    """
    class FakeConfigParser:
        """Dummy ConfigParser class.
        """
        def __init__(self, *args, **kwargs):
            pass
        
        def read(self, *args, **kwargs):
            """Dummy read method
            """
            del args, kwargs
            self = self

        def get(self, section, item, **kwargs):
            """Dummy get method
            """
            del kwargs
            self = self
            if(section == "avhrr-granules" and
               item == "filename"):
                return "myfile_%Y%m%d_%H%M.cooltype"
            if(section == "avhrr-granules" and
               item == "type"):
                return "cooltype"
            if(section == "avhrr-granules" and
               item == "granularity"):
                return 60
            if(section == "avhrr-granules" and
               item == "full_scan_period"):
                return 0.1667
            if(section == "avhrr-granules" and
               item == "scan_width"):
                return 2048
            if(section == "avhrr-granules" and
               item == "dir"):
                return "/path/to/my/data"
            if(section == "satellite" and
               item == "instruments"):
                return "('avhrr', )"
            if(section == "satellite" and
               item == "satname"):
                return "metop"
            if(section == "satellite" and
               item == "number"):
                return "02"
            if(section == "satellite" and
               item == "variant"):
                return "regional"
            

    global OldConfigParser, ConfigParser

    OldConfigParser = ConfigParser
    ConfigParser = FakeConfigParser

def unpatch_configparser():
    """Unpatch fake ConfigParser.
    """
    global ConfigParser
    ConfigParser = OldConfigParser
    



class TestGranules(unittest.TestCase):
    """Testing granules.
    """

    def setUp(self):
        """Patching stuff.
        """
        patch_configparser()
        patch_glob()

    def tearDown(self):
        """Unpatching stuff.
        """
        unpatch_configparser()
        unpatch_glob()

    def test_init(self):
        """Testing init function.
        """

        g_1 = Granule("/path/to/my/data/myfile_20101010_1010.cooltype")

        self.assertEquals(g_1.satname, "metop")
        self.assertEquals(g_1.number, "02")
        self.assertEquals(g_1.variant, "regional")
        self.assertEquals(g_1.time_slot,
                          datetime.datetime(2010, 10, 10, 10, 10))
        self.assertEquals(g_1.instrument_name, "avhrr")
        
        sat = Satellite("metop", "02", "regional")
        g_1 = Granule(instrument="avhrr", satellite=sat,
                     time_slot=datetime.datetime(2010, 10, 10, 10, 10))
        
        self.assertEquals(os.path.join(g_1.directory, g_1.file_name),
                          "/path/to/my/data/myfile_20101010_1010.cooltype")
        

    def test_cmp(self):
        """test the __cmp__ function.
        """
        g_1 = Granule("/path/to/my/data/myfile_20101010_1010.cooltype")
        g_2 = Granule("/path/to/my/data/myfile_20101010_1011.cooltype")
        self.assertTrue(g_1 < g_2)
        self.assertTrue(g_2 > g_1)
        self.assertTrue(g_1 == g_1)


OldGranule = None

def patch_granule():
    """Faking Granule.
    """
    class FakeArea:
        """Fake area class.
        """
        def __init__(self, inside):
            self.inside = inside
            
        def overlaps(self, other):
            """Fake overlaping function.
            """
            del other
            return self.inside
    
    class FakeGranule:
        """Fake granule class.
        """
        def __init__(self, time_slot, satellite):
            self.time_slot = time_slot
            self.satellite = satellite
            self.span = 1

        def __cmp__(self, other):
            return cmp(self.time_slot, other.time_slot)

        @property
        def gross_area(self):
            """Approximate area of the granule.
            """
            if(self.time_slot > 2 and
               self.time_slot < 8):
                return FakeArea(inside=True)
            else:
                return FakeArea(inside=False)

        def __repr__(self):
            return "G:" + str(self.time_slot)
        
    global Granule, OldGranule
    OldGranule = Granule
    Granule = FakeGranule

def patch_granule_with_time():
    """Faking Granule.
    """
    class FakeArea:
        """Fake area class.
        """
        def __init__(self, inside):
            self.inside = inside
            
        def overlaps(self, other):
            """Fake overlaping function.
            """
            return self.inside
    
    class FakeGranule:
        """Fake granule class.
        """
        satname = "bli"
        number = "blu"
        variant = "bla"
        def __init__(self, time_slot, satellite):
            self.time_slot = time_slot
            self.satellite = satellite
            self.span = datetime.timedelta(minutes=1)

        def __cmp__(self, other):
            return cmp(self.time_slot, other.time_slot)

        @property
        def gross_area(self):
            """Approximate area of the granule.
            """
            start_time = datetime.datetime(2010, 10, 10, 0, 2)
            end_time = datetime.datetime(2010, 10, 10, 0, 8)
            if(self.time_slot > start_time and
               self.time_slot < end_time):
                return FakeArea(inside=True)
            else:
                return FakeArea(inside=False)

        def __repr__(self):
            return "G:" + str(self.time_slot)
        
    global Granule, OldGranule
    OldGranule = Granule
    Granule = FakeGranule

def unpatch_granule():

    global Granule
    Granule = OldGranule
    

class TestSegmentedSwath(unittest.TestCase):
    """Testing SegmentedSwath.
    """
    def setUp(self):
        patch_granule()

    def tearDown(self):
        unpatch_granule()

    def test_init(self):
        """Test initialisation.
        """
        swath = SegmentedSwath("bla", "bli", "blu", "blo")
        self.assertEquals(swath.area, "bla")

    def test_add(self):
        """Test adding.
        """
        swath = SegmentedSwath("bla", "bli", "blu", "blo")
        granule = Granule(5, "kurt")
        
        swath.add(granule)
        self.assertEquals(granule.time_slot, swath.granules[0].time_slot)
        times = [granule.time_slot for granule in swath.planned_granules]
        self.assertEquals(times, [3, 4, 6, 7])

        self.assertEquals(swath.timeout, 7)
        
        granule = Granule(4, "kurt")
        swath.add(granule)

        times = [granule.time_slot for granule in swath.planned_granules]
        self.assertEquals(times, [3, 6, 7])

        times = [granule.time_slot for granule in swath.granules]
        times.sort()
        self.assertEquals(times, [4, 5])

        self.assertEquals(swath.timeout, 7)
        
        granule = Granule(6, "kurt")
        swath.add(granule)
        
        times = [granule.time_slot for granule in swath.planned_granules]
        self.assertEquals(times, [3, 7])

        times = [granule.time_slot for granule in swath.granules]
        times.sort()
        self.assertEquals(times, [4, 5, 6])

        self.assertEquals(swath.timeout, 7)

        granule = Granule(7, "kurt")
        swath.add(granule)

        times = [granule.time_slot for granule in swath.planned_granules]
        self.assertEquals(times, [3])

        times = [granule.time_slot for granule in swath.granules]
        times.sort()
        self.assertEquals(times, [4, 5, 6, 7])

        self.assertEquals(swath.timeout, 3)

        granule = Granule(3, "kurt")
        swath.add(granule)

        times = [granule.time_slot for granule in swath.planned_granules]
        self.assertEquals(times, [])

        times = [granule.time_slot for granule in swath.granules]
        times.sort()
        self.assertEquals(times, [3, 4, 5, 6, 7])

        self.assertTrue((datetime.datetime.now() - swath.timeout).seconds < 1)
        self.assertTrue(isinstance(swath.timeout, datetime.datetime))



    def test_compute_plan(self):
        """Test planning of comming granules.
        """
        swath = SegmentedSwath("bla", "bli", "blu", "blo")

        swath._compute_plan(5, 1)
        times = [granule.time_slot for granule in swath.planned_granules]
        self.assertEquals(times, [3, 4, 5, 6, 7])


def patch_now():
    """Patching the now function from datetime.
    """
    def fakenow():
        """Fake now function.
        """
        return datetime.datetime(2010, 10, 10, 0, 10)
    
    datetime.datetime.oldnow = datetime.datetime.now
    datetime.datetime.now = fakenow

def unpatch_now():
    """Returning the original now.
    """
    datetime.datetime.now = datetime.datetime.oldnow
    delattr(datetime.datetime, "oldnow")

class TestGatherer(unittest.TestCase):
    """Testing Gatherer.
    """

    def setUp(self):
        patch_granule_with_time()
        
    def tearDown(self):
        unpatch_granule()
        
    def test_init(self):
        """Testing initialisation.
        """
        gatherer = Gatherer(["bli", "blu"],
                            satname="gla", number="glo", variant="glu")
        self.assertTrue("bli" in gatherer.swaths)
        self.assertTrue(isinstance(gatherer.swaths["bli"], SegmentedSwath))
        self.assertTrue("blu" in gatherer.swaths)
        self.assertTrue(isinstance(gatherer.swaths["blu"], SegmentedSwath))

    def test_add(self):
        """Testing adding of granules to the gatherer.
        """
        timeliness = (datetime.datetime.utcnow() -
                      datetime.datetime(2010, 10, 9, 23, 0))

        gatherer = Gatherer(["bli", "blu"],
                            timeliness=timeliness,
                            satname="gla", number="glo", variant="glu")
        gatherer.add(Granule(datetime.datetime(2010, 10, 10, 0, 5), "blaf"))
        gatherer.add(Granule(datetime.datetime(2010, 10, 10, 0, 3), "blaf"))
        gatherer.add(Granule(datetime.datetime(2010, 10, 10, 0, 4), "blaf"))

        times = [granule.time_slot
                 for granule in gatherer.swaths["bli"].granules]
        ideal_times = [datetime.datetime(2010, 10, 10, 0, 5),
                       datetime.datetime(2010, 10, 10, 0, 3),
                       datetime.datetime(2010, 10, 10, 0, 4)]
        self.assertEquals(times, ideal_times)

        # put the timeliness to zero and add a new granule:
        # This should add the granule as normal.
        
        gatherer.timeliness = datetime.timedelta(seconds=0)
        gatherer.add(Granule(datetime.datetime(2010, 10, 10, 0, 7), "blaf"))

        times = [granule.time_slot
                 for granule in gatherer.swaths["bli"].granules]
        ideal_times = [datetime.datetime(2010, 10, 10, 0, 5),
                       datetime.datetime(2010, 10, 10, 0, 3),
                       datetime.datetime(2010, 10, 10, 0, 4),
                       datetime.datetime(2010, 10, 10, 0, 7)]
        self.assertEquals(times, ideal_times)

        gatherer.timeliness = timeliness
        gatherer.add(Granule(datetime.datetime(2010, 10, 10, 0, 6), "blaf"))

        self.assertFalse(gatherer.swaths["bli"].granules)
        self.assertFalse(gatherer.swaths["blu"].granules)
        self.assertEquals(len(gatherer.finished_swaths), 2)


    def test_timeout(self):
        """Test the timeout.
        """
        timeliness = (datetime.datetime.utcnow() -
                      datetime.datetime(2010, 10, 9, 23, 0))

        gatherer = Gatherer(["kurt", "blu"],
                            timeliness=timeliness,
                            satname="gla", number="glo", variant="glu")
        gatherer.add(Granule(datetime.datetime(2010, 10, 10, 0, 5), "blaf"))
        gatherer.add(Granule(datetime.datetime(2010, 10, 10, 0, 3), "blaf"))
        gatherer.add(Granule(datetime.datetime(2010, 10, 10, 0, 4), "blaf"))

        gatherer.timeout()
        times = [granule.time_slot
                 for granule in gatherer.swaths["kurt"].granules]
        ideal_times = [datetime.datetime(2010, 10, 10, 0, 5),
                       datetime.datetime(2010, 10, 10, 0, 3),
                       datetime.datetime(2010, 10, 10, 0, 4)]
        self.assertEquals(times, ideal_times)

        gatherer.timeliness = datetime.timedelta(seconds=0)
        gatherer.timeout()
        times = [granule.time_slot
                 for granule in gatherer.finished_swaths[0].granules]
        ideal_times = [datetime.datetime(2010, 10, 10, 0, 5),
                       datetime.datetime(2010, 10, 10, 0, 3),
                       datetime.datetime(2010, 10, 10, 0, 4)]
        self.assertEquals(times, ideal_times)

        self.assertFalse(gatherer.swaths["kurt"].granules)
        self.assertFalse(gatherer.swaths["kurt"].planned_granules)
        self.assertFalse(gatherer.swaths["blu"].granules)
        self.assertFalse(gatherer.swaths["blu"].planned_granules)
        

        
if __name__ == "__main__":
    unittest.main()
