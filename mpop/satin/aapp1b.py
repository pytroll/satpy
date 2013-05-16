#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2012, 2013 SMHI

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>
#   Nina Håkansson <nina.hakansson@smhi.se>
#   Oana Nicola <oananicola@yahoo.com>
#   Lars Ørum Rasmussen <ras@dmi.dk>

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

"""Reader for aapp level 1b data.

http://research.metoffice.gov.uk/research/interproj/nwpsaf/aapp/
NWPSAF-MF-UD-003_Formats.pdf
"""


import numpy as np
import os
import logging
import datetime
import glob
from ConfigParser import ConfigParser
from mpop import CONFIG_PATH

LOG = logging.getLogger('aapp1b')

def load(satscene, *args, **kwargs):
    """Read data from file and load it into *satscene*.
    A possible *calibrate* keyword argument is passed to the AAPP reader. 
    Should be 0 for off (counts), 1 for default (brightness temperatures and
    reflectances), and 2 for radiances only.
    """
    del args

    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name + "-level2",
                                    raw = True):
        options[option] = value

    options["calibrate"] = kwargs.get("calibrate", True)

    LOG.info("Loading instrument '%s'" % satscene.instrument_name)
    try:
        CASES[satscene.instrument_name](satscene, options)
    except KeyError:
        raise KeyError("Unknown instrument '%s'" % satscene.instrument_name)

def load_avhrr(satscene, options):
    """Read avhrr data from file and load it into *satscene*.
    """
    if "filename" not in options:
        raise IOError("No filename given, cannot load.")

    chns = satscene.channels_to_load & set(AVHRR_CHANNEL_NAMES)
    LOG.info("Loading channels " + str(sorted(list(chns))))
    if len(chns) == 0:
        return

    values = {"orbit": satscene.orbit,
              "satname": satscene.satname,
              "number": satscene.number,
              "instrument": satscene.instrument_name,
              "satellite": satscene.fullname
              }

    filename = os.path.join(satscene.time_slot.strftime(options["dir"])%values,
                            satscene.time_slot.strftime(options["filename"])
                            %values)

    file_list = glob.glob(filename)

    if len(file_list) > 1:
        raise IOError("More than one l1b file matching!")
    elif len(file_list) == 0:
        raise IOError("No l1b file matching!: " + filename)

    
    filename = file_list[0]

    LOG.debug("Loading from " + filename)

    scene = AAPP1b(filename)
    scene.read()
    scene.calibrate(chns, calibrate=options.get('calibrate', 1))

    scene.navigate()
    for chn in chns:
        if scene.channels.has_key(chn):
            satscene[chn].data = scene.channels[chn]
            satscene[chn].info['units'] = scene.units[chn]

    try:
        from pyresample import geometry
    except ImportError, ex_:
        LOG.debug("Could not load pyresample: " + str(ex_))
        satscene.lat = scene.lats
        satscene.lon = scene.lons
    else:
        satscene.area = geometry.SwathDefinition(lons=scene.lons,
                                                 lats=scene.lats)

AVHRR_CHANNEL_NAMES = ("1", "2", "3A", "3B", "4", "5")

# AAPP 1b header

_HEADERTYPE = np.dtype([("siteid", "S3"),
                        ("blank", "S1"),
                        ("l1bversnb", "<i2"),
                        ("l1bversyr", "<i2"),
                        ("l1bversdy", "<i2"),
                        ("reclg", "<i2"),
                        ("blksz", "<i2"),
                        ("hdrcnt", "<i2"),
                        ("filler0", "S6"),
                        ("dataname", "S42"),
                        ("prblkid", "S8"),
                        ("satid", "<i2"),
                        ("instid", "<i2"),
                        ("datatype", "<i2"),
                        ("tipsrc", "<i2"),
                        ("startdatajd", "<i4"),
                        ("startdatayr", "<i2"),
                        ("startdatady", "<i2"),
                        ("startdatatime", "<i4"),
                        ("enddatajd", "<i4"),
                        ("enddatayr", "<i2"),
                        ("enddatady", "<i2"),
                        ("enddatatime", "<i4"),
                        ("cpidsyr", "<i2"),
                        ("cpidsdy", "<i2"),
                        ("filler1", "S8"),
                        # data set quality indicators
                        ("inststat1", "<i4"),
                        ("filler2", "S2"),
                        ("statchrecnb", "<i2"),
                        ("inststat2", "<i4"),
                        ("scnlin", "<i2"),
                        ("callocscnlin", "<i2"),
                        ("misscnlin", "<i2"),
                        ("datagaps", "<i2"),
                        ("okdatafr", "<i2"),
                        ("pacsparityerr", "<i2"),
                        ("auxsyncerrsum", "<i2"),
                        ("timeseqerr", "<i2"),
                        ("timeseqerrcode", "<i2"),
                        ("socclockupind", "<i2"),
                        ("locerrind", "<i2"),
                        ("locerrcode", "<i2"),
                        ("pacsstatfield", "<i2"),
                        ("pacsdatasrc", "<i2"),
                        ("filler3", "S4"),
                        ("spare1", "S8"),
                        ("spare2", "S8"),
                        ("filler4", "S10"),
                        # Calibration
                        ("racalind", "<i2"),
                        ("solarcalyr", "<i2"),
                        ("solarcaldy", "<i2"),
                        ("pcalalgind", "<i2"),
                        ("pcalalgopt", "<i2"),
                        ("scalalgind", "<i2"),
                        ("scalalgopt", "<i2"),
                        ("irttcoef", "<i2", (4, 6)),
                        ("filler5", "<i4", (2, )),
                        # radiance to temperature conversion
                        ("albcnv", "<i4", (2, 3)),
                        ("radtempcnv", "<i4", (3, 3)),
                        ("filler6", "<i4", (3, )),
                        # Navigation
                        ("modelid", "S8"),
                        ("nadloctol", "<i2"),
                        ("locbit", "<i2"),
                        ("filler7", "S2"),
                        ("rollerr", "<i2"),
                        ("pitcherr", "<i2"),
                        ("yawerr", "<i2"),
                        ("epoyr", "<i2"),
                        ("epody", "<i2"),
                        ("epotime", "<i4"),
                        ("smaxis", "<i4"),
                        ("eccen", "<i4"),
                        ("incli", "<i4"),
                        ("argper", "<i4"),
                        ("rascnod", "<i4"),
                        ("manom", "<i4"),
                        ("xpos", "<i4"),
                        ("ypos", "<i4"),
                        ("zpos", "<i4"),
                        ("xvel", "<i4"),
                        ("yvel", "<i4"),
                        ("zvel", "<i4"),
                        ("earthsun", "<i4"),
                        ("filler8", "S16"),
                        # analog telemetry conversion
                        ("pchtemp", "<i2", (5, )),
                        ("reserved1", "<i2"),
                        ("pchtempext", "<i2", (5, )),
                        ("reserved2", "<i2"),
                        ("pchpow", "<i2", (5, )),
                        ("reserved3", "<i2"),
                        ("rdtemp", "<i2", (5, )),
                        ("reserved4", "<i2"),
                        ("bbtemp1", "<i2", (5, )),
                        ("reserved5", "<i2"),
                        ("bbtemp2", "<i2", (5, )),
                        ("reserved6", "<i2"),
                        ("bbtemp3", "<i2", (5, )),
                        ("reserved7", "<i2"),
                        ("bbtemp4", "<i2", (5, )),
                        ("reserved8", "<i2"),
                        ("eleccur", "<i2", (5, )),
                        ("reserved9", "<i2"),
                        ("motorcur", "<i2", (5, )),
                        ("reserved10", "<i2"),
                        ("earthpos", "<i2", (5, )),
                        ("reserved11", "<i2"),
                        ("electemp", "<i2", (5, )),
                        ("reserved12", "<i2"),
                        ("chtemp", "<i2", (5, )),
                        ("reserved13", "<i2"),
                        ("bptemp", "<i2", (5, )),
                        ("reserved14", "<i2"),
                        ("mhtemp", "<i2", (5, )),
                        ("reserved15", "<i2"),
                        ("adcontemp", "<i2", (5, )),
                        ("reserved16", "<i2"),
                        ("d4bvolt", "<i2", (5, )),
                        ("reserved17", "<i2"),
                        ("d5bvolt", "<i2", (5, )),
                        ("reserved18", "<i2"),
                        ("bbtempchn3B", "<i2", (5, )),
                        ("reserved19", "<i2"),
                        ("bbtempchn4", "<i2", (5, )),
                        ("reserved20", "<i2"),
                        ("bbtempchn5", "<i2", (5, )),
                        ("reserved21", "<i2"),
                        ("refvolt", "<i2", (5, )),
                        ("reserved22", "<i2"),
])

# AAPP 1b scanline

_SCANTYPE = np.dtype([("scnlin", "<i2"),
                      ("scnlinyr", "<i2"),
                      ("scnlindy", "<i2"),
                      ("clockdrift", "<i2"),
                      ("scnlintime", "<i4"),
                      ("scnlinbit", "<i2"),
                      ("filler0", "S10"),
                      ("qualind", "<i4"),
                      ("scnlinqual", "<i4"),
                      ("calqual", "<i2", (3, )),
                      ("cbiterr", "<i2"),
                      ("filler1", "S8"),
                      # Calibration
                      ("calvis", "<i4", (3, 3, 5)),
                      ("calir", "<i4", (3, 2, 3)),
                      ("filler2", "<i4", (3, )),
                      # Navigation
                      ("navstat", "<i4"),
                      ("attangtime", "<i4"),
                      ("rollang", "<i2"),
                      ("pitchang", "<i2"),
                      ("yawang", "<i2"),
                      ("scalti", "<i2"),
                      ("ang", "<i2", (51, 3)),
                      ("filler3", "<i2", (3, )),
                      ("pos", "<i4", (51, 2)),
                      ("filler4", "<i4", (2, )),
                      ("telem", "<i2", (103, )),
                      ("filler5", "<i2"),
                      ("hrpt", "<i2", (2048, 5)),
                      ("filler6", "<i4", (2, )),
                      # tip minor frame header
                      ("tipmfhd", "<i2", (7, 5)),
                      # cpu telemetry
                      ("cputel", "S6", (2, 5)),
                      ("filler7", "<i2", (67, )),
                      ])

class AAPP1b(object):
    """AAPP-level 1b data reader
    """
    def __init__(self, fname):
        self.filename = fname
        self.channels = dict([(i, None) for i in AVHRR_CHANNEL_NAMES])
        self.units = dict([(i, 'counts') for i in AVHRR_CHANNEL_NAMES])

        self._data = None
        self._header = None
        self._is3b = None
        self.lons = None
        self.lats = None

    def read(self):
        """Read the data.
        """
        tic = datetime.datetime.now()
        with open(self.filename, "rb") as fp_:
            header =  np.fromfile(fp_, dtype=_HEADERTYPE, count=1)
            fp_.seek(10664 * 2, 1)
            data = np.fromfile(fp_, dtype=_SCANTYPE)

        LOG.debug("Reading time " + str(datetime.datetime.now() - tic))
        self._header = header
        self._data = data

    def navigate(self):
        """Return the longitudes and latitudes of the scene.
        """
        tic = datetime.datetime.now()
        lons40km = self._data["pos"][:, :, 1] * 1e-4
        lats40km = self._data["pos"][:, :, 0] * 1e-4

        try:
            from geotiepoints import SatelliteInterpolator
        except ImportError:
            LOG.warning("Could not interpolate lon/lats, "
                        "python-geotiepoints missing.")
            self.lons, self.lats = lons40km, lats40km
        else:
            cols40km = np.arange(24, 2048, 40)
            cols1km = np.arange(2048)
            lines = lons40km.shape[0]
            rows40km = np.arange(lines)
            rows1km = np.arange(lines)

            along_track_order = 1
            cross_track_order = 3

            satint = SatelliteInterpolator((lons40km, lats40km),
                                           (rows40km, cols40km),
                                           (rows1km, cols1km),
                                           along_track_order,
                                           cross_track_order)
            self.lons, self.lats = satint.interpolate()
            LOG.debug("Navigation time " + str(datetime.datetime.now() - tic))

    def calibrate(self, chns=("1", "2", "3A", "3B", "4", "5"),
                  calibrate=1):
        """Calibrate the data
        """
        tic = datetime.datetime.now()

        if "1" in chns:
            self.channels['1'] = _vis_calibrate(self._data, 0, calibrate)
            if calibrate == 0:
                self.units['1'] = ''
            else:
                self.units['1'] = '%'
            
        if "2" in chns:
            self.channels['2'] = _vis_calibrate(self._data, 1, calibrate)
            if calibrate == 0:
                self.units['2'] = ''
            else:
                self.units['2'] = '%'

        if "3A" in chns or "3B" in chns:
            # Is it 3A or 3B:
            is3b = np.expand_dims(np.bitwise_and(
                np.right_shift(self._data['scnlinbit'], 0), 1) == 1, 1)
            self._is3b = is3b

        if "3A" in chns:
            ch3a = _vis_calibrate(self._data, 2, calibrate)
            self.channels['3A'] = np.ma.masked_array(ch3a, is3b * ch3a)
            if calibrate == 0:
                self.units['3A'] = ''		
            else:
                self.units['3A'] = '%'

        if "3B" in chns:
            ch3b = _ir_calibrate(self._header, self._data, 0, calibrate)
            self.channels['3B'] = np.ma.masked_array(ch3b, 
                                                     np.logical_or((is3b==False)
                                                                   * ch3b, 
                                                                   ch3b<0.1))
            if calibrate == 1:
                self.units['3B'] = 'K'
            elif calibrate == 2:
                self.units['3B'] = 'W*m-2*sr-1*cm ?'
            else:
                self.units['3B'] = ''

        if "4" in chns:
            self.channels['4'] = _ir_calibrate(self._header, 
                                               self._data, 1, calibrate)
            if calibrate == 1:
                self.units['4'] = 'K'
            elif calibrate == 2:
                self.units['4'] = 'W*m-2*sr-1*cm ?'
            else:
                self.units['4'] = ''


        if "5" in chns:
            self.channels['5'] = _ir_calibrate(self._header, 
                                               self._data, 2, calibrate)
            if calibrate == 1:
                self.units['5'] = 'K'
            elif calibrate == 2:
                self.units['5'] = 'W*m-2*sr-1*cm ?'
            else:
                self.units['5'] = ''

        LOG.debug("Calibration time " + str(datetime.datetime.now() - tic))
        

def _vis_calibrate(data, chn, calib_type):
    """Visible channel calibration only.
    *calib_type* = 0: Counts
    *calib_type* = 1: Reflectances
    *calib_type* = 2: Radiances
    """
    # Calibration count to albedo, the calibration is performed separately for
    # two value ranges.
    
    channel = data["hrpt"][:, :, chn].astype(np.float)
    if calib_type == 0:
        return channel

    if calib_type == 2:
        LOG.info("Radiances are not yet supported for " + 
                 "the VIS/NIR channels!")

    mask1 = channel <= np.expand_dims(data["calvis"][:, chn, 2, 4], 1)
    mask2 = channel > np.expand_dims(data["calvis"][:, chn, 2, 4], 1)

    channel[mask1] = (channel * np.expand_dims(data["calvis"][:, chn, 2, 0] * 
                                               1e-10, 1) + 
                      np.expand_dims(data["calvis"][:, chn, 2, 1] * 
                                     1e-7, 1))[mask1]

    channel[mask2] = (channel * np.expand_dims(data["calvis"][:, chn, 2, 2] * 
                                               1e-10, 1) +
                      np.expand_dims(data["calvis"][:, chn, 2, 3] * 
                                     1e-7, 1))[mask2]

    channel[channel < 0] = np.nan
    return np.ma.masked_array(channel, np.isnan(channel))


def _ir_calibrate(header, data, irchn, calib_type):
    """IR calibration
    *calib_type* = 0: Counts
    *calib_type* = 1: Reflectances
    *calib_type* = 2: Radiances
    """

    count = data['hrpt'][:, :, irchn + 2].astype(np.float)

    if calib_type == 0:
        return count

    ir_const_1 = 1.1910659e-5
    ir_const_2 = 1.438833

    k1_ = np.expand_dims(data['calir'][:, irchn, 0, 0] / 1.0e9, 1)
    k2_ = np.expand_dims(data['calir'][:, irchn, 0, 1] / 1.0e6, 1)
    k3_ = np.expand_dims(data['calir'][:, irchn, 0, 2] / 1.0e6, 1)

    # Central wavenumber:
    cwnum = header['radtempcnv'][0, irchn, 0]
    if irchn == 0:
        cwnum = cwnum/1.0e2
    else:
        cwnum = cwnum/1.0e3

    bandcor_2 = header['radtempcnv'][0, irchn, 1]/1e5
    bandcor_3 = header['radtempcnv'][0, irchn, 2]/1e6

    # Count to radiance conversion:
    rad = k1_ * count*count + k2_*count + k3_

    if calib_type == 2:
        return rad

    all_zero = np.logical_and(np.logical_and(np.equal(k1_, 0),
                                             np.equal(k2_, 0)),
                              np.equal(k3_, 0))    
    idx = np.indices((all_zero.shape[0],))
    suspect_line_nums = np.repeat(idx[0], all_zero[:, 0])
    if suspect_line_nums.any():
        LOG.info("Suspect scan lines: " + str(suspect_line_nums))

    t_planck = (ir_const_2*cwnum) / np.log(1 + ir_const_1*cwnum*cwnum*cwnum/rad)

    # Band corrections applied to t_planck to get correct
    # brightness temperature for channel:
    if bandcor_2 < 0: # Post AAPP-v4
        tb_ = bandcor_2 + bandcor_3 * t_planck
    else: # AAPP 1 to 4
        tb_ = (t_planck - bandcor_2) / bandcor_3

    tb_[tb_ <= 0] = np.nan
    # Data with count=0 are often related to erroneous (bad) lines, but in case
    # of saturation (channel 3b) count=0 can be observed and associated to a
    # real measurement. So we leave out this filtering to the user!
    # tb_[count == 0] = np.nan
    tb_[rad == 0] = np.nan
    return np.ma.masked_array(tb_, np.isnan(tb_))


def show(data, negate=False):
    """Show the stetched data.
    """
    import Image as pil
    data = np.array((data - data.min()) * 255.0 /
                    (data.max() - data.min()), np.uint8)
    if negate:
        data = 255 - data
    img = pil.fromarray(data)
    img.show()

CASES = {
    "avhrr": load_avhrr,
    }

if __name__ == "__main__":

    import sys
    from mpop.utils import debug_on

    debug_on()
    SCENE = AAPP1b(sys.argv[1])
    SCENE.read()
    SCENE.calibrate()
    SCENE.navigate()
    for i_ in AVHRR_CHANNEL_NAMES:
        data_ = SCENE.channels[i_]
        print >> sys.stderr, "%-3s" % i_, \
            "%6.2f%%" % (100.*(float(np.ma.count(data_))/data_.size)), \
            "%6.2f, %6.2f, %6.2f" % (data_.min(), data_.mean(), data_.max())    
    show(SCENE.channels['4'], negate=True)
