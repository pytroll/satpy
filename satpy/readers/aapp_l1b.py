#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2012, 2013, 2014, 2015, 2016, 2017

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>
#   Nina Håkansson <nina.hakansson@smhi.se>
#   Oana Nicola <oananicola@yahoo.com>
#   Lars Ørum Rasmussen <ras@dmi.dk>
#   Panu Lahtinen <panu.lahtinen@fmi.fi>

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

Options for loading:

 - pre_launch_coeffs (False): use pre-launch coefficients if True, operational
   otherwise (if available).

http://research.metoffice.gov.uk/research/interproj/nwpsaf/aapp/
NWPSAF-MF-UD-003_Formats.pdf
"""

import logging
from datetime import datetime, timedelta

import numpy as np

from satpy.dataset import Dataset
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)

CHANNEL_NAMES = ['1', '2', '3a', '3b', '4', '5']

ANGLES = {'sensor_zenith_angle': 'satz',
          'solar_zenith_angle': 'sunz',
          'sun_sensor_azimuth_difference_angle': 'azidiff'}


class AVHRRAAPPL1BFile(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(AVHRRAAPPL1BFile, self).__init__(filename, filename_info,
                                               filetype_info)
        self.channels = {i: None for i in AVHRR_CHANNEL_NAMES}
        self.units = {i: 'counts' for i in AVHRR_CHANNEL_NAMES}

        self._data = None
        self._header = None
        self._is3b = None
        self._shape = None
        self.lons = None
        self.lats = None
        self.area = None
        self.read()

        self.sunz, self.satz, self.azidiff = None, None, None

    @property
    def start_time(self):
        return datetime(self._data['scnlinyr'][0], 1, 1) + timedelta(
            days=int(self._data['scnlindy'][0]) - 1,
            milliseconds=int(self._data['scnlintime'][0]))

    @property
    def end_time(self):
        return datetime(self._data['scnlinyr'][-1], 1, 1) + timedelta(
            days=int(self._data['scnlindy'][-1]) - 1,
            milliseconds=int(self._data['scnlintime'][-1]))

    def shape(self):
        # return self._data.shape
        return self._shape

    def get_dataset(self, key, info):
        """Get a dataset from the file."""

        if key.name in CHANNEL_NAMES:
            dataset = self.calibrate([key])[0]
            # dataset.info.update(info)
        elif key.name in ['longitude', 'latitude']:
            if self.lons is None or self.lats is None:
                self.navigate()
            if key.name == 'longitude':
                return Dataset(self.lons, id=key, **info)
            else:
                return Dataset(self.lats, id=key, **info)
        else:  # Get sun-sat angles
            if key.name in ANGLES:
                if isinstance(getattr(self, ANGLES[key.name]), np.ndarray):
                    dataset = Dataset(
                        getattr(self, ANGLES[key.name]),
                        copy=False)
                else:
                    dataset = self.get_angles(key.name)
            else:
                logger.exception(
                    "Not a supported sun-sensor viewing angle: %s", key.name)
                raise

        # TODO get metadata

        if not self._shape:
            self._shape = dataset.shape

        return dataset

    def read(self):
        """Read the data.
        """
        tic = datetime.now()
        with open(self.filename, "rb") as fp_:
            header = np.memmap(fp_, dtype=_HEADERTYPE, mode="r", shape=(1, ))
            data = np.memmap(fp_, dtype=_SCANTYPE, offset=22016, mode="r")

        logger.debug("Reading time %s", str(datetime.now() - tic))

        self._header = header
        self._data = data

    def get_angles(self, angle_id):
        """Get sun-satellite viewing angles"""

        tic = datetime.now()

        sunz40km = self._data["ang"][:, :, 0] * 1e-2
        satz40km = self._data["ang"][:, :, 1] * 1e-2
        azidiff40km = self._data["ang"][:, :, 2] * 1e-2

        try:
            from geotiepoints.interpolator import Interpolator
        except ImportError:
            logger.warning("Could not interpolate sun-sat angles, "
                           "python-geotiepoints missing.")
            self.sunz, self.satz, self.azidiff = sunz40km, satz40km, azidiff40km
        else:
            cols40km = np.arange(24, 2048, 40)
            cols1km = np.arange(2048)
            lines = sunz40km.shape[0]
            rows40km = np.arange(lines)
            rows1km = np.arange(lines)

            along_track_order = 1
            cross_track_order = 3

            satint = Interpolator(
                [sunz40km, satz40km, azidiff40km], (rows40km, cols40km),
                (rows1km, cols1km), along_track_order, cross_track_order)
            self.sunz, self.satz, self.azidiff = satint.interpolate()

            logger.debug("Interpolate sun-sat angles: time %s",
                         str(datetime.now() - tic))

        return Dataset(getattr(self, ANGLES[angle_id]), copy=False)

    def navigate(self):
        """Return the longitudes and latitudes of the scene.
        """
        tic = datetime.now()
        lons40km = self._data["pos"][:, :, 1] * 1e-4
        lats40km = self._data["pos"][:, :, 0] * 1e-4

        try:
            from geotiepoints import SatelliteInterpolator
        except ImportError:
            logger.warning("Could not interpolate lon/lats, "
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

            satint = SatelliteInterpolator(
                (lons40km, lats40km), (rows40km, cols40km), (rows1km, cols1km),
                along_track_order, cross_track_order)
            self.lons, self.lats = satint.interpolate()
            logger.debug("Navigation time %s", str(datetime.now() - tic))

    def calibrate(self,
                  dataset_ids,
                  pre_launch_coeffs=False,
                  calib_coeffs=None):
        """Calibrate the data
        """
        tic = datetime.now()

        if calib_coeffs is None:
            calib_coeffs = {}

        chns = dict((dataset_id.name, dataset_id)
                    for dataset_id in dataset_ids)

        res = []
        # FIXME this should be done in _vis_calibrate
        units = {'reflectance': '%',
                 'brightness_temperature': 'K',
                 'counts': '',
                 'radiance': 'W*m-2*sr-1*cm ?'}

        if ("3a" in chns or "3b" in chns) and self._is3b is None:
            # Is it 3a or 3b:
            is3b = np.expand_dims(
                np.bitwise_and(
                    np.right_shift(self._data['scnlinbit'], 0), 1) == 1, 1)
            self._is3b = np.repeat(is3b,
                                   self._data['hrpt'][0].shape[0], axis=1)

        for idx, name in enumerate(['1', '2', '3a']):
            if name in chns:
                coeffs = calib_coeffs.get('ch' + name)
                # FIXME data should be masked before calibration
                ds = Dataset(
                    _vis_calibrate(self._data,
                                   idx,
                                   chns[name].calibration,
                                   pre_launch_coeffs,
                                   coeffs,
                                   mask=(name == '3a' and self._is3b)),
                    units=units[chns[name].calibration],
                    id=chns[name],
                    **chns[name]._asdict())
                res.append(ds)

        for idx, name in enumerate(['3b', '4', '5']):
            if name in chns:
                ds = Dataset(
                    _ir_calibrate(self._header,
                                  self._data,
                                  idx,
                                  chns[name].calibration,
                                  mask=(name == '3b' and
                                        (np.logical_not(self._is3b)))),
                    units=units[chns[name].calibration],
                    id=chns[name],
                    **chns[name]._asdict())
                res.append(ds)

        logger.debug("Calibration time %s", str(datetime.now() - tic))

        return res


AVHRR_CHANNEL_NAMES = ("1", "2", "3a", "3b", "4", "5")

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
                        ("bbtempchn3b", "<i2", (5, )),
                        ("reserved19", "<i2"),
                        ("bbtempchn4", "<i2", (5, )),
                        ("reserved20", "<i2"),
                        ("bbtempchn5", "<i2", (5, )),
                        ("reserved21", "<i2"),
                        ("refvolt", "<i2", (5, )),
                        ("reserved22", "<i2"), ])

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
                      ("filler7", "<i2", (67, )), ])


def _vis_calibrate(data,
                   chn,
                   calib_type,
                   pre_launch_coeffs=False,
                   calib_coeffs=None,
                   mask=False):
    """Visible channel calibration only.

    *calib_type* in count, reflectance, radiance
    """
    # Calibration count to albedo, the calibration is performed separately for
    # two value ranges.

    if calib_type not in ['counts', 'radiance', 'reflectance']:
        raise ValueError('Calibration ' + calib_type + ' unknown!')

    arr = data["hrpt"][:, :, chn]
    channel = np.ma.array(arr.astype(np.float), mask=mask * arr)
    if calib_type == 'counts':
        return channel

    if calib_type == 'radiance':
        logger.info("Radiances are not yet supported for " +
                    "the VIS/NIR channels!")

    if pre_launch_coeffs:
        coeff_idx = 2
    else:
        # check that coeffs are valid
        if np.all(data["calvis"][:, chn, 0, 4] == 0):
            logger.info(
                "No valid operational coefficients, fall back to pre-launch")
            coeff_idx = 2
        else:
            coeff_idx = 0

    intersection = data["calvis"][:, chn, coeff_idx, 4]

    if calib_coeffs is not None:
        logger.info("Updating from external calibration coefficients.")
        # intersection = np.expand_dims
        slope1 = np.expand_dims(calib_coeffs[0], 1)
        intercept1 = np.expand_dims(calib_coeffs[1], 1)
        slope2 = np.expand_dims(calib_coeffs[2], 1)
        intercept2 = np.expand_dims(calib_coeffs[3], 1)
    else:
        slope1 = np.expand_dims(data["calvis"][:, chn, coeff_idx, 0] * 1e-10,
                                1)
        intercept1 = np.expand_dims(data["calvis"][:, chn, coeff_idx, 1] *
                                    1e-7, 1)
        slope2 = np.expand_dims(data["calvis"][:, chn, coeff_idx, 2] * 1e-10,
                                1)
        intercept2 = np.expand_dims(data["calvis"][:, chn, coeff_idx, 3] *
                                    1e-7, 1)

        if chn == 2:
            slope2[slope2 < 0] += 0.4294967296

    mask1 = channel <= np.expand_dims(intersection, 1)
    mask2 = channel > np.expand_dims(intersection, 1)

    channel[mask1] = (channel * slope1 + intercept1)[mask1]

    channel[mask2] = (channel * slope2 + intercept2)[mask2]

    channel[channel < 0] = np.nan
    return np.ma.masked_invalid(channel)


def _ir_calibrate(header, data, irchn, calib_type, mask=False):
    """IR calibration
    *calib_type* in brightness_temperature, radiance, count
    """

    count = data['hrpt'][:, :, irchn + 2].astype(np.float)

    if calib_type == 0:
        return count

    k1_ = np.expand_dims(data['calir'][:, irchn, 0, 0] / 1.0e9, 1)
    k2_ = np.expand_dims(data['calir'][:, irchn, 0, 1] / 1.0e6, 1)
    k3_ = np.expand_dims(data['calir'][:, irchn, 0, 2] / 1.0e6, 1)

    # Count to radiance conversion:
    rad = k1_ * count * count + k2_ * count + k3_

    all_zero = np.logical_and(
        np.logical_and(
            np.equal(k1_, 0), np.equal(k2_, 0)), np.equal(k3_, 0))
    idx = np.indices((all_zero.shape[0], ))
    suspect_line_nums = np.repeat(idx[0], all_zero[:, 0])
    if suspect_line_nums.any():
        logger.info("Suspicious scan lines: %s", str(suspect_line_nums))

    if calib_type == 2:
        return rad

    # Central wavenumber:
    cwnum = header['radtempcnv'][0, irchn, 0]
    if irchn == 0:
        cwnum = cwnum / 1.0e2
    else:
        cwnum = cwnum / 1.0e3

    bandcor_2 = header['radtempcnv'][0, irchn, 1] / 1e5
    bandcor_3 = header['radtempcnv'][0, irchn, 2] / 1e6

    # Count to radiance conversion:
    rad = k1_ * count * count + k2_ * count + k3_

    if calib_type == 2:
        return rad

    all_zero = np.logical_and(
        np.logical_and(
            np.equal(k1_, 0), np.equal(k2_, 0)), np.equal(k3_, 0))
    idx = np.indices((all_zero.shape[0], ))
    suspect_line_nums = np.repeat(idx[0], all_zero[:, 0])
    if suspect_line_nums.any():
        logger.info("Suspect scan lines: %s", str(suspect_line_nums))

    ir_const_1 = 1.1910659e-5
    ir_const_2 = 1.438833

    t_planck = (ir_const_2 * cwnum) / \
        np.log(1 + ir_const_1 * cwnum * cwnum * cwnum / rad)

    # Band corrections applied to t_planck to get correct
    # brightness temperature for channel:
    if bandcor_2 < 0:  # Post AAPP-v4
        tb_ = bandcor_2 + bandcor_3 * t_planck
    else:  # AAPP 1 to 4
        tb_ = (t_planck - bandcor_2) / bandcor_3

    # tb_[tb_ <= 0] = np.nan
    # Data with count=0 are often related to erroneous (bad) lines, but in case
    # of saturation (channel 3b) count=0 can be observed and associated to a
    # real measurement. So we leave out this filtering to the user!
    # tb_[count == 0] = np.nan
    # tb_[rad == 0] = np.nan
    tb_ = np.ma.masked_array(tb_, copy=False, mask=mask)
    tb_ = np.ma.masked_invalid(tb_, copy=False)
    if calib_type == 'brightness_temperature':
        tb_ = np.ma.masked_less(tb_, 0.1, copy=False)

    return tb_


if __name__ == '__main__':

    def norm255(a__):
        """normalize array to uint8.
        """
        arr = a__ * 1.0
        arr = (arr - arr.min()) * 255.0 / (arr.max() - arr.min())
        return arr.astype(np.uint8)

    def show(a__):
        """show array.
        """
        from PIL import Image
        Image.fromarray(norm255(a__), "L").show()

    import sys
    res = read_raw(sys.argv[1])
