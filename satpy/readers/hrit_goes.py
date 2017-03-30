#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2014, 2015, 2016 Adam.Dybbroe

# Author(s):

#   Adam.Dybbroe <adam.dybbroe@smhi.se>
#   Cooke, Michael.C, UK Met Office
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

"""HRIT format reader.

References:
      LRIT/HRIT Mission Specific Implementation, February 2012
      GVARRDL98.pdf
      05057_SPE_MSG_LRIT_HRI
"""

import logging
from datetime import datetime, timedelta

import numpy as np

from pyresample import geometry
from satpy.readers.hrit_base import (HRITFileHandler, ancillary_text,
                                     annotation_header, base_hdr_map,
                                     image_data_function, make_time_cds_short,
                                     time_cds_short)


class CalibrationError(Exception):
    pass

logger = logging.getLogger('hrit_goes')


# goes implementation:
key_header = np.dtype([('key_number', 'u1'),
                       ('seed', '>f8')])

segment_identification = np.dtype([('GP_SC_ID', '>i2'),
                                   ('spectral_channel_id', '>i1'),
                                   ('segment_sequence_number', '>u2'),
                                   ('planned_start_segment_number', '>u2'),
                                   ('planned_end_segment_number', '>u2'),
                                   ('data_field_representation', '>i1')])

image_segment_line_quality = np.dtype([('line_number_in_grid', '>i4'),
                                       ('line_mean_acquisition',
                                        [('days', '>u2'),
                                         ('milliseconds', '>u4')]),
                                       ('line_validity', 'u1'),
                                       ('line_radiometric_quality', 'u1'),
                                       ('line_geometric_quality', 'u1')])

goms_variable_length_headers = {
    image_segment_line_quality: 'image_segment_line_quality'}

goms_text_headers = {image_data_function: 'image_data_function',
                     annotation_header: 'annotation_header',
                     ancillary_text: 'ancillary_text'}

goes_hdr_map = base_hdr_map.copy()
goes_hdr_map.update({7: key_header,
                     128: segment_identification,
                     129: image_segment_line_quality
                     })


orbit_coef = np.dtype([('StartTime', time_cds_short),
                       ('EndTime', time_cds_short),
                       ('X', '>f8', (8, )),
                       ('Y', '>f8', (8, )),
                       ('Z', '>f8', (8, )),
                       ('VX', '>f8', (8, )),
                       ('VY', '>f8', (8, )),
                       ('VZ', '>f8', (8, ))])

attitude_coef = np.dtype([('StartTime', time_cds_short),
                          ('EndTime', time_cds_short),
                          ('XofSpinAxis', '>f8', (8, )),
                          ('YofSpinAxis', '>f8', (8, )),
                          ('ZofSpinAxis', '>f8', (8, ))])

cuc_time = np.dtype([('coarse', 'u1', (4, )),
                     ('fine', 'u1', (3, ))])

time_cds_expanded = np.dtype([('days', '>u2'),
                              ('milliseconds', '>u4'),
                              ('microseconds', '>u2'),
                              ('nanoseconds', '>u2')])


def make_time_cds_expanded(tcds_array):
    return (datetime(1958, 1, 1) +
            timedelta(days=int(tcds_array['days']),
                      milliseconds=int(tcds_array['milliseconds']),
                      microseconds=float(tcds_array['microseconds'] +
                                         tcds_array['nanoseconds'] / 1000.)))

satellite_status = np.dtype([("TagType", "<u4"),
                             ("TagLength", "<u4"),
                             ("SatelliteID", "<u8"),
                             ("SatelliteName", "S256"),
                             ("NominalLongitude", "<f8"),
                             ("SatelliteCondition", "<u4"),
                             ("TimeOffset", "<f8")])

image_acquisition = np.dtype([("TagType", "<u4"),
                              ("TagLength", "<u4"),
                              ("Status", "<u4"),
                              ("StartDelay", "<i4"),
                              ("Cel", "<f8")])


# XXX arb
#prologue = np.dtype([('SatelliteStatus', satellite_status),
#                     ('ImageAcquisition', image_acquisition, (10, )),
#                     ('ImageCalibration', "<i4", (10, 1024))])
prologue = np.dtype([
  # common generic header
  ("CommonHeaderVersion", "u1"),
  ("Junk1", "u1", 3),
  ("NominalSGSProductTime", time_cds_short),
  ("SGSProductQuality", "u1"),
  ("SGSProductCompleteness", "u1"),
  ("SGSProductTimeliness", "u1"),
  ("SGSProcessingInstanceId", "u1"),
  ("BaseAlgorithmVersion", np.str_, 16),
  ("ProductAlgorithmVersion", np.str_, 16),
  # product header
  ("ImageProductHeaderVersion", "u1"),
  ("Junk2", "u1", 3),
  ("ImageProductHeaderLength", "<u4"),
  ("ImageProductVersion", "u1"),
  # first block-0
  ("SatelliteID", "u1"),
  ("Junk3", "u1", 173), # move to "word" 175
  ("SubSatLatitude",     ">f4"),
  ("SubSatLongitude",    ">f4"), # ">f4" seems better than "<f4". still wrong though. 
  ("Junk4", "u1", 112), # move to "word" 295
  ("ReferenceLongitude", ">f4"),
  ("ReferenceDistance",  ">f4"),
  ("ReferenceLatitude",  ">f4")
  ])


def recarray2dict(arr):
    res = {}
    for dtuple in arr.dtype.descr:
        key = dtuple[0]
        ntype = dtuple[1]
        data = arr[key]
        if isinstance(ntype, list):
            res[key] = recarray2dict(data)
        else:
            res[key] = data

    return res


class HRITGOESPrologueFileHandler(HRITFileHandler):

    """GOES HRIT format reader
    """

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super(HRITGOESPrologueFileHandler, self).__init__(filename, filename_info,
                                                          filetype_info,
                                                          (goes_hdr_map,
                                                           goms_variable_length_headers,
                                                           goms_text_headers))

        self.prologue = {}
        self.read_prologue()

    def read_prologue(self):
        """Read the prologue metadata."""
        with open(self.filename, "rb") as fp_:
            fp_.seek(self.mda['total_header_length'])
            data = np.fromfile(fp_, dtype=prologue, count=1)[0]

            self.prologue.update(recarray2dict(data))

        self.process_prologue()

    def process_prologue(self):
        """Reprocess prologue to correct types."""
        pass


radiometric_processing = np.dtype([("TagType", "<u4"),
                                   ("TagLength", "<u4"),
                                   ("RPSummary",
                                    [("Impulse", "<u4"),
                                     ("IsStrNoiseCorrection", "<u4"),
                                        ("IsOptic", "<u4"),
                                        ("IsBrightnessAligment", "<u4")]),
                                   ("OpticCorrection",
                                    [("Degree", "<i4"),
                                     ("A", "<f8", (16, ))]),
                                   ("RPQuality",
                                    [("EffDinRange", "<f8"),
                                     ("EathDarkening", "<f8"),
                                        ("Zone", "<f8"),
                                        ("Impulse", "<f8"),
                                        ("Group", "<f8"),
                                        ("DefectCount", "<u4"),
                                        ("DefectProcent", "<f8"),
                                        ("S_Noise_DT_Preflight", "<f8"),
                                        ("S_Noise_DT_Bort", "<f8"),
                                        ("S_Noise_DT_Video", "<f8"),
                                        ("S_Noise_DT_1_5", "<f8"),
                                        ("CalibrStability", "<f8"),
                                        ("TemnSKO", "<f8", (2, )),
                                        ("StructSKO", "<f8", (2, )),
                                        ("Struct_1_5", "<f8"),
                                        ("Zone_1_ 5", "<f8"),
                                        ("RadDif", "<f8")])])

geometric_processing = np.dtype([("TagType", "<u4"),
                                 ("TagLength", "<u4"),
                                 ("TGeomNormInfo",
                                  [("IsExist", "<u4"),
                                   ("IsNorm", "<u4"),
                                      ("SubLon", "<f8"),
                                      ("TypeProjection", "<u4"),
                                      ("PixInfo", "<f8", (4, ))]),
                                 ("SatInfo",
                                  [("TISO",
                                    [("T0", "<f8"),
                                     ("dT", "<f8"),
                                        ("ASb", "<f8"),
                                        ("Evsk", "<f8", (3, 3, 4)),
                                        ("ARx", "<f8", (4, )),
                                        ("ARy", "<f8", (4, )),
                                        ("ARz", "<f8", (4, )),
                                        ("AVx", "<f8", (4, )),
                                        ("AVy", "<f8", (4, )),
                                        ("AVz", "<f8", (4, ))]),
                                      ("Type", "<i4")]),
                                 ("TimeProcessing", "<f8"),
                                 ("ApriorAccuracy", "<f8"),
                                 ("RelativeAccuracy", "<f8", (2, ))])


C1 = 1.19104273e-5
C2 = 1.43877523

SPACECRAFTS = {
               # these are GP_SC_ID
               18007: "GOES-7",
               18008: "GOES-8",
               18009: "GOES-9",
               18010: "GOES-10",
               18011: "GOES-11",
               18012: "GOES-12",
               18013: "GOES-13",
               18014: "GOES-14",
               18015: "GOES-15",
               # these are in block-0
               7: "GOES-7",
               8: "GOES-8",
               9: "GOES-9",
               10: "GOES-10",
               11: "GOES-11",
               12: "GOES-12",
               13: "GOES-13",
               14: "GOES-14",
               15: "GOES-15"}

class HRITGOESFileHandler(HRITFileHandler):
    """GOES HRIT format reader."""

    def __init__(self, filename, filename_info, filetype_info,
                 prologue):
        """Initialize the reader."""
        super(HRITGOESFileHandler, self).__init__(filename, filename_info,
                                                  filetype_info,
                                                  (goes_hdr_map,
                                                   goms_variable_length_headers,
                                                   goms_text_headers))
        self.prologue = prologue.prologue
        self.chid = self.mda['spectral_channel_id']
        # XXX no epilogue
        #sublon = self.epilogue['GeometricProcessing']['TGeomNormInfo']['SubLon']
        #sublon = sublon[self.chid]
        sublon = self.prologue['SubSatLongitude'];
        #sublon = -75.0 # XXX arb hard-coded since extraction returns -0.0883789
        self.mda['projection_parameters']['SSP_longitude'] = sublon # degrees, so no need to np.rad2deg(sublon)
        
        #satellite_id = self.prologue['SatelliteStatus']['SatelliteID']
        #self.platform_name = SPACECRAFTS[satellite_id]
        satellite_id = self.prologue['SatelliteID']
        self.platform_name = SPACECRAFTS[satellite_id]
        #logger.debug("satellite_id  "+str(satellite_id))
        #logger.debug("platform_name "+self.platform_name)
        #logger.debug("SSP_longitude "+str(self.mda['projection_parameters']['SSP_longitude']))
        #logger.debug("SSP_distance  "+str(self.prologue['ReferenceDistance']))
        #logger.debug("SSP_latitude  "+str(self.prologue['SubSatLatitude']))


    def get_dataset(self, key, info, out=None,
                    xslice=slice(None), yslice=slice(None)):
        """Get the data  from the files."""
        logger.debug("hrit_goes/get_dataset pre")
        res = super(HRITGOESFileHandler, self).get_dataset(key, info, out,
                                                           xslice, yslice)

        logger.debug("hrit_goes/get_dataset post")
        if res is not None:
            out = res

        logger.debug("hrit_goes/get_dataset res")
        self.calibrate(out, key.calibration)
        out.info['units'] = info['units']
        out.info['standard_name'] = info['standard_name']
        out.info['platform_name'] = self.platform_name
        out.info['sensor'] = 'goes_imager'

    def calibrate(self, data, calibration):
        """Calibrate the data."""
        logger.debug("hrit_goes/calibrate")
        tic = datetime.now()

        if calibration == 'counts':
            return

        if calibration == 'radiance':
            self._vis_calibrate(data)
        elif calibration == 'brightness_temperature':
            self._ir_calibrate(data)
        else:
            raise NotImplementedError("Don't know how to calibrate to " +
                                      str(calibration))

        logger.debug("Calibration time " + str(datetime.now() - tic))

    def _vis_calibrate(self, data):
        """Visible channel calibration only."""
        lut = self.prologue['ImageCalibration'][self.chid] / 1000.0

        data.data[:] = lut[data.data.astype(np.uint16)]

    def _ir_calibrate(self, data):
        """IR calibration."""

        lut = self.prologue['ImageCalibration'][self.chid] / 1000.0

        data.data[:] = lut[data.data.astype(np.uint16)]


def show(data, negate=False):
    """Show the stretched data.
    """
    from PIL import Image as pil
    data = np.array((data - data.min()) * 255.0 /
                    (data.max() - data.min()), np.uint8)
    if negate:
        data = 255 - data
    img = pil.fromarray(data)
    img.show()


if __name__ == "__main__":

    # TESTFILE = ("/media/My Passport/HIMAWARI-8/HISD/Hsfd/" +
    #            "201502/07/201502070200/00/B13/" +
    #            "HS_H08_20150207_0200_B13_FLDK_R20_S0101.DAT")
    TESTFILE = ("/local_disk/data/himawari8/testdata/" +
                "HS_H08_20130710_0300_B13_FLDK_R20_S1010.DAT")
    #"HS_H08_20130710_0300_B01_FLDK_R10_S1010.DAT")
    SCENE = ahisf([TESTFILE])
    SCENE.read_band(TESTFILE)
    SCENE.calibrate(['13'])
    # SCENE.calibrate(['13'], calibrate=0)

    # print SCENE._data['13']['counts'][0].shape

    show(SCENE.channels['13'], negate=False)

    import matplotlib.pyplot as plt
    plt.imshow(SCENE.channels['13'])
    plt.colorbar()
    plt.show()
