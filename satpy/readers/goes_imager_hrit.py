#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2014-2018 Pytroll developpers

# Author(s):

#   Andrew Brooks
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

"""GOES HRIT format reader
****************************

References:
      LRIT/HRIT Mission Specific Implementation, February 2012
      GVARRDL98.pdf
      05057_SPE_MSG_LRIT_HRI
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import dask.array as da

from pyresample import geometry

from satpy.readers.eum_base import (time_cds_short, recarray2dict)
from satpy.readers.hrit_base import (HRITFileHandler, ancillary_text,
                                     annotation_header, base_hdr_map,
                                     image_data_function)


class CalibrationError(Exception):
    pass


logger = logging.getLogger('hrit_goes')

# Geometric constants [meters]
EQUATOR_RADIUS = 6378169.00
POLE_RADIUS = 6356583.80
ALTITUDE = 35785831.00

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


sgs_time = np.dtype([('century', 'u1'),
                     ('year', 'u1'),
                     ('doy1', 'u1'),
                     ('doy_hours', 'u1'),
                     ('hours_mins', 'u1'),
                     ('mins_secs', 'u1'),
                     ('secs_msecs', 'u1'),
                     ('msecs', 'u1')])


def make_sgs_time(sgs_time_array):
    year = ((sgs_time_array['century'] >> 4) * 1000 +
            (sgs_time_array['century'] & 15) * 100 +
            (sgs_time_array['year'] >> 4) * 10 +
            (sgs_time_array['year'] & 15))
    doy = ((sgs_time_array['doy1'] >> 4) * 100 +
           (sgs_time_array['doy1'] & 15) * 10 +
           (sgs_time_array['doy_hours'] >> 4))
    hours = ((sgs_time_array['doy_hours'] & 15) * 10 +
             (sgs_time_array['hours_mins'] >> 4))
    mins = ((sgs_time_array['hours_mins'] & 15) * 10 +
            (sgs_time_array['mins_secs'] >> 4))
    secs = ((sgs_time_array['mins_secs'] & 15) * 10 +
            (sgs_time_array['secs_msecs'] >> 4))
    msecs = ((sgs_time_array['secs_msecs'] & 15) * 100 +
             (sgs_time_array['msecs'] >> 4) * 10 +
             (sgs_time_array['msecs'] & 15))
    return (datetime(int(year), 1, 1) +
            timedelta(days=int(doy - 1),
                      hours=int(hours),
                      minutes=int(mins),
                      seconds=int(secs),
                      milliseconds=int(msecs)))


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


gvar_float = '>i4'


def make_gvar_float(float_val):
    sign = 1
    if float_val < 0:
        float_val = -float_val
        sign = -1

    exp = (float_val >> 24) - 64
    mant = float_val & ((1 << 24) - 1)
    if mant == 0:
        return 0.
    res = sign * mant * 2.0**(-24 + exp * 4)
    return res


prologue = np.dtype([
    # common generic header
    ("CommonHeaderVersion", "u1"),
    ("Junk1", "u1", 3),
    ("NominalSGSProductTime", time_cds_short),
    ("SGSProductQuality", "u1"),
    ("SGSProductCompleteness", "u1"),
    ("SGSProductTimeliness", "u1"),
    ("SGSProcessingInstanceId", "u1"),
    ("BaseAlgorithmVersion", "S1", 16),
    ("ProductAlgorithmVersion", "S1", 16),
    # product header
    ("ImageProductHeaderVersion", "u1"),
    ("Junk2", "u1", 3),
    ("ImageProductHeaderLength", ">u4"),
    ("ImageProductVersion", "u1"),
    # first block-0
    ("SatelliteID", "u1"),
    ("SPSID", "u1"),
    ("IScan", "u1", 4),
    ("IDSub", "u1", 16),
    ("TCurr", sgs_time),
    ("TCHED", sgs_time),
    ("TCTRL", sgs_time),
    ("TLHED", sgs_time),
    ("TLTRL", sgs_time),
    ("TIPFS", sgs_time),
    ("TINFS", sgs_time),
    ("TISPC", sgs_time),
    ("TIECL", sgs_time),
    ("TIBBC", sgs_time),
    ("TISTR", sgs_time),
    ("TLRAN", sgs_time),
    ("TIIRT", sgs_time),
    ("TIVIT", sgs_time),
    ("TCLMT", sgs_time),
    ("TIONA", sgs_time),
    ("RelativeScanCount", '>u2'),
    ("AbsoluteScanCount", '>u2'),
    ("NorthernmostScanLine", '>u2'),
    ("WesternmostPixel", '>u2'),
    ("EasternmostPixel", '>u2'),
    ("NorthernmostFrameLine", '>u2'),
    ("SouthernmostFrameLine", '>u2'),
    ("0Pixel", '>u2'),
    ("0ScanLine", '>u2'),
    ("0Scan", '>u2'),
    ("SubSatScan", '>u2'),
    ("SubSatPixel", '>u2'),
    ("SubSatLatitude", gvar_float),
    ("SubSatLongitude", gvar_float),
    ("Junk4", "u1", 96),  # move to "word" 295
    ("IMCIdentifier", "S4"),
    ("Zeros", "u1", 12),
    ("ReferenceLongitude", gvar_float),
    ("ReferenceDistance",  gvar_float),
    ("ReferenceLatitude",  gvar_float)
])


class HRITGOESPrologueFileHandler(HRITFileHandler):

    """GOES HRIT format reader"""

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
            data = np.fromfile(fp_, dtype=prologue, count=1)
            self.prologue.update(recarray2dict(data))

        self.process_prologue()

    def process_prologue(self):
        """Reprocess prologue to correct types."""

        for key in ['TCurr', 'TCHED', 'TCTRL', 'TLHED', 'TLTRL', 'TIPFS',
                    'TINFS', 'TISPC', 'TIECL', 'TIBBC', 'TISTR', 'TLRAN',
                    'TIIRT', 'TIVIT', 'TCLMT', 'TIONA']:
            try:
                self.prologue[key] = make_sgs_time(self.prologue[key])
            except ValueError:
                self.prologue.pop(key, None)
                logger.debug("Invalid data for %s", key)

        for key in ['SubSatLatitude', "SubSatLongitude", "ReferenceLongitude",
                    "ReferenceDistance", "ReferenceLatitude"]:
            self.prologue[key] = make_gvar_float(self.prologue[key])


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

        sublon = self.prologue['SubSatLongitude']
        self.mda['projection_parameters']['SSP_longitude'] = sublon

        satellite_id = self.prologue['SatelliteID']
        self.platform_name = SPACECRAFTS[satellite_id]

    def get_dataset(self, key, info):
        """Get the data  from the files."""
        logger.debug("Getting raw data")
        res = super(HRITGOESFileHandler, self).get_dataset(key, info)

        self.mda['calibration_parameters'] = self._get_calibration_params()

        res = self.calibrate(res, key.calibration)
        new_attrs = info.copy()
        new_attrs.update(res.attrs)
        res.attrs = new_attrs
        res.attrs['platform_name'] = self.platform_name
        res.attrs['sensor'] = 'goes_imager'
        return res

    def _get_calibration_params(self):
        """Get the calibration parameters from the metadata."""
        params = {}
        idx_table = []
        val_table = []
        for elt in self.mda['image_data_function'].split(b'\r\n'):
            try:
                key, val = elt.split(b':=')
                try:
                    idx_table.append(int(key))
                    val_table.append(float(val))
                except ValueError:
                    params[key] = val
            except ValueError:
                pass
        params['indices'] = np.array(idx_table)
        params['values'] = np.array(val_table, dtype=np.float32)
        return params

    def calibrate(self, data, calibration):
        """Calibrate the data."""
        logger.debug("Calibration")
        tic = datetime.now()
        if calibration == 'counts':
            return data
        if calibration == 'reflectance':
            res = self._calibrate(data)
        elif calibration == 'brightness_temperature':
            res = self._calibrate(data)
        else:
            raise NotImplementedError("Don't know how to calibrate to " +
                                      str(calibration))

        logger.debug("Calibration time " + str(datetime.now() - tic))
        return res

    def _calibrate(self, data):
        """Calibrate *data*."""
        idx = self.mda['calibration_parameters']['indices']
        val = self.mda['calibration_parameters']['values']
        data.data = da.where(data.data == 0, np.nan, data.data)
        ddata = data.data.map_blocks(np.interp, idx, val, dtype=val.dtype)
        res = xr.DataArray(ddata,
                           dims=data.dims, attrs=data.attrs,
                           coords=data.coords)
        res = res.clip(min=0)
        units = {'percent': '%'}
        unit = self.mda['calibration_parameters'][b'_UNIT']
        res.attrs['units'] = units.get(unit, unit)
        return res

    def get_area_def(self, dsid):
        """Get the area definition of the band."""
        cfac = np.int32(self.mda['cfac'])
        lfac = np.int32(self.mda['lfac'])
        coff = np.float32(self.mda['coff'])
        loff = np.float32(self.mda['loff'])

        a = EQUATOR_RADIUS
        b = POLE_RADIUS
        h = ALTITUDE

        lon_0 = self.prologue['SubSatLongitude']

        nlines = int(self.mda['number_of_lines'])
        ncols = int(self.mda['number_of_columns'])

        loff = nlines - loff

        area_extent = self.get_area_extent((nlines, ncols),
                                           (loff, coff),
                                           (lfac, cfac),
                                           h)

        proj_dict = {'a': float(a),
                     'b': float(b),
                     'lon_0': float(lon_0),
                     'h': float(h),
                     'proj': 'geos',
                     'units': 'm'}

        area = geometry.AreaDefinition(
            'some_area_name',
            "On-the-fly area",
            'geosmsg',
            proj_dict,
            ncols,
            nlines,
            area_extent)

        self.area = area

        return area
