#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""HRIT format reader.

References:
    ELECTRO-L GROUND SEGMENT MSU-GS INSTRUMENT,
      LRIT/HRIT Mission Specific Implementation, February 2012

"""

import logging
from datetime import datetime

import numpy as np
import xarray as xr

from satpy.readers._geos_area import get_area_definition, get_area_extent
from satpy.readers.hrit_base import (
    HRITFileHandler,
    ancillary_text,
    annotation_header,
    base_hdr_map,
    image_data_function,
    time_cds_short,
)

logger = logging.getLogger('hrit_electrol')


# goms implementation:
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

goms_hdr_map = base_hdr_map.copy()
goms_hdr_map.update({7: key_header,
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


prologue = np.dtype([('SatelliteStatus', satellite_status),
                     ('ImageAcquisition', image_acquisition, (10, )),
                     ('ImageCalibration', "<i4", (10, 1024))])


def recarray2dict(arr):
    """Change record array to a dictionary."""
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


class HRITGOMSPrologueFileHandler(HRITFileHandler):
    """GOMS HRIT format reader."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super(HRITGOMSPrologueFileHandler, self).__init__(filename, filename_info,
                                                          filetype_info,
                                                          (goms_hdr_map,
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

epilogue = np.dtype([('RadiometricProcessing', radiometric_processing, (10, )),
                     ('GeometricProcessing', geometric_processing, (10, ))])
# FIXME: Add rest of the epilogue


class HRITGOMSEpilogueFileHandler(HRITFileHandler):
    """GOMS HRIT format reader."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super(HRITGOMSEpilogueFileHandler, self).__init__(filename, filename_info,
                                                          filetype_info,
                                                          (goms_hdr_map,
                                                           goms_variable_length_headers,
                                                           goms_text_headers))
        self.epilogue = {}
        self.read_epilogue()

    def read_epilogue(self):
        """Read the prologue metadata."""
        with open(self.filename, "rb") as fp_:
            fp_.seek(self.mda['total_header_length'])
            data = np.fromfile(fp_, dtype=epilogue, count=1)[0]

            self.epilogue.update(recarray2dict(data))


C1 = 1.19104273e-5
C2 = 1.43877523


# Defined in MSG Level 1.5 Image Data Format Description
# https://www-cdn.eumetsat.int/files/2020-05/pdf_ten_05105_msg_img_data.pdf
SPACECRAFTS = {19001: "Electro-L N1",
               19002: "Electro-L N2",
               19003: "Electro-L N3"}


class HRITGOMSFileHandler(HRITFileHandler):
    """GOMS HRIT format reader."""

    def __init__(self, filename, filename_info, filetype_info,
                 prologue, epilogue):
        """Initialize the reader."""
        super(HRITGOMSFileHandler, self).__init__(filename, filename_info,
                                                  filetype_info,
                                                  (goms_hdr_map,
                                                   goms_variable_length_headers,
                                                   goms_text_headers))
        self.prologue = prologue.prologue
        self.epilogue = epilogue.epilogue
        self.chid = self.mda['spectral_channel_id']
        sublon = self.epilogue['GeometricProcessing']['TGeomNormInfo']['SubLon']
        sublon = sublon[self.chid]
        self.mda['projection_parameters']['SSP_longitude'] = np.rad2deg(sublon)
        self.mda['orbital_parameters']['satellite_nominal_longitude'] = np.rad2deg(
            self.prologue['SatelliteStatus']['NominalLongitude'])
        satellite_id = self.prologue['SatelliteStatus']['SatelliteID']
        self.platform_name = SPACECRAFTS[satellite_id]

    def get_dataset(self, key, info):
        """Get the data  from the files."""
        res = super(HRITGOMSFileHandler, self).get_dataset(key, info)

        res = self.calibrate(res, key['calibration'])
        res.attrs['units'] = info['units']
        res.attrs['standard_name'] = info['standard_name']
        res.attrs['wavelength'] = info['wavelength']
        res.attrs['platform_name'] = self.platform_name
        res.attrs['sensor'] = 'msu-gs'
        res.attrs['orbital_parameters'] = {
            'satellite_nominal_longitude': self.mda['orbital_parameters']['satellite_nominal_longitude'],
            'satellite_nominal_latitude': 0.,
            'projection_longitude': self.mda['projection_parameters']['SSP_longitude'],
            'projection_latitude': 0.,
            'projection_altitude': 35785831.00
        }

        return res

    def calibrate(self, data, calibration):
        """Calibrate the data."""
        tic = datetime.now()
        if calibration == 'counts':
            res = data
        elif calibration in ['radiance', 'brightness_temperature']:
            res = self._calibrate(data)
        else:
            raise NotImplementedError("Don't know how to calibrate to " +
                                      str(calibration))

        res.attrs['standard_name'] = calibration
        res.attrs['calibration'] = calibration

        logger.debug("Calibration time " + str(datetime.now() - tic))
        return res

    @staticmethod
    def _getitem(block, lut):
        return lut[block]

    def _calibrate(self, data):
        """Visible/IR channel calibration."""
        lut = self.prologue['ImageCalibration'][self.chid]
        if abs(lut).max() > 16777216:
            lut = lut.astype(np.float64)
        else:
            lut = lut.astype(np.float32)
        lut /= 1000
        lut[0] = np.nan
        # Dask/XArray don't support indexing in 2D (yet).
        res = data.data.map_blocks(self._getitem, lut, dtype=lut.dtype)
        res = xr.DataArray(res, dims=data.dims,
                           attrs=data.attrs, coords=data.coords)
        res = res.where(data > 0)
        return res

    def get_area_def(self, dsid):
        """Get the area definition of the band."""
        pdict = {}
        pdict['cfac'] = np.int32(self.mda['cfac'])
        pdict['lfac'] = np.int32(self.mda['lfac'])
        pdict['coff'] = np.float32(self.mda['coff'])
        pdict['loff'] = np.float32(self.mda['loff'])

        pdict['a'] = 6378169.00
        pdict['b'] = 6356583.80
        pdict['h'] = 35785831.00
        pdict['scandir'] = 'N2S'

        pdict['ssp_lon'] = self.mda['projection_parameters']['SSP_longitude']

        pdict['nlines'] = int(self.mda['number_of_lines'])
        pdict['ncols'] = int(self.mda['number_of_columns'])

        pdict['loff'] = pdict['nlines'] - pdict['loff']

        pdict['a_name'] = 'geosgoms'
        pdict['a_desc'] = 'Electro-L/GOMS channel area'
        pdict['p_id'] = 'goms'

        area_extent = get_area_extent(pdict)
        area = get_area_definition(pdict, area_extent)

        self.area = area

        return area
