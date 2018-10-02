#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2010-2018 PyTroll Community

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Bybbroe <adam.dybbroe@smhi.se>
#   Sauli Joro <sauli.joro@eumetsat.int>

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

"""SEVIRI HRIT format reader.

References:
    MSG Level 1.5 Image Data FormatDescription

TODO:
- HRV navigation

"""

import logging
from datetime import datetime

import numpy as np

from pyresample import geometry

from satpy.readers.eum_base import (time_cds_short,
                                    recarray2dict)
from satpy.readers.hrit_base import (HRITFileHandler, ancillary_text,
                                     annotation_header, base_hdr_map,
                                     image_data_function)

from satpy.readers.msg_base import SEVIRICalibrationHandler
from satpy.readers.msg_base import (CHANNEL_NAMES, CALIB, SATNUM)

from satpy.readers.native_msg_hdr import (hrit_prologue, hrit_epilogue,
                                          impf_configuration)

logger = logging.getLogger('hrit_msg')

# MSG implementation:
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

msg_variable_length_headers = {
    image_segment_line_quality: 'image_segment_line_quality'}

msg_text_headers = {image_data_function: 'image_data_function',
                    annotation_header: 'annotation_header',
                    ancillary_text: 'ancillary_text'}

msg_hdr_map = base_hdr_map.copy()
msg_hdr_map.update({7: key_header,
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


class HRITMSGPrologueFileHandler(HRITFileHandler):
    """SEVIRI HRIT prologue reader.
    """

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super(HRITMSGPrologueFileHandler, self).__init__(filename, filename_info,
                                                         filetype_info,
                                                         (msg_hdr_map,
                                                          msg_variable_length_headers,
                                                          msg_text_headers))
        self.prologue = {}
        self.read_prologue()

        service = filename_info['service']
        if service == '':
            self.mda['service'] = '0DEG'
        else:
            self.mda['service'] = service

    def read_prologue(self):
        """Read the prologue metadata."""

        with open(self.filename, "rb") as fp_:
            fp_.seek(self.mda['total_header_length'])
            data = np.fromfile(fp_, dtype=hrit_prologue, count=1)
            self.prologue.update(recarray2dict(data))
            try:
                impf = np.fromfile(fp_, dtype=impf_configuration, count=1)[0]
            except IndexError:
                logger.info('No IMPF configuration field found in prologue.')
            else:
                self.prologue.update(recarray2dict(impf))


class HRITMSGEpilogueFileHandler(HRITFileHandler):
    """SEVIRI HRIT epilogue reader.
    """

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super(HRITMSGEpilogueFileHandler, self).__init__(filename, filename_info,
                                                         filetype_info,
                                                         (msg_hdr_map,
                                                          msg_variable_length_headers,
                                                          msg_text_headers))
        self.epilogue = {}
        self.read_epilogue()

        service = filename_info['service']
        if service == '':
            self.mda['service'] = '0DEG'
        else:
            self.mda['service'] = service

    def read_epilogue(self):
        """Read the epilogue metadata."""

        with open(self.filename, "rb") as fp_:
            fp_.seek(self.mda['total_header_length'])
            data = np.fromfile(fp_, dtype=hrit_epilogue, count=1)
            self.epilogue.update(recarray2dict(data))


class HRITMSGFileHandler(HRITFileHandler, SEVIRICalibrationHandler):
    """SEVIRI HRIT format reader
    """

    def __init__(self, filename, filename_info, filetype_info,
                 prologue, epilogue):
        """Initialize the reader."""
        super(HRITMSGFileHandler, self).__init__(filename, filename_info,
                                                 filetype_info,
                                                 (msg_hdr_map,
                                                  msg_variable_length_headers,
                                                  msg_text_headers))

        self.prologue = prologue.prologue
        self.epilogue = epilogue.epilogue
        self._filename_info = filename_info

        self._get_header()

    def _get_header(self):
        """Read the header info, and fill the metadata dictionary"""

        earth_model = self.prologue['GeometricProcessing']['EarthModel']
        self.mda['offset_corrected'] = earth_model['TypeOfEarthModel'] == 1
        b = (earth_model['NorthPolarRadius'] +
             earth_model['SouthPolarRadius']) / 2.0 * 1000
        self.mda['projection_parameters'][
            'a'] = earth_model['EquatorialRadius'] * 1000
        self.mda['projection_parameters']['b'] = b
        ssp = self.prologue['ImageDescription'][
            'ProjectionDescription']['LongitudeOfSSP']
        self.mda['projection_parameters']['SSP_longitude'] = ssp
        self.mda['projection_parameters']['SSP_latitude'] = 0.0
        self.platform_id = self.prologue["SatelliteStatus"][
            "SatelliteDefinition"]["SatelliteId"]
        self.platform_name = "Meteosat-" + SATNUM[self.platform_id]
        self.mda['platform_name'] = self.platform_name
        service = self._filename_info['service']
        if service == '':
            self.mda['service'] = '0DEG'
        else:
            self.mda['service'] = service
        self.channel_name = CHANNEL_NAMES[self.mda['spectral_channel_id']]

    @property
    def start_time(self):

        return self.epilogue['ImageProductionStats'][
            'ActualScanningSummary']['ForwardScanStart']

    @property
    def end_time(self):

        return self.epilogue['ImageProductionStats'][
            'ActualScanningSummary']['ForwardScanEnd']

    def get_xy_from_linecol(self, line, col, offsets, factors):
        """Get the intermediate coordinates from line & col.

        Intermediate coordinates are actually the instruments scanning angles.
        """
        loff, coff = offsets
        lfac, cfac = factors
        x__ = (col - coff) / cfac * 2**16
        y__ = - (line - loff) / lfac * 2**16

        return x__, y__

    def get_area_extent(self, size, offsets, factors, platform_height):
        """Get the area extent of the file."""
        nlines, ncols = size
        h = platform_height

        loff, coff = offsets
        loff -= nlines
        offsets = loff, coff
        # count starts at 1
        cols = 1 - 0.5
        lines = 1 - 0.5
        ll_x, ll_y = self.get_xy_from_linecol(-lines, cols, offsets, factors)

        cols += ncols
        lines += nlines
        ur_x, ur_y = self.get_xy_from_linecol(-lines, cols, offsets, factors)

        aex = (np.deg2rad(ll_x) * h, np.deg2rad(ll_y) * h,
               np.deg2rad(ur_x) * h, np.deg2rad(ur_y) * h)

        if not self.mda['offset_corrected']:
            xadj = 1500
            yadj = 1500
            aex = (aex[0] + xadj, aex[1] + yadj,
                   aex[2] + xadj, aex[3] + yadj)

        return aex

    def get_area_def(self, dsid):
        """Get the area definition of the band."""
        if dsid.name != 'HRV':
            return super(HRITMSGFileHandler, self).get_area_def(dsid)

        cfac = np.int32(self.mda['cfac'])
        lfac = np.int32(self.mda['lfac'])
        loff = np.float32(self.mda['loff'])

        a = self.mda['projection_parameters']['a']
        b = self.mda['projection_parameters']['b']
        h = self.mda['projection_parameters']['h']
        lon_0 = self.mda['projection_parameters']['SSP_longitude']

        nlines = int(self.mda['number_of_lines'])
        ncols = int(self.mda['number_of_columns'])

        segment_number = self.mda['segment_sequence_number']

        current_first_line = (segment_number -
                              self.mda['planned_start_segment_number']) * nlines
        bounds = self.epilogue['ImageProductionStats']['ActualL15CoverageHRV']

        upper_south_line = bounds[
            'LowerNorthLineActual'] - current_first_line - 1
        upper_south_line = min(max(upper_south_line, 0), nlines)

        lower_coff = (5566 - bounds['LowerEastColumnActual'] + 1)
        upper_coff = (5566 - bounds['UpperEastColumnActual'] + 1)

        lower_area_extent = self.get_area_extent((upper_south_line, ncols),
                                                 (loff, lower_coff),
                                                 (lfac, cfac),
                                                 h)

        upper_area_extent = self.get_area_extent((nlines - upper_south_line,
                                                  ncols),
                                                 (loff - upper_south_line,
                                                  upper_coff),
                                                 (lfac, cfac),
                                                 h)

        proj_dict = {'a': float(a),
                     'b': float(b),
                     'lon_0': float(lon_0),
                     'h': float(h),
                     'proj': 'geos',
                     'units': 'm'}

        lower_area = geometry.AreaDefinition(
            'some_area_name',
            "On-the-fly area",
            'geosmsg',
            proj_dict,
            ncols,
            upper_south_line,
            lower_area_extent)

        upper_area = geometry.AreaDefinition(
            'some_area_name',
            "On-the-fly area",
            'geosmsg',
            proj_dict,
            ncols,
            nlines - upper_south_line,
            upper_area_extent)

        area = geometry.StackedAreaDefinition(lower_area, upper_area)

        self.area = area.squeeze()
        return area

    def get_dataset(self, key, info):
        res = super(HRITMSGFileHandler, self).get_dataset(key, info)
        res = self.calibrate(res, key.calibration)
        res.attrs['units'] = info['units']
        res.attrs['wavelength'] = info['wavelength']
        res.attrs['standard_name'] = info['standard_name']
        res.attrs['platform_name'] = self.platform_name
        res.attrs['sensor'] = 'seviri'
        res.attrs['satellite_longitude'] = self.mda[
            'projection_parameters']['SSP_longitude']
        res.attrs['satellite_latitude'] = self.mda[
            'projection_parameters']['SSP_latitude']
        res.attrs['satellite_altitude'] = self.mda['projection_parameters']['h']
        return res

    def calibrate(self, data, calibration):
        """Calibrate the data."""
        tic = datetime.now()
        channel_name = self.channel_name

        if calibration == 'counts':
            res = data
        elif calibration in ['radiance', 'reflectance', 'brightness_temperature']:

            coeffs = self.prologue["RadiometricProcessing"]
            coeffs = coeffs["Level15ImageCalibration"]
            gain = coeffs['CalSlope'][self.mda['spectral_channel_id'] - 1]
            offset = coeffs['CalOffset'][self.mda['spectral_channel_id'] - 1]
            data = data.where(data > 0)
            res = self._convert_to_radiance(data.astype(np.float32), gain, offset)
            line_mask = self.mda['image_segment_line_quality']['line_validity'] >= 2
            line_mask &= self.mda['image_segment_line_quality']['line_validity'] <= 3
            line_mask &= self.mda['image_segment_line_quality']['line_radiometric_quality'] == 4
            line_mask &= self.mda['image_segment_line_quality']['line_geometric_quality'] == 4
            res *= np.choose(line_mask, [1, np.nan])[:, np.newaxis].astype(np.float32)

        if calibration == 'reflectance':
            solar_irradiance = CALIB[self.platform_id][channel_name]["F"]
            res = self._vis_calibrate(res, solar_irradiance)

        elif calibration == 'brightness_temperature':
            cal_type = self.prologue['ImageDescription'][
                'Level15ImageProduction']['PlannedChanProcessing'][self.mda['spectral_channel_id']]
            res = self._ir_calibrate(res, channel_name, cal_type)

        logger.debug("Calibration time " + str(datetime.now() - tic))
        return res


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
