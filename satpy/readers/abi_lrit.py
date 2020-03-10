#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2018 Satpy developers
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
"""GOES-R ABI LRIT format reader.

References:
      LRIT Receiver Specifications

"""

import logging
from datetime import datetime

import numpy as np
import xarray as xr
import dask.array as da

from pyresample import geometry

from satpy.readers.hrit_base import (HRITFileHandler, ancillary_text,
                                     annotation_header, base_hdr_map,
                                     image_data_function)


class CalibrationError(Exception):
    """Dummy error-class."""

    pass


logger = logging.getLogger(__name__)

# Geometric constants [meters]
EQUATOR_RADIUS = 6378169.00
POLE_RADIUS = 6356583.80
ALTITUDE = 35785831.00

# goes implementation:
key_header = np.dtype([('key_number', 'u1'),
                       ('seed', '>f8')])

segment_identification = np.dtype([('image_identifier', '>u2'),
                                   ('segment_sequence_number', '>u2'),
                                   ('start_column_of_segment', '>u2'),
                                   ('start_line_of_segment', '>u2'),
                                   ('max_segment', '>u2'),
                                   ('max_column', '>u2'),
                                   ('max_row', '>u2')])

noaa_lrit_header = np.dtype([('agency_signature', '|S4'),
                             ('product_id', '>u2'),
                             ('product_subid', '>u2'),
                             ('parameter', '>u2'),
                             ('noaa_specific_compression', 'u1')])

header_structure_record = np.dtype([('header_structure', '|S1')])

rice_compression = np.dtype([('flags', '>u2'),
                             ('pixels_per_block', 'u1'),
                             ('scan_lines_per_packet', 'u1')])


goms_variable_length_headers = {
}

goms_text_headers = {image_data_function: 'image_data_function',
                     annotation_header: 'annotation_header',
                     ancillary_text: 'ancillary_text',
                     header_structure_record: 'header_structure_record',
                     }

goes_hdr_map = base_hdr_map.copy()
goes_hdr_map.update({7: key_header,
                     128: segment_identification,
                     129: noaa_lrit_header,
                     130: header_structure_record,
                     131: rice_compression,
                     })

SPACECRAFTS = {
    b'G16': "GOES-16",
    b'G17': 'GOES-17',
}


def anc2dict(txt):
    """Make ancillary text to dict."""
    items = txt.split(b';')
    res = {}
    for item in items:
        key, val = item.split(b'=')
        key = key.strip()
        val = val.strip()
        res[key] = val
    return res


class LRITGOESFileHandler(HRITFileHandler):
    """GOES HRIT format reader."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super(LRITGOESFileHandler, self).__init__(filename, filename_info,
                                                  filetype_info,
                                                  (goes_hdr_map,
                                                   goms_variable_length_headers,
                                                   goms_text_headers))

        self.flags = self.mda.get('flags', 49)
        self.pixels_per_block = self.mda.get('pixel_per_block', 16)
        self.scan_lines_per_packet = self.mda.get('scan_lines_per_packet', 1)
        self.rice = False

        anc = anc2dict(self.mda['ancillary_text'])

        self.platform_name = SPACECRAFTS[anc[b'Satellite']]

    def decompress(self, hdr_info):
        """Decompress the file."""
        noaa_compression = self.mda.get('noaa_specific_compression', 0)
        if noaa_compression == 1:
            # Rice compression
            self.rice = True
            logger.warning('Ignoring Rice Compression flag!')
        else:
            raise NotImplementedError("Don't know how do decompress this file!")

    def get_dataset(self, key, info):
        """Get the data  from the files."""
        logger.debug("Getting raw data")
        res = super(LRITGOESFileHandler, self).get_dataset(key, info)
        self.mda['calibration_parameters'] = self._get_calibration_params()

        res = self.calibrate(res, key.calibration)
        new_attrs = info.copy()
        new_attrs.update(res.attrs)
        res.attrs = new_attrs
        res.attrs['platform_name'] = self.platform_name
        res.attrs['sensor'] = 'goes_imager'
        res.attrs['orbital_parameters'] = {'projection_longitude': self.mda['projection_parameters']['SSP_longitude'],
                                           'projection_latitude': 0.0,
                                           'projection_altitude': ALTITUDE}
        return res

    def _get_calibration_params(self):
        """Get the calibration parameters from the metadata."""
        params = {}
        idx_table = []
        val_table = []
        for elt in self.mda['image_data_function'].split(b'\n'):
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
        data.data = da.where(data.data == 255, np.nan, data.data)
        ddata = data.data.map_blocks(np.interp, idx, val, dtype=val.dtype)
        res = xr.DataArray(ddata,
                           dims=data.dims, attrs=data.attrs,
                           coords=data.coords)
        res = res.clip(min=0)
        units = {b'percent': '%', b'degree Kelvin': 'K'}
        unit = self.mda['calibration_parameters'][b'_UNIT']
        if unit == b'1':
            unit = b'percent'
            res *= 100
        res.attrs['units'] = units.get(unit, unit)
        return res

    def get_area_def(self, dsid):
        """Get the area definition of the band."""
        raise NotImplementedError
        cfac = np.int32(self.mda['cfac'])
        lfac = np.int32(self.mda['lfac'])
        coff = np.float32(self.mda['coff'])
        loff = np.float32(self.mda['loff'])

        a = EQUATOR_RADIUS
        b = POLE_RADIUS
        h = ALTITUDE

        # TODO: get this from the metadata
        lon_0 = -75

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
