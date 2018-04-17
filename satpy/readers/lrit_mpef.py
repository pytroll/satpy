#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 Pytroll developers

# Author(s):

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

"""LRIT MPEF format reader.

References:
    MSG Ground Segment LRIT HRIT Mission Specific Implementation
"""

import logging

import numpy as np

from pyresample import geometry
from satpy.readers.hrit_base import (HRITFileHandler, base_hdr_map,
                                     time_cds_short)
from satpy.readers.hrit_msg import (segment_identification,
                                    image_segment_line_quality,
                                    msg_variable_length_headers,
                                    msg_text_headers)

logger = logging.getLogger('lrit_mpef')


gp_config_item_version = ">i4"

image_details = np.dtype([('Pad1', 'S2'),
                          ('ExpectedImageStart', time_cds_short),
                          ('ImageReceivedFlag', "u1"),
                          ('Pad2', 'S1'),
                          ('UsedImageStart', time_cds_short),
                          ('Pad3', 'S2'),
                          ('UsedImageEnd', time_cds_short)])

mpef_prod_hdr = np.dtype([('MPEF_File_Id', ">i2"),
                          ('MPEF_header_version', "u1"),
                          ('ManualDissAuthRequested', "u1"),
                          ('ManualDisseminationAuth', "u1"),
                          ('DisseminationAuth', 'u1'),
                          ('NominalTime', time_cds_short),
                          ('ProductQuality', "u1"),
                          ('ProductCompleteness', "u1"),
                          ('ProductTimeliness', "u1"),
                          ('InstanceID', "u1"),
                          ('ImagesUsed', image_details, 4),
                          ('BaseAlgorithVersion', gp_config_item_version),
                          ('ProductAlgorithmVersion', gp_config_item_version),
                          ('Filler', 'S52')])

prologue = np.dtype([('MPEF_Product_Header', mpef_prod_hdr)])

cth_hdr = np.dtype([('ProductHeaderVersion', "u1"),
                    ('Filler', 'S95')])

prologue_cth = np.dtype([('MPEF_Product_Header', mpef_prod_hdr),
                         ('MPEF_Product_Specific_Header', cth_hdr)])

prologue_map = {'LRIT_CTH_PRO': prologue_cth}


mpef_hdr_map = base_hdr_map.copy()
mpef_hdr_map.update({128: segment_identification,
                     129: image_segment_line_quality
                     })


def recarray2dict(arr):
    """Transform a recarray to dict."""
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


class LRITMPEFPrologueFileHandler(HRITFileHandler):
    """MPEF LRIT format reader."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super(LRITMPEFPrologueFileHandler, self).__init__(filename, filename_info,
                                                          filetype_info,
                                                          (mpef_hdr_map,
                                                           msg_variable_length_headers,
                                                           msg_text_headers))

        self.prologue = {}
        self.read_prologue(prologue_map[filetype_info['file_type']])

    def read_prologue(self, product_prologue):
        """Read the prologue metadata."""
        with open(self.filename, "rb") as fp_:
            fp_.seek(self.mda['total_header_length'])

            data = np.fromfile(fp_, dtype=product_prologue, count=1)[0]
            self.prologue.update(recarray2dict(data))

        self.process_prologue()

    def process_prologue(self):
        """Reprocess prologue to correct types."""
        pass


class LRITMPEFCTHFileHandler(HRITFileHandler):
    """LRIT MPEF CTH format reader."""

    def __init__(self, filename, filename_info, filetype_info,
                 prologue):
        """Initialize the reader."""
        super(LRITMPEFCTHFileHandler, self).__init__(filename, filename_info,
                                                     filetype_info,
                                                     (mpef_hdr_map,
                                                      msg_variable_length_headers,
                                                      msg_text_headers))
        self.prologue = prologue.prologue

        #
        # self.chid = self.mda['spectral_channel_id']
        # sublon = self.epilogue['GeometricProcessing']['TGeomNormInfo']['SubLon']
        # sublon = sublon[self.chid]
        # self.mda['projection_parameters']['SSP_longitude'] = np.rad2deg(sublon)
        # satellite_id = self.prologue['SatelliteStatus']['SatelliteID']
        # self.platform_name = SPACECRAFTS[satellite_id]

    def get_dataset(self, key, info, out=None,
                    xslice=slice(None), yslice=slice(None)):
        """Get the data  from the files."""
        res = super(LRITMPEFCTHFileHandler, self).get_dataset(key, info)

        # Do something here

        return res

    def get_area_def(self, dsid):
        """Get the area definition of the band."""

        cfac = np.int32(self.mda['cfac'])
        lfac = np.int32(self.mda['lfac'])
        coff = np.float32(self.mda['coff'])
        loff = np.float32(self.mda['loff'])

        a = 6378169.00
        b = 6356583.80
        h = 35785831.00

        lon_0 = self.mda['projection_parameters']['SSP_longitude']

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
