#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 Trygve Aspenes

# Author(s):

#   Trygve Aspenes <trygveas@met.no>

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
"""SAFE SAR L2 OCN format."""

import logging
import os
import xml.etree.ElementTree as ET

from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def dictify(r, root=True):
    """Convert an ElementTree into a dict."""
    if root:
        return {r.tag: dictify(r, False)}
    d = {}
    if r.text and r.text.strip():
        try:
            return int(r.text)
        except ValueError:
            try:
                return float(r.text)
            except ValueError:
                return r.text
    for x in r.findall("./*"):
        print x, x.tag
        if x.tag in d and not isinstance(d[x.tag], list):
            d[x.tag] = [d[x.tag]]
            d[x.tag].append(dictify(x, False))
        else:
            d[x.tag] = dictify(x, False)
    return d


class SAFEXML(BaseFileHandler):
    """XML file reader for the SAFE format."""

    def __init__(self, filename, filename_info, filetype_info):
        print 'SAFEXML init'
        super(SAFEXML, self).__init__(filename, filename_info, filetype_info)

        #self._start_time = filename_info['fstart_time']
        #self._end_time = filename_info['fend_time']
        #self._polarization = filename_info['fpolarization']
        self.root = ET.parse(self.filename)
        #rt = self.root.getroot()
        #for coordinates in rt.findall('gml:coordinates'):
        #    print coordinates
        #print 'After coordinates'
        #print dictify(self.root.getroot())
        #self.hdr = {}
        #if header_file is not None:
        #    self.hdr = header_file.get_metadata()
        #    print 'self.hdr', self.hdr
        print "SAFEXML END INIT"

    def get_metadata(self):
        """Convert the xml metadata to dict."""
        print "get_metadata"
        return dictify(self.root.getroot())

    def get_dataset(self, key, info):
        print "get_dataset XML"
        return

        #    @property
#    def start_time(self):
#        return self._start_time

#    @property
#    def end_time(self):
#        return self._end_time


class SAFENC(BaseFileHandler):
    """Measurement file reader."""

    def __init__(self, filename, filename_info, filetype_info, manifest_fh):
        print "INIT SAFENC"
        super(SAFENC, self).__init__(filename, filename_info,
                                      filetype_info)

        self.mainfest = manifest_fh
        print "manifest_fh ", manifest_fh
        self.manifest.get_metadata()

        self._start_time = filename_info['start_time']
        self._end_time = filename_info['end_time']

        self._polarization = filename_info['polarization']

        self.lats = None
        self.lons = None
        self._shape = None
        self.area = None
    
        self.nc = xr.open_dataset(filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  chunks={'owiAzSize': CHUNK_SIZE,
                                          'owiRaSize': CHUNK_SIZE})
        self.nc = self.nc.rename({'owiAzSize': 'x'})
        self.nc = self.nc.rename({'owiRaSize': 'y'})
        print self.nc
        print self.nc['owiWindDirection']
        self.filename = filename
        print "END INIT"
        #self.get_gdal_filehandle()

    def get_dataset(self, key, info):
        """Load a dataset."""
        logger.debug("REader %s %s",key, info)
        #if self._polarization != key.polarization:
        #    return

        logger.debug('Reading keyname %s.', key.name)
        if key.name in ['owiLat', 'owiLon']:
            logger.debug('Constructing coordinate arrays ll.')

            if self.lons is None or self.lats is None:
                self.lons = self.nc['owiLon']
                self.lats = self.nc['owiLat']

            if key.name == 'owiLat':
                res = self.lats
            else:
                res = self.lons
            res.attrs = info
        else:
            logger.debug("Read data")
            res = self.nc[key.name]
            res = xr.DataArray(res, dims=['y', 'x'])
            res.attrs.update(info)
            if '_FillValue' in res.attrs:
                res = res.where(res != res.attrs['_FillValue'])
                res.attrs['_FillValue'] = np.nan

            
            print "DATA:", self.nc[key.name]
            print "END"

        #print self.nc.attrs
        if 'missionName' in self.nc.attrs:
            res.attrs.update({'platform_name': self.nc.attrs['missionName']})

        print "res.shape: ",res.shape
        if not self._shape:
            self._shape = res.shape

        return res

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    def get_area_def(self, ds_id):
        data = self[ds_id.name]

        proj_dict = {
            'proj': 'latlong',
            'datum': 'WGS84',
            'ellps': 'WGS84',
            'no_defs': True
        }

        area_extent = [data.attrs.get('ProjectionMinLongitude'), data.attrs.get('ProjectionMinLatitude'),
                       data.attrs.get('ProjectionMaxLongitude'), data.attrs.get('ProjectionMaxLatitude')]

        area = geometry.AreaDefinition(
            'sar_ocn_area',
            'name_of_proj',
            'id_of_proj',
            proj_dict,
            int(self.filename_info['dim0']),
            int(self.filename_info['dim1']),
            np.asarray(area_extent)
        )

        return area
