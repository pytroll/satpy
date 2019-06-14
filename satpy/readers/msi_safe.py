#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2017 Satpy developers
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
"""SAFE MSI L1C reader.
"""

import logging

import glymur
import numpy as np
from xarray import DataArray
import dask.array as da
import xml.etree.ElementTree as ET
from pyresample import geometry
from dask import delayed

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


PLATFORMS = {'S2A': "Sentinel-2A",
             'S2B': "Sentinel-2B",
             'S2C': "Sentinel-2C",
             'S2D': "Sentinel-2D"}


class SAFEMSIL1C(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info, mda):
        super(SAFEMSIL1C, self).__init__(filename, filename_info,
                                         filetype_info)

        self._start_time = filename_info['observation_time']
        self._end_time = filename_info['observation_time']
        self._channel = filename_info['band_name']
        self._mda = mda
        self.platform_name = PLATFORMS[filename_info['fmission_id']]

    def get_dataset(self, key, info):
        """Load a dataset."""
        if self._channel != key.name:
            return

        logger.debug('Reading %s.', key.name)
        # FIXME: get this from MTD_MSIL1C.xml
        quantification_value = 10000.
        jp2 = glymur.Jp2k(self.filename)
        bitdepth = 0
        for seg in jp2.codestream.segment:
            try:
                bitdepth = max(bitdepth, seg.bitdepth[0])
            except AttributeError:
                pass

        jp2.dtype = (np.uint8 if bitdepth <= 8 else np.uint16)

        # Initialize the jp2 reader / doesn't work in a multi-threaded context.
        # jp2[0, 0]
        # data = da.from_array(jp2, chunks=CHUNK_SIZE) / quantification_value * 100

        data = da.from_delayed(delayed(jp2.read)(), jp2.shape, jp2.dtype)
        data = data.rechunk(CHUNK_SIZE) / quantification_value * 100

        proj = DataArray(data, dims=['y', 'x'])
        proj.attrs = info.copy()
        proj.attrs['units'] = '%'
        proj.attrs['platform_name'] = self.platform_name
        return proj

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._start_time

    def get_area_def(self, dsid):
        if self._channel != dsid.name:
            return
        return self._mda.get_area_def(dsid)


class SAFEMSIMDXML(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(SAFEMSIMDXML, self).__init__(filename, filename_info,
                                           filetype_info)
        self._start_time = filename_info['observation_time']
        self._end_time = filename_info['observation_time']
        self.root = ET.parse(self.filename)
        self.tile = filename_info['gtile_number']
        self.platform_name = PLATFORMS[filename_info['fmission_id']]

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._start_time

    def get_area_def(self, dsid):
        """Get the area definition of the dataset."""
        geocoding = self.root.find('.//Tile_Geocoding')
        epsg = geocoding.find('HORIZONTAL_CS_CODE').text
        rows = int(geocoding.find('Size[@resolution="' + str(dsid.resolution) + '"]/NROWS').text)
        cols = int(geocoding.find('Size[@resolution="' + str(dsid.resolution) + '"]/NCOLS').text)
        geoposition = geocoding.find('Geoposition[@resolution="' + str(dsid.resolution) + '"]')
        ulx = float(geoposition.find('ULX').text)
        uly = float(geoposition.find('ULY').text)
        xdim = float(geoposition.find('XDIM').text)
        ydim = float(geoposition.find('YDIM').text)
        area_extent = (ulx, uly + rows * ydim, ulx + cols * xdim, uly)
        area = geometry.AreaDefinition(
                    self.tile,
                    "On-the-fly area",
                    self.tile,
                    {'init': epsg},
                    cols,
                    rows,
                    area_extent)
        return area

    @staticmethod
    def _do_interp(minterp, xcoord, ycoord):
        interp_points2 = np.vstack((xcoord.ravel(), ycoord.ravel()))
        res = minterp(interp_points2)
        return res.reshape(xcoord.shape)

    def interpolate_angles(self, angles, resolution):
        # FIXME: interpolate in cartesian coordinates if the lons or lats are
        # problematic
        from geotiepoints.multilinear import MultilinearInterpolator

        geocoding = self.root.find('.//Tile_Geocoding')
        rows = int(geocoding.find('Size[@resolution="' + str(resolution) + '"]/NROWS').text)
        cols = int(geocoding.find('Size[@resolution="' + str(resolution) + '"]/NCOLS').text)

        smin = [0, 0]
        smax = np.array(angles.shape) - 1
        orders = angles.shape
        minterp = MultilinearInterpolator(smin, smax, orders)
        minterp.set_values(da.atleast_2d(angles.ravel()))

        x = da.arange(rows, dtype=angles.dtype, chunks=CHUNK_SIZE) / (rows-1) * (angles.shape[0] - 1)
        y = da.arange(cols, dtype=angles.dtype, chunks=CHUNK_SIZE) / (cols-1) * (angles.shape[1] - 1)
        xcoord, ycoord = da.meshgrid(x, y)
        return da.map_blocks(self._do_interp, minterp, xcoord, ycoord, dtype=angles.dtype, chunks=xcoord.chunks)

    def _get_coarse_dataset(self, key, info):
        """Get the coarse dataset refered to by `key` from the XML data."""
        angles = self.root.find('.//Tile_Angles')
        if key in ['solar_zenith_angle', 'solar_azimuth_angle']:
            elts = angles.findall(info['xml_tag'] + '/Values_List/VALUES')
            return np.array([[val for val in elt.text.split()] for elt in elts],
                            dtype=np.float)

        elif key in ['satellite_zenith_angle', 'satellite_azimuth_angle']:
            arrays = []
            elts = angles.findall(info['xml_tag'] + '[@bandId="1"]')
            for elt in elts:
                items = elt.findall(info['xml_item'] + '/Values_List/VALUES')
                arrays.append(np.array([[val for val in item.text.split()] for item in items],
                                       dtype=np.float))
            return np.nanmean(np.dstack(arrays), -1)
        else:
            return

    def get_dataset(self, key, info):
        """Get the dataset refered to by `key`."""

        angles = self._get_coarse_dataset(key, info)
        if angles is None:
            return

        # Fill gaps at edges of swath
        darr = DataArray(angles, dims=['y', 'x'])
        darr = darr.bfill('x')
        darr = darr.ffill('x')
        angles = darr.data

        res = self.interpolate_angles(angles, key.resolution)

        proj = DataArray(res, dims=['y', 'x'])
        proj.attrs = info.copy()
        proj.attrs['units'] = 'degrees'
        proj.attrs['platform_name'] = self.platform_name
        return proj
