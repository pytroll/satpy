#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Martin Raspaud

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
"""SAFE SAR-C format."""

import logging
import os
import xml.etree.ElementTree as ET

import numpy as np
from osgeo import gdal
import dask.array as da
from xarray import DataArray
import xarray.ufuncs as xu
from dask.base import tokenize

from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE

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
        if x.tag in d and not isinstance(d[x.tag], list):
            d[x.tag] = [d[x.tag]]
            d[x.tag].append(dictify(x, False))
        else:
            d[x.tag] = dictify(x, False)
    return d


class SAFEXML(BaseFileHandler):
    """XML file reader for the SAFE format."""

    def __init__(self, filename, filename_info, filetype_info,
                 header_file=None):
        super(SAFEXML, self).__init__(filename, filename_info, filetype_info)

        self._start_time = filename_info['start_time']
        self._end_time = filename_info['end_time']
        self._polarization = filename_info['polarization']
        self.root = ET.parse(self.filename)
        self.hdr = {}
        if header_file is not None:
            self.hdr = header_file.get_metadata()

    def get_metadata(self):
        """Convert the xml metadata to dict."""
        return dictify(self.root.getroot())

    @staticmethod
    def read_xml_array(elts, variable_name):
        """Read an array from an xml elements *elts*."""
        y = []
        x = []
        data = []
        for elt in elts:
            newx = elt.find('pixel').text.split()
            y += [int(elt.find('line').text)] * len(newx)
            x += [int(val) for val in newx]
            data += [float(val)
                     for val in elt.find(variable_name).text.split()]

        return np.asarray(data), (x, y)

    @staticmethod
    def interpolate_xml_array(data, low_res_coords, shape):
        """Interpolate arbitrary size dataset to a full sized grid."""
        xpoints, ypoints = low_res_coords

        return interpolate_xarray_linear(xpoints, ypoints, data, shape)

    def get_dataset(self, key, info):
        """Load a dataset."""
        if self._polarization != key.polarization:
            return

        xml_items = info['xml_item']
        xml_tags = info['xml_tag']

        if not isinstance(xml_items, list):
            xml_items = [xml_items]
            xml_tags = [xml_tags]

        for xml_item, xml_tag in zip(xml_items, xml_tags):
            data_items = self.root.findall(".//" + xml_item)
            if not data_items:
                continue
            data, low_res_coords = self.read_xml_array(data_items, xml_tag)

        if key.name.endswith('squared'):
            data **= 2

        data = self.interpolate_xml_array(data, low_res_coords, data.shape)

    def get_noise_correction(self, shape):
        """Get the noise correction array."""
        data_items = self.root.findall(".//noiseVector")
        data, low_res_coords = self.read_xml_array(data_items, 'noiseLut')
        if not data_items:
            data_items = self.root.findall(".//noiseRangeVector")
            data, low_res_coords = self.read_xml_array(data_items, 'noiseRangeLut')
        return self.interpolate_xml_array(data, low_res_coords, shape)

    def get_calibration(self, name, shape):
        """Get the calibration array."""
        data_items = self.root.findall(".//calibrationVector")
        data, low_res_coords = self.read_xml_array(data_items, name)
        return self.interpolate_xml_array(data ** 2, low_res_coords, shape)

    def get_calibration_constant(self):
        """Load the calibration constant."""
        return float(self.root.find('.//absoluteCalibrationConstant').text)

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time


def interpolate_slice(slice_rows, slice_cols, interpolator):
    """Interpolate the given slice of the larger array."""
    fine_rows = np.arange(slice_rows.start, slice_rows.stop, slice_rows.step)
    fine_cols = np.arange(slice_cols.start, slice_cols.stop, slice_cols.step)
    return interpolator(fine_cols, fine_rows)


def interpolate_xarray(xpoints, ypoints, values, shape, kind='cubic',
                       blocksize=CHUNK_SIZE):
    """Interpolate, generating a dask array."""
    vchunks = range(0, shape[0], blocksize)
    hchunks = range(0, shape[1], blocksize)

    token = tokenize(blocksize, xpoints, ypoints, values, kind, shape)
    name = 'interpolate-' + token

    from scipy.interpolate import interp2d
    interpolator = interp2d(xpoints, ypoints, values, kind=kind)

    dskx = {(name, i, j): (interpolate_slice,
                           slice(vcs, min(vcs + blocksize, shape[0])),
                           slice(hcs, min(hcs + blocksize, shape[1])),
                           interpolator)
            for i, vcs in enumerate(vchunks)
            for j, hcs in enumerate(hchunks)
            }

    res = da.Array(dskx, name, shape=list(shape),
                   chunks=(blocksize, blocksize),
                   dtype=values.dtype)
    return DataArray(res, dims=('y', 'x'))


def interpolate_xarray_linear(xpoints, ypoints, values, shape):
    """Interpolate linearly, generating a dask array."""
    from scipy.interpolate.interpnd import (LinearNDInterpolator,
                                            _ndim_coords_from_arrays)
    points = _ndim_coords_from_arrays(np.vstack((np.asarray(ypoints),
                                                 np.asarray(xpoints))).T)

    interpolator = LinearNDInterpolator(points, values)

    def intp(grid_x, grid_y, interpolator):
        return interpolator((grid_y, grid_x))

    grid_x, grid_y = da.meshgrid(da.arange(shape[1], chunks=CHUNK_SIZE),
                                 da.arange(shape[0], chunks=CHUNK_SIZE))
    # workaround for non-thread-safe first call of the interpolator:
    interpolator((0, 0))
    res = da.map_blocks(intp, grid_x, grid_y, interpolator=interpolator)

    return DataArray(res, dims=('y', 'x'))


class SAFEGRD(BaseFileHandler):
    """Measurement file reader."""

    def __init__(self, filename, filename_info, filetype_info, calfh, noisefh):
        super(SAFEGRD, self).__init__(filename, filename_info,
                                      filetype_info)

        self._start_time = filename_info['start_time']
        self._end_time = filename_info['end_time']

        self._polarization = filename_info['polarization']

        self.lats = None
        self.lons = None

        self.calibration = calfh
        self.noise = noisefh

        self.filename = filename
        self.get_gdal_filehandle()

    def get_gdal_filehandle(self):
        """Try to create the filehandle using gdal."""
        if os.path.exists(self.filename):
            self.filehandle = gdal.Open(self.filename)
            logger.debug("Loading dataset {}".format(self.filename))
        else:
            raise IOError("Path {} does not exist.".format(self.filename))

    def get_dataset(self, key, info):
        """Load a dataset."""
        if self._polarization != key.polarization:
            return

        logger.debug('Reading %s.', key.name)

        if key.name in ['longitude', 'latitude']:
            logger.debug('Constructing coordinate arrays.')

            if self.lons is None or self.lats is None:
                self.lons, self.lats = self.get_lonlats()

            if key.name == 'latitude':
                data = self.lats
            else:
                data = self.lons
            data.attrs = info

        else:
            data = self.read_band()
            logger.debug('Reading noise data.')

            noise = self.noise.get_noise_correction(data.shape)

            logger.debug('Reading calibration data.')

            cal = self.calibration.get_calibration('gamma', data.shape)
            cal_constant = self.calibration.get_calibration_constant()

            logger.debug('Calibrating.')

            data = data.astype(np.float64)
            data = (data * data + cal_constant - noise) / cal

            data = xu.sqrt(data.clip(min=0))

            data.attrs = info

            del noise, cal

            data.attrs['units'] = 'sigma'

        return data

    def read_band(self, blocksize=CHUNK_SIZE):
        """Read the band in blocks."""
        band = self.filehandle

        shape = band.RasterYSize, band.RasterXSize
        vchunks = range(0, shape[0], blocksize)
        hchunks = range(0, shape[1], blocksize)

        token = tokenize(blocksize, band)
        name = 'read_band-' + token

        dskx = {(name, i, j): (band.GetRasterBand(1).ReadAsArray,
                               hcs, vcs,
                               min(blocksize,  shape[1] - hcs),
                               min(blocksize,  shape[0] - vcs))
                for i, vcs in enumerate(vchunks)
                for j, hcs in enumerate(hchunks)
                }

        res = da.Array(dskx, name, shape=list(shape),
                       chunks=(blocksize, blocksize),
                       dtype=np.uint16)
        return DataArray(res, dims=('y', 'x'))

    def get_lonlats(self):
        """Obtain GCPs and construct latitude and longitude arrays.

        Args:
           band (gdal band): Measurement band which comes with GCP's
           array_shape (tuple) : The size of the data array
        Returns:
           coordinates (tuple): A tuple with longitude and latitude arrays
        """
        band = self.filehandle

        shape = band.RasterYSize, band.RasterXSize

        (xpoints, ypoints), (gcp_lons, gcp_lats) = self.get_gcps()

        # FIXME: do interpolation on cartesion coordinates if the area is
        # problematic.

        longitudes = interpolate_xarray(xpoints, ypoints, gcp_lons, shape)
        latitudes = interpolate_xarray(xpoints, ypoints, gcp_lats, shape)

        return longitudes, latitudes

    def get_gcps(self):
        """Read GCP from the GDAL band.

        Args:
           band (gdal band): Measurement band which comes with GCP's
           coordinates (tuple): A tuple with longitude and latitude arrays

        Returns:
           points (tuple): Pixel and Line indices 1d arrays
           gcp_coords (tuple): longitude and latitude 1d arrays

        """
        gcps = self.filehandle.GetGCPs()

        gcp_array = np.array(
            [(p.GCPLine, p.GCPPixel, p.GCPY, p.GCPX) for p in gcps])

        ypoints = np.unique(gcp_array[:, 0])
        xpoints = np.unique(gcp_array[:, 1])

        gcp_lats = gcp_array[:, 2].reshape(ypoints.shape[0], xpoints.shape[0])
        gcp_lons = gcp_array[:, 3].reshape(ypoints.shape[0], xpoints.shape[0])

        return (xpoints, ypoints), (gcp_lons, gcp_lats)

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time
