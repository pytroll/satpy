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
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.windows import Window
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
    def read_azimuth_array(elts):
        """Read the azimuth noise vectors."""
        y = []
        x = []
        data = []
        for elt in elts:
            first_pixel = int(elt.find('firstRangeSample').text)
            last_pixel = int(elt.find('lastRangeSample').text)
            lines = elt.find('line').text.split()
            lut = elt.find('noiseAzimuthLut').text.split()
            pixels = [first_pixel, last_pixel]
            swath = elt.find('swath').text
            corr = 1
            if swath == 'EW1':
                corr = 1.5
            if swath == 'EW4':
                corr = 1.2
            if swath == 'EW5':
                corr = 1.5

            for pixel in pixels:
                y += [int(val) for val in lines]
                x += [pixel] * len(lines)
                data += [float(val) * corr for val in lut]

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
            range_noise = self.interpolate_xml_array(data, low_res_coords, shape)
            data_items = self.root.findall(".//noiseAzimuthVector")
            data, low_res_coords = self.read_azimuth_array(data_items)
            azimuth_noise = self.interpolate_xml_array(data, low_res_coords, shape)
            noise = range_noise * azimuth_noise
        else:
            noise = self.interpolate_xml_array(data, low_res_coords, shape)
        return noise

    def get_calibration(self, name, shape):
        """Get the calibration array."""
        data_items = self.root.findall(".//calibrationVector")
        data, low_res_coords = self.read_xml_array(data_items, name)
        return self.interpolate_xml_array(data, low_res_coords, shape)

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
        self.alts = None

        self.calibration = calfh
        self.noise = noisefh

        self.filehandle = rasterio.open(self.filename, 'r')

    def get_dataset(self, key, info):
        """Load a dataset."""
        if self._polarization != key.polarization:
            return

        logger.debug('Reading %s.', key.name)

        if key.name in ['longitude', 'latitude']:
            logger.debug('Constructing coordinate arrays.')

            if self.lons is None or self.lats is None:
                self.lons, self.lats, self.alts = self.get_lonlatalts()

            if key.name == 'latitude':
                data = self.lats
            else:
                data = self.lons
            data.attrs.update(info)

        else:
            calibration = key.calibration or 'gamma'
            data = self.read_band()
            logger.debug('Reading noise data.')

            noise = self.noise.get_noise_correction(data.shape).fillna(0)

            logger.debug('Reading calibration data.')

            cal = self.calibration.get_calibration(calibration, data.shape)
            cal_constant = self.calibration.get_calibration_constant()

            logger.debug('Calibrating.')
            data = data.where(data > 0)
            data = data.astype(np.float64)
            dn = data * data
            data = ((dn - noise).clip(min=0) + cal_constant) / (cal * cal) # + 0.002

            data = np.sqrt(data.clip(min=0))

            data.attrs.update(info)

            del noise, cal

            data.attrs['units'] = calibration

        return data

    def read_band(self, blocksize=CHUNK_SIZE):
        """Read the band in blocks."""
        band = self.filehandle

        shape = band.shape
        vchunks = range(0, shape[0], blocksize)
        hchunks = range(0, shape[1], blocksize)

        token = tokenize(blocksize, band)
        name = 'read_band-' + token

        dskx = {(name, i, j): (band.read, 1, None,
                               Window(hcs, vcs,
                                      min(blocksize,  shape[1] - hcs),
                                      min(blocksize,  shape[0] - vcs)))
                for i, vcs in enumerate(vchunks)
                for j, hcs in enumerate(hchunks)
                }

        res = da.Array(dskx, name, shape=list(shape),
                       chunks=(blocksize, blocksize),
                       dtype=band.dtypes[0])
        return DataArray(res, dims=('y', 'x'))

    def get_lonlatalts(self):
        """Obtain GCPs and construct latitude and longitude arrays.

        Args:
           band (gdal band): Measurement band which comes with GCP's
           array_shape (tuple) : The size of the data array
        Returns:
           coordinates (tuple): A tuple with longitude and latitude arrays
        """
        band = self.filehandle

        (xpoints, ypoints), (gcp_lons, gcp_lats, gcp_alts), gcps = self.get_gcps()

        # FIXME: do interpolation on cartesion coordinates if the area is
        # problematic.

        longitudes = interpolate_xarray(xpoints, ypoints, gcp_lons, band.shape)
        latitudes = interpolate_xarray(xpoints, ypoints, gcp_lats, band.shape)
        altitudes = interpolate_xarray(xpoints, ypoints, gcp_alts, band.shape)

        longitudes.attrs['gcps'] = gcps
        latitudes.attrs['gcps'] = gcps
        altitudes.attrs['gcps'] = gcps

        return longitudes, latitudes, altitudes

    def get_gcps(self):
        """Read GCP from the GDAL band.

        Args:
           band (gdal band): Measurement band which comes with GCP's
           coordinates (tuple): A tuple with longitude and latitude arrays

        Returns:
           points (tuple): Pixel and Line indices 1d arrays
           gcp_coords (tuple): longitude and latitude 1d arrays

        """
        gcps = self.filehandle.gcps

        gcp_array = np.array([(p.row, p.col, p.x, p.y, p.z) for p in gcps[0]])

        ypoints = np.unique(gcp_array[:, 0])
        xpoints = np.unique(gcp_array[:, 1])

        gcp_lons = gcp_array[:, 2].reshape(ypoints.shape[0], xpoints.shape[0])
        gcp_lats = gcp_array[:, 3].reshape(ypoints.shape[0], xpoints.shape[0])
        gcp_alts = gcp_array[:, 4].reshape(ypoints.shape[0], xpoints.shape[0])

        return (xpoints, ypoints), (gcp_lons, gcp_lats, gcp_alts), gcps

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time
