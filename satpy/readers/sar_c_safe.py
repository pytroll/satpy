#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2019 Satpy developers
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
"""SAFE SAR-C reader.

This module implements a reader for Sentinel 1 SAR-C GRD (level1) SAFE format as
provided by ESA. The format is comprised of a directory containing multiple
files, most notably two measurement files in geotiff and a few xml files for
calibration, noise and metadata.

References:
  - *Level 1 Product Formatting*
    https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1-product-formatting

  - J. Park, A. A. Korosov, M. Babiker, S. Sandven and J. Won,
    *"Efficient Thermal Noise Removal for Sentinel-1 TOPSAR Cross-Polarization Channel,"*
    in IEEE Transactions on Geoscience and Remote Sensing, vol. 56, no. 3,
    pp. 1555-1565, March 2018.
    doi: `10.1109/TGRS.2017.2765248 <https://doi.org/10.1109/TGRS.2017.2765248>`_

"""

import logging
import xml.etree.ElementTree as ET
from functools import lru_cache
from threading import Lock

import numpy as np
import rasterio
import xarray as xr
from dask import array as da
from dask.base import tokenize
from xarray import DataArray

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler

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
        """Init the xml filehandler."""
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
    def read_azimuth_noise_array(elts):
        """Read the azimuth noise vectors.

        The azimuth noise is normalized per swath to account for gain
        differences between the swaths in EW mode.

        This is based on the this reference:
        J. Park, A. A. Korosov, M. Babiker, S. Sandven and J. Won,
        "Efficient Thermal Noise Removal for Sentinel-1 TOPSAR Cross-Polarization Channel,"
        in IEEE Transactions on Geoscience and Remote Sensing, vol. 56, no. 3,
        pp. 1555-1565, March 2018.
        doi: 10.1109/TGRS.2017.2765248
        """
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
    def interpolate_xml_array(data, low_res_coords, shape, chunks):
        """Interpolate arbitrary size dataset to a full sized grid."""
        xpoints, ypoints = low_res_coords

        return interpolate_xarray_linear(xpoints, ypoints, data, shape, chunks=chunks)

    def get_dataset(self, key, info):
        """Load a dataset."""
        if self._polarization != key["polarization"]:
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

        if key['name'].endswith('squared'):
            data **= 2

        data = self.interpolate_xml_array(data, low_res_coords, data.shape)

    @lru_cache(maxsize=10)
    def get_noise_correction(self, shape, chunks=None):
        """Get the noise correction array."""
        data_items = self.root.findall(".//noiseVector")
        data, low_res_coords = self.read_xml_array(data_items, 'noiseLut')
        if not data_items:
            data_items = self.root.findall(".//noiseRangeVector")
            data, low_res_coords = self.read_xml_array(data_items, 'noiseRangeLut')
            range_noise = self.interpolate_xml_array(data, low_res_coords, shape, chunks=chunks)
            data_items = self.root.findall(".//noiseAzimuthVector")
            data, low_res_coords = self.read_azimuth_noise_array(data_items)
            azimuth_noise = self.interpolate_xml_array(data, low_res_coords, shape, chunks=chunks)
            noise = range_noise * azimuth_noise
        else:
            noise = self.interpolate_xml_array(data, low_res_coords, shape, chunks=chunks)
        return noise

    @lru_cache(maxsize=10)
    def get_calibration(self, calibration, shape, chunks=None):
        """Get the calibration array."""
        calibration_name = calibration.name or 'gamma'
        if calibration_name == 'sigma_nought':
            calibration_name = 'sigmaNought'
        elif calibration_name == 'beta_nought':
            calibration_name = 'betaNought'
        data_items = self.root.findall(".//calibrationVector")
        data, low_res_coords = self.read_xml_array(data_items, calibration_name)
        return self.interpolate_xml_array(data, low_res_coords, shape, chunks=chunks)

    def get_calibration_constant(self):
        """Load the calibration constant."""
        return float(self.root.find('.//absoluteCalibrationConstant').text)

    @property
    def start_time(self):
        """Get the start time."""
        return self._start_time

    @property
    def end_time(self):
        """Get the end time."""
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


def intp(grid_x, grid_y, interpolator):
    """Interpolate."""
    return interpolator((grid_y, grid_x))


def interpolate_xarray_linear(xpoints, ypoints, values, shape, chunks=CHUNK_SIZE):
    """Interpolate linearly, generating a dask array."""
    from scipy.interpolate.interpnd import (LinearNDInterpolator,
                                            _ndim_coords_from_arrays)

    if isinstance(chunks, (list, tuple)):
        vchunks, hchunks = chunks
    else:
        vchunks, hchunks = chunks, chunks

    points = _ndim_coords_from_arrays(np.vstack((np.asarray(ypoints),
                                                 np.asarray(xpoints))).T)

    interpolator = LinearNDInterpolator(points, values)

    grid_x, grid_y = da.meshgrid(da.arange(shape[1], chunks=hchunks),
                                 da.arange(shape[0], chunks=vchunks))

    # workaround for non-thread-safe first call of the interpolator:
    interpolator((0, 0))
    res = da.map_blocks(intp, grid_x, grid_y, interpolator=interpolator)

    return DataArray(res, dims=('y', 'x'))


class SAFEGRD(BaseFileHandler):
    """Measurement file reader.

    The measurement files are in geotiff format and read using rasterio. For
    performance reasons, the reading adapts the chunk size to match the file's
    block size.
    """

    def __init__(self, filename, filename_info, filetype_info, calfh, noisefh):
        """Init the grd filehandler."""
        super(SAFEGRD, self).__init__(filename, filename_info,
                                      filetype_info)

        self._start_time = filename_info['start_time']
        self._end_time = filename_info['end_time']

        self._polarization = filename_info['polarization']

        self._mission_id = filename_info['mission_id']

        self.calibration = calfh
        self.noise = noisefh
        self.read_lock = Lock()

        self.filehandle = rasterio.open(self.filename, 'r', sharing=False)

    def get_dataset(self, key, info):
        """Load a dataset."""
        if self._polarization != key["polarization"]:
            return

        logger.debug('Reading %s.', key['name'])

        if key['name'] in ['longitude', 'latitude', 'altitude']:
            logger.debug('Constructing coordinate arrays.')
            arrays = dict()
            arrays['longitude'], arrays['latitude'], arrays['altitude'] = self.get_lonlatalts()

            data = arrays[key['name']]
            data.attrs.update(info)

        else:
            data = xr.open_rasterio(self.filehandle,
                                    chunks={'band': 1, 'x': CHUNK_SIZE, 'y': CHUNK_SIZE},
                                    lock=self.read_lock).squeeze()
            data = self._calibrate(data, key)
            data.attrs.update(info)
            data.attrs.update({'platform_name': self._mission_id})

            data = self._change_quantity(data, key['quantity'])

        return data

    @staticmethod
    def _change_quantity(data, quantity):
        """Change quantity to dB if needed."""
        if quantity == 'dB':
            data.data = 10 * np.log10(data.data)
            data.attrs['units'] = 'dB'
        else:
            data.attrs['units'] = '1'

        return data

    def _calibrate(self, data, key):
        """Calibrate the data."""
        chunks = CHUNK_SIZE
        logger.debug('Reading noise data.')
        noise = self.noise.get_noise_correction(data.shape, chunks=chunks).fillna(0)
        logger.debug('Reading calibration data.')
        cal = self.calibration.get_calibration(key['calibration'], data.shape, chunks=chunks)
        cal_constant = self.calibration.get_calibration_constant()
        logger.debug('Calibrating.')
        data = data.where(data > 0)
        data = data.astype(np.float64)
        dn = data * data
        data = dn - noise
        data = ((data + cal_constant) / (cal ** 2)).clip(min=0)

        return data

    @lru_cache(maxsize=2)
    def get_lonlatalts(self):
        """Obtain GCPs and construct latitude and longitude arrays.

        Args:
           band (gdal band): Measurement band which comes with GCP's
           array_shape (tuple) : The size of the data array
        Returns:
           coordinates (tuple): A tuple with longitude and latitude arrays
        """
        band = self.filehandle

        (xpoints, ypoints), (gcp_lons, gcp_lats, gcp_alts), (gcps, crs) = self.get_gcps()

        # FIXME: do interpolation on cartesian coordinates if the area is
        # problematic.

        longitudes = interpolate_xarray(xpoints, ypoints, gcp_lons, band.shape)
        latitudes = interpolate_xarray(xpoints, ypoints, gcp_lats, band.shape)
        altitudes = interpolate_xarray(xpoints, ypoints, gcp_alts, band.shape)

        longitudes.attrs['gcps'] = gcps
        longitudes.attrs['crs'] = crs
        latitudes.attrs['gcps'] = gcps
        latitudes.attrs['crs'] = crs
        altitudes.attrs['gcps'] = gcps
        altitudes.attrs['crs'] = crs

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
        """Get the start time."""
        return self._start_time

    @property
    def end_time(self):
        """Get the end time."""
        return self._end_time
