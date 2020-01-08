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
"""SAFE SAR-C reader
*********************

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

import numpy as np
import rasterio
from rasterio.windows import Window
import dask.array as da
from xarray import DataArray
from dask.base import tokenize
from threading import Lock

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

    def get_calibration(self, name, shape, chunks=None):
        """Get the calibration array."""
        data_items = self.root.findall(".//calibrationVector")
        data, low_res_coords = self.read_xml_array(data_items, name)
        return self.interpolate_xml_array(data, low_res_coords, shape, chunks=chunks)

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


def intp(grid_x, grid_y, interpolator):
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
        super(SAFEGRD, self).__init__(filename, filename_info,
                                      filetype_info)

        self._start_time = filename_info['start_time']
        self._end_time = filename_info['end_time']

        self._polarization = filename_info['polarization']

        self._mission_id = filename_info['mission_id']

        self.lats = None
        self.lons = None
        self.alts = None

        self.calibration = calfh
        self.noise = noisefh
        self.read_lock = Lock()

        self.filehandle = rasterio.open(self.filename, 'r', sharing=False)

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
            if calibration == 'sigma_nought':
                calibration = 'sigmaNought'
            elif calibration == 'beta_nought':
                calibration = 'betaNought'

            data = self.read_band()
            # chunks = data.chunks  # This seems to be slower for some reason
            chunks = CHUNK_SIZE
            logger.debug('Reading noise data.')
            noise = self.noise.get_noise_correction(data.shape, chunks=chunks).fillna(0)

            logger.debug('Reading calibration data.')

            cal = self.calibration.get_calibration(calibration, data.shape, chunks=chunks)
            cal_constant = self.calibration.get_calibration_constant()

            logger.debug('Calibrating.')
            data = data.where(data > 0)
            data = data.astype(np.float64)
            dn = data * data
            data = ((dn - noise).clip(min=0) + cal_constant)

            data = (np.sqrt(data) / cal).clip(min=0)
            data.attrs.update(info)

            del noise, cal

            data.attrs.update({'platform_name': self._mission_id})

            data.attrs['units'] = calibration

        return data

    def read_band_blocks(self, blocksize=CHUNK_SIZE):
        """Read the band in native blocks."""
        # For sentinel 1 data, the block are 1 line, and dask seems to choke on that.
        band = self.filehandle

        shape = band.shape
        token = tokenize(blocksize, band)
        name = 'read_band-' + token
        dskx = dict()
        if len(band.block_shapes) != 1:
            raise NotImplementedError('Bands with multiple shapes not supported.')
        else:
            chunks = band.block_shapes[0]

        def do_read(the_band, the_window, the_lock):
            with the_lock:
                return the_band.read(1, None, window=the_window)

        for ji, window in band.block_windows(1):
            dskx[(name, ) + ji] = (do_read, band, window, self.read_lock)

        res = da.Array(dskx, name, shape=list(shape),
                       chunks=chunks,
                       dtype=band.dtypes[0])
        return DataArray(res, dims=('y', 'x'))

    def read_band(self, blocksize=CHUNK_SIZE):
        """Read the band in chunks."""
        band = self.filehandle

        shape = band.shape
        if len(band.block_shapes) == 1:
            total_size = blocksize * blocksize * 1.0
            lines, cols = band.block_shapes[0]
            if cols > lines:
                hblocks = cols
                vblocks = int(total_size / cols / lines)
            else:
                hblocks = int(total_size / cols / lines)
                vblocks = lines
        else:
            hblocks = blocksize
            vblocks = blocksize
        vchunks = range(0, shape[0], vblocks)
        hchunks = range(0, shape[1], hblocks)

        token = tokenize(hblocks, vblocks, band)
        name = 'read_band-' + token

        def do_read(the_band, the_window, the_lock):
            with the_lock:
                return the_band.read(1, None, window=the_window)

        dskx = {(name, i, j): (do_read, band,
                               Window(hcs, vcs,
                                      min(hblocks,  shape[1] - hcs),
                                      min(vblocks,  shape[0] - vcs)),
                               self.read_lock)
                for i, vcs in enumerate(vchunks)
                for j, hcs in enumerate(hchunks)
                }

        res = da.Array(dskx, name, shape=list(shape),
                       chunks=(vblocks, hblocks),
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

        (xpoints, ypoints), (gcp_lons, gcp_lats, gcp_alts), (gcps, crs) = self.get_gcps()

        # FIXME: do interpolation on cartesion coordinates if the area is
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
        return self._start_time

    @property
    def end_time(self):
        return self._end_time
