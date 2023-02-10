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

import functools
import logging
from threading import Lock

import defusedxml.ElementTree as ET
import numpy as np
import rasterio
import rioxarray
import xarray as xr
from dask import array as da
from dask.base import tokenize
from xarray import DataArray

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


def dictify(r):
    """Convert an ElementTree into a dict."""
    return {r.tag: _dictify(r)}


def _dictify(r):
    """Convert an xml element to dict."""
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
            d[x.tag].append(_dictify(x))
        else:
            d[x.tag] = _dictify(x)
    return d


def _get_calibration_name(calibration):
    """Get the proper calibration name."""
    calibration_name = getattr(calibration, "name", calibration) or 'gamma'
    if calibration_name == 'sigma_nought':
        calibration_name = 'sigmaNought'
    elif calibration_name == 'beta_nought':
        calibration_name = 'betaNought'
    return calibration_name


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
        else:
            self.hdr = self.get_metadata()
        self._image_shape = (self.hdr['product']['imageAnnotation']['imageInformation']['numberOfLines'],
                             self.hdr['product']['imageAnnotation']['imageInformation']['numberOfSamples'])

    def get_metadata(self):
        """Convert the xml metadata to dict."""
        return dictify(self.root.getroot())

    @property
    def start_time(self):
        """Get the start time."""
        return self._start_time

    @property
    def end_time(self):
        """Get the end time."""
        return self._end_time


class SAFEXMLAnnotation(SAFEXML):
    """XML file reader for the SAFE format, Annotation file."""

    def __init__(self, filename, filename_info, filetype_info,
                 header_file=None):
        """Init the XML annotation reader."""
        super().__init__(filename, filename_info, filetype_info, header_file)
        self.get_incidence_angle = functools.lru_cache(maxsize=10)(
            self._get_incidence_angle_uncached
        )

    def get_dataset(self, key, info, chunks=None):
        """Load a dataset."""
        if self._polarization != key["polarization"]:
            return

        if key["name"] == "incidence_angle":
            return self.get_incidence_angle(chunks=chunks or CHUNK_SIZE)

    def _get_incidence_angle_uncached(self, chunks):
        """Get the incidence angle array."""
        incidence_angle = XMLArray(self.root, ".//geolocationGridPoint", "incidenceAngle")
        return incidence_angle.expand(self._image_shape, chunks=chunks)


class SAFEXMLCalibration(SAFEXML):
    """XML file reader for the SAFE format, Calibration file."""

    def __init__(self, filename, filename_info, filetype_info,
                 header_file=None):
        """Init the XML calibration reader."""
        super().__init__(filename, filename_info, filetype_info, header_file)
        self.get_calibration = functools.lru_cache(maxsize=10)(
            self._get_calibration_uncached
        )

    def get_dataset(self, key, info, chunks=None):
        """Load a dataset."""
        if self._polarization != key["polarization"]:
            return
        if key["name"] == "calibration_constant":
            return self.get_calibration_constant()
        return self.get_calibration(key["name"], chunks=chunks or CHUNK_SIZE)

    def get_calibration_constant(self):
        """Load the calibration constant."""
        return float(self.root.find('.//absoluteCalibrationConstant').text)

    def _get_calibration_uncached(self, calibration, chunks=None):
        """Get the calibration array."""
        calibration_name = _get_calibration_name(calibration)
        calibration_vector = self._get_calibration_vector(calibration_name, chunks)
        return calibration_vector

    def _get_calibration_vector(self, calibration_name, chunks):
        """Get the calibration vector."""
        calibration_vector = XMLArray(self.root, ".//calibrationVector", calibration_name)
        return calibration_vector.expand(self._image_shape, chunks=chunks)


class SAFEXMLNoise(SAFEXML):
    """XML file reader for the SAFE format, Noise file."""

    def __init__(self, filename, filename_info, filetype_info,
                 header_file=None):
        """Init the xml filehandler."""
        super().__init__(filename, filename_info, filetype_info, header_file)

        self.azimuth_noise_reader = AzimuthNoiseReader(self.root, self._image_shape)
        self.get_noise_correction = functools.lru_cache(maxsize=10)(
            self._get_noise_correction_uncached
        )

    def get_dataset(self, key, info, chunks=None):
        """Load a dataset."""
        if self._polarization != key["polarization"]:
            return
        if key["name"] == "noise":
            return self.get_noise_correction(chunks=chunks or CHUNK_SIZE)

    def _get_noise_correction_uncached(self, chunks=None):
        """Get the noise correction array."""
        try:
            noise = self.read_legacy_noise(chunks)
        except KeyError:
            range_noise = self.read_range_noise_array(chunks)
            azimuth_noise = self.azimuth_noise_reader.read_azimuth_noise_array(chunks)
            noise = range_noise * azimuth_noise
        return noise

    def read_legacy_noise(self, chunks):
        """Read noise for legacy GRD data."""
        noise = XMLArray(self.root, ".//noiseVector", "noiseLut")
        return noise.expand(self._image_shape, chunks)

    def read_range_noise_array(self, chunks):
        """Read the range-noise array."""
        range_noise = XMLArray(self.root, ".//noiseRangeVector", "noiseRangeLut")
        return range_noise.expand(self._image_shape, chunks)


class AzimuthNoiseReader:
    """Class to parse and read azimuth-noise data.

    The azimuth noise vector is provided as a series of blocks, each comprised
    of a column of data to fill the block and a start and finish column number,
    and a start and finish line.
    For example, we can see here a (fake) azimuth noise array::

        [[ 1.  1.  1. nan nan nan nan nan nan nan]
         [ 1.  1.  1. nan nan nan nan nan nan nan]
         [ 2.  2.  3.  3.  3.  4.  4.  4.  4. nan]
         [ 2.  2.  3.  3.  3.  4.  4.  4.  4. nan]
         [ 2.  2.  3.  3.  3.  4.  4.  4.  4. nan]
         [ 2.  2.  5.  5.  5.  5.  6.  6.  6.  6.]
         [ 2.  2.  5.  5.  5.  5.  6.  6.  6.  6.]
         [ 2.  2.  5.  5.  5.  5.  6.  6.  6.  6.]
         [ 2.  2.  7.  7.  7.  7.  7.  8.  8.  8.]
         [ 2.  2.  7.  7.  7.  7.  7.  8.  8.  8.]]

    As is shown here, the blocks may not cover the full array, and hence it has
    to be gap-filled with NaNs.
    """

    def __init__(self, root, shape):
        """Set up the azimuth noise reader."""
        self.root = root
        self.elements = self.root.findall(".//noiseAzimuthVector")
        self._image_shape = shape
        self.blocks = []

    def read_azimuth_noise_array(self, chunks=CHUNK_SIZE):
        """Read the azimuth noise vectors."""
        self._read_azimuth_noise_blocks(chunks)
        populated_array = self._assemble_azimuth_noise_blocks(chunks)

        return populated_array

    def _read_azimuth_noise_blocks(self, chunks):
        """Read the azimuth noise blocks."""
        self.blocks = []
        for elt in self.elements:
            block = _AzimuthBlock(elt)
            new_arr = block.expand(chunks)
            self.blocks.append(new_arr)

    def _assemble_azimuth_noise_blocks(self, chunks):
        """Assemble the azimuth noise blocks into one single array."""
        # The strategy here is a bit convoluted. The job would be trivial if
        # performed on regular numpy arrays, but here we want to keep the data
        # as xarray/dask array as much as possible.
        # Using a pure xarray approach was tested (with `combine_first`,
        # `interpolate_na`, etc), but was found to be memory-hungry at the time
        # of implementation (March 2021). Hence the usage of a custom algorithm,
        # relying mostly on dask arrays.
        slices = self._create_dask_slices_from_blocks(chunks)
        populated_array = da.vstack(slices).rechunk(chunks)
        populated_array = xr.DataArray(populated_array, dims=['y', 'x'],
                                       coords={'x': np.arange(self._image_shape[1]),
                                               'y': np.arange(self._image_shape[0])})
        return populated_array

    def _create_dask_slices_from_blocks(self, chunks):
        """Create full-width slices from azimuth noise blocks."""
        current_line = 0
        slices = []
        while current_line < self._image_shape[0]:
            new_slice = self._create_dask_slice_from_block_line(current_line, chunks)
            slices.append(new_slice)
            current_line += new_slice.shape[0]
        return slices

    def _create_dask_slice_from_block_line(self, current_line, chunks):
        """Create a dask slice from the blocks at the current line."""
        pieces = self._get_array_pieces_for_current_line(current_line)
        dask_pieces = self._get_padded_dask_pieces(pieces, chunks)
        new_slice = da.hstack(dask_pieces)

        return new_slice

    def _get_array_pieces_for_current_line(self, current_line):
        """Get the array pieces that cover the current line."""
        current_blocks = self._find_blocks_covering_line(current_line)
        current_blocks.sort(key=(lambda x: x.coords['x'][0]))
        next_line = self._get_next_start_line(current_blocks, current_line)
        current_y = np.arange(current_line, next_line)
        pieces = [arr.sel(y=current_y) for arr in current_blocks]
        return pieces

    def _find_blocks_covering_line(self, current_line):
        """Find the blocks covering a given line."""
        current_blocks = []
        for block in self.blocks:
            if block.coords['y'][0] <= current_line <= block.coords['y'][-1]:
                current_blocks.append(block)
        return current_blocks

    def _get_next_start_line(self, current_blocks, current_line):
        next_line = min((arr.coords['y'][-1] for arr in current_blocks)) + 1
        blocks_starting_soon = [block for block in self.blocks if current_line < block.coords["y"][0] < next_line]
        if blocks_starting_soon:
            next_start_line = min((arr.coords["y"][0] for arr in blocks_starting_soon))
            next_line = min(next_line, next_start_line)
        return next_line

    def _get_padded_dask_pieces(self, pieces, chunks):
        """Get the padded pieces of a slice."""
        pieces = sorted(pieces, key=(lambda x: x.coords['x'][0]))
        dask_pieces = []
        previous_x_end = -1
        piece = pieces[0]
        next_x_start = piece.coords['x'][0].item()
        y_shape = len(piece.coords['y'])

        x_shape = (next_x_start - previous_x_end - 1)
        self._fill_dask_pieces(dask_pieces, (y_shape, x_shape), chunks)

        for i, piece in enumerate(pieces):
            dask_pieces.append(piece.data)
            previous_x_end = piece.coords['x'][-1].item()
            try:
                next_x_start = pieces[i + 1].coords['x'][0].item()
            except IndexError:
                next_x_start = self._image_shape[1]

            x_shape = (next_x_start - previous_x_end - 1)
            self._fill_dask_pieces(dask_pieces, (y_shape, x_shape), chunks)

        return dask_pieces

    @staticmethod
    def _fill_dask_pieces(dask_pieces, shape, chunks):
        if shape[1] > 0:
            new_piece = da.full(shape, np.nan, chunks=chunks)
            dask_pieces.append(new_piece)


def interpolate_slice(slice_rows, slice_cols, interpolator):
    """Interpolate the given slice of the larger array."""
    fine_rows = np.arange(slice_rows.start, slice_rows.stop, slice_rows.step)
    fine_cols = np.arange(slice_cols.start, slice_cols.stop, slice_cols.step)
    return interpolator(fine_cols, fine_rows)


class _AzimuthBlock:
    """Implementation of an single azimuth-noise block."""

    def __init__(self, xml_element):
        """Set up the block from an XML element."""
        self.element = xml_element

    def expand(self, chunks):
        """Build an azimuth block from xml data."""
        corr = 1
        # This isn't needed with newer data (> 2020). When was the change operated?
        #
        #         The azimuth noise is normalized per swath to account for gain
        #         differences between the swaths in EW mode.
        #
        #         This is based on the this reference:
        #         J. Park, A. A. Korosov, M. Babiker, S. Sandven and J. Won,
        #         "Efficient Thermal Noise Removal for Sentinel-1 TOPSAR Cross-Polarization Channel,"
        #         in IEEE Transactions on Geoscience and Remote Sensing, vol. 56, no. 3,
        #         pp. 1555-1565, March 2018.
        #         doi: 10.1109/TGRS.2017.2765248
        #
        # For old data. < 2020
        # swath = elt.find('swath').text
        # if swath == 'EW1':
        #     corr = 1.5
        # if swath in ['EW4', 'IW3']:
        #     corr = 1.2
        # if swath == 'EW5':
        #     corr = 1.5
        data = self.lut * corr

        x_coord = np.arange(self.first_pixel, self.last_pixel + 1)
        y_coord = np.arange(self.first_line, self.last_line + 1)

        new_arr = (da.ones((len(y_coord), len(x_coord)), chunks=chunks) *
                   np.interp(y_coord, self.lines, data)[:, np.newaxis])
        new_arr = xr.DataArray(new_arr,
                               dims=['y', 'x'],
                               coords={'x': x_coord,
                                       'y': y_coord})
        return new_arr

    @property
    def first_pixel(self):
        return int(self.element.find('firstRangeSample').text)

    @property
    def last_pixel(self):
        return int(self.element.find('lastRangeSample').text)

    @property
    def first_line(self):
        return int(self.element.find('firstAzimuthLine').text)

    @property
    def last_line(self):
        return int(self.element.find('lastAzimuthLine').text)

    @property
    def lines(self):
        lines = self.element.find('line').text.split()
        return np.array(lines).astype(int)

    @property
    def lut(self):
        lut = self.element.find('noiseAzimuthLut').text.split()
        return np.array(lut).astype(float)


class XMLArray:
    """A proxy for getting xml data as an array."""

    def __init__(self, root, list_tag, element_tag):
        """Set up the XML array."""
        self.root = root
        self.list_tag = list_tag
        self.element_tag = element_tag
        self.data, self.low_res_coords = self._read_xml_array()

    def expand(self, shape, chunks=None):
        """Generate the full-blown array."""
        return self.interpolate_xml_array(shape, chunks=chunks)

    def _read_xml_array(self):
        """Read an array from xml."""
        elements = self.get_data_items()
        y = []
        x = []
        data = []
        for elt in elements:
            new_x = elt.find('pixel').text.split()
            y += [int(elt.find('line').text)] * len(new_x)
            x += [int(val) for val in new_x]
            data += [float(val)
                     for val in elt.find(self.element_tag).text.split()]

        return np.asarray(data), (x, y)

    def get_data_items(self):
        """Get the data items for this array."""
        data_items = self.root.findall(self.list_tag)
        if not data_items:
            raise KeyError("Can't find data items for xml tag " + self.list_tag)
        return data_items

    def interpolate_xml_array(self, shape, chunks):
        """Interpolate arbitrary size dataset to a full sized grid."""
        xpoints, ypoints = self.low_res_coords
        return interpolate_xarray_linear(xpoints, ypoints, self.data, shape, chunks=chunks)


def interpolate_xarray(xpoints, ypoints, values, shape,
                       blocksize=CHUNK_SIZE):
    """Interpolate, generating a dask array."""
    from scipy.interpolate import RectBivariateSpline

    vchunks = range(0, shape[0], blocksize)
    hchunks = range(0, shape[1], blocksize)

    token = tokenize(blocksize, xpoints, ypoints, values, shape)
    name = 'interpolate-' + token

    spline = RectBivariateSpline(xpoints, ypoints, values.T)

    def interpolator(xnew, ynew):
        """Interpolator function."""
        return spline(xnew, ynew).T

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
    from scipy.interpolate.interpnd import LinearNDInterpolator, _ndim_coords_from_arrays

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

    def __init__(self, filename, filename_info, filetype_info, calfh, noisefh, annotationfh):
        """Init the grd filehandler."""
        super(SAFEGRD, self).__init__(filename, filename_info,
                                      filetype_info)

        self._start_time = filename_info['start_time']
        self._end_time = filename_info['end_time']

        self._polarization = filename_info['polarization']

        self._mission_id = filename_info['mission_id']

        self.calibration = calfh
        self.noise = noisefh
        self.annotation = annotationfh
        self.read_lock = Lock()

        self.filehandle = rasterio.open(self.filename, 'r', sharing=False)
        self.get_lonlatalts = functools.lru_cache(maxsize=2)(
            self._get_lonlatalts_uncached
        )

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
            data = rioxarray.open_rasterio(self.filename, chunks=(1, CHUNK_SIZE, CHUNK_SIZE)).squeeze()
            data = data.assign_coords(x=np.arange(len(data.coords['x'])),
                                      y=np.arange(len(data.coords['y'])))
            data = self._calibrate_and_denoise(data, key)
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

    def _calibrate_and_denoise(self, data, key):
        """Calibrate and denoise the data."""
        chunks = CHUNK_SIZE

        dn = self._get_digital_number(data)
        dn = self._denoise(dn, chunks)
        data = self._calibrate(dn, chunks, key)

        return data

    def _get_digital_number(self, data):
        """Get the digital numbers (uncalibrated data)."""
        data = data.where(data > 0)
        data = data.astype(np.float64)
        dn = data * data
        return dn

    def _denoise(self, dn, chunks):
        """Denoise the data."""
        logger.debug('Reading noise data.')
        noise = self.noise.get_noise_correction(chunks=chunks).fillna(0)
        dn = dn - noise
        return dn

    def _calibrate(self, dn, chunks, key):
        """Calibrate the data."""
        logger.debug('Reading calibration data.')
        cal = self.calibration.get_calibration(key['calibration'], chunks=chunks)
        cal_constant = self.calibration.get_calibration_constant()
        logger.debug('Calibrating.')
        data = ((dn + cal_constant) / (cal ** 2)).clip(min=0)
        return data

    def _get_lonlatalts_uncached(self):
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
