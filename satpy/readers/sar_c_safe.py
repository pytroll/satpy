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
import json
import logging
import warnings
from collections import defaultdict
from datetime import timezone as tz
from functools import cached_property
from pathlib import Path
from threading import Lock

import defusedxml.ElementTree as ET
import numpy as np
import rasterio
import rioxarray  # noqa F401  # xarray open_dataset use engine rasterio, which use rioxarray
import xarray as xr
from dask import array as da
from geotiepoints.geointerpolator import lonlat2xyz, xyz2lonlat
from geotiepoints.interpolator import MultipleSplineInterpolator
from xarray import DataArray

from satpy.dataset.data_dict import DatasetDict
from satpy.dataset.dataid import DataID
from satpy.readers import open_file_or_filename
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.yaml_reader import GenericYAMLReader
from satpy.utils import get_legacy_chunk_size

logger = logging.getLogger(__name__)
CHUNK_SIZE = get_legacy_chunk_size()


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
                return np.float32(r.text)
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
    calibration_name = getattr(calibration, "name", calibration) or "gamma"
    if calibration_name == "sigma_nought":
        calibration_name = "sigmaNought"
    elif calibration_name == "beta_nought":
        calibration_name = "betaNought"
    return calibration_name


class SAFEXML(BaseFileHandler):
    """XML file reader for the SAFE format."""

    def __init__(self, filename, filename_info, filetype_info,
                 header_file=None, image_shape=None):
        """Init the xml filehandler."""
        super().__init__(filename, filename_info, filetype_info)

        self._start_time = filename_info["start_time"].replace(tzinfo=tz.utc)
        self._end_time = filename_info["end_time"].replace(tzinfo=tz.utc)
        self._polarization = filename_info["polarization"]
        if isinstance(self.filename, str):
            self.filename = Path(self.filename)
        with self.filename.open() as fd:
            self.root = ET.parse(fd)
        self._image_shape = image_shape

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
        self.hdr = self.get_metadata()
        self._image_shape = (self.hdr["product"]["imageAnnotation"]["imageInformation"]["numberOfLines"],
                             self.hdr["product"]["imageAnnotation"]["imageInformation"]["numberOfSamples"])

    @property
    def image_shape(self):
        """Return the image shape of this dataset."""
        return self._image_shape

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


class Calibrator(SAFEXML):
    """XML file reader for the SAFE format, Calibration file."""

    def __init__(self, filename, filename_info, filetype_info,
                 header_file=None, image_shape=None):
        """Init the XML calibration reader."""
        super().__init__(filename, filename_info, filetype_info, header_file, image_shape)
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
        return np.float32(self.root.find(".//absoluteCalibrationConstant").text)

    def _get_calibration_uncached(self, calibration, chunks=None):
        """Get the calibration array."""
        calibration_name = _get_calibration_name(calibration)
        calibration_vector = self._get_calibration_vector(calibration_name, chunks)
        return calibration_vector

    def _get_calibration_vector(self, calibration_name, chunks):
        """Get the calibration vector."""
        calibration_vector = XMLArray(self.root, ".//calibrationVector", calibration_name)
        return calibration_vector.expand(self._image_shape, chunks=chunks)

    def __call__(self, dn, calibration_type, chunks=None):
        """Calibrate the data."""
        logger.debug("Reading calibration data.")
        cal = self.get_calibration(calibration_type, chunks=chunks)
        cal_constant = self.get_calibration_constant()
        logger.debug("Calibrating.")
        data = ((dn + cal_constant) / (cal ** 2)).clip(min=0)
        return data

class Denoiser(SAFEXML):
    """XML file reader for the SAFE format, Noise file."""

    def __init__(self, filename, filename_info, filetype_info,
                 header_file=None, image_shape=None):
        """Init the xml filehandler."""
        super().__init__(filename, filename_info, filetype_info, header_file, image_shape)

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

    def __call__(self, dn, chunks):
        """Denoise the data."""
        logger.debug("Reading noise data.")
        noise = self.get_noise_correction(chunks=chunks).fillna(0)
        dn = dn - noise
        return dn



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
        populated_array = xr.DataArray(populated_array, dims=["y", "x"],
                                       coords={"x": np.arange(self._image_shape[1]),
                                               "y": np.arange(self._image_shape[0])})
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
        current_blocks.sort(key=(lambda x: x.coords["x"][0]))
        next_line = self._get_next_start_line(current_blocks, current_line)
        current_y = np.arange(current_line, next_line, dtype=np.uint16)
        pieces = [arr.sel(y=current_y) for arr in current_blocks]
        return pieces

    def _find_blocks_covering_line(self, current_line):
        """Find the blocks covering a given line."""
        current_blocks = []
        for block in self.blocks:
            if block.coords["y"][0] <= current_line <= block.coords["y"][-1]:
                current_blocks.append(block)
        return current_blocks

    def _get_next_start_line(self, current_blocks, current_line):
        next_line = min((arr.coords["y"][-1] for arr in current_blocks)) + 1
        blocks_starting_soon = [block for block in self.blocks if current_line < block.coords["y"][0] < next_line]
        if blocks_starting_soon:
            next_start_line = min((arr.coords["y"][0] for arr in blocks_starting_soon))
            next_line = min(next_line, next_start_line)
        return next_line

    def _get_padded_dask_pieces(self, pieces, chunks):
        """Get the padded pieces of a slice."""
        pieces = sorted(pieces, key=(lambda x: x.coords["x"][0]))
        dask_pieces = []
        previous_x_end = -1
        piece = pieces[0]
        next_x_start = piece.coords["x"][0].item()
        y_shape = len(piece.coords["y"])

        x_shape = (next_x_start - previous_x_end - 1)
        self._fill_dask_pieces(dask_pieces, (y_shape, x_shape), chunks)

        for i, piece in enumerate(pieces):
            dask_pieces.append(piece.data)
            previous_x_end = piece.coords["x"][-1].item()
            try:
                next_x_start = pieces[i + 1].coords["x"][0].item()
            except IndexError:
                next_x_start = self._image_shape[1]

            x_shape = (next_x_start - previous_x_end - 1)
            self._fill_dask_pieces(dask_pieces, (y_shape, x_shape), chunks)

        return dask_pieces

    @staticmethod
    def _fill_dask_pieces(dask_pieces, shape, chunks):
        if shape[1] > 0:
            new_piece = da.full(shape, np.nan, chunks=chunks, dtype=np.float32)
            dask_pieces.append(new_piece)


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

        x_coord = np.arange(self.first_pixel, self.last_pixel + 1, dtype=np.uint16)
        y_coord = np.arange(self.first_line, self.last_line + 1, dtype=np.uint16)
        new_arr = (da.ones((len(y_coord), len(x_coord)), dtype=np.float32, chunks=chunks) *
                   np.interp(y_coord, self.lines, data)[:, np.newaxis].astype(np.float32))
        new_arr = xr.DataArray(new_arr,
                               dims=["y", "x"],
                               coords={"x": x_coord,
                                       "y": y_coord})
        return new_arr

    @property
    def first_pixel(self):
        return np.uint16(self.element.find("firstRangeSample").text)

    @property
    def last_pixel(self):
        return np.uint16(self.element.find("lastRangeSample").text)

    @property
    def first_line(self):
        return np.uint16(self.element.find("firstAzimuthLine").text)

    @property
    def last_line(self):
        return np.uint16(self.element.find("lastAzimuthLine").text)

    @property
    def lines(self):
        lines = self.element.find("line").text.split()
        return np.array(lines).astype(np.uint16)

    @property
    def lut(self):
        lut = self.element.find("noiseAzimuthLut").text.split()
        return np.array(lut, dtype=np.float32)


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
            new_x = elt.find("pixel").text.split()
            y += [int(elt.find("line").text)] * len(new_x)
            x += [int(val) for val in new_x]
            data += [np.float32(val)
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

    points = _ndim_coords_from_arrays(np.vstack((np.asarray(ypoints, dtype=np.uint16),
                                                 np.asarray(xpoints, dtype=np.uint16))).T)

    interpolator = LinearNDInterpolator(points, values)

    grid_x, grid_y = da.meshgrid(da.arange(shape[1], chunks=hchunks, dtype=np.uint16),
                                 da.arange(shape[0], chunks=vchunks, dtype=np.uint16))

    # workaround for non-thread-safe first call of the interpolator:
    interpolator((0, 0))
    res = da.map_blocks(intp, grid_x, grid_y, interpolator=interpolator).astype(values.dtype)

    return DataArray(res, dims=("y", "x"))


class SAFEGRD(BaseFileHandler):
    """Measurement file reader.

    The measurement files are in geotiff format and read using rasterio. For
    performance reasons, the reading adapts the chunk size to match the file's
    block size.
    """

    def __init__(self, filename, filename_info, filetype_info, calibrator, denoiser):
        """Init the grd filehandler."""
        super().__init__(filename, filename_info, filetype_info)
        self._start_time = filename_info["start_time"].replace(tzinfo=tz.utc)
        self._end_time = filename_info["end_time"].replace(tzinfo=tz.utc)

        self._polarization = filename_info["polarization"]

        self._mission_id = filename_info["mission_id"]

        self.calibrator = calibrator
        self.denoiser = denoiser
        self.read_lock = Lock()

        self.get_lonlatalts = functools.lru_cache(maxsize=2)(
            self._get_lonlatalts_uncached
        )

    def get_dataset(self, key, info):
        """Load a dataset."""
        if self._polarization != key["polarization"]:
            return

        logger.debug("Reading %s.", key["name"])

        if key["name"] in ["longitude", "latitude", "altitude"]:
            logger.debug("Constructing coordinate arrays.")
            arrays = dict()
            arrays["longitude"], arrays["latitude"], arrays["altitude"] = self.get_lonlatalts()

            data = arrays[key["name"]]
            data.attrs.update(info)

        else:
            data = self._calibrate_and_denoise(self._data, key)
            data.attrs.update(info)
            data.attrs.update({"platform_name": self._mission_id})

            data = self._change_quantity(data, key["quantity"])

        return data

    @cached_property
    def _data(self):
        data = xr.open_dataarray(open_file_or_filename(self.filename, mode="rb"), engine="rasterio",
                                 chunks="auto"
                                ).squeeze()
        self.chunks = data.data.chunksize
        data = data.assign_coords(x=np.arange(len(data.coords["x"])),
                                      y=np.arange(len(data.coords["y"])))

        return data

    @staticmethod
    def _change_quantity(data, quantity):
        """Change quantity to dB if needed."""
        if quantity == "dB":
            data.data = 10 * np.log10(data.data)
            data.attrs["units"] = "dB"
        else:
            data.attrs["units"] = "1"

        return data

    def _calibrate_and_denoise(self, data, key):
        """Calibrate and denoise the data."""
        dn = self._get_digital_number(data)
        dn = self.denoiser(dn, self.chunks)
        data = self.calibrator(dn, key["calibration"], self.chunks)

        return data

    def _get_digital_number(self, data):
        """Get the digital numbers (uncalibrated data)."""
        data = data.where(data > 0)
        data = data.astype(np.float32)
        dn = data * data
        return dn

    def _get_lonlatalts_uncached(self):
        """Obtain GCPs and construct latitude and longitude arrays.

        Args:
           band (gdal band): Measurement band which comes with GCP's
           array_shape (tuple) : The size of the data array
        Returns:
           coordinates (tuple): A tuple with longitude and latitude arrays
        """
        shape = self._data.shape

        (xpoints, ypoints), (gcp_lons, gcp_lats, gcp_alts), (gcps, crs) = self.get_gcps()

        fine_points = [np.arange(size) for size in shape]
        x, y, z = lonlat2xyz(gcp_lons, gcp_lats)


        interpolator = MultipleSplineInterpolator((ypoints, xpoints), x, y, z, gcp_alts, kx=2, ky=2)
        hx, hy, hz, altitudes = interpolator.interpolate(fine_points, chunks=self.chunks)


        longitudes, latitudes = xyz2lonlat(hx, hy, hz)
        altitudes = xr.DataArray(altitudes, dims=["y", "x"])
        longitudes = xr.DataArray(longitudes, dims=["y", "x"])
        latitudes = xr.DataArray(latitudes, dims=["y", "x"])

        longitudes.attrs["gcps"] = gcps
        longitudes.attrs["crs"] = crs
        latitudes.attrs["gcps"] = gcps
        latitudes.attrs["crs"] = crs
        altitudes.attrs["gcps"] = gcps
        altitudes.attrs["crs"] = crs

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
        gcps = get_gcps_from_array(self._data)
        crs = self._data.rio.crs

        gcp_list = [(feature["properties"]["row"], feature["properties"]["col"], *feature["geometry"]["coordinates"])
                    for feature in gcps["features"]]
        gcp_array = np.array(gcp_list)

        ypoints = np.unique(gcp_array[:, 0]).astype(np.uint16)
        xpoints = np.unique(gcp_array[:, 1]).astype(np.uint16)

        gcp_lons = gcp_array[:, 2].reshape(ypoints.shape[0], xpoints.shape[0])
        gcp_lats = gcp_array[:, 3].reshape(ypoints.shape[0], xpoints.shape[0])
        gcp_alts = gcp_array[:, 4].reshape(ypoints.shape[0], xpoints.shape[0])

        rio_gcps = [rasterio.control.GroundControlPoint(*gcp) for gcp in gcp_list]

        return (xpoints, ypoints), (gcp_lons, gcp_lats, gcp_alts), (rio_gcps, crs)

    def get_bounding_box(self):
        """Get the bounding box for the data coverage."""
        (xpoints, ypoints), (gcp_lons, gcp_lats, gcp_alts), (rio_gcps, crs) = self.get_gcps()
        bblons = np.hstack((gcp_lons[0, :-1], gcp_lons[:-1, -1], gcp_lons[-1, :1:-1], gcp_lons[:1:-1, 0]))
        bblats = np.hstack((gcp_lats[0, :-1], gcp_lats[:-1, -1], gcp_lats[-1, :1:-1], gcp_lats[:1:-1, 0]))
        return bblons.tolist(), bblats.tolist()

    @property
    def start_time(self):
        """Get the start time."""
        return self._start_time

    @property
    def end_time(self):
        """Get the end time."""
        return self._end_time


class SAFESARReader(GenericYAMLReader):
    """A reader for SAFE SAR-C data for Sentinel 1 satellites."""

    def __init__(self, config, filter_parameters=None):
        """Set up the SAR reader."""
        super().__init__(config)
        self.filter_parameters = filter_parameters
        self.files_by_type = defaultdict(list)
        self.storage_items = []

    @property
    def start_time(self):
        """Get the start time."""
        return self.storage_items.values()[0].filename_info["start_time"].replace(tzinfo=tz.utc)

    @property
    def end_time(self):
        """Get the end time."""
        return self.storage_items.values()[0].filename_info["end_time"].replace(tzinfo=tz.utc)

    def load(self, dataset_keys, **kwargs):
        """Load some data."""
        if kwargs:
            warnings.warn(f"Don't know how to handle kwargs {kwargs}")
        datasets = DatasetDict()
        for key in dataset_keys:
            for handler in self.storage_items.values():
                val = handler.get_dataset(key, info=dict())
                if val is not None:
                    val.attrs["start_time"] = handler.start_time
                    if key["name"] not in ["longitude", "latitude"]:
                        lonlats = self.load([DataID(self._id_keys, name="longitude", polarization=key["polarization"]),
                                             DataID(self._id_keys, name="latitude", polarization=key["polarization"])])
                        gcps = get_gcps_from_array(val)
                        from pyresample.future.geometry import SwathDefinition
                        val.attrs["area"] = SwathDefinition(lonlats["longitude"], lonlats["latitude"],
                                                            attrs=dict(gcps=gcps,
                                                                       bounding_box=handler.get_bounding_box()))
                    datasets[key] = val
                    continue
        return datasets

    def create_storage_items(self, files, **kwargs):
        """Create the storage items."""
        self.files_by_type = self._get_files_by_type(files)
        image_shapes = self._get_image_shapes()
        calibrators = self._create_calibrators(image_shapes)
        denoisers = self._create_denoisers(image_shapes)
        measurement_handlers = self._create_measurement_handlers(calibrators, denoisers)

        self.storage_items = measurement_handlers


    def _get_files_by_type(self, files):
        files_by_type = defaultdict(list)
        for file_type, type_info in self.config["file_types"].items():
            files_by_type[file_type].extend(self.filename_items_for_filetype(files, type_info))
        return files_by_type


    def _get_image_shapes(self):
        image_shapes = dict()
        for annotation_file, annotation_info in self.files_by_type["safe_annotation"]:
            annotation_fh = SAFEXMLAnnotation(annotation_file,
                                              filename_info=annotation_info,
                                              filetype_info=None)
            image_shapes[annotation_info["polarization"]] = annotation_fh.image_shape
        return image_shapes


    def _create_calibrators(self, image_shapes):
        calibrators = dict()
        for calibration_file, calibration_info in self.files_by_type["safe_calibration"]:
            polarization = calibration_info["polarization"]
            calibrators[polarization] = Calibrator(calibration_file,
                                                   filename_info=calibration_info,
                                                   filetype_info=None,
                                                   image_shape=image_shapes[polarization])

        return calibrators


    def _create_denoisers(self, image_shapes):
        denoisers = dict()
        for noise_file, noise_info in self.files_by_type["safe_noise"]:
            polarization = noise_info["polarization"]
            denoisers[polarization] = Denoiser(noise_file,
                                               filename_info=noise_info,
                                               filetype_info=None,
                                               image_shape=image_shapes[polarization])

        return denoisers


    def _create_measurement_handlers(self, calibrators, denoisers):
        measurement_handlers = dict()
        for measurement_file, measurement_info in self.files_by_type["safe_measurement"]:
            polarization = measurement_info["polarization"]
            measurement_handlers[polarization] = SAFEGRD(measurement_file,
                                                         filename_info=measurement_info,
                                                         calibrator=calibrators[polarization],
                                                         denoiser=denoisers[polarization],
                                                         filetype_info=None)

        return measurement_handlers


def get_gcps_from_array(val):
    """Get the gcps from the spatial_ref coordinate as a geojson dict."""
    gcps = val.coords["spatial_ref"].attrs["gcps"]
    if isinstance(gcps, str):
        gcps = json.loads(gcps)
    return gcps
