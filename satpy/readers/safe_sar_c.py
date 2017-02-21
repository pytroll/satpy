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
"""SAFE SAR-C format.
"""

import logging
import os
import xml.etree.ElementTree as ET

import numpy as np
from osgeo import gdal

from geotiepoints.geointerpolator import GeoInterpolator
from satpy.dataset import Dataset
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

    def __init__(self, filename, filename_info, filetype_info, header_file=None):
        super(SAFEXML, self).__init__(filename, filename_info,
                                      filetype_info)

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
    def interpolate_xml_array(data, low_res_coords, full_res_size):
        """Interpolate arbitrary size dataset to a full sized grid."""
        from scipy.interpolate import griddata
        grid_x, grid_y = np.mgrid[0:full_res_size[0], 0:full_res_size[1]]
        x, y = low_res_coords

        return griddata(np.vstack((np.asarray(y), np.asarray(x))).T,
                        data,
                        (grid_x, grid_y),
                        method='linear')

    def get_dataset(self, key, info):
        """Load a dataset."""
        if self._polarization != key.polarization:
            return

        data_items = self.root.findall(".//" + info['xml_item'])
        data, low_res_coords = self.read_xml_array(data_items,
                                                   info['xml_tag'])
        if key.name.endswith('squared'):
            data **= 2

        data = self.interpolate_xml_array(data, low_res_coords, data.shape)

    def get_noise_correction(self, shape):
        data_items = self.root.findall(".//noiseVector")
        data, low_res_coords = self.read_xml_array(data_items, 'noiseLut')
        return self.interpolate_xml_array(data, low_res_coords, shape)

    def get_calibration(self, name, shape):
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


class SAFEGRD(BaseFileHandler):

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

        band = self.filehandle

        if key.name in ['longitude', 'latitude']:
            logger.debug('Constructing coordinate arrays.')

            if self.lons is None or self.lats is None:
                self.lons, self.lats = self.get_lonlats()

            if key.name == 'latitude':
                proj = Dataset(self.lats, id=key, **info)
            else:
                proj = Dataset(self.lons, id=key, **info)

        else:
            data = band.GetRasterBand(1).ReadAsArray().astype(np.float)
            logger.debug('Reading noise data.')

            noise = self.noise.get_noise_correction(data.shape)

            logger.debug('Reading calibration data.')

            cal = self.calibration.get_calibration('gamma', data.shape)
            cal_constant = self.calibration.get_calibration_constant()

            logger.debug('Calibrating.')

            data **= 2
            data += cal_constant - noise
            data /= cal
            data[data < 0] = 0
            del noise, cal

            proj = Dataset(np.sqrt(data),
                           copy=False,
                           units='sigma')
        del band
        return proj

    def get_lonlats(self):
        """Obtain GCPs and construct latitude and longitude arrays.

        Args:
           band (gdal band): Measurement band which comes with GCP's
           array_shape (tuple) : The size of the data array
        Returns:
           coordinates (tuple): A tuple with longitude and latitude arrays
        """
        band = self.filehandle

        band_x_size = band.RasterXSize
        band_y_size = band.RasterYSize

        (xpoints, ypoints), (gcp_lons, gcp_lats) = self.get_gcps()
        fine_cols = np.arange(band_x_size)
        fine_rows = np.arange(band_y_size)

        satint = GeoInterpolator((gcp_lons, gcp_lats),
                                 (ypoints, xpoints),
                                 (fine_rows, fine_cols), 2, 2)

        longitudes, latitudes = satint.interpolate()

        # FIXME: check if the array is C-contigious, and make it such if it
        # isn't
        if longitudes.flags['CONTIGUOUS'] is False:
            longitudes = np.ascontiguousarray(longitudes)
        if latitudes.flags['CONTIGUOUS'] is False:
            latitudes = np.ascontiguousarray(latitudes)

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
