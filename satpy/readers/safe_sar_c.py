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
"""Compact viirs format.
"""

import logging
import os
from datetime import datetime
from xml.etree import ElementTree as ET

import numpy as np
from osgeo import gdal

from satpy.projectable import Projectable
from satpy.readers.file_handlers import BaseFileHandler

from geotiepoints.geointerpolator import GeoInterpolator

logger = logging.getLogger(__name__)


class SAFEXML(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(SAFEXML, self).__init__(filename, filename_info,
                                      filetype_info)

        self._start_time = filename_info['start_time']
        self._end_time = filename_info['end_time']
        self._polarization = filename_info['polarization']
        self.root = ET.parse(self.filename)

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

        self.band = None
        self.lats = None
        self.lons = None

        self._polarization = filename_info['polarization']

        self.calibration = calfh
        self.noise = noisefh
        self.filehandle = gdal.Open(self.filename)

    def get_dataset(self, key, info):
        """Load a dataset."""
        if self._polarization != key.polarization:
            return
        logger.debug('Reading %s.', key.name)

        band = self.filehandle

        band_x_size = band.RasterXSize
        band_y_size = band.RasterYSize

        if key.name in ['longitude', 'latitude']:
            logger.debug('Constructing coordinate arrays.')

            if self.lons is None or self.lats is None:
                self.lons, self.lats = self.get_lonlats(band, (band_y_size, band_x_size))

            if key.name == 'latitude':
                proj = Projectable(self.lats, id=key, **info)
            else:
                proj = Projectable(self.lons, id=key, **info)
        else:
            data = band.GetRasterBand(1).ReadAsArray().astype(np.float)
            logger.debug('Reading noise data.')

            noise = self.noise.get_noise_correction(data.shape)

            logger.debug('Reading calibration data.')

            cal = self.calibration.get_calibration('gamma', data.shape)
            cal_constant = self.calibration.get_calibration_constant()

            logger.debug('Calibrating.')

            # val = np.sqrt((data ** 2. + cal_constant - noise) / sigma ** 2)
            # val = np.sqrt((data ** 2. - noise))
            # data = (data ** 2. + cal_constant - noise) / sigma_sqr
            data **= 2
            data += cal_constant - noise
            data /= cal
            data[data < 0] = 0
            del noise, cal

            proj = Projectable(np.sqrt(data),
                               copy=False,
                               units='sigma')
        del band
        return proj

    def get_lonlats(self, band, array_shape):
        """
        Obtain GCPs and construct latitude and longitude arrays
 	    Args:
           band (gdal band): Measurement band which comes with GCP's
           array_shape (tuple) : The size of the data array
        Returns:
           coordinates (tuple): A tuple with longitude and latitude arrays
        """
        (xpoints, ypoints), (gcp_lons, gcp_lats), (fine_cols, fine_rows) = self.get_gcps(band, array_shape)
        satint = GeoInterpolator((gcp_lons, gcp_lats),
		                         (ypoints, xpoints),
		                         (fine_rows, fine_cols), 2,2)

        longitudes, latitudes = satint.interpolate()

        # FIXME: check if the array is C-contigious, and make it such if it isn't
        if longitudes.flags['CONTIGUOUS'] is False:
            longitudes = np.ascontiguousarray(longitudes)
        if latitudes.flags['CONTIGUOUS'] is False:
            latitudes = np.ascontiguousarray(latitudes)

        return longitudes, latitudes

    def get_gcps(self, band, array_shape):
        """
        Read GCP from the GDAL band

        Args:
           band (gdal band): Measurement band which comes with GCP's
           coordinates (tuple): A tuple with longitude and latitude arrays

        Returns:
           points (tuple): Pixel and Line indices 1d arrays
           gcp_coords (tuple): longitude and latitude 1d arrays
           fine_grid_points (tuple): pixel and line indices 1d arrays for the new coordinate grid
        """

        gcps = band.GetGCPs()

        points_array = np.array([(point.GCPLine, point.GCPPixel) for point in gcps])
        ypoints = np.unique(points_array[:,0])
        xpoints = np.unique(points_array[:,1])
        xpoints.sort(); ypoints.sort()

        gcp_coords = np.array([(points.GCPY, points.GCPX) for points in gcps])
        lats = gcp_coords[:,0].reshape(ypoints.shape[0], xpoints.shape[0])
        lons = gcp_coords[:,1].reshape(ypoints.shape[0], xpoints.shape[0])

        fine_cols = np.arange(array_shape[1])
        fine_rows = np.arange(array_shape[0])

        return (xpoints, ypoints), (lons, lats), (fine_cols, fine_rows)

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time
