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
from satpy.readers import DatasetID
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


class SAFEGRD(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(SAFEGRD, self).__init__(filename, filename_info,
                                      filetype_info)

        self._start_time = filename_info['start_time']
        self._end_time = filename_info['end_time']

        self.band = None

        self.channel = filename_info['polarization']

        dirname, basename = os.path.split(filename)
        basedir, measurements = os.path.split(dirname)
        basename, ext = os.path.splitext(basename)
        self.cal_file = os.path.join(
            basedir, 'annotation', 'calibration', 'calibration-' + basename + '.xml')
        self.noise_file = os.path.join(
            basedir, 'annotation', 'calibration', 'noise-' + basename + '.xml')
        self.cal = None

    def get_dataset(self, key, info):
        """Load a dataset
        """
        if self.channel != key.name:
            return
        logger.debug('Reading %s.', key.name)

        from scipy.interpolate import griddata

        self.band = gdal.Open(self.filename)

        data = self.band.GetRasterBand(1).ReadAsArray()

        logger.debug('Reading noise data.')

        root = ET.parse(self.noise_file)

        noise_data = root.findall(
            ".//noiseVector")
        y = []
        x = []
        noise = []
        for elt in noise_data:
            newx = map(int, elt.find('pixel').text.split())
            y += [int(elt.find('line').text)] * len(newx)
            x += newx
            noise += map(float, elt.find('noiseLut').text.split())

        grid_x, grid_y = np.mgrid[0:data.shape[0], 0:data.shape[1]]

        noise = griddata(np.vstack((np.asarray(y), np.asarray(
            x))).T, noise, (grid_x, grid_y), method='linear')

        logger.debug('Reading calibration data.')
        root = ET.parse(self.cal_file)

        cal_constant = float(root.find('.//absoluteCalibrationConstant').text)

        cal_data = root.findall(
            ".//calibrationVector")

        y = []
        x = []
        sigma = []
        for elt in cal_data:
            newx = map(int, elt.find('pixel').text.split())
            y += [int(elt.find('line').text)] * len(newx)
            x += newx
            sigma += map(float, elt.find('gamma').text.split())

        grid_x, grid_y = np.mgrid[0:data.shape[0], 0:data.shape[1]]

        sigma = griddata(np.vstack((np.asarray(y), np.asarray(
            x))).T, sigma, (grid_x, grid_y), method='linear')

        logger.debug('Calibrating.')

        #val = np.sqrt((data ** 2. + cal_constant - noise) / sigma ** 2)
        #val = np.sqrt((data ** 2. - noise))
        val = (data ** 2. + cal_constant - noise) / sigma ** 2
        val[val < 0] = 0
        val = np.sqrt(val)

        proj = Projectable(val,
                           copy=False,
                           units='sigma')
        return proj

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time
