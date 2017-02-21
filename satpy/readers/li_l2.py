#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016.

# Author(s):

#
#   Thomas Leppelt <thomas.leppelt@gmail.com>
#   Sauli Joro <sauli.joro@icloud.com>

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Interface to MTG-LI L2 product NetCDF files

The reader is based on preliminary test data provided by EUMETSAT.
The data description is described in the 
    "LI L2 Product User Guide [LIL2PUG] Draft version" documentation
"""
import h5netcdf
import logging
import numpy as np
import os.path
from collections import defaultdict
from datetime import datetime, timedelta
from pyresample import geometry
from satpy.dataset import Dataset
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


class LIFileHandler(BaseFileHandler):
    """MTG LI File Reader
    """

    def __init__(self, filename, filename_info, filetype_info):
        super(LIFileHandler, self).__init__(filename, filename_info,
                                        filetype_info)
        logger.debug("READING: %s" % filename)
        logger.debug("START: %s" % self.start_time)
        logger.debug("END: %s" % self.end_time)
        
        self.nc = h5netcdf.File(filename, 'r')
        self.filename = filename
        self.cache = {}

    @property
    def start_time(self):
        return self.filename_info['start_time']

    @property
    def end_time(self):
        return self.filename_info['end_time']

    def get_dataset(self, key, info=None):
        """Load a dataset
        """
        if key in self.cache:
            return self.cache[key]
        # Type dictionary
        typedict = {"af": "flash_accumulation",
                    "afa": "accumulated_flash_area",
                    "afr": "flash_radiance",
                    "lgr": "radiance",
                    "lef": "radiance",
                    "lfl": "radiance"}

        #Get lightning data out of NetCDF container
        logger.debug("KEY: %s" % key.name)
        # Get grid dimensions from file
        refdim = self.nc['grid_position'][:]
        # Get number of lines and columns
        self.nlines = int(refdim[2])
        self.ncols = int(refdim[3])
        # Create reference grid
        grid = np.full((refdim[2], refdim[3]), np.NaN)
        # Get product value
        values = self.nc[typedict[key.name]][:]
        rows = self.nc['row'][:]
        cols = self.nc['column'][:]
        # Convert xy coordinates to flattend indices
        ids = np.ravel_multi_index([rows, cols], grid.shape)
        # Replace NaN values with data
        np.put(grid, ids, values)
        # Correct for bottom left origin in LI row/column indices.
        rotgrid = np.flipud(grid)
        logger.debug('DATA SHAPE: %s' % str(rotgrid.shape))
        # Rotate the grid by 90 degree clockwise
        logger.warning("LI data has been roteted to fit to reference grid. \
                        Works only for test dataset")
        rotgrid = np.rot90(rotgrid, 3)

        logger.debug('[ Dimension ] : %s' % (refdim))
        logger.debug("ROW/COLS: %d / %d" % (self.nlines, self.ncols))
        logger.debug('[ Number of values ] : %d' % (len(values)))
        logger.debug('[Min/Max] : <%d> / <%d>' % (np.min(values), np.max(values)))
        logger.debug("START: %s" % self.start_time)
        logger.debug("END: %s" % self.end_time)
        ds = (np.ma.masked_invalid(rotgrid[:]))
        # Create projectable object
        out = Dataset(ds, dtype=np.float32)
        self.cache[key] = out

        return(out)

    def get_area_def(self, key, info):
        """ Projection information are hard coded for 0 degree geos projection
        Test dataset doen't provide the values in the file container.
        Only fill values are inserted
        """
        # TODO Get projection information from input file
        a = 6378169.
        h=35785831.
        b=6356583.8
        lon_0 = 0.
        area_extent = (-5432229.9317116784, -5429229.5285458621,
                        5429229.5285458621, 5432229.9317116784)

        proj_dict = {'a': float(a),
                     'b': float(b),
                     'lon_0': float(lon_0),
                     'h': float(h),
                     'proj': 'geos',
                     'units': 'm'}

        area = geometry.AreaDefinition(
            'Test_area_name',
            "Test area",
            'geosli',
            proj_dict,
            self.ncols,
            self.nlines,
            area_extent)

        self.area = area

        return area

