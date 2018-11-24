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
"LI L2 Product User Guide [LIL2PUG] Draft version" documentation.

"""
import h5netcdf
import logging
import numpy as np
from datetime import datetime
from pyresample import geometry
from satpy.readers.file_handlers import BaseFileHandler
# FIXME: This is not xarray/dask compatible
# TODO: Once migrated to xarray/dask, remove ignored path in setup.cfg
from satpy.dataset import Dataset

logger = logging.getLogger(__name__)


class LIFileHandler(BaseFileHandler):
    """MTG LI File Reader."""

    def __init__(self, filename, filename_info, filetype_info):
        super(LIFileHandler, self).__init__(filename, filename_info, filetype_info)
        self.nc = h5netcdf.File(self.filename, 'r')
        # Get grid dimensions from file
        refdim = self.nc['grid_position'][:]
        # Get number of lines and columns
        self.nlines = int(refdim[2])
        self.ncols = int(refdim[3])
        self.cache = {}
        logger.debug('Dimension : {}'.format(refdim))
        logger.debug('Row/Cols: {} / {}'.format(self.nlines, self.ncols))
        logger.debug('Reading: {}'.format(self.filename))
        logger.debug('Start: {}'.format(self.start_time))
        logger.debug('End: {}'.format(self.end_time))

    @property
    def start_time(self):
        return datetime.strptime(self.nc.attrs['sensing_start'], '%Y%m%d%H%M%S')

    @property
    def end_time(self):
        return datetime.strptime(self.nc.attrs['end_time'], '%Y%m%d%H%M%S')

    def get_dataset(self, key, info=None, out=None, xslice=None, yslice=None):
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

        # Get lightning data out of NetCDF container
        logger.debug("Key: {}".format(key.name))
        # Create reference grid
        grid = np.full((self.nlines, self.ncols), np.NaN)
        # Set slices to full disc extent
        if xslice is None:
            xslice = slice(0, self.ncols, None)
        if yslice is None:
            yslice = slice(0, self.nlines, None)
        logger.debug("Slices - x: {}, y: {}".format(xslice, yslice))
        # Get product values
        values = self.nc[typedict[key.name]]
        rows = self.nc['row']
        cols = self.nc['column']
        logger.debug('[ Number of values ] : {}'.format((len(values))))
        logger.debug('[Min/Max] : <{}> / <{}>'.format(np.min(values),
                                                      np.max(values)))
        # Convert xy coordinates to flatten indices
        ids = np.ravel_multi_index([rows, cols], grid.shape)
        # Replace NaN values with data
        np.put(grid, ids, values)

        # Correct for bottom left origin in LI row/column indices.
        rotgrid = np.flipud(grid)
        logger.debug('Data shape: {}, {}'.format(yslice, xslice))
        # Rotate the grid by 90 degree clockwise
        rotgrid = np.rot90(rotgrid, 3)
        logger.warning("LI data has been rotated to fit to reference grid. \
                        Works only for test dataset")
        # Slice the gridded lighting data
        slicegrid = rotgrid[yslice, xslice]
        # Mask invalid values
        ds = np.ma.masked_where(np.isnan(slicegrid), slicegrid)
        # Create dataset object
        out.data[:] = np.ma.getdata(ds)
        out.mask[:] = np.ma.getmask(ds)
        out.info.update(key.to_dict())

        return(out)

    def get_area_def(self, key, info=None):
        """Create AreaDefinition for specified product.

        Projection information are hard coded for 0 degree geos projection
        Test dataset doesn't provide the values in the file container.
        Only fill values are inserted.

        """
        # TODO Get projection information from input file
        a = 6378169.
        h = 35785831.
        b = 6356583.8
        lon_0 = 0.
        # area_extent = (-5432229.9317116784, -5429229.5285458621,
        #                5429229.5285458621, 5432229.9317116784)
        area_extent = (-5570248.4773392612, -5567248.074173444,
                       5567248.074173444, 5570248.4773392612)
        proj_dict = {'a': float(a),
                     'b': float(b),
                     'lon_0': float(lon_0),
                     'h': float(h),
                     'proj': 'geos',
                     'units': 'm'}

        area = geometry.AreaDefinition(
            'LI_area_name',
            "LI area",
            'geosli',
            proj_dict,
            self.ncols,
            self.nlines,
            area_extent)
        self.area = area
        logger.debug("Dataset area definition: \n {}".format(area))
        return area
