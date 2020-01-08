#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2018 Satpy developers
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
"""The SCMI AWIPS writer is used to create AWIPS-compatible tiled NetCDF4 files.

The Advanced Weather Interactive Processing System (AWIPS) is a
program used by the United States National Weather Service (NWS) and others
to view
different forms of weather imagery. Sectorized Cloud and Moisture Imagery
(SCMI) is a netcdf format accepted by AWIPS to store one image broken up
in to one or more "tiles". Once AWIPS is configured for specific products
the SCMI NetCDF backend can be used to provide compatible products to the
system. The files created by this backend are compatible with AWIPS II (AWIPS I is no
longer supported).

The SCMI writer takes remapped binary image data and creates an
AWIPS-compatible NetCDF4 file. The SCMI writer and the AWIPS client may
need to be configured to make things appear the way the user wants in
the AWIPS client. The SCMI writer can only produce files for datasets mapped
to areas with specific projections:

 - lcc
 - geos
 - merc
 - stere

This is a limitation of the AWIPS client and not of the SCMI writer.

Numbered versus Lettered Grids
------------------------------

By default the SCMI writer will save tiles by number starting with '1'
representing the upper-left image tile. Tile numbers then increase
along the column and then on to the next row.

By specifying `lettered_grid` as `True` tiles can be designated with a
letter. Lettered grids or sectors are preconfigured in the `scmi.yaml`
configuration file. The lettered tile locations are static and will not
change with the data being written to them. Each lettered tile is split
in to a certain number of subtiles (`num_subtiles`), default 2 rows by
2 columns. Lettered tiles are meant to make it easier for receiving
AWIPS clients/stations to filter what tiles they receive; saving time,
bandwidth, and space.

Any tiles (numbered or lettered) not containing any valid data are not
created.

Updating tiles
--------------

There are some input data cases where we want to put new data in a tile
file written by a previous execution. An example is a pre-tiled input dataset
that is processed one tile at a time. One input tile may map to one or more
output SCMI tiles, but may not perfectly align with the SCMI tile, leaving
empty/unused space in the SCMI tile. The next input tile may be able to fill
in that empty space and should be allowed to write the "new" data to the file.
This is the default behavior of the SCMI writer. In cases where data overlaps
the existing data in the tile, the newer data has priority.

Shifting Lettered Grids
-----------------------

Due to the static nature of the lettered grids, there is sometimes a
need to shift the locations of where these tiles are by up to 0.5 pixels in
each dimension to align with the data being processed. This means that the
tiles for a 1000m resolution grid may be shifted up to 500m in each direction
from the original definition of the lettered "sector". This can cause
differences in the location of the tiles between executions depending on the
locations of the input data. In the worst case tile A01 from one execution
could be shifted up to 1 grid cell from tile A01 in another execution (one
is shifted 0.5 pixels to the left, the other is shifted 0.5 to the right).

This shifting makes the calculations for generating tiles easier and
more accurate. By default, the lettered tile locations are changed to match
the location of the data. This works well when output tiles will not be
updated (see above) in future processing. In cases where output tiles will be
filled in or updated with more data the ``use_sector_reference`` keyword
argument can be set to ``True`` to tell the SCMI writer to shift the data's
geolocation by up to 0.5 pixels in each dimension instead of shifting the
lettered tile locations.

"""
import os
import logging
import string
import sys
from datetime import datetime, timedelta
from netCDF4 import Dataset

import numpy as np
from pyproj import Proj
import dask.array as da
from satpy.writers import Writer, DecisionTree, Enhancer, get_enhanced_image
from pyresample.geometry import AreaDefinition
from collections import namedtuple

try:
    from pyresample.utils import proj4_radius_parameters
except ImportError:
    raise ImportError("SCMI Writer requires pyresample>=1.7.0")

LOG = logging.getLogger(__name__)
# AWIPS 2 seems to not like data values under 0
AWIPS_USES_NEGATIVES = False
AWIPS_DATA_DTYPE = np.int16
DEFAULT_OUTPUT_PATTERN = '{source_name}_AII_{platform_name}_{sensor}_' \
                         '{name}_{sector_id}_{tile_id}_' \
                         '{start_time:%Y%m%d_%H%M}.nc'

# misc. global attributes
SCMI_GLOBAL_ATT = dict(
    satellite_id=None,  # GOES-H8
    pixel_y_size=None,  # km
    start_date_time=None,  # 2015181030000,  # %Y%j%H%M%S
    pixel_x_size=None,  # km
    product_name=None,  # "HFD-010-B11-M1C01",
    production_location=None,  # "MSC",
)


UNIT_CONV = {
    'micron': 'microm',
    'mm h-1': 'mm/h',
    '1': '*1',
    'none': '*1',
    'percent': '%',
    'Kelvin': 'kelvin',
    'K': 'kelvin',
}

TileInfo = namedtuple('TileInfo', ['tile_count', 'image_shape', 'tile_shape',
                                   'tile_row_offset', 'tile_column_offset', 'tile_id',
                                   'x', 'y', 'tile_slices', 'data_slices'])
XYFactors = namedtuple('XYFactors', ['mx', 'bx', 'my', 'by'])


def fix_awips_file(fn):
    """Hack the NetCDF4 files to workaround NetCDF-Java bugs used by AWIPS.

    This should not be needed for new versions of AWIPS.

    """
    # hack to get files created by new NetCDF library
    # versions to be read by AWIPS buggy java version
    # of NetCDF
    LOG.info("Modifying SCMI NetCDF file to work with AWIPS")
    import h5py
    h = h5py.File(fn, 'a')
    if '_NCProperties' in h.attrs:
        del h.attrs['_NCProperties']
    h.close()


class NumberedTileGenerator(object):
    """Helper class to generate per-tile metadata for numbered tiles."""

    def __init__(self, area_definition,
                 tile_shape=None, tile_count=None):
        """Initialize and generate tile information for this sector/grid for later use."""
        self.area_definition = area_definition
        self._rows = self.area_definition.y_size
        self._cols = self.area_definition.x_size

        # get tile shape, number of tiles, etc.
        self._get_tile_properties(tile_shape, tile_count)
        # scaling parameters for the overall images X and Y coordinates
        # they must be the same for all X and Y variables for all tiles
        # and must be stored in the file as 0, 1, 2, 3, ...
        # (X factor, X offset, Y factor, Y offset)
        self.mx, self.bx, self.my, self.by = self._get_xy_scaling_parameters()
        self.xy_factors = XYFactors(self.mx, self.bx, self.my, self.by)
        self._tile_cache = []

    def _get_tile_properties(self, tile_shape, tile_count):
        """Generate tile information for numbered tiles."""
        if tile_shape is not None:
            tile_shape = (int(min(tile_shape[0], self._rows)), int(min(tile_shape[1], self._cols)))
            tile_count = (int(np.ceil(self._rows / float(tile_shape[0]))),
                          int(np.ceil(self._cols / float(tile_shape[1]))))
        elif tile_count:
            tile_shape = (int(np.ceil(self._rows / float(tile_count[0]))),
                          int(np.ceil(self._cols / float(tile_count[1]))))
        else:
            raise ValueError("Either 'tile_count' or 'tile_shape' must be provided")

        # number of pixels per each tile
        self.tile_shape = tile_shape
        # number of tiles in each direction (rows, columns)
        self.tile_count = tile_count
        # number of tiles in the entire image
        self.total_tiles = tile_count[0] * tile_count[1]
        # number of pixels in the whole image (rows, columns)
        self.image_shape = (self.tile_shape[0] * self.tile_count[0],
                            self.tile_shape[1] * self.tile_count[1])

        # X and Y coordinates of the whole image
        self.x, self.y = self._get_xy_arrays()

    def _get_xy_arrays(self):
        """Get the overall X/Y coordinate variable arrays."""
        gd = self.area_definition
        ts = self.tile_shape
        tc = self.tile_count
        # Since our tiles may go over the edge of the original "grid" we
        # need to make sure we calculate X/Y to the edge of all of the tiles
        imaginary_data_size = (ts[0] * tc[0], ts[1] * tc[1])
        ps_x = gd.pixel_size_x
        ps_y = gd.pixel_size_y
        # tiles start from upper-left
        new_extents = (
            gd.area_extent[0],
            gd.area_extent[1] - ps_y * (imaginary_data_size[1] - gd.y_size),
            gd.area_extent[2] + ps_x * (imaginary_data_size[0] - gd.x_size),
            gd.area_extent[3])
        imaginary_grid_def = AreaDefinition(
            gd.area_id,
            gd.name,
            gd.proj_id,
            gd.proj_dict,
            imaginary_data_size[1],
            imaginary_data_size[0],
            new_extents,
        )

        x, y = imaginary_grid_def.get_proj_coords()
        x = x[0].squeeze()  # all rows should have the same coordinates
        y = y[:, 0].squeeze()  # all columns should have the same coordinates
        # scale the X and Y arrays to fit in the file for 16-bit integers
        # AWIPS is dumb and requires the integer values to be 0, 1, 2, 3, 4
        # Max value of a signed 16-bit integer is 32767 meaning
        # 32768 values.
        if x.shape[0] > 2**15:
            # awips uses 0, 1, 2, 3 so we can't use the negative end of the variable space
            raise ValueError("X variable too large for AWIPS-version of 16-bit integer space")
        if y.shape[0] > 2**15:
            # awips uses 0, 1, 2, 3 so we can't use the negative end of the variable space
            raise ValueError("Y variable too large for AWIPS-version of 16-bit integer space")
        # NetCDF library doesn't handle numpy arrays nicely anymore for some
        # reason and has been masking values that shouldn't be
        return np.ma.masked_array(x), np.ma.masked_array(y)

    def _get_xy_scaling_parameters(self):
        """Get the X/Y coordinate limits for the full resulting image."""
        gd = self.area_definition
        bx = self.x.min()
        mx = gd.pixel_size_x
        by = self.y.min()
        my = -abs(gd.pixel_size_y)
        return mx, bx, my, by

    def _tile_number(self, ty, tx):
        """Get tile number from tile row/column."""
        # e.g.
        # 001 002 003 004
        # 005 006 ...
        return ty * self.tile_count[1] + tx + 1

    def _tile_identifier(self, ty, tx):
        """Get tile identifier for numbered tiles."""
        return "T{:03d}".format(self._tile_number(ty, tx))

    def _generate_tile_info(self):
        """Get numbered tile metadata."""
        x = self.x
        y = self.y
        ts = self.tile_shape
        tc = self.tile_count

        if self._tile_cache:
            for tile_info in self._tile_cache:
                yield tile_info

        for ty in range(tc[0]):
            for tx in range(tc[1]):
                tile_id = self._tile_identifier(ty, tx)
                tile_row_offset = ty * ts[0]
                tile_column_offset = tx * ts[1]

                # store tile data to an intermediate array
                # the tile may be larger than the remaining data, handle that:
                max_row_idx = min((ty + 1) * ts[0], self._rows) - (ty * ts[0])
                max_col_idx = min((tx + 1) * ts[1], self._cols) - (tx * ts[1])
                tile_slices = (slice(0, max_row_idx), slice(0, max_col_idx))
                data_slices = (slice(ty * ts[0], (ty + 1) * ts[0]),
                               slice(tx * ts[1], (tx + 1) * ts[1]))

                tmp_x = x[data_slices[1]]
                tmp_y = y[data_slices[0]]

                tile_info = TileInfo(
                    tc, self.image_shape, ts,
                    tile_row_offset, tile_column_offset, tile_id,
                    tmp_x, tmp_y, tile_slices, data_slices)
                self._tile_cache.append(tile_info)
                yield tile_info

    def __call__(self, data):
        """Provide simple call interface for getting tile metadata."""
        if self._tile_cache:
            tile_infos = self._tile_cache
        else:
            tile_infos = self._generate_tile_info()

        for tile_info in tile_infos:
            tile_data = data[tile_info.data_slices]
            if not tile_data.size:
                LOG.info("Tile {} is empty, skipping...".format(tile_info[2]))
                continue
            yield tile_info, tile_data


class LetteredTileGenerator(NumberedTileGenerator):
    """Helper class to generate per-tile metadata for lettered tiles."""

    def __init__(self, area_definition, extents,
                 cell_size=(2000000, 2000000),
                 num_subtiles=None, use_sector_reference=False):
        """Initialize tile information for later generation."""
        # (row subtiles, col subtiles)
        self.num_subtiles = num_subtiles or (2, 2)
        self.cell_size = cell_size  # (row tile height, col tile width)
        # x/y
        self.ll_extents = extents[:2]  # (x min, y min)
        self.ur_extents = extents[2:]  # (x max, y max)
        self.use_sector_reference = use_sector_reference
        super(LetteredTileGenerator, self).__init__(area_definition)

    def _get_tile_properties(self, tile_shape, tile_count):
        """Calculate tile information for this particular sector/grid."""
        # ignore tile_shape and tile_count
        # they come from the base class, but aren't used here
        del tile_shape, tile_count

        # get original image's X/Y
        ad = self.area_definition
        x, y = ad.get_proj_vectors()

        ll_xy = self.ll_extents
        ur_xy = self.ur_extents
        cw = abs(ad.pixel_size_x)
        ch = abs(ad.pixel_size_y)
        st = self.num_subtiles
        cs = self.cell_size  # row height, column width
        # make sure the number of total tiles is a factor of the subtiles
        # meaning each letter has the full number of subtiles
        # Tile numbering/naming starts from the upper left corner
        ul_xy = (ll_xy[0], ur_xy[1])

        # Adjust the upper-left corner to 'perfectly' match the data
        # X/Y are center of pixels, adjust by half a pixels to get upper-left pixel corner
        shift_x = float(ul_xy[0] - (x.min() - cw / 2.)) % cw  # could be negative
        shift_y = float(ul_xy[1] - (y.max() + ch / 2.)) % ch  # could be negative
        # if we're really close to 0 then don't worry about it
        if abs(shift_x) < 1e-10 or abs(shift_x - cw) < 1e-10:
            shift_x = 0
        if abs(shift_y) < 1e-10 or abs(shift_y - ch) < 1e-10:
            shift_y = 0
        if self.use_sector_reference:
            LOG.debug("Adjusting X/Y by ({}, {}) so it better matches lettered grid".format(shift_x, shift_y))
            x = x + shift_x
            y = y + shift_y
        else:
            LOG.debug("Adjusting lettered grid by ({}, {}) so it better matches data X/Y".format(shift_x, shift_y))
            ul_xy = (ul_xy[0] - shift_x, ul_xy[1] - shift_y)  # outer edge of grid
            # always keep the same distance between the extents
            ll_xy = (ul_xy[0], ll_xy[1] - shift_y)
            ur_xy = (ur_xy[0] - shift_x, ul_xy[1])

        fcs_y, fcs_x = (np.ceil(float(cs[0]) / st[0]), np.ceil(float(cs[1]) / st[1]))
        # need X/Y for *whole* tiles
        max_cols = np.ceil((ur_xy[0] - ul_xy[0]) / fcs_x)
        max_rows = np.ceil((ul_xy[1] - ll_xy[1]) / fcs_y)
        # don't create partial alpha-tiles
        max_cols = int(np.ceil(max_cols / st[1]) * st[1])
        max_rows = int(np.ceil(max_rows / st[0]) * st[0])

        # make tile cell size a factor of pixel size
        num_pixels_x = int(np.floor(fcs_x / cw))
        num_pixels_y = int(np.floor(fcs_y / ch))
        # NOTE: this does not change the *total* number of columns/rows that
        # will be produced. This is important because otherwise the number
        # of lettered tiles could depend on the input data which is not what we
        # want
        fcs_x = num_pixels_x * cw
        fcs_y = num_pixels_y * ch
        # NOTE: this takes the center of the pixel relative to the upper-left outer edge:
        min_col = max(int(np.floor((x.min() - ul_xy[0]) / fcs_x)), 0)
        max_col = min(int(np.floor((x.max() - ul_xy[0]) / fcs_x)), max_cols - 1)
        min_row = max(int(np.floor((ul_xy[1] - y.max()) / fcs_y)), 0)
        max_row = min(int(np.floor((ul_xy[1] - y.min()) / fcs_y)), max_rows - 1)
        num_cols = max_col - min_col + 1
        num_rows = max_row - min_row + 1

        total_alphas = (max_cols * max_rows) / (st[0] * st[1])
        if total_alphas > 26:
            raise ValueError("Too many lettered grid cells '{}' (sector cell size too small). "
                             "Maximum of 26".format(total_alphas))

        self.tile_shape = (num_pixels_y, num_pixels_x)
        self.total_tile_count = (max_rows, max_cols)
        self.tile_count = (num_rows, num_cols)
        self.total_tiles = num_rows * num_cols
        self.image_shape = (num_pixels_y * num_rows, num_pixels_x * num_cols)
        self.min_col = min_col
        self.max_col = max_col
        self.min_row = min_row
        self.max_row = max_row
        self.ul_xy = ul_xy
        self.mx = cw
        self.bx = ul_xy[0] + cw / 2.0  # X represents the center of the pixel
        self.my = -ch
        self.by = ul_xy[1] - ch / 2.0  # Y represents the center of the pixel
        self.x = x
        self.y = y

    def _get_xy_scaling_parameters(self):
        """Get the X/Y coordinate limits for the full resulting image."""
        return self.mx, self.bx, self.my, self.by

    def _tile_identifier(self, ty, tx):
        """Get tile identifier (name) for a particular tile row/column."""
        st = self.num_subtiles
        ttc = self.total_tile_count
        alpha_num = int((ty // st[0]) * (ttc[1] // st[1]) + (tx // st[1]))
        alpha = string.ascii_uppercase[alpha_num]
        tile_num = int((ty % st[0]) * st[1] + (tx % st[1])) + 1
        return "T{}{:02d}".format(alpha, tile_num)

    def _generate_tile_info(self):
        """Create generator of individual tile metadata."""
        if self._tile_cache:
            for tile_info in self._tile_cache:
                yield tile_info

        ts = self.tile_shape
        ul_xy = self.ul_xy
        x, y = self.x, self.y
        cw = abs(float(self.area_definition.pixel_size_x))
        ch = abs(float(self.area_definition.pixel_size_y))

        # where does the data fall in our lettered grid
        for gy in range(self.min_row, self.max_row + 1):
            for gx in range(self.min_col, self.max_col + 1):
                tile_id = self._tile_identifier(gy, gx)
                # ul_xy is outer-edge of upper-left corner
                # x/y are center of each data pixel
                x_left = ul_xy[0] + gx * ts[1] * cw
                x_right = x_left + ts[1] * cw
                y_top = ul_xy[1] - gy * ts[0] * ch
                y_bot = y_top - ts[0] * ch
                x_mask = np.nonzero((x >= x_left) & (x < x_right))[0]
                y_mask = np.nonzero((y > y_bot) & (y <= y_top))[0]
                if not x_mask.any() or not y_mask.any():
                    # no data in this tile
                    LOG.debug("Tile '{}' doesn't have any data in it".format(tile_id))
                    continue
                x_slice = slice(x_mask[0], x_mask[-1] + 1)  # assume it's continuous
                y_slice = slice(y_mask[0], y_mask[-1] + 1)

                # theoretically we can precompute the X/Y now
                # instead of taking the x/y data and mapping it
                # to the tile
                tmp_x = np.ma.arange(x_left + cw / 2., x_right, cw)
                tmp_y = np.ma.arange(y_top - ch / 2., y_bot, -ch)
                data_x_idx_min = np.nonzero(np.isclose(tmp_x, x[x_slice.start]))[0][0]
                data_x_idx_max = np.nonzero(np.isclose(tmp_x, x[x_slice.stop - 1]))[0][0]
                # I have a half pixel error some where
                data_y_idx_min = np.nonzero(np.isclose(tmp_y, y[y_slice.start]))[0][0]
                data_y_idx_max = np.nonzero(np.isclose(tmp_y, y[y_slice.stop - 1]))[0][0]
                # now put the data in the grid tile

                tile_slices = (slice(data_y_idx_min, data_y_idx_max + 1),
                               slice(data_x_idx_min, data_x_idx_max + 1))
                data_slices = (y_slice, x_slice)

                tile_info = TileInfo(
                    self.tile_count, self.image_shape, ts,
                    gy * ts[0], gx * ts[1], tile_id, tmp_x, tmp_y, tile_slices, data_slices)
                self._tile_cache.append(tile_info)
                yield tile_info


class SCMIDatasetDecisionTree(DecisionTree):
    """Load AWIPS-specific metadata from YAML configuration."""

    def __init__(self, decision_dicts, **kwargs):
        """Initialize decision tree with specific keys to look for."""
        # Fields used to match a product object to it's correct configuration
        attrs = kwargs.pop('attrs',
                           ["name",
                            "standard_name",
                            "satellite",
                            "instrument",
                            "area_id",
                            "units",
                            "reader"]
                           )
        super(SCMIDatasetDecisionTree, self).__init__(decision_dicts, attrs, **kwargs)


class AttributeHelper(object):
    """Helper object which wraps around metadata to provide SCMI attributes."""

    def __init__(self, ds_info):
        """Initialize metadata for future attribute collection."""
        self.ds_info = ds_info

    def apply_attributes(self, nc, table, prefix=''):
        """Apply fixed attributes or look up attributes needed and apply them."""
        for name, value in sorted(table.items()):
            if name in nc.ncattrs():
                LOG.debug('already have a value for %s' % name)
                continue
            if value is not None:
                setattr(nc, name, value)
            else:
                funcname = prefix+name  # _global_ + product_tile_height
                func = getattr(self, funcname, None)
                if func is not None:
                    value = func()
                    if value is not None:
                        setattr(nc, name, value)
                else:
                    LOG.info('no routine matching %s' % funcname)

    def _scene_time(self):
        """Get default start time of this observation."""
        return self.ds_info["start_time"] + timedelta(minutes=int(os.environ.get("DEBUG_TIME_SHIFT", 0)))

    def _global_product_name(self):
        """Get default global product name attribute."""
        return self.ds_info["name"]

    def _global_pixel_x_size(self):
        """Get default global x size attribute."""
        return self.ds_info["area"].pixel_size_x / 1000.

    def _global_pixel_y_size(self):
        """Get default global y size attribute."""
        return self.ds_info["area"].pixel_size_y / 1000.

    def _global_start_date_time(self):
        """Get default global start time attribute."""
        when = self._scene_time()
        return when.strftime('%Y-%m-%dT%H:%M:%S')

    def _global_production_location(self):
        """Get default global production_location attribute."""
        org = os.environ.get('ORGANIZATION', None)
        if org is not None:
            return org
        else:
            LOG.warning('environment ORGANIZATION not set for .production_location attribute, using hostname')
            import socket
            return socket.gethostname()  # FUTURE: something more correct but this will do for now


class NetCDFWriter(object):
    """Write a basic AWIPS compatible NetCDF4 SCMI file representing one "tile" of data."""

    _kind = None  # 'albedo', 'brightness_temp'
    _band = None
    _include_fgf = True
    _fill_value = 0
    image_var_name = 'data'
    fgf_y = None
    fgf_x = None
    projection = None

    def __init__(self, filename, include_fgf=True, ds_info=None, compress=False,
                 is_geographic=False):
        """Initialize variable and dimension names and metadata helper objects."""
        self._nc = None
        self.filename = filename
        self._include_fgf = include_fgf
        self._compress = compress
        self.helper = AttributeHelper(ds_info)
        self.image_data = None
        self.is_geographic = is_geographic
        self.exists = os.path.isfile(self.filename)
        if self.is_geographic:
            self.row_dim_name = 'lat'
            self.col_dim_name = 'lon'
            self.y_var_name = 'lat'
            self.x_var_name = 'lon'
        else:
            self.row_dim_name = 'y'
            self.col_dim_name = 'x'
            self.y_var_name = 'y'
            self.x_var_name = 'x'

    @property
    def nc(self):
        """Access the NetCDF file object if not already created."""
        if self._nc is None:
            self._nc = Dataset(self.filename, 'r+' if self.exists else 'w')
        return self._nc

    def create_dimensions(self, lines, columns):
        """Create NetCDF dimensions."""
        # Create Dimensions
        if self.exists:
            LOG.debug("Skipping creating dimensions because file already exists.")
            return
        _nc = self.nc
        _nc.createDimension(self.row_dim_name, lines)
        _nc.createDimension(self.col_dim_name, columns)

    def create_variables(self, bitdepth, fill_value, scale_factor=None, add_offset=None,
                         valid_min=None, valid_max=None):
        """Create data and geolcoation NetCDF variables."""
        if self.exists:
            LOG.debug("Skipping creating variables because file already exists.")
            self.image_data = self.nc[self.image_var_name]
            self.fgf_y = self.nc[self.y_var_name]
            self.fgf_x = self.nc[self.x_var_name]
            return

        fgf_coords = "%s %s" % (self.y_var_name, self.x_var_name)

        self.image_data = self.nc.createVariable(self.image_var_name,
                                                 AWIPS_DATA_DTYPE,
                                                 dimensions=(self.row_dim_name, self.col_dim_name),
                                                 fill_value=fill_value,
                                                 zlib=self._compress)
        self.image_data.coordinates = fgf_coords
        self.apply_data_attributes(bitdepth, scale_factor, add_offset,
                                   valid_min=valid_min, valid_max=valid_max)

        if self._include_fgf:
            self.fgf_y = self.nc.createVariable(
                self.y_var_name, 'i2', dimensions=(self.row_dim_name,), zlib=self._compress)
            self.fgf_x = self.nc.createVariable(
                self.x_var_name, 'i2', dimensions=(self.col_dim_name,), zlib=self._compress)

    def apply_data_attributes(self, bitdepth, scale_factor, add_offset,
                              valid_min=None, valid_max=None):
        """Assign various data variable metadata."""
        # NOTE: grid_mapping is set by `set_projection_attrs`
        self.image_data.scale_factor = np.float32(scale_factor)
        self.image_data.add_offset = np.float32(add_offset)
        u = self.helper.ds_info.get('units', '1')
        self.image_data.units = UNIT_CONV.get(u, u)
        file_bitdepth = self.image_data.dtype.itemsize * 8
        is_unsigned = self.image_data.dtype.kind == 'u'
        if not AWIPS_USES_NEGATIVES and not is_unsigned:
            file_bitdepth -= 1
            is_unsigned = True

        if bitdepth >= file_bitdepth:
            bitdepth = file_bitdepth
            num_fills = 1
        else:
            bitdepth = bitdepth
            num_fills = 0
        if valid_min is not None and valid_max is not None:
            self.image_data.valid_min = valid_min
            self.image_data.valid_max = valid_max
        elif not is_unsigned:
            # signed data type
            self.image_data.valid_min = -2**(bitdepth - 1)
            # 1 less for data type (65535), another 1 less for fill value (fill value = max file value)
            self.image_data.valid_max = 2**(bitdepth - 1) - 1 - num_fills
        else:
            # unsigned data type
            self.image_data.valid_min = 0
            self.image_data.valid_max = 2**bitdepth - 1 - num_fills

        if "standard_name" in self.helper.ds_info:
            self.image_data.standard_name = self.helper.ds_info["standard_name"]
        elif self.helper.ds_info.get("standard_name") in ["reflectance", "albedo"]:
            self.image_data.standard_name = "toa_bidirectional_reflectance"
        else:
            self.image_data.standard_name = self.helper.ds_info.get("standard_name") or ''

    def set_fgf(self, x, mx, bx, y, my, by, units=None, downsample_factor=1):
        """Assign geolocation x/y variables metadata."""
        if self.exists:
            LOG.debug("Skipping setting FGF variable attributes because file already exists.")
            return

        # assign values before scale factors to avoid implicit scale reversal
        LOG.debug('y variable shape is {}'.format(self.fgf_y.shape))
        self.fgf_y.scale_factor = np.float64(my * float(downsample_factor))
        self.fgf_y.add_offset = np.float64(by)
        if self.is_geographic:
            self.fgf_y.units = units if units is not None else 'degrees_north'
            self.fgf_y.standard_name = "latitude"
        else:
            self.fgf_y.units = units if units is not None else 'meters'
            self.fgf_y.standard_name = "projection_y_coordinate"
        self.fgf_y[:] = y

        self.fgf_x.scale_factor = np.float64(mx * float(downsample_factor))
        self.fgf_x.add_offset = np.float64(bx)
        if self.is_geographic:
            self.fgf_x.units = units if units is not None else 'degrees_east'
            self.fgf_x.standard_name = "longitude"
        else:
            self.fgf_x.units = units if units is not None else 'meters'
            self.fgf_x.standard_name = "projection_x_coordinate"
        self.fgf_x[:] = x

    def set_image_data(self, data):
        """Write image variable data."""
        LOG.debug('writing image data')
        if not hasattr(data, 'mask'):
            data = np.ma.masked_array(data, np.isnan(data))
        # note: autoscaling will be applied to make int16
        self.image_data[:, :] = np.require(data, dtype=np.float32)

    def set_projection_attrs(self, area_id, proj4_info):
        """Assign projection attributes per GRB standard."""
        if self.exists:
            LOG.debug("Skipping setting projection attributes because file already exists.")
            return
        proj4_info['a'], proj4_info['b'] = proj4_radius_parameters(proj4_info)
        if proj4_info["proj"] == "geos":
            p = self.projection = self.nc.createVariable("fixedgrid_projection", 'i4')
            self.image_data.grid_mapping = "fixedgrid_projection"
            p.short_name = area_id
            p.grid_mapping_name = "geostationary"
            p.sweep_angle_axis = proj4_info.get("sweep", "y")
            p.perspective_point_height = proj4_info['h']
            p.latitude_of_projection_origin = np.float32(0.0)
            p.longitude_of_projection_origin = np.float32(proj4_info.get('lon_0', 0.0))  # is the float32 needed?
        elif proj4_info["proj"] == "lcc":
            p = self.projection = self.nc.createVariable("lambert_projection", 'i4')
            self.image_data.grid_mapping = "lambert_projection"
            p.short_name = area_id
            p.grid_mapping_name = "lambert_conformal_conic"
            p.standard_parallel = proj4_info["lat_0"]  # How do we specify two standard parallels?
            p.longitude_of_central_meridian = proj4_info["lon_0"]
            p.latitude_of_projection_origin = proj4_info.get('lat_1', proj4_info['lat_0'])  # Correct?
        elif proj4_info['proj'] == 'stere':
            p = self.projection = self.nc.createVariable("polar_projection", 'i4')
            self.image_data.grid_mapping = "polar_projection"
            p.short_name = area_id
            p.grid_mapping_name = "polar_stereographic"
            p.standard_parallel = proj4_info["lat_ts"]
            p.straight_vertical_longitude_from_pole = proj4_info.get("lon_0", 0.0)
            p.latitude_of_projection_origin = proj4_info["lat_0"]  # ?
        elif proj4_info['proj'] == 'merc':
            p = self.projection = self.nc.createVariable("mercator_projection", 'i4')
            self.image_data.grid_mapping = "mercator_projection"
            p.short_name = area_id
            p.grid_mapping_name = "mercator"
            p.standard_parallel = proj4_info.get('lat_ts', proj4_info.get('lat_0', 0.0))
            p.longitude_of_projection_origin = proj4_info.get("lon_0", 0.0)
        # AWIPS 2 Doesn't actually support this yet
        # elif proj4_info['proj'] in ['latlong', 'longlat', 'lonlat', 'latlon']:
        #     p = self.projection = self._nc.createVariable("latitude_longitude_projection", 'i4')
        #     self.image_data.grid_mapping = "latitude_longitude_projection"
        #     p.short_name = area_id
        #     p.grid_mapping_name = 'latitude_longitude'
        else:
            raise ValueError("SCMI can not handle projection '{}'".format(proj4_info['proj']))

        p.semi_major_axis = np.float64(proj4_info["a"])
        p.semi_minor_axis = np.float64(proj4_info["b"])
        p.false_easting = np.float32(proj4_info.get("x", 0.0))
        p.false_northing = np.float32(proj4_info.get("y", 0.0))

    def set_global_attrs(self, physical_element, awips_id, sector_id,
                         creating_entity, total_tiles, total_pixels,
                         tile_row, tile_column, tile_height, tile_width, creator=None):
        """Assign NetCDF global attributes."""
        if self.exists:
            LOG.debug("Skipping setting global attributes because file already exists.")
            return

        self.nc.Conventions = "CF-1.7"
        if creator is None:
            from satpy import __version__
            self.nc.creator = "Satpy Version {} - SCMI Writer".format(__version__)
        else:
            self.nc.creator = creator
        self.nc.creation_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        # name as it shows in the product browser (physicalElement)
        self.nc.physical_element = physical_element
        self.nc.satellite_id = creating_entity
        # identifying name to match against AWIPS common descriptions (ex. "AWIPS_product_name")
        self.nc.awips_id = awips_id
        self.nc.sector_id = sector_id
        self.nc.tile_row_offset = tile_row
        self.nc.tile_column_offset = tile_column
        self.nc.product_tile_height = tile_height
        self.nc.product_tile_width = tile_width
        self.nc.number_product_tiles = total_tiles[0] * total_tiles[1]
        self.nc.product_rows = total_pixels[0]
        self.nc.product_columns = total_pixels[1]

        self.helper.apply_attributes(self.nc, SCMI_GLOBAL_ATT, '_global_')

    def close(self):
        """Close the NetCDF file if created."""
        if self._nc is not None:
            self._nc.sync()
            self._nc.close()
            self._nc = None


class NetCDFWrapper(object):
    """Object to wrap all NetCDF data-based operations in to a single call.

    This makes it possible to do SCMI writing with dask's delayed `da.store` function.

    """

    def __init__(self, filename, sector_id, ds_info, awips_info,
                 xy_factors, tile_info, compress=False, fix_awips=False,
                 update_existing=True):
        """Assign instance attributes for later use."""
        self.filename = filename
        self.sector_id = sector_id
        self.ds_info = ds_info
        self.awips_info = awips_info
        self.tile_info = tile_info
        self.xy_factors = xy_factors
        self.compress = compress
        self.fix_awips = fix_awips
        self.update_existing = update_existing
        self.exists = os.path.isfile(self.filename)

    def __setitem__(self, key, data):
        """Write an entire tile to a file."""
        if np.isnan(data).all():
            LOG.info("Tile {} contains all invalid data, skipping...".format(self.filename))
            return

        ds_info = self.ds_info
        awips_info = self.awips_info
        tile_info = self.tile_info
        area_def = ds_info['area']
        if hasattr(area_def, 'crs'):
            is_geographic = area_def.crs.is_geographic
        else:
            is_geographic = Proj(area_def.proj_dict).is_latlong()
        nc = NetCDFWriter(self.filename, ds_info=self.ds_info,
                          compress=self.compress,
                          is_geographic=is_geographic)

        LOG.debug("Scaling %s data to fit in netcdf file...", ds_info["name"])
        bit_depth = ds_info.get("bit_depth", 16)
        valid_min = ds_info.get('valid_min')
        if valid_min is None and self.update_existing and self.exists:
            # reuse the valid_min that was previously computed
            valid_min = nc.nc['data'].valid_min
        elif valid_min is None:
            valid_min = np.nanmin(data)

        valid_max = ds_info.get('valid_max')
        if valid_max is None and self.update_existing and self.exists:
            # reuse the valid_max that was previously computed
            valid_max = nc.nc['data'].valid_max
        elif valid_max is None:
            valid_max = np.nanmax(data)

        LOG.debug("Using product valid min {} and valid max {}".format(valid_min, valid_max))
        is_cat = 'flag_meanings' in ds_info
        fills, factor, offset = self._calc_factor_offset(
            data=data, bitdepth=bit_depth, min=valid_min, max=valid_max, dtype=AWIPS_DATA_DTYPE, flag_meanings=is_cat)
        if is_cat:
            data = data.astype(AWIPS_DATA_DTYPE)

        tmp_tile = np.empty(tile_info.tile_shape, dtype=data.dtype)
        tmp_tile[:] = np.nan

        LOG.info("Writing tile '%s' to '%s'", self.tile_info[2], self.filename)
        LOG.debug("Creating dimensions...")
        nc.create_dimensions(tmp_tile.shape[0], tmp_tile.shape[1])
        LOG.debug("Creating variables...")
        nc.create_variables(bit_depth, fills[0], factor, offset)
        LOG.debug("Creating global attributes...")
        nc.set_global_attrs(awips_info['physical_element'],
                            awips_info['awips_id'], self.sector_id,
                            awips_info['creating_entity'],
                            tile_info.tile_count, tile_info.image_shape,
                            tile_info.tile_row_offset, tile_info.tile_column_offset,
                            tmp_tile.shape[0], tmp_tile.shape[1])
        LOG.debug("Creating projection attributes...")
        nc.set_projection_attrs(area_def.area_id, area_def.proj_dict)
        LOG.debug("Writing X/Y navigation data...")
        mx, bx, my, by = self.xy_factors
        nc.set_fgf(tile_info.x, mx, bx, tile_info.y, my, by)

        tmp_tile[tile_info.tile_slices] = data
        if self.exists and self.update_existing:
            # use existing data where possible
            existing_data = nc.nc['data'][:]
            # where we don't have new data but we also have good existing data
            old_mask = np.isnan(tmp_tile) & ~existing_data.mask
            tmp_tile[old_mask] = existing_data[old_mask]

        LOG.debug("Writing image data...")
        np.clip(tmp_tile, valid_min, valid_max, out=tmp_tile)
        nc.set_image_data(tmp_tile)
        nc.close()

        if self.fix_awips and not self.exists:
            fix_awips_file(self.filename)

    def _calc_factor_offset(self, data=None, dtype=np.int16, bitdepth=None,
                            min=None, max=None, num_fills=1, flag_meanings=False):
        """Compute netcdf variable factor and offset."""
        if num_fills > 1:
            raise NotImplementedError("More than one fill value is not implemented yet")

        dtype = np.dtype(dtype)
        file_bitdepth = dtype.itemsize * 8
        is_unsigned = dtype.kind == 'u'
        if not AWIPS_USES_NEGATIVES and not is_unsigned:
            file_bitdepth -= 1
            is_unsigned = True

        if bitdepth is None:
            bitdepth = file_bitdepth
        if bitdepth >= file_bitdepth:
            bitdepth = file_bitdepth
        else:
            # don't take away from the data bitdepth if there is room in
            # file data type to allow for extra fill values
            num_fills = 0
        if min is None:
            min = data.min()
        if max is None:
            max = data.max()

        if not is_unsigned:
            # max value
            fills = [2**(file_bitdepth - 1) - 1]
        else:
            # max value
            fills = [2**file_bitdepth - 1]

        if flag_meanings:
            # AWIPS doesn't like Identity conversion so we can't have
            # a factor of 1 and an offset of 0
            mx = 0.5
            bx = 0
        else:
            mx = float(max - min) / (2**bitdepth - 1 - num_fills)
            bx = min
            if not is_unsigned:
                bx += 2**(bitdepth - 1) * mx

        return fills, mx, bx


class SCMIWriter(Writer):
    """Writer for AWIPS NetCDF4 SCMI files.

    These files are **not** the official GOES-R style files, but rather a
    custom "Polar SCMI" file scheme originally developed at the University
    of Wisconsin - Madison, Space Science and Engineering Center (SSEC) for
    use by the CSPP Polar2Grid project. Despite the name these files should
    support data from polar-orbitting satellites (after resampling) and
    geostationary satellites in single band (luminance) or RGB image format.

    """

    def __init__(self, compress=False, fix_awips=False, **kwargs):
        """Initialize writer and decision trees."""
        super(SCMIWriter, self).__init__(default_config_filename="writers/scmi.yaml", **kwargs)
        self.keep_intermediate = False
        self.overwrite_existing = True
        self.scmi_sectors = self.config['sectors']
        self.scmi_datasets = SCMIDatasetDecisionTree([self.config['datasets']])
        self.compress = compress
        self.fix_awips = fix_awips
        self._fill_sector_info()
        self._enhancer = None

    @property
    def enhancer(self):
        """Get lazy loaded enhancer object only if needed."""
        if self._enhancer is None:
            self._enhancer = Enhancer(ppp_config_dir=self.ppp_config_dir)
        return self._enhancer

    @classmethod
    def separate_init_kwargs(cls, kwargs):
        """Separate keyword arguments by initialization and saving keyword arguments."""
        # FUTURE: Don't pass Scene.save_datasets kwargs to init and here
        init_kwargs, kwargs = super(SCMIWriter, cls).separate_init_kwargs(
            kwargs)
        for kw in ['compress', 'fix_awips']:
            if kw in kwargs:
                init_kwargs[kw] = kwargs.pop(kw)

        return init_kwargs, kwargs

    def _fill_sector_info(self):
        """Convert sector extents if needed."""
        for sector_info in self.scmi_sectors.values():
            p = Proj(sector_info['projection'])
            if 'lower_left_xy' in sector_info:
                sector_info['lower_left_lonlat'] = p(*sector_info['lower_left_xy'], inverse=True)
            else:
                sector_info['lower_left_xy'] = p(*sector_info['lower_left_lonlat'])
            if 'upper_right_xy' in sector_info:
                sector_info['upper_right_lonlat'] = p(*sector_info['upper_right_xy'], inverse=True)
            else:
                sector_info['upper_right_xy'] = p(*sector_info['upper_right_lonlat'])

    def _get_sector_info(self, sector_id, lettered_grid):
        """Get metadata for the current sector if configured.

        This is not necessary for numbered grids. If found, the sector info
        will provide the overall tile layout for this grid/sector. This allows
        for consistent tile numbering/naming regardless of where the data being
        converted actually is.

        """
        try:
            sector_info = self.scmi_sectors[sector_id]
        except KeyError:
            if lettered_grid:
                raise ValueError("Unknown sector '{}'".format(sector_id))
            else:
                sector_info = None
        return sector_info

    def _get_tile_generator(self, area_def, lettered_grid, sector_id,
                            num_subtiles, tile_size, tile_count,
                            use_sector_reference=False):
        """Get the appropriate tile generator class for lettered or numbered tiles."""
        sector_info = self._get_sector_info(sector_id, lettered_grid)
        # Create a tile generator for this grid definition
        if lettered_grid:
            tile_gen = LetteredTileGenerator(
                area_def,
                sector_info['lower_left_xy'] + sector_info['upper_right_xy'],
                cell_size=sector_info['resolution'],
                num_subtiles=num_subtiles,
                use_sector_reference=use_sector_reference,
                )
        else:
            tile_gen = NumberedTileGenerator(
                area_def,
                tile_shape=tile_size,
                tile_count=tile_count,
            )
        return tile_gen

    def _get_awips_info(self, ds_info, source_name=None, physical_element=None):
        """Get metadata for this product when shown in AWIPS if configured in the YAML file."""
        try:
            awips_info = self.scmi_datasets.find_match(**ds_info).copy()
            awips_info['awips_id'] = "AWIPS_" + ds_info['name']

            if not physical_element:
                physical_element = awips_info.get('physical_info')
            if not physical_element:
                physical_element = ds_info['name']
            if "{" in physical_element:
                physical_element = physical_element.format(**ds_info)
            awips_info['physical_element'] = physical_element

            if source_name:
                awips_info['source_name'] = source_name
            if awips_info['source_name'] is None:
                raise TypeError("'source_name' keyword must be specified")

            def_ce = "{}-{}".format(ds_info["platform_name"].upper(), ds_info["sensor"].upper())
            awips_info.setdefault('creating_entity', def_ce)
            return awips_info
        except KeyError:
            LOG.error("Could not get information on dataset from backend configuration file")
            raise

    def _group_by_area(self, datasets):
        """Group datasets by their area."""
        def _area_id(area_def):
            return area_def.name + str(area_def.area_extent) + str(area_def.shape)

        # get all of the datasets stored by area
        area_datasets = {}
        for x in datasets:
            area_id = _area_id(x.attrs['area'])
            area, ds_list = area_datasets.setdefault(area_id, (x.attrs['area'], []))
            ds_list.append(x)
        return area_datasets

    def _split_rgbs(self, ds):
        """Split a single RGB dataset in to multiple."""
        for component in 'RGB':
            band_data = ds.sel(bands=component)
            band_data.attrs['name'] += '_{}'.format(component)
            band_data.attrs['valid_min'] = 0.0
            band_data.attrs['valid_max'] = 1.0
            yield band_data

    def _enhance_and_split_rgbs(self, datasets):
        """Handle multi-band images by splitting in to separate products."""
        new_datasets = []
        for ds in datasets:
            if ds.ndim == 2:
                new_datasets.append(ds)
                continue
            elif ds.ndim > 3 or ds.ndim < 1 or (ds.ndim == 3 and 'bands' not in ds.coords):
                LOG.error("Can't save datasets with more or less than 2 dimensions "
                          "that aren't RGBs to SCMI format: {}".format(ds.name))
            else:
                # this is an RGB
                img = get_enhanced_image(ds.squeeze(), enhance=self.enhancer)
                res_data = img.finalize(fill_value=0, dtype=np.float32)[0]
                new_datasets.extend(self._split_rgbs(res_data))

        return new_datasets

    def save_dataset(self, dataset, **kwargs):
        """Save a single DataArray to one or more NetCDF4 SCMI files."""
        LOG.warning("For best performance use `save_datasets`")
        return self.save_datasets([dataset], **kwargs)

    def get_filename(self, area_def, tile_info, sector_id, **kwargs):
        """Generate output NetCDF file from metadata."""
        # format the filename
        kwargs["start_time"] += timedelta(minutes=int(os.environ.get("DEBUG_TIME_SHIFT", 0)))
        return super(SCMIWriter, self).get_filename(
            area_id=area_def.area_id,
            rows=area_def.y_size,
            columns=area_def.x_size,
            sector_id=sector_id,
            tile_id=tile_info.tile_id,
            **kwargs)

    def check_tile_exists(self, output_filename):
        """Check if tile exists and report error accordingly."""
        if os.path.isfile(output_filename):
            if not self.overwrite_existing:
                LOG.error("AWIPS file already exists: %s", output_filename)
                raise RuntimeError("AWIPS file already exists: %s" % (output_filename,))
            else:
                LOG.info("AWIPS file already exists, will update with new data: %s", output_filename)

    def save_datasets(self, datasets, sector_id=None,
                      source_name=None, filename=None,
                      tile_count=(1, 1), tile_size=None,
                      lettered_grid=False, num_subtiles=None,
                      use_end_time=False, use_sector_reference=False,
                      compute=True, **kwargs):
        """Write a series of DataArray objects to multiple NetCDF4 SCMI files.

        Args:
            datasets (iterable): Series of gridded :class:`~xarray.DataArray`
                objects with the necessary metadata to be converted to a valid
                SCMI product file.
            sector_id (str): Name of the region or sector that the provided
                data is on. This name will be written to the NetCDF file and
                will be used as the sector in the AWIPS client. For lettered
                grids this name should match the name configured in the writer
                YAML. This is required but is defined as a keyword argument
                for better error handling in Satpy.
            source_name (str): Name of producer of these files (ex. "SSEC").
                This name is used to create the output filename.
            filename (str): Filename format pattern to be filled in with
                dataset metadata for each tile. See YAML configuration file
                for default.
            tile_count (tuple): For numbered tiles only, how many tile rows
                and tile columns to produce. Default to ``(1, 1)``, a single
                giant tile. Either ``tile_count``, ``tile_size``, or
                ``lettered_grid`` should be specified.
            tile_size (tuple): For numbered tiles only, how many pixels each
                tile should be. This takes precedence over ``tile_count`` if
                specified. Either ``tile_count``, ``tile_size``, or
                ``lettered_grid`` should be specified.
            lettered_grid (bool): Whether to use a preconfigured grid and
                label tiles with letters and numbers instead of only numbers.
                For example, tiles will be named "A01", "A02", "B01", and so
                on in the first row of data and continue on to "A03", "A04",
                and "B03" in the default case where ``num_subtiles`` is (2, 2).
                Letters start in the upper-left corner and will go from A up to
                Z, if necessary.
            num_subtiles (tuple): For lettered tiles only, how many rows and
                columns to split each lettered tile in to. By default 2 rows
                and 2 columns will be created. For example, the tile for
                letter "A" will have "A01" and "A02" in the top row and "A03"
                and "A04" in the second row.
            use_end_time (bool): Instead of using the ``start_time`` for the
                product filename and time written to the file, use the
                ``end_time``. This is useful for multi-day composites where
                the ``end_time`` is a better representation of what data is
                in the file.
            use_sector_reference (bool): For lettered tiles only, whether to
                shift the data locations to align with the preconfigured
                grid's pixels. By default this is False meaning that the
                grid's tiles will be shifted to align with the data locations.
                If True, the data is shifted. At most the data will be shifted
                by 0.5 pixels. See :mod:`satpy.writers.scmi` for more
                information.
            compute (bool): Compute and write the output immediately using
                dask. Default to ``False``.

        """
        if sector_id is None:
            raise TypeError("Keyword 'sector_id' is required")

        area_datasets = self._group_by_area(datasets)
        sources_targets = []
        for area_def, ds_list in area_datasets.values():
            tile_gen = self._get_tile_generator(
                area_def, lettered_grid, sector_id, num_subtiles, tile_size,
                tile_count, use_sector_reference=use_sector_reference)
            for dataset in self._enhance_and_split_rgbs(ds_list):
                LOG.info("Preparing product %s to be written to AWIPS SCMI NetCDF file", dataset.attrs["name"])
                awips_info = self._get_awips_info(dataset.attrs, source_name=source_name)
                for tile_info, tmp_tile in tile_gen(dataset):
                    # make sure this entire tile is loaded as one single array
                    tmp_tile.data = tmp_tile.data.rechunk(tmp_tile.shape)
                    ds_info = dataset.attrs.copy()
                    if use_end_time:
                        # replace start_time with end_time for multi-day composites
                        ds_info['start_time'] = ds_info['end_time']

                    output_filename = filename or self.get_filename(area_def, tile_info, sector_id,
                                                                    source_name=awips_info['source_name'],
                                                                    **ds_info)
                    self.check_tile_exists(output_filename)
                    nc_wrapper = NetCDFWrapper(output_filename, sector_id, ds_info, awips_info,
                                               tile_gen.xy_factors, tile_info,
                                               compress=self.compress, fix_awips=self.fix_awips)
                    sources_targets.append((tmp_tile.data, nc_wrapper))

        if compute and sources_targets:
            # the NetCDF creation is per-file so we don't need to lock
            return da.store(*zip(*sources_targets), lock=False)
        return sources_targets


def _create_debug_array(sector_info, num_subtiles, font_path='Verdana.ttf'):
    from PIL import Image, ImageDraw, ImageFont
    from pkg_resources import resource_filename as get_resource_filename
    size = (1000, 1000)
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)

    if ':' in font_path:
        # load from a python package
        font_path = get_resource_filename(*font_path.split(':'))
    font = ImageFont.truetype(font_path, 25)

    ll_extent = sector_info['lower_left_xy']
    ur_extent = sector_info['upper_right_xy']
    total_meters_x = ur_extent[0] - ll_extent[0]
    total_meters_y = ur_extent[1] - ll_extent[1]
    fcs_x = np.ceil(float(sector_info['resolution'][1]) / num_subtiles[1])
    fcs_y = np.ceil(float(sector_info['resolution'][0]) / num_subtiles[0])
    total_cells_x = np.ceil(total_meters_x / fcs_x)
    total_cells_y = np.ceil(total_meters_y / fcs_y)
    total_cells_x = np.ceil(total_cells_x / num_subtiles[1]) * num_subtiles[1]
    total_cells_y = np.ceil(total_cells_y / num_subtiles[0]) * num_subtiles[0]
    # total_alpha_cells_x = int(total_cells_x / num_subtiles[1])
    # total_alpha_cells_y = int(total_cells_y / num_subtiles[0])

    # "round" the total meters up to the number of alpha cells
    # total_meters_x = total_cells_x * fcs_x
    # total_meters_y = total_cells_y * fcs_y

    # Pixels per tile
    ppt_x = np.floor(float(size[0]) / total_cells_x)
    ppt_y = np.floor(float(size[1]) / total_cells_y)
    half_ppt_x = np.floor(ppt_x / 2.)
    half_ppt_y = np.floor(ppt_y / 2.)
    # Meters per pixel
    meters_ppx = fcs_x / ppt_x
    meters_ppy = fcs_y / ppt_y
    for idx, alpha in enumerate(string.ascii_uppercase):
        for i in range(4):
            st_x = i % num_subtiles[1]
            st_y = int(i / num_subtiles[1])
            t = "{}{:02d}".format(alpha, i + 1)
            t_size = font.getsize(t)
            cell_x = (idx * num_subtiles[1] + st_x) % total_cells_x
            cell_y = int(idx / (total_cells_x / num_subtiles[1])) * num_subtiles[0] + st_y
            if cell_x > total_cells_x:
                continue
            elif cell_y > total_cells_y:
                continue
            x = ppt_x * cell_x + half_ppt_x
            y = ppt_y * cell_y + half_ppt_y
            # draw box around the tile edge
            # PIL Documentation: "The second point is just outside the drawn rectangle."
            # we want to be just inside 0 and just inside the outer edge of the tile
            draw_rectangle(draw,
                           (x - half_ppt_x, y - half_ppt_y,
                            x + half_ppt_x, y + half_ppt_y), outline=255, fill=75, width=3)
            draw.text((x - t_size[0] / 2., y - t_size[1] / 2.), t, fill=255, font=font)

    img.save("test.png")

    from pyresample.utils import proj4_str_to_dict
    new_extents = (
        ll_extent[0],
        ur_extent[1] - 1001. * meters_ppy,
        ll_extent[0] + 1001. * meters_ppx,
        ur_extent[1],
    )
    grid_def = AreaDefinition(
        'debug_grid',
        'debug_grid',
        'debug_grid',
        proj4_str_to_dict(sector_info['projection']),
        1000,
        1000,
        new_extents
    )
    return grid_def, np.array(img)


def draw_rectangle(draw, coordinates, outline=None, fill=None, width=1):
    """Draw simple rectangle in to a numpy array image."""
    for i in range(width):
        rect_start = (coordinates[0] + i, coordinates[1] + i)
        rect_end = (coordinates[2] - i, coordinates[3] - i)
        draw.rectangle((rect_start, rect_end), outline=outline, fill=fill)


def create_debug_lettered_tiles(init_args, create_args):
    """Create SCMI files with tile identifiers "burned" in to the image data for debugging."""
    import xarray as xr
    create_args['lettered_grid'] = True
    create_args['num_subtiles'] = (2, 2)  # default, don't use command line argument

    writer = SCMIWriter(**init_args)

    sector_id = create_args['sector_id']
    sector_info = writer.scmi_sectors[sector_id]
    area_def, arr = _create_debug_array(sector_info, create_args['num_subtiles'])

    now = datetime.utcnow()
    product = xr.DataArray(da.from_array(arr, chunks='auto'), attrs=dict(
        name='debug_{}'.format(sector_id),
        platform_name='DEBUG',
        sensor='TILES',
        start_time=now,
        end_time=now,
        area=area_def,
        standard_name="toa_bidirectional_reflectance",
        units='1',
        valid_min=0,
        valid_max=255,
    ))
    created_files = writer.save_dataset(
        product,
        **create_args
    )
    return created_files


def add_backend_argument_groups(parser):
    """Add command line arguments for this writer used for debugging."""
    group_1 = parser.add_argument_group(title="Backend Initialization")
    group_1.add_argument("--backend-configs", nargs="*", dest="backend_configs",
                         help="alternative backend configuration files")
    group_1.add_argument("--compress", action="store_true",
                         help="zlib compress each netcdf file")
    group_1.add_argument("--fix-awips", action="store_true",
                         help="modify NetCDF output to work with the old/broken AWIPS NetCDF library")
    group_2 = parser.add_argument_group(title="Backend Output Creation")
    group_2.add_argument("--tiles", dest="tile_count", nargs=2, type=int, default=[1, 1],
                         help="Number of tiles to produce in Y (rows) and X (cols) direction respectively")
    group_2.add_argument("--tile-size", dest="tile_size", nargs=2, type=int, default=None,
                         help="Specify how many pixels are in each tile (overrides '--tiles')")
    # group.add_argument('--tile-offset', nargs=2, default=(0, 0),
    #                    help="Start counting tiles from this offset ('row_offset col_offset')")
    group_2.add_argument("--letters", dest="lettered_grid", action='store_true',
                         help="Create tiles from a static letter-based grid based on the product projection")
    group_2.add_argument("--letter-subtiles", nargs=2, type=int, default=(2, 2),
                         help="Specify number of subtiles in each lettered tile: \'row col\'")
    group_2.add_argument("--output-pattern", default=DEFAULT_OUTPUT_PATTERN,
                         help="output filenaming pattern")
    group_2.add_argument("--source-name", default='SSEC',
                         help="specify processing source name used in attributes and filename (default 'SSEC')")
    group_2.add_argument("--sector-id", required=True,
                         help="specify name for sector/region used in attributes and filename (example 'LCC')")
    return group_1, group_2


def main():
    """Command line interface mimicing CSPP Polar2Grid."""
    import argparse
    parser = argparse.ArgumentParser(description="Create SCMI AWIPS compatible NetCDF files")
    subgroups = add_backend_argument_groups(parser)
    parser.add_argument("--create-debug", action='store_true',
                        help='Create debug NetCDF files to show tile locations in AWIPS')
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through '
                             'ERROR-WARNING-INFO-DEBUG (default INFO)')
    parser.add_argument('-l', '--log', dest="log_fn", default=None,
                        help="specify the log filename")
    args = parser.parse_args()

    init_args = {ga.dest: getattr(args, ga.dest) for ga in subgroups[0]._group_actions}
    create_args = {ga.dest: getattr(args, ga.dest) for ga in subgroups[1]._group_actions}

    # Logs are renamed once data the provided start date is known
    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=levels[min(3, args.verbosity)], filename=args.log_fn)

    if args.create_debug:
        create_debug_lettered_tiles(init_args, create_args)
        return
    else:
        raise NotImplementedError("Command line interface not implemented yet for SCMI writer")


if __name__ == '__main__':
    sys.exit(main())
