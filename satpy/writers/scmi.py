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
import xarray as xr

import numpy as np
from pyproj import Proj
import dask
import dask.array as da
from satpy.writers import Writer, DecisionTree, Enhancer, get_enhanced_image
from satpy import __version__
from pyresample.geometry import AreaDefinition
from trollsift.parser import StringFormatter
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
        self._rows = self.area_definition.height
        self._cols = self.area_definition.width

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
            gd.area_extent[1] - ps_y * (imaginary_data_size[0] - gd.height),
            gd.area_extent[2] + ps_x * (imaginary_data_size[1] - gd.width),
            gd.area_extent[3])
        imaginary_grid_def = AreaDefinition(
            gd.area_id,
            gd.description,
            gd.proj_id,
            gd.proj_dict,
            imaginary_data_size[1],
            imaginary_data_size[0],
            new_extents,
        )

        x, y = imaginary_grid_def.get_proj_coords()
        x = x[0].squeeze()  # all rows should have the same coordinates
        y = y[:, 0].squeeze()  # all columns should have the same coordinates
        return x, y

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

    def __call__(self):
        """Provide simple call interface for getting tile metadata."""
        if self._tile_cache:
            tile_infos = self._tile_cache
        else:
            tile_infos = self._generate_tile_info()

        for tile_info in tile_infos:
            # TODO: Return the slice instead of the actual data array
            #   Use the slicing start/end to determine if it is empty
            # tile_data = data[tile_info.data_slices]
            # if not tile_data.size:
            #     LOG.info("Tile {} is empty, skipping...".format(tile_info[2]))
            #     continue
            yield tile_info


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


def _get_factor_offset_fill(input_data_arr, vmin, vmax, encoding):
    dtype_str = encoding['dtype']
    dtype = np.dtype(getattr(np, dtype_str))
    file_bit_depth = dtype.itemsize * 8
    unsigned_in_signed = encoding.get('_Unsigned') == "true"
    is_unsigned = dtype.kind == 'u'
    bit_depth = input_data_arr.attrs.get('bit_depth', file_bit_depth)
    num_fills = 1  # future: possibly support more than one fill value
    if bit_depth is None:
        bit_depth = file_bit_depth
    if bit_depth >= file_bit_depth:
        bit_depth = file_bit_depth
    else:
        # don't take away from the data bit depth if there is room in
        # file data type to allow for extra fill values
        num_fills = 0

    if is_unsigned or unsigned_in_signed:
        # max value
        fills = [2 ** file_bit_depth - 1]
    else:
        # max value
        fills = [2 ** (file_bit_depth - 1) - 1]

    mx = float(vmax - vmin) / (2 ** bit_depth - 1 - num_fills)
    bx = vmin
    if not is_unsigned and not unsigned_in_signed:
        bx += 2 ** (bit_depth - 1) * mx
    return mx, bx, fills[0]


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
            self._nc = Dataset(self.filename, 'r+' if self.exists else 'w')  # noqa
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
    """Object to wrap all NetCDF data-based operations into a single call.

    This makes it possible to do SCMI writing with dask's delayed `da.store` function.

    """

    def __init__(self, filename, sector_id, ds_infos, awips_info,
                 xy_factors, tile_info, compress=False, fix_awips=False,
                 update_existing=True):
        """Assign instance attributes for later use."""
        self.filename = filename
        self.sector_id = sector_id
        self.ds_infos = ds_infos
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

        if len(self.ds_infos) > 1:
            raise NotImplementedError("Can't handle multiple variables in one file yet.")
        ds_info = self.ds_infos[0]
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


class NetCDFTemplate:
    """Helper class to convert a dictionary-based NetCDF template to an :class:`xarray.Dataset`."""

    def __init__(self, template_dict):
        """Parse template dictionary and prepare for rendering."""
        self.is_single_variable = template_dict.get('single_variable', False)
        self.global_attributes = template_dict.get('global_attributes', {})

        default_var_config = {
            "default": {
                "encoding": {"dtype": "uint16"},
            }
        }
        self.variables = template_dict.get('variables', default_var_config)

        default_coord_config = {
            "default": {
                "encoding": {"dtype": "uint16"},
            }
        }
        self.coordinates = template_dict.get('coordinates', default_coord_config)

        self._var_tree = SCMIDatasetDecisionTree([self.variables])
        self._coord_tree = SCMIDatasetDecisionTree([self.coordinates])
        self._str_formatter = StringFormatter()

    def _get_attr_value(self, attr_name, input_metadata, value=None, raw_key=None, raw_value=None, prefix="_"):
        """Determine attribute value using the provided configuration information.

        If `value` and `raw_key` are not provided, this method will search
        for a method named ``<prefix><attr_name>``, which will be called with
        one argument (`input_metadata`) to get the value to return. See
        the documentation for the `prefix` keyword argument below for more
        information.

        Args:
            attr_name (str): Name of the attribute whose value we are
                generating.
            input_metadata (dict): Dictionary of metadata from the input
                DataArray and other context information. Used to provide
                information to `value` or access data from using `raw_key`
                if provided.
            value (Any): Value to assign to this attribute. If a string, it
                may be a python format string which will be provided the data
                from `input_metadata`. For example, ``{name}`` will be filled
                with the value for the ``"name"`` in `input_metadata`. It can
                also include environment variables (ex. ``"${MY_ENV_VAR}"``)
                which will be expanded. String formatting is accomplished by
                the special :class:`trollsift.parser.StringFormatter` which
                allows for special common conversions.
            raw_key (str): Key to access value from `input_metadata`, but
                without any string formatting applied to it. This allows for
                metadata of non-string types to be requested.
            raw_value (Any): Static hardcoded value to set this attribute
                to. Overrides all other options.
            prefix (bool): Prefix to use when `value` and `raw_key` are
                both ``None``. Default is ``"_"``. This will be used to find
                custom attribute handlers in subclasses. For example, if
                `value` and `raw_key` are both ``None`` and `attr_name`
                is ``"my_attr"``, then the method ``self._my_attr`` will be
                called as ``return self._my_attr(input_metadata)``.
                See :meth:`NetCDFTemplate.render_global_attributes` for
                additional information (prefix is ``"_global_"``).

        """
        if raw_value is not None:
            return raw_value
        if raw_key is not None and raw_key in input_metadata:
            value = input_metadata[raw_key]
            return value

        if isinstance(value, str):
            try:
                value = os.path.expandvars(value)
                value = self._str_formatter.format(value, **input_metadata)
            except (KeyError, ValueError):
                LOG.debug("Can't format string '{}' with provided "
                          "input metadata.".format(value))
                value = None
                # raise ValueError("Can't format string '{}' with provided "
                #                  "input metadata.".format(value))
        if value is not None:
            return value

        meth_name = prefix + attr_name
        func = getattr(self, meth_name, None)
        if func is not None:
            value = func(input_metadata)
        if value is not None:
            return value
        else:
            LOG.debug('no routine matching %s' % (meth_name,))

    def _render_attrs(self, attr_configs, input_metadata, prefix="_"):
        attrs = {}
        for attr_name, attr_config_dict in attr_configs.items():
            val = self._get_attr_value(attr_name, input_metadata,
                                       prefix=prefix, **attr_config_dict)
            if val is None:
                # NetCDF attributes can't have a None value
                continue
            attrs[attr_name] = val
        return attrs

    def _render_global_attributes(self, input_metadata):
        attr_configs = self.global_attributes
        return self._render_attrs(attr_configs, input_metadata,
                                  prefix="_global_")

    def _render_variable_attributes(self, var_config, input_metadata):
        attr_configs = var_config['attributes']
        var_attrs = self._render_attrs(attr_configs, input_metadata, prefix="_data_")

        # Add variables that should be applied to all variables
        # TODO: valid_min - based on dtype and needs to handle _Unsigned.
        #   Also need to convert the valid min and valid max if specified by the DataArray?
        #   Double check what I did before.
        #   Need to handle _FillValue too (don't say that 0 is valid_min if it is a _FillValue)
        # TODO: valid_max
        return var_attrs

    def _render_coordinate_attributes(self, coord_config, input_metadata):
        attr_configs = coord_config['attributes']
        coord_attrs = self._render_attrs(attr_configs, input_metadata, prefix="_coord_")
        return coord_attrs

    def _render_variable_encoding(self, var_config, input_data_arr):
        new_encoding = input_data_arr.encoding.copy()
        # determine fill value and
        if 'encoding' in var_config:
            new_encoding.update(var_config['encoding'])
        # handled during delayed 'to_netcdf' by taking min/max of data
        new_encoding.setdefault('scale_factor', 'auto')
        new_encoding.setdefault('add_offset', 'auto')
        new_encoding.setdefault('_FillValue', 'auto')
        return new_encoding

    def _render_variable(self, data_arr):
        var_config = self._var_tree.find_match(**data_arr.attrs)
        new_var_name = var_config.get('var_name', data_arr.attrs['name'])
        new_data_arr = data_arr.copy()
        # remove coords which may cause issues later on
        new_data_arr = new_data_arr.reset_coords(drop=True)

        var_encoding = self._render_variable_encoding(var_config, data_arr)
        new_data_arr.encoding = var_encoding
        var_attrs = self._render_variable_attributes(var_config, data_arr.attrs)
        new_data_arr.attrs = var_attrs
        return new_var_name, new_data_arr

    def _get_matchable_coordinate_metadata(self, coord_name, coord_attrs):
        match_kwargs = {}
        if 'name' not in coord_attrs:
            match_kwargs['name'] = coord_name
        match_kwargs.update(coord_attrs)
        return match_kwargs

    def _render_coordinates(self, ds):
        new_coords = {}
        for coord_name, coord_arr in ds.coords.items():
            match_kwargs = self._get_matchable_coordinate_metadata(coord_name, coord_arr.attrs)
            coord_config = self._coord_tree.find_match(**match_kwargs)
            coord_attrs = self._render_coordinate_attributes(coord_config, coord_arr.attrs)
            coord_encoding = self._render_variable_encoding(coord_config, coord_arr)
            new_coords[coord_name] = ds.coords[coord_name].copy()
            new_coords[coord_name].attrs = coord_attrs
            new_coords[coord_name].encoding = coord_encoding
        return new_coords

    def render(self, dataset_or_data_arrays, shared_attrs=None):
        """Create :class:`xarray.Dataset` from provided data."""
        data_arrays = dataset_or_data_arrays
        if isinstance(data_arrays, xr.Dataset):
            data_arrays = data_arrays.data_vars.values()

        new_ds = xr.Dataset()
        for data_arr in data_arrays:
            new_var_name, new_data_arr = self._render_variable(data_arr)
            new_ds[new_var_name] = new_data_arr

        new_coords = self._render_coordinates(new_ds)
        new_ds.coords.update(new_coords)
        # use first data array as "representative" for global attributes
        # XXX: Should we use global attributes if dataset_or_data_arrays is a Dataset
        new_ds.attrs = self._render_global_attributes(data_arrays[0].attrs)
        return new_ds


class AWIPSNetCDFTemplate(NetCDFTemplate):
    """NetCDF template renderer specifically for tiled AWIPS files."""

    def __init__(self, template_dict, swap_end_time=False):
        """Handle AWIPS special cases and initialize template helpers."""
        self._swap_end_time = swap_end_time
        if swap_end_time:
            self._swap_attributes_end_time(template_dict)
        super().__init__(template_dict)

    def _swap_attributes_end_time(self, template_dict):
        """Swap every use of 'start_time' to use 'end_time' instead."""
        variable_attributes = [var_section['attributes'] for var_section in template_dict.get('variables', {}).values()]
        global_attributes = template_dict.get('global_attributes', {})
        for attr_section in variable_attributes + [global_attributes]:
            for attr_name in attr_section:
                attr_config = attr_section[attr_name]
                if '{start_time' in attr_config.get('value', ''):
                    attr_config['value'] = attr_config['value'].replace('{start_time', '{end_time')
                if attr_config.get('raw_key', '') == 'start_time':
                    attr_config['raw_key'] = 'end_time'

    def _data_units(self, input_metadata):
        units = input_metadata.get('units', '1')
        # we *know* AWIPS can't handle some units
        return UNIT_CONV.get(units, units)

    def _global_start_date_time(self, input_metadata):
        start_time = input_metadata['start_time']
        if self._swap_end_time:
            start_time = input_metadata['end_time']
        new_stime = start_time + timedelta(minutes=int(os.environ.get("DEBUG_TIME_SHIFT", 0)))
        return new_stime.strftime("%Y-%m-%dT%H:%M:%S")

    def _global_awips_id(self, input_metadata):
        return "AWIPS_" + input_metadata['name']

    def _global_production_location(self, input_metadata):
        """Get default global production_location attribute."""
        del input_metadata
        org = os.environ.get('ORGANIZATION', None)
        if org is not None:
            return org
        else:
            LOG.warning('environment ORGANIZATION not set for .production_location attribute, using hostname')
            import socket
            return socket.gethostname()  # FUTURE: something more correct but this will do for now

    def _get_data_vmin_vmax(self, input_data_arr):
        input_metadata = input_data_arr.attrs
        valid_range = input_metadata.get("valid_range", input_metadata.get("valid_range"))
        if valid_range:
            valid_min, valid_max = valid_range
        else:
            valid_min = input_metadata.get("valid_min", input_metadata.get("valid_min"))
            valid_max = input_metadata.get("valid_max", input_metadata.get("valid_max"))
        return valid_min, valid_max

    def _render_variable_encoding(self, var_config, input_data_arr):
        new_encoding = super()._render_variable_encoding(var_config, input_data_arr)
        vmin, vmax = self._get_data_vmin_vmax(input_data_arr)
        has_flag_meanings = 'flag_meanings' in input_data_arr.attrs
        is_int = np.issubdtype(input_data_arr.dtype, np.integer)
        is_cat = has_flag_meanings or is_int
        if is_cat:
            # AWIPS doesn't like Identity conversion so we can't have
            # a factor of 1 and an offset of 0
            new_encoding['scale_factor'] = 0.5
            new_encoding['add_offset'] = 0
            # no _FillValue
        elif vmin is not None and vmax is not None:
            # calculate scale_factor and add_offset
            sf, ao, fill = _get_factor_offset_fill(
                input_data_arr, vmin, vmax, new_encoding
            )
            new_encoding['scale_factor'] = sf
            new_encoding['add_offset'] = ao
            new_encoding['_FillValue'] = fill
        return new_encoding

    def _get_projection_attrs(self, area_def):
        """Assign projection attributes per CF standard."""
        proj_attrs = area_def.crs.to_cf()
        proj_encoding = {"dtype": "i4"}
        proj_attrs['short_name'] = area_def.area_id
        gmap_name = proj_attrs['grid_mapping_name']

        preferred_names = {
            'geostationary': 'fixedgrid_projection',
            'lambert_conformal_conic': 'lambert_projection',
            'polar_stereographic': 'polar_projection',
            'mercator': 'mercator_projection',
        }
        if gmap_name not in preferred_names:
            LOG.warning("Data is in projection {} which may not be supported "
                        "by AWIPS".format(gmap_name))
        area_id_as_var_name = area_def.area_id.replace('-', '_').lower()
        proj_name = preferred_names.get(gmap_name, area_id_as_var_name)
        return proj_name, proj_attrs, proj_encoding

    def _set_xy_coords_attrs(self, new_ds, crs):
        y_attrs = new_ds.coords['y'].attrs
        if crs.is_geographic:
            if y_attrs.get('units') is None:
                y_attrs['units'] = 'degrees_north'
            if y_attrs.get('standard_name') is None:
                y_attrs['standard_name'] = 'latitude'
        else:
            if y_attrs.get('units') is None:
                y_attrs['units'] = 'meters'
            if y_attrs.get('standard_name') is None:
                y_attrs['standard_name'] = 'projection_y_coordinate'

        x_attrs = new_ds.coords['x'].attrs
        if crs.is_geographic:
            if x_attrs.get('units') is None:
                x_attrs['units'] = 'degrees_east'
            if x_attrs.get('standard_name') is None:
                x_attrs['standard_name'] = 'longitude'
        else:
            if x_attrs.get('units') is None:
                x_attrs['units'] = 'meters'
            if x_attrs.get('standard_name') is None:
                x_attrs['standard_name'] = 'projection_x_coordinate'

    def apply_area_def(self, new_ds, area_def):
        """Apply information we can gather from the AreaDefinition."""
        gmap_name, gmap_attrs, gmap_encoding = self._get_projection_attrs(area_def)
        gmap_data_arr = xr.DataArray(0, attrs=gmap_attrs)
        gmap_data_arr.encoding = gmap_encoding
        new_ds[gmap_name] = gmap_data_arr
        self._set_xy_coords_attrs(new_ds, area_def.crs)
        for data_arr in new_ds.data_vars.values():
            if 'y' in data_arr.dims and 'x' in data_arr.dims:
                data_arr.attrs['grid_mapping'] = gmap_name

        new_ds.attrs['pixel_x_size'] = area_def.pixel_size_x / 1000.0
        new_ds.attrs['pixel_y_size'] = area_def.pixel_size_y / 1000.0
        return new_ds

    def apply_tile_coord_encoding(self, new_ds, xy_factors):
        if 'x' in new_ds.coords:
            new_ds.coords['x'].encoding['dtype'] = 'int16'
            new_ds.coords['x'].encoding['scale_factor'] = np.float64(xy_factors.mx)
            new_ds.coords['x'].encoding['add_offset'] = np.float64(xy_factors.bx)
            new_ds.coords['x'].encoding['_FillValue'] = -1
        if 'y' in new_ds.coords:
            new_ds.coords['y'].encoding['dtype'] = 'int16'
            new_ds.coords['y'].encoding['scale_factor'] = np.float64(xy_factors.my)
            new_ds.coords['y'].encoding['add_offset'] = np.float64(xy_factors.by)
            new_ds.coords['y'].encoding['_FillValue'] = -1
        return new_ds

    def apply_tile_info(self, new_ds, tile_info):
        total_tiles = tile_info.tile_count
        total_pixels = tile_info.image_shape
        tile_row = tile_info.tile_row_offset
        tile_column = tile_info.tile_column_offset
        tile_height = new_ds.sizes['y']
        tile_width = new_ds.sizes['x']
        new_ds.attrs['tile_row_offset'] = tile_row
        new_ds.attrs['tile_column_offset'] = tile_column
        new_ds.attrs['product_tile_height'] = tile_height
        new_ds.attrs['product_tile_width'] = tile_width
        new_ds.attrs['number_product_tiles'] = total_tiles[0] * total_tiles[1]
        new_ds.attrs['product_rows'] = total_pixels[0]
        new_ds.attrs['product_columns'] = total_pixels[1]
        return new_ds

    def apply_misc_metadata(self, new_ds, sector_id, creator=None):
        if creator is None:
            creator = "Satpy Version {} - SCMI Writer".format(__version__)

        new_ds.attrs['Conventions'] = "CF-1.7"
        new_ds.attrs['creator'] = creator
        new_ds.attrs['creation_time'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        new_ds.attrs['sector_id'] = sector_id
        return new_ds

    def render(self, dataset_or_data_arrays, area_def, xy_factors,
               tile_info, sector_id, creator=None):
        """Create a :class:`xarray.Dataset` from template using information provided."""
        new_ds = super().render(dataset_or_data_arrays)
        new_ds = self.apply_area_def(new_ds, area_def)
        new_ds = self.apply_tile_coord_encoding(new_ds, xy_factors)
        new_ds = self.apply_tile_info(new_ds, tile_info)
        new_ds = self.apply_misc_metadata(new_ds, sector_id, creator)
        return new_ds


def _assign_autoscale_encoding_parameters(dataset_to_save):
    for data_var in dataset_to_save.data_vars.values():
        # assume add_offset and scale_factor are both auto
        sf_is_auto = data_var.encoding.get('scale_factor') == 'auto'
        ao_is_auto = data_var.encoding.get('add_offset') == 'auto'
        fv_is_auto = data_var.encoding.get('_FillValue') == 'auto'
        if not any((sf_is_auto, ao_is_auto, fv_is_auto)):
            continue
        if not all((sf_is_auto, ao_is_auto, fv_is_auto)):
            raise ValueError("Auto-scaling must be requested for all or none "
                             "of the associated attributes (scale_factor, "
                             "add_offset, _FillValue).")
        vmin = data_var.min(skipna=True).data.item()
        vmax = data_var.max(skipna=True).data.item()
        sf, ao, fill = _get_factor_offset_fill(
            data_var, vmin, vmax, data_var.encoding
        )
        data_var.encoding['scale_factor'] = sf
        data_var.encoding['add_offset'] = ao
        data_var.encoding['_FillValue'] = fill


def _is_empty_tile(dataset_to_save):
    # check if this tile is empty
    # if so, don't create it
    for data_var in dataset_to_save.data_vars.values():
        # TODO: Does this work for category products?
        if data_var.ndim and data_var.notnull().any():
            break
    else:
        return True
    return False


def _copy_to_existing(dataset_to_save, output_filename):
    # if we leave the dataset open NetCDF will fail because the same
    # file will be opened for reading and writing
    # somehow though, copying makes it work and also closing it doesn't
    # fail. The copy itself makes things work, but explicit closing seemed
    # like a good idea too.
    existing_dataset = xr.open_dataset(output_filename)
    existing_dataset = existing_dataset.copy(deep=True)
    existing_dataset.close()
    # update existing data with new valid data
    for var_name, var_data_arr in existing_dataset.data_vars.items():
        if var_name not in dataset_to_save:
            continue
        new_data_arr = dataset_to_save[var_name]
        # TODO: Make sure category products work
        valid_existing = new_data_arr.notnull()
        var_data_arr.data[valid_existing] = new_data_arr.data[valid_existing]
    return existing_dataset


def to_nonempty_netcdf(dataset_to_save, output_filename, update_existing=True,
                       fix_awips=False):
    """Save :class:`xarray.Dataset` to a NetCDF file if not all fills.

    In addition to checking certain Dataset variables for fill values,
    this function can also "update" an existing NetCDF file with the
    new valid data provided.

    This function will also allow for 'auto' scale_factor and add_offset
    creation by taking the minimum and maximum value of the variable.

    """
    if _is_empty_tile(dataset_to_save):
        LOG.debug("Skipping tile creation for {} because it would be "
                  "empty.".format(output_filename))
        return

    _assign_autoscale_encoding_parameters(dataset_to_save)

    # TODO: Add ability to update existing files
    if update_existing and os.path.isfile(output_filename):
        dataset_to_save = _copy_to_existing(dataset_to_save, output_filename)
        mode = 'a'
    else:
        mode = 'w'
    dataset_to_save.to_netcdf(output_filename, mode=mode)
    if fix_awips:
        fix_awips_file(output_filename)


delayed_to_notempty_netcdf = dask.delayed(to_nonempty_netcdf, pure=True)


def tile_filler(data_arr_data, tile_shape, tile_slices, fill_value):
    empty_tile = np.full(tile_shape, fill_value, dtype=data_arr_data.dtype)
    empty_tile[tile_slices] = data_arr_data
    return empty_tile


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
        self.scmi_sectors = self.config['sectors']
        self.templates = self.config['templates']
        # self.scmi_datasets = SCMIDatasetDecisionTree([self.config['variables']])
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

    def _group_by_area(self, datasets):
        """Group datasets by their area."""
        def _area_id(area_def):
            return area_def.description + str(area_def.area_extent) + str(area_def.shape)

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

    def _tile_filler(self, tile_info, data_arr):
        fill = np.nan if np.issubdtype(data_arr.dtype, np.floating) else data_arr.attrs.get('_FillValue', 0)
        data_arr_data = data_arr.data[tile_info.data_slices]
        data_arr_data = data_arr_data.rechunk(data_arr_data.shape)
        new_data = da.map_blocks(tile_filler, data_arr_data,
                                 tile_info.tile_shape, tile_info.tile_slices,
                                 fill, dtype=data_arr.dtype, chunks=tile_info.tile_shape)
        return xr.DataArray(new_data, dims=('y', 'x'),
                            attrs=data_arr.attrs.copy())

    def _slice_and_update_coords(self, tile_info, data_arrays):
        new_x = xr.DataArray(tile_info.x, dims=('x',))
        if 'x' in data_arrays[0].coords:
            old_x = data_arrays[0].coords['x']
            new_x.attrs.update(old_x.attrs)
            new_x.encoding = old_x.encoding
        new_y = xr.DataArray(tile_info.y, dims=('y',))
        if 'y' in data_arrays[0].coords:
            old_y = data_arrays[0].coords['y']
            new_x.attrs.update(old_y.attrs)
            new_y.encoding = old_y.encoding

        for data_arr in data_arrays:
            new_data_arr = self._tile_filler(tile_info, data_arr)
            new_data_arr.coords['x'] = new_x
            new_data_arr.coords['y'] = new_y
            yield new_data_arr

    def _iter_tile_info_and_datasets(self, tile_gen, data_arrays, single_variable=True):
        all_data_arrays = self._enhance_and_split_rgbs(data_arrays)
        if single_variable:
            all_data_arrays = [[single_data_arr] for single_data_arr in all_data_arrays]
        else:
            all_data_arrays = [all_data_arrays]
        for data_arrays_set in all_data_arrays:
            for tile_info in tile_gen():
                data_arrays_tile_set = list(self._slice_and_update_coords(tile_info, data_arrays_set))
                yield tile_info, data_arrays_tile_set

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
            rows=area_def.height,
            columns=area_def.width,
            sector_id=sector_id,
            tile_id=tile_info.tile_id,
            **kwargs)

    def check_tile_exists(self, output_filename):
        """Check if tile exists and report error accordingly."""
        if os.path.isfile(output_filename):
            LOG.info("AWIPS file already exists, will update with new data: %s", output_filename)

    def _save_nonempty_mfdatasets(self, datasets_to_save, output_filenames, **kwargs):
        for dataset_to_save, output_filename in zip(datasets_to_save, output_filenames):
            delayed_res = delayed_to_notempty_netcdf(
                dataset_to_save, output_filename, **kwargs)
            yield delayed_res

    def save_datasets(self, datasets, sector_id=None,
                      source_name=None, filename=None,
                      tile_count=(1, 1), tile_size=None,
                      lettered_grid=False, num_subtiles=None,
                      use_end_time=False, use_sector_reference=False,
                      template='polar', compute=True, **kwargs):
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
            template (str or dict): Name of the template configured in the
                writer YAML file. This can also be a dictionary with a full
                template configuration. See the :mod:`satpy.writers.scmi`
                documentation for more information on templates. Defaults to
                the 'polar' builtin template.
            compute (bool): Compute and write the output immediately using
                dask. Default to ``False``.

        """
        if sector_id is None:
            raise TypeError("Keyword 'sector_id' is required")
        if source_name is None:
            raise TypeError("Keyword 'source_name' is required")
        if self.fix_awips and not compute:
            LOG.debug("Can't 'fix_awips' with delayed computation, "
                      "forcing immediate computation.")

        if not isinstance(template, dict):
            template = self.config['templates'][template]
        template = AWIPSNetCDFTemplate(template, swap_end_time=use_end_time)
        delayeds = []
        area_datasets = self._group_by_area(datasets)
        datasets_to_save = []
        output_filenames = []
        # TODO: Combine these for loops into one helper iterator.
        #    Will require putting tile_gen.xy_factors into TileInfo
        for area_def, data_arrays in area_datasets.values():
            tile_gen = self._get_tile_generator(
                area_def, lettered_grid, sector_id, num_subtiles, tile_size,
                tile_count, use_sector_reference=use_sector_reference)
            for tile_info, data_arrs in self._iter_tile_info_and_datasets(
                    tile_gen, data_arrays, single_variable=template.is_single_variable):
                # use the first data array as a "representative" for the group
                ds_info = data_arrs[0].attrs.copy()
                # TODO: Create Dataset object of all of the sliced-DataArrays (optional)
                output_filename = filename or self.get_filename(area_def, tile_info, sector_id,
                                                                source_name=source_name,
                                                                **ds_info)
                self.check_tile_exists(output_filename)
                # TODO: Provide attribute caching for things that likely won't change
                new_ds = template.render(data_arrs, area_def, tile_gen.xy_factors,
                                         tile_info, sector_id)
                if self.compress:
                    new_ds.encoding['zlib'] = True

                datasets_to_save.append(new_ds)
                output_filenames.append(output_filename)
        if not datasets_to_save:
            # no tiles produced
            return delayeds

        for delayed_result in self._save_nonempty_mfdatasets(datasets_to_save, output_filenames):
            if compute:
                delayed_result.compute()
                continue
            delayeds.append(delayed_result)

        if self.fix_awips:
            for fn in output_filenames:
                fix_awips_file(fn)
        return delayeds


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
