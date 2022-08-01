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
"""The AWIPS Tiled writer is used to create AWIPS-compatible tiled NetCDF4 files.

The Advanced Weather Interactive Processing System (AWIPS) is a
program used by the United States National Weather Service (NWS) and others
to view
different forms of weather imagery. The original Sectorized Cloud and Moisture
Imagery (SCMI) functionality in AWIPS was a NetCDF4 format supported by AWIPS
to store one image broken up in to one or more "tiles". This format has since
been expanded to support many other products and so the writer for this format
in Satpy is generically called the "AWIPS Tiled" writer. You may still see
SCMI referenced in this documentation or in the source code for the writer.
Once AWIPS is configured for specific products this writer can be used to
provide compatible products to the system.

The AWIPS Tiled writer takes 2D (y, x) geolocated data and creates one or more
AWIPS-compatible NetCDF4 files. The writer and the AWIPS client may
need to be configured to make things appear the way the user wants in
the AWIPS client. The writer can only produce files for datasets mapped
to areas with specific projections:

 - lcc
 - geos
 - merc
 - stere

This is a limitation of the AWIPS client and not of the writer. In the case
where AWIPS has been updated to support additional projections, this writer
may also need to be updated to support those projections.

AWIPS Configuration
-------------------

Depending on how this writer is used and the data it is provided, AWIPS may
need additional configuration on the server side to properly ingest the files
produced. This will require administrator privileges to the ingest server(s)
and is not something that can be configured on the client. Note that any
changes required must be done on all servers that you wish to ingest your data
files. The generic "polar" template this writer defaults to should limit the
number of modifications needed for any new data fields that AWIPS previously
was unaware of. Once the data is ingested, the client can be used to customize
how the data looks on screen.

AWIPS requires files to follow a specific naming scheme so they can be routed
to specific "decoders". For the files produced by this writer, this typically
means editing the "goesr" decoder configuration in a directory like::

/awips2/edex/data/utility/common_static/site/<site>/distribution/goesr.xml

The "goesr" decoder is a subclass of the "satellite" decoder. You may see
either name show up in the AWIPS ingest logs. With the correct
regular expression in the above file, your files should be passed to the
right decoder, opened, and parsed for data.

To tell AWIPS exactly what attributes and variables mean in your file, you'll
need to create or configure an XML file in::

/awips2/edex/data/utility/common_static/site/<site>/satellite/goesr/descriptions/

See the existing files in this directory for examples. The "polar" template
(see below) that this writer uses by default is already configured in the
"Polar" subdirectory assuming that the TOWR-S RPM package has been installed
on your AWIPS ingest server.

Templates
---------

This writer allows for a "template" to be specified to control how the output
files are structured and created. Templates can be configured in the writer
YAML file (``awips_tiled.yaml``) or passed as a dictionary to the ``template``
keyword argument. Templates have three main sections:

1. global_attributes
2. coordinates
3. variables

Additionally, you can specify whether a template should produce files with
one variable per file by specifying ``single_variable: true`` or multiple
variables per file by specifying ``single_variable: false``. You can also
specify the output filename for a template using a Python format string.
See ``awips_tiled.yaml`` for examples. Lastly, a ``add_sector_id_global``
boolean parameter can be specified to add the user-provided ``sector_id``
keyword argument as a global attribute to the file.

The ``global_attributes`` section takes names of global attributes and
then a series of options to "render" that attribute from the metadata
provided when creating files. For example::

    product_name:
        value: "{name}"

For more information see the
:meth:`satpy.writers.awips_tiled.NetCDFTemplate.get_attr_value` method.

The ``coordinates`` and ``variables`` are similar to each other in that they
define how a variable should be created, the attributes it should have, and
the encoding to write to the file. Coordinates typically don't need to be
modified as tiled files usually have only ``x`` and ``y`` dimension variables.
The Variables on the other hand use a decision tree to determine what section
applies for a particular DataArray being saved. The basic structure is::

    variables:
      arbitrary_section_name:
        <decision tree matching parameters>
        var_name: "output_netcdf_variable_name"
        attributes:
          <attributes similar to global attributes>
        encoding:
          <xarray encoding parameters>

The "decision tree matching parameters" can be one or more of "name",
"standard_name', "satellite", "sensor", "area_id', "units", or "reader".
The writer will choose the best section for the DataArray being saved
(the most matches). If none of these parameters are specified in a section
then it will be used when no other matches are found (the "default" section).

The "encoding" parameters can be anything accepted by xarray's ``to_netcdf``
method. See :meth:`xarray.Dataset.to_netcdf` for more information on the
`encoding`` keyword argument.

For more examples see the existing builtin templates defined in
``awips_tiled.yaml``.

Builtin Templates
^^^^^^^^^^^^^^^^^

There are only a few templates provided in Sapty currently.

* **polar**: A custom format developed for the CSPP Polar2Grid project at the
  University of Wisconsin - Madison Space Science and Engineering Center
  (SSEC). This format is made available through the TOWR-S package that can be
  installed for GOES-R support in AWIPS. This format is meant to be very
  generic and should theoretically allow any variable to get ingested into
  AWIPS.
* **glm_l2_radc**: This format is used to produce standard files for the gridded
  GLM products produced by the CSPP Geo Gridded GLM package. Support for this
  format is also available in the TOWR-S package on an AWIPS ingest server.
  This format is specific to gridded GLM on the CONUS sector and is not meant
  to work for other data.
* **glm_l2_radf**: This format is used to produce standard files for the gridded
  GLM productes produced by the CSPP Geo Gridded GLM package. Support for this
  format is also available in the TOWR-S package on an AWIPS ingest server.
  This format is specific to gridded GLM on the Full Disk sector and is not
  meant to work for other data.

Numbered versus Lettered Grids
------------------------------

By default this writer will save tiles by number starting with '1'
representing the upper-left image tile. Tile numbers then increase
along the column and then on to the next row.

By specifying `lettered_grid` as `True` tiles can be designated with a
letter. Lettered grids or sectors are preconfigured in the `awips_tiled.yaml`
configuration file. The lettered tile locations are static and will not
change with the data being written to them. Each lettered tile is split
into a certain number of subtiles (`num_subtiles`), default 2 rows by
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
output AWIPS tiles, but may not perfectly aligned, leaving
empty/unused space in the output tile. The next input tile may be able to fill
in that empty space and should be allowed to write the "new" data to the file.
This is the default behavior of the AWIPS tiled writer. In cases where data
overlaps the existing data in the tile, the newer data has priority.

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
argument can be set to ``True`` to tell the writer to shift the data's
geolocation by up to 0.5 pixels in each dimension instead of shifting the
lettered tile locations.

"""
import logging
import os
import string
import sys
import warnings
from collections import namedtuple
from datetime import datetime, timedelta

import dask
import dask.array as da
import numpy as np
import xarray as xr
from pyproj import CRS, Proj, Transformer
from pyresample.geometry import AreaDefinition
from trollsift.parser import Parser, StringFormatter

from satpy import __version__
from satpy.writers import DecisionTree, Enhancer, Writer, get_enhanced_image

LOG = logging.getLogger(__name__)
DEFAULT_OUTPUT_PATTERN = '{source_name}_AII_{platform_name}_{sensor}_' \
                         '{name}_{sector_id}_{tile_id}_' \
                         '{start_time:%Y%m%d_%H%M}.nc'

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
                                   'tile_number',
                                   'x', 'y', 'xy_factors', 'tile_slices', 'data_slices'])
XYFactors = namedtuple('XYFactors', ['mx', 'bx', 'my', 'by'])


def fix_awips_file(fn):
    """Hack the NetCDF4 files to workaround NetCDF-Java bugs used by AWIPS.

    This should not be needed for new versions of AWIPS.

    """
    # hack to get files created by new NetCDF library
    # versions to be read by AWIPS buggy java version
    # of NetCDF
    LOG.info("Modifying output NetCDF file to work with AWIPS")
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

        # number of pixels per each tile (rows, cols)
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
            gd.crs,
            imaginary_data_size[1],
            imaginary_data_size[0],
            new_extents,
        )

        x, y = imaginary_grid_def.get_proj_vectors()
        return x, y

    def _get_xy_scaling_parameters(self):
        """Get the X/Y coordinate limits for the full resulting image."""
        gd = self.area_definition
        bx = self.x.min()
        mx = gd.pixel_size_x
        by = self.y.max()
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

                tile_number = self._tile_number(ty, tx)
                tile_info = TileInfo(
                    tc, self.image_shape, ts,
                    tile_row_offset, tile_column_offset, tile_id,
                    tile_number,
                    tmp_x, tmp_y, self.xy_factors, tile_slices, data_slices)
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

    def __init__(self, area_definition, extents, sector_crs,
                 cell_size=(2000000, 2000000),
                 num_subtiles=None, use_sector_reference=False):
        """Initialize tile information for later generation.

        Args:
            area_definition (AreaDefinition): Area of the data being saved.
            extents (tuple): Four element tuple of the configured lettered
                 area.
            sector_crs (pyproj.CRS): CRS of the configured lettered sector
                area.
            cell_size (tuple): Two element tuple of resolution of each tile
                in sector projection units (y, x).
        """
        # (row subtiles, col subtiles)
        self.num_subtiles = num_subtiles or (2, 2)
        self.cell_size = cell_size  # (row tile height, col tile width)
        # x/y
        self.ll_extents = extents[:2]  # (x min, y min)
        self.ur_extents = extents[2:]  # (x max, y max)
        self.use_sector_reference = use_sector_reference
        self._transformer = Transformer.from_crs(sector_crs, area_definition.crs)
        super().__init__(area_definition)

    def _get_tile_properties(self, tile_shape, tile_count):
        """Calculate tile information for this particular sector/grid."""
        # ignore tile_shape and tile_count
        # they come from the base class, but aren't used here
        del tile_shape, tile_count

        # get original image's X/Y
        ad = self.area_definition
        x, y = ad.get_proj_vectors()

        ll_xy = self._transformer.transform(*self.ll_extents)
        ur_xy = self._transformer.transform(*self.ur_extents)
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
            LOG.debug("Adjusting X/Y by (%f, %f) so it better matches lettered grid", shift_x, shift_y)
            x = x + shift_x
            y = y + shift_y
        else:
            LOG.debug("Adjusting lettered grid by (%f, %f) so it better matches data X/Y", shift_x, shift_y)
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
                    LOG.debug("Tile '%s' doesn't have any data in it", tile_id)
                    continue
                x_slice = slice(x_mask[0], x_mask[-1] + 1)  # assume it's continuous
                y_slice = slice(y_mask[0], y_mask[-1] + 1)

                # theoretically we can precompute the X/Y now
                # instead of taking the x/y data and mapping it
                # to the tile
                tmp_x = np.arange(x_left + cw / 2., x_right, cw)
                tmp_y = np.arange(y_top - ch / 2., y_bot, -ch)
                data_x_idx_min = np.nonzero(np.isclose(tmp_x, x[x_slice.start]))[0][0]
                data_x_idx_max = np.nonzero(np.isclose(tmp_x, x[x_slice.stop - 1]))[0][0]
                # I have a half pixel error some where
                data_y_idx_min = np.nonzero(np.isclose(tmp_y, y[y_slice.start]))[0][0]
                data_y_idx_max = np.nonzero(np.isclose(tmp_y, y[y_slice.stop - 1]))[0][0]
                # now put the data in the grid tile

                tile_slices = (slice(data_y_idx_min, data_y_idx_max + 1),
                               slice(data_x_idx_min, data_x_idx_max + 1))
                data_slices = (y_slice, x_slice)

                tile_number = self._tile_number(gy, gx)
                tile_info = TileInfo(
                    self.tile_count, self.image_shape, ts,
                    gy * ts[0], gx * ts[1], tile_id, tile_number,
                    tmp_x, tmp_y, self.xy_factors, tile_slices, data_slices)
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

    mx = (vmax - vmin) / (2 ** bit_depth - 1 - num_fills)
    bx = vmin
    if not is_unsigned and not unsigned_in_signed:
        bx += 2 ** (bit_depth - 1) * mx
    return mx, bx, fills[0]


def _get_data_vmin_vmax(input_data_arr):
    input_metadata = input_data_arr.attrs
    valid_range = input_metadata.get("valid_range")
    if valid_range:
        valid_min, valid_max = valid_range
    else:
        valid_min = input_metadata.get("valid_min")
        valid_max = input_metadata.get("valid_max")
    return valid_min, valid_max


def _add_valid_ranges(data_arrs):
    """Add 'valid_range' metadata if not present.

    If valid_range or valid_min/valid_max are not present in a DataArrays
    metadata (``.attrs``), then lazily compute it with dask so it can be
    computed later when we write tiles out.

    AWIPS requires that scale_factor/add_offset/_FillValue be the **same**
    for all tiles. We must do this calculation before splitting the data into
    tiles otherwise the values will be different.

    """
    for data_arr in data_arrs:
        vmin, vmax = _get_data_vmin_vmax(data_arr)
        if vmin is None:
            # XXX: Do we need to handle category products here?
            vmin = data_arr.min(skipna=True).data
            vmax = data_arr.max(skipna=True).data
            # we don't want to effect the original attrs
            data_arr = data_arr.copy(deep=False)
            # these are dask arrays, they need to get computed later
            data_arr.attrs['valid_range'] = (vmin, vmax)
        yield data_arr


class AWIPSTiledVariableDecisionTree(DecisionTree):
    """Load AWIPS-specific metadata from YAML configuration."""

    def __init__(self, decision_dicts, **kwargs):
        """Initialize decision tree with specific keys to look for."""
        # Fields used to match a product object to it's correct configuration
        attrs = kwargs.pop('attrs',
                           ["name",
                            "standard_name",
                            "satellite",
                            "sensor",
                            "area_id",
                            "units",
                            "reader"]
                           )
        super(AWIPSTiledVariableDecisionTree, self).__init__(decision_dicts, attrs, **kwargs)


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

        self._var_tree = AWIPSTiledVariableDecisionTree([self.variables])
        self._coord_tree = AWIPSTiledVariableDecisionTree([self.coordinates])
        self._filename_format_str = template_dict.get('filename')
        self._str_formatter = StringFormatter()
        self._template_dict = template_dict

    def get_filename(self, base_dir='', **kwargs):
        """Generate output NetCDF file from metadata."""
        # format the filename
        if self._filename_format_str is None:
            raise ValueError("Template does not have a configured "
                             "'filename' pattern.")
        fn_format_str = os.path.join(base_dir, self._filename_format_str)
        filename_parser = Parser(fn_format_str)
        output_filename = filename_parser.compose(kwargs)
        dirname = os.path.dirname(output_filename)
        if dirname and not os.path.isdir(dirname):
            LOG.info("Creating output directory: %s", dirname)
            os.makedirs(dirname)
        return output_filename

    def get_attr_value(self, attr_name, input_metadata, value=None, raw_key=None, raw_value=None, prefix="_"):
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
            prefix (str): Prefix to use when `value` and `raw_key` are
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
                LOG.debug("Can't format string '%s' with provided "
                          "input metadata.", value)
                value = None
                # raise ValueError("Can't format string '{}' with provided "
                #                  "input metadata.".format(value))
        if value is not None:
            return value

        meth_name = prefix + attr_name
        func = getattr(self, meth_name, None)
        if func is not None:
            value = func(input_metadata)
        if value is None:
            LOG.debug('no routine matching %s', meth_name)
        return value

    def _render_attrs(self, attr_configs, input_metadata, prefix="_"):
        attrs = {}
        for attr_name, attr_config_dict in attr_configs.items():
            val = self.get_attr_value(attr_name, input_metadata,
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
        if "dtype" not in new_encoding:
            new_encoding['dtype'] = 'int16'
            new_encoding['_Unsigned'] = 'true'
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
        if shared_attrs is None:
            shared_attrs = data_arrays[0].attrs
        new_ds.attrs = self._render_global_attributes(shared_attrs)
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
        return start_time.strftime("%Y-%m-%dT%H:%M:%S")

    def _global_awips_id(self, input_metadata):
        return "AWIPS_" + input_metadata['name']

    def _global_physical_element(self, input_metadata):
        var_config = self._var_tree.find_match(**input_metadata)
        attr_config = {"physical_element": var_config["attributes"]["physical_element"]}
        result = self._render_attrs(attr_config, input_metadata, prefix="_data_")
        return result["physical_element"]

    def _global_production_location(self, input_metadata):
        """Get default global production_location attribute."""
        del input_metadata
        org = os.environ.get('ORGANIZATION', None)
        if org is not None:
            prod_location = org
        else:
            LOG.warning('environment ORGANIZATION not set for .production_location attribute, using hostname')
            import socket
            prod_location = socket.gethostname()  # FUTURE: something more correct but this will do for now

        if len(prod_location) > 31:
            warnings.warn("Production location attribute is longer than 31 "
                          "characters (AWIPS limit). Set it to a smaller "
                          "value with the 'ORGANIZATION' environment "
                          "variable. Defaults to hostname and is currently "
                          "set to '{}'.".format(prod_location))
            prod_location = prod_location[:31]
        return prod_location

    _global_production_site = _global_production_location

    @staticmethod
    def _get_vmin_vmax(var_config, input_data_arr):
        if 'valid_range' in var_config:
            return var_config['valid_range']
        data_vmin, data_vmax = _get_data_vmin_vmax(input_data_arr)
        return data_vmin, data_vmax

    def _render_variable_encoding(self, var_config, input_data_arr):
        new_encoding = super()._render_variable_encoding(var_config, input_data_arr)
        vmin, vmax = self._get_vmin_vmax(var_config, input_data_arr)
        has_flag_meanings = 'flag_meanings' in input_data_arr.attrs
        is_int = np.issubdtype(input_data_arr.dtype, np.integer)
        is_cat = has_flag_meanings or is_int
        has_sf = new_encoding.get('scale_factor') is not None
        if not has_sf and is_cat:
            # AWIPS doesn't like Identity conversion so we can't have
            # a factor of 1 and an offset of 0
            # new_encoding['scale_factor'] = None
            # new_encoding['add_offset'] = None
            if '_FillValue' in input_data_arr.attrs:
                new_encoding['_FillValue'] = input_data_arr.attrs['_FillValue']
        elif not has_sf and vmin is not None and vmax is not None:
            # calculate scale_factor and add_offset
            sf, ao, fill = _get_factor_offset_fill(
                input_data_arr, vmin, vmax, new_encoding
            )
            # NOTE: These could be dask arrays that will be computed later
            #   when we go to write the files.
            new_encoding['scale_factor'] = sf
            new_encoding['add_offset'] = ao
            new_encoding['_FillValue'] = fill
            new_encoding['coordinates'] = ' '.join([ele for ele in input_data_arr.dims])
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
            LOG.warning("Data is in projection %s which may not be supported "
                        "by AWIPS", gmap_name)
        area_id_as_var_name = area_def.area_id.replace('-', '_').lower()
        proj_name = preferred_names.get(gmap_name, area_id_as_var_name)
        return proj_name, proj_attrs, proj_encoding

    def _set_xy_coords_attrs(self, new_ds, crs):
        y_attrs = new_ds.coords['y'].attrs
        if crs.is_geographic:
            self._fill_units_and_standard_name(y_attrs, 'degrees_north', 'latitude')
        else:
            self._fill_units_and_standard_name(y_attrs, 'meters', 'projection_y_coordinate')
            y_attrs['axis'] = 'Y'

        x_attrs = new_ds.coords['x'].attrs
        if crs.is_geographic:
            self._fill_units_and_standard_name(x_attrs, 'degrees_east', 'longitude')
        else:
            self._fill_units_and_standard_name(x_attrs, 'meters', 'projection_x_coordinate')
            x_attrs['axis'] = 'X'

    @staticmethod
    def _fill_units_and_standard_name(attrs, units, standard_name):
        """Fill in units and standard_name if not set in `attrs`."""
        if attrs.get('units') is None:
            attrs['units'] = units
        if attrs['units'] in ('meter', 'metre'):
            # AWIPS doesn't like 'meter'
            attrs['units'] = 'meters'
        if attrs.get('standard_name') is None:
            attrs['standard_name'] = standard_name

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
        """Add encoding information specific to the coordinate variables."""
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
        """Apply attributes associated with the current tile."""
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

    def _add_sector_id_global(self, new_ds, sector_id):
        if not self._template_dict.get('add_sector_id_global'):
            return

        if sector_id is None:
            raise ValueError("Keyword 'sector_id' is required for this "
                             "template.")
        new_ds.attrs['sector_id'] = sector_id

    def apply_misc_metadata(self, new_ds, sector_id=None, creator=None, creation_time=None):
        """Add attributes that don't fit into any other category."""
        if creator is None:
            creator = "Satpy Version {} - AWIPS Tiled Writer".format(__version__)
        if creation_time is None:
            creation_time = datetime.utcnow()

        self._add_sector_id_global(new_ds, sector_id)
        new_ds.attrs['Conventions'] = "CF-1.7"
        new_ds.attrs['creator'] = creator
        new_ds.attrs['creation_time'] = creation_time.strftime('%Y-%m-%dT%H:%M:%S')
        return new_ds

    def _render_variable_attributes(self, var_config, input_metadata):
        attrs = super()._render_variable_attributes(var_config, input_metadata)
        # AWIPS validation checks
        if len(attrs.get("units", "")) > 26:
            warnings.warn(
                "AWIPS 'units' must be limited to a maximum of 26 characters. "
                "Units '{}' is too long and will be truncated.".format(attrs["units"]))
            attrs["units"] = attrs["units"][:26]
        return attrs

    def render(self, dataset_or_data_arrays, area_def,
               tile_info, sector_id, creator=None, creation_time=None,
               shared_attrs=None, extra_global_attrs=None):
        """Create a :class:`xarray.Dataset` from template using information provided."""
        new_ds = super().render(dataset_or_data_arrays, shared_attrs=shared_attrs)
        new_ds = self.apply_area_def(new_ds, area_def)
        new_ds = self.apply_tile_coord_encoding(new_ds, tile_info.xy_factors)
        new_ds = self.apply_tile_info(new_ds, tile_info)
        new_ds = self.apply_misc_metadata(new_ds, sector_id, creator, creation_time)
        if extra_global_attrs:
            new_ds.attrs.update(extra_global_attrs)
        return new_ds


def _notnull(data_arr, check_categories=True):
    is_int = np.issubdtype(data_arr.dtype, np.integer)
    fill_value = data_arr.encoding.get('_FillValue', data_arr.attrs.get('_FillValue'))
    if is_int and fill_value is not None:
        # some DQF datasets are always valid
        if check_categories:
            return data_arr != fill_value
        return False
    return data_arr.notnull()


def _any_notnull(data_arr, check_categories):
    not_null = _notnull(data_arr, check_categories)
    if not_null is False:
        return False
    return not_null.any()


def _is_empty_tile(dataset_to_save, check_categories):
    # check if this tile is empty
    # if so, don't create it
    for data_var in dataset_to_save.data_vars.values():
        if data_var.ndim == 2 and _any_notnull(data_var, check_categories):
            return False
    return True


def _copy_to_existing(dataset_to_save, output_filename):
    # Experimental: This function doesn't seem to behave well with xarray file
    #   caching and/or multiple dask workers. It causes tests to hang, but
    #   only sometimes. Limiting dask to 1 worker seems to fix this.
    #   I (David Hoese) was unable to make a script that reproduces this
    #   without using this writer (makes it difficult to file a bug report).
    existing_dataset = xr.open_dataset(output_filename)
    # the below used to trick xarray into working, but this doesn't work
    # in newer versions. This was a hack in the first place so I'm keeping it
    # here for reference.
    # existing_dataset = existing_dataset.copy(deep=True)
    # existing_dataset.close()

    # update existing data with new valid data
    for var_name, var_data_arr in dataset_to_save.data_vars.items():
        if var_name not in existing_dataset:
            continue
        if var_data_arr.ndim != 2:
            continue
        existing_data_arr = existing_dataset[var_name]
        valid_current = _notnull(var_data_arr)
        new_data = existing_data_arr.data[:]
        new_data[valid_current] = var_data_arr.data[valid_current]
        var_data_arr.data[:] = new_data
        var_data_arr.encoding.update(existing_data_arr.encoding)
        var_data_arr.encoding.pop('source', None)

    return dataset_to_save


def _extract_factors(dataset_to_save):
    factors = {}
    for data_var in dataset_to_save.data_vars.values():
        enc = data_var.encoding
        data_var.attrs.pop('valid_range', None)
        factor_set = (enc.pop('scale_factor', None),
                      enc.pop('add_offset', None),
                      enc.pop('_FillValue', None))
        factors[data_var.name] = factor_set
    return factors


def _reapply_factors(dataset_to_save, factors):
    for var_name, factor_set in factors.items():
        data_arr = dataset_to_save[var_name]
        if factor_set[0] is not None:
            data_arr.encoding['scale_factor'] = factor_set[0]
        if factor_set[1] is not None:
            data_arr.encoding['add_offset'] = factor_set[1]
        if factor_set[2] is not None:
            data_arr.encoding['_FillValue'] = factor_set[2]
    return dataset_to_save


def to_nonempty_netcdf(dataset_to_save: xr.Dataset,
                       factors: dict,
                       output_filename: str,
                       update_existing: bool = True,
                       check_categories: bool = True):
    """Save :class:`xarray.Dataset` to a NetCDF file if not all fills.

    In addition to checking certain Dataset variables for fill values,
    this function can also "update" an existing NetCDF file with the
    new valid data provided.

    """
    dataset_to_save = _reapply_factors(dataset_to_save, factors)
    if _is_empty_tile(dataset_to_save, check_categories):
        LOG.debug("Skipping tile creation for %s because it would be "
                  "empty.", output_filename)
        return None, None, None

    # TODO: Allow for new variables to be created
    if update_existing and os.path.isfile(output_filename):
        dataset_to_save = _copy_to_existing(dataset_to_save, output_filename)
        mode = 'a'
    else:
        mode = 'w'
    return dataset_to_save, output_filename, mode
    # return dataset_to_save.to_netcdf(output_filename, mode=mode)
    # if fix_awips:
    #     fix_awips_file(output_filename)


delayed_to_notempty_netcdf = dask.delayed(to_nonempty_netcdf, pure=True)


def tile_filler(data_arr_data, tile_shape, tile_slices, fill_value):
    """Create an empty tile array and fill the proper locations with data."""
    empty_tile = np.full(tile_shape, fill_value, dtype=data_arr_data.dtype)
    empty_tile[tile_slices] = data_arr_data
    return empty_tile


class AWIPSTiledWriter(Writer):
    """Writer for AWIPS NetCDF4 Tile files.

    See :mod:`satpy.writers.awips_tiled` documentation for more information
    on templates and produced file format.

    """

    def __init__(self, compress=False, fix_awips=False, **kwargs):
        """Initialize writer and decision trees."""
        super(AWIPSTiledWriter, self).__init__(default_config_filename="writers/awips_tiled.yaml", **kwargs)
        self.base_dir = kwargs.get('base_dir', '')
        self.awips_sectors = self.config['sectors']
        self.templates = self.config['templates']
        self.compress = compress
        self.fix_awips = fix_awips
        self._fill_sector_info()
        self._enhancer = None

        if self.fix_awips:
            warnings.warn("'fix_awips' flag no longer has any effect and is "
                          "deprecated. Modern versions of AWIPS should not "
                          "require this hack.", DeprecationWarning)
            self.fix_awips = False

    @property
    def enhancer(self):
        """Get lazy loaded enhancer object only if needed."""
        if self._enhancer is None:
            self._enhancer = Enhancer()
        return self._enhancer

    @classmethod
    def separate_init_kwargs(cls, kwargs):
        """Separate keyword arguments by initialization and saving keyword arguments."""
        # FUTURE: Don't pass Scene.save_datasets kwargs to init and here
        init_kwargs, kwargs = super(AWIPSTiledWriter, cls).separate_init_kwargs(
            kwargs)
        for kw in ['compress', 'fix_awips']:
            if kw in kwargs:
                init_kwargs[kw] = kwargs.pop(kw)

        return init_kwargs, kwargs

    def _fill_sector_info(self):
        """Convert sector extents if needed."""
        for sector_info in self.awips_sectors.values():
            sector_info['projection'] = CRS.from_user_input(sector_info['projection'])
            p = Proj(sector_info['projection'])
            if 'lower_left_xy' in sector_info:
                sector_info['lower_left_lonlat'] = p(*sector_info['lower_left_xy'], inverse=True)
            else:
                sector_info['lower_left_xy'] = p(*sector_info['lower_left_lonlat'])
            if 'upper_right_xy' in sector_info:
                sector_info['upper_right_lonlat'] = p(*sector_info['upper_right_xy'], inverse=True)
            else:
                sector_info['upper_right_xy'] = p(*sector_info['upper_right_lonlat'])

    def _get_lettered_sector_info(self, sector_id):
        """Get metadata for the current sector if configured.

        This is not necessary for numbered grids. If found, the sector info
        will provide the overall tile layout for this grid/sector. This allows
        for consistent tile numbering/naming regardless of where the data being
        converted actually is.

        """
        if sector_id is None:
            raise TypeError("Keyword 'sector_id' is required for lettered grids.")
        try:
            return self.awips_sectors[sector_id]
        except KeyError:
            raise ValueError("Unknown sector '{}'".format(sector_id))

    def _get_tile_generator(self, area_def, lettered_grid, sector_id,
                            num_subtiles, tile_size, tile_count,
                            use_sector_reference=False):
        """Get the appropriate tile generator class for lettered or numbered tiles."""
        # Create a tile generator for this grid definition
        if lettered_grid:
            sector_info = self._get_lettered_sector_info(sector_id)
            tile_gen = LetteredTileGenerator(
                area_def,
                sector_info['lower_left_xy'] + sector_info['upper_right_xy'],
                sector_crs=sector_info['projection'],
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
                          "that aren't RGBs to AWIPS Tiled format: %s", ds.name)
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
            new_y.attrs.update(old_y.attrs)
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

    def _iter_area_tile_info_and_datasets(self, area_datasets, template,
                                          lettered_grid, sector_id,
                                          num_subtiles, tile_size, tile_count,
                                          use_sector_reference):
        for area_def, data_arrays in area_datasets.values():
            data_arrays = list(_add_valid_ranges(data_arrays))
            tile_gen = self._get_tile_generator(
                area_def, lettered_grid, sector_id, num_subtiles, tile_size,
                tile_count, use_sector_reference=use_sector_reference)
            for tile_info, data_arrs in self._iter_tile_info_and_datasets(
                    tile_gen, data_arrays, single_variable=template.is_single_variable):
                yield area_def, tile_info, data_arrs

    def save_dataset(self, dataset, **kwargs):
        """Save a single DataArray to one or more NetCDF4 Tile files."""
        LOG.warning("For best performance use `save_datasets`")
        return self.save_datasets([dataset], **kwargs)

    def get_filename(self, template, area_def, tile_info, sector_id, **kwargs):
        """Generate output NetCDF file from metadata."""
        # format the filename
        try:
            return super(AWIPSTiledWriter, self).get_filename(
                area_id=area_def.area_id,
                rows=area_def.height,
                columns=area_def.width,
                sector_id=sector_id,
                tile_id=tile_info.tile_id,
                tile_number=tile_info.tile_number,
                **kwargs)
        except RuntimeError:
            # the user didn't provide a specific filename, use the template
            return template.get_filename(
                base_dir=self.base_dir,
                area_id=area_def.area_id,
                rows=area_def.height,
                columns=area_def.width,
                sector_id=sector_id,
                tile_id=tile_info.tile_id,
                tile_number=tile_info.tile_number,
                **kwargs)

    def check_tile_exists(self, output_filename):
        """Check if tile exists and report error accordingly."""
        if os.path.isfile(output_filename):
            LOG.info("AWIPS file already exists, will update with new data: %s", output_filename)

    def _save_nonempty_mfdatasets(self, datasets_to_save, output_filenames, **kwargs):
        for dataset_to_save, output_filename in zip(datasets_to_save, output_filenames):
            factors = _extract_factors(dataset_to_save)
            delayed_res = delayed_to_notempty_netcdf(
                dataset_to_save, factors, output_filename, **kwargs)
            yield delayed_res

    def _adjust_metadata_times(self, ds_info):
        debug_shift_time = int(os.environ.get("DEBUG_TIME_SHIFT", 0))
        if debug_shift_time:
            ds_info["start_time"] += timedelta(minutes=debug_shift_time)
            ds_info["end_time"] += timedelta(minutes=debug_shift_time)

    def _get_tile_data_info(self, data_arrs, creation_time, source_name):
        # use the first data array as a "representative" for the group
        ds_info = data_arrs[0].attrs.copy()
        # we want to use our own creation_time
        ds_info['creation_time'] = creation_time
        if source_name is not None:
            ds_info['source_name'] = source_name
        self._adjust_metadata_times(ds_info)
        return ds_info

    # TODO: Add additional untiled variable support
    def save_datasets(self, datasets, sector_id=None,
                      source_name=None,
                      tile_count=(1, 1), tile_size=None,
                      lettered_grid=False, num_subtiles=None,
                      use_end_time=False, use_sector_reference=False,
                      template='polar', check_categories=True,
                      extra_global_attrs=None, environment_prefix='DR',
                      compute=True, **kwargs):
        """Write a series of DataArray objects to multiple NetCDF4 Tile files.

        Args:
            datasets (iterable): Series of gridded :class:`~xarray.DataArray`
                objects with the necessary metadata to be converted to a valid
                tile product file.
            sector_id (str): Name of the region or sector that the provided
                data is on. This name will be written to the NetCDF file and
                will be used as the sector in the AWIPS client for the 'polar'
                template. For lettered
                grids this name should match the name configured in the writer
                YAML. This is required for some templates (ex. default 'polar'
                template) but is defined as a keyword argument
                for better error handling in Satpy.
            source_name (str): Name of producer of these files (ex. "SSEC").
                This name is used to create the output filename for some
                templates.
            environment_prefix (str): Prefix of filenames for some templates.
                For operational real-time data this is usually "OR", "OT" for
                test data, "IR" for test system real-time data, and "IT" for
                test system test data. This defaults to "DR" for "Developer
                Real-time" to avoid anyone accidentally producing files that
                could be mistaken for the operational system.
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
                by 0.5 pixels. See :mod:`satpy.writers.awips_tiled` for more
                information.
            template (str or dict): Name of the template configured in the
                writer YAML file. This can also be a dictionary with a full
                template configuration. See the :mod:`satpy.writers.awips_tiled`
                documentation for more information on templates. Defaults to
                the 'polar' builtin template.
            check_categories (bool): Whether category and flag products should
                be included in the checks for empty or not empty tiles. In
                some cases (ex. data quality flags) category products may look
                like all valid data (a non-empty tile) but shouldn't be used
                to determine the emptiness of the overall tile (good quality
                versus non-existent). Default is True. Set to False to ignore
                category (integer dtype or "flag_meanings" defined) when
                checking for valid data.
            extra_global_attrs (dict): Additional global attributes to be
                added to every produced file. These attributes are applied
                at the end of template rendering and will therefore overwrite
                template generated values with the same global attribute name.
            compute (bool): Compute and write the output immediately using
                dask. Default to ``False``.

        """
        if not isinstance(template, dict):
            template = self.config['templates'][template]
        template = AWIPSNetCDFTemplate(template, swap_end_time=use_end_time)
        area_data_arrs = self._group_by_area(datasets)
        datasets_to_save = []
        output_filenames = []
        creation_time = datetime.utcnow()
        area_tile_data_gen = self._iter_area_tile_info_and_datasets(
            area_data_arrs, template, lettered_grid, sector_id, num_subtiles,
            tile_size, tile_count, use_sector_reference)
        for area_def, tile_info, data_arrs in area_tile_data_gen:
            # TODO: Create Dataset object of all of the sliced-DataArrays (optional)
            ds_info = self._get_tile_data_info(data_arrs,
                                               creation_time,
                                               source_name)
            output_filename = self.get_filename(template, area_def,
                                                tile_info, sector_id,
                                                environment_prefix=environment_prefix,
                                                **ds_info)
            self.check_tile_exists(output_filename)
            # TODO: Provide attribute caching for things that likely won't change (functools lrucache)
            new_ds = template.render(data_arrs, area_def,
                                     tile_info, sector_id,
                                     creation_time=creation_time,
                                     shared_attrs=ds_info,
                                     extra_global_attrs=extra_global_attrs)
            if self.compress:
                new_ds.encoding['zlib'] = True
                for var in new_ds.variables.values():
                    var.encoding['zlib'] = True

            datasets_to_save.append(new_ds)
            output_filenames.append(output_filename)
        if not datasets_to_save:
            # no tiles produced
            return []

        delayed_gen = self._save_nonempty_mfdatasets(datasets_to_save, output_filenames,
                                                     check_categories=check_categories,
                                                     update_existing=True)
        delayeds = self._delay_netcdf_creation(delayed_gen)

        if not compute:
            return delayeds
        return dask.compute(delayeds)

    def _delay_netcdf_creation(self, delayed_gen, precompute=True, use_distributed=False):
        """Workaround random dask and xarray hanging executions.

        In previous implementations this writer called 'to_dataset' directly
        in a delayed function. This seems to cause random deadlocks where
        execution would hang indefinitely.

        """
        delayeds = []
        if precompute:
            dataset_iter = self._get_delayed_iter(use_distributed)
            for dataset_to_save, output_filename, mode in dataset_iter(delayed_gen):
                delayed_save = dataset_to_save.to_netcdf(output_filename, mode, compute=False)
                delayeds.append(delayed_save)
        else:
            for delayed_result in delayed_gen:
                delayeds.append(delayed_result)
        return delayeds

    @staticmethod
    def _get_delayed_iter(use_distributed=False):
        if use_distributed:
            def dataset_iter(_delayed_gen):
                from dask.distributed import as_completed, get_client
                client = get_client()
                futures = client.compute(list(_delayed_gen))
                for _, (dataset_to_save, output_filename, mode) in as_completed(futures, with_results=True):
                    if dataset_to_save is None:
                        continue
                    yield dataset_to_save, output_filename, mode
        else:
            def dataset_iter(_delayed_gen):
                # compute all datasets
                results = dask.compute(_delayed_gen)[0]
                for result in results:
                    if result[0] is None:
                        continue
                    yield result
        return dataset_iter


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
            if (cell_x > total_cells_x) or (cell_y > total_cells_y):
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
        sector_info['projection'],
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


def create_debug_lettered_tiles(**writer_kwargs):
    """Create tile files with tile identifiers "burned" in to the image data for debugging."""
    writer_kwargs['lettered_grid'] = True
    writer_kwargs['num_subtiles'] = (2, 2)  # default, don't use command line argument

    init_kwargs, save_kwargs = AWIPSTiledWriter.separate_init_kwargs(**writer_kwargs)
    writer = AWIPSTiledWriter(**init_kwargs)

    sector_id = save_kwargs['sector_id']
    sector_info = writer.awips_sectors[sector_id]
    area_def, arr = _create_debug_array(sector_info, save_kwargs['num_subtiles'])

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
        **save_kwargs
    )
    return created_files


def main():
    """Command line interface mimicing CSPP Polar2Grid."""
    import argparse
    parser = argparse.ArgumentParser(description="Create AWIPS compatible NetCDF tile files")
    parser.add_argument("--create-debug", action='store_true',
                        help='Create debug NetCDF files to show tile locations in AWIPS')
    parser.add_argument('-v', '--verbose', dest='verbosity', action="count", default=0,
                        help='each occurrence increases verbosity 1 level through '
                             'ERROR-WARNING-INFO-DEBUG (default INFO)')
    parser.add_argument('-l', '--log', dest="log_fn", default=None,
                        help="specify the log filename")

    group_1 = parser.add_argument_group(title="Writer Initialization")
    group_1.add_argument("--backend-configs", nargs="*", dest="backend_configs",
                         help="alternative backend configuration files")
    group_1.add_argument("--compress", action="store_true",
                         help="zlib compress each netcdf file")
    # group_1.add_argument("--fix-awips", action="store_true",
    #                      help="modify NetCDF output to work with the old/broken AWIPS NetCDF library")
    group_2 = parser.add_argument_group(title="Wrtier Save")
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
    group_2.add_argument("--template", default='polar',
                         help="specify the template name to use (default: polar)")
    args = parser.parse_args()

    # Logs are renamed once data the provided start date is known
    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    logging.basicConfig(level=levels[min(3, args.verbosity)], filename=args.log_fn)

    if args.create_debug:
        writer_kwargs = vars(args)
        create_debug_lettered_tiles(**writer_kwargs)
        return
    else:
        raise NotImplementedError("Command line interface not implemented yet for AWIPS tiled writer")


if __name__ == '__main__':
    sys.exit(main())
