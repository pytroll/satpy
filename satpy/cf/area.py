# Copyright (c) 2017-2023 Satpy developers
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
"""CF processing of pyresample area information."""
import logging

import xarray as xr
from packaging.version import Version
from pyresample.geometry import AreaDefinition, SwathDefinition

logger = logging.getLogger(__name__)


def _add_lonlat_coords(data_arr: xr.DataArray) -> xr.DataArray:
    """Add 'longitude' and 'latitude' coordinates to DataArray."""
    data_arr = data_arr.copy()
    area = data_arr.attrs["area"]
    ignore_dims = {dim: 0 for dim in data_arr.dims if dim not in ["x", "y"]}
    chunks = getattr(data_arr.isel(**ignore_dims), "chunks", None)
    lons, lats = area.get_lonlats(chunks=chunks)
    data_arr["longitude"] = xr.DataArray(lons, dims=["y", "x"],
                                         attrs={"name": "longitude",
                                                 "standard_name": "longitude",
                                                 "units": "degrees_east"},
                                         name="longitude")
    data_arr["latitude"] = xr.DataArray(lats, dims=["y", "x"],
                                        attrs={"name": "latitude",
                                                "standard_name": "latitude",
                                                "units": "degrees_north"},
                                        name="latitude")
    return data_arr


def _create_grid_mapping(area):
    """Create the grid mapping instance for `area`."""
    import pyproj

    if Version(pyproj.__version__) < Version("2.4.1"):
        # technically 2.2, but important bug fixes in 2.4.1
        raise ImportError("'cf' writer requires pyproj 2.4.1 or greater")
    # let pyproj do the heavily lifting (pyproj 2.0+ required)
    grid_mapping = area.crs.to_cf()
    return area.area_id, grid_mapping


def _add_grid_mapping(data_arr: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Convert an area to at CF grid mapping."""
    data_arr = data_arr.copy()
    area = data_arr.attrs["area"]
    gmapping_var_name, attrs = _create_grid_mapping(area)
    data_arr.attrs["grid_mapping"] = gmapping_var_name
    return data_arr, xr.DataArray(0, attrs=attrs, name=gmapping_var_name)


def area2cf(data_arr: xr.DataArray, include_lonlats: bool = False, got_lonlats: bool = False) -> list[xr.DataArray]:
    """Convert an area to at CF grid mapping or lon and lats."""
    res = []
    include_lonlats = include_lonlats or isinstance(data_arr.attrs["area"], SwathDefinition)
    is_area_def = isinstance(data_arr.attrs["area"], AreaDefinition)
    if not got_lonlats and include_lonlats:
        data_arr = _add_lonlat_coords(data_arr)
    if is_area_def:
        data_arr, gmapping = _add_grid_mapping(data_arr)
        res.append(gmapping)
    res.append(data_arr)
    return res
