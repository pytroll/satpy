"""Set CF-compliant spatial and temporal coordinates."""
from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from contextlib import suppress

import numpy as np
import xarray as xr
from dask.base import tokenize
from pyproj import CRS
from pyresample.geometry import AreaDefinition, SwathDefinition

logger = logging.getLogger(__name__)


EPOCH = u"seconds since 1970-01-01 00:00:00"


def add_xy_coords_attrs(data_arr: xr.DataArray) -> xr.DataArray:
    """Add relevant attributes to x, y coordinates."""
    # If there are no coords, return dataarray
    if not data_arr.coords.keys() & {"x", "y", "crs"}:
        return data_arr
    # If projected area
    if _is_projected(data_arr):
        data_arr = _add_xy_projected_coords_attrs(data_arr)
    else:
        data_arr = _add_xy_geographic_coords_attrs(data_arr)
    if "crs" in data_arr.coords:
        data_arr = data_arr.drop_vars("crs")
    return data_arr


def _is_projected(data_arr: xr.DataArray) -> bool:
    """Guess whether data are projected or not."""
    crs = _try_to_get_crs(data_arr)
    if crs:
        return crs.is_projected
    units = _try_get_units_from_coords(data_arr)
    if units:
        if units.endswith("m"):
            return True
        if units.startswith("degrees"):
            return False
    logger.warning("Failed to tell if data are projected. Assuming yes.")
    return True


def _is_area(data_arr: xr.DataArray) -> bool:
     return isinstance(data_arr.attrs["area"], AreaDefinition)


def _is_swath(data_arr: xr.DataArray) -> bool:
     return isinstance(data_arr.attrs["area"], SwathDefinition)


def _try_to_get_crs(data_arr: xr.DataArray) -> CRS:
    """Try to get a CRS from attributes."""
    if "area" in data_arr.attrs:
        if _is_area(data_arr):
            return data_arr.attrs["area"].crs
        if not _is_swath(data_arr):
            logger.warning(
                f"Could not tell CRS from area of type {type(data_arr.attrs['area']).__name__:s}. "
                "Assuming projected CRS.")
    if "crs" in data_arr.coords:
        return data_arr.coords["crs"].item()


def _try_get_units_from_coords(data_arr: xr.DataArray) -> str | None:
    """Try to retrieve coordinate x/y units."""
    for c in ["x", "y"]:
        with suppress(KeyError):
            # If the data has only 1 dimension, it has only one of x or y coords
            if "units" in data_arr.coords[c].attrs:
                return data_arr.coords[c].attrs["units"]
    return None


def _add_xy_projected_coords_attrs(data_arr: xr.DataArray, x: str = "x", y: str = "y") -> xr.DataArray:
    """Add relevant attributes to x, y coordinates of a projected CRS."""
    if x in data_arr.coords:
        data_arr[x].attrs["standard_name"] = "projection_x_coordinate"
        data_arr[x].attrs["units"] = "m"
    if y in data_arr.coords:
        data_arr[y].attrs["standard_name"] = "projection_y_coordinate"
        data_arr[y].attrs["units"] = "m"
    return data_arr


def _add_xy_geographic_coords_attrs(data_arr: xr.DataArray, x: str = "x", y: str = "y") -> xr.DataArray:
    """Add relevant attributes to x, y coordinates of a geographic CRS."""
    if x in data_arr.coords:
        data_arr[x].attrs["standard_name"] = "longitude"
        data_arr[x].attrs["units"] = "degrees_east"
    if y in data_arr.coords:
        data_arr[y].attrs["standard_name"] = "latitude"
        data_arr[y].attrs["units"] = "degrees_north"
    return data_arr


def set_cf_time_info(data_arr: xr.DataArray, epoch: str | None) -> xr.DataArray:
    """Set CF time attributes and encoding.

    It expand the DataArray with a time dimension if does not yet exists.

    The function assumes

        - that x and y dimensions have at least shape > 1
        - the time coordinate has size 1

    """
    if epoch is None:
        epoch = EPOCH

    data_arr["time"].encoding["units"] = epoch
    data_arr["time"].attrs["standard_name"] = "time"
    data_arr["time"].attrs.pop("bounds", None)

    if "time" not in data_arr.dims and data_arr["time"].size not in data_arr.shape:
        data_arr = data_arr.expand_dims("time")

    return data_arr


def has_projection_coords(data_arrays: dict[str, xr.DataArray]) -> bool:
    """Check if DataArray collection has a "longitude" or "latitude" DataArray."""
    return any(_is_lon_or_lat_dataarray(data_arr) for data_arr in data_arrays.values())


def _is_lon_or_lat_dataarray(data_arr: xr.DataArray) -> bool:
    """Check if the DataArray represents the latitude or longitude coordinate."""
    return data_arr.attrs.get("standard_name", "") in ("longitude", "latitude")


def _get_is_nondimensional_coords_dict(data_arrays: dict[str, xr.DataArray]) -> dict[str, bool]:
    tokens = defaultdict(set)
    for data_arr in data_arrays.values():
        for coord_name in data_arr.coords:
            if not _is_lon_or_lat_dataarray(data_arr[coord_name]) and coord_name not in data_arr.dims:
                tokens[coord_name].add(tokenize(data_arr[coord_name].data))
    return dict([(coord_name, len(tokens) == 1) for coord_name, tokens in tokens.items()])


def _warn_if_pretty_but_not_unique(pretty, coord_name):
    """Warn if coordinates cannot be pretty-formatted due to non-uniqueness."""
    if pretty:
        warnings.warn(
            f'Cannot pretty-format "{coord_name}" coordinates because they are '
            'not identical among the given datasets',
            stacklevel=2
        )


def _rename_coords(data_arrays: dict[str, xr.DataArray], coord_name: str) -> dict[str, xr.DataArray]:
    """Rename coordinates in the datasets."""
    for name, dataarray in data_arrays.items():
        if coord_name in dataarray.coords:
            rename = {coord_name: f"{name}_{coord_name}"}
            data_arrays[name] = dataarray.rename(rename)
    return data_arrays


def ensure_unique_nondimensional_coords(
        data_arrays: dict[str, xr.DataArray],
        pretty: bool = False
) -> dict[str, xr.DataArray]:
    """Make non-dimensional coordinates unique among all datasets.

    Non-dimensional coordinates, such as scanline timestamps,
    may occur in multiple datasets with the same name and dimension
    but different values.

    In order to avoid conflicts, prepend the dataset name to the coordinate name.
    If a non-dimensional coordinate is unique among all datasets and ``pretty=True``,
    its name will not be modified.

    Since all datasets must have the same projection coordinates,
    this is not applied to latitude and longitude.

    Args:
        data_arrays:
            Dictionary of (dataset name, dataset)
        pretty:
            Don't modify coordinate names, if possible. Makes the file prettier, but possibly less consistent.

    Returns:
        Dictionary holding the updated datasets

    """
    # Determine which non-dimensional coordinates are unique
    # - coords_unique has structure: {coord_name: True/False}
    is_coords_unique_dict = _get_is_nondimensional_coords_dict(data_arrays)

    # Prepend dataset name, if not unique or no pretty-format desired
    new_dict_dataarrays = data_arrays.copy()
    for coord_name, unique in is_coords_unique_dict.items():
        if not pretty or not unique:
            _warn_if_pretty_but_not_unique(pretty, coord_name)
            new_dict_dataarrays = _rename_coords(new_dict_dataarrays, coord_name)
    return new_dict_dataarrays


def check_unique_projection_coords(data_arrays: dict[str, xr.DataArray]) -> None:
    """Check that all datasets share the same projection coordinates x/y."""
    unique_x = set()
    unique_y = set()
    for dataarray in data_arrays.values():
        if "y" in dataarray.dims:
            token_y = tokenize(dataarray["y"].data)
            unique_y.add(token_y)
        if "x" in dataarray.dims:
            token_x = tokenize(dataarray["x"].data)
            unique_x.add(token_x)
    if len(unique_x) > 1 or len(unique_y) > 1:
        raise ValueError("Datasets to be saved in one file (or one group) must have identical projection coordinates."
                         "Please group them by area or save them in separate files.")


def add_coordinates_attrs_coords(data_arrays: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    """Add to DataArrays the coordinates specified in the 'coordinates' attribute.

    It deal with the 'coordinates' attributes indicating lat/lon coords
    The 'coordinates' attribute is dropped from each DataArray

    If the `coordinates` attribute of a data array links to other dataarrays in the scene, for example
    `coordinates='lon lat'`, add them as coordinates to the data array and drop that attribute.

    In the final call to `xr.Dataset.to_netcdf()` all coordinate relations will be resolved
    and the `coordinates` attributes be set automatically.
    """
    for dataarray_name in data_arrays.keys():
        data_arrays = _add_declared_coordinates(data_arrays,
                                                dataarray_name=dataarray_name)
        # Drop 'coordinates' attribute in any case to avoid conflicts in xr.Dataset.to_netcdf()
        data_arrays[dataarray_name].attrs.pop("coordinates", None)
    return data_arrays


def _add_declared_coordinates(data_arrays: dict[str, xr.DataArray], dataarray_name: str) -> dict[str, xr.DataArray]:
    """Add declared coordinates to the dataarray if they exist."""
    dataarray = data_arrays[dataarray_name]
    declared_coordinates = _get_coordinates_list(dataarray)
    for coord in declared_coordinates:
        if coord not in dataarray.coords:
            data_arrays = _try_add_coordinate(data_arrays,
                                              dataarray_name=dataarray_name,
                                              coord=coord)
    return data_arrays


def _try_add_coordinate(
        data_arrays: dict[str, xr.DataArray],
        dataarray_name: str,
        coord: str
) -> dict[str, xr.DataArray]:
    """Try to add a coordinate to the dataarray, warn if not possible."""
    try:
        dataarray_dims = set(data_arrays[dataarray_name].dims)
        coordinate_dims = set(data_arrays[coord].dims)
        dimensions_to_squeeze = list(coordinate_dims - dataarray_dims)
        data_arrays[dataarray_name][coord] = data_arrays[coord].squeeze(dimensions_to_squeeze, drop=True)
    except KeyError:
        warnings.warn(
            f'Coordinate "{coord}" referenced by dataarray {dataarray_name} does not '
            'exist, dropping reference.',
            stacklevel=2
        )
    return data_arrays


def _get_coordinates_list(data_arr: xr.DataArray) -> list[str]:
    """Return a list with the coordinates names specified in the 'coordinates' attribute."""
    declared_coordinates = data_arr.attrs.get("coordinates", [])
    if isinstance(declared_coordinates, str):
        declared_coordinates = declared_coordinates.split(" ")
    return declared_coordinates


def add_time_bounds_dimension(ds: xr.Dataset, time: str = "time") -> xr.Dataset:
    """Add time bound dimension to xr.Dataset."""
    start_times = []
    end_times = []
    for _var_name, data_array in ds.items():
        start_times.append(data_array.attrs.get("start_time", None))
        end_times.append(data_array.attrs.get("end_time", None))

    start_time = min(start_time for start_time in start_times
                     if start_time is not None)
    end_time = min(end_time for end_time in end_times
                   if end_time is not None)
    ds["time_bnds"] = xr.DataArray([[np.datetime64(start_time, "ns"),
                                     np.datetime64(end_time, "ns")]],
                                   dims=["time", "bnds_1d"])
    ds[time].attrs["bounds"] = "time_bnds"
    ds[time].attrs["standard_name"] = "time"
    return ds
