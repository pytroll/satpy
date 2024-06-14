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
"""Utility to generate a CF-compliant Datasets."""
import logging
import warnings

import xarray as xr

from satpy.writers.cf_writer import CF_DTYPES, CF_VERSION

logger = logging.getLogger(__name__)


def _get_extra_ds(dataarray, keys=None):
    """Get the ancillary_variables DataArrays associated to a dataset."""
    dict_datarrays = {}
    # Retrieve ancillary variable datarrays
    for ancillary_dataarray in dataarray.attrs.get("ancillary_variables", []):
        ancillary_variable = ancillary_dataarray.name
        if keys and ancillary_variable not in keys:
            keys.append(ancillary_variable)
            dict_datarrays.update(_get_extra_ds(ancillary_dataarray, keys=keys))
    # Add input dataarray
    dict_datarrays[dataarray.attrs["name"]] = dataarray
    return dict_datarrays


def _get_group_dataarrays(group_members, list_dataarrays):
    """Yield DataArrays that are part of a specific group."""
    return [da for da in list_dataarrays if da.attrs["name"] in group_members]


def _get_groups(groups, list_datarrays):
    """Return a dictionary with the list of xr.DataArray associated to each group.

    If no groups (groups=None), return all DataArray attached to a single None key.
    Else, collect the DataArrays associated to each group.
    """
    if groups is None:
        return {None: list_datarrays}

    return {group_name: _get_group_dataarrays(group_members, list_datarrays)
            for group_name, group_members in groups.items()}


def _collect_cf_dataset(list_dataarrays,
                        epoch=None,
                        flatten_attrs=False,
                        exclude_attrs=None,
                        include_lonlats=True,
                        pretty=False,
                        include_orig_name=True,
                        numeric_name_prefix="CHANNEL_"):
    """Process a list of xr.DataArray and return a dictionary with CF-compliant xr.Dataset.

    Args:
        list_dataarrays (list): List of DataArrays to make CF compliant and merge into an xr.Dataset.
        epoch (str, optional): Reference time for encoding the time coordinates.
            Example format: "seconds since 1970-01-01 00:00:00".
            If None, the default reference time is defined using `from satpy.cf.coords import EPOCH`.
        flatten_attrs (bool, optional): If True, flatten dict-type attributes.
        exclude_attrs (list, optional): List of xr.DataArray attribute names to be excluded.
        include_lonlats (bool, optional): If True, includes 'latitude' and 'longitude' coordinates also for a
            satpy.Scene defined on an AreaDefinition.
            If the 'area' attribute is a SwathDefinition, it always includes latitude and longitude coordinates.
        pretty (bool, optional): Don't modify coordinate names, if possible.
            Makes the file prettier, but possibly less consistent.
        include_orig_name (bool, optional): Include the original dataset name as a variable attribute in the xr.Dataset.
        numeric_name_prefix (str, optional): Prefix to add to each variable with a name starting with a digit.
            Use '' or None to leave this out.

    Returns:
        xr.Dataset: A partially CF-compliant xr.Dataset.
    """
    from satpy.cf.area import area2cf
    from satpy.cf.coords import (
        add_coordinates_attrs_coords,
        check_unique_projection_coords,
        ensure_unique_nondimensional_coords,
        has_projection_coords,
    )
    from satpy.cf.data_array import make_cf_data_array

    # Create dictionary of input datarrays
    # --> Since keys=None, it doesn't never retrieve ancillary variables !!!
    dict_dataarrays = {}
    for dataarray in list_dataarrays:
        dict_dataarrays.update(_get_extra_ds(dataarray))

    # Check if one DataArray in the collection has 'longitude' or 'latitude'
    got_lonlats = has_projection_coords(dict_dataarrays)

    # Sort dictionary by keys name
    dict_dataarrays = dict(sorted(dict_dataarrays.items()))

    dict_cf_dataarrays = {}
    for dataarray in dict_dataarrays.values():
        dataarray_type = dataarray.dtype
        if dataarray_type not in CF_DTYPES:
            warnings.warn(
                f"dtype {dataarray_type} not compatible with {CF_VERSION}.",
                stacklevel=3
            )
        # Deep copy the datarray since adding/modifying attributes and coordinates
        dataarray = dataarray.copy(deep=True)

        # Add CF-compliant area information from the pyresample area
        # - If include_lonlats=True, add latitude and longitude coordinates
        # - Add grid_mapping attribute to the DataArray
        # - Return the CRS DataArray as first list element
        # - Return the CF-compliant input DataArray as second list element
        try:
            list_new_dataarrays = area2cf(dataarray,
                                          include_lonlats=include_lonlats,
                                          got_lonlats=got_lonlats)
        except KeyError:
            list_new_dataarrays = [dataarray]

        # Ensure each DataArray is CF-compliant
        # --> NOTE: Here the CRS DataArray is repeatedly overwrited
        # --> NOTE: If the input list_dataarrays have different pyresample areas with the same name
        #           area information can be lost here !!!
        for new_dataarray in list_new_dataarrays:
            new_dataarray = make_cf_data_array(new_dataarray,
                                               epoch=epoch,
                                               flatten_attrs=flatten_attrs,
                                               exclude_attrs=exclude_attrs,
                                               include_orig_name=include_orig_name,
                                               numeric_name_prefix=numeric_name_prefix)
            dict_cf_dataarrays[new_dataarray.name] = new_dataarray

    # Check all DataArrays have same projection coordinates
    check_unique_projection_coords(dict_cf_dataarrays)

    # Add to DataArrays the coordinates specified in the 'coordinates' attribute
    # - Deal with the 'coordinates' attributes indicating lat/lon coords
    # - The 'coordinates' attribute is dropped from each DataArray
    dict_cf_dataarrays = add_coordinates_attrs_coords(dict_cf_dataarrays)

    # Ensure non-dimensional coordinates to be unique across DataArrays
    # --> If not unique, prepend the DataArray name to the coordinate
    # --> If unique, does not prepend the DataArray name only if pretty=True
    # --> 'longitude' and 'latitude' coordinates are not prepended
    dict_cf_dataarrays = ensure_unique_nondimensional_coords(dict_cf_dataarrays, pretty=pretty)

    # Create a xr.Dataset
    ds = xr.Dataset(dict_cf_dataarrays)
    return ds


def collect_cf_datasets(list_dataarrays,
                        header_attrs=None,
                        exclude_attrs=None,
                        flatten_attrs=False,
                        pretty=True,
                        include_lonlats=True,
                        epoch=None,
                        include_orig_name=True,
                        numeric_name_prefix="CHANNEL_",
                        groups=None):
    """Process a list of xr.DataArray and return a dictionary with CF-compliant xr.Datasets.

    If the xr.DataArrays does not share the same dimensions, it creates a collection
    of xr.Datasets sharing the same dimensions.

    Args:
        list_dataarrays (list): List of DataArrays to make CF compliant and merge into groups of xr.Datasets.
        header_attrs (dict): Global attributes of the output xr.Dataset.
        epoch (str, optional): Reference time for encoding the time coordinates.
            Example format: "seconds since 1970-01-01 00:00:00".
            If None, the default reference time is retrieved using `from satpy.cf.coords import EPOCH`.
        flatten_attrs (bool, optional): If True, flatten dict-type attributes.
        exclude_attrs (list, optional): List of xr.DataArray attribute names to be excluded.
        include_lonlats (bool, optional): If True, includes 'latitude' and 'longitude' coordinates also
            for a satpy.Scene defined on an AreaDefinition.
            If the 'area' attribute is a SwathDefinition, it always includes latitude and longitude coordinates.
        pretty (bool, optional): Don't modify coordinate names, if possible.
            Makes the file prettier, but possibly less consistent.
        include_orig_name (bool, optional): Include the original dataset name as a variable attribute in the xr.Dataset.
        numeric_name_prefix (str, optional): Prefix to add to each variable with a name starting with a digit.
            Use '' or None to leave this out.
        groups (dict, optional): Group datasets according to the given assignment:
            `{'<group_name>': ['dataset_name1', 'dataset_name2', ...]}`.
            Used to create grouped netCDFs using the CF_Writer. If None, no groups will be created.

    Returns:
        tuple: A tuple containing:
            - grouped_datasets (dict): A dictionary of CF-compliant xr.Dataset: {group_name: xr.Dataset}.
            - header_attrs (dict): Global attributes to be attached to the xr.Dataset / netCDF4.
    """
    from satpy.cf.attrs import preprocess_header_attrs
    from satpy.cf.coords import add_time_bounds_dimension

    if not list_dataarrays:
        raise RuntimeError("None of the requested datasets have been "
                           "generated or could not be loaded. Requested "
                           "composite inputs may need to have matching "
                           "dimensions (eg. through resampling).")

    header_attrs = preprocess_header_attrs(header_attrs=header_attrs,
                                           flatten_attrs=flatten_attrs)

    # Retrieve groups
    # - If groups is None: {None: list_dataarrays}
    # - if groups not None: {group_name: [xr.DataArray, xr.DataArray ,..], ...}
    # Note: if all dataset names are wrong, behave like groups = None !
    grouped_dataarrays = _get_groups(groups, list_dataarrays)
    is_grouped = len(grouped_dataarrays) >= 2

    # If not grouped, add CF conventions.
    # - If 'Conventions' key already present, do not overwrite !
    if "Conventions" not in header_attrs and not is_grouped:
        header_attrs["Conventions"] = CF_VERSION

    # Create dictionary of group xr.Datasets
    # --> If no groups (groups=None) --> group_name=None
    grouped_datasets = {}
    for group_name, group_dataarrays in grouped_dataarrays.items():
        ds = _collect_cf_dataset(
            list_dataarrays=group_dataarrays,
            epoch=epoch,
            flatten_attrs=flatten_attrs,
            exclude_attrs=exclude_attrs,
            include_lonlats=include_lonlats,
            pretty=pretty,
            include_orig_name=include_orig_name,
            numeric_name_prefix=numeric_name_prefix)

        if not is_grouped:
            ds.attrs = header_attrs

        if "time" in ds:
            ds = add_time_bounds_dimension(ds, time="time")

        grouped_datasets[group_name] = ds
    return grouped_datasets, header_attrs
