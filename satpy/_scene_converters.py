# Copyright (c) 2023 Satpy developers
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
"""Helper functions for converting the Scene object to some other object."""

import xarray as xr

from satpy.dataset import DataID


def _get_dataarrays_from_identifiers(scn, identifiers):
    """Return a list of DataArray based on a single or list of identifiers.

    An identifier can be a DataID or a string with name of a valid DataID.
    """
    if isinstance(identifiers, (str, DataID)):
        identifiers = [identifiers]

    if identifiers is not None:
        dataarrays = [scn[ds] for ds in identifiers]
    else:
        dataarrays = [scn._datasets.get(ds) for ds in scn._wishlist]
        dataarrays = [dataarray for dataarray in dataarrays if dataarray is not None]
    return dataarrays


def to_xarray(scn,
              datasets=None,  # DataID
              header_attrs=None,
              exclude_attrs=None,
              flatten_attrs=False,
              pretty=True,
              include_lonlats=True,
              epoch=None,
              include_orig_name=True,
              numeric_name_prefix='CHANNEL_'):
    """Merge all xr.DataArray(s) of a satpy.Scene to a CF-compliant xarray object.

    If all Scene DataArrays are on the same area, it returns an xr.Dataset.
    If Scene DataArrays are on different areas, currently it fails, although
    in future we might return a DataTree object, grouped by area.

    Parameters
    ----------
    scn: satpy.Scene
        Satpy Scene.
    datasets (iterable):
        List of Satpy Scene datasets to include in the output xr.Dataset.
        Elements can be string name, a wavelength as a number, a DataID,
        or DataQuery object.
        If None (the default), it include all loaded Scene datasets.
    header_attrs:
        Global attributes of the output xr.Dataset.
    epoch (str):
        Reference time for encoding the time coordinates (if available).
        Example format: "seconds since 1970-01-01 00:00:00".
        If None, the default reference time is retrieved using "from satpy.cf_writer import EPOCH"
    flatten_attrs (bool):
        If True, flatten dict-type attributes.
    exclude_attrs (list):
        List of xr.DataArray attribute names to be excluded.
    include_lonlats (bool):
        If True, it includes 'latitude' and 'longitude' coordinates.
        If the 'area' attribute is a SwathDefinition, it always includes
        latitude and longitude coordinates.
    pretty (bool):
        Don't modify coordinate names, if possible. Makes the file prettier,
        but possibly less consistent.
    include_orig_name (bool).
        Include the original dataset name as a variable attribute in the xr.Dataset.
    numeric_name_prefix (str):
        Prefix to add the each variable with name starting with a digit.
        Use '' or None to leave this out.

    Returns
    -------
    ds, xr.Dataset
        A CF-compliant xr.Dataset

    """
    from satpy.writers.cf_writer import EPOCH, collect_cf_datasets

    if epoch is None:
        epoch = EPOCH

    # Get list of DataArrays
    if datasets is None:
        datasets = list(scn.keys())  # list all loaded DataIDs
    list_dataarrays = _get_dataarrays_from_identifiers(scn, datasets)

    # Check that some DataArray could be returned
    if len(list_dataarrays) == 0:
        return xr.Dataset()

    # Collect xr.Dataset for each group
    grouped_datasets, header_attrs = collect_cf_datasets(list_dataarrays=list_dataarrays,
                                                         header_attrs=header_attrs,
                                                         exclude_attrs=exclude_attrs,
                                                         flatten_attrs=flatten_attrs,
                                                         pretty=pretty,
                                                         include_lonlats=include_lonlats,
                                                         epoch=epoch,
                                                         include_orig_name=include_orig_name,
                                                         numeric_name_prefix=numeric_name_prefix,
                                                         groups=None)
    if len(grouped_datasets) == 1:
        ds = grouped_datasets[None]
        return ds
    else:
        msg = """The Scene object contains datasets with different areas.
                  Resample the Scene to have matching dimensions using i.e. scn.resample(resampler="native") """
        raise NotImplementedError(msg)
