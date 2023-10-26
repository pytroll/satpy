#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2019 Satpy developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Utilities for various satpy tests."""

from contextlib import contextmanager
from datetime import datetime
from unittest import mock

import dask.array as da
import numpy as np
from pyresample import create_area_def
from pyresample.geometry import BaseDefinition, SwathDefinition
from xarray import DataArray

from satpy import Scene
from satpy.composites import GenericCompositor, IncompatibleAreas
from satpy.dataset import DataID, DataQuery
from satpy.dataset.dataid import default_id_keys_config, minimal_default_keys_config
from satpy.modifiers import ModifierBase
from satpy.readers.file_handlers import BaseFileHandler

FAKE_FILEHANDLER_START = datetime(2020, 1, 1, 0, 0, 0)
FAKE_FILEHANDLER_END = datetime(2020, 1, 1, 1, 0, 0)


def make_dataid(**items):
    """Make a DataID with default keys."""
    return DataID(default_id_keys_config, **items)


def make_cid(**items):
    """Make a DataID with a minimal set of keys to id composites."""
    return DataID(minimal_default_keys_config, **items)


def make_dsq(**items):
    """Make a dataset query."""
    return DataQuery(**items)


def spy_decorator(method_to_decorate):
    """Fancy decorator to wrap an object while still calling it.

    See https://stackoverflow.com/a/41599695/433202

    """
    tmp_mock = mock.MagicMock()

    def wrapper(self, *args, **kwargs):
        tmp_mock(*args, **kwargs)
        return method_to_decorate(self, *args, **kwargs)

    wrapper.mock = tmp_mock
    return wrapper


def convert_file_content_to_data_array(file_content, attrs=tuple(),
                                       dims=('z', 'y', 'x')):
    """Help old reader tests that still use numpy arrays.

    A lot of old reader tests still use numpy arrays and depend on the
    "var_name/attr/attr_name" convention established before Satpy used xarray
    and dask. While these conventions are still used and should be supported,
    readers need to use xarray DataArrays instead.

    If possible, new tests should be based on pure DataArray objects instead
    of the "var_name/attr/attr_name" style syntax provided by the utility
    file handlers.

    Args:
        file_content (dict): Dictionary of string file keys to fake file data.
        attrs (iterable): Series of attributes to copy to DataArray object from
            file content dictionary. Defaults to no attributes.
        dims (iterable): Dimension names to use for resulting DataArrays.
            The second to last dimension is used for 1D arrays, so for
            dims of ``('z', 'y', 'x')`` this would use ``'y'``. Otherwise, the
            dimensions are used starting with the last, so 2D arrays are
            ``('y', 'x')``
            Dimensions are used in reverse order so the last dimension
            specified is used as the only dimension for 1D arrays and the
            last dimension for other arrays.

    """
    for key, val in file_content.items():
        da_attrs = {}
        for a in attrs:
            if key + '/attr/' + a in file_content:
                da_attrs[a] = file_content[key + '/attr/' + a]

        if isinstance(val, np.ndarray):
            val = da.from_array(val, chunks=4096)
            if val.ndim == 1:
                da_dims = dims[-2]
            elif val.ndim > 1:
                da_dims = tuple(dims[-val.ndim:])
            else:
                da_dims = None

            file_content[key] = DataArray(val, dims=da_dims, attrs=da_attrs)


def _filter_datasets(all_ds, names_or_ids):
    """Help filtering DataIDs by name or DataQuery."""
    # DataID will match a str to the name
    # need to separate them out
    str_filter = [ds_name for ds_name in names_or_ids if isinstance(ds_name, str)]
    id_filter = [ds_id for ds_id in names_or_ids if not isinstance(ds_id, str)]
    for ds_id in all_ds:
        if ds_id in id_filter or ds_id['name'] in str_filter:
            yield ds_id


def _swath_def_of_data_arrays(rows, cols):
    return SwathDefinition(
        DataArray(da.zeros((rows, cols)), dims=('y', 'x')),
        DataArray(da.zeros((rows, cols)), dims=('y', 'x')),
    )


class FakeModifier(ModifierBase):
    """Act as a modifier that performs different modifications."""

    def _handle_res_change(self, datasets, info):
        # assume this is used on the 500m version of ds5
        info['resolution'] = 250
        rep_data_arr = datasets[0]
        y_size = rep_data_arr.sizes['y']
        x_size = rep_data_arr.sizes['x']
        data = da.zeros((y_size * 2, x_size * 2))
        if isinstance(rep_data_arr.attrs['area'], SwathDefinition):
            area = _swath_def_of_data_arrays(y_size * 2, x_size * 2)
            info['area'] = area
        else:
            raise NotImplementedError("'res_change' modifier can't handle "
                                      "AreaDefinition changes yet.")
        return data

    def __call__(self, datasets, optional_datasets=None, **kwargs):
        """Modify provided data depending on the modifier name and input data."""
        if self.attrs['optional_prerequisites']:
            for opt_dep in self.attrs['optional_prerequisites']:
                opt_dep_name = opt_dep if isinstance(opt_dep, str) else opt_dep.get('name', '')
                if 'NOPE' in opt_dep_name or 'fail' in opt_dep_name:
                    continue
                assert (optional_datasets is not None and
                        len(optional_datasets))
        resolution = datasets[0].attrs.get('resolution')
        mod_name = self.attrs['modifiers'][-1]
        data = datasets[0].data
        i = datasets[0].attrs.copy()
        if mod_name == 'res_change' and resolution is not None:
            data = self._handle_res_change(datasets, i)
        elif 'incomp_areas' in mod_name:
            raise IncompatibleAreas(
                "Test modifier 'incomp_areas' always raises IncompatibleAreas")
        self.apply_modifier_info(datasets[0].attrs, i)
        return DataArray(data,
                         dims=datasets[0].dims,
                         # coords=datasets[0].coords,
                         attrs=i)


class FakeCompositor(GenericCompositor):
    """Act as a compositor that produces fake RGB data."""

    def __call__(self, projectables, nonprojectables=None, **kwargs):
        """Produce test compositor data depending on modifiers and input data provided."""
        if projectables:
            projectables = self.match_data_arrays(projectables)
        if nonprojectables:
            self.match_data_arrays(nonprojectables)
        info = self.attrs.copy()
        if self.attrs['name'] in ('comp14', 'comp26'):
            # used as a test when composites update the dataset id with
            # information from prereqs
            info['resolution'] = 555
        if self.attrs['name'] in ('comp24', 'comp25'):
            # other composites that copy the resolution from inputs
            info['resolution'] = projectables[0].attrs.get('resolution')
        if len(projectables) != len(self.attrs['prerequisites']):
            raise ValueError("Not enough prerequisite datasets passed")

        info.update(kwargs)
        if projectables:
            info['area'] = projectables[0].attrs['area']
            dim_sizes = projectables[0].sizes
        else:
            # static_image
            dim_sizes = {'y': 4, 'x': 5}
        return DataArray(data=da.zeros((dim_sizes['y'], dim_sizes['x'], 3)),
                         attrs=info,
                         dims=['y', 'x', 'bands'],
                         coords={'bands': ['R', 'G', 'B']})


class FakeFileHandler(BaseFileHandler):
    """Fake file handler to be used by test readers."""

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Initialize file handler and accept all keyword arguments."""
        self.kwargs = kwargs
        super().__init__(filename, filename_info, filetype_info)

    @property
    def start_time(self):
        """Get static start time datetime object."""
        return FAKE_FILEHANDLER_START

    @property
    def end_time(self):
        """Get static end time datetime object."""
        return FAKE_FILEHANDLER_END

    @property
    def sensor_names(self):
        """Get sensor name from filetype configuration."""
        sensor = self.filetype_info.get('sensor', 'fake_sensor')
        return {sensor}

    def get_dataset(self, data_id: DataID, ds_info: dict):
        """Get fake DataArray for testing."""
        if data_id['name'] == 'ds9_fail_load':
            raise KeyError("Can't load '{}' because it is supposed to "
                           "fail.".format(data_id['name']))
        attrs = data_id.to_dict()
        attrs.update(ds_info)
        attrs['sensor'] = self.filetype_info.get('sensor', 'fake_sensor')
        attrs['platform_name'] = 'fake_platform'
        attrs['start_time'] = self.start_time
        attrs['end_time'] = self.end_time
        res = attrs.get('resolution', 250)
        rows = cols = {
            250: 20,
            500: 10,
            1000: 5,
        }.get(res, 5)
        return DataArray(data=da.zeros((rows, cols)),
                         attrs=attrs,
                         dims=['y', 'x'])

    def available_datasets(self, configured_datasets=None):
        """Report YAML datasets available unless 'not_available' is specified during creation."""
        not_available_names = self.kwargs.get("not_available", [])
        for is_avail, ds_info in (configured_datasets or []):
            if is_avail is not None:
                # some other file handler said it has this dataset
                # we don't know any more information than the previous
                # file handler so let's yield early
                yield is_avail, ds_info
                continue
            ft_matches = self.file_type_matches(ds_info['file_type'])
            if not ft_matches:
                yield None, ds_info
                continue
            # mimic what happens when a reader "knows" about one variable
            # but the files loaded don't have that variable
            is_avail = ds_info["name"] not in not_available_names
            yield is_avail, ds_info


class CustomScheduler(object):
    """Scheduler raising an exception if data are computed too many times."""

    def __init__(self, max_computes=1):
        """Set starting and maximum compute counts."""
        self.max_computes = max_computes
        self.total_computes = 0

    def __call__(self, dsk, keys, **kwargs):
        """Compute dask task and keep track of number of times we do so."""
        import dask
        self.total_computes += 1
        if self.total_computes > self.max_computes:
            raise RuntimeError("Too many dask computations were scheduled: "
                               "{}".format(self.total_computes))
        return dask.get(dsk, keys, **kwargs)


@contextmanager
def assert_maximum_dask_computes(max_computes=1):
    """Context manager to make sure dask computations are not executed more than ``max_computes`` times."""
    import dask
    with dask.config.set(scheduler=CustomScheduler(max_computes=max_computes)) as new_config:
        yield new_config


def make_fake_scene(content_dict, daskify=False, area=True,
                    common_attrs=None):
    """Create a fake Scene.

    Create a fake Scene object from fake data.  Data are provided in
    the ``content_dict`` argument.  In ``content_dict``, keys should be
    strings or DataID, and values may be either numpy.ndarray
    or xarray.DataArray, in either case with exactly two dimensions.
    The function will convert each of the numpy.ndarray objects into
    an xarray.DataArray and assign those as datasets to a Scene object.
    A fake AreaDefinition will be assigned for each array, unless disabled
    by passing ``area=False``.  When areas are automatically generated,
    arrays with the same shape will get the same area.

    This function is exclusively intended for testing purposes.

    If regular ndarrays are passed and the keyword argument daskify is
    True, DataArrays will be created as dask arrays.  If False (default),
    regular DataArrays will be created.  When the user passes xarray.DataArray
    objects then this flag has no effect.

    Args:
        content_dict (Mapping): Mapping where keys correspond to objects
            accepted by ``Scene.__setitem__``, i.e. strings or DataID,
            and values may be either ``numpy.ndarray`` or
            ``xarray.DataArray``.
        daskify (bool): optional, to use dask when converting
            ``numpy.ndarray`` to ``xarray.DataArray``.  No effect when the
            values in ``content_dict`` are already ``xarray.DataArray``.
        area (bool or BaseDefinition): Can be ``True``, ``False``, or an
            instance of ``pyresample.geometry.BaseDefinition`` such as
            ``AreaDefinition`` or ``SwathDefinition``.  If ``True``, which is
            the default, automatically generate areas with the name "test-area".
            If ``False``, values will not have assigned areas.  If an instance
            of ``pyresample.geometry.BaseDefinition``, those instances will be
            used for all generated fake datasets.  Warning: Passing an area as
            a string (``area="germ"``) is not supported.
        common_attrs (Mapping): optional, additional attributes that will
            be added to every dataset in the scene.

    Returns:
        Scene object with datasets corresponding to content_dict.
    """
    if common_attrs is None:
        common_attrs = {}
    sc = Scene()
    for (did, arr) in content_dict.items():
        extra_attrs = common_attrs.copy()
        if area:
            extra_attrs["area"] = _get_fake_scene_area(arr, area)
        sc[did] = _get_did_for_fake_scene(area, arr, extra_attrs, daskify)
    return sc


def _get_fake_scene_area(arr, area):
    """Get area for fake scene.  Helper for make_fake_scene."""
    if isinstance(area, BaseDefinition):
        return area
    return create_area_def(
        "test-area",
        {"proj": "eqc", "lat_ts": 0, "lat_0": 0, "lon_0": 0,
         "x_0": 0, "y_0": 0, "ellps": "sphere", "units": "m",
         "no_defs": None, "type": "crs"},
        units="m",
        shape=arr.shape,
        resolution=1000,
        center=(0, 0))


def _get_did_for_fake_scene(area, arr, extra_attrs, daskify):
    """Add instance to fake scene.  Helper for make_fake_scene."""
    from satpy.resample import add_crs_xy_coords
    if isinstance(arr, DataArray):
        new = arr.copy()  # don't change attributes of input
        new.attrs.update(extra_attrs)
    else:
        if daskify:
            arr = da.from_array(arr)
        new = DataArray(
                arr,
                dims=("y", "x"),
                attrs=extra_attrs)
    if area:
        new = add_crs_xy_coords(new, extra_attrs["area"])
    return new


def assert_attrs_equal(attrs, attrs_exp, tolerance=0):
    """Test that attributes are equal.

    Walks dictionary recursively. Numerical attributes are compared with
    the given relative tolerance.
    """
    keys_diff = set(attrs).difference(set(attrs_exp))
    assert not keys_diff, "Different set of keys: {}".format(keys_diff)
    for key in attrs_exp:
        err_msg = "Attribute {} does not match expectation".format(key)
        if isinstance(attrs[key], dict):
            assert_attrs_equal(attrs[key], attrs_exp[key], tolerance)
        else:
            try:
                np.testing.assert_allclose(
                    attrs[key],
                    attrs_exp[key],
                    rtol=tolerance,
                    err_msg=err_msg
                )
            except TypeError:
                assert attrs[key] == attrs_exp[key], err_msg
