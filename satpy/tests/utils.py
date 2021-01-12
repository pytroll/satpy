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

from datetime import datetime
from unittest import mock

from satpy.dataset import DataID, DataQuery
from satpy.dataset.dataid import default_id_keys_config, minimal_default_keys_config
from satpy.readers.yaml_reader import FileYAMLReader


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
    """Fancy decorate to wrap an object while still calling it.

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
    from xarray import DataArray
    import dask.array as da
    import numpy as np
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


def test_datasets():
    """Get list of various test datasets."""
    d = [
        make_dataid(name='ds1', resolution=250, calibration='reflectance'),
        make_dataid(name='ds2', resolution=250, calibration='reflectance'),
        make_dataid(name='ds3'),
        make_dataid(name='ds4', calibration='reflectance'),
        make_dataid(name='ds4', calibration='radiance'),
        make_dataid(name='ds5', resolution=250),
        make_dataid(name='ds5', resolution=500),
        make_dataid(name='ds5', resolution=1000),
        make_dataid(name='ds6', wavelength=(0.1, 0.2, 0.3), resolution=250, calibration='reflectance'),
        make_dataid(name='ds7', wavelength=(0.4, 0.5, 0.6)),
        make_dataid(name='ds8', wavelength=(0.7, 0.8, 0.9)),
        make_dataid(name='ds9_fail_load', wavelength=(1.0, 1.1, 1.2)),
        make_dataid(name='ds10', wavelength=(0.75, 0.85, 0.95)),
        make_dataid(name='ds11', resolution=500),
        make_dataid(name='ds11', resolution=1000),
        make_dataid(name='ds12', resolution=500),
        make_dataid(name='ds12', resolution=1000),
        make_dataid(name='B02', wavelength=(30.49, 30.51, 30.53), resolution=1000),
        make_dataid(name='B03', wavelength=(30.62, 30.64, 30.66), resolution=500),
        make_dataid(name='B04', wavelength=(30.83, 30.85, 30.87), resolution=1000),
    ]
    return d


def _create_fake_compositor(ds_id, prereqs, opt_prereqs):
    import numpy as np
    from xarray import DataArray
    c = mock.MagicMock()
    c.attrs = {
        'prerequisites': tuple(prereqs),
        'optional_prerequisites': tuple(opt_prereqs),
    }
    # special case
    c.attrs.update(ds_id.to_dict())
    c.id = ds_id

    se = mock.MagicMock()

    def _se(datasets, optional_datasets=None, ds_id=ds_id, **kwargs):
        if ds_id['name'] in ('comp14', 'comp26'):
            # used as a test when composites update the dataset id with
            # information from prereqs
            ds_id = DataID(ds_id.id_keys, resolution=555, **ds_id)
        if len(datasets) != len(prereqs):
            raise ValueError("Not enough prerequisite datasets passed")
        return DataArray(data=np.arange(75).reshape(5, 5, 3),
                         attrs=ds_id.to_dict(),
                         dims=['y', 'x', 'bands'],
                         coords={'bands': ['R', 'G', 'B']})
    se.side_effect = _se
    c.side_effect = se
    return c


def _create_fake_modifiers(name, prereqs, opt_prereqs):
    import numpy as np
    from xarray import DataArray
    from satpy.composites import CompositeBase, IncompatibleAreas

    attrs = {
        'name': name,
        'prerequisites': tuple(prereqs),
        'optional_prerequisites': tuple(opt_prereqs)
    }

    def _mod_loader(*args, **kwargs):
        class FakeMod(CompositeBase):
            def __init__(self, *args, **kwargs):

                super(FakeMod, self).__init__(*args, **kwargs)

            def __call__(self, datasets, optional_datasets, **info):
                if self.attrs['optional_prerequisites']:
                    for opt_dep in self.attrs['optional_prerequisites']:
                        if 'NOPE' in opt_dep or 'fail' in opt_dep:
                            continue
                        assert optional_datasets is not None and \
                            len(optional_datasets)
                resolution = datasets[0].attrs.get('resolution')
                if name == 'res_change' and resolution is not None:
                    i = datasets[0].attrs.copy()
                    i['resolution'] *= 5
                elif 'incomp_areas' in name:
                    raise IncompatibleAreas(
                        "Test modifier 'incomp_areas' always raises IncompatibleAreas")
                else:
                    i = datasets[0].attrs
                info = datasets[0].attrs.copy()
                self.apply_modifier_info(i, info)
                return DataArray(np.ma.MaskedArray(datasets[0]), attrs=info)

        m = FakeMod(*args, **kwargs)
        # m.attrs = attrs
        m._call_mock = mock.patch.object(
            FakeMod, '__call__', wraps=m.__call__).start()
        return m

    return _mod_loader, attrs


def test_composites(sensor_name):
    """Create some test composites."""
    from satpy import DatasetDict
    # Composite ID -> (prereqs, optional_prereqs)
    comps = {
        make_cid(name='comp1'): (['ds1'], []),
        make_cid(name='comp2'): (['ds1', 'ds2'], []),
        make_cid(name='comp3'): (['ds1', 'ds2', 'ds3'], []),
        make_cid(name='comp4'): (['comp2', 'ds3'], []),
        make_cid(name='comp5'): (['ds1', 'ds2'], ['ds3']),
        make_cid(name='comp6'): (['ds1', 'ds2'], ['comp2']),
        make_cid(name='comp7'): (['ds1', 'comp2'], ['ds2']),
        make_cid(name='comp8'): (['ds_NOPE', 'comp2'], []),
        make_cid(name='comp9'): (['ds1', 'comp2'], ['ds_NOPE']),
        make_cid(name='comp10'): ([make_dsq(name='ds1', modifiers=('mod1',)), 'comp2'], []),
        make_cid(name='comp11'): ([0.22, 0.48, 0.85], []),
        make_cid(name='comp12'): ([make_dsq(wavelength=0.22, modifiers=('mod1',)),
                                   make_dsq(wavelength=0.48, modifiers=('mod1',)),
                                   make_dsq(wavelength=0.85, modifiers=('mod1',))], []),
        make_cid(name='comp13'): ([make_dsq(name='ds5', modifiers=('res_change',))], []),
        make_cid(name='comp14'): (['ds1'], []),
        make_cid(name='comp15'): (['ds1', 'ds9_fail_load'], []),
        make_cid(name='comp16'): (['ds1'], ['ds9_fail_load']),
        make_cid(name='comp17'): (['ds1', 'comp15'], []),
        make_cid(name='comp18'): (['ds3',
                                   make_dsq(name='ds4', modifiers=('mod1', 'mod3',)),
                                   make_dsq(name='ds5', modifiers=('mod1', 'incomp_areas'))], []),
        make_cid(name='comp18_2'): (['ds3',
                                     make_dsq(name='ds4', modifiers=('mod1', 'mod3',)),
                                     make_dsq(name='ds5', modifiers=('mod1', 'incomp_areas_opt'))], []),
        make_cid(name='comp19'): ([make_dsq(name='ds5', modifiers=('res_change',)), 'comp13', 'ds2'], []),
        make_cid(name='comp20'): ([make_dsq(name='ds5', modifiers=('mod_opt_prereq',))], []),
        make_cid(name='comp21'): ([make_dsq(name='ds5', modifiers=('mod_bad_opt',))], []),
        make_cid(name='comp22'): ([make_dsq(name='ds5', modifiers=('mod_opt_only',))], []),
        make_cid(name='comp23'): ([0.8], []),
        make_cid(name='static_image'): ([], []),
        make_cid(name='comp24', resolution=500): ([make_dsq(name='ds11', resolution=500),
                                                   make_dsq(name='ds12', resolution=500)], []),
        make_cid(name='comp24', resolution=1000): ([make_dsq(name='ds11', resolution=1000),
                                                    make_dsq(name='ds12', resolution=1000)], []),
        make_cid(name='comp25', resolution=500): ([make_dsq(name='comp24', resolution=500),
                                                   make_dsq(name='ds5', resolution=500)], []),
        make_cid(name='comp25', resolution=1000): ([make_dsq(name='comp24', resolution=1000),
                                                    make_dsq(name='ds5', resolution=1000)], []),
        make_cid(name='ahi_green'): ([make_dsq(wavelength=30.5, modifiers=('sunz_corr', 'rayleigh_corr')),
                                      make_dsq(wavelength=30.85, modifiers=('sunz_corr',))], []),
        make_cid(name='comp26'): (['comp14'], []),
    }
    # Modifier name -> (prereqs (not including to-be-modified), opt_prereqs)
    mods = {
        'mod1': (['ds2'], []),
        'mod2': (['comp3'], []),
        'mod3': (['ds2'], []),
        'res_change': ([], []),
        'incomp_areas': (['ds1'], []),
        'incomp_areas_opt': ([make_dataid(name='ds1', modifiers=('incomp_areas',))], ['ds2']),
        'mod_opt_prereq': (['ds1'], ['ds2']),
        'mod_bad_opt': (['ds1'], ['ds9_fail_load']),
        'mod_opt_only': ([], ['ds2']),
        'mod_wl': ([make_dsq(wavelength=0.2, modifiers=('mod1',))], []),
        'sunz_corr': ([], ['sunz_angles_NOPE']),
        'rayleigh_corr': ([make_dsq(wavelength=30.64, modifiers=('sunz_corr',))], [])
    }
    comps = {sensor_name: DatasetDict((k, _create_fake_compositor(k, *v)) for k, v in comps.items())}
    mods = {sensor_name: dict((k, _create_fake_modifiers(k, *v)) for k, v in mods.items())}

    return comps, mods


def _filter_datasets(all_ds, names_or_ids):
    """Help filtering DataIDs by name or DataQuery."""
    # DataID will match a str to the name
    # need to separate them out
    str_filter = [ds_name for ds_name in names_or_ids if isinstance(ds_name, str)]
    id_filter = [ds_id for ds_id in names_or_ids if not isinstance(ds_id, str)]
    for ds_id in all_ds:
        if ds_id in id_filter or ds_id['name'] in str_filter:
            yield ds_id


class FakeReader(FileYAMLReader):
    """Fake reader to make testing basic Scene/reader functionality easier."""

    def __init__(self, name, sensor_name='fake_sensor', datasets=None,
                 available_datasets=None, start_time=None, end_time=None,
                 filter_datasets=True):
        """Initialize reader and mock necessary properties and methods.

        By default any 'datasets' provided will be filtered by what datasets
        are configured at the top of this module in 'test_datasets'. This can
        be disabled by specifying `filter_datasets=False`.

        """
        reader_config = {'reader': {'name': name, 'config_files': ['fake.yaml']}, 'file_types': {}}
        super(FakeReader, self).__init__(reader_config)

        if start_time is None:
            start_time = datetime.utcnow()
        self._start_time = start_time
        if end_time is None:
            end_time = start_time
        self._end_time = end_time
        self._sensor_name = set([sensor_name])

        all_ds = test_datasets()
        if datasets is not None and filter_datasets:
            all_ds = list(_filter_datasets(all_ds, datasets))
        elif datasets:
            all_ds = datasets
        if available_datasets is not None:
            available_datasets = list(_filter_datasets(all_ds, available_datasets))
        else:
            available_datasets = all_ds

        self.all_ids = {ds_id: {} for ds_id in all_ds}
        self.available_ids = {ds_id: {} for ds_id in available_datasets}

        # Wrap load method in mock object so we can record call information
        self.load = mock.patch.object(self, 'load', wraps=self.load).start()

    @property
    def start_time(self):
        """Get the start time."""
        return self._start_time

    @property
    def end_time(self):
        """Get the end time."""
        return self._end_time

    @property
    def sensor_names(self):
        """Get the sensor names."""
        return self._sensor_name

    def load(self, dataset_keys):
        """Load some data."""
        from satpy import DatasetDict
        from xarray import DataArray
        import numpy as np
        dataset_ids = self.all_ids.keys()
        loaded_datasets = DatasetDict()
        for k in dataset_keys:
            if k['name'] == 'ds9_fail_load':
                continue
            for ds in dataset_ids:
                if ds == k:
                    loaded_datasets[ds] = DataArray(data=np.arange(25).reshape(5, 5),
                                                    attrs=ds.to_dict(),
                                                    dims=['y', 'x'])
        return loaded_datasets


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
            ``numpy.ndarray`` to ``xarray.DataArray`.  No effect when the
            values in ``content_dict`` are already ``xarray.DataArray`.
        area (bool or BaseDefinition): Can be ``True``, ``False``, or an
            instance of ``pyresample.geometry.BaseDefinition`` such as
            ``AreaDefinition`` or ``SwathDefinition``.  If ``True``, which is
            the default, automatically generate areas.  If ``False``, values
            will not have assigned areas.  If an instance of
            ``pyresample.geometry.BaseDefinition``, those instances will be
            used for all generated fake datasets.  Warning: Passing an area as
            a string (``area="germ"``) is not supported.
        common_attrs (Mapping): optional, additional attributes that will
            be added to every dataset in the scene.

    Returns:
        Scene object with datasets corresponding to content_dict.
    """
    import pyresample
    import satpy
    import xarray as xr
    if common_attrs is None:
        common_attrs = {}
    if daskify:
        import dask.array
    sc = satpy.Scene()
    for (did, arr) in content_dict.items():
        extra_attrs = common_attrs.copy()
        if isinstance(area, pyresample.geometry.BaseDefinition):
            extra_attrs["area"] = area
        elif area:
            extra_attrs["area"] = pyresample.create_area_def(
                    "test-area",
                    {"proj": "eqc", "lat_ts": 0, "lat_0": 0, "lon_0": 0,
                     "x_0": 0, "y_0": 0, "ellps": "sphere", "units": "m",
                     "no_defs": None, "type": "crs"},
                    units="m",
                    shape=arr.shape,
                    resolution=1000,
                    center=(0, 0))
        if isinstance(arr, xr.DataArray):
            sc[did] = arr.copy()  # don't change attributes of input
            sc[did].attrs.update(extra_attrs)
        else:
            if daskify:
                arr = dask.array.from_array(arr)
            sc[did] = xr.DataArray(
                    arr,
                    dims=("y", "x"),
                    attrs=extra_attrs)
    return sc
