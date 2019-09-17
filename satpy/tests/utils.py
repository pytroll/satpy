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
from satpy.readers.yaml_reader import FileYAMLReader

try:
    from unittest import mock
except ImportError:
    import mock


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
    from satpy import DatasetID
    d = [
        DatasetID(name='ds1'),
        DatasetID(name='ds2'),
        DatasetID(name='ds3'),
        DatasetID(name='ds4', calibration='reflectance'),
        DatasetID(name='ds4', calibration='radiance'),
        DatasetID(name='ds5', resolution=250),
        DatasetID(name='ds5', resolution=500),
        DatasetID(name='ds5', resolution=1000),
        DatasetID(name='ds6', wavelength=(0.1, 0.2, 0.3)),
        DatasetID(name='ds7', wavelength=(0.4, 0.5, 0.6)),
        DatasetID(name='ds8', wavelength=(0.7, 0.8, 0.9)),
        DatasetID(name='ds9_fail_load', wavelength=(1.0, 1.1, 1.2)),
        DatasetID(name='ds10', wavelength=(0.75, 0.85, 0.95)),
        DatasetID(name='ds11', resolution=500),
        DatasetID(name='ds11', resolution=1000),
        DatasetID(name='ds12', resolution=500),
        DatasetID(name='ds12', resolution=1000),
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
        if ds_id.name == 'comp14':
            # used as a test when composites update the dataset id with
            # information from prereqs
            ds_id = ds_id._replace(resolution=555)
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
    from satpy import DatasetID

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
                resolution = DatasetID.from_dict(datasets[0].attrs).resolution
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
    from satpy import DatasetID, DatasetDict
    # Composite ID -> (prereqs, optional_prereqs)
    comps = {
        DatasetID(name='comp1'): (['ds1'], []),
        DatasetID(name='comp2'): (['ds1', 'ds2'], []),
        DatasetID(name='comp3'): (['ds1', 'ds2', 'ds3'], []),
        DatasetID(name='comp4'): (['comp2', 'ds3'], []),
        DatasetID(name='comp5'): (['ds1', 'ds2'], ['ds3']),
        DatasetID(name='comp6'): (['ds1', 'ds2'], ['comp2']),
        DatasetID(name='comp7'): (['ds1', 'comp2'], ['ds2']),
        DatasetID(name='comp8'): (['ds_NOPE', 'comp2'], []),
        DatasetID(name='comp9'): (['ds1', 'comp2'], ['ds_NOPE']),
        DatasetID(name='comp10'): ([DatasetID('ds1', modifiers=('mod1',)), 'comp2'], []),
        DatasetID(name='comp11'): ([0.22, 0.48, 0.85], []),
        DatasetID(name='comp12'): ([DatasetID(wavelength=0.22, modifiers=('mod1',)),
                                    DatasetID(wavelength=0.48, modifiers=('mod1',)),
                                    DatasetID(wavelength=0.85, modifiers=('mod1',))], []),
        DatasetID(name='comp13'): ([DatasetID(name='ds5', modifiers=('res_change',))], []),
        DatasetID(name='comp14'): (['ds1'], []),
        DatasetID(name='comp15'): (['ds1', 'ds9_fail_load'], []),
        DatasetID(name='comp16'): (['ds1'], ['ds9_fail_load']),
        DatasetID(name='comp17'): (['ds1', 'comp15'], []),
        DatasetID(name='comp18'): (['ds3',
                                    DatasetID(name='ds4', modifiers=('mod1', 'mod3',)),
                                    DatasetID(name='ds5', modifiers=('mod1', 'incomp_areas'))], []),
        DatasetID(name='comp18_2'): (['ds3',
                                      DatasetID(name='ds4', modifiers=('mod1', 'mod3',)),
                                      DatasetID(name='ds5', modifiers=('mod1', 'incomp_areas_opt'))], []),
        DatasetID(name='comp19'): ([DatasetID('ds5', modifiers=('res_change',)), 'comp13', 'ds2'], []),
        DatasetID(name='comp20'): ([DatasetID(name='ds5', modifiers=('mod_opt_prereq',))], []),
        DatasetID(name='comp21'): ([DatasetID(name='ds5', modifiers=('mod_bad_opt',))], []),
        DatasetID(name='comp22'): ([DatasetID(name='ds5', modifiers=('mod_opt_only',))], []),
        DatasetID(name='comp23'): ([0.8], []),
        DatasetID(name='static_image'): ([], []),
        DatasetID(name='comp24', resolution=500): ([DatasetID(name='ds11', resolution=500),
                                                    DatasetID(name='ds12', resolution=500)], []),
        DatasetID(name='comp24', resolution=1000): ([DatasetID(name='ds11', resolution=1000),
                                                     DatasetID(name='ds12', resolution=1000)], []),
        DatasetID(name='comp25', resolution=500): ([DatasetID(name='comp24', resolution=500),
                                                    DatasetID(name='ds5', resolution=500)], []),
        DatasetID(name='comp25', resolution=1000): ([DatasetID(name='comp24', resolution=1000),
                                                     DatasetID(name='ds5', resolution=1000)], []),
    }
    # Modifier name -> (prereqs (not including to-be-modified), opt_prereqs)
    mods = {
        'mod1': (['ds2'], []),
        'mod2': (['comp3'], []),
        'mod3': (['ds2'], []),
        'res_change': ([], []),
        'incomp_areas': (['ds1'], []),
        'incomp_areas_opt': ([DatasetID(name='ds1', modifiers=('incomp_areas',))], ['ds2']),
        'mod_opt_prereq': (['ds1'], ['ds2']),
        'mod_bad_opt': (['ds1'], ['ds9_fail_load']),
        'mod_opt_only': ([], ['ds2']),
        'mod_wl': ([DatasetID(wavelength=0.2, modifiers=('mod1',))], []),
    }

    comps = {sensor_name: DatasetDict((k, _create_fake_compositor(k, *v)) for k, v in comps.items())}
    mods = {sensor_name: dict((k, _create_fake_modifiers(k, *v)) for k, v in mods.items())}

    return comps, mods


def _filter_datasets(all_ds, names_or_ids):
    """Help filtering DatasetIDs by name or DatasetID."""
    # DatasetID will match a str to the name
    # need to separate them out
    str_filter = [ds_name for ds_name in names_or_ids if isinstance(ds_name, str)]
    id_filter = [ds_id for ds_id in names_or_ids if not isinstance(ds_id, str)]
    for ds_id in all_ds:
        if ds_id in id_filter or ds_id.name in str_filter:
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
        with mock.patch('satpy.readers.yaml_reader.recursive_dict_update') as rdu, \
                mock.patch('satpy.readers.yaml_reader.open'), \
                mock.patch('satpy.readers.yaml_reader.yaml.load'):
            rdu.return_value = {'reader': {'name': name}, 'file_types': {}}
            super(FakeReader, self).__init__(['fake.yaml'])

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
            if k == 'ds9_fail_load':
                continue
            for ds in dataset_ids:
                if ds == k:
                    loaded_datasets[ds] = DataArray(data=np.arange(25).reshape(5, 5),
                                                    attrs=ds.to_dict(),
                                                    dims=['y', 'x'])
        return loaded_datasets
