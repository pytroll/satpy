#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017.

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   David Hoese <david.hoese@ssec.wisc.edu>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Utilities for various satpy tests.
"""

from datetime import datetime

try:
    from unittest import mock
except ImportError:
    import mock


def test_datasets():
    """Get list of various test datasets"""
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
                elif name == 'incomp_areas':
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
        DatasetID(name='comp19'): ([DatasetID('ds5', modifiers=('res_change',)), 'comp13', 'ds2'], []),
        DatasetID(name='comp20'): ([DatasetID(name='ds5', modifiers=('mod_opt_prereq',))], []),
        DatasetID(name='comp21'): ([DatasetID(name='ds5', modifiers=('mod_bad_opt',))], []),
        DatasetID(name='comp22'): ([DatasetID(name='ds5', modifiers=('mod_opt_only',))], []),
        DatasetID(name='comp23'): ([0.8], []),
    }
    # Modifier name -> (prereqs (not including to-be-modified), opt_prereqs)
    mods = {
        'mod1': (['ds2'], []),
        'mod2': (['comp3'], []),
        'mod3': (['ds2'], []),
        'res_change': ([], []),
        'incomp_areas': (['ds1'], []),
        'mod_opt_prereq': (['ds1'], ['ds2']),
        'mod_bad_opt': (['ds1'], ['ds9_fail_load']),
        'mod_opt_only': ([], ['ds2']),
        'mod_wl': ([DatasetID(wavelength=0.2, modifiers=('mod1',))], []),
    }

    comps = {sensor_name: DatasetDict((k, _create_fake_compositor(k, *v)) for k, v in comps.items())}
    mods = {sensor_name: dict((k, _create_fake_modifiers(k, *v)) for k, v in mods.items())}

    return comps, mods


def _get_dataset_key(self, key, **kwargs):
    from satpy.readers import get_key
    return get_key(key, self.datasets, **kwargs)


def _reader_load(self, dataset_keys):
    from satpy import DatasetDict
    from xarray import DataArray
    import numpy as np
    dataset_ids = self.datasets
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


def create_fake_reader(reader_name, sensor_name='fake_sensor', datasets=None,
                       start_time=None, end_time=None):
    from functools import partial
    if start_time is None:
        start_time = datetime.utcnow()
    if end_time is None:
        end_time = start_time
    r = mock.MagicMock()
    ds = test_datasets()
    if datasets is not None:
        ds = [d for d in ds if d.name in datasets]

    r.datasets = ds
    r.start_time = start_time
    r.end_time = end_time
    r.sensor_names = set([sensor_name])
    r.get_dataset_key = partial(_get_dataset_key, r)
    r.all_dataset_ids = r.datasets
    r.available_dataset_ids = r.datasets
    r.load.side_effect = partial(_reader_load, r)
    return r
