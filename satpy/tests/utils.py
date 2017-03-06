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

import mock
from datetime import datetime, timedelta


def test_datasets():
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
    ]
    return d


def _create_fake_compositor(ds_id, prereqs, opt_prereqs):
    import numpy as np
    from satpy import Dataset
    c = mock.MagicMock()
    c.info = {
        'prerequisites': tuple(prereqs),
        'optional_prerequisites': tuple(opt_prereqs),
    }
    # special case
    if ds_id.name == 'comp14':
        # used as a test when composites update the dataset id with
        # information from prereqs
        ds_id = ds_id._replace(resolution=555)
    c.info.update(ds_id.to_dict())
    c.id = ds_id

    def se(datasets, optional_datasets=None, **kwargs):
        if len(datasets) != len(prereqs):
            raise ValueError("Not enough prerequisite datasets passed")
        return Dataset(data=np.arange(5), **ds_id.to_dict())
    c.side_effect = se
    return c


def _create_fake_modifiers(name, prereqs, opt_prereqs):
    import numpy as np
    from satpy.dataset import Dataset
    from satpy.composites import CompositeBase, IncompatibleAreas

    def _mod_loader(*args, **kwargs):
        class FakeMod(CompositeBase):
            def __init__(self, *args, **kwargs):
                self.info = {}

            def __call__(self, datasets, optional_datasets, **info):
                if name == 'res_change' and datasets[0].id.resolution is not None:
                    i = datasets[0].info.copy()
                    i['resolution'] *= 5
                elif name == 'incomp_areas':
                    raise IncompatibleAreas("Test modifier 'incomp_areas' always raises IncompatibleAreas")
                else:
                    i = datasets[0].info
                info = datasets[0].info.copy()
                self.apply_modifier_info(i, info)
                return Dataset(data=np.ma.MaskedArray(datasets[0]), **info)

        m = FakeMod()
        m.info = {
            'prerequisites': tuple(prereqs),
            'optional_prerequisites': tuple(opt_prereqs)
        }
        return m

    return _mod_loader, {}


def test_composites(sensor_name):
    from satpy import DatasetID, DatasetDict
    # Composite ID -> (prereqs, optional_prereqs)
    # TODO: Composites with DatasetIDs as prereqs
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
                                    DatasetID(wavelength=0.85, modifiers=('mod1',))],
                                    []),
        DatasetID(name='comp13'): ([DatasetID(name='ds5', modifiers=('res_change',))], []),
        DatasetID(name='comp14'): (['ds1'], []),
        DatasetID(name='comp15'): (['ds1', 'ds9_fail_load'], []),
        DatasetID(name='comp16'): (['ds1'], ['ds9_fail_load']),
        DatasetID(name='comp17'): (['ds1', 'comp15'], []),
        DatasetID(name='comp18'): ([DatasetID(name='ds1', modifiers=('incomp_areas',))], []),
    }
    # Modifier name -> (prereqs (not including to-be-modified), opt_prereqs)
    mods = {
        'mod1': (['ds2'], []),
        'mod2': (['comp3'], []),
        'res_change': ([], []),
        'incomp_areas': (['ds1'], []),
    }

    comps = {sensor_name: DatasetDict((k, _create_fake_compositor(k, *v)) for k, v in comps.items())}
    mods = {sensor_name: dict((k, _create_fake_modifiers(k, *v)) for k, v in mods.items())}

    return comps, mods


def _get_dataset_key(self,
                     key,
                     calibration=None,
                     resolution=None,
                     polarization=None,
                     modifiers=None,
                     aslist=False):
    from satpy import DatasetID
    if isinstance(key, DatasetID) and not key.modifiers:
        try:
            return _get_dataset_key(self, key.name or key.wavelength)
        except KeyError:
            pass

    dataset_ids = self.datasets
    for ds in dataset_ids:
        # should do wavelength and string matching for equality
        if key == ds:
            return ds
    raise KeyError("No fake test key '{}'".format(key))


def _reader_load(self, dataset_keys):
    from satpy import DatasetDict, Dataset
    import numpy as np
    dataset_ids = self.datasets
    loaded_datasets = DatasetDict()
    for k in dataset_keys:
        if k == 'ds9_fail_load':
            continue
        for ds in dataset_ids:
            if ds == k:
                loaded_datasets[ds] = Dataset(data=np.arange(5),
                                              **ds.to_dict())
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
    r.load = partial(_reader_load, r)
    return r