#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2020 Satpy developers
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
"""Utilities for dealing with ancillary variables."""

from .dataid import DataID, default_id_keys_config


def dataset_walker(datasets):
    """Walk through *datasets* and their ancillary data.

    Yields datasets and their parent.
    """
    for dataset in datasets:
        yield dataset, None
        for anc_ds in dataset.attrs.get('ancillary_variables', []):
            try:
                anc_ds.attrs
                yield anc_ds, dataset
            except AttributeError:
                continue


def replace_anc(dataset, parent_dataset):
    """Replace *dataset* the *parent_dataset*'s `ancillary_variables` field."""
    if parent_dataset is None:
        return
    id_keys = parent_dataset.attrs.get(
            '_satpy_id_keys',
            dataset.attrs.get(
                '_satpy_id_keys',
                default_id_keys_config))
    current_dataid = DataID(id_keys, **dataset.attrs)
    for idx, ds in enumerate(parent_dataset.attrs['ancillary_variables']):
        if current_dataid == DataID(id_keys, **ds.attrs):
            parent_dataset.attrs['ancillary_variables'][idx] = dataset
            return
