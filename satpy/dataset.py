#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2019 Satpy developers
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
"""Dataset objects."""

import sys
import logging
import numbers
from collections import namedtuple
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


class MetadataObject(object):
    """A general metadata object."""

    def __init__(self, **attributes):
        """Initialize the class with *attributes*."""
        self.attrs = attributes

    @property
    def id(self):
        """Return the DatasetID of the object."""
        return DatasetID.from_dict(self.attrs)


def average_datetimes(dt_list):
    """Average a series of datetime objects.

    .. note::

        This function assumes all datetime objects are naive and in the same
        time zone (UTC).

    Args:
        dt_list (iterable): Datetime objects to average

    Returns: Average datetime as a datetime object

    """
    if sys.version_info < (3, 3):
        # timestamp added in python 3.3
        import time

        def timestamp_func(dt):
            return time.mktime(dt.timetuple())
    else:
        timestamp_func = datetime.timestamp

    total = [timestamp_func(dt) for dt in dt_list]
    return datetime.fromtimestamp(sum(total) / len(total))


def combine_metadata(*metadata_objects, **kwargs):
    """Combine the metadata of two or more Datasets.

    If any keys are not equal or do not exist in all provided dictionaries
    then they are not included in the returned dictionary.
    By default any keys with the word 'time' in them and consisting
    of datetime objects will be averaged. This is to handle cases where
    data were observed at almost the same time but not exactly.

    Args:
        *metadata_objects: MetadataObject or dict objects to combine
        average_times (bool): Average any keys with 'time' in the name

    Returns:
        dict: the combined metadata

    """
    average_times = kwargs.get('average_times', True)  # python 2 compatibility (no kwarg after *args)
    shared_keys = None
    info_dicts = []
    # grab all of the dictionary objects provided and make a set of the shared keys
    for metadata_object in metadata_objects:
        if isinstance(metadata_object, dict):
            metadata_dict = metadata_object
        elif hasattr(metadata_object, "attrs"):
            metadata_dict = metadata_object.attrs
        else:
            continue
        info_dicts.append(metadata_dict)

        if shared_keys is None:
            shared_keys = set(metadata_dict.keys())
        else:
            shared_keys &= set(metadata_dict.keys())

    # combine all of the dictionaries
    shared_info = {}
    for k in shared_keys:
        values = [nfo[k] for nfo in info_dicts]
        any_arrays = any([isinstance(val, np.ndarray) for val in values])
        if any_arrays:
            if all(np.all(val == values[0]) for val in values[1:]):
                shared_info[k] = values[0]
        elif 'time' in k and isinstance(values[0], datetime) and average_times:
            shared_info[k] = average_datetimes(values)
        elif all(val == values[0] for val in values[1:]):
            shared_info[k] = values[0]

    return shared_info


def get_keys_from_config(common_id_keys, config):
    """Gather keys for a new DatasetID then ones available in configured dataset."""
    id_keys = {}
    for key, val in common_id_keys.items():
        if key in config:
            id_keys[key] = val
        elif val is not None and val.get('compulsory') is True:
            id_keys[key] = val
    if not id_keys:
        raise ValueError('Metada does not contain enough information to create a DatasetID.')
    return id_keys


def new_dataset_id_from_keys(id_keys):
    """Create a new DatasetID from a configuration."""
    types = {}
    defaults = []
    for key, val in id_keys.items():
        if val is None:
            defaults.append(None)
        else:
            defaults.append(val.get('default'))
            if 'type' in val:
                types[key] = val['type']
    klass = make_dsid_class(types, **dict(zip(id_keys.keys(), defaults)))
    return klass


def wavelength_match(a, b):
    """Return if two wavelengths are equal.

    Args:
        a (tuple or scalar): (min wl, nominal wl, max wl) or scalar wl
        b (tuple or scalar): (min wl, nominal wl, max wl) or scalar wl
    """
    if ((type(a) == type(b)) or
        (isinstance(a, numbers.Number) and
            isinstance(b, numbers.Number))):
        return a == b
    elif a is None or b is None:
        return False
    elif isinstance(a, (list, tuple)) and len(a) == 3:
        return a[0] <= b <= a[2]
    elif isinstance(b, (list, tuple)) and len(b) == 3:
        return b[0] <= a <= b[2]
    else:
        raise ValueError("Can only compare wavelengths of length 1 or 3")


wlklass = namedtuple("WavelengthRange", "min central max")


class WavelengthRange(wlklass):
    """A named tuple for wavelength ranges."""

    def __eq__(self, other):
        """Return if two wavelengths are equal.

        Args:
            other (tuple or scalar): (min wl, nominal wl, max wl) or scalar wl

        Return:
            True if other is a scalar and min <= other <= max, or if other is
            a tuple equal to self, False otherwise.
        """
        if other is None:
            return False
        elif isinstance(other, numbers.Number):
            return self.min <= other <= self.max
        else:
            return super().__eq__(other)

    def __ne__(self, other):
        """Return the opposite of `__eq__`."""
        return not self == other

    def __lt__(self, other):
        """Compare to another wavelength."""
        if other is None:
            return False
        return super().__lt__(other)

    def __gt__(self, other):
        """Compare to another wavelength."""
        if other is None:
            return True
        return super().__gt__(other)

    def __hash__(self):
        """Hash this tuple."""
        return tuple.__hash__(self)

    @classmethod
    def convert(cls, wl):
        """Convert `wl` to this type if possible."""
        if isinstance(wl, (tuple, list)):
            return cls(*wl)
        return wl


class ModifierTuple(tuple):
    """A tuple holder for modifiers."""

    @classmethod
    def convert(cls, modifiers):
        """Convert `modifiers` to this type if possible."""
        if modifiers is None:
            return cls()
        elif not isinstance(modifiers, (cls, tuple, list)):
            raise TypeError("'DatasetID' modifiers must be a tuple or None, "
                            "not {}".format(type(modifiers)))
        return cls(modifiers)


def make_dsid_class(types=None, **kwargs):
    """Make a new DatasetID class."""
    fields = kwargs.keys()
    defaults = kwargs.values()
    klass = namedtuple("DatasetID", " ".join(fields), defaults=defaults)
    if types is None:
        types = {}

    # TODO: put this documentation somewhere sphinx can find it.
    class DatasetID(klass):
        """Identifier for all `Dataset` objects.

        DatasetID is a namedtuple that holds identifying and classifying
        information about a Dataset. There are two identifying elements,
        ``name`` and ``wavelength``. These can be used to generically refer to a
        Dataset. The other elements of a DatasetID are meant to further
        distinguish a Dataset from the possible variations it may have. For
        example multiple Datasets may be called by one ``name`` but may exist
        in multiple resolutions or with different calibrations such as "radiance"
        and "reflectance". If an element is `None` then it is considered not
        applicable.

        A DatasetID can also be used in Satpy to query for a Dataset. This way
        a fully qualified DatasetID can be found even if some of the DatasetID
        elements are unknown. In this case a `None` signifies something that is
        unknown or not applicable to the requested Dataset.

        Args:
            name (str): String identifier for the Dataset
            wavelength (float, tuple): Single float wavelength when querying for
                                    a Dataset. Otherwise 3-element tuple of
                                    floats specifying the minimum, nominal,
                                    and maximum wavelength for a Dataset.
                                    `None` if not applicable.
            resolution (int, float): Per data pixel/area resolution. If resolution
                                    varies across the Dataset then nadir view
                                    resolution is preferred. Usually this is in
                                    meters, but for lon/lat gridded data angle
                                    degrees may be used.
            polarization (str): 'V' or 'H' polarizations of a microwave channel.
                                `None` if not applicable.
            calibration (str): String identifying the calibration level of the
                            Dataset (ex. 'radiance', 'reflectance', etc).
                            `None` if not applicable.
            level (int, float): Pressure/altitude level of the dataset. This is
                                typically in hPa, but may be in inverse meters
                                for altitude datasets (1/meters).
            modifiers (tuple): Tuple of strings identifying what corrections or
                            other modifications have been performed on this
                            Dataset (ex. 'sunz_corrected', 'rayleigh_corrected',
                            etc). `None` or empty tuple if not applicable.
        """

        def __new__(cls, *args, **kwargs):
            """Create new DatasetID."""
            for ckey, the_type in types.items():
                if ckey in kwargs:
                    # TODO: do we really need a convert method or should we fix __new__?
                    kwargs[ckey] = the_type.convert(kwargs[ckey])
            newargs = []
            for key, val in zip(cls._fields, args):
                if key in types:
                    val = types[key].convert(val)
                newargs.append(val)

            return super(DatasetID, cls).__new__(cls, *newargs, **kwargs)

        def __hash__(self):
            """Hash."""
            return hash((self._fields, tuple.__hash__(self)))

        def __eq__(self, other):
            """Compare the DatasetIDs."""
            sdict = self._asdict()
            odict = other._asdict()
            for key, val in sdict.items():
                if key in odict and odict[key] != val:
                    return False
            return True

        @classmethod
        def from_dict(cls, d, **kwargs):
            """Convert a dict to an ID."""
            newkwargs = dict()
            for k in cls._fields:
                if k in d:
                    newkwargs[k] = d[k]

            return cls(**newkwargs)

        def to_dict(self, trim=True):
            """Convert the ID to a dict."""
            if trim:
                return self._to_trimmed_dict()
            else:
                return self._asdict()

        def _to_trimmed_dict(self):
            return {key: getattr(self, key) for key in self._fields
                    if getattr(self, key) is not None}

    return DatasetID

    pass


# TODO: remove this static list
DATASET_KEYS = ("name", "wavelength", "resolution", "polarization",
                "calibration", "level", "modifiers")

"""
  identification_keys:
    name:
      compulsory: true
    wavelength:
      type: !!python/name:satpy.dataset.WavelengthRange
    resolution:
    view:
      default: nadir
    calibration:
    modifiers:
      compulsory: true
      default: []
      type: !!python/name:satpy.dataset.ModifierTuple
"""

default_id_keys_config = {'name': {
                              'compulsory': True
                          },
                          'wavelength': {
                              'type': WavelengthRange,
                          },
                          'resolution': None,
                          'polarization': None,
                          'calibration': None,
                          'level': None,
                          'modifiers': {
                              'compulsory': True,
                              'default': tuple(),
                              'type': ModifierTuple,
                          },
                          }

default_types = {'wavelength': WavelengthRange,
                 'modifiers': ModifierTuple}
default_id_keys = {'name': None, 'wavelength': None, 'resolution': None,
                   'polarization': None, 'calibration': None, 'level': None,
                   'modifiers': ModifierTuple()}
default_DatasetID = make_dsid_class(default_types, **default_id_keys)

DatasetID = default_DatasetID


def create_filtered_dsid(dataset_key, **dfilter):
    """Create a DatasetID matching *dataset_key* and *dfilter*.

    If a proprety is specified in both *dataset_key* and *dfilter*, the former
    has priority.

    """
    try:
        ds_dict = dataset_key.to_dict()
    except AttributeError:
        if isinstance(dataset_key, str):
            ds_dict = {'name': dataset_key}
        elif isinstance(dataset_key, numbers.Number):
            ds_dict = {'wavelength': dataset_key}
    for key, value in dfilter.items():
        if value is not None:
            ds_dict.setdefault(key, value)
    return DatasetID.from_dict(ds_dict)


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
    current_dsid = DatasetID.from_dict(dataset.attrs)
    for idx, ds in enumerate(parent_dataset.attrs['ancillary_variables']):
        if current_dsid == DatasetID.from_dict(ds.attrs):
            parent_dataset.attrs['ancillary_variables'][idx] = dataset
            return
