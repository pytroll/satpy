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
"""Dataset identifying objects."""

import logging
import numbers
import warnings
from collections import namedtuple
from collections.abc import Collection
from contextlib import suppress
from copy import copy, deepcopy
from datetime import datetime
from enum import IntEnum

import numpy as np

logger = logging.getLogger(__name__)


class ValueList(IntEnum):
    """A static value list."""

    @classmethod
    def convert(cls, value):
        """Convert value to an instance of this class."""
        try:
            return cls[value]
        except KeyError:
            raise ValueError('{} invalid value for {}'.format(value, cls))

    def __eq__(self, other):
        """Check equality."""
        return self.name == other

    def __ne__(self, other):
        """Check non-equality."""
        return self.name != other

    def __hash__(self):
        """Hash the object."""
        return hash(self.name)

    def __repr__(self):
        """Represent the values."""
        return '<' + str(self) + '>'


try:
    wlklass = namedtuple("WavelengthRange", "min central max unit", defaults=('µm',))
except NameError:  # python 3.6
    wlklass = namedtuple("WavelengthRange", "min central max unit")
    wlklass.__new__.__defaults__ = ('µm',)


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
            return other in self
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

    def __str__(self):
        """Format for print out."""
        return "{0.central}{0.unit} ({0.min}-{0.max}{0.unit})".format(self)

    def __contains__(self, other):
        """Check if this range contains *other*."""
        if other is None:
            return False
        elif isinstance(other, numbers.Number):
            return self.min <= other <= self.max
        with suppress(AttributeError):
            if self.unit != other.unit:
                raise NotImplementedError("Can't compare wavelength ranges with different units.")
            return self.min <= other.min and self.max >= other.max
        return False

    def distance(self, value):
        """Get the distance from value."""
        if self == value:
            try:
                return abs(value.central - self.central)
            except AttributeError:
                return abs(value - self.central)
        else:
            return np.inf

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
            return None
        elif not isinstance(modifiers, (cls, tuple, list)):
            raise TypeError("'DataID' modifiers must be a tuple or None, "
                            "not {}".format(type(modifiers)))
        return cls(modifiers)

    def __eq__(self, other):
        """Check equality."""
        if isinstance(other, list):
            other = tuple(other)
        return super().__eq__(other)

    def __ne__(self, other):
        """Check non-equality."""
        if isinstance(other, list):
            other = tuple(other)
        return super().__ne__(other)

    def __hash__(self):
        """Hash this tuple."""
        return tuple.__hash__(self)


default_id_keys_config = {'name': {
                              'required': True,
                          },
                          'wavelength': {
                              'type': WavelengthRange,
                          },
                          'resolution': None,
                          'calibration': {
                              'enum': [
                                  'reflectance',
                                  'brightness_temperature',
                                  'radiance',
                                  'counts'
                                  ]
                          },
                          'modifiers': {
                              'required': True,
                              'default': ModifierTuple(),
                              'type': ModifierTuple,
                          },
                          }


default_co_keys_config = {'name': {
                              'required': True,
                          },
                          'resolution': None,
                          }


minimal_default_keys_config = {'name': {
                                  'required': True,
                              },
                               'resolution': None
                              }


class MetadataObject(object):
    """A general metadata object."""

    def __init__(self, **attributes):
        """Initialize the class with *attributes*."""
        self.attrs = attributes

    @property
    def id(self):
        """Return the DataID of the object."""
        try:
            return self.attrs['_satpy_id']
        except KeyError:
            id_keys = self.attrs.get('_satpy_id_keys', minimal_default_keys_config)
            return DataID(id_keys, **self.attrs)


def average_datetimes(dt_list):
    """Average a series of datetime objects.

    .. note::

        This function assumes all datetime objects are naive and in the same
        time zone (UTC).

    Args:
        dt_list (iterable): Datetime objects to average

    Returns: Average datetime as a datetime object

    """
    total = [datetime.timestamp(dt) for dt in dt_list]
    return datetime.fromtimestamp(sum(total) / len(total))


def combine_metadata(*metadata_objects, **kwargs):
    """Combine the metadata of two or more Datasets.

    If the values corresponding to any keys are not equal or do not
    exist in all provided dictionaries then they are not included in
    the returned dictionary.  By default any keys with the word 'time'
    in them and consisting of datetime objects will be averaged. This
    is to handle cases where data were observed at almost the same time
    but not exactly.  In the interest of time, arrays are compared by
    object identity rather than by their contents.

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
        if _share_metadata_key(k, values, average_times):
            if 'time' in k and isinstance(values[0], datetime) and average_times:
                shared_info[k] = average_datetimes(values)
            else:
                shared_info[k] = values[0]

    return shared_info


def get_keys_from_config(common_id_keys, config):
    """Gather keys for a new DataID from the ones available in configured dataset."""
    id_keys = {}
    for key, val in common_id_keys.items():
        if key in config:
            id_keys[key] = val
        elif val is not None and val.get('required') is True:
            id_keys[key] = val
    if not id_keys:
        raise ValueError('Metadata does not contain enough information to create a DataID.')
    return id_keys


def _share_metadata_key(k, values, average_times):
    """Combine metadata. Helper for combine_metadata, decide if key is shared."""
    any_arrays = any([hasattr(val, "__array__") for val in values])
    # in the real world, the `ancillary_variables` attribute may be
    # List[xarray.DataArray], this means our values are now
    # List[List[xarray.DataArray]].
    # note that this list_of_arrays check is also true for any
    # higher-dimensional ndarray, but we only use this check after we have
    # checked any_arrays so this false positive should have no impact
    list_of_arrays = any(
            [isinstance(val, Collection) and len(val) > 0 and
             all([hasattr(subval, "__array__")
                 for subval in val])
             for val in values])
    if any_arrays:
        return _share_metadata_key_array(values)
    elif list_of_arrays:
        return _share_metadata_key_list_arrays(values)
    elif 'time' in k and isinstance(values[0], datetime) and average_times:
        return True
    elif all(val == values[0] for val in values[1:]):
        return True
    return False


def _share_metadata_key_array(values):
    """Combine metadata. Helper for combine_metadata, check object identity in list of arrays."""
    for val in values[1:]:
        if val is not values[0]:
            return False
    return True


def _share_metadata_key_list_arrays(values):
    """Combine metadata. Helper for combine_metadata, check object identity in list of list of arrays."""
    for val in values[1:]:
        for arr, ref in zip(val, values[0]):
            if arr is not ref:
                return False
    return True


class DataID(dict):
    """Identifier for all `DataArray` objects.

    FIXME: talk about None not being a valid value

    DataID is a dict that holds identifying and classifying
    information about a DataArray. There are two identifying elements,
    ``name`` and ``wavelength``. These can be used to generically refer to a
    Dataset. The other elements of a DataID are meant to further
    distinguish a Dataset from the possible variations it may have. For
    example multiple Datasets may be called by one ``name`` but may exist
    in multiple resolutions or with different calibrations such as "radiance"
    and "reflectance". If an element is `None` then it is considered not
    applicable.

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

    def __init__(self, id_keys, **keyval_dict):
        """Init the DataID."""
        self._hash = None
        self._id_keys = self.fix_id_keys(id_keys or {})
        if keyval_dict:
            curated = self.convert_dict(keyval_dict)
        else:
            curated = {}
        # if curated.get('name') == 'ds5' and curated.get('modifiers'):
        #     import ipdb; ipdb.set_trace()
        super(DataID, self).__init__(curated)

    @property
    def id_keys(self):
        """Get the id_keys."""
        return deepcopy(self._id_keys)

    @staticmethod
    def fix_id_keys(id_keys):
        """Flesh out enums in the id keys as gotten from a config."""
        new_id_keys = id_keys.copy()
        for key, val in id_keys.items():
            if not val:
                continue
            if 'enum' in val and 'type' in val:
                raise ValueError('Cannot have both type and enum for the same id key.')
            new_val = copy(val)
            if 'enum' in val:
                new_val['type'] = ValueList(key, ' '.join(new_val.pop('enum')))
            new_id_keys[key] = new_val
        return new_id_keys

    def convert_dict(self, keyvals):
        """Convert a dictionary's values to the types defined in this object's id_keys."""
        curated = {}
        if not keyvals:
            return curated
        for key, val in self._id_keys.items():
            if val is not None:
                if key in keyvals or val.get('default') or val.get('required'):
                    curated_val = keyvals.get(key, val.get('default'))
                    if 'required' in val and curated_val is None:
                        raise ValueError('Required field {} missing.'.format(key))
                    if 'type' in val:
                        curated[key] = val['type'].convert(curated_val)
                    elif curated_val is not None:
                        curated[key] = curated_val
            else:
                try:
                    curated_val = keyvals[key]
                except KeyError:
                    pass
                else:
                    if curated_val is not None:
                        curated[key] = curated_val
        return curated

    def from_dict(self, keyvals):
        """Create a DataID from a dictionary."""
        return self.__class__(self._id_keys, **keyvals)

    @classmethod
    def from_dataarray(cls, array, default_keys=minimal_default_keys_config):
        """Create a DataID from a dataarray."""
        try:
            id_keys = array.attrs['_satpy_id'].id_keys
        except KeyError:
            id_keys = array.attrs.get('_satpy_id_keys', default_keys)
        return cls(id_keys, **array.attrs)

    def _asdict(self):
        return dict(self.items())

    def to_dict(self):
        """Convert the ID to a dict."""
        return self._asdict()

    def __getattr__(self, key):
        """Support old syntax for getting items."""
        if key != '_id_keys' and key in self._id_keys:
            warnings.warn('Attribute access to DataIDs is deprecated, use key access instead.')
            return self[key]
        else:
            return super().__getattr__(key)

    def __deepcopy__(self, memo=None):
        """Copy this object.

        Returns self as it's immutable.
        """
        return self

    def __copy__(self):
        """Copy this object.

        Returns self as it's immutable.
        """
        return self

    def __repr__(self):
        """Represent the id."""
        items = ("{}={}".format(key, repr(val)) for key, val in self.items())
        return self.__class__.__name__ + "(" + ", ".join(items) + ")"

    def _replace(self, **kwargs):
        """Make a new instance with replaced items."""
        info = dict(self.items())
        info.update(kwargs)
        return self.from_dict(info)

    def __hash__(self):
        """Hash the object."""
        if self._hash is None:
            self._hash = hash(tuple(sorted(self.items())))
        return self._hash

    def _immutable(self, *args, **kws):
        """Raise and error."""
        raise TypeError('Cannot change a DataID')

    def __lt__(self, other):
        """Check lesser than."""
        return tuple(self.values()) < tuple(other.values())

    __setitem__ = _immutable
    __delitem__ = _immutable
    pop = _immutable
    popitem = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable


class DataQuery:
    """The data query object.

    A DataQuery can be used in Satpy to query for a Dataset. This way
    a fully qualified DataID can be found even if some of the DataID
    elements are unknown. In this case a `*` signifies something that is
    unknown or not applicable to the requested Dataset.
    """

    def __init__(self, **kwargs):
        """Initialize the query."""
        self._dict = kwargs.copy()
        self._fields = tuple(self._dict.keys())
        self._values = tuple(self._dict.values())

    def __getitem__(self, key):
        """Get an item."""
        return self._dict[key]

    def __eq__(self, other):
        """Compare the DataQuerys.

        A DataQuery is considered equal to another DataQuery or DataID
        if they have common keys that have equal values.
        """
        sdict = self._asdict()
        try:
            odict = other._asdict()
        except AttributeError:
            return False
        common_keys = False
        for key, val in sdict.items():
            if key in odict:
                common_keys = True
                if odict[key] != val and val is not None:
                    return False
        return common_keys

    def __hash__(self):
        """Hash."""
        fields = []
        values = []
        for field, value in sorted(self._dict.items()):
            if value != '*':
                fields.append(field)
                if isinstance(value, (list, set)):
                    value = tuple(value)
                values.append(value)
        return hash(tuple(zip(fields, values)))

    def get(self, key, default=None):
        """Get an item."""
        return self._dict.get(key, default)

    @classmethod
    def from_dict(cls, the_dict):
        """Convert a dict to an ID."""
        return cls(**the_dict)

    def _asdict(self):
        return dict(zip(self._fields, self._values))

    def to_dict(self, trim=True):
        """Convert the ID to a dict."""
        if trim:
            return self._to_trimmed_dict()
        else:
            return self._asdict()

    def _to_trimmed_dict(self):
        return {key: val for key, val in self._dict.items()
                if val != '*'}

    def __repr__(self):
        """Represent the query."""
        items = ("{}={}".format(key, repr(val)) for key, val in zip(self._fields, self._values))
        return self.__class__.__name__ + "(" + ", ".join(items) + ")"

    def filter_dataids(self, dataid_container):
        """Filter DataIDs based on this query."""
        keys = iter(dataid_container)
        for key, val in self._dict.items():
            if val != '*':
                keys = [k for k in keys
                        if k.get(key) == val]
        return keys

    def sort_dataids(self, dataids):
        """Sort the DataIDs based on this query.

        Returns the sorted dataids and the list of distances.

        The sorting is performed based on the types of the keys to search on
        (as they are defined in the DataIDs from `dataids`).
        If that type defines a `distance` method, then it is used to find how
        'far' the DataID is from the current query.
        If the type is a number, a simple subtraction is performed.
        For other types, the distance is 0 if the values are identical, np.inf
        otherwise.

        For example, with the default DataID, we use the following criteria:

        1. Central wavelength is nearest to the `key` wavelength if
           specified.
        2. Least modified dataset if `modifiers` is `None` in `key`.
           Otherwise, the modifiers are ignored.
        3. Highest calibration if `calibration` is `None` in `key`.
           Calibration priority is chosen by `satpy.CALIBRATION_ORDER`.
        4. Best resolution (smallest number) if `resolution` is `None`
           in `key`. Otherwise, the resolution is ignored.

        """
        distances = []
        sorted_dataids = []
        keys = set(self._dict.keys())
        for dataid in dataids:
            keys |= set(dataid.keys())
        for dataid in sorted(dataids):
            sorted_dataids.append(dataid)
            distance = 0
            for key in keys:
                val = self._dict.get(key, '*')
                if val == '*':
                    try:
                        # for enums
                        distance += dataid.get(key).value
                    except AttributeError:
                        if isinstance(dataid.get(key), numbers.Number):
                            distance += dataid.get(key)
                        elif isinstance(dataid.get(key), tuple):
                            distance += len(dataid.get(key))
                else:
                    try:
                        dataid_val = dataid[key]
                    except KeyError:
                        distance = np.inf
                        break
                    try:
                        distance += dataid_val.distance(val)
                    except AttributeError:
                        if dataid_val != val:
                            distance = np.inf
                            break
            distances.append(distance)
        distances, dataids = zip(*sorted(zip(distances, sorted_dataids)))
        return dataids, distances


class DatasetID:
    """Deprecated datasetid."""

    def __init__(self, *args, **kwargs):
        """Fake init."""
        raise TypeError("DatasetID should not be used directly")

    def from_dict(self, *args, **kwargs):
        """Fake fun."""
        raise TypeError("DatasetID should not be used directly")


def create_filtered_query(dataset_key, filter_query):
    """Create a DataQuery matching *dataset_key* and *filter_query*.

    If a property is specified in both *dataset_key* and *filter_query*, the former
    has priority.

    """
    try:
        ds_dict = dataset_key.to_dict()
    except AttributeError:
        if isinstance(dataset_key, str):
            ds_dict = {'name': dataset_key}
        elif isinstance(dataset_key, numbers.Number):
            ds_dict = {'wavelength': dataset_key}
        else:
            raise TypeError("Don't know how to interpret a dataset_key of type {}".format(type(dataset_key)))
    if filter_query is not None:
        for key, value in filter_query._dict.items():
            if value != '*':
                ds_dict.setdefault(key, value)

    return DataQuery.from_dict(ds_dict)


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
    id_keys = parent_dataset.attrs.get('_satpy_id_keys', dataset.attrs.get('_satpy_id_keys'))
    current_dataid = DataID(id_keys, **dataset.attrs)
    for idx, ds in enumerate(parent_dataset.attrs['ancillary_variables']):
        if current_dataid == DataID(id_keys, **ds.attrs):
            parent_dataset.attrs['ancillary_variables'][idx] = dataset
            return
