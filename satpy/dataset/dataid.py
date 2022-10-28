#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2021 Satpy developers
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
from contextlib import suppress
from copy import copy, deepcopy
from enum import Enum, IntEnum
from typing import NoReturn

import numpy as np

logger = logging.getLogger(__name__)


def get_keys_from_config(common_id_keys, config):
    """Gather keys for a new DataID from the ones available in configured dataset."""
    id_keys = {}
    for key, val in common_id_keys.items():
        if key in config:
            id_keys[key] = val
        elif val is not None and (val.get('required') is True or val.get('default') is not None):
            id_keys[key] = val
    if not id_keys:
        raise ValueError('Metadata does not contain enough information to create a DataID.')
    return id_keys


class ValueList(IntEnum):
    """A static value list.

    This class is meant to be used for dynamically created Enums. Due to this
    it should not be used as a normal Enum class or there may be some
    unexpected behavior. For example, this class contains custom pickling and
    unpickling handling that may break in subclasses.

    """

    @classmethod
    def convert(cls, value):
        """Convert value to an instance of this class."""
        try:
            return cls[value]
        except KeyError:
            raise ValueError('{} invalid value for {}'.format(value, cls))

    @classmethod
    def _unpickle(cls, enum_name, enum_members, enum_member):
        """Create dynamic class that was previously pickled.

        See :meth:`__reduce_ex__` for implementation details.

        """
        enum_cls = cls(enum_name, enum_members)
        return enum_cls[enum_member]

    def __reduce_ex__(self, proto):
        """Reduce the object for pickling."""
        return (ValueList._unpickle,
                (self.__class__.__name__, list(self.__class__.__members__.keys()), self.name))

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


wlklass = namedtuple("WavelengthRange", "min central max unit", defaults=('µm',))  # type: ignore


class WavelengthRange(wlklass):
    """A named tuple for wavelength ranges.

    The elements of the range are min, central and max values, and optionally a unit
    (defaults to µm). No clever unit conversion is done here, it's just used for checking
    that two ranges are comparable.
    """

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
        if isinstance(other, numbers.Number):
            return other in self
        if isinstance(other, (tuple, list)) and len(other) == 3:
            return self[:3] == other
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
        return "{0.central} {0.unit} ({0.min}-{0.max} {0.unit})".format(self)

    def __contains__(self, other):
        """Check if this range contains *other*."""
        if other is None:
            return False
        if isinstance(other, numbers.Number):
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
                if isinstance(value, (tuple, list)):
                    return abs(value[1] - self.central)
                return abs(value - self.central)
        else:
            return np.inf

    @classmethod
    def convert(cls, wl):
        """Convert `wl` to this type if possible."""
        if isinstance(wl, (tuple, list)):
            return cls(*wl)
        return wl

    def to_cf(self):
        """Serialize for cf export."""
        return str(self)

    @classmethod
    def from_cf(cls, blob):
        """Return a WavelengthRange from a cf blob."""
        try:
            obj = cls._read_cf_from_string_export(blob)
        except TypeError:
            obj = cls._read_cf_from_string_list(blob)
        return obj

    @classmethod
    def _read_cf_from_string_export(cls, blob):
        """Read blob as a string created by `to_cf`."""
        pattern = "{central:f} {unit:s} ({min:f}-{max:f} {unit2:s})"
        from trollsift import Parser
        parser = Parser(pattern)
        res_dict = parser.parse(blob)
        res_dict.pop('unit2')
        obj = cls(**res_dict)
        return obj

    @classmethod
    def _read_cf_from_string_list(cls, blob):
        """Read blob as a list of strings (legacy formatting)."""
        min_wl, central_wl, max_wl, unit = blob
        obj = cls(float(min_wl), float(central_wl), float(max_wl), unit)
        return obj


class ModifierTuple(tuple):
    """A tuple holder for modifiers."""

    @classmethod
    def convert(cls, modifiers):
        """Convert `modifiers` to this type if possible."""
        if modifiers is None:
            return None
        if not isinstance(modifiers, (cls, tuple, list)):
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


#: Default ID keys DataArrays.
default_id_keys_config = {'name': {
                              'required': True,
                          },
                          'wavelength': {
                              'type': WavelengthRange,
                          },
                          'resolution': {
                              'transitive': False,
                              },
                          'calibration': {
                              'enum': [
                                  'reflectance',
                                  'brightness_temperature',
                                  'radiance',
                                  'counts'
                                  ],
                              'transitive': True,
                          },
                          'modifiers': {
                              'default': ModifierTuple(),
                              'type': ModifierTuple,
                          },
                          }

#: Default ID keys for coordinate DataArrays.
default_co_keys_config = {'name': {
                              'required': True,
                          },
                          'resolution': {
                              'transitive': True,
                          }
                          }

#: Minimal ID keys for DataArrays, for example composites.
minimal_default_keys_config = {'name': {
                                  'required': True,
                              },
                               'resolution': {
                                   'transitive': True,
                               }
                              }


class DataID(dict):
    """Identifier for all `DataArray` objects.

    DataID is a dict that holds identifying and classifying
    information about a DataArray.
    """

    def __init__(self, id_keys, **keyval_dict):
        """Init the DataID.

        The *id_keys* dictionary has to be formed as described in :doc:`../dev_guide/satpy_internals`.
        The other keyword arguments are values to be assigned to the keys. Note that
        `None` isn't a valid value and will simply be ignored.
        """
        self._hash = None
        self._orig_id_keys = id_keys
        self._id_keys = self.fix_id_keys(id_keys or {})
        if keyval_dict:
            curated = self.convert_dict(keyval_dict)
        else:
            curated = {}
        super(DataID, self).__init__(curated)

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
            if val is None:
                val = {}
            if key in keyvals or val.get('default') is not None or val.get('required'):
                curated_val = keyvals.get(key, val.get('default'))
                if 'required' in val and curated_val is None:
                    raise ValueError('Required field {} missing.'.format(key))
                if 'type' in val:
                    curated[key] = val['type'].convert(curated_val)
                elif curated_val is not None:
                    curated[key] = curated_val

        return curated

    @classmethod
    def _unpickle(cls, id_keys, keyval):
        """Create a new instance of the DataID after pickling."""
        return cls(id_keys, **keyval)

    def __reduce__(self):
        """Reduce the object for pickling."""
        return (self._unpickle, (self._orig_id_keys, self.to_dict()))

    def from_dict(self, keyvals):
        """Create a DataID from a dictionary."""
        return self.__class__(self._id_keys, **keyvals)

    @classmethod
    def from_dataarray(cls, array, default_keys=minimal_default_keys_config):
        """Get the DataID using the dataarray attributes."""
        if '_satpy_id' in array.attrs:
            return array.attrs['_satpy_id']
        return cls.new_id_from_dataarray(array, default_keys)

    @classmethod
    def new_id_from_dataarray(cls, array, default_keys=minimal_default_keys_config):
        """Create a new DataID from a dataarray's attributes."""
        try:
            id_keys = array.attrs['_satpy_id'].id_keys
        except KeyError:
            id_keys = array.attrs.get('_satpy_id_keys', default_keys)
        return cls(id_keys, **array.attrs)

    @property
    def id_keys(self):
        """Get the id_keys."""
        return deepcopy(self._id_keys)

    def create_filter_query_without_required_fields(self, query):
        """Remove the required fields from *query*."""
        try:
            new_query = query.to_dict()
        except AttributeError:
            new_query = query.copy()
        for key, val in self._id_keys.items():
            if val and (val.get('transitive') is not True):
                new_query.pop(key, None)
        return DataQuery.from_dict(new_query)

    def _asdict(self):
        return dict(self.items())

    def to_dict(self):
        """Convert the ID to a dict."""
        res_dict = dict()
        for key, value in self._asdict().items():
            if isinstance(value, Enum):
                res_dict[key] = value.name
            else:
                res_dict[key] = value
        return res_dict

    def __getattr__(self, key):
        """Support old syntax for getting items."""
        if key in self._id_keys:
            warnings.warn('Attribute access to DataIDs is deprecated, use key access instead.',
                          stacklevel=2)
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

    def _immutable(self, *args, **kws) -> NoReturn:
        """Raise and error."""
        raise TypeError('Cannot change a DataID')

    def __lt__(self, other):
        """Check lesser than."""
        list_self, list_other = [], []
        for key in self._id_keys:
            if key not in self and key not in other:
                continue
            elif key in self and key in other:
                list_self.append(self[key])
                list_other.append(other[key])
            elif key in self:
                val = self[key]
                list_self.append(val)
                list_other.append(_generalize_value_for_comparison(val))
            elif key in other:
                val = other[key]
                list_other.append(val)
                list_self.append(_generalize_value_for_comparison(val))
        return tuple(list_self) < tuple(list_other)

    __setitem__ = _immutable
    __delitem__ = _immutable
    pop = _immutable  # type: ignore
    popitem = _immutable
    clear = _immutable
    update = _immutable  # type: ignore
    setdefault = _immutable  # type: ignore

    def _find_modifiers_key(self):
        for key, val in self.items():
            if isinstance(val, ModifierTuple):
                return key
        raise KeyError

    def create_less_modified_query(self):
        """Create a query with one less modifier."""
        new_dict = self.to_dict()
        new_dict['modifiers'] = tuple(new_dict['modifiers'][:-1])
        return DataQuery.from_dict(new_dict)

    def is_modified(self):
        """Check if this is modified."""
        try:
            key = self._find_modifiers_key()
        except KeyError:
            return False
        return bool(self[key])


def _generalize_value_for_comparison(val):
    """Get a generalize value for comparisons."""
    if isinstance(val, numbers.Number):
        return 0
    if isinstance(val, str):
        return ""
    if isinstance(val, tuple):
        return tuple()

    raise NotImplementedError("Don't know how to generalize " + str(type(val)))


class DataQuery:
    """The data query object.

    A DataQuery can be used in Satpy to query for a Dataset. This way
    a fully qualified DataID can be found even if some DataID
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

    def items(self):
        """Get the items of this query."""
        return self._dict.items()

    def _asdict(self):
        return self._dict.copy()

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
        keys = list(filter(self._match_dataid, dataid_container))

        return keys

    def _match_dataid(self, dataid):
        """Match the dataid with the current query."""
        if self._shares_required_keys(dataid):
            keys_to_check = set(dataid.keys()) & set(self._fields)
        else:
            keys_to_check = set(dataid._id_keys.keys()) & set(self._fields)
        if not keys_to_check:
            return False
        return all(self._match_query_value(key, dataid.get(key)) for key in keys_to_check)

    def _shares_required_keys(self, dataid):
        """Check if dataid shares required keys with the current query."""
        for key, val in dataid._id_keys.items():
            try:
                if val.get('required', False):
                    if key in self._fields:
                        return True
            except AttributeError:
                continue
        return False

    def _match_query_value(self, key, id_val):
        val = self._dict[key]
        if val == '*':
            return True
        if isinstance(id_val, tuple) and isinstance(val, (tuple, list)):
            return tuple(val) == id_val
        if not isinstance(val, list):
            val = [val]
        return id_val in val

    def sort_dataids_with_preference(self, all_ids, preference):
        """Sort `all_ids` given a sorting `preference` (DataQuery or None)."""
        try:
            res = preference.to_dict()
        except AttributeError:
            res = dict()
        res.update(self.to_dict())
        optimistic_query = DataQuery.from_dict(res)
        sorted_ids, distances = optimistic_query.sort_dataids(all_ids)
        if distances[0] == np.inf:  # nothing matches the optimistic query
            sorted_ids, distances = self.sort_dataids(all_ids)
        return sorted_ids, distances

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
           Calibration priority is the order of the calibration list defined as
           reflectance, brightness temperature, radiance counts if not overridden in the
           reader configuration.
        4. Best resolution (smallest number) if `resolution` is `None`
           in `key`. Otherwise, the resolution is ignored.

        """
        distances = []
        sorted_dataids = []
        big_distance = 100000
        keys = set(self._dict.keys())
        for dataid in dataids:
            keys |= set(dataid.keys())
        for dataid in sorted(dataids):
            sorted_dataids.append(dataid)
            distance = 0
            for key in keys:
                if distance == np.inf:
                    break
                val = self._dict.get(key, '*')
                if val == '*':
                    distance = self._add_absolute_distance(dataid, key, distance)
                else:
                    try:
                        dataid_val = dataid[key]
                    except KeyError:
                        distance += big_distance
                        continue
                    distance = self._add_distance_from_query(dataid_val, val, distance)
            distances.append(distance)
        distances, dataids = zip(*sorted(zip(distances, sorted_dataids)))
        return dataids, distances

    @staticmethod
    def _add_absolute_distance(dataid, key, distance):
        try:
            # for enums
            distance += dataid.get(key).value
        except AttributeError:
            if isinstance(dataid.get(key), numbers.Number):
                distance += dataid.get(key)
            elif isinstance(dataid.get(key), tuple):
                distance += len(dataid.get(key))
        return distance

    @staticmethod
    def _add_distance_from_query(dataid_val, requested_val, distance):
        try:
            distance += dataid_val.distance(requested_val)
        except AttributeError:
            if not isinstance(requested_val, list):
                requested_val = [requested_val]
            if dataid_val not in requested_val:
                distance = np.inf
            elif isinstance(dataid_val, numbers.Number):
                # so as to get the highest resolution first
                # FIXME: this ought to be clarified, not sure that
                # higher resolution is preferable is all cases.
                # Moreover this might break with other numerical
                # values.
                distance += dataid_val
        return distance

    def create_less_modified_query(self):
        """Create a query with one less modifier."""
        new_dict = self.to_dict()
        new_dict['modifiers'] = tuple(new_dict['modifiers'][:-1])
        return DataQuery.from_dict(new_dict)

    def is_modified(self):
        """Check if this is modified."""
        return bool(self._dict.get('modifiers'))


def create_filtered_query(dataset_key, filter_query):
    """Create a DataQuery matching *dataset_key* and *filter_query*.

    If a property is specified in both *dataset_key* and *filter_query*, the former
    has priority.

    """
    ds_dict = _create_id_dict_from_any_key(dataset_key)
    _update_dict_with_filter_query(ds_dict, filter_query)

    return DataQuery.from_dict(ds_dict)


def _update_dict_with_filter_query(ds_dict, filter_query):
    if filter_query is not None:
        for key, value in filter_query.items():
            if value != '*':
                ds_dict.setdefault(key, value)


def _create_id_dict_from_any_key(dataset_key):
    try:
        ds_dict = dataset_key.to_dict()
    except AttributeError:
        if isinstance(dataset_key, str):
            ds_dict = {'name': dataset_key}
        elif isinstance(dataset_key, numbers.Number):
            ds_dict = {'wavelength': dataset_key}
        else:
            raise TypeError("Don't know how to interpret a dataset_key of type {}".format(type(dataset_key)))
    return ds_dict
