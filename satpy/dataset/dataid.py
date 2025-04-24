# Copyright (c) 2015-2023 Satpy developers
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
from __future__ import annotations

import logging
import numbers
from copy import copy, deepcopy
from enum import Enum
from functools import partial
from typing import Any, NoReturn

import numpy as np

from satpy.dataset.id_keys import ModifierTuple, ValueList, default_id_keys_config, minimal_default_keys_config

logger = logging.getLogger(__name__)


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
            if "enum" in val and "type" in val:
                raise ValueError("Cannot have both type and enum for the same id key.")
            new_val = copy(val)
            if "enum" in val:
                new_val["type"] = ValueList(key, " ".join(new_val.pop("enum")))
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
            if key in keyvals or val.get("default") is not None or val.get("required"):
                curated_val = keyvals.get(key, val.get("default"))
                if "required" in val and curated_val is None:
                    raise ValueError("Required field {} missing.".format(key))
                if "type" in val:
                    curated[key] = val["type"].convert(curated_val)
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
        if "_satpy_id" in array.attrs:
            return array.attrs["_satpy_id"]
        return cls.new_id_from_dataarray(array, default_keys)

    @classmethod
    def new_id_from_dataarray(cls, array, default_keys=minimal_default_keys_config):
        """Create a new DataID from a dataarray's attributes."""
        try:
            id_keys = array.attrs["_satpy_id"].id_keys
        except KeyError:
            id_keys = array.attrs.get("_satpy_id_keys", default_keys)
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
            if val and (val.get("transitive") is not True):
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
        raise TypeError("Cannot change a DataID")

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
        new_dict["modifiers"] = tuple(new_dict["modifiers"][:-1])
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

    A DataQuery can be used in Satpy to query a dict using ``DataID`` objects
    as keys. In a plain Python builtin ``dict`` object a fully matching
    ``DataQuery`` can be used to access the value of the matching ``DataID``.
    Using Satpy's special :class:``~satpy.dataid.data_dict.DatasetDict`` a
    ``DataQuery`` will match the closest matching ``DataID``. In this case a
    ``"*"`` in the query signifies something that is unknown or not applicable
    to the requested Dataset. See the ``DatasetDict`` class for more information
    including retrieving all items matching a ``DataQuery``.
    """

    def __init__(self, **kwargs):
        """Initialize the query."""
        self._dict = kwargs.copy()
        self._fields = tuple(self._dict.keys())
        self._values = tuple(self._dict.values())

    def __getitem__(self, key):
        """Get an item."""
        return self._dict[key]

    def __eq__(self, other: Any) -> bool:
        """Compare the DataQuerys.

        A DataQuery is considered equal to another DataQuery if all keys
        are shared between them and are equal. A DataQuery is considered
        equal to a DataID if all elements in the query are equal to those
        elements in the DataID. The DataID is still considered equal if it
        contains additional elements. Any DataQuery elements with the value
        ``"*"`` are ignored.

        """
        return self.equal(other, shared_keys=False)

    def equal(self, other: Any, shared_keys: bool = False) -> bool:
        """Compare this DataQuery to another DataQuery or a DataID.

        Args:
            other: Other DataQuery or DataID to compare against.
            shared_keys: Limit keys being compared to those shared
                by both objects. If False (default), then all of the
                current query's keys are used when compared against
                a DataID. If compared against another DataQuery then
                all keys are compared between the two queries.

        """
        sdict = self._asdict()
        try:
            odict = other._asdict()
        except AttributeError:
            return False

        if not sdict and not odict:
            return True

        # if other is a DataID then must match this query exactly
        o_is_id = hasattr(other, "id_keys")
        keys_to_match = _keys_to_compare(sdict, odict, o_is_id, shared_keys)
        if not keys_to_match:
            return False

        for key in keys_to_match:
            if not _compare_key_equality(sdict, odict, key, o_is_id):
                return False
        return True

    def __hash__(self):
        """Hash."""
        fields = []
        values = []
        for field, value in sorted(self._to_trimmed_dict().items()):
            fields.append(field)
            if isinstance(value, list):
                # list or tuple is ordered (ex. modifiers)
                value = tuple(value)
            elif isinstance(value, set):
                # a set is unordered, but must be sorted for consistent hashing
                value = tuple(sorted(value))
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
                if val != "*"}

    def __repr__(self):
        """Represent the query."""
        items = ("{}={}".format(key, repr(val)) for key, val in zip(self._fields, self._values))
        return self.__class__.__name__ + "(" + ", ".join(items) + ")"

    def filter_dataids(self, dataid_container, shared_keys: bool = False):
        """Filter DataIDs based on this query."""
        func = partial(self.equal, shared_keys=shared_keys)
        keys = list(filter(func, dataid_container))
        return keys

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
                val = self._dict.get(key, "*")
                if val == "*":
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
        new_dict["modifiers"] = tuple(new_dict["modifiers"][:-1])
        return DataQuery.from_dict(new_dict)

    def is_modified(self):
        """Check if this is modified."""
        return bool(self._dict.get("modifiers"))


def create_filtered_query(dataset_key, filter_query):
    """Create a DataQuery matching *dataset_key* and *filter_query*.

    If a property is specified in both *dataset_key* and *filter_query*, the former
    has priority.

    """
    ds_dict = _create_id_dict_from_any_key(dataset_key)
    _update_dict_with_filter_query(ds_dict, filter_query)

    return DataQuery.from_dict(ds_dict)


def _update_dict_with_filter_query(ds_dict: dict[str, Any], filter_query: dict[str, Any]) -> None:
    if filter_query is not None:
        for key, value in filter_query.items():
            if value != "*":
                ds_dict.setdefault(key, value)


def _create_id_dict_from_any_key(dataset_key: DataQuery | DataID | str | numbers.Number) -> dict[str, Any]:
    if hasattr(dataset_key, "to_dict"):
        ds_dict = dataset_key.to_dict()
    elif isinstance(dataset_key, str):
        ds_dict = {"name": dataset_key}
    elif isinstance(dataset_key, numbers.Number):
        ds_dict = {"wavelength": dataset_key}
    else:
        raise TypeError("Don't know how to interpret a dataset_key of type {}".format(type(dataset_key)))
    return ds_dict


def update_id_with_query(orig_id: DataID, query: DataQuery) -> DataID:
    """Update a DataID with additional info from a query used to find it."""
    query_dict = query.to_dict()
    if not query_dict:
        return orig_id

    new_id_dict = orig_id.to_dict()
    orig_id_keys = orig_id.id_keys
    for query_key, query_val in query_dict.items():
        # XXX: What if the query_val is a list?
        if new_id_dict.get(query_key) is None:
            new_id_dict[query_key] = query_val
    # don't replace ID key information if we don't have to
    id_keys = orig_id_keys if all(key in orig_id_keys for key in new_id_dict) else default_id_keys_config
    new_id = DataID(id_keys, **new_id_dict)
    return new_id


def _keys_to_compare(sdict: dict, odict: dict, o_is_id: bool, shared_keys: bool) -> set:
    keys_to_match = set(sdict.keys())
    if not o_is_id:
        # if another DataQuery, then compare both sets of keys
        keys_to_match |= set(odict.keys())
    if shared_keys:
        # only compare with the keys that both objects share
        keys_to_match &= set(odict.keys())
    return keys_to_match


def _compare_key_equality(sdict: dict, odict: dict, key: str, o_is_id: bool) -> bool:
    if key not in sdict:
        return False
    sval = sdict[key]
    if sval == "*":
        return True

    if key not in odict:
        return False
    oval = odict[key]
    if oval == "*":
        # Gotcha: if a DataID contains a "*" this could cause
        #    unexpected matches. A DataID is not expected to use "*"
        return True

    return _compare_values(sval, oval, o_is_id)


def _compare_values(sval: Any, oval: Any, o_is_id: bool) -> bool:
    if isinstance(sval, list) or isinstance(oval, list):
        # multiple options to match
        if not isinstance(sval, list):
            # query to query comparison, make a list to iterate over
            sval = [sval]
        if o_is_id:
            return oval in sval

        # we're matching against a DataQuery who could have its own list
        if not isinstance(oval, list):
            oval = [oval]
        s_in_o = any(_sval in oval for _sval in sval)
        o_in_s = any(_oval in sval for _oval in oval)
        return s_in_o or o_in_s
    return oval == sval
