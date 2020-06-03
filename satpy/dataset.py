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

import logging
import numbers
from collections import namedtuple
from collections.abc import Collection
from datetime import datetime
from enum import IntEnum
from copy import copy, deepcopy
import warnings
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
        return self.name == other

    def __ne__(self, other):
        return self.name != other

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return '<' + str(self) + '>'


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

    def distance(self, value):
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
            raise TypeError("'DatasetID' modifiers must be a tuple or None, "
                            "not {}".format(type(modifiers)))
        return cls(modifiers)

    def __eq__(self, other):
        if isinstance(other, list):
            other = tuple(other)
        return super().__eq__(other)

    def __ne__(self, other):
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
                          'modifiers': {
                              'required': True,
                              'default': ModifierTuple(),
                              'type': ModifierTuple,
                          },
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
        """Return the DatasetID of the object."""
        #print(self.attrs)
        # if self.attrs['name'] is None:
        #     import ipdb; ipdb.set_trace()
        id_keys = self.attrs.get('_id_keys', minimal_default_keys_config)
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
    """Gather keys for a new DatasetID from the ones available in configured dataset."""
    id_keys = {}
    for key, val in common_id_keys.items():
        if key in config:
            id_keys[key] = val
        elif val is not None and val.get('required') is True:
            id_keys[key] = val
    if not id_keys:
        raise ValueError('Metada does not contain enough information to create a DatasetID.')
    return id_keys


def _share_metadata_key(k, values, average_times):
    """Helper for combine_metadata, decide if key is shared."""
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
    """Helper for combine_metadata, check object identity in list of arrays."""
    for val in values[1:]:
        if val is not values[0]:
            return False
    return True


def _share_metadata_key_list_arrays(values):
    """Helper for combine_metadata, check object identity in list of list of arrays."""
    for val in values[1:]:
        for arr, ref in zip(val, values[0]):
            if arr is not ref:
                return False
    return True


def new_dataset_id_class_from_keys(id_keys):
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
            elif 'enum' in val:
                types[key] = ValueList(key, ' '.join(val['enum']))
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



class DataArrayID:
    pass


class DataID(dict):
    def __init__(self, id_keys, **keyval_dict):
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
        return deepcopy(self._id_keys)

    @staticmethod
    def fix_id_keys(id_keys):
        """Sanitize the id keys."""
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
        return self.__class__(self._id_keys, **keyvals)

    @classmethod
    def from_dataarray(cls, array, default_keys=minimal_default_keys_config):
        id_keys = array.attrs.get('_id_keys', default_keys)
        return cls(id_keys, **array.attrs)

    def _asdict(self):
        return dict(self.items())

    def to_dict(self):
        """Convert the ID to a dict."""

        return self._asdict()

    def __getattr__(self, key):
        if key in self._id_keys:
            warnings.warn('Access to DataID attributes is deprecated, use [] instead')
            return self[key]
        else:
            return super().__getattr__(key)

    def __repr__(self):
        """Represent the id."""
        items = ("{}={}".format(key, repr(val)) for key, val in self.items())
        return self.__class__.__name__ + "(" + ", ".join(items) + ")"

    def _replace(self, **kwargs):
        """Make a new instance with replaced items."""
        info = dict(self.items())
        info.update(kwargs)
        return self.from_dict(info)
    # types = {}
    # defaults = []
    # for key, val in id_keys.items():
    #     if val is None:
    #         defaults.append(None)
    #     else:
    #         defaults.append(val.get('default'))
    #         if 'type' in val:
    #             types[key] = val['type']
    #         elif 'enum' in val:
    #             types[key] = ValueList(key, ' '.join(val['enum']))




    #         for ckey, the_type in types.items():
    #             if ckey in kwargs:
    #                 # TODO: do we really need a convert method or should we fix __new__?
    #                 kwargs[ckey] = the_type.convert(kwargs[ckey])
    #         newargs = []
    #         for key, val in zip(cls._fields, args):
    #             if key in types:
    #                 val = types[key].convert(val)
    #             if val is not None:
    #                 newargs.append(val)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(tuple(sorted(self.items())))
        return self._hash

    def _immutable(self, *args, **kws):
        raise TypeError('Cannot change a DatasetID')

    def __lt__(self, other):
        return tuple(self.values()) < tuple(other.values())

    __setitem__ = _immutable
    __delitem__ = _immutable
    pop = _immutable
    popitem = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable



def make_dsid_class(types=None, **kwargs):
    """Make a new DatasetID class."""
    return DataID

    fields = kwargs.keys()
    defaults = kwargs.values()
    klass = namedtuple("DatasetID", " ".join(fields), defaults=defaults)
    if types is None:
        types = {}

    # TODO: put this documentation somewhere sphinx can find it.
    class DatasetID(klass, DataArrayID):
        """Identifier for all `Dataset` objects.

        FIXME: talk about None not being a valid value

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
                if val is not None:
                    newargs.append(val)

            return super(DatasetID, cls).__new__(cls, *newargs, **kwargs)

        def __hash__(self):
            """Hash."""
            return hash((self._fields, tuple.__hash__(self)))

        def __eq__(self, other):
            """Compare the DatasetIDs."""
            if isinstance(other, DatasetQuery):
                return other.__eq__(self)
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
                val = d.get(k)
                if val is not None:
                    newkwargs[k] = val

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


class DatasetQuery:
    """The dataset query object."""

    def __init__(self, **kwargs):
        """Initialize the query."""
        self._dict = kwargs.copy()
        self._fields = tuple(self._dict.keys())
        self._values = tuple(self._dict.values())

    def __getitem__(self, key):
        """Get an item."""
        return self._dict[key]

    def __eq__(self, other):
        """Compare the DatasetIDs."""
        sdict = self._asdict()
        try:
            odict = other._asdict()
        except AttributeError:
            return False
        for key, val in sdict.items():
            if key in odict and odict[key] != val and val is not None:
                return False
        return True

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
        items = ("{}={}".format(key, val) for key, val in zip(self._fields, self._values))
        return self.__class__.__name__ + "(" + ", ".join(items) + ")"

    def filter_dsids(self, dsid_container):
        """Filter datasetids based on this query."""
        keys = iter(dsid_container)
        for key, val in self._dict.items():
            if val != '*':
                keys = [k for k in keys
                        if k.get(key) == val]
        return keys

    def sort_dsids(self, dsids):
        """Sort the datasetids based on this query.

        Returns the sorted dsids and the list of distances.
        """
        distances = []
        sorted_dsids = []
        keys = set(self._dict.keys())
        for dsid in dsids:
            keys |= set(dsid.keys())
        for dsid in sorted(dsids):
            sorted_dsids.append(dsid)
            distance = 0
            for key in keys:
                val = self._dict.get(key, '*')
                if val == '*':
                    try:
                        # for enums
                        distance += dsid.get(key).value
                    except AttributeError:
                        if isinstance(dsid.get(key), numbers.Number):
                            distance += dsid.get(key)
                        elif isinstance(dsid.get(key), tuple):
                            distance += len(dsid.get(key))
                else:
                    try:
                        dsid_val = dsid[key]
                    except KeyError:
                        distance = np.inf
                        break
                    try:
                        distance += dsid_val.distance(val)
                    except AttributeError:
                        if dsid_val != val:
                            distance = np.inf
                            break
            distances.append(distance)
        distances, dsids = zip(*sorted(zip(distances, sorted_dsids)))
        return dsids, distances


"""
  identification_keys:
    name:
      required: true
    wavelength:
      type: !!python/name:satpy.dataset.WavelengthRange
    resolution:
    view:
      default: nadir
    calibration:
    modifiers:
      required: true
      default: []
      type: !!python/name:satpy.dataset.ModifierTuple
"""


def make_trimmed_dsid_from_keys(_id_keys=default_id_keys_config, **items):
    keys = get_keys_from_config(_id_keys, items)
    dsid_class = new_dataset_id_class_from_keys(keys)
    return dsid_class.from_dict(items)


class DatasetID:
    """Fake datasetid."""

    def __init__(self, *args, **kwargs):
        """Fake init."""
        raise TypeError("DatasetID should not be used directly")

    def from_dict(self, *args, **kwargs):
        """Fake fun."""
        raise TypeError("DatasetID should not be used directly")


def create_filtered_query(dataset_key, filter_query):
    """Create a DatasetQuery matching *dataset_key* and *filter_query*.

    If a proprety is specified in both *dataset_key* and *filter_query*, the former
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

    return DatasetQuery.from_dict(ds_dict)


def create_filtered_id(dataset_key, filter_query):
    """Create a DatasetID matching *dataset_key* and *filter_query*.

    If a proprety is specified in both *dataset_key* and *filter_query*, the former
    has priority.

    """
    additional_info = {}
    if filter_query is not None:
        for key, val in filter_query._dict.items():
            if val != '*':
                additional_info[key] = val
    if not additional_info:
        return dataset_key
    else:
        raise NotImplementedError("Missmatch {} vs {}".format(str(dataset_key), str(filter_query)))


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
    id_keys = parent_dataset.attrs.get('_id_keys', dataset.attrs.get('_id_keys'))
    current_dsid = DataID(id_keys, **dataset.attrs)
    for idx, ds in enumerate(parent_dataset.attrs['ancillary_variables']):
        if current_dsid == DataID(id_keys, **ds.attrs):
            parent_dataset.attrs['ancillary_variables'][idx] = dataset
            return
