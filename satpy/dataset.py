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


DATASET_KEYS = ("name", "wavelength", "resolution", "polarization",
                "calibration", "level", "modifiers")
DatasetID = namedtuple("DatasetID", " ".join(DATASET_KEYS))
DatasetID.__new__.__defaults__ = (None, None, None, None, None, None, tuple())


class DatasetID(DatasetID):
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
        ret = super(DatasetID, cls).__new__(cls, *args, **kwargs)
        if ret.modifiers is not None and not isinstance(ret.modifiers, tuple):
            raise TypeError("'DatasetID' modifiers must be a tuple or None, "
                            "not {}".format(type(ret.modifiers)))
        return ret

    @staticmethod
    def name_match(a, b):
        """Return if two string names are equal.

        Args:
            a (str): DatasetID.name or other string
            b (str): DatasetID.name or other string
        """
        return a == b

    @staticmethod
    def wavelength_match(a, b):
        """Return if two wavelengths are equal.

        Args:
            a (tuple or scalar): (min wl, nominal wl, max wl) or scalar wl
            b (tuple or scalar): (min wl, nominal wl, max wl) or scalar wl
        """
        if type(a) == (type(b) or
                       isinstance(a, numbers.Number) and
                       isinstance(b, numbers.Number)):
            return a == b
        elif a is None or b is None:
            return False
        elif isinstance(a, (list, tuple)) and len(a) == 3:
            return a[0] <= b <= a[2]
        elif isinstance(b, (list, tuple)) and len(b) == 3:
            return b[0] <= a <= b[2]
        else:
            raise ValueError("Can only compare wavelengths of length 1 or 3")

    def _comparable(self):
        """Get a comparable version of the DatasetID.

        Without this DatasetIDs often raise an exception when compared in
        Python 3 due to None not being comparable with other types.
        """
        return self._replace(
            name='' if self.name is None else self.name,
            wavelength=tuple() if self.wavelength is None else self.wavelength,
            resolution=0 if self.resolution is None else self.resolution,
            polarization='' if self.polarization is None else self.polarization,
            calibration='' if self.calibration is None else self.calibration,
        )

    def __lt__(self, other):
        """Less than."""
        """Compare DatasetIDs with special handling of `None` values"""
        # modifiers should never be None when sorted, should be tuples
        if isinstance(other, DatasetID):
            other = other._comparable()
        return super(DatasetID, self._comparable()).__lt__(other)

    def __eq__(self, other):
        """Check for equality."""
        if isinstance(other, str):
            return self.name_match(self.name, other)
        elif isinstance(other, numbers.Number) or \
                isinstance(other, (tuple, list)) and len(other) == 3:
            return self.wavelength_match(self.wavelength, other)
        else:
            return super(DatasetID, self).__eq__(other)

    def __hash__(self):
        """Generate the hash of the ID."""
        return tuple.__hash__(self)

    @classmethod
    def from_dict(cls, d, **kwargs):
        """Convert a dict to an ID."""
        args = []
        for k in DATASET_KEYS:
            val = kwargs.get(k, d.get(k))
            # force modifiers to tuple
            if k == 'modifiers' and val is not None:
                val = tuple(val)
            args.append(val)

        return cls(*args)

    def to_dict(self, trim=True):
        """Convert the ID to a dict."""
        if trim:
            return self._to_trimmed_dict()
        else:
            return dict(zip(DATASET_KEYS, self))

    def _to_trimmed_dict(self):
        return {key: getattr(self, key) for key in DATASET_KEYS
                if getattr(self, key) is not None}


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


class Dataset(object):
    """Placeholder for the deprecated class."""

    pass
