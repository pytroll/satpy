#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2019 Satpy developers
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
"""Interface for BaseFileHandlers."""

from abc import ABCMeta

import numpy as np
from pathlib import PurePath

from pyresample.geometry import SwathDefinition
from satpy.dataset import combine_metadata


class BaseFileHandler(metaclass=ABCMeta):
    """Base file handler."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize file handler."""
        if isinstance(filename, PurePath):
            self.filename = str(filename)
        else:
            self.filename = filename
        self.navigation_reader = None
        self.filename_info = filename_info
        self.filetype_info = filetype_info
        self.metadata = filename_info.copy()

    def __str__(self):
        """Customize __str__."""
        return "<{}: '{}'>".format(self.__class__.__name__, self.filename)

    def __repr__(self):
        """Customize __repr__."""
        return str(self)

    def get_dataset(self, dataset_id, ds_info):
        """Get dataset."""
        raise NotImplementedError

    def get_area_def(self, dsid):
        """Get area definition."""
        raise NotImplementedError

    def get_bounding_box(self):
        """Get the bounding box of the files, as a (lons, lats) tuple.

        The tuple return should a lons and lats list of coordinates traveling
        clockwise around the points available in the file.
        """
        raise NotImplementedError

    @staticmethod
    def _combine(infos, func, *keys):
        res = {}
        for key in keys:
            if key in infos[0]:
                res[key] = func([i[key] for i in infos])
        return res

    def combine_info(self, all_infos):
        """Combine metadata for multiple datasets.

        When loading data from multiple files it can be non-trivial to combine
        things like start_time, end_time, start_orbit, end_orbit, etc.

        By default this method will produce a dictionary containing all values
        that were equal across **all** provided info dictionaries.

        Additionally it performs the logical comparisons to produce the
        following if they exist:

         - start_time
         - end_time
         - start_orbit
         - end_orbit
         - satellite_altitude
         - satellite_latitude
         - satellite_longitude
         - orbital_parameters

         Also, concatenate the areas.

        """
        combined_info = combine_metadata(*all_infos)

        new_dict = self._combine(all_infos, min, 'start_time', 'start_orbit')
        new_dict.update(self._combine(all_infos, max, 'end_time', 'end_orbit'))
        new_dict.update(self._combine(all_infos, np.mean,
                                      'satellite_longitude',
                                      'satellite_latitude',
                                      'satellite_altitude'))

        # Average orbital parameters
        orb_params = [info.get('orbital_parameters', {}) for info in all_infos]
        if all(orb_params):
            # Collect all available keys
            orb_params_comb = {}
            for d in orb_params:
                orb_params_comb.update(d)

            # Average known keys
            keys = ['projection_longitude', 'projection_latitude', 'projection_altitude',
                    'satellite_nominal_longitude', 'satellite_nominal_latitude',
                    'satellite_actual_longitude', 'satellite_actual_latitude', 'satellite_actual_altitude',
                    'nadir_longitude', 'nadir_latitude']
            orb_params_comb.update(self._combine(orb_params, np.mean, *keys))
            new_dict['orbital_parameters'] = orb_params_comb

        try:
            area = SwathDefinition(lons=np.ma.vstack([info['area'].lons for info in all_infos]),
                                   lats=np.ma.vstack([info['area'].lats for info in all_infos]))
            area.name = '_'.join([info['area'].name for info in all_infos])
            combined_info['area'] = area
        except KeyError:
            pass

        new_dict.update(combined_info)
        return new_dict

    @property
    def start_time(self):
        """Get start time."""
        return self.filename_info['start_time']

    @property
    def end_time(self):
        """Get end time."""
        return self.filename_info.get('end_time', self.start_time)

    @property
    def sensor_names(self):
        """List of sensors represented in this file."""
        raise NotImplementedError

    def file_type_matches(self, ds_ftype):
        """Match file handler's type to this dataset's file type.

        Args:
            ds_ftype (str or list): File type or list of file types that a
                dataset is configured to be loaded from.

        Returns: ``True`` if this file handler object's type matches the
            dataset's file type(s), ``False`` otherwise.

        """
        if isinstance(ds_ftype, str) and ds_ftype == self.filetype_info['file_type']:
            return True
        elif self.filetype_info['file_type'] in ds_ftype:
            return True
        return None

    def available_datasets(self, configured_datasets=None):
        """Get information of available datasets in this file.

        This is used for dynamically specifying what datasets are available
        from a file in addition to what's configured in a YAML configuration
        file. Note that this method is called for each file handler for each
        file type; care should be taken when possible to reduce the amount
        of redundant datasets produced.

        This method should **not** update values of the dataset information
        dictionary **unless** this file handler has a matching file type
        (the data could be loaded from this object in the future) and at least
        **one** :class:`satpy.dataset.DatasetID` key is also modified.
        Otherwise, this file type may override the information provided by
        a more preferred file type (as specified in the YAML file).
        It is recommended that any non-ID metadata be updated during the
        :meth:`BaseFileHandler.get_dataset` part of loading.
        This method is not guaranteed that it will be called before any
        other file type's handler.
        The availability "boolean" not being ``None`` does not mean that a
        file handler called later can't provide an additional dataset, but
        it must provide more identifying (DatasetID) information to do so
        and should yield its new dataset in addition to the previous one.

        Args:
            configured_datasets (list): Series of (bool or None, dict) in the
                same way as is returned by this method (see below). The bool
                is whether or not the dataset is available from at least one
                of the current file handlers. It can also be ``None`` if
                no file handler knows before us knows how to handle it.
                The dictionary is existing dataset metadata. The dictionaries
                are typically provided from a YAML configuration file and may
                be modified, updated, or used as a "template" for additional
                available datasets. This argument could be the result of a
                previous file handler's implementation of this method.

        Returns: Iterator of (bool or None, dict) pairs where dict is the
            dataset's metadata. If the dataset is available in the current
            file type then the boolean value should be ``True``, ``False``
            if we **know** about the dataset but it is unavailable, or
            ``None`` if this file object is not responsible for it.

        Example 1 - Supplement existing configured information::

            def available_datasets(self, configured_datasets=None):
                "Add information to configured datasets."
                # we know the actual resolution
                res = self.resolution

                # update previously configured datasets
                for is_avail, ds_info in (configured_datasets or []):
                    # some other file handler knows how to load this
                    # don't override what they've done
                    if is_avail is not None:
                        yield is_avail, ds_info

                    matches = self.file_type_matches(ds_info['file_type'])
                    if matches and ds_info.get('resolution') != res:
                        # we are meant to handle this dataset (file type matches)
                        # and the information we can provide isn't available yet
                        new_info = ds_info.copy()
                        new_info['resolution'] = res
                        yield True, new_info
                    elif is_avail is None:
                        # we don't know what to do with this
                        # see if another future file handler does
                        yield is_avail, ds_info

        Example 2 - Add dynamic datasets from the file::

            def available_datasets(self, configured_datasets=None):
                "Add information to configured datasets."
                # pass along existing datasets
                for is_avail, ds_info in (configured_datasets or []):
                    yield is_avail, ds_info

                # get dynamic variables known to this file (that we created)
                for var_name, val in self.dynamic_variables.items():
                    ds_info = {
                        'file_type': self.filetype_info['file_type'],
                        'resolution': 1000,
                        'name': var_name,
                    }
                    yield True, ds_info

        """
        for is_avail, ds_info in (configured_datasets or []):
            if is_avail is not None:
                # some other file handler said it has this dataset
                # we don't know any more information than the previous
                # file handler so let's yield early
                yield is_avail, ds_info
                continue
            yield self.file_type_matches(ds_info['file_type']), ds_info
