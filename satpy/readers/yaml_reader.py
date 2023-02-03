#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2022 Satpy developers
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
"""Base classes and utilities for all readers configured by YAML files."""

import glob
import itertools
import logging
import os
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, deque
from contextlib import suppress
from fnmatch import fnmatch
from weakref import WeakValueDictionary

import numpy as np
import xarray as xr
import yaml
from pyresample.boundary import AreaDefBoundary, Boundary
from pyresample.geometry import AreaDefinition, StackedAreaDefinition, SwathDefinition
from trollsift.parser import globify, parse

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader  # type: ignore

from satpy import DatasetDict
from satpy._compat import cache
from satpy.aux_download import DataDownloadMixin
from satpy.dataset import DataID, DataQuery, get_key
from satpy.dataset.dataid import default_co_keys_config, default_id_keys_config, get_keys_from_config
from satpy.resample import add_crs_xy_coords, get_area_def
from satpy.utils import recursive_dict_update

logger = logging.getLogger(__name__)


def listify_string(something):
    """Take *something* and make it a list.

    *something* is either a list of strings or a string, in which case the
    function returns a list containing the string.
    If *something* is None, an empty list is returned.
    """
    if isinstance(something, str):
        return [something]
    if something is not None:
        return list(something)
    return list()


def _get_filebase(path, pattern):
    """Get the end of *path* of same length as *pattern*."""
    # convert any `/` on Windows to `\\`
    path = os.path.normpath(path)
    # A pattern can include directories
    tail_len = len(pattern.split(os.path.sep))
    return os.path.join(*str(path).split(os.path.sep)[-tail_len:])


def _match_filenames(filenames, pattern):
    """Get the filenames matching *pattern*."""
    matching = set()
    glob_pat = globify(pattern)
    for filename in filenames:
        if fnmatch(_get_filebase(filename, pattern), glob_pat):
            matching.add(filename)

    return matching


def _verify_reader_info_assign_config_files(config, config_files):
    try:
        reader_info = config['reader']
    except KeyError:
        raise KeyError(
            "Malformed config file {}: missing reader 'reader'".format(
                config_files))
    else:
        reader_info['config_files'] = config_files


def load_yaml_configs(*config_files, loader=Loader):
    """Merge a series of YAML reader configuration files.

    Args:
        *config_files (str): One or more pathnames
            to YAML-based reader configuration files that will be merged
            to create a single configuration.
        loader: Yaml loader object to load the YAML with. Defaults to
            `CLoader` if libyaml is available, `Loader` otherwise.

    Returns: dict
        Dictionary representing the entire YAML configuration with the
        addition of `config['reader']['config_files']` (the list of
        YAML pathnames that were merged).

    """
    config = {}
    logger.debug('Reading %s', str(config_files))
    for config_file in config_files:
        with open(config_file, 'r', encoding='utf-8') as fd:
            config = recursive_dict_update(config, yaml.load(fd, Loader=loader))
    _verify_reader_info_assign_config_files(config, config_files)
    return config


class AbstractYAMLReader(metaclass=ABCMeta):
    """Base class for all readers that use YAML configuration files.

    This class should only be used in rare cases. Its child class
    `FileYAMLReader` should be used in most cases.

    """

    def __init__(self, config_dict):
        """Load information from YAML configuration file about how to read data files."""
        if isinstance(config_dict, str):
            raise ValueError("Passing config files to create a Reader is "
                             "deprecated. Use ReaderClass.from_config_files "
                             "instead.")
        self.config = config_dict
        self.info = self.config['reader']
        self.name = self.info['name']
        self.file_patterns = []
        for file_type, filetype_info in self.config['file_types'].items():
            filetype_info.setdefault('file_type', file_type)
            # correct separator if needed
            file_patterns = [os.path.join(*pattern.split('/'))
                             for pattern in filetype_info['file_patterns']]
            filetype_info['file_patterns'] = file_patterns
            self.file_patterns.extend(file_patterns)

        if 'sensors' in self.info and not isinstance(self.info['sensors'], (list, tuple)):
            self.info['sensors'] = [self.info['sensors']]
        self.datasets = self.config.get('datasets', {})
        self._id_keys = self.info.get('data_identification_keys', default_id_keys_config)
        self._co_keys = self.info.get('coord_identification_keys', default_co_keys_config)
        self.info['filenames'] = []
        self.all_ids = {}
        self.load_ds_ids_from_config()

    @classmethod
    def from_config_files(cls, *config_files, **reader_kwargs):
        """Create a reader instance from one or more YAML configuration files."""
        config_dict = load_yaml_configs(*config_files)
        return config_dict['reader']['reader'](config_dict, **reader_kwargs)

    @property
    def sensor_names(self):
        """Names of sensors whose data is being loaded by this reader."""
        return self.info['sensors'] or []

    @property
    def all_dataset_ids(self):
        """Get DataIDs of all datasets known to this reader."""
        return self.all_ids.keys()

    @property
    def all_dataset_names(self):
        """Get names of all datasets known to this reader."""
        # remove the duplicates from various calibration and resolutions
        return set(ds_id['name'] for ds_id in self.all_dataset_ids)

    @property
    def available_dataset_ids(self):
        """Get DataIDs that are loadable by this reader."""
        logger.warning(
            "Available datasets are unknown, returning all datasets...")
        return self.all_dataset_ids

    @property
    def available_dataset_names(self):
        """Get names of datasets that are loadable by this reader."""
        return (ds_id['name'] for ds_id in self.available_dataset_ids)

    @property
    @abstractmethod
    def start_time(self):
        """Start time of the reader."""

    @property
    @abstractmethod
    def end_time(self):
        """End time of the reader."""

    @abstractmethod
    def filter_selected_filenames(self, filenames):
        """Filter provided filenames by parameters in reader configuration.

        Returns: iterable of usable files

        """

    @abstractmethod
    def load(self, dataset_keys):
        """Load *dataset_keys*."""

    def supports_sensor(self, sensor):
        """Check if *sensor* is supported.

        Returns True is *sensor* is None.
        """
        if sensor and not (set(self.info.get("sensors")) &
                           set(listify_string(sensor))):
            return False
        return True

    def select_files_from_directory(
            self, directory=None, fs=None):
        """Find files for this reader in *directory*.

        If directory is None or '', look in the current directory.

        Searches the local file system by default.  Can search on a remote
        filesystem by passing an instance of a suitable implementation of
        ``fsspec.spec.AbstractFileSystem``.

        Args:
            directory (Optional[str]): Path to search.
            fs (Optional[FileSystem]): fsspec FileSystem implementation to use.
                                       Defaults to None, using local file
                                       system.

        Returns:
            list of strings describing matching files
        """
        filenames = set()
        if directory is None:
            directory = ''
        # all the glob patterns that we are going to look at
        all_globs = {os.path.join(directory, globify(pattern))
                     for pattern in self.file_patterns}
        # custom filesystem or not
        if fs is None:
            matcher = glob.iglob
        else:
            matcher = fs.glob
        # get all files matching these patterns
        for glob_pat in all_globs:
            filenames.update(matcher(glob_pat))
        return filenames

    def select_files_from_pathnames(self, filenames):
        """Select the files from *filenames* this reader can handle."""
        selected_filenames = []
        filenames = set(filenames)  # make a copy of the inputs

        for pattern in self.file_patterns:
            matching = _match_filenames(filenames, pattern)
            filenames -= matching
            for fname in matching:
                if fname not in selected_filenames:
                    selected_filenames.append(fname)
        if len(selected_filenames) == 0:
            logger.warning("No filenames found for reader: %s", self.name)
        return selected_filenames

    def get_dataset_key(self, key, **kwargs):
        """Get the fully qualified `DataID` matching `key`.

        See `satpy.readers.get_key` for more information about kwargs.

        """
        return get_key(key, self.all_ids.keys(), **kwargs)

    def load_ds_ids_from_config(self):
        """Get the dataset ids from the config."""
        ids = []
        for dataset in self.datasets.values():
            # xarray doesn't like concatenating attributes that are lists
            # https://github.com/pydata/xarray/issues/2060
            if 'coordinates' in dataset and \
                    isinstance(dataset['coordinates'], list):
                dataset['coordinates'] = tuple(dataset['coordinates'])
            id_keys = get_keys_from_config(self._id_keys, dataset)

            # Build each permutation/product of the dataset
            id_kwargs = self._build_id_permutations(dataset, id_keys)

            for id_params in itertools.product(*id_kwargs):
                dsid = DataID(id_keys, **dict(zip(id_keys, id_params)))
                ids.append(dsid)

                # create dataset infos specifically for this permutation
                ds_info = dataset.copy()
                for key in dsid.keys():
                    if isinstance(ds_info.get(key), dict):
                        with suppress(KeyError):
                            # KeyError is suppressed in case the key does not represent interesting metadata,
                            # eg a custom type
                            ds_info.update(ds_info[key][dsid.get(key)])
                    # this is important for wavelength which was converted
                    # to a tuple
                    ds_info[key] = dsid.get(key)
                self.all_ids[dsid] = ds_info
        return ids

    def _build_id_permutations(self, dataset, id_keys):
        """Build each permutation/product of the dataset."""
        id_kwargs = []
        for key, idval in id_keys.items():
            val = dataset.get(key, idval.get('default') if idval is not None else None)
            val_type = None
            if idval is not None:
                val_type = idval.get('type')
            if val_type is not None and issubclass(val_type, tuple):
                # special case: wavelength can be [min, nominal, max]
                # but is still considered 1 option
                id_kwargs.append((val,))
            elif isinstance(val, (list, tuple, set)):
                # this key has multiple choices
                # (ex. 250 meter, 500 meter, 1000 meter resolutions)
                id_kwargs.append(val)
            elif isinstance(val, dict):
                id_kwargs.append(val.keys())
            else:
                # this key only has one choice so make it a one
                # item iterable
                id_kwargs.append((val,))
        return id_kwargs


class FileYAMLReader(AbstractYAMLReader, DataDownloadMixin):
    """Primary reader base class that is configured by a YAML file.

    This class uses the idea of per-file "file handler" objects to read file
    contents and determine what is available in the file. This differs from
    the base :class:`AbstractYAMLReader` which does not depend on individual
    file handler objects. In almost all cases this class should be used over
    its base class and can be used as a reader by itself and requires no
    subclassing.

    """

    # WeakValueDictionary objects must be created at the class level or else
    # dask will not be able to serialize them on a distributed environment
    _coords_cache: WeakValueDictionary = WeakValueDictionary()

    def __init__(self,
                 config_dict,
                 filter_parameters=None,
                 filter_filenames=True,
                 **kwargs):
        """Set up initial internal storage for loading file data."""
        super(FileYAMLReader, self).__init__(config_dict)

        self.file_handlers = {}
        self.available_ids = {}
        self.filter_filenames = self.info.get('filter_filenames', filter_filenames)
        self.filter_parameters = filter_parameters or {}
        self.register_data_files()

    @property
    def sensor_names(self):
        """Names of sensors whose data is being loaded by this reader."""
        if not self.file_handlers:
            return self.info['sensors']

        file_handlers = (handlers[0] for handlers in
                         self.file_handlers.values())
        sensor_names = set()
        for fh in file_handlers:
            try:
                sensor_names.update(fh.sensor_names)
            except NotImplementedError:
                continue
        if not sensor_names:
            return self.info['sensors']
        return sorted(sensor_names)

    @property
    def available_dataset_ids(self):
        """Get DataIDs that are loadable by this reader."""
        return self.available_ids.keys()

    @property
    def start_time(self):
        """Start time of the earlier file used by this reader."""
        if not self.file_handlers:
            raise RuntimeError("Start time unknown until files are selected")
        return min(x[0].start_time for x in self.file_handlers.values())

    @property
    def end_time(self):
        """End time of the latest file used by this reader."""
        if not self.file_handlers:
            raise RuntimeError("End time unknown until files are selected")
        return max(x[-1].end_time for x in self.file_handlers.values())

    @staticmethod
    def check_file_covers_area(file_handler, check_area):
        """Check if the file covers the current area.

        If the file doesn't provide any bounding box information or 'area'
        was not provided in `filter_parameters`, the check returns True.
        """
        try:
            gbb = Boundary(*file_handler.get_bounding_box())
        except NotImplementedError as err:
            logger.debug("Bounding box computation not implemented: %s",
                         str(err))
        else:
            abb = AreaDefBoundary(get_area_def(check_area), frequency=1000)

            intersection = gbb.contour_poly.intersection(abb.contour_poly)
            if not intersection:
                return False
        return True

    def find_required_filehandlers(self, requirements, filename_info):
        """Find the necessary file handlers for the given requirements.

        We assume here requirements are available.

        Raises:
            KeyError, if no handler for the given requirements is available.
            RuntimeError, if there is a handler for the given requirements,
            but it doesn't match the filename info.

        """
        req_fh = []
        filename_info = set(filename_info.items())
        if requirements:
            for requirement in requirements:
                for fhd in self.file_handlers[requirement]:
                    if set(fhd.filename_info.items()).issubset(filename_info):
                        req_fh.append(fhd)
                        break
                else:
                    raise RuntimeError("No matching requirement file of type "
                                       "{}".format(requirement))
                    # break everything and continue to next
                    # filetype!
        return req_fh

    def sorted_filetype_items(self):
        """Sort the instance's filetypes in using order."""
        processed_types = []
        file_type_items = deque(self.config['file_types'].items())
        while len(file_type_items):
            filetype, filetype_info = file_type_items.popleft()

            requirements = filetype_info.get('requires')
            if requirements is not None:
                # requirements have not been processed yet -> wait
                missing = [req for req in requirements
                           if req not in processed_types]
                if missing:
                    file_type_items.append((filetype, filetype_info))
                    continue

            processed_types.append(filetype)
            yield filetype, filetype_info

    @staticmethod
    def filename_items_for_filetype(filenames, filetype_info):
        """Iterate over the filenames matching *filetype_info*."""
        if not isinstance(filenames, set):
            # we perform set operations later on to improve performance
            filenames = set(filenames)
        for pattern in filetype_info['file_patterns']:
            matched_files = set()
            matches = _match_filenames(filenames, pattern)
            for filename in matches:
                try:
                    filename_info = parse(
                        pattern, _get_filebase(filename, pattern))
                except ValueError:
                    logger.debug("Can't parse %s with %s.", filename, pattern)
                    continue
                matched_files.add(filename)
                yield filename, filename_info
            filenames -= matched_files

    def _new_filehandler_instances(self, filetype_info, filename_items, fh_kwargs=None):
        """Generate new filehandler instances."""
        requirements = filetype_info.get('requires')
        filetype_cls = filetype_info['file_reader']

        if fh_kwargs is None:
            fh_kwargs = {}

        for filename, filename_info in filename_items:
            try:
                req_fh = self.find_required_filehandlers(requirements,
                                                         filename_info)
            except KeyError as req:
                msg = "No handler for reading requirement {} for {}".format(
                    req, filename)
                warnings.warn(msg)
                continue
            except RuntimeError as err:
                warnings.warn(str(err) + ' for {}'.format(filename))
                continue

            yield filetype_cls(filename, filename_info, filetype_info, *req_fh, **fh_kwargs)

    def time_matches(self, fstart, fend):
        """Check that a file's start and end time mtach filter_parameters of this reader."""
        start_time = self.filter_parameters.get('start_time')
        end_time = self.filter_parameters.get('end_time')
        fend = fend or fstart
        if start_time and fend and fend < start_time:
            return False
        if end_time and fstart and fstart > end_time:
            return False
        return True

    def metadata_matches(self, sample_dict, file_handler=None):
        """Check that file metadata matches filter_parameters of this reader."""
        # special handling of start/end times
        if not self.time_matches(
                sample_dict.get('start_time'), sample_dict.get('end_time')):
            return False
        for key, val in self.filter_parameters.items():
            if key != 'area' and key not in sample_dict:
                continue

            if key in ['start_time', 'end_time']:
                continue
            elif key == 'area' and file_handler:
                if not self.check_file_covers_area(file_handler, val):
                    logger.info('Filtering out %s based on area',
                                file_handler.filename)
                    break
            elif key in sample_dict and val != sample_dict[key]:
                # don't use this file
                break
        else:
            # all the metadata keys are equal
            return True
        return False

    def filter_filenames_by_info(self, filename_items):
        """Filter out file using metadata from the filenames.

        Currently only uses start and end time. If only start time is available
        from the filename, keep all the filename that have a start time before
        the requested end time.
        """
        for filename, filename_info in filename_items:
            fend = filename_info.get('end_time')
            fstart = filename_info.setdefault('start_time', fend)
            if fend and fend < fstart:
                # correct for filenames with 1 date and 2 times
                fend = fend.replace(year=fstart.year,
                                    month=fstart.month,
                                    day=fstart.day)
                filename_info['end_time'] = fend
            if self.metadata_matches(filename_info):
                yield filename, filename_info

    def filter_fh_by_metadata(self, filehandlers):
        """Filter out filehandlers using provide filter parameters."""
        for filehandler in filehandlers:
            filehandler.metadata['start_time'] = filehandler.start_time
            filehandler.metadata['end_time'] = filehandler.end_time
            if self.metadata_matches(filehandler.metadata, filehandler):
                yield filehandler

    def filter_selected_filenames(self, filenames):
        """Filter provided files based on metadata in the filename."""
        if not isinstance(filenames, set):
            # we perform set operations later on to improve performance
            filenames = set(filenames)
        for _, filetype_info in self.sorted_filetype_items():
            filename_iter = self.filename_items_for_filetype(filenames,
                                                             filetype_info)
            if self.filter_filenames:
                filename_iter = self.filter_filenames_by_info(filename_iter)

            for fn, _ in filename_iter:
                yield fn

    def _new_filehandlers_for_filetype(self, filetype_info, filenames, fh_kwargs=None):
        """Create filehandlers for a given filetype."""
        filename_iter = self.filename_items_for_filetype(filenames,
                                                         filetype_info)
        if self.filter_filenames:
            # preliminary filter of filenames based on start/end time
            # to reduce the number of files to open
            filename_iter = self.filter_filenames_by_info(filename_iter)
        filehandler_iter = self._new_filehandler_instances(filetype_info,
                                                           filename_iter,
                                                           fh_kwargs=fh_kwargs)
        filtered_iter = self.filter_fh_by_metadata(filehandler_iter)
        return list(filtered_iter)

    def create_filehandlers(self, filenames, fh_kwargs=None):
        """Organize the filenames into file types and create file handlers."""
        filenames = list(OrderedDict.fromkeys(filenames))
        logger.debug("Assigning to %s: %s", self.info['name'], filenames)

        self.info.setdefault('filenames', []).extend(filenames)
        filename_set = set(filenames)
        created_fhs = {}
        # load files that we know about by creating the file handlers
        for filetype, filetype_info in self.sorted_filetype_items():
            filehandlers = self._new_filehandlers_for_filetype(filetype_info,
                                                               filename_set,
                                                               fh_kwargs=fh_kwargs)

            if filehandlers:
                created_fhs[filetype] = filehandlers
                self.file_handlers[filetype] = sorted(
                    self.file_handlers.get(filetype, []) + filehandlers,
                    key=lambda fhd: (fhd.start_time, fhd.filename))

        # load any additional dataset IDs determined dynamically from the file
        # and update any missing metadata that only the file knows
        self.update_ds_ids_from_file_handlers()
        return created_fhs

    def _file_handlers_available_datasets(self):
        """Generate a series of available dataset information.

        This is done by chaining file handler's
        :meth:`satpy.readers.file_handlers.BaseFileHandler.available_datasets`
        together. See that method's documentation for more information.

        Returns:
            Generator of (bool, dict) where the boolean tells whether the
            current dataset is available from any of the file handlers. The
            boolean can also be None in the case where no loaded file handler
            is configured to load the dataset. The
            dictionary is the metadata provided either by the YAML
            configuration files or by the file handler itself if it is a new
            dataset. The file handler may have also supplemented or modified
            the information.

        """
        # flatten all file handlers in to one list
        flat_fhs = (fh for fhs in self.file_handlers.values() for fh in fhs)
        id_values = list(self.all_ids.values())
        configured_datasets = ((None, ds_info) for ds_info in id_values)
        for fh in flat_fhs:
            # chain the 'available_datasets' methods together by calling the
            # current file handler's method with the previous ones result
            configured_datasets = fh.available_datasets(configured_datasets=configured_datasets)
        return configured_datasets

    def update_ds_ids_from_file_handlers(self):
        """Add or modify available dataset information.

        Each file handler is consulted on whether or not it can load the
        dataset with the provided information dictionary.
        See
        :meth:`satpy.readers.file_handlers.BaseFileHandler.available_datasets`
        for more information.

        """
        avail_datasets = self._file_handlers_available_datasets()
        new_ids = {}
        for is_avail, ds_info in avail_datasets:
            # especially from the yaml config
            coordinates = ds_info.get('coordinates')
            if isinstance(coordinates, list):
                # xarray doesn't like concatenating attributes that are
                # lists: https://github.com/pydata/xarray/issues/2060
                ds_info['coordinates'] = tuple(ds_info['coordinates'])

            ds_info.setdefault('modifiers', tuple())  # default to no mods

            # Create DataID for this dataset
            ds_id = DataID(self._id_keys, **ds_info)
            # all datasets
            new_ids[ds_id] = ds_info
            # available datasets
            # False == we have the file type but it doesn't have this dataset
            # None == we don't have the file type object to ask
            if is_avail:
                self.available_ids[ds_id] = ds_info
        self.all_ids = new_ids

    @staticmethod
    def _load_dataset(dsid, ds_info, file_handlers, dim='y', **kwargs):
        """Load only a piece of the dataset."""
        slice_list = []
        failure = True
        for fh in file_handlers:
            try:
                projectable = fh.get_dataset(dsid, ds_info)
                if projectable is not None:
                    slice_list.append(projectable)
                    failure = False
            except KeyError:
                logger.warning("Failed to load {} from {}".format(dsid, fh),
                               exc_info=True)

        if failure:
            raise KeyError(
                "Could not load {} from any provided files".format(dsid))

        if dim not in slice_list[0].dims:
            return slice_list[0]
        res = xr.concat(slice_list, dim=dim)

        combined_info = file_handlers[0].combine_info(
            [p.attrs for p in slice_list])

        res.attrs = combined_info
        return res

    def _load_dataset_data(self, file_handlers, dsid, **kwargs):
        ds_info = self.all_ids[dsid]
        proj = self._load_dataset(dsid, ds_info, file_handlers, **kwargs)
        # FIXME: areas could be concatenated here
        # Update the metadata
        proj.attrs['start_time'] = file_handlers[0].start_time
        proj.attrs['end_time'] = file_handlers[-1].end_time
        proj.attrs['reader'] = self.name
        return proj

    def _preferred_filetype(self, filetypes):
        """Get the preferred filetype out of the *filetypes* list.

        At the moment, it just returns the first filetype that has been loaded.
        """
        if not isinstance(filetypes, list):
            filetypes = [filetypes]

        # look through the file types and use the first one that we have loaded
        for filetype in filetypes:
            if filetype in self.file_handlers:
                return filetype
        return None

    def _load_area_def(self, dsid, file_handlers, **kwargs):
        """Load the area definition of *dsid*."""
        return _load_area_def(dsid, file_handlers)

    def _get_file_handlers(self, dsid):
        """Get the file handler to load this dataset."""
        ds_info = self.all_ids[dsid]

        filetype = self._preferred_filetype(ds_info['file_type'])
        if filetype is None:
            logger.warning("Required file type '%s' not found or loaded for "
                           "'%s'", ds_info['file_type'], dsid['name'])
        else:
            return self.file_handlers[filetype]

    def _load_dataset_area(self, dsid, file_handlers, coords, **kwargs):
        """Get the area for *dsid*."""
        try:
            return self._load_area_def(dsid, file_handlers, **kwargs)
        except NotImplementedError:
            if any(x is None for x in coords):
                logger.warning(
                    "Failed to load coordinates for '{}'".format(dsid))
                return None

            area = self._make_area_from_coords(coords)
            if area is None:
                logger.debug("No coordinates found for %s", str(dsid))
            return area

    def _make_area_from_coords(self, coords):
        """Create an appropriate area with the given *coords*."""
        if len(coords) == 2:
            lons, lats = self._get_lons_lats_from_coords(coords)
            sdef = self._make_swath_definition_from_lons_lats(lons, lats)
            return sdef
        if len(coords) != 0:
            raise NameError("Don't know what to do with coordinates " + str(
                coords))

    def _get_lons_lats_from_coords(self, coords):
        """Get lons and lats from the coords list."""
        lons, lats = None, None
        for coord in coords:
            if coord.attrs.get('standard_name') == 'longitude':
                lons = coord
            elif coord.attrs.get('standard_name') == 'latitude':
                lats = coord
        if lons is None or lats is None:
            raise ValueError('Missing longitude or latitude coordinate: ' + str(coords))
        return lons, lats

    def _make_swath_definition_from_lons_lats(self, lons, lats):
        """Make a swath definition instance from lons and lats."""
        key = None
        try:
            key = (lons.data.name, lats.data.name)
            sdef = FileYAMLReader._coords_cache.get(key)
        except AttributeError:
            sdef = None
        if sdef is None:
            sdef = SwathDefinition(lons, lats)
            sensor_str = '_'.join(self.info['sensors'])
            shape_str = '_'.join(map(str, lons.shape))
            sdef.name = "{}_{}_{}_{}".format(sensor_str, shape_str,
                                             lons.attrs.get('name', lons.name),
                                             lats.attrs.get('name', lats.name))
            if key is not None:
                FileYAMLReader._coords_cache[key] = sdef
        return sdef

    def _load_dataset_with_area(self, dsid, coords, **kwargs):
        """Load *dsid* and its area if available."""
        file_handlers = self._get_file_handlers(dsid)
        if not file_handlers:
            return

        try:
            ds = self._load_dataset_data(file_handlers, dsid, **kwargs)
        except (KeyError, ValueError) as err:
            logger.exception("Could not load dataset '%s': %s", dsid, str(err))
            return None

        coords = self._assign_coords_from_dataarray(coords, ds)

        area = self._load_dataset_area(dsid, file_handlers, coords, **kwargs)

        if area is not None:
            ds.attrs['area'] = area
            ds = add_crs_xy_coords(ds, area)
        return ds

    @staticmethod
    def _assign_coords_from_dataarray(coords, ds):
        """Assign coords from the *ds* dataarray if needed."""
        if not coords:
            coords = []
            for coord in ds.coords.values():
                if coord.attrs.get('standard_name') in ['longitude', 'latitude']:
                    coords.append(coord)
        return coords

    def _load_ancillary_variables(self, datasets, **kwargs):
        """Load the ancillary variables of `datasets`."""
        all_av_ids = self._gather_ancillary_variables_ids(datasets)
        loadable_av_ids = [av_id for av_id in all_av_ids if av_id not in datasets]
        if not all_av_ids:
            return
        if loadable_av_ids:
            self.load(loadable_av_ids, previous_datasets=datasets, **kwargs)

        for dataset in datasets.values():
            new_vars = []
            for av_id in dataset.attrs.get('ancillary_variables', []):
                if isinstance(av_id, DataID):
                    new_vars.append(datasets[av_id])
                else:
                    new_vars.append(av_id)
            dataset.attrs['ancillary_variables'] = new_vars

    def _gather_ancillary_variables_ids(self, datasets):
        """Gather ancillary variables' ids.

        This adds/modifies the dataset's `ancillary_variables` attr.
        """
        all_av_ids = set()
        for dataset in datasets.values():
            ancillary_variables = dataset.attrs.get('ancillary_variables', [])
            if not isinstance(ancillary_variables, (list, tuple, set)):
                ancillary_variables = ancillary_variables.split(' ')
            av_ids = []
            for key in ancillary_variables:
                try:
                    av_ids.append(self.get_dataset_key(key))
                except KeyError:
                    logger.warning("Can't load ancillary dataset %s", str(key))

            all_av_ids |= set(av_ids)
            dataset.attrs['ancillary_variables'] = av_ids
        return all_av_ids

    def get_dataset_key(self, key, available_only=False, **kwargs):
        """Get the fully qualified `DataID` matching `key`.

        This will first search through available DataIDs, datasets that
        should be possible to load, and fallback to "known" datasets, those
        that are configured but aren't loadable from the provided files.
        Providing ``available_only=True`` will stop this fallback behavior
        and raise a ``KeyError`` exception if no available dataset is found.

        Args:
            key (str, float, DataID, DataQuery): Key to search for in this reader.
            available_only (bool): Search only loadable datasets for the
                provided key. Loadable datasets are always searched first,
                but if ``available_only=False`` (default) then all known
                datasets will be searched.
            kwargs: See :func:`satpy.readers.get_key` for more information about
                kwargs.

        Returns:
            Best matching DataID to the provided ``key``.

        Raises:
            KeyError: if no key match is found.

        """
        try:
            return get_key(key, self.available_dataset_ids, **kwargs)
        except KeyError:
            if available_only:
                raise
            return get_key(key, self.all_dataset_ids, **kwargs)

    def load(self, dataset_keys, previous_datasets=None, **kwargs):
        """Load `dataset_keys`.

        If `previous_datasets` is provided, do not reload those.
        """
        all_datasets = previous_datasets or DatasetDict()
        datasets = DatasetDict()

        # Include coordinates in the list of datasets to load
        dsids = [self.get_dataset_key(ds_key) for ds_key in dataset_keys]
        coordinates = self._get_coordinates_for_dataset_keys(dsids)
        all_dsids = list(set().union(*coordinates.values())) + dsids
        for dsid in all_dsids:
            if dsid in all_datasets:
                continue
            coords = [all_datasets.get(cid, None)
                      for cid in coordinates.get(dsid, [])]
            ds = self._load_dataset_with_area(dsid, coords, **kwargs)
            if ds is not None:
                all_datasets[dsid] = ds
                if dsid in dsids:
                    datasets[dsid] = ds
        self._load_ancillary_variables(all_datasets, **kwargs)

        return datasets

    def _get_coordinates_for_dataset_keys(self, dsids):
        """Get all coordinates."""
        coordinates = {}
        for dsid in dsids:
            cids = self._get_coordinates_for_dataset_key(dsid)
            coordinates.setdefault(dsid, []).extend(cids)
        return coordinates

    def _get_coordinates_for_dataset_key(self, dsid):
        """Get the coordinate dataset keys for *dsid*."""
        ds_info = self.all_ids[dsid]
        cids = []
        for cinfo in ds_info.get('coordinates', []):
            if not isinstance(cinfo, dict):
                cinfo = {'name': cinfo}

            for key in self._co_keys:
                if key == 'name':
                    continue
                if key in ds_info:
                    if ds_info[key] is not None:
                        cinfo[key] = ds_info[key]
            cid = DataQuery.from_dict(cinfo)

            cids.append(self.get_dataset_key(cid))

        return cids


def _load_area_def(dsid, file_handlers):
    """Load the area definition of *dsid*."""
    area_defs = [fh.get_area_def(dsid) for fh in file_handlers]
    area_defs = [area_def for area_def in area_defs
                 if area_def is not None]

    final_area = StackedAreaDefinition(*area_defs)
    return final_area.squeeze()


def _set_orientation(dataset, upper_right_corner):
    """Set the orientation of geostationary datasets.

    Allows to flip geostationary imagery when loading the datasets.
    Example call: scn.load(['VIS008'], upper_right_corner='NE')

    Args:
        dataset: Dataset to be flipped.
        upper_right_corner (str): Direction of the upper right corner of the image after flipping.
                                Possible options are 'NW', 'NE', 'SW', 'SE', or 'native'.
                                The common upright image orientation corresponds to 'NE'.
                                Defaults to 'native' (no flipping is applied).

    """
    # do some checks and early returns
    if upper_right_corner == 'native':
        logger.debug("Requested orientation for Dataset {} is 'native' (default). "
                     "No flipping is applied.".format(dataset.attrs.get('name')))
        return dataset

    if upper_right_corner not in ['NW', 'NE', 'SE', 'SW', 'native']:
        raise ValueError("Target orientation for Dataset {} not recognized. "
                         "Kwarg upper_right_corner should be "
                         "'NW', 'NE', 'SW', 'SE' or 'native'.".format(dataset.attrs.get('name', 'unknown_name')))

    if 'area' not in dataset.attrs:
        logger.info("Dataset {} is missing the area attribute "
                    "and will not be flipped.".format(dataset.attrs.get('name', 'unknown_name')))
        return dataset

    if isinstance(dataset.attrs['area'], SwathDefinition):
        logger.info("Dataset {} is in a SwathDefinition "
                    "and will not be flipped.".format(dataset.attrs.get('name', 'unknown_name')))
        return dataset

    projection_type = _get_projection_type(dataset.attrs['area'])
    accepted_geos_proj_types = ['Geostationary Satellite (Sweep Y)', 'Geostationary Satellite (Sweep X)']
    if projection_type not in accepted_geos_proj_types:
        logger.info("Dataset {} is not in one of the known geostationary projections {} "
                    "and cannot be flipped.".format(dataset.attrs.get('name', 'unknown_name'),
                                                    accepted_geos_proj_types))
        return dataset

    target_eastright, target_northup = _get_target_scene_orientation(upper_right_corner)

    area_extents_to_update = _get_dataset_area_extents_array(dataset.attrs['area'])
    current_eastright, current_northup = _get_current_scene_orientation(area_extents_to_update)

    if target_northup == current_northup and target_eastright == current_eastright:
        logger.info("Dataset {} is already in the target orientation "
                    "and will not be flipped.".format(dataset.attrs.get('name', 'unknown_name')))
        return dataset

    if target_northup != current_northup:
        dataset, area_extents_to_update = _flip_dataset_data_and_area_extents(dataset, area_extents_to_update,
                                                                              'upsidedown')
    if target_eastright != current_eastright:
        dataset, area_extents_to_update = _flip_dataset_data_and_area_extents(dataset, area_extents_to_update,
                                                                              'leftright')

    dataset.attrs['area'] = _get_new_flipped_area_definition(dataset.attrs['area'], area_extents_to_update,
                                                             flip_areadef_stacking=target_northup != current_northup)

    return dataset


def _get_projection_type(dataset_area_attr):
    """Get the projection type from the crs coordinate operation method name."""
    if isinstance(dataset_area_attr, StackedAreaDefinition):
        # assumes all AreaDefinitions in a tackedAreaDefinition have the same projection
        area_crs = dataset_area_attr.defs[0].crs
    else:
        area_crs = dataset_area_attr.crs

    return area_crs.coordinate_operation.method_name


def _get_target_scene_orientation(upper_right_corner):
    """Get the target scene orientation from the target upper_right_corner.

    'NE' corresponds to target_eastright and target_northup being True.
    """
    target_northup = upper_right_corner in ['NW', 'NE']

    target_eastright = upper_right_corner in ['NE', 'SE']

    return target_eastright, target_northup


def _get_dataset_area_extents_array(dataset_area_attr):
    """Get dataset area extents in a numpy array for further flipping."""
    if isinstance(dataset_area_attr, StackedAreaDefinition):
        # array of area extents if the Area is a StackedAreaDefinition
        area_extents_to_update = np.asarray([list(area_def.area_extent) for area_def in dataset_area_attr.defs])
    else:
        # array with a single item if Area is in one piece
        area_extents_to_update = np.asarray([list(dataset_area_attr.area_extent)])
    return area_extents_to_update


def _get_current_scene_orientation(area_extents_to_update):
    """Get the current scene orientation from the area_extents."""
    # assumes all AreaDefinitions inside a StackedAreaDefinition have the same orientation
    current_northup = area_extents_to_update[0, 3] - area_extents_to_update[0, 1] > 0
    current_eastright = area_extents_to_update[0, 2] - area_extents_to_update[0, 0] > 0

    return current_eastright, current_northup


def _flip_dataset_data_and_area_extents(dataset, area_extents_to_update, flip_direction):
    """Flip the data and area extents array for a dataset."""
    logger.info("Flipping Dataset {} {}.".format(dataset.attrs.get('name', 'unknown_name'), flip_direction))
    if flip_direction == 'upsidedown':
        dataset = dataset[::-1, :]
        area_extents_to_update[:, [1, 3]] = area_extents_to_update[:, [3, 1]]
    elif flip_direction == 'leftright':
        dataset = dataset[:, ::-1]
        area_extents_to_update[:, [0, 2]] = area_extents_to_update[:, [2, 0]]
    else:
        raise ValueError("Flip direction not recognized. Should be either 'upsidedown' or 'leftright'.")

    return dataset, area_extents_to_update


def _get_new_flipped_area_definition(dataset_area_attr, area_extents_to_update, flip_areadef_stacking):
    """Get a new area definition with updated area_extents for flipped geostationary datasets."""
    if len(area_extents_to_update) == 1:
        # just update the area extents using the AreaDefinition copy method
        new_area_def = dataset_area_attr.copy(area_extent=area_extents_to_update[0])
    else:
        # update the stacked AreaDefinitions singularly
        new_area_defs_to_stack = []
        for n_area_def, area_def in enumerate(dataset_area_attr.defs):
            new_area_defs_to_stack.append(area_def.copy(area_extent=area_extents_to_update[n_area_def]))

        # flip the order of stacking if the area is upside down
        if flip_areadef_stacking:
            new_area_defs_to_stack = new_area_defs_to_stack[::-1]

        # regenerate the StackedAreaDefinition
        new_area_def = StackedAreaDefinition(*new_area_defs_to_stack)

    return new_area_def


class GEOFlippableFileYAMLReader(FileYAMLReader):
    """Reader for flippable geostationary data."""

    def _load_dataset_with_area(self, dsid, coords, upper_right_corner='native', **kwargs):
        ds = super(GEOFlippableFileYAMLReader, self)._load_dataset_with_area(dsid, coords, **kwargs)

        if ds is not None:
            ds = _set_orientation(ds, upper_right_corner)

        return ds


class GEOSegmentYAMLReader(GEOFlippableFileYAMLReader):
    """Reader for segmented geostationary data.

    This reader pads the data to full geostationary disk if necessary.

    This reader uses an optional ``pad_data`` keyword argument that can be
    passed to :meth:`Scene.load` to control if padding is done (True by
    default). Passing `pad_data=False` will return data unpadded.

    When using this class in a reader's YAML configuration, segmented file
    types (files that may have multiple segments) should specify an extra
    ``expected_segments`` piece of file_type metadata. This tells this reader
    how many total segments it should expect when padding data. Alternatively,
    the file patterns for a file type can include a ``total_segments``
    field which will be used if ``expected_segments`` is not defined. This
    will default to 1 segment.

    """

    def create_filehandlers(self, filenames, fh_kwargs=None):
        """Create file handler objects and determine expected segments for each."""
        created_fhs = super(GEOSegmentYAMLReader, self).create_filehandlers(
            filenames, fh_kwargs=fh_kwargs)

        # add "expected_segments" information
        for fhs in created_fhs.values():
            for fh in fhs:
                # check the filename for total_segments parameter as a fallback
                ts = fh.filename_info.get('total_segments', 1)
                # if the YAML has segments explicitly specified then use that
                fh.filetype_info.setdefault('expected_segments', ts)
                # add segment key-values for FCI filehandlers
                if 'segment' not in fh.filename_info:
                    fh.filename_info['segment'] = fh.filename_info.get('count_in_repeat_cycle', 1)
        return created_fhs

    def _load_dataset(self, dsid, ds_info, file_handlers, dim='y', pad_data=True):
        """Load only a piece of the dataset."""
        if not pad_data:
            return FileYAMLReader._load_dataset(dsid, ds_info,
                                                file_handlers)

        counter, expected_segments, slice_list, failure, projectable = \
            _find_missing_segments(file_handlers, ds_info, dsid)

        if projectable is None or failure:
            raise KeyError(
                "Could not load {} from any provided files".format(dsid))

        filetype = file_handlers[0].filetype_info['file_type']
        self.empty_segment = xr.full_like(projectable, np.nan)
        for i, sli in enumerate(slice_list):
            if sli is None:
                slice_list[i] = self._get_empty_segment(dim=dim, idx=i, filetype=filetype)

        while expected_segments > counter:
            slice_list.append(self._get_empty_segment(dim=dim, idx=counter, filetype=filetype))
            counter += 1

        if dim not in slice_list[0].dims:
            return slice_list[0]
        res = xr.concat(slice_list, dim=dim)

        combined_info = file_handlers[0].combine_info(
            [p.attrs for p in slice_list])

        res.attrs = combined_info
        return res

    def _get_empty_segment(self, **kwargs):
        return self.empty_segment

    def _load_area_def(self, dsid, file_handlers, pad_data=True):
        """Load the area definition of *dsid* with padding."""
        if not pad_data:
            return _load_area_def(dsid, file_handlers)
        return self._load_area_def_with_padding(dsid, file_handlers)

    def _load_area_def_with_padding(self, dsid, file_handlers):
        """Load the area definition of *dsid* with padding."""
        # Pad missing segments between the first available and expected
        area_defs = self._pad_later_segments_area(file_handlers, dsid)

        # Add missing start segments
        area_defs = self._pad_earlier_segments_area(file_handlers, dsid, area_defs)

        # Stack the area definitions
        area_def = _stack_area_defs(area_defs)

        return area_def

    def _pad_later_segments_area(self, file_handlers, dsid):
        """Pad area definitions for missing segments that are later in sequence than the first available."""
        expected_segments = file_handlers[0].filetype_info['expected_segments']
        filetype = file_handlers[0].filetype_info['file_type']
        available_segments = [int(fh.filename_info.get('segment', 1)) for
                              fh in file_handlers]

        area_defs = self._get_segments_areadef_with_later_padded(file_handlers, filetype, dsid, available_segments,
                                                                 expected_segments)

        return area_defs

    def _get_segments_areadef_with_later_padded(self, file_handlers, filetype, dsid, available_segments,
                                                expected_segments):
        seg_size = None
        area_defs = {}
        for segment in range(available_segments[0], expected_segments + 1):
            try:
                idx = available_segments.index(segment)
                fh = file_handlers[idx]
                area = fh.get_area_def(dsid)
            except ValueError:
                area = self._get_new_areadef_for_padded_segment(area, filetype, seg_size, segment, padding_type='later')

            area_defs[segment] = area
            seg_size = area.shape
        return area_defs

    def _pad_earlier_segments_area(self, file_handlers, dsid, area_defs):
        """Pad area definitions for missing segments that are earlier in sequence than the first available."""
        available_segments = [int(fh.filename_info.get('segment', 1)) for
                              fh in file_handlers]
        area = file_handlers[0].get_area_def(dsid)
        seg_size = area.shape
        filetype = file_handlers[0].filetype_info['file_type']

        for segment in range(available_segments[0] - 1, 0, -1):
            area = self._get_new_areadef_for_padded_segment(area, filetype, seg_size, segment, padding_type='earlier')
            area_defs[segment] = area
            seg_size = area.shape

        return area_defs

    def _get_new_areadef_for_padded_segment(self, area, filetype, seg_size, segment, padding_type):
        logger.debug("Padding to full disk with segment nr. %d", segment)
        new_height_px, new_ll_y, new_ur_y = self._get_y_area_extents_for_padded_segment(area, filetype, padding_type,
                                                                                        seg_size, segment)

        fill_extent = (area.area_extent[0], new_ll_y,
                       area.area_extent[2], new_ur_y)
        area = AreaDefinition('fill', 'fill', 'fill', area.crs,
                              seg_size[1], new_height_px,
                              fill_extent)
        return area

    def _get_y_area_extents_for_padded_segment(self, area, filetype, padding_type, seg_size, segment):
        new_height_proj_coord, new_height_px = self._get_new_areadef_heights(area, seg_size,
                                                                             segment_n=segment,
                                                                             filetype=filetype)
        if padding_type == 'later':
            new_ll_y = area.area_extent[1] + new_height_proj_coord
            new_ur_y = area.area_extent[1]
        elif padding_type == 'earlier':
            new_ll_y = area.area_extent[3]
            new_ur_y = area.area_extent[3] - new_height_proj_coord
        else:
            raise ValueError("Padding type not recognised.")
        return new_height_px, new_ll_y, new_ur_y

    def _get_new_areadef_heights(self, previous_area, previous_seg_size, **kwargs):
        new_height_px = previous_seg_size[0]
        new_height_proj_coord = previous_area.area_extent[1] - previous_area.area_extent[3]

        return new_height_proj_coord, new_height_px


def _stack_area_defs(area_def_dict):
    """Stack given dict of area definitions and return a StackedAreaDefinition."""
    area_defs = [area_def_dict[area_def] for
                 area_def in sorted(area_def_dict.keys())
                 if area_def is not None]

    area_def = StackedAreaDefinition(*area_defs)
    area_def = area_def.squeeze()

    return area_def


def _find_missing_segments(file_handlers, ds_info, dsid):
    """Find missing segments."""
    slice_list = []
    failure = True
    counter = 1
    expected_segments = 1
    # get list of file handlers in segment order
    # (ex. first segment, second segment, etc)
    handlers = sorted(file_handlers, key=lambda x: x.filename_info.get('segment', 1))
    projectable = None
    for fh in handlers:
        if fh.filetype_info['file_type'] in ds_info['file_type']:
            expected_segments = fh.filetype_info['expected_segments']

        while int(fh.filename_info.get('segment', 1)) > counter:
            slice_list.append(None)
            counter += 1
        try:
            projectable = fh.get_dataset(dsid, ds_info)
            if projectable is not None:
                slice_list.append(projectable)
                failure = False
                counter += 1
        except KeyError:
            logger.warning("Failed to load %s from %s", str(dsid), str(fh),
                           exc_info=True)

    # The last segment is missing?
    if len(slice_list) < expected_segments:
        slice_list.append(None)

    return counter, expected_segments, slice_list, failure, projectable


def _get_empty_segment_with_height(empty_segment, new_height, dim):
    """Get a new empty segment with the specified height."""
    if empty_segment.shape[0] > new_height:
        # if current empty segment is too tall, slice the DataArray
        return empty_segment[:new_height, :]
    if empty_segment.shape[0] < new_height:
        # if current empty segment is too short, concatenate a slice of the DataArray
        return xr.concat([empty_segment, empty_segment[:new_height - empty_segment.shape[0], :]], dim=dim)
    return empty_segment


class GEOVariableSegmentYAMLReader(GEOSegmentYAMLReader):
    """GEOVariableSegmentYAMLReader for handling segmented GEO products with segments of variable height.

    This YAMLReader overrides parts of the GEOSegmentYAMLReader to account for formats where the segments can
    have variable heights. It computes the sizes of the padded segments using the information available in the
    file(handlers), so that gaps of any size can be filled as needed.

    This implementation was motivated by the FCI L1c format, where the segments (called chunks in the FCI world)
    can have variable heights. It is however generic, so that any future reader can use it. The requirement
    for the reader is to have a method called `get_segment_position_info`, returning a dictionary containing
    the positioning info for each segment (see example in
    :func:`satpy.readers.fci_l1c_nc.FCIL1cNCFileHandler.get_segment_position_info`).

    For more information on please see the documentation of :func:`satpy.readers.yaml_reader.GEOSegmentYAMLReader`.
    """

    def __init__(self,
                 config_dict,
                 filter_parameters=None,
                 filter_filenames=True,
                 **kwargs):
        """Initialise the GEOVariableSegmentYAMLReader object."""
        super().__init__(config_dict, filter_parameters, filter_filenames, **kwargs)
        self.segment_heights = cache(self._segment_heights)
        self.segment_infos = dict()

    def _extract_segment_location_dicts(self, filetype):
        self._initialise_segment_infos(filetype)
        self._collect_segment_position_infos(filetype)
        return

    def _collect_segment_position_infos(self, filetype):
        # collect the segment positioning infos for all available segments
        for fh in self.file_handlers[filetype]:
            chk_infos = fh.get_segment_position_info()
            chk_infos.update({'segment_nr': fh.filename_info['segment'] - 1})
            self.segment_infos[filetype]['available_segment_infos'].append(chk_infos)

    def _initialise_segment_infos(self, filetype):
        # initialise the segment info for this filetype
        filetype_fhs_sample = self.file_handlers[filetype][0]
        exp_segment_nr = filetype_fhs_sample.filetype_info['expected_segments']
        grid_width_to_grid_type = _get_grid_width_to_grid_type(filetype_fhs_sample.get_segment_position_info())
        self.segment_infos.update({filetype: {'available_segment_infos': [],
                                              'expected_segments': exp_segment_nr,
                                              'grid_width_to_grid_type': grid_width_to_grid_type}})

    def _get_empty_segment(self, dim=None, idx=None, filetype=None):
        grid_width = self.empty_segment.shape[1]
        segment_height = self.segment_heights(filetype, grid_width)[idx]
        return _get_empty_segment_with_height(self.empty_segment, segment_height, dim=dim)

    def _segment_heights(self, filetype, grid_width):
        """Compute optimal padded segment heights (in number of pixels) based on the location of available segments."""
        self._extract_segment_location_dicts(filetype)
        grid_type = self.segment_infos[filetype]['grid_width_to_grid_type'][grid_width]
        segment_heights = _compute_optimal_missing_segment_heights(self.segment_infos[filetype], grid_type, grid_width)
        return segment_heights

    def _get_new_areadef_heights(self, previous_area, previous_seg_size, segment_n=None, filetype=None):
        # retrieve the segment height in number of pixels
        grid_width = previous_seg_size[1]
        new_height_px = self.segment_heights(filetype, grid_width)[segment_n - 1]
        # scale the previous vertical area extent using the new pixel height
        prev_area_extent = previous_area.area_extent[1] - previous_area.area_extent[3]
        new_height_proj_coord = prev_area_extent * new_height_px / previous_seg_size[0]

        return new_height_proj_coord, new_height_px


def _get_grid_width_to_grid_type(seg_info):
    grid_width_to_grid_type = dict()
    for grid_type, grid_type_seg_info in seg_info.items():
        grid_width_to_grid_type.update({grid_type_seg_info['grid_width']: grid_type})
    return grid_width_to_grid_type


def _compute_optimal_missing_segment_heights(seg_infos, grid_type, expected_vertical_size):
    # initialise positioning arrays
    segment_start_rows, segment_end_rows, segment_heights = _init_positioning_arrays_for_variable_padding(
        seg_infos['available_segment_infos'], grid_type, seg_infos['expected_segments'])

    # populate start row of first segment and end row of last segment with known values
    segment_start_rows[0] = 1
    segment_end_rows[seg_infos['expected_segments'] - 1] = expected_vertical_size

    # find missing segments and group contiguous missing segments together
    missing_segments = np.where(segment_heights == 0)[0]
    groups_missing_segments = np.split(missing_segments, np.where(np.diff(missing_segments) > 1)[0] + 1)

    for group in groups_missing_segments:
        _compute_positioning_data_for_missing_group(segment_start_rows, segment_end_rows, segment_heights, group)

    return segment_heights.astype('int')


def _compute_positioning_data_for_missing_group(segment_start_rows, segment_end_rows, segment_heights, group):
    _populate_group_start_end_row_using_neighbour_segments(group, segment_end_rows, segment_start_rows)
    proposed_sizes_missing_segments = _compute_proposed_sizes_of_missing_segments_in_group(group, segment_end_rows,
                                                                                           segment_start_rows)
    _populate_start_end_rows_of_missing_segments_with_proposed_sizes(group, proposed_sizes_missing_segments,
                                                                     segment_start_rows, segment_end_rows,
                                                                     segment_heights)


def _populate_start_end_rows_of_missing_segments_with_proposed_sizes(group, proposed_sizes_missing_segments,
                                                                     segment_start_rows, segment_end_rows,
                                                                     segment_heights):
    for n in range(len(group)):
        # start of first and end of last missing segment have been populated already
        if n != 0:
            segment_start_rows[group[n]] = segment_start_rows[group[n - 1]] + proposed_sizes_missing_segments[n] + 1
        if n != len(group) - 1:
            segment_end_rows[group[n]] = segment_start_rows[group[n]] + proposed_sizes_missing_segments[n]
        segment_heights[group[n]] = proposed_sizes_missing_segments[n]


def _compute_proposed_sizes_of_missing_segments_in_group(group, segment_end_rows, segment_start_rows):
    size_group_gap = segment_end_rows[group[-1]] - segment_start_rows[group[0]] + 1
    proposed_sizes_missing_segments = split_integer_in_most_equal_parts(size_group_gap, len(group))
    return proposed_sizes_missing_segments


def _populate_group_start_end_row_using_neighbour_segments(group, segment_end_rows, segment_start_rows):
    # if group is at the start/end of the full-disk, we know the start/end value already
    if segment_start_rows[group[0]] == 0:
        _populate_group_start_row_using_previous_segment(group, segment_end_rows, segment_start_rows)
    if segment_end_rows[group[-1]] == 0:
        _populate_group_end_row_using_later_segment(group, segment_end_rows, segment_start_rows)


def _populate_group_end_row_using_later_segment(group, segment_end_rows, segment_start_rows):
    segment_end_rows[group[-1]] = segment_start_rows[group[-1] + 1] - 1


def _populate_group_start_row_using_previous_segment(group, segment_end_rows, segment_start_rows):
    segment_start_rows[group[0]] = segment_end_rows[group[0] - 1] + 1


def _init_positioning_arrays_for_variable_padding(chk_infos, grid_type, exp_segment_nr):
    segment_heights = np.zeros(exp_segment_nr)
    segment_start_rows = np.zeros(exp_segment_nr)
    segment_end_rows = np.zeros(exp_segment_nr)

    _populate_positioning_arrays_with_available_segment_info(chk_infos, grid_type, segment_start_rows, segment_end_rows,
                                                             segment_heights)
    return segment_start_rows, segment_end_rows, segment_heights


def _populate_positioning_arrays_with_available_segment_info(chk_infos, grid_type, segment_start_rows, segment_end_rows,
                                                             segment_heights):
    for chk_info in chk_infos:
        current_fh_segment_nr = chk_info['segment_nr']
        segment_heights[current_fh_segment_nr] = chk_info[grid_type]['segment_height']
        segment_start_rows[current_fh_segment_nr] = chk_info[grid_type]['start_position_row']
        segment_end_rows[current_fh_segment_nr] = chk_info[grid_type]['end_position_row']


def split_integer_in_most_equal_parts(x, n):
    """Split an integer number x in n parts that are as equally-sizes as possible."""
    if x % n == 0:
        return np.repeat(x // n, n).astype('int')
    else:
        # split the remainder amount over the last remainder parts
        remainder = int(x % n)
        mod = int(x // n)
        ar = np.repeat(mod, n)
        ar[-remainder:] = mod + 1
        return ar.astype('int')
