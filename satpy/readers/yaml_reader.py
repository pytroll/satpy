#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016, 2017.

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

# New stuff

import glob
import itertools
import logging
import os
import warnings
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import deque, OrderedDict
from fnmatch import fnmatch
from weakref import WeakValueDictionary

import six
import xarray as xr
import yaml

try:
    from yaml import UnsafeLoader
except ImportError:
    from yaml import Loader as UnsafeLoader

from pyresample.geometry import StackedAreaDefinition, SwathDefinition
from pyresample.boundary import AreaDefBoundary, Boundary
from satpy.resample import get_area_def
from satpy.config import recursive_dict_update
from satpy.dataset import DATASET_KEYS, DatasetID
from satpy.readers import DatasetDict, get_key
from satpy.readers.utils import get_area_slices, get_sub_area
from trollsift.parser import globify, parse
from satpy import CHUNK_SIZE

logger = logging.getLogger(__name__)


def listify_string(something):
    """Takes *something* and make it a list.

    *something* is either a list of strings or a string, in which case the
    function returns a list containing the string.
    If *something* is None, an empty list is returned.
    """
    if isinstance(something, (str, six.text_type)):
        return [something]
    elif something is not None:
        return list(something)
    else:
        return list()


def get_filebase(path, pattern):
    """Get the end of *path* of same length as *pattern*."""
    # A pattern can include directories
    tail_len = len(pattern.split(os.path.sep))
    return os.path.join(*str(path).split(os.path.sep)[-tail_len:])


def match_filenames(filenames, pattern):
    """Get the filenames matching *pattern*."""
    matching = []

    for filename in filenames:
        if fnmatch(get_filebase(filename, pattern), globify(pattern)):
            matching.append(filename)

    return matching


class AbstractYAMLReader(six.with_metaclass(ABCMeta, object)):

    def __init__(self, config_files):
        self.config = {}
        self.config_files = config_files
        for config_file in config_files:
            with open(config_file) as fd:
                self.config = recursive_dict_update(self.config, yaml.load(fd, Loader=UnsafeLoader))

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
        self.info['filenames'] = []
        self.ids = {}
        self.load_ds_ids_from_config()

    @property
    def sensor_names(self):
        return self.info['sensors'] or []

    @property
    def all_dataset_ids(self):
        return self.ids.keys()

    @property
    def all_dataset_names(self):
        # remove the duplicates from various calibration and resolutions
        return set(ds_id.name for ds_id in self.all_dataset_ids)

    @property
    def available_dataset_ids(self):
        logger.warning(
            "Available datasets are unknown, returning all datasets...")
        return self.all_dataset_ids

    @property
    def available_dataset_names(self):
        return (ds_id.name for ds_id in self.available_dataset_ids)

    @abstractproperty
    def start_time(self):
        """Start time of the reader."""

    @abstractproperty
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
        else:
            return True

    def select_files_from_directory(self, directory=None):
        """Find files for this reader in *directory*.

        If directory is None or '', look in the current directory.
        """
        filenames = []
        if directory is None:
            directory = ''
        for pattern in self.file_patterns:
            matching = glob.iglob(os.path.join(directory, globify(pattern)))
            filenames.extend(matching)
        return filenames

    def select_files_from_pathnames(self, filenames):
        """Select the files from *filenames* this reader can handle."""
        selected_filenames = []

        for pattern in self.file_patterns:
            matching = match_filenames(filenames, pattern)
            for fname in matching:
                if fname not in selected_filenames:
                    selected_filenames.append(fname)
        if len(selected_filenames) == 0:
            logger.warning("No filenames found for reader: %s", self.name)
        return selected_filenames

    def get_dataset_key(self, key, **kwargs):
        """Get the fully qualified `DatasetID` matching `key`.

        See `satpy.readers.get_key` for more information about kwargs.

        """
        return get_key(key, self.ids.keys(), **kwargs)

    def load_ds_ids_from_config(self):
        """Get the dataset ids from the config."""
        ids = []
        for dataset in self.datasets.values():
            # xarray doesn't like concatenating attributes that are lists
            # https://github.com/pydata/xarray/issues/2060
            if 'coordinates' in dataset and \
                    isinstance(dataset['coordinates'], list):
                dataset['coordinates'] = tuple(dataset['coordinates'])
            # Build each permutation/product of the dataset
            id_kwargs = []
            for key in DATASET_KEYS:
                val = dataset.get(key)
                if key in ["wavelength", "modifiers"] and isinstance(val,
                                                                     list):
                    # special case: wavelength can be [min, nominal, max]
                    # but is still considered 1 option
                    # it also needs to be a tuple so it can be used in
                    # a dictionary key (DatasetID)
                    id_kwargs.append((tuple(val), ))
                elif key == "modifiers" and val is None:
                    # empty modifiers means no modifiers applied
                    id_kwargs.append((tuple(), ))
                elif isinstance(val, (list, tuple, set)):
                    # this key has multiple choices
                    # (ex. 250 meter, 500 meter, 1000 meter resolutions)
                    id_kwargs.append(val)
                elif isinstance(val, dict):
                    id_kwargs.append(val.keys())
                else:
                    # this key only has one choice so make it a one
                    # item iterable
                    id_kwargs.append((val, ))
            for id_params in itertools.product(*id_kwargs):
                dsid = DatasetID(*id_params)
                ids.append(dsid)

                # create dataset infos specifically for this permutation
                ds_info = dataset.copy()
                for key in DATASET_KEYS:
                    if isinstance(ds_info.get(key), dict):
                        ds_info.update(ds_info[key][getattr(dsid, key)])
                    # this is important for wavelength which was converted
                    # to a tuple
                    ds_info[key] = getattr(dsid, key)
                self.ids[dsid] = ds_info

        return ids


class FileYAMLReader(AbstractYAMLReader):
    """Implementation of the YAML reader."""

    def __init__(self,
                 config_files,
                 filter_parameters=None,
                 filter_filenames=True,
                 **kwargs):
        super(FileYAMLReader, self).__init__(config_files)

        self.file_handlers = {}
        self.filter_filenames = self.info.get('filter_filenames', filter_filenames)
        self.filter_parameters = filter_parameters or {}
        if kwargs:
            logger.warning("Unrecognized/unused reader keyword argument(s) '{}'".format(kwargs))
        self.coords_cache = WeakValueDictionary()

    @property
    def sensor_names(self):
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
        for ds_id in self.all_dataset_ids:
            fts = self.ids[ds_id]["file_type"]
            if isinstance(fts, str) and fts in self.file_handlers:
                yield ds_id
            elif any(ft in self.file_handlers for ft in fts):
                yield ds_id

    @property
    def start_time(self):
        if not self.file_handlers:
            raise RuntimeError("Start time unknown until files are selected")
        return min(x[0].start_time for x in self.file_handlers.values())

    @property
    def end_time(self):
        if not self.file_handlers:
            raise RuntimeError("End time unknown until files are selected")
        return max(x[-1].end_time for x in self.file_handlers.values())

    @staticmethod
    def check_file_covers_area(file_handler, check_area):
        """Checks if the file covers the current area.

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
        """Iterator over the filenames matching *filetype_info*."""
        matched_files = []
        for pattern in filetype_info['file_patterns']:
            for filename in match_filenames(filenames, pattern):
                if filename in matched_files:
                    continue
                try:
                    filename_info = parse(
                        pattern, get_filebase(filename, pattern))
                except ValueError:
                    logger.debug("Can't parse %s with %s.", filename, pattern)
                    continue
                matched_files.append(filename)
                yield filename, filename_info

    def new_filehandler_instances(self, filetype_info, filename_items, fh_kwargs=None):
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
        start_time = self.filter_parameters.get('start_time')
        end_time = self.filter_parameters.get('end_time')
        fend = fend or fstart
        if start_time and fend and fend < start_time:
            return False
        if end_time and fstart and fstart > end_time:
            return False
        return True

    def metadata_matches(self, sample_dict, file_handler=None):
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
        for filetype, filetype_info in self.sorted_filetype_items():
            filename_iter = self.filename_items_for_filetype(filenames,
                                                             filetype_info)
            if self.filter_filenames:
                filename_iter = self.filter_filenames_by_info(filename_iter)

            for fn, _ in filename_iter:
                yield fn

    def new_filehandlers_for_filetype(self, filetype_info, filenames, fh_kwargs=None):
        """Create filehandlers for a given filetype."""
        filename_iter = self.filename_items_for_filetype(filenames,
                                                         filetype_info)
        if self.filter_filenames:
            # preliminary filter of filenames based on start/end time
            # to reduce the number of files to open
            filename_iter = self.filter_filenames_by_info(filename_iter)
        filehandler_iter = self.new_filehandler_instances(filetype_info,
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
            filehandlers = self.new_filehandlers_for_filetype(filetype_info,
                                                              filename_set,
                                                              fh_kwargs=fh_kwargs)

            filename_set -= set([fhd.filename for fhd in filehandlers])
            if filehandlers:
                created_fhs[filetype] = filehandlers
                self.file_handlers[filetype] = sorted(
                    self.file_handlers.get(filetype, []) + filehandlers,
                    key=lambda fhd: (fhd.start_time, fhd.filename))

        # update existing dataset IDs with information from the file handler
        self.update_ds_ids_from_file_handlers()

        # load any additional dataset IDs determined dynamically from the file
        self.add_ds_ids_from_files()
        return created_fhs

    def update_ds_ids_from_file_handlers(self):
        """Update DatasetIDs with information from loaded files.

        This is useful, for example, if dataset resolution may change
        depending on what files were loaded.

        """
        for file_handlers in self.file_handlers.values():
            fh = file_handlers[0]
            # update resolution in the dataset IDs for this files resolution
            res = getattr(fh, 'resolution', None)
            if res is None:
                continue

            for ds_id, ds_info in list(self.ids.items()):
                file_types = ds_info['file_type']
                if not isinstance(file_types, list):
                    file_types = [file_types]
                if fh.filetype_info['file_type'] not in file_types:
                    continue
                if ds_id.resolution is not None:
                    continue
                ds_info['resolution'] = res
                new_id = DatasetID.from_dict(ds_info)
                self.ids[new_id] = ds_info
                del self.ids[ds_id]

    def add_ds_ids_from_files(self):
        """Check files for more dynamically discovered datasets."""
        for file_handlers in self.file_handlers.values():
            try:
                fh = file_handlers[0]
                avail_ids = fh.available_datasets()
            except NotImplementedError:
                continue

            # dynamically discover other available datasets
            for ds_id, ds_info in avail_ids:
                # don't overwrite an existing dataset
                # especially from the yaml config
                coordinates = ds_info.get('coordinates')
                if isinstance(coordinates, list):
                    # xarray doesn't like concatenating attributes that are
                    # lists: https://github.com/pydata/xarray/issues/2060
                    ds_info['coordinates'] = tuple(ds_info['coordinates'])
                self.ids.setdefault(ds_id, ds_info)

    @staticmethod
    def _load_dataset(dsid, ds_info, file_handlers, dim='y'):
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

    def _load_dataset_data(self,
                           file_handlers,
                           dsid,
                           xslice=slice(None),
                           yslice=slice(None)):
        ds_info = self.ids[dsid]
        proj = self._load_dataset(dsid, ds_info, file_handlers)
        # FIXME: areas could be concatenated here
        # Update the metadata
        proj.attrs['start_time'] = file_handlers[0].start_time
        proj.attrs['end_time'] = file_handlers[-1].end_time
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

    def _load_area_def(self, dsid, file_handlers):
        """Load the area definition of *dsid*."""
        area_defs = [fh.get_area_def(dsid) for fh in file_handlers]
        area_defs = [area_def for area_def in area_defs
                     if area_def is not None]

        final_area = StackedAreaDefinition(*area_defs)
        return final_area.squeeze()

    def _get_coordinates_for_dataset_key(self, dsid):
        """Get the coordinate dataset keys for *dsid*."""
        ds_info = self.ids[dsid]
        cids = []

        for cinfo in ds_info.get('coordinates', []):
            if not isinstance(cinfo, dict):
                cinfo = {'name': cinfo}

            cinfo['resolution'] = ds_info['resolution']
            if 'polarization' in ds_info:
                cinfo['polarization'] = ds_info['polarization']
            cid = DatasetID(**cinfo)
            cids.append(self.get_dataset_key(cid))

        return cids

    def _get_coordinates_for_dataset_keys(self, dsids):
        """Get all coordinates."""
        coordinates = {}
        for dsid in dsids:
            cids = self._get_coordinates_for_dataset_key(dsid)
            coordinates.setdefault(dsid, []).extend(cids)
        return coordinates

    def _get_file_handlers(self, dsid):
        """Get the file handler to load this dataset."""
        ds_info = self.ids[dsid]

        filetype = self._preferred_filetype(ds_info['file_type'])
        if filetype is None:
            logger.warning("Required file type '%s' not found or loaded for "
                           "'%s'", ds_info['file_type'], dsid.name)
        else:
            return self.file_handlers[filetype]

    def _make_area_from_coords(self, coords):
        """Create an appropriate area with the given *coords*."""
        if len(coords) == 2:
            lon_sn = coords[0].attrs.get('standard_name')
            lat_sn = coords[1].attrs.get('standard_name')
            if lon_sn == 'longitude' and lat_sn == 'latitude':
                key = None
                try:
                    key = (coords[0].data.name, coords[1].data.name)
                    sdef = self.coords_cache.get(key)
                except AttributeError:
                    sdef = None
                if sdef is None:
                    sdef = SwathDefinition(*coords)
                    if key is not None:
                        self.coords_cache[key] = sdef
                sensor_str = '_'.join(self.info['sensors'])
                shape_str = '_'.join(map(str, coords[0].shape))
                sdef.name = "{}_{}_{}_{}".format(sensor_str, shape_str,
                                                 coords[0].attrs['name'],
                                                 coords[1].attrs['name'])
                return sdef
            else:
                raise ValueError(
                    'Coordinates info object missing standard_name key: ' +
                    str(coords))
        elif len(coords) != 0:
            raise NameError("Don't know what to do with coordinates " + str(
                coords))

    def _load_dataset_area(self, dsid, file_handlers, coords):
        """Get the area for *dsid*."""
        try:
            return self._load_area_def(dsid, file_handlers)
        except NotImplementedError:
            if any(x is None for x in coords):
                logger.warning(
                    "Failed to load coordinates for '{}'".format(dsid))
                return None

            area = self._make_area_from_coords(coords)
            if area is None:
                logger.debug("No coordinates found for %s", str(dsid))
            return area

    # TODO: move this out of here.
    def _get_slices(self, area):
        """Get the slices of raw data covering area.

        Args:
            area: the area to slice.

        Returns:
            slice_kwargs: kwargs to pass on to loading giving the span of the
                data to load.
            area: the trimmed area corresponding to the slices.
        """
        slice_kwargs = {}

        if area is not None and self.filter_parameters.get('area') is not None:
            try:
                slices = get_area_slices(area, self.filter_parameters['area'])
                area = get_sub_area(area, *slices)
                slice_kwargs['xslice'], slice_kwargs['yslice'] = slices
            except (NotImplementedError, AttributeError):
                logger.info("Cannot compute specific slice of data to load.")

        return slice_kwargs, area

    def _load_dataset_with_area(self, dsid, coords):
        """Loads *dsid* and it's area if available."""
        file_handlers = self._get_file_handlers(dsid)
        if not file_handlers:
            return

        area = self._load_dataset_area(dsid, file_handlers, coords)
        slice_kwargs, area = self._get_slices(area)

        try:
            ds = self._load_dataset_data(file_handlers, dsid, **slice_kwargs)
        except (KeyError, ValueError) as err:
            logger.exception("Could not load dataset '%s': %s", dsid, str(err))
            return None

        if area is not None:
            ds.attrs['area'] = area
            calc_coords = (('x' not in ds.coords) or('y' not in ds.coords)) and hasattr(area, 'get_proj_vectors_dask')
            if calc_coords and hasattr(area, 'get_proj_vectors'):
                ds['x'], ds['y'] = area.get_proj_vectors()
            elif calc_coords:
                # older pyresample with dask-only method
                ds['x'], ds['y'] = area.get_proj_vectors_dask(CHUNK_SIZE)
        return ds

    def _load_ancillary_variables(self, datasets):
        """Load the ancillary variables of `datasets`."""
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
        loadable_av_ids = [av_id for av_id in all_av_ids if av_id not in datasets]
        if not all_av_ids:
            return
        if loadable_av_ids:
            self.load(loadable_av_ids, previous_datasets=datasets)

        for dataset in datasets.values():
            new_vars = []
            for av_id in dataset.attrs.get('ancillary_variables', []):
                if isinstance(av_id, DatasetID):
                    new_vars.append(datasets[av_id])
                else:
                    new_vars.append(av_id)
            dataset.attrs['ancillary_variables'] = new_vars

    def load(self, dataset_keys, previous_datasets=None):
        """Load `dataset_keys`.

        If `previous_datasets` is provided, do not reload those."""
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
            ds = self._load_dataset_with_area(dsid, coords)
            if ds is not None:
                all_datasets[dsid] = ds
                if dsid in dsids:
                    datasets[dsid] = ds
        self._load_ancillary_variables(all_datasets)

        return datasets
