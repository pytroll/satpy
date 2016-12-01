#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016.

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

import copy
import glob
import itertools
import logging
import numbers
import os
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import namedtuple
from fnmatch import fnmatch

import numpy as np
import six
import yaml

from pyresample import geometry
from satpy.config import recursive_dict_update
from satpy.projectable import Projectable
from satpy.readers import AreaID, DatasetDict, DatasetID
from trollsift.parser import globify, parse

LOG = logging.getLogger(__name__)

Shuttle = namedtuple('Shuttle', ['data', 'mask', 'info'])


class AbstractYAMLReader(six.with_metaclass(ABCMeta, object)):
    __metaclass__ = ABCMeta

    def __init__(self, config_files,
                 start_time=None,
                 end_time=None,
                 area=None):
        self.config = {}
        self._start_time = start_time
        self._end_time = end_time
        self._area = area
        self.config_files = config_files
        for config_file in config_files:
            with open(config_file) as fd:
                self.config = recursive_dict_update(self.config, yaml.load(fd))

        self.info = self.config['reader']
        self.name = self.info['name']
        self.file_patterns = []
        for file_type in self.config['file_types'].values():
            self.file_patterns.extend(file_type['file_patterns'])

        self.sensor_names = self.info['sensors']
        self.datasets = self.config['datasets']
        self.info['filenames'] = []
        self.ids = {}
        self.get_dataset_ids()

    @property
    def all_dataset_ids(self):
        return self.ids.keys()

    @property
    def all_dataset_names(self):
        # remove the duplicates from various calibration and resolutions
        return set(ds_id.name for ds_id in self.all_dataset_ids)

    @property
    def available_dataset_ids(self):
        LOG.warning("Available datasets are unknown, returning all datasets...")
        return self.all_dataset_ids

    @property
    def available_dataset_names(self):
        return (ds_id.name for ds_id in self.available_dataset_ids)

    @abstractproperty
    def start_time(self):
        raise NotImplementedError()

    @abstractproperty
    def end_time(self):
        raise NotImplementedError()

    @abstractmethod
    def load(self):
        raise NotImplementedError()

    def select_files(self, base_dir=None, filenames=None, sensor=None):
        if isinstance(sensor, (str, six.text_type)):
            sensor_set = set([sensor])
        elif sensor is not None:
            sensor_set = set(sensor)
        else:
            sensor_set = set()

        if sensor is not None and not (set(self.info.get("sensors")) &
                                       sensor_set):
            return filenames, []

        file_set = set()
        if filenames:
            file_set |= set(filenames)

        if not filenames:
            self.info["filenames"] = self.find_filenames(base_dir)
        else:
            self.info["filenames"] = self.match_filenames(filenames, base_dir)
        if not self.info["filenames"]:
            LOG.warning("No filenames found for reader: %s", self.name)
        file_set -= set(self.info['filenames'])
        LOG.debug("Assigned to %s: %s", self.info[
            'name'], self.info['filenames'])

        return file_set, self.info['filenames']

    def match_filenames(self, filenames, base_dir=None):
        result = []
        for file_pattern in self.file_patterns:
            if base_dir is not None:
                file_pattern = os.path.join(base_dir, file_pattern)
            pattern = globify(file_pattern)
            if not filenames:
                return result
            for filename in list(filenames):
                if fnmatch(
                        os.path.basename(filename), os.path.basename(pattern)):
                    result.append(filename)
                    filenames.remove(filename)
        return result

    def find_filenames(self, directory, file_patterns=None):
        if file_patterns is None:
            file_patterns = self.file_patterns
            # file_patterns.extend(item['file_patterns'] for item in
            # self.config['file_types'])
        filelist = []
        if directory is None:
            directory = ''
        for pattern in file_patterns:
            filelist.extend(glob.iglob(os.path.join(directory, globify(
                pattern))))
        return filelist

    def get_dataset_key(self,
                        key,
                        calibration=None,
                        resolution=None,
                        polarization=None,
                        modifiers=None,
                        aslist=False):
        """Get the fully qualified dataset corresponding to *key*, either by name or centerwavelength.

        If `key` is a `DatasetID` object its name is searched if it exists, otherwise its wavelength is used.
        """
        # TODO This can be made simpler
        # get by wavelength

        if isinstance(key, numbers.Number):
            datasets = [ds for ds in self.ids
                        if ds.wavelength and (ds.wavelength[0] <= key <=
                                              ds.wavelength[2])]
            datasets = sorted(datasets,
                              key=lambda ch: abs(ch.wavelength[1] - key))
            if not datasets:
                raise KeyError("Can't find any projectable at %gum" % key)
        elif isinstance(key, DatasetID):
            if key.name is not None:
                datasets = self.get_dataset_key(key.name, aslist=True)
            elif key.wavelength is not None:
                datasets = self.get_dataset_key(key.wavelength, aslist=True)
            else:
                raise KeyError("Can't find any projectable '{}'".format(key))

            if calibration is None and key.calibration is not None:
                calibration = [key.calibration]
            if resolution is None and key.resolution is not None:
                resolution = [key.resolution]
            if polarization is None and key.polarization is not None:
                polarization = [key.polarization]
            if modifiers is None and key.modifiers is not None:
                modifiers = key.modifiers
        # get by name
        else:
            datasets = [ds_id for ds_id in self.ids if ds_id.name == key]
            if not datasets:
                raise KeyError("Can't find any projectable called '{}'".format(
                    key))

        if resolution is not None:
            if not isinstance(resolution, (tuple, list, set)):
                resolution = [resolution]
            datasets = [
                ds_id for ds_id in datasets if ds_id.resolution in resolution
            ]

        # default calibration choices
        if calibration is None:
            calibration = ["brightness_temperature",
                           "reflectance", 'radiance', 'counts']
        else:
            calibration = [x
                           for x in ["brightness_temperature", "reflectance",
                                     "radiance", "counts"] if x in calibration]

        new_datasets = []
        
        for cal_name in calibration:
            # order calibration from highest level to lowest level
            for ds_id in datasets:
                if ds_id.calibration == cal_name:
                    new_datasets.append(ds_id)
        for ds_id in datasets:
            if ds_id.calibration is None:
                new_datasets.append(ds_id)
        datasets = new_datasets

        if polarization is not None:
            datasets = [
                ds_id for ds_id in datasets
                if ds_id.polarization in polarization
            ]

        if modifiers is not None:
            datasets = [
                ds_id for ds_id in datasets
                if ds_id.modifiers == modifiers
            ]

        if not datasets:
            raise KeyError("Can't find any projectable matching '{}'".format(
                str(key)))
        if aslist:
            return datasets
        else:
            return datasets[0]

    def get_dataset_ids(self):
        """Get the dataset ids from the config."""
        ids = []
        for dskey, dataset in self.datasets.items():
            # Build each permutation/product of the dataset
            id_kwargs = []
            for key in DatasetID._fields:
                val = dataset.get(key)
                if key in ["wavelength", "modifiers"] and isinstance(val, list):
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
                for key in dsid._fields:
                    if isinstance(ds_info.get(key), dict):
                        ds_info.update(ds_info[key][getattr(dsid, key)])
                    # this is important for wavelength which was converted
                    # to a tuple
                    ds_info[key] = getattr(dsid, key)
                self.ids[dsid] = ds_info

        return ids


class FileYAMLReader(AbstractYAMLReader):

    def __init__(self, config_files,
                 start_time=None,
                 end_time=None,
                 area=None):
        super(FileYAMLReader, self).__init__(config_files,
                                             start_time=start_time,
                                             end_time=end_time,
                                             area=area)

        self.file_handlers = {}

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
        return min(x.start_time for x in self.file_handlers.values()[0])

    @property
    def end_time(self):
        if not self.file_handlers:
            raise RuntimeError("End time unknown until files are selected")
        return max(x.end_time for x in self.file_handlers.values()[0])

    def select_files(self,
                     base_dir=None,
                     filenames=None,
                     sensor=None):
        res = super(FileYAMLReader, self).select_files(base_dir, filenames,
                                                       sensor)

        # Organize filenames in to file types and create file handlers
        remaining_filenames = set(self.info['filenames'])
        for filetype, filetype_info in self.config['file_types'].items():
            filetype_cls = filetype_info['file_reader']
            patterns = filetype_info['file_patterns']
            file_handlers = []
            for pattern in patterns:
                used_filenames = set()

                levels = len(pattern.split('/'))

                for filename in remaining_filenames:
                    filebase = os.path.join(
                        *filename.split(os.path.sep)[-levels:])

                    if fnmatch(filebase, globify(pattern)):
                        # we know how to use this file (even if we may not use
                        # it later)
                        used_filenames.add(filename)
                        filename_info = parse(pattern,
                                              filebase)
                        file_handler = filetype_cls(filename, filename_info,
                                                    filetype_info)

                        # Only add this file handler if it is within the time
                        # we want
                        if self._start_time and file_handler.start_time < self._start_time:
                            continue
                        if self._end_time and file_handler.end_time > self._end_time:
                            continue

                        # TODO: Area filtering

                        file_handlers.append(file_handler)
                remaining_filenames -= used_filenames
            # Only create an entry in the file handlers dictionary if
            # we have those files
            if file_handlers:
                # Sort the file handlers by start time
                file_handlers.sort(key=lambda fh: fh.start_time)
                self.file_handlers[filetype] = file_handlers

        return res

    def _load_dataset(self, file_handlers, dsid, ds_info):
        try:
            # Can we allow the file handlers to do inplace data writes?
            all_shapes = [fh.get_shape(dsid, ds_info) for fh in file_handlers]
            # rows accumlate, columns stay the same
            overall_shape = (sum([x[0] for x in all_shapes]),) + all_shapes[0][1:]
        except (NotImplementedError, Exception):
            # FIXME: Is NotImplementedError included in Exception for all
            # versions of Python?
            all_shapes = None
            overall_shape = None

        cls = ds_info.get("container", Projectable)
        if overall_shape is None:
            # can't optimize by using inplace loading
            projectables = []
            for fh in file_handlers:
                projectable = fh.get_dataset(dsid, ds_info)
                if projectable is not None:
                    projectables.append(projectable)

            # Join them all together
            all_shapes = [x.shape for x in projectables]
            combined_info = file_handlers[0].combine_info(
                [p.info for p in projectables])
            proj = cls(np.ma.vstack(projectables), **combined_info)
            del projectables  # clean up some space since we don't need these anymore
        else:
            # we can optimize
            # create a projectable object for the file handler to fill in
            # proj = cls(np.empty(overall_shape,
            #           dtype=ds_info.get('dtype', np.float32)))

            # overwrite single boolean 'False'
            # proj.mask = np.ma.make_mask_none(overall_shape)
            out_info = {}
            data = np.empty(overall_shape,
                            dtype=ds_info.get('dtype',
                                              np.float32))
            mask = np.ma.make_mask_none(overall_shape)

            offset = 0
            for idx, fh in enumerate(file_handlers):
                granule_height = all_shapes[idx][0]
                # XXX: Does this work with masked arrays and subclasses of them?
                # Otherwise, have to send in separate data, mask, and info parameters to be filled in
                # TODO: Combine info in a sane way
                shuttle = Shuttle(data[offset:offset + granule_height],
                                  mask[offset:offset + granule_height],
                                  out_info)
                fh.get_dataset(dsid,
                               ds_info,
                               out=shuttle)
                offset += granule_height
            out_info.pop('area', None)
            proj = cls(data, mask=mask,
                       copy=False, **out_info)
        # FIXME: areas could be concatenated here
        # Update the metadata
        proj.info['start_time'] = file_handlers[0].start_time
        proj.info['end_time'] = file_handlers[-1].end_time

        return all_shapes, proj

    def _load_area(self, navid, file_handlers, nav_info, all_shapes, shape):
        try:
            area_defs = [fh.get_area_def(navid, nav_info)
                         for fh in file_handlers]
        except NotImplementedError:
            pass
        else:
            final_area = copy.deepcopy(area_defs[0])

            for area_def in area_defs[1:]:
                different_items = (set(final_area.proj_dict.items()) ^
                                   set(area_def.proj_dict.items()))
                if different_items:
                    break
                if (final_area.area_extent[0] == area_def.area_extent[0] and
                        final_area.area_extent[2] == area_def.area_extent[2] and
                        np.isclose(final_area.area_extent[1], area_def.area_extent[3])):
                    current_extent = list(final_area.area_extent)
                    current_extent[1] = area_def.area_extent[1]
                    final_area.area_extent = current_extent
                    final_area.y_size += area_def.y_size
                    try:
                        final_area.shape = (
                            final_area.y_size, final_area.x_size)
                    except AttributeError:
                        pass
                else:
                    break
            else:
                return final_area
        lons = np.ma.empty(shape, dtype=nav_info.get('dtype', np.float32))
        # overwrite single boolean 'False'
        lons.mask = np.zeros(shape, dtype=np.bool)
        lats = np.ma.empty(shape, dtype=nav_info.get('dtype', np.float32))
        lats.mask = np.zeros(shape, dtype=np.bool)
        offset = 0
        for idx, fh in enumerate(file_handlers):
            granule_height = all_shapes[idx][0]
            fh.get_lonlats(navid,
                           nav_info,
                           lon_out=lons[offset:offset + granule_height],
                           lat_out=lats[offset:offset + granule_height])
            offset += granule_height

        lons[lons < -180] = np.ma.masked
        lons[lons > 180] = np.ma.masked
        lats[lats < -90] = np.ma.masked
        lats[lats > 90] = np.ma.masked

        area = geometry.SwathDefinition(lons, lats)
        # FIXME: How do we name areas?
        area.name = navid.name
        area.info = getattr(area, 'info', {})
        area.info.update(nav_info)
        return area

    def _preferred_filetype(self, filetypes):
        if not isinstance(filetypes, list):
            filetypes = [filetypes]

        # look through the file types and use the first one that we have loaded
        for ft in filetypes:
            if ft in self.file_handlers:
                return ft
        return None

    def load(self, dataset_keys):
        loaded_navs = {}
        datasets = DatasetDict()

        for dataset_key in dataset_keys:
            dsid = self.get_dataset_key(dataset_key)
            ds_info = self.ids[dsid]

            # Get the file handler to load this dataset (list or single string)
            filetype = self._preferred_filetype(ds_info['file_type'])
            if filetype is None:
                LOG.warning(
                    "Required file type '{}' not found or loaded for '{}'".format(
                        ds_info['file_type'], dsid.name))
                continue
            file_handlers = self.file_handlers[filetype]

            all_shapes, proj = self._load_dataset(file_handlers, dsid, ds_info)
            datasets[dsid] = proj

            if isinstance(proj, Projectable) and ('area' not in proj.info or proj.info['area'] is None):
                # we need to load the area because the file handlers didn't
                navid = AreaID(ds_info.get('navigation'), dsid.resolution)
                if navid.name is None or navid.name not in self.config[
                        'navigations']:
                    try:
                        nav_filetype = filetype
                        navid = dsid
                        nav_info = ds_info
                        nav_fhs = self.file_handlers[nav_filetype]

                        ds_area = self._load_area(navid, nav_fhs, nav_info,
                                                  all_shapes, proj.shape)
                        loaded_navs[navid.name] = ds_area
                        proj.info["area"] = ds_area

                    except NotImplementedError as err:
                        # we don't know how to load navigation
                        LOG.warning(
                            "Can't load navigation for {}: {}".format(dsid, str(err)))
                elif navid.name in loaded_navs:
                    proj.info["area"] = loaded_navs[navid.name]
                else:
                    nav_info = self.config['navigations'][navid.name]
                    nav_filetype = self._preferred_filetype(nav_info['file_type'])
                    if nav_filetype is None:
                        raise RuntimeError(
                            "Required file type '{}' not found or loaded".format(
                                nav_info['file_type']))
                    nav_fhs = self.file_handlers[nav_filetype]

                    ds_area = self._load_area(navid, nav_fhs, nav_info,
                                              all_shapes, proj.shape)
                    loaded_navs[navid.name] = ds_area
                    proj.info["area"] = ds_area

        return datasets
