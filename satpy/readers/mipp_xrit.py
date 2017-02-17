#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010-2016.

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Esben S. Nielsen <esn@dmi.dk>
#   Panu Lahtinen <panu.lahtinen@fmi.fi>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

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
"""Interface to Eumetcast level 1.5 HRIT/LRIT format. Uses the MIPP reader.
"""
import logging
import os
from fnmatch import fnmatch

from mipp import CalibrationError, ReaderError, xrit
from satpy.dataset import Dataset
from satpy.readers import DatasetDict
from satpy.readers.helper_functions import area_defs_to_extent
from satpy.readers.yaml_reader import AbstractYAMLReader
from trollsift.parser import globify, parse

LOGGER = logging.getLogger(__name__)

try:
    # Work around for on demand import of pyresample. pyresample depends
    # on scipy.spatial which memory leaks on multiple imports
    IS_PYRESAMPLE_LOADED = False
    from pyresample import geometry
    IS_PYRESAMPLE_LOADED = True
except ImportError:
    LOGGER.warning("pyresample missing. Can only work in satellite projection")


class xRITFile(AbstractYAMLReader):
    '''Class for reading XRIT data.
    '''

    def __init__(self, config_files,
                 start_time=None,
                 end_time=None,
                 area=None):
        super(xRITFile, self).__init__(config_files,
                                       start_time=start_time,
                                       end_time=end_time,
                                       area=area)
        self.info['filenames'] = []
        self.file_patterns = []
        for file_type in self.config['file_types'].values():
            self.file_patterns.extend(file_type['file_patterns'])

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    def load(self, dataset_keys, area=None, start_time=None, end_time=None):
        image_files = []
        pattern = self.file_patterns[0]
        prologue_file = None
        epilogue_file = None
        for filename in self.info['filenames']:
            try:
                file_info = parse(pattern, os.path.basename(filename))
            except ValueError:
                continue
            if file_info["segment"] == "EPI":
                epilogue_file = filename
            elif file_info["segment"] == "PRO":
                prologue_file = filename
            else:
                image_files.append(filename)

        start_times = set()
        datasets = DatasetDict()
        area_converted_to_extent = False
        area_extent = None
        for ds in dataset_keys:

            channel_files = []
            for filename in image_files:
                file_info = parse(pattern, os.path.basename(filename))
                if file_info["dataset_name"] == ds.name:
                    channel_files.append(filename)
                start_times.add(file_info['start_time'])

            if not channel_files:
                continue
            kwargs = {}
            if 'platform_name' in self.info:
                kwargs['platform_name'] = self.info['platform_name']
            # Convert area definitions to maximal area_extent
            if not area_converted_to_extent and area is not None:
                metadata = xrit.sat.load_files(prologue_file,
                                               channel_files,
                                               epilogue_file,
                                               only_metadata=True,
                                               **kwargs)
                # otherwise use the default value (MSG3 extent at
                # lon0=0.0), that is, do not pass default_extent=area_extent
                area_extent = area_defs_to_extent(
                    [area], metadata.proj4_params)
                area_converted_to_extent = True

            try:
                calibrate = 1
                if ds.calibration == 'counts':
                    calibrate = 0
                elif ds.calibration == 'radiance':
                    calibrate = 2
                image = xrit.sat.load_files(prologue_file,
                                            channel_files,
                                            epilogue_file,
                                            mask=True,
                                            calibrate=calibrate,
                                            **kwargs)
                if area_extent:
                    metadata, data = image(area_extent)
                else:
                    metadata, data = image()
            except CalibrationError:
                LOGGER.warning(
                    "Loading non calibrated data since calibration failed.")
                image = xrit.sat.load_files(prologue_file,
                                            channel_files,
                                            epilogue_file,
                                            mask=True,
                                            calibrate=False,
                                            **kwargs)
                if area_extent:
                    metadata, data = image(area_extent)
                else:
                    metadata, data = image()

            except ReaderError as err:
                # if dataset can't be found, go on with next dataset
                LOGGER.error(str(err))
                continue
            if len(metadata.instruments) != 1:
                sensor = None
            else:
                sensor = metadata.instruments[0]

            units = {'ALBEDO(%)': '%',
                     'KELVIN': 'K'}

            standard_names = {'1': 'counts',
                              'W m-2 sr-1 m-1':
                              'toa_outgoing_radiance_per_unit_wavelength',
                              '%': 'toa_bidirectional_reflectance',
                              'K':
                              'toa_brightness_temperature'}

            unit = units.get(metadata.calibration_unit,
                             metadata.calibration_unit)
            projectable = Dataset(
                data,
                name=ds.name,
                units=unit,
                standard_name=standard_names[unit],
                sensor=sensor,
                start_time=min(start_times),
                id=ds)

            # Build an area on the fly from the mipp metadata
            proj_params = getattr(metadata, "proj4_params").split(" ")
            proj_dict = {}
            for param in proj_params:
                key, val = param.split("=")
                proj_dict[key] = val

            if IS_PYRESAMPLE_LOADED:
                # Build area_def on-the-fly
                projectable.info["area"] = geometry.AreaDefinition(
                    str(metadata.area_extent) + str(data.shape),
                    "On-the-fly area", proj_dict["proj"], proj_dict,
                    data.shape[1], data.shape[0], metadata.area_extent)
            else:
                LOGGER.info("Could not build area, pyresample missing...")

            datasets[ds] = projectable

        return datasets

    def select_files(self,
                     base_dir=None,
                     filenames=None,
                     sensor=None):
        file_set, info_filenames = super(xRITFile, self).select_files(
            base_dir, filenames, sensor)

        # for pattern in self.file_patterns:
        #    for filename in filenames:
        #        parse(pattern, os.path.basename(filename))

        matching_filenames = []

        # Organize filenames in to file types and create file handlers
        remaining_filenames = set(self.info['filenames'])
        start_times = []
        end_times = []
        for filetype, filetype_info in self.config['file_types'].items():
            patterns = filetype_info['file_patterns']
            for pattern in patterns:
                used_filenames = set()
                for filename in remaining_filenames:
                    if fnmatch(os.path.basename(filename), globify(pattern)):
                        # we know how to use this file (even if we may not use
                        # it later)
                        used_filenames.add(filename)
                        filename_info = parse(pattern,
                                              os.path.basename(filename))
                        # Only add this file handler if it is within the time
                        # we want
                        file_start = filename_info['start_time']
                        file_end = filename_info.get('end_time', file_start)
                        if self._start_time and file_start < self._start_time:
                            continue
                        if self._end_time and file_end > self._end_time:
                            continue

                        start_times.append(file_start)
                        end_times.append(file_end)
                        matching_filenames.append(filename)
                        # TODO: Area filtering

                remaining_filenames -= used_filenames

        if matching_filenames:
            # Assign the start time and end time
            self._start_time = min(start_times)
            self._end_time = max(end_times)
        self.info['filenames'] = matching_filenames
        return file_set, info_filenames
