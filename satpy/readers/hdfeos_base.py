#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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
"""Base HDF-EOS reader."""

from __future__ import annotations

import re
import logging

from datetime import datetime
import xarray as xr
import numpy as np

from pyhdf.error import HDF4Error
from pyhdf.SD import SD

from satpy import CHUNK_SIZE, DataID
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


def interpolate(clons, clats, csatz, src_resolution, dst_resolution):
    """Interpolate two parallel datasets jointly."""
    if csatz is None:
        return _interpolate_no_angles(clons, clats, src_resolution, dst_resolution)
    return _interpolate_with_angles(clons, clats, csatz, src_resolution, dst_resolution)


def _interpolate_with_angles(clons, clats, csatz, src_resolution, dst_resolution):
    from geotiepoints.modisinterpolator import modis_1km_to_250m, modis_1km_to_500m, modis_5km_to_1km
    # (src_res, dst_res, is satz not None) -> interp function
    interpolation_functions = {
        (5000, 1000): modis_5km_to_1km,
        (1000, 500): modis_1km_to_500m,
        (1000, 250): modis_1km_to_250m
    }
    return _find_and_run_interpolation(interpolation_functions, src_resolution, dst_resolution,
                                       (clons, clats, csatz))


def _interpolate_no_angles(clons, clats, src_resolution, dst_resolution):
    interpolation_functions = {}

    try:
        from geotiepoints.simple_modis_interpolator import (
            modis_1km_to_250m as simple_1km_to_250m,
            modis_1km_to_500m as simple_1km_to_500m)
    except ImportError:
        raise NotImplementedError(
            f"Interpolation from {src_resolution}m to {dst_resolution}m "
            "without satellite zenith angle information is not "
            "implemented. Try updating your version of "
            "python-geotiepoints.")
    else:
        interpolation_functions[(1000, 500)] = simple_1km_to_500m
        interpolation_functions[(1000, 250)] = simple_1km_to_250m

    return _find_and_run_interpolation(interpolation_functions, src_resolution, dst_resolution,
                                       (clons, clats))


def _find_and_run_interpolation(interpolation_functions, src_resolution, dst_resolution, args):
    try:
        interpolation_function = interpolation_functions[(src_resolution, dst_resolution)]
    except KeyError:
        error_message = "Interpolation from {}m to {}m not implemented".format(
            src_resolution, dst_resolution)
        raise NotImplementedError(error_message)

    logger.debug("Interpolating from {} to {}".format(src_resolution, dst_resolution))
    return interpolation_function(*args)


class HDFEOSBaseFileReader(BaseFileHandler):
    """Base file handler for HDF EOS data for both L1b and L2 products."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the base reader."""
        BaseFileHandler.__init__(self, filename, filename_info, filetype_info)
        try:
            self.sd = SD(self.filename)
        except HDF4Error as err:
            error_message = "Could not load data from file {}: {}".format(self.filename, err)
            raise ValueError(error_message)

        self.metadata = self._load_all_metadata_attributes()

    def _load_all_metadata_attributes(self):
        metadata = {}
        attrs = self.sd.attributes()
        for md_key in ("CoreMetadata.0", "StructMetadata.0", "ArchiveMetadata.0"):
            try:
                str_val = attrs[md_key]
            except KeyError:
                continue
            else:
                metadata.update(self.read_mda(str_val))
        return metadata

    @staticmethod
    def read_mda(attribute):
        """Read the EOS metadata."""
        lines = attribute.split('\n')
        mda = {}
        current_dict = mda
        path = []
        prev_line = None
        for line in lines:
            if not line:
                continue
            if line == 'END':
                break
            if prev_line:
                line = prev_line + line
            key, val = line.split('=')
            key = key.strip()
            val = val.strip()
            try:
                val = eval(val)
            except NameError:
                pass
            except SyntaxError:
                prev_line = line
                continue
            prev_line = None
            if key in ['GROUP', 'OBJECT']:
                new_dict = {}
                path.append(val)
                current_dict[val] = new_dict
                current_dict = new_dict
            elif key in ['END_GROUP', 'END_OBJECT']:
                if val != path[-1]:
                    raise SyntaxError
                path = path[:-1]
                current_dict = mda
                for item in path:
                    current_dict = current_dict[item]
            elif key in ['CLASS', 'NUM_VAL']:
                pass
            else:
                current_dict[key] = val
        return mda

    @property
    def metadata_platform_name(self):
        """Platform name from the internal file metadata."""
        try:
            # Example: 'Terra' or 'Aqua'
            return self.metadata['INVENTORYMETADATA']['ASSOCIATEDPLATFORMINSTRUMENTSENSOR'][
                'ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER']['ASSOCIATEDPLATFORMSHORTNAME']['VALUE']
        except KeyError:
            return self._platform_name_from_filename()

    def _platform_name_from_filename(self):
        platform_indicator = self.filename_info["platform_indicator"]
        if platform_indicator in ("t", "O"):
            # t1.* or MOD*
            return "Terra"
        # a1.* or MYD*
        return "Aqua"

    @property
    def start_time(self):
        """Get the start time of the dataset."""
        try:
            date = (self.metadata['INVENTORYMETADATA']['RANGEDATETIME']['RANGEBEGINNINGDATE']['VALUE'] + ' ' +
                    self.metadata['INVENTORYMETADATA']['RANGEDATETIME']['RANGEBEGINNINGTIME']['VALUE'])
            return datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')
        except KeyError:
            return self._start_time_from_filename()

    def _start_time_from_filename(self):
        for fn_key in ("start_time", "acquisition_time"):
            if fn_key in self.filename_info:
                return self.filename_info[fn_key]
        raise RuntimeError("Could not determine file start time")

    @property
    def end_time(self):
        """Get the end time of the dataset."""
        try:
            date = (self.metadata['INVENTORYMETADATA']['RANGEDATETIME']['RANGEENDINGDATE']['VALUE'] + ' ' +
                    self.metadata['INVENTORYMETADATA']['RANGEDATETIME']['RANGEENDINGTIME']['VALUE'])
            return datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')
        except KeyError:
            return self.start_time

    def _read_dataset_in_file(self, dataset_name):
        if dataset_name not in self.sd.datasets():
            error_message = "Dataset name {} not included in available datasets {}".format(
                dataset_name, self.sd.datasets()
            )
            raise KeyError(error_message)

        dataset = self.sd.select(dataset_name)
        return dataset

    def load_dataset(self, dataset_name, is_category=False):
        """Load the dataset from HDF EOS file."""
        from satpy.readers.hdf4_utils import from_sds

        dataset = self._read_dataset_in_file(dataset_name)
        dask_arr = from_sds(dataset, chunks=CHUNK_SIZE)
        dims = ('y', 'x') if dask_arr.ndim == 2 else None
        data = xr.DataArray(dask_arr, dims=dims,
                            attrs=dataset.attributes())
        data = self._scale_and_mask_data_array(data, is_category=is_category)

        return data

    def _scale_and_mask_data_array(self, data, is_category=False):
        good_mask, new_fill = self._get_good_data_mask(data, is_category=is_category)
        scale_factor = data.attrs.pop('scale_factor', None)
        add_offset = data.attrs.pop('add_offset', None)
        # don't scale category products, even though scale_factor may equal 1
        # we still need to convert integers to floats
        if scale_factor is not None and not is_category:
            data = data * np.float32(scale_factor)
            if add_offset is not None and add_offset != 0:
                data = data + add_offset

        if good_mask is not None:
            data = data.where(good_mask, new_fill)
        return data

    def _get_good_data_mask(self, data_arr, is_category=False):
        try:
            fill_value = data_arr.attrs["_FillValue"]
        except KeyError:
            return None, None

        # preserve integer data types if possible
        if is_category and np.issubdtype(data_arr.dtype, np.integer):
            # no need to mask, the fill value is already what it needs to be
            return None, None
        new_fill = np.nan
        data_arr.attrs.pop('_FillValue', None)
        good_mask = data_arr != fill_value
        return good_mask, new_fill

    def _add_satpy_metadata(self, data_id: DataID, data_arr: xr.DataArray):
        """Add metadata that is specific to Satpy."""
        new_attrs = {
            'platform_name': 'EOS-' + self.metadata_platform_name,
            'sensor': 'modis',
        }

        res = data_id["resolution"]
        rps = self._resolution_to_rows_per_scan(res)
        new_attrs["rows_per_scan"] = rps

        data_arr.attrs.update(new_attrs)

    def _resolution_to_rows_per_scan(self, resolution: int) -> int:
        known_rps = {
            5000: 2,
            1000: 10,
            500: 20,
            250: 40,
        }
        return known_rps.get(resolution, 10)


class HDFEOSGeoReader(HDFEOSBaseFileReader):
    """Handler for the geographical datasets."""

    # list of geographical datasets handled by the georeader
    # mapping to the default variable name if not specified in YAML
    DATASET_NAMES = {
        'longitude': 'Longitude',
        'latitude': 'Latitude',
        'satellite_azimuth_angle': ('SensorAzimuth', 'Sensor_Azimuth'),
        'satellite_zenith_angle': ('SensorZenith', 'Sensor_Zenith'),
        'solar_azimuth_angle': ('SolarAzimuth', 'SolarAzimuth'),
        'solar_zenith_angle': ('SolarZenith', 'Solar_Zenith'),
    }

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the geographical reader."""
        HDFEOSBaseFileReader.__init__(self, filename, filename_info, filetype_info)
        self.cache = {}

    @staticmethod
    def is_geo_loadable_dataset(dataset_name: str) -> bool:
        """Determine if this dataset should be loaded as a Geo dataset."""
        return dataset_name in HDFEOSGeoReader.DATASET_NAMES

    @staticmethod
    def read_geo_resolution(metadata):
        """Parse metadata to find the geolocation resolution."""
        # level 1 files
        try:
            return HDFEOSGeoReader._geo_resolution_for_l1b(metadata)
        except KeyError:
            try:
                return HDFEOSGeoReader._geo_resolution_for_l2_l1b(metadata)
            except (AttributeError, KeyError):
                raise RuntimeError("Could not determine resolution from file metadata")

    @staticmethod
    def _geo_resolution_for_l1b(metadata):
        ds = metadata['INVENTORYMETADATA']['COLLECTIONDESCRIPTIONCLASS']['SHORTNAME']['VALUE']
        if ds.endswith('D03') or ds.endswith('HKM') or ds.endswith('QKM'):
            return 1000
        # 1km files have 5km geolocation usually
        return 5000

    @staticmethod
    def _geo_resolution_for_l2_l1b(metadata):
        # data files probably have this level 2 files
        # this does not work for L1B 1KM data files because they are listed
        # as 1KM data but the geo data inside is at 5km
        latitude_dim = metadata['SwathStructure']['SWATH_1']['DimensionMap']['DimensionMap_2']['GeoDimension']
        resolution_regex = re.compile(r'(?P<resolution>\d+)(km|KM)')
        resolution_match = resolution_regex.search(latitude_dim)
        return int(resolution_match.group('resolution')) * 1000

    @property
    def geo_resolution(self):
        """Resolution of the geographical data retrieved in the metadata."""
        return self.read_geo_resolution(self.metadata)

    def _load_ds_by_name(self, ds_name):
        """Attempt loading using multiple common names."""
        var_names = self.DATASET_NAMES[ds_name]
        if isinstance(var_names, (list, tuple)):
            try:
                return self.load_dataset(var_names[0])
            except KeyError:
                return self.load_dataset(var_names[1])
        return self.load_dataset(var_names)

    def get_interpolated_dataset(self, name1, name2, resolution, offset=0):
        """Load and interpolate datasets."""
        try:
            result1 = self.cache[(name1, resolution)]
            result2 = self.cache[(name2, resolution)]
        except KeyError:
            result1 = self._load_ds_by_name(name1)
            result2 = self._load_ds_by_name(name2) - offset
            try:
                sensor_zenith = self._load_ds_by_name('satellite_zenith_angle')
            except KeyError:
                # no sensor zenith angle, do "simple" interpolation
                sensor_zenith = None

            result1, result2 = interpolate(
                result1, result2, sensor_zenith,
                self.geo_resolution, resolution
            )
            self.cache[(name1, resolution)] = result1
            self.cache[(name2, resolution)] = result2 + offset

    def get_dataset(self, dataset_id: DataID, dataset_info: dict) -> xr.DataArray:
        """Get the geolocation dataset."""
        # Name of the dataset as it appears in the HDF EOS file
        in_file_dataset_name = dataset_info.get('file_key')
        # Name of the dataset in the YAML file
        dataset_name = dataset_id['name']
        # Resolution asked
        resolution = dataset_id['resolution']
        if in_file_dataset_name is not None:
            # if the YAML was configured with a specific name use that
            data = self.load_dataset(in_file_dataset_name)
        else:
            # otherwise use the default name for this variable
            data = self._load_ds_by_name(dataset_name)
        if resolution != self.geo_resolution:
            if in_file_dataset_name is not None:
                # they specified a custom variable name but
                # we don't know how to interpolate this yet
                raise NotImplementedError(
                    "Interpolation for variable '{}' is not "
                    "configured".format(dataset_name))

            # The data must be interpolated
            logger.debug("Loading %s", dataset_name)
            if dataset_name in ['longitude', 'latitude']:
                self.get_interpolated_dataset('longitude', 'latitude',
                                              resolution)
            elif dataset_name in ['satellite_azimuth_angle', 'satellite_zenith_angle']:
                # Sensor dataset names differs between L1b and L2 products
                self.get_interpolated_dataset('satellite_azimuth_angle', 'satellite_zenith_angle',
                                              resolution, offset=90)
            elif dataset_name in ['solar_azimuth_angle', 'solar_zenith_angle']:
                # Sensor dataset names differs between L1b and L2 products
                self.get_interpolated_dataset('solar_azimuth_angle', 'solar_zenith_angle',
                                              resolution, offset=90)

            data = self.cache[dataset_name, resolution]

        for key in ('standard_name', 'units'):
            if key in dataset_info:
                data.attrs[key] = dataset_info[key]
        self._add_satpy_metadata(dataset_id, data)

        return data
