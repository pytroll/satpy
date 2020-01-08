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

import re
import logging

from datetime import datetime
import xarray as xr
import numpy as np

from pyhdf.error import HDF4Error
from pyhdf.SD import SD

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


def interpolate(clons, clats, csatz, src_resolution, dst_resolution):
    """Interpolate two parallel datasets jointly."""
    from geotiepoints.modisinterpolator import modis_1km_to_250m, modis_1km_to_500m, modis_5km_to_1km

    interpolation_functions = {
        (5000, 1000): modis_5km_to_1km,
        (1000, 500): modis_1km_to_500m,
        (1000, 250): modis_1km_to_250m
    }

    try:
        interpolation_function = interpolation_functions[(src_resolution, dst_resolution)]
    except KeyError:
        error_message = "Interpolation from {}m to {}m not implemented".format(
            src_resolution, dst_resolution)
        raise NotImplementedError(error_message)

    logger.debug("Interpolating from {} to {}".format(src_resolution, dst_resolution))

    return interpolation_function(clons, clats, csatz)


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

        # Read metadata
        self.metadata = self.read_mda(self.sd.attributes()['CoreMetadata.0'])
        self.metadata.update(self.read_mda(
            self.sd.attributes()['StructMetadata.0'])
        )
        self.metadata.update(self.read_mda(
            self.sd.attributes()['ArchiveMetadata.0'])
        )

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
    def start_time(self):
        """Get the start time of the dataset."""
        date = (self.metadata['INVENTORYMETADATA']['RANGEDATETIME']['RANGEBEGINNINGDATE']['VALUE'] + ' ' +
                self.metadata['INVENTORYMETADATA']['RANGEDATETIME']['RANGEBEGINNINGTIME']['VALUE'])
        return datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')

    @property
    def end_time(self):
        """Get the end time of the dataset."""
        date = (self.metadata['INVENTORYMETADATA']['RANGEDATETIME']['RANGEENDINGDATE']['VALUE'] + ' ' +
                self.metadata['INVENTORYMETADATA']['RANGEDATETIME']['RANGEENDINGTIME']['VALUE'])
        return datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')

    def _read_dataset_in_file(self, dataset_name):
        if dataset_name not in self.sd.datasets():
            error_message = "Dataset name {} not included in available datasets {}".format(
                dataset_name, self.sd.datasets()
            )
            raise KeyError(error_message)

        dataset = self.sd.select(dataset_name)
        return dataset

    def load_dataset(self, dataset_name):
        """Load the dataset from HDF EOS file."""
        from satpy.readers.hdf4_utils import from_sds

        dataset = self._read_dataset_in_file(dataset_name)
        fill_value = dataset._FillValue
        dask_arr = from_sds(dataset, chunks=CHUNK_SIZE)
        dims = ('y', 'x') if dask_arr.ndim == 2 else None
        data = xr.DataArray(dask_arr, dims=dims,
                            attrs=dataset.attributes())

        # preserve integer data types if possible
        if np.issubdtype(data.dtype, np.integer):
            new_fill = fill_value
        else:
            new_fill = np.nan
            data.attrs.pop('_FillValue', None)
        good_mask = data != fill_value

        scale_factor = data.attrs.get('scale_factor')
        if scale_factor is not None:
            data = data * scale_factor

        data = data.where(good_mask, new_fill)
        return data


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
    def read_geo_resolution(metadata):
        """Parse metadata to find the geolocation resolution.

        It is implemented as a staticmethod to match read_mda pattern.

        """
        # level 1 files
        try:
            ds = metadata['INVENTORYMETADATA']['COLLECTIONDESCRIPTIONCLASS']['SHORTNAME']['VALUE']
            if ds.endswith('D03'):
                return 1000
            else:
                # 1km files have 5km geolocation usually
                return 5000
        except KeyError:
            pass

        # data files probably have this level 2 files
        # this does not work for L1B 1KM data files because they are listed
        # as 1KM data but the geo data inside is at 5km
        try:
            latitude_dim = metadata['SwathStructure']['SWATH_1']['DimensionMap']['DimensionMap_2']['GeoDimension']
            resolution_regex = re.compile(r'(?P<resolution>\d+)(km|KM)')
            resolution_match = resolution_regex.search(latitude_dim)
            return int(resolution_match.group('resolution')) * 1000
        except (AttributeError, KeyError):
            pass

        raise RuntimeError("Could not determine resolution from file metadata")

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

    def get_interpolated_dataset(self, name1, name2, resolution, sensor_zenith, offset=0):
        """Load and interpolate datasets."""
        try:
            result1 = self.cache[(name1, resolution)]
            result2 = self.cache[(name2, resolution)]
        except KeyError:
            result1 = self._load_ds_by_name(name1)
            result2 = self._load_ds_by_name(name2) - offset
            result1, result2 = interpolate(
                result1, result2, sensor_zenith,
                self.geo_resolution, resolution
            )
            self.cache[(name1, resolution)] = result1
            self.cache[(name2, resolution)] = result2 + offset

    def get_dataset(self, dataset_keys, dataset_info):
        """Get the geolocation dataset."""
        # Name of the dataset as it appears in the HDF EOS file
        in_file_dataset_name = dataset_info.get('file_key')
        # Name of the dataset in the YAML file
        dataset_name = dataset_keys.name
        # Resolution asked
        resolution = dataset_keys.resolution
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
            sensor_zenith = self._load_ds_by_name('satellite_zenith_angle')
            logger.debug("Loading %s", dataset_name)
            if dataset_name in ['longitude', 'latitude']:
                self.get_interpolated_dataset('longitude', 'latitude',
                                              resolution, sensor_zenith)
            elif dataset_name in ['satellite_azimuth_angle', 'satellite_zenith_angle']:
                # Sensor dataset names differs between L1b and L2 products
                self.get_interpolated_dataset('satellite_azimuth_angle', 'satellite_zenith_angle',
                                              resolution, sensor_zenith, offset=90)
            elif dataset_name in ['solar_azimuth_angle', 'solar_zenith_angle']:
                # Sensor dataset names differs between L1b and L2 products
                self.get_interpolated_dataset('solar_azimuth_angle', 'solar_zenith_angle',
                                              resolution, sensor_zenith, offset=90)

            data = self.cache[dataset_name, resolution]

        for key in ('standard_name', 'units'):
            if key in dataset_info:
                data.attrs[key] = dataset_info[key]

        return data
