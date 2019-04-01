#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019

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
import logging
from functools import lru_cache

from datetime import datetime
import xarray as xr
import numpy as np

from pyhdf.error import HDF4Error
from pyhdf.SD import SD

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


class HDFEOSBaseFileReader(BaseFileHandler):
    """Base file handler for HDF EOS data for both L1b and L2 products. """
    def __init__(self, filename, filename_info, filetype_info):
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
        date = (self.metadata['INVENTORYMETADATA']['RANGEDATETIME']['RANGEBEGINNINGDATE']['VALUE'] + ' ' +
                self.metadata['INVENTORYMETADATA']['RANGEDATETIME']['RANGEBEGINNINGTIME']['VALUE'])
        return datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')

    @property
    def end_time(self):
        date = (self.metadata['INVENTORYMETADATA']['RANGEDATETIME']['RANGEENDINGDATE']['VALUE'] + ' ' +
                self.metadata['INVENTORYMETADATA']['RANGEDATETIME']['RANGEENDINGTIME']['VALUE'])
        return datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')

    @lru_cache(32)
    def _read_dataset_in_file(self, dataset_name):
        if dataset_name not in self.sd.datasets():
            error_message = "Dataset name {} not included in available datasets {}".format(
                dataset_name, self.sd.datasets()
            )
            raise KeyError(error_message)

        dataset = self.sd.select(dataset_name)
        return dataset

    def load_dataset(self, dataset_name):
        """Load the dataset from HDF EOS file. """
        from satpy.readers.hdf4_utils import from_sds

        dataset = self._read_dataset_in_file(dataset_name)
        fill_value = dataset._FillValue
        scale_factor = np.float32(dataset.scale_factor)
        data = xr.DataArray(from_sds(dataset, chunks=CHUNK_SIZE),
                            dims=['y', 'x']).astype(np.float32)
        data_mask = data.where(data != fill_value)
        data = data_mask * scale_factor
        return data


class HDFEOSGeoReader(HDFEOSBaseFileReader):
    """Handler for the geographical datasets. """

    # list of geographical datasets handled by the georeader
    DATASET_NAMES = ['longitude', 'latitude',
                     'satellite_azimuth_angle', 'satellite_zenith_angle',
                     'solar_azimuth_angle', 'solar_zenith_angle']

    def __init__(self, filename, filename_info, filetype_info):
        HDFEOSBaseFileReader.__init__(self, filename, filename_info, filetype_info)

    @staticmethod
    def read_geo_resolution(metadata):
        """Parses metada to find the geolocation resolution.
        It is implemented as a staticmethod to match read_mda pattern.

        """
        import re
        try:
            latitude_dim = metadata['SwathStructure']['SWATH_1']['DimensionMap']['DimensionMap_2']['GeoDimension']
        except KeyError as e:
            logger.debug("Resolution not found in metadata: {}".format(e))
            return None
        resolution_regex = re.compile(r'(?P<resolution>\d+)(km|KM)')
        resolution_match = resolution_regex.search(latitude_dim)
        return int(resolution_match.group('resolution')) * 1000

    @property
    def geo_resolution(self):
        """Resolution of the geographical data retrieved in the metada. """
        return self.read_geo_resolution(self.metadata)

    def get_dataset(self, dataset_keys, dataset_info):
        """Get the geolocation dataset."""
        # Name of the dataset as it appears in the HDF EOS file
        in_file_dataset_name = dataset_info['file_key']
        # Name of the dataset in the YAML file
        dataset_name = dataset_keys.name
        # Resolution asked
        resolution = dataset_keys.resolution

        data = self.load_dataset(in_file_dataset_name)

        if resolution != self.geo_resolution:

            # The data must be interpolated
            interpolated_dataset = {}

            def interpolate(clons, clats, csatz):
                from geotiepoints.modisinterpolator import modis_1km_to_250m, modis_1km_to_500m, modis_5km_to_1km

                interpolation_functions = {
                    (5000, 1000): modis_5km_to_1km,
                    (1000, 500): modis_1km_to_500m,
                    (1000, 250): modis_1km_to_250m
                }

                try:
                    interpolation_function = interpolation_functions[(self.geo_resolution, resolution)]
                except KeyError:
                    error_message = "Interpolation from {}m to {}m not implemented".format(
                        self.geo_resolution, resolution)
                    raise NotImplementedError(error_message)

                logger.debug("Interpolating from {} to {}".format(self.geo_resolution, resolution))

                return interpolation_function(clons, clats, csatz)

            # Sensor zenith dataset name differs between L1b and L2 products
            sensor_zentih = None
            try:
                sensor_zenith = self.load_dataset('SensorZenith')
            except KeyError:
                sensor_zenith = self.load_dataset('Sensor_Zenith')

            if dataset_name in ['longitude', 'latitude']:
                latitude = self.load_dataset('Longitude')
                longitude = self.load_dataset('Latitude')
                longitude, latitude = interpolate(
                    longitude, latitude, sensor_zenith
                )
                interpolated_dataset['longitude'] = longitude
                interpolated_dataset['latitude'] = latitude

            # Warning: Are these interpolations originally correct?
            # Does geotiepoints actually interpolate azimuth coordinates?
            else:
                if dataset_name in ['satellite_azimuth_angle', 'satellite_zenith_angle']:
                    sensor_azimuth_a = self.load_dataset('SensorAzimuth')
                    sensor_azimuth_b = self.load_dataset('SensorZenith') - 90
                    sensor_azimuth_a, sensor_azimuth_b = interpolate(
                        sensor_azimuth_a, sensor_azimuth_b, sensor_zenith
                    )
                    interpolated_dataset['satellite_azimuth_angle'] = sensor_azimuth_a
                    interpolated_dataset['satellite_zentih_angle'] = sensor_azimuth_b + 90

                elif dataset_name in ['solar_azimuth_angle', 'solar_zenith_angle']:
                    solar_azimuth_a = self.load_dataset('SolarAzimuth')
                    solar_azimuth_b = self.load_dataset('SolarZenith') - 90
                    solar_azimuth_a, solar_azimuth_b = interpolate(
                        solar_azimuth_a, solar_azimuth_b, sensor_zentih
                    )
                    interpolated_dataset['solar_azimuth_angle'] = solar_azimuth_a
                    interpolated_dataset['solar_zentih_angle'] = solar_azimuth_b + 90

            data = interpolated_dataset[dataset_name]

        return data
