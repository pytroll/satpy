#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017.
#
# Author(s):
#
#   David Hoese <david.hoese@ssec.wisc.edu>
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
"""Interface to CLAVR-X HDF4 products.
"""
from datetime import datetime, timedelta
import numpy as np
import logging
from collections import defaultdict

from satpy.dataset import DatasetID, Dataset
from satpy.readers.yaml_reader import FileYAMLReader
from satpy.readers.hdf4_utils import HDF4FileHandler, SDS

LOG = logging.getLogger(__name__)


CF_UNITS = {
    'none': '1',
}


class CLAVRXFileHandler(HDF4FileHandler):
    sensors = {
        'MODIS': 'modis',
        'VIIRS': 'viirs',
        'AVHRR': 'avhrr',
    }
    platforms = {
        'SNPP': 'npp',
    }
    rows_per_scan = {
        'viirs': 16,
        'modis': 10,
    }
    nadir_resolution = {
        'viirs': 742,
        'modis': 1000,
        'avhrr': 1050,
    }

    def get_sensor(self, sensor):
        for k, v in self.sensors.items():
            if k in sensor:
                return v
        raise ValueError("Unknown sensor '{}'".format(sensor))

    def get_platform(self, platform):
        for k, v in self.platforms.items():
            if k in platform:
                return v
        return platform

    def get_rows_per_scan(self, sensor):
        for k, v in self.rows_per_scan.items():
            if sensor.startswith(k):
                return v

    def get_nadir_resolution(self, sensor):
        for k, v in self.nadir_resolution.items():
            if sensor.startswith(k):
                return v
        res = self.filename_info.get('resolution')
        if res.endswith('m'):
            return int(res[:-1])
        elif res is not None:
            return int(res)

    @property
    def start_time(self):
        return self.filename_info['start_time']

    @property
    def end_time(self):
        return self.filename_info.get('end_time', self.start_time)

    def available_dataset_ids(self):
        """Automatically determine datasets provided by this file"""
        sensor = self.get_sensor(self['/attr/sensor'])
        nadir_resolution = self.get_nadir_resolution(sensor)
        for var_name, val in self.file_content.items():
            if isinstance(val, SDS):
                ds_info = {
                    'file_type': self.filetype_info['file_type'],
                    'coordinates': ['longitude', 'latitude'],
                    'resolution': nadir_resolution,
                }
                yield DatasetID(name=var_name, resolution=nadir_resolution), ds_info

    def get_shape(self, dataset_id, ds_info):
        var_name = ds_info.get('file_key', dataset_id.name)
        return self[var_name + '/shape']

    def get_data_type(self, dataset_id, ds_info):
        base_default = super(CLAVRXFileHandler, self).get_data_type(
            dataset_id, ds_info)
        var_name = ds_info.get('file_key', dataset_id.name)
        if self.get(var_name + '/attr/SCALED', 1) or self.get(var_name + '/attr/flag_meanings'):
            return self.get(var_name + '/attr/dtype', base_default)
        return base_default

    def get_metadata(self, dataset_id, ds_info):
        var_name = ds_info.get('file_key', dataset_id.name)
        i = {}
        i.update(ds_info)
        for a in ['standard_name', 'units', 'long_name', 'actual_range', 'flag_meanings', 'flag_values', 'flag_masks']:
            attr_path = var_name + '/attr/' + a
            if attr_path in self:
                i[a] = self[attr_path]

        flag_meanings = i.get('flag_meanings')
        if not self.get(var_name + '/attr/SCALED', 1) and not flag_meanings:
            i['flag_meanings'] = '<flag_meanings_unknown>'
            i.setdefault('flag_values', [None])

        u = i.get('units')
        if u in CF_UNITS:
            # CF compliance
            i['units'] = CF_UNITS[u]

        i['sensor'] = self.get_sensor(self['/attr/sensor'])
        i['platform'] = self.get_platform(self['/attr/platform'])
        i['resolution'] = dataset_id.resolution or self.get_nadir_resolution(i['sensor'])
        i['rows_per_scan'] = self.get_rows_per_scan(i['sensor'])
        i['reader'] = 'clavrx'

        return i

    def get_dataset(self, dataset_id, ds_info, out=None,
                    xslice=slice(None), yslice=slice(None)):
        var_name = ds_info.get('file_key', dataset_id.name)
        # FUTURE: Metadata retrieval may be separate
        i = self.get_metadata(dataset_id, ds_info)
        data = self[var_name][yslice, xslice]
        fill = self[var_name + '/attr/_FillValue']
        factor = self.get(var_name + '/attr/scale_factor')
        offset = self.get(var_name + '/attr/add_offset')
        valid_range = self.get(var_name + '/attr/valid_range')

        mask = data == fill
        if valid_range is not None:
            mask |= (data < valid_range[0]) | (data > valid_range[1])
        data = data.astype(out.data.dtype)
        if factor is not None and offset is not None:
            data *= factor
            data += offset

        out.data[:] = data
        out.mask[:] |= mask
        out.info.update(i)
        return out


class CLAVRXYAMLReader(FileYAMLReader):
    def create_filehandlers(self, filenames):
        super(CLAVRXYAMLReader, self).create_filehandlers(filenames)
        self.load_ds_ids_from_files()

    def load_ds_ids_from_files(self):
        for file_handlers in self.file_handlers.values():
            fh = file_handlers[0]
            for ds_id, ds_info in fh.available_dataset_ids():
                # don't overwrite an existing dataset
                # especially from the yaml config
                self.ids.setdefault(ds_id, ds_info)

