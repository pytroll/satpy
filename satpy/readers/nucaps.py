#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016.

# Author(s):

#
#   David Hoese <david.hoese@ssec.wisc.edu>
#

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

"""Interface to NUCAPS Retrieval NetCDF files

"""
import os.path
from datetime import datetime, timedelta
import numpy as np
import h5py
import logging
from collections import defaultdict

from satpy.dataset import Dataset
from satpy.readers.yaml_reader import FileYAMLReader
from satpy.readers.netcdf_utils import NetCDF4FileHandler

NO_DATE = datetime(1958, 1, 1)
EPSILON_TIME = timedelta(days=2)
LOG = logging.getLogger(__name__)

# It's difficult to do processing without knowing the pressure levels beforehand
ALL_PRESSURE_LEVELS = [
    0.0161, 0.0384, 0.0769, 0.137, 0.2244, 0.3454, 0.5064, 0.714,
    0.9753, 1.2972, 1.6872, 2.1526, 2.7009, 3.3398, 4.077, 4.9204,
    5.8776, 6.9567, 8.1655, 9.5119, 11.0038, 12.6492, 14.4559, 16.4318,
    18.5847, 20.9224, 23.4526, 26.1829, 29.121, 32.2744, 35.6505,
    39.2566, 43.1001, 47.1882, 51.5278, 56.126, 60.9895, 66.1253,
    71.5398, 77.2396, 83.231, 89.5204, 96.1138, 103.017, 110.237,
    117.777, 125.646, 133.846, 142.385, 151.266, 160.496, 170.078,
    180.018, 190.32, 200.989, 212.028, 223.441, 235.234, 247.408,
    259.969, 272.919, 286.262, 300, 314.137, 328.675, 343.618, 358.966,
    374.724, 390.893, 407.474, 424.47, 441.882, 459.712, 477.961,
    496.63, 515.72, 535.232, 555.167, 575.525, 596.306, 617.511, 639.14,
    661.192, 683.667, 706.565, 729.886, 753.628, 777.79, 802.371,
    827.371, 852.788, 878.62, 904.866, 931.524, 958.591, 986.067,
    1013.95, 1042.23, 1070.92, 1100
]


class NUCAPSFileHandler(NetCDF4FileHandler):
    """NUCAPS File Reader
    """

    def __contains__(self, item):
        return item in self.file_content

    def _parse_datetime(self, datestr):
        """Parse NUCAPS datetime string.
        """
        return datetime.strptime(datestr, "%Y-%m-%dT%H:%M:%S.%fZ")

    @property
    def start_time(self):
        return self._parse_datetime(self['/attr/time_coverage_start'])

    @property
    def end_time(self):
        return self._parse_datetime(self['/attr/time_coverage_end'])

    @property
    def start_orbit_number(self):
        """Return orbit number for the beginning of the swath.
        """
        return int(self['/attr/start_orbit_number'])

    @property
    def end_orbit_number(self):
        """Return orbit number for the end of the swath.
        """
        return int(self['/attr/end_orbit_number'])

    @property
    def platform_name(self):
        """Return standard platform name for the file's data.
        """
        res = self['/attr/platform_name']
        if isinstance(res, np.ndarray):
            return str(res.astype(str))
        else:
            return res

    @property
    def sensor_name(self):
        """Return standard sensor or instrument name for the file's data.
        """
        res = self['/attr/instrument_name']
        if isinstance(res, np.ndarray):
            return str(res.astype(str))
        else:
            return res

    def get_shape(self, ds_id, ds_info):
        """Return data array shape for item specified.
        """
        var_path = ds_info.get('file_key', '{}'.format(ds_id.name))
        shape = self[var_path + "/shape"]
        if "index" in ds_info:
            shape = shape[1:]
        if "pressure_index" in ds_info:
            shape = shape[:-1]
        return shape

    def adjust_scaling_factors(self, factors, file_units, output_units):
        if factors is None or factors[0] is None:
            factors = [1, 0]
        if file_units == output_units:
            LOG.debug("File units and output units are the same (%s)", file_units)
            return factors
        factors = np.array(factors)

        if file_units == "W cm-2 sr-1" and output_units == "W m-2 sr-1":
            LOG.debug("Adjusting scaling factors to convert '%s' to '%s'", file_units, output_units)
            factors[::2] = np.where(factors[::2] != -999, factors[::2] * 10000.0, -999)
            factors[1::2] = np.where(factors[1::2] != -999, factors[1::2] * 10000.0, -999)
            return factors
        elif file_units == "1" and output_units == "%":
            LOG.debug("Adjusting scaling factors to convert '%s' to '%s'", file_units, output_units)
            factors[::2] = np.where(factors[::2] != -999, factors[::2] * 100.0, -999)
            factors[1::2] = np.where(factors[1::2] != -999, factors[1::2] * 100.0, -999)
            return factors
        else:
            return factors

    def combine_info(self, all_infos):
        info = super(NUCAPSFileHandler, self).combine_info(all_infos)
        info['Quality_Flag'] = np.concatenate(tuple(nfo['Quality_Flag'] for nfo in all_infos))
        return info

    def get_dataset(self, dataset_id, ds_info, out=None):
        var_path = ds_info.get('file_key', '{}'.format(dataset_id.name))
        dtype = ds_info.get('dtype', np.float32)
        if var_path + '/shape' not in self:
            # loading a scalar value
            shape = 1
        else:
            shape = self.get_shape(dataset_id, ds_info)
        file_units = ds_info.get('file_units')
        if file_units is None:
            try:
                file_units = self[var_path + '/attr/units']
                # they were almost completely CF compliant...
                if file_units == "none":
                    file_units = "1"
            except KeyError:
                # no file units specified
                file_units = None

        if out is None:
            out = np.ma.empty(shape, dtype=dtype)
            out.mask = np.zeros(shape, dtype=np.bool)

        try:
            valid_min, valid_max = self[var_path + '/attr/valid_range']
        except KeyError:
            try:
                valid_min = self[var_path + '/attr/valid_min']
                valid_max = self[var_path + '/attr/valid_max']
            except KeyError:
                valid_min = valid_max = None
        if var_path + '/attr/_FillValue' in self:
            fill_value = self[var_path + '/attr/_FillValue']
        else:
            fill_value = None

        d_tmp = np.require(self[var_path][:], dtype=dtype)
        if "index" in ds_info:
            d_tmp = d_tmp[int(ds_info["index"])]
        if "pressure_index" in ds_info:
            d_tmp = d_tmp[..., int(ds_info["pressure_index"])]
            # this is a pressure based field
            # include surface_pressure as metadata
            ds_info.setdefault('surface_pressure', self['Surface_Pressure'][:])
            # include all the pressure levels
            ds_info.setdefault('pressure_levels', self['Pressure'][0])
        out.data[:] = d_tmp
        del d_tmp

        scale_factor_path = var_path + '/attr/scale_factor'
        if scale_factor_path in self:
            scale_factor = self[scale_factor_path]
            scale_offset = self[var_path + '/attr/add_offset']
        else:
            scale_factor = None
            scale_offset = None

        if valid_min is not None and valid_max is not None:
            # the original .cfg/INI based reader only checked valid_max
            out.mask[:] |= (out.data > valid_max) # | (out < valid_min)
        if fill_value is not None:
            out.mask[:] |= out.data == fill_value

        factors = (scale_factor, scale_offset)
        factors = self.adjust_scaling_factors(factors, file_units, ds_info.get("units"))
        if factors[0] != 1 or factors[1] != 0:
            out.data[:] *= factors[0]
            out.data[:] += factors[1]

        ds_info.update({
            "units": ds_info.get("units", file_units),
            "platform": self.platform_name,
            "sensor": self.sensor_name,
            "start_orbit": self.start_orbit_number,
            "end_orbit": self.end_orbit_number,
        })
        ds_info.update(dataset_id.to_dict())
        if 'standard_name' not in ds_info:
            sname_path = var_path + '/attr/standard_name'
            ds_info['standard_name'] = self.get(sname_path)
        ds_info.update({'Quality_Flag': self['Quality_Flag'][:]})

        cls = ds_info.pop("container", Dataset)
        return cls(out, **ds_info)


class NUCAPSReader(FileYAMLReader):
    """Reader for NUCAPS NetCDF4 files.
    """
    def __init__(self, config_files, mask_surface=True, mask_quality=True,
                 start_time=None, end_time=None, area=None, **kwargs):
        """Configure reader behavior.

        :param mask_surface: mask anything below the surface pressure (surface_pressure metadata required)
        :param mask_quality: mask anything where the `quality_flag` metadata is ``!= 1``.

        """
        self.pressure_dataset_names = defaultdict(list)
        super(NUCAPSReader, self).__init__(config_files,
                                           start_time=start_time,
                                           end_time=end_time,
                                           area=area,
                                           **kwargs)
        self.mask_surface = self.info.get('mask_surface', mask_surface)
        self.mask_quality = self.info.get('mask_quality', mask_quality)

    def load_ds_ids_from_config(self):
        super(NUCAPSReader, self).load_ds_ids_from_config()
        for ds_id in list(self.ids.keys()):
            ds_info = self.ids[ds_id]
            if ds_info.get('pressure_based', False):
                for idx, lvl_num in enumerate(ALL_PRESSURE_LEVELS):
                    if lvl_num < 5.0:
                        suffix = "_{:0.03f}mb".format(lvl_num)
                    else:
                        suffix = "_{:0.0f}mb".format(lvl_num)

                    new_info = ds_info.copy()
                    new_info['pressure_level'] = lvl_num
                    new_info['pressure_index'] = idx
                    new_info['file_key'] = '{}'.format(ds_id.name)
                    new_info['name'] = ds_id.name + suffix
                    new_ds_id = ds_id._replace(name=new_info['name'])
                    new_info['id'] = new_ds_id
                    self.ids[new_ds_id] = new_info
                    self.pressure_dataset_names[ds_id.name].append(new_info['name'])

    def load(self, dataset_keys, pressure_levels=None):
        """Load data from one or more set of files.

        :param pressure_levels: mask out certain pressure levels:
                                True for all levels
                                (min, max) for a range of pressure levels
                                [...] list of levels to include
        """
        if pressure_levels is not None:
            # Filter out datasets that don't fit in the correct pressure level
            for ds_id in dataset_keys.copy():
                ds_info = self.ids[ds_id]
                ds_level = ds_info.get("pressure_level")
                if ds_level is not None:
                    if pressure_levels is True:
                        # they want all pressure levels
                        continue
                    elif len(pressure_levels) == 2 and pressure_levels[0] <= ds_level <= pressure_levels[1]:
                        # given a min and a max pressure level
                        continue
                    elif np.isclose(pressure_levels, ds_level).any():
                        # they asked for this specific pressure level
                        continue
                    else:
                        # they don't want this dataset at this pressure level
                        LOG.debug("Removing dataset to load: %s", ds_id)
                        dataset_keys.remove(ds_id)
                        continue

            # Add pressure levels to the datasets to load if needed so
            # we can do further filtering after loading
            plevels_ds_id = self.get_dataset_key('Pressure_Levels')
            remove_plevels = False
            if plevels_ds_id not in dataset_keys:
                dataset_keys.add(plevels_ds_id)
                remove_plevels = True

        datasets_loaded = super(NUCAPSReader, self).load(dataset_keys)

        if pressure_levels is not None:
            if remove_plevels:
                plevels_ds = datasets_loaded.pop(plevels_ds_id)
                dataset_keys.remove(plevels_ds_id)
            else:
                plevels_ds = datasets_loaded[plevels_ds_id]

            for ds_id in datasets_loaded.keys():
                ds_obj = datasets_loaded[ds_id]
                if plevels_ds is None:
                    LOG.debug("No 'pressure_levels' metadata included in dataset")
                    continue
                if plevels_ds.shape[0] != ds_obj.shape[-1]:
                    # LOG.debug("Dataset '{}' doesn't contain multiple pressure levels".format(ds_id))
                    continue

                if pressure_levels is True:
                    levels_mask = np.ones(plevels_ds.shape, dtype=np.bool)
                elif len(pressure_levels) == 2:
                    # given a min and a max pressure level
                    levels_mask = (plevels_ds <= pressure_levels[1]) & (plevels_ds >= pressure_levels[0])
                else:
                    levels_mask = np.zeros(plevels_ds.shape, dtype=np.bool)
                    for idx, ds_level in enumerate(plevels_ds):
                        levels_mask[idx] = np.isclose(pressure_levels, ds_level).any()

                datasets_loaded[ds_id] = ds_obj[..., levels_mask]
                datasets_loaded[ds_id].info["pressure_levels"] = plevels_ds[levels_mask]

        if self.mask_surface:
            LOG.debug("Filtering pressure levels at or below the surface pressure")
            for ds_id in dataset_keys:
                ds = datasets_loaded[ds_id]
                if "surface_pressure" not in ds.info or "pressure_levels" not in ds.info:
                    continue
                data_pressure = ds.info["pressure_levels"]
                surface_pressure = ds.info["surface_pressure"]
                if isinstance(surface_pressure, float):
                    # scalar needs to become array for each record
                    surface_pressure = np.repeat(surface_pressure, ds.shape[0])
                if surface_pressure.ndim == 1 and surface_pressure.shape[0] == ds.shape[0]:
                    # surface is one element per record
                    LOG.debug("Filtering %s at and below the surface pressure", ds_id)
                    if ds.ndim == 2:
                        surface_pressure = np.repeat(surface_pressure[:, None], data_pressure.shape[0], axis=1)
                        data_pressure = np.repeat(data_pressure[None, :], surface_pressure.shape[0], axis=0)
                        ds.mask[data_pressure >= surface_pressure] = True
                    else:
                        # entire dataset represents one pressure level
                        data_pressure = ds.info["pressure_level"]
                        ds.mask[data_pressure >= surface_pressure] = True
                else:
                    LOG.warning("Not sure how to handle shape of 'surface_pressure' metadata")

        if self.mask_quality:
            LOG.debug("Filtering data based on quality flags")
            for ds_id in dataset_keys:
                ds = datasets_loaded[ds_id]
                if "quality_flag" not in ds.info:
                    continue
                quality_flag = ds.info["quality_flag"]
                LOG.debug("Masking %s where quality flag doesn't equal 1", ds_id)
                ds.mask[quality_flag != 0, ...] = True

        return datasets_loaded


