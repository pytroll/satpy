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
"""Interface to MiRS product."""

import datetime
import logging
import os
from collections import Counter

import dask.array as da
import numpy as np
import xarray as xr

from satpy import CHUNK_SIZE
from satpy.aux_download import retrieve
from satpy.readers.file_handlers import BaseFileHandler

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    # try getting setuptools/distribute's version of resource retrieval first
    from pkg_resources import resource_string as get_resource_string
except ImportError:
    from pkgutil import get_data as get_resource_string  # type: ignore

#

# 'Polo' variable in MiRS files use these values for H/V polarization
POLO_V = 2
POLO_H = 3

amsu = "amsu-mhs"
PLATFORMS = {"n18": "NOAA-18",
             "n19": "NOAA-19",
             "np": "NOAA-19",
             "m2": "MetOp-A",
             "m1": "MetOp-B",
             "m3": "MetOp-C",
             "ma2": "MetOp-A",
             "ma1": "MetOp-B",
             "ma3": "MetOp-C",
             "npp": "NPP",
             "f17": "DMSP-F17",
             "f18": "DMSP-F18",
             "gpm": "GPM",
             "n20": "NOAA-20",
             }
SENSOR = {"n18": amsu,
          "n19": amsu,
          "n20": 'atms',
          "np": amsu,
          "m1": amsu,
          "m2": amsu,
          "m3": amsu,
          "ma1": amsu,
          "ma2": amsu,
          "ma3": amsu,
          "npp": "atms",
          "jpss": "atms",
          "f17": "ssmis",
          "f18": "ssmis",
          "gpm": "GPI",
          }


def read_atms_coeff_to_string(fn):
    """Read the coefficients into a string."""
    if os.path.isfile(fn):
        coeff_str = open(fn, "r").readlines()
    else:
        parts = fn.split(":")
        mod_part, file_part = parts if len(parts) == 2 else ("", parts[0])
        mod_part = mod_part or __package__  # self.__module__
        coeff_str = get_resource_string(mod_part, file_part).decode().split("\n")

    return coeff_str


def read_atms_limb_correction_coefficients(fn):
    """Read the limb correction files."""
    coeff_str = read_atms_coeff_to_string(fn)
    n_chn = 22
    n_fov = 96
    # make the string a generator
    coeff_lines = (line.strip() for line in coeff_str)

    all_coeffs = np.zeros((n_chn, n_fov, n_chn), dtype=np.float32)
    all_amean = np.zeros((n_chn, n_fov, n_chn), dtype=np.float32)
    all_dmean = np.zeros(n_chn, dtype=np.float32)
    all_nchx = np.zeros(n_chn, dtype=np.int32)
    all_nchanx = np.zeros((n_chn, n_chn), dtype=np.int32)
    all_nchanx[:] = 9999
    # There should be 22 sections
    for chan_idx in range(n_chn):
        # blank line at the start of each section
        _ = next(coeff_lines)
        # section header
        next_line = next(coeff_lines)

        _nx, nchx, dmean = [x.strip() for x in next_line.split(" ") if x]
        all_nchx[chan_idx] = nchx = int(nchx)
        all_dmean[chan_idx] = float(dmean)

        # coeff locations (indexes to put the future coefficients in)
        next_line = next(coeff_lines)
        locations = [int(x.strip()) for x in next_line.split(" ") if x]
        if len(locations) != nchx:
            raise RuntimeError
        for x in range(nchx):
            all_nchanx[chan_idx, x] = locations[x] - 1

        # Read 'nchx' coefficients for each of 96 FOV
        for fov_idx in range(n_fov):
            # chan_num, fov_num, *coefficients, error
            coeff_line_parts = [x.strip() for x in next(coeff_lines).split(" ") if x][2:]
            coeffs = [float(x) for x in coeff_line_parts[:nchx]]
            ameans = [float(x) for x in coeff_line_parts[nchx:-1]]
            # not used but nice to know the purpose of the last column.
            # _error_val = float(coeff_line_parts[-1])
            for x in range(nchx):
                all_coeffs[chan_idx, fov_idx, all_nchanx[chan_idx, x]] = coeffs[x]
                all_amean[all_nchanx[chan_idx, x], fov_idx, chan_idx] = ameans[x]

    return all_dmean, all_coeffs, all_amean, all_nchx, all_nchanx


def apply_atms_limb_correction(datasets, channel_idx, dmean,
                               coeffs, amean, nchx, nchanx):
    """Calculate the correction for each channel."""
    ds = datasets[channel_idx]

    fov_line_correct = []
    for fov_idx in range(ds.shape[1]):
        coeff_sum = np.zeros(ds.shape[0], dtype=ds.dtype)
        for k in range(nchx[channel_idx]):
            chn_repeat = nchanx[channel_idx, k]
            coef = coeffs[channel_idx, fov_idx, chn_repeat] * (
                   datasets[chn_repeat, :, fov_idx] -
                   amean[chn_repeat, fov_idx, channel_idx])
            coeff_sum = np.add(coef, coeff_sum)
        fov_line_correct.append(np.add(coeff_sum, dmean[channel_idx]))
    return np.stack(fov_line_correct, axis=1)


def get_coeff_by_sfc(coeff_fn, bt_data, idx):
    """Read coefficients for specific filename (land or sea)."""
    sfc_coeff = read_atms_limb_correction_coefficients(coeff_fn)
    # transpose bt_data for correction
    bt_data = bt_data.transpose("Channel", "y", "x")
    c_size = bt_data[idx, :, :].chunks
    correction = da.map_blocks(apply_atms_limb_correction,
                               bt_data, idx,
                               *sfc_coeff, chunks=c_size, meta=np.array((), dtype=bt_data.dtype))
    return correction


def limb_correct_atms_bt(bt_data, surf_type_mask, coeff_fns, ds_info):
    """Gather data needed for limb correction."""
    idx = ds_info['channel_index']
    LOG.info("Starting ATMS Limb Correction...")

    sea_bt = get_coeff_by_sfc(coeff_fns['sea'], bt_data, idx)
    land_bt = get_coeff_by_sfc(coeff_fns['land'], bt_data, idx)

    LOG.info("Finishing limb correction")
    is_sea = (surf_type_mask == 0)
    new_data = np.where(is_sea, sea_bt, land_bt)

    bt_corrected = xr.DataArray(new_data, dims=("y", "x"),
                                attrs=ds_info)

    return bt_corrected


class MiRSL2ncHandler(BaseFileHandler):
    """MiRS handler for NetCDF4 files using xarray.

    The MiRS retrieval algorithm runs on multiple
    sensors.  For the ATMS sensors, a limb correction
    is applied by default.  In order to change that
    behavior, use the keyword argument ``limb_correction=False``::


        from satpy import Scene, find_files_and_readers

        filenames = find_files_and_readers(base_dir, reader="mirs")
        scene = Scene(filenames, reader_kwargs={'limb_correction': False})

    """

    def __init__(self, filename, filename_info, filetype_info,
                 limb_correction=True):
        """Init method."""
        super(MiRSL2ncHandler, self).__init__(filename,
                                              filename_info,
                                              filetype_info,
                                              )

        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  decode_coords=True,
                                  chunks={'Field_of_view': CHUNK_SIZE,
                                          'Scanline': CHUNK_SIZE})
        # y,x is used in satpy, bands rather than channel using in xrimage
        self.nc = self.nc.rename_dims({"Scanline": "y",
                                       "Field_of_view": "x"})
        self.nc = self.nc.rename({"Latitude": "latitude",
                                  "Longitude": "longitude"})

        self.platform_name = self._get_platform_name
        self.sensor = self._get_sensor
        self.limb_correction = limb_correction

    @property
    def platform_shortname(self):
        """Get platform shortname."""
        return self.filename_info['platform_shortname']

    @property
    def _get_platform_name(self):
        """Get platform name."""
        try:
            res = PLATFORMS[self.filename_info['platform_shortname'].lower()]
        except KeyError:
            res = "mirs"
        return res.lower()

    @property
    def _get_sensor(self):
        """Get sensor."""
        try:
            res = SENSOR[self.filename_info["platform_shortname"].lower()]
        except KeyError:
            res = self.sensor_names
        return res

    @property
    def sensor_names(self):
        """Return standard sensor names for the file's data."""
        return list(set(SENSOR.values()))

    @property
    def start_time(self):
        """Get start time."""
        # old file format
        if self.filename_info.get("date", False):
            s_time = datetime.datetime.combine(
                self.force_date("date"),
                self.force_time("start_time")
            )
            self.filename_info["start_time"] = s_time
        return self.filename_info["start_time"]

    @property
    def end_time(self):
        """Get end time."""
        # old file format
        if self.filename_info.get("date", False):
            end_time = datetime.datetime.combine(
                self.force_date("date"),
                self.force_time("end_time")
            )
            self.filename_info["end_time"] = end_time
        return self.filename_info["end_time"]

    def force_date(self, key):
        """Force datetime.date for combine."""
        if isinstance(self.filename_info[key], datetime.datetime):
            return self.filename_info[key].date()
        return self.filename_info[key]

    def force_time(self, key):
        """Force datetime.time for combine."""
        if isinstance(self.filename_info.get(key), datetime.datetime):
            return self.filename_info.get(key).time()
        return self.filename_info.get(key)

    @property
    def _get_coeff_filenames(self):
        """Retrieve necessary files for coefficients if needed."""
        coeff_fn = {'sea': None, 'land': None}
        if self.platform_name == "noaa-20":
            coeff_fn['land'] = retrieve("readers/limbcoef_atmsland_noaa20.txt")
            coeff_fn['sea'] = retrieve("readers/limbcoef_atmssea_noaa20.txt")
        if self.platform_name == 'npp':
            coeff_fn['land'] = retrieve("readers/limbcoef_atmsland_snpp.txt")
            coeff_fn['sea'] = retrieve("readers/limbcoef_atmssea_snpp.txt")

        return coeff_fn

    def update_metadata(self, ds_info):
        """Get metadata."""
        metadata = {}
        metadata.update(ds_info)
        metadata.update({
            'sensor': self.sensor,
            'platform_name': self.platform_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
        })
        return metadata

    @staticmethod
    def _nan_for_dtype(data_arr_dtype):
        # don't force the conversion from 32-bit float to 64-bit float
        # if we don't have to
        if data_arr_dtype.type == np.float32:
            return np.float32(np.nan)
        if np.issubdtype(data_arr_dtype, np.timedelta64):
            return np.timedelta64('NaT')
        if np.issubdtype(data_arr_dtype, np.datetime64):
            return np.datetime64('NaT')
        return np.nan

    @staticmethod
    def _scale_data(data_arr, scale_factor, add_offset):
        """Scale data, if needed."""
        scaling_needed = not (scale_factor == 1 and add_offset == 0)
        if scaling_needed:
            data_arr = data_arr * scale_factor + add_offset
        return data_arr

    def _fill_data(self, data_arr, fill_value, scale_factor, add_offset):
        """Fill missing data with NaN."""
        if fill_value is not None:
            fill_value = self._scale_data(fill_value, scale_factor, add_offset)
            fill_out = self._nan_for_dtype(data_arr.dtype)
            data_arr = data_arr.where(data_arr != fill_value, fill_out)
        return data_arr

    def _apply_valid_range(self, data_arr, valid_range, scale_factor, add_offset):
        """Get and apply valid_range."""
        if valid_range is not None:
            valid_min, valid_max = valid_range
            valid_min = self._scale_data(valid_min, scale_factor, add_offset)
            valid_max = self._scale_data(valid_max, scale_factor, add_offset)

            if valid_min is not None and valid_max is not None:
                data_arr = data_arr.where((data_arr >= valid_min) &
                                          (data_arr <= valid_max))
        return data_arr

    def apply_attributes(self, data, ds_info):
        """Combine attributes from file and yaml and apply.

        File attributes should take precedence over yaml if both are present

        """
        try:
            global_attr_fill = self.nc.missing_value
        except AttributeError:
            global_attr_fill = 1.0

        # let file metadata take precedence over ds_info from yaml,
        # but if yaml has more to offer, include it here, but fix
        # units.
        ds_info.update(data.attrs)

        # special cases
        if ds_info['name'] in ["latitude", "longitude"]:
            ds_info["standard_name"] = ds_info.get("standard_name",
                                                   ds_info['name'])

        # try to assign appropriate units (if "Kelvin" covert to K)
        units_convert = {"Kelvin": "K"}
        data_unit = ds_info.get('units', None)
        ds_info['units'] = units_convert.get(data_unit, data_unit)

        scale = ds_info.pop('scale_factor', 1.0)
        offset = ds_info.pop('add_offset', 0.)
        fill_value = ds_info.pop("_FillValue", global_attr_fill)
        valid_range = ds_info.pop('valid_range', None)

        data = self._scale_data(data, scale, offset)
        data = self._fill_data(data, fill_value, scale, offset)
        data = self._apply_valid_range(data, valid_range, scale, offset)

        data.attrs = ds_info

        return data, ds_info

    def get_dataset(self, ds_id, ds_info):
        """Get datasets."""
        if 'dependencies' in ds_info.keys():
            idx = ds_info['channel_index']
            data = self['BT']
            data = data.rename(new_name_or_name_dict=ds_info["name"])
            data, ds_info = self.apply_attributes(data, ds_info)

            if self.sensor.lower() == "atms" and self.limb_correction:
                sfc_type_mask = self['Sfc_type']
                data = limb_correct_atms_bt(data, sfc_type_mask,
                                            self._get_coeff_filenames,
                                            ds_info)

                self.nc = self.nc.merge(data)
            else:
                LOG.info("No Limb Correction applied.")
                data = data[:, :, idx]
        else:
            data = self[ds_id['name']]
            data, ds_info = self.apply_attributes(data, ds_info)

        data.attrs = self.update_metadata(ds_info)

        return data

    def available_datasets(self, configured_datasets=None):
        """Dynamically discover what variables can be loaded from this file.

        See :meth:`satpy.readers.file_handlers.BaseHandler.available_datasets`
        for more information.

        """
        handled_vars = set()
        for is_avail, ds_info in (configured_datasets or []):
            if is_avail is not None:
                # some other file handler said it has this dataset
                # we don't know any more information than the previous
                # file handler so let's yield early
                yield is_avail, ds_info
                continue

            yaml_info = {}
            if self.file_type_matches(ds_info['file_type']):
                handled_vars.add(ds_info['name'])
                yaml_info = ds_info
            if ds_info['name'] == 'BT':
                yield from self._available_btemp_datasets(yaml_info)
            yield True, ds_info
        yield from self._available_new_datasets(handled_vars)

    def _count_channel_repeat_number(self):
        """Count channel/polarization pair repetition."""
        freq = self.nc.coords.get('Freq', self.nc.get('Freq'))
        polo = self.nc['Polo']

        chn_total = Counter()
        normals = []
        for idx, (f, p) in enumerate(zip(freq, polo)):
            normal_f = str(int(f))
            normal_p = 'v' if p == POLO_V else 'h'
            chn_total[normal_f + normal_p] += 1
            normals.append((idx, f, p, normal_f, normal_p))

        return chn_total, normals

    def _available_btemp_datasets(self, yaml_info):
        """Create metadata for channel BTs."""
        chn_total, normals = self._count_channel_repeat_number()
        # keep track of current channel count for string description
        chn_cnt = Counter()
        for idx, _f, _p, normal_f, normal_p in normals:
            chn_cnt[normal_f + normal_p] += 1
            p_count = str(chn_cnt[normal_f + normal_p]
                          if chn_total[normal_f + normal_p] > 1 else '')

            new_name = "btemp_{}{}{}".format(normal_f, normal_p, p_count)

            desc_bt = "Channel {} Brightness Temperature at {}GHz {}{}"
            desc_bt = desc_bt.format(idx, normal_f, normal_p, p_count)
            ds_info = yaml_info.copy()
            ds_info.update({
                'file_type': self.filetype_info['file_type'],
                'name': new_name,
                'description': desc_bt,
                'channel_index': idx,
                'frequency': "{}GHz".format(normal_f),
                'polarization': normal_p,
                'dependencies': ('BT', 'Sfc_type'),
                'coordinates': ['longitude', 'latitude']
            })
            yield True, ds_info

    def _get_ds_info_for_data_arr(self, var_name):
        ds_info = {
            'file_type': self.filetype_info['file_type'],
            'name': var_name,
            'coordinates': ["longitude", "latitude"]
        }
        return ds_info

    def _is_2d_yx_data_array(self, data_arr):
        has_y_dim = data_arr.dims[0] == "y"
        has_x_dim = data_arr.dims[1] == "x"
        return has_y_dim and has_x_dim

    def _available_new_datasets(self, handled_vars):
        """Metadata for available variables other than BT."""
        possible_vars = list(self.nc.items()) + list(self.nc.coords.items())
        for var_name, data_arr in possible_vars:
            if var_name in handled_vars:
                continue
            if data_arr.ndim != 2:
                # we don't currently handle non-2D variables
                continue
            if not self._is_2d_yx_data_array(data_arr):
                # we need 'traditional' y/x dimensions currently
                continue

            ds_info = self._get_ds_info_for_data_arr(var_name)
            yield True, ds_info

    def __getitem__(self, item):
        """Wrap around `self.nc[item]`."""
        data = self.nc[item]

        # 'Freq' dimension causes issues in other processing
        if 'Freq' in data.coords:
            data = data.drop_vars('Freq')

        return data
