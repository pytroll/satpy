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

import os
import logging
import datetime
import numpy as np
import xarray as xr
import dask.array as da
from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

try:
    # try getting setuptools/distribute's version of resource retrieval first
    from pkg_resources import resource_string as get_resource_string
except ImportError:
    from pkgutil import get_data as get_resource_string

#

# 'Polo' variable in MiRS files use these values for H/V polarization
POLO_V = 2
POLO_H = 3

# number of channels
# n_channels = 22
# number of fields of view
N_FOV = 96

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

LIMB_SEA_FILE = os.environ.get("ATMS_LIMB_SEA",
                               "/Users/joleenf/data/mirs/coeff/"
                               "limball_atmssea.txt")
LIMB_LAND_FILE = os.environ.get("ATMS_LIMB_LAND",
                                "/Users/joleenf/data/mirs/coeff/"
                                "limball_atmsland.txt")


def read_atms_limb_correction_coeffs(fn):
    """Read provided limb correction files for atms."""
    if os.path.isfile(fn):
        coeff_str = open(fn, "r").readlines()
    else:
        parts = fn.split(":")
        mod_part, file_part = parts if len(parts) == 2 else ("", parts[0])
        mod_part = mod_part or __package__  # self.__module__
        coeff_str = get_resource_string(mod_part, file_part).decode().split("\n")
    # make it a generator
    coeff_str = (line.strip() for line in coeff_str)

    all_coeffs = np.zeros((22, 96, 22), dtype=np.float32)
    all_amean = np.zeros((22, 96, 22), dtype=np.float32)
    all_dmean = np.zeros(22, dtype=np.float32)
    all_nchx = np.zeros(22, dtype=np.int32)
    all_nchanx = np.zeros((22, 22), dtype=np.int32)
    all_nchanx[:] = 9999
    # There should be 22 sections
    for chan_idx in range(22):
        # blank line at the start of each section
        _ = next(coeff_str)

        # section header
        nx, nchx, dmean = [x.strip() for x in next(coeff_str).split(" ") if x]
        nx = int(nx)  # Question, was this supposed to be used somewhere?
        all_nchx[chan_idx] = nchx = int(nchx)
        all_dmean[chan_idx] = float(dmean)

        # coeff locations (indexes to put the future coefficients in)
        locations = [int(x.strip()) for x in next(coeff_str).split(" ") if x]
        assert len(locations) == nchx
        for x in range(nchx):
            all_nchanx[chan_idx, x] = locations[x] - 1

        # Read 'nchx' coefficients for each of 96 FOV (N_FOV).
        for fov_idx in range(N_FOV):
            # chan_num, fov_num, *coefficients, error
            coeff_line_parts = [x.strip() for x in next(coeff_str).split(" ") if x][2:]
            coeffs = [float(x) for x in coeff_line_parts[:nchx]]
            ameans = [float(x) for x in coeff_line_parts[nchx:-1]]
            # error_val = float(coeff_line_parts[-1])
            for x in range(nchx):
                all_coeffs[chan_idx, fov_idx, all_nchanx[chan_idx, x]] = coeffs[x]
                all_amean[all_nchanx[chan_idx, x], fov_idx, chan_idx] = ameans[x]

    return all_dmean, all_coeffs, all_amean, all_nchx, all_nchanx


def apply_atms_limb_correction(datasets, channel_idx, dmean, coeffs, amean, nchx, nchanx):
    """Apply the atms limb correction to the brightness temperature data."""
    datasets = datasets.persist()
    ds = datasets[channel_idx]
    new_ds = np.zeros(ds.shape, dtype=ds.dtype)
    for fov_idx in range(N_FOV):
        new_ds[:, fov_idx] = da.map_blocks(coeff_cums_calc,
                                           datasets, channel_idx,
                                           fov_idx, dmean, coeffs,
                                           amean, nchx,
                                           nchanx, dtype=ds.dtype)

    return new_ds


def coeff_cums_calc(datasets, channel_idx, fov_idx,
                    dmean, coeffs, amean, nchx, nchanx):
    """Calculate the correction for each channel."""
    coeff_sum = np.zeros(datasets.shape[1], dtype=datasets[0].dtype)
    for k in range(nchx[channel_idx]):
        coef = coeffs[channel_idx, fov_idx, nchanx[channel_idx, k]] * (
                datasets[nchanx[channel_idx, k], :, fov_idx] -
                amean[nchanx[channel_idx, k], fov_idx, channel_idx])
        coeff_sum = coef + coeff_sum
    return coeff_sum + dmean[channel_idx]


class MiRSL2ncHandler(BaseFileHandler):
    """MiRS handler for NetCDF4 files using xarray."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init method."""
        super(MiRSL2ncHandler, self).__init__(filename,
                                              filename_info,
                                              filetype_info)

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

        if len(self.nc.coords.values()) == 0:
            self.nc = self.nc.assign_coords(self.new_coords())

        self.platform_name = self._get_platform_name
        self.sensor = self._get_sensor

    def new_coords(self):
        """Define coordinates when file does not use variable attributes."""
        if not self.nc.coords.keys():
            # this file did not define variable coordinates
            new_coords = {'latitude': self['latitude'],
                          'longitude': self['longitude']}
        else:
            # let xarray handle coordinates defined by variable attributes.
            new_coords = self.nc.coords.keys()
        return new_coords

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
        else:
            return self.filename_info[key]

    def force_time(self, key):
        """Force datetime.time for combine."""
        if isinstance(self.filename_info.get(key), datetime.datetime):
            return self.filename_info.get(key).time()
        else:
            return self.filename_info.get(key)

    def limb_correct_atms_bt(self, bt_data, ds_info):
        """Gather data needed for limb correction."""
        idx = ds_info['channel_index']
        LOG.info("Starting ATMS Limb Correction...")
        # transpose bt_data for correction
        bt_data = bt_data.transpose("Channel", "y", "x")

        deps = ds_info['dependencies']
        if len(deps) != 2:
            LOG.error("Expected 1 dependencies to create corrected BT product, got %d" % (len(deps),))
            raise ValueError("Expected 1 dependencies to create corrected BT product, got %d" % (len(deps),))

        surf_type_name = deps[1]
        surf_type_mask = self[surf_type_name]

        sea = read_atms_limb_correction_coeffs(LIMB_SEA_FILE)
        sea_bt = apply_atms_limb_correction(bt_data, idx, *sea)

        land = read_atms_limb_correction_coeffs(LIMB_LAND_FILE)
        land_bt = apply_atms_limb_correction(bt_data, idx, *land)

        LOG.info("Finishing limb correction")
        is_sea = (surf_type_mask == 0)
        bt_data = bt_data[idx, :, :]
        new_data = (sea_bt.squeeze() * [is_sea] +
                    land_bt.squeeze() * [~is_sea])

        # for consistency when not doing limb correction, return a dask.array
        new_data = da.from_array(new_data.squeeze())

        bt_data = xr.DataArray(new_data.squeeze(), dims=bt_data.dims,
                               coords=bt_data.coords, attrs=ds_info,
                               name=bt_data.name)
        return bt_data

    def get_metadata(self, ds_info):
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
        elif np.issubdtype(data_arr_dtype, np.timedelta64):
            return np.timedelta64('NaT')
        elif np.issubdtype(data_arr_dtype, np.datetime64):
            return np.datetime64('NaT')
        return np.nan

    def _scale_data(self, data_arr, attrs):
        # handle scaling
        # take special care for integer/category fields
        scale_factor = attrs.pop('scale_factor', 1.)
        add_offset = attrs.pop('add_offset', 0.)
        scaling_needed = not (scale_factor == 1 and add_offset == 0)
        if scaling_needed:
            data_arr = data_arr * scale_factor + add_offset
        return data_arr, attrs

    def _fill_data(self, data_arr, attrs):
        try:
            global_attr_fill = self.nc.missing_value
        except AttributeError:
            global_attr_fill = None
        fill_value = attrs.pop('_FillValue', global_attr_fill)

        fill_out = self._nan_for_dtype(data_arr.dtype)
        if fill_value is not None:
            data_arr = data_arr.where(data_arr != fill_value, fill_out)
        return data_arr, attrs

    def get_dataset(self, ds_id, ds_info):
        """Get datasets."""
        if 'dependencies' in ds_info.keys():
            idx = ds_info['channel_index']
            data = self['BT']
            LOG.debug('Calc {} {}'.format(idx, ds_id))
            data = data.rename(new_name_or_name_dict=ds_info["name"])

            # only correct for 'BT' data
            if 'BT' not in ds_info['dependencies']:
                do_not_apply = True
            else:
                do_not_apply = True   # TODO:  Change this to false when coeff loc is resolved.

            if self.sensor.lower() != "atms" or do_not_apply:
                LOG.info("Limb Correction will not be applied to non-ATMS BTs")
                data = data[:, :, idx]
            else:
                data = self.limb_correct_atms_bt(data, ds_info)
                self.nc = self.nc.merge(data)
        else:
            data = self[ds_id['name']]

        data.attrs = self.get_metadata(ds_info)
        return data

    def _available_if_this_file_type(self, configured_datasets):
        for is_avail, ds_info in (configured_datasets or []):
            if is_avail is not None:
                # some other file handler said it has this dataset
                # we don't know any more information than the previous
                # file handler so let's yield early
                yield is_avail, ds_info
                continue
            yield self.file_type_matches(ds_info['file_type']), ds_info

    def _available_new_datasets(self):
        possible_vars = list(self.nc.data_vars.items())
        for var_name, val in possible_vars:
            if val.ndim < 2:
                # only handle 2d variables and 3-D BT.
                # This brings in uncorrected BT(YM) as well... agh!
                continue

            if isinstance(val, xr.DataArray):
                # get specific brightness temperature band products
                if (var_name == 'BT' and
                        self.filetype_info['file_type'] in 'mirs_atms'):
                    freq = self.nc.coords.get('Freq', self.nc.get('Freq'))
                    polo = self.nc['Polo']
                    from collections import Counter
                    c = Counter()
                    normals = []
                    for idx, (f, p) in enumerate(zip(freq, polo)):
                        normal_f = str(int(f))
                        normal_p = 'v' if p == POLO_V else 'h'
                        c[normal_f + normal_p] += 1
                        normals.append((idx, f, p, normal_f, normal_p))

                    c2 = Counter()
                    for idx, _f, _p, normal_f, normal_p in normals:
                        c2[normal_f + normal_p] += 1
                        p_count = str(c2[normal_f + normal_p]
                                      if c[normal_f + normal_p] > 1 else '')

                        new_name = "btemp_{}{}{}".format(normal_f,
                                                         normal_p,
                                                         p_count)

                        desc_bt = ("Channel {} Brightness Temperature"
                                   " at {}GHz {}{}".format(idx,
                                                           normal_f,
                                                           normal_p,
                                                           p_count))
                        ds_info = {
                            'file_type': self.filetype_info['file_type'],
                            'name': new_name,
                            'description': desc_bt,
                            'units': 'K',
                            'channel_index': idx,
                            'frequency': "{}GHz".format(normal_f),
                            'polarization': normal_p,
                            'dependencies': ('BT', 'Sfc_type'),
                            'coordinates': ["longitude", "latitude"]
                        }
                        yield True, ds_info

                else:
                    # only yield 'y','x' data other than BT for now.
                    if self.nc[var_name].ndim == 2:
                        ds_info = {
                            'file_type': self.filetype_info['file_type'],
                            'name': var_name,
                            'coordinates': ["longitude", "latitude"]
                        }
                        yield True, ds_info

    def _available_coordinates(self):
        for var_name, data_arr in list(self.nc.coords.items()):
            # don't yield ndim == 1, it cannot be displayed.
            if data_arr.ndim == 2:
                attrs = data_arr.attrs.copy()
                ds_info = {
                    'file_type': self.filetype_info['file_type'],
                    'name': var_name,
                    'standard_name': var_name,
                    'coordinates': ['longitude', 'latitude']
                }
                data_arr.attrs = attrs
                yield True, ds_info

    def available_datasets(self, configured_datasets=None):
        """Dynamically discover what variables can be loaded from this file.

        See :meth:`satpy.readers.file_handlers.BaseHandler.available_datasets`
        for more information.

        """
        yield from self._available_if_this_file_type(configured_datasets)
        yield from self._available_coordinates()
        yield from self._available_new_datasets()

    def __getitem__(self, item):
        """Wrap around `self.nc[item]`.

        Some datasets use a 32-bit float scaling factor like the 'x' and 'y'
        variables which causes inaccurate unscaled data values. This method
        forces the scale factor to a 64-bit float first.

        """
        data = self.nc[item]
        attrs = data.attrs.copy()
        data, attrs = self._scale_data(data, attrs)
        data, attrs = self._fill_data(data, attrs)

        # 'Freq' dimension causes issues in other processing
        if 'Freq' in data.coords:
            data = data.drop_vars('Freq')

        return data
