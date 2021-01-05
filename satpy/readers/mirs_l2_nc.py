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
"""Interface to MIRS product."""

from satpy.readers.file_handlers import BaseFileHandler
from pyresample.geometry import SwathDefinition
import numpy as np
import datetime
import os
import logging
import xarray as xr
import tempfile

LOAD_CHUNK_SIZE = int(os.getenv('PYTROLL_LOAD_CHUNK_SIZE', -1))
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
          "n20": amsu,
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

LIMB_SEA_FILE = os.environ.get("ATMS_LIMB_SEA", "satpy.readers:limball_atmssea.txt")
LIMB_LAND_FILE = os.environ.get("ATMS_LIMB_LAND", "satpy.readers:limball_atmsland.txt")


class MIRSL2ncHandler(BaseFileHandler):
    """MIRS handler for NetCDF4 files using xarray."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init method."""
        super(MIRSL2ncHandler, self).__init__(filename,
                                              filename_info,
                                              filetype_info)

        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  chunks={'Field_of_view': LOAD_CHUNK_SIZE,
                                          'Scanline': LOAD_CHUNK_SIZE})

        self.nc = self.nc.rename_dims({"Scanline": "y", "Field_of_view": "x"})

        self.platform_name = self._get_platform_name
        self.sensor = self._get_sensor
        self.nlines = self.nc.dims['y']
        self.ncols = self.nc.dims['x']
        self.lons = None
        self.lats = None
        self.area = None
        self.coords = {}

        self.sea_bt_data = []
        self.land_bt_data = []

    def get_swath(self):
        """Get lonlats."""
        if self.area is None:
            if self.lons is None or self.lats is None:
                self.lons = self['Longitude']
                self.lats = self['Latitude']
            self.area = SwathDefinition(lons=self.lons, lats=self.lats)
            self.area.name = '_'.join([self.platform_name,
                                       str(self.start_time),
                                       str(self.end_time)])
        return self.area

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

    def _write_temporary_btdata(self, bt_data):
        tempfile_wrapper = tempfile.NamedTemporaryFile(suffix="dat", prefix="bt_data")
        with tempfile_wrapper as filename:
            try:
                fp = np.memmap(filename, dtype=bt_data.dtype, mode='w+', shape=bt_data.shape)
            except (OSError, ValueError):
                LOG.error("Could not extract data from file")
                LOG.debug("Extraction exception: ", exc_info=True)
                raise

        fp[:] = bt_data[:]
        return fp

    def read_atms_limb_correction_coeffs(self, fn):
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

        all_coeffs = np.zeros((22, N_FOV, 22), dtype=np.float32)
        all_amean = np.zeros((22, N_FOV, 22), dtype=np.float32)
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
            nx = int(nx)
            all_nchx[chan_idx] = nchx = int(nchx)
            all_dmean[chan_idx] = float(dmean)

            # coeff locations (indexes to put the future coefficients in)
            locations = [int(x.strip()) for x in next(coeff_str).split(" ") if x]
            assert (len(locations) == nchx)
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

    def apply_atms_limb_correction(self, datasets, dmean, coeffs, amean, nchx, nchanx):
        """Apply the atms limb correction to the brightness temperature data."""
        all_new_ds = []
        coeff_sum = np.zeros(datasets.shape[1], dtype=datasets[0].dtype)
        for channel_idx in range(datasets.shape[0]):
            ds = datasets[channel_idx]
            new_ds = ds.copy()
            all_new_ds.append(new_ds)
            for fov_idx in range(N_FOV):
                coeff_sum[:] = 0
                for k in range(nchx[channel_idx]):
                    coef = coeffs[channel_idx, fov_idx, nchanx[channel_idx, k]] * (
                            datasets[nchanx[channel_idx, k], :, fov_idx] -
                            amean[nchanx[channel_idx, k], fov_idx, channel_idx])
                    coeff_sum += coef
                new_ds[:, fov_idx] = coeff_sum + dmean[channel_idx]

        return all_new_ds

    def limb_correct_atms_bt(self, bt_data, ds_info):
        """Gather data needed for limb correction."""
        idx = ds_info['channel_index']
        LOG.info("Starting ATMS Limb Correction...")
        # transpose bt_data for correction
        bt_data = bt_data.transpose("Channel", "y", "x")

        # write bt_data to a temp file
        fp = self._write_temporary_btdata(bt_data)

        deps = ds_info['dependencies']
        if len(deps) != 2:
            LOG.error("Expected 1 dependencies to create corrected BT product, got %d" % (len(deps),))
            raise ValueError("Expected 1 dependencies to create corrected BT product, got %d" % (len(deps),))

        surf_type_name = deps[1]
        surf_type_mask = self[surf_type_name]

        sea = self.read_atms_limb_correction_coeffs(LIMB_SEA_FILE)
        sea_bt = self.apply_atms_limb_correction(fp, *sea)
        self.sea_bt_data = sea_bt

        land = self.read_atms_limb_correction_coeffs(LIMB_LAND_FILE)
        land_bt = self.apply_atms_limb_correction(fp, *land)
        self.land_bt_data = land_bt

        LOG.info("Finishing limb correction")
        is_sea = (surf_type_mask == 0)
        bt_data = bt_data[idx, :, :]
        new_data = (self.sea_bt_data[idx].squeeze()*[is_sea]) +\
                   (self.land_bt_data[idx].squeeze()*[~is_sea])

        bt_data = xr.DataArray(new_data.squeeze(), dims=bt_data.dims,
                               coords=bt_data.coords, attrs=ds_info,
                               name=bt_data.name)

        return bt_data

    def get_metadata(self, data, ds_info):
        """Get metadata."""
        metadata = {}
        metadata.update(ds_info)
        metadata.update({
            'sensor': self.sensor,
            'platform_name': self.platform_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'coordinates': self.get_lonlat_coords(data, ds_info),
            'area': self.get_swath(),
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
        fill_value = attrs.pop('_FillValue', None)
        fill_out = self._nan_for_dtype(data_arr.dtype)
        if fill_value is not None:
            data_arr = data_arr.where(data_arr != fill_value, fill_out)
        return data_arr, attrs

    def get_dataset(self, ds_id, ds_info):
        """Get datasets."""
        if 'dependencies' in ds_info.keys():
            idx = ds_info['channel_index']
            data = self['BT']
            data = data.rename(new_name_or_name_dict=ds_info["name"])

            if self.sensor.lower() != "atms":
                LOG.info("Limb Correction will not be applied to non-ATMS BTs")
                data = data[:, :, idx]
            else:
                data = self.limb_correct_atms_bt(data, ds_info)
                self.nc = self.nc.merge(data)
        else:
            data = self[ds_id]

        data.attrs = self.get_metadata(data, ds_info)

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

    def get_lonlat_coords(self, data_arr, ds_info):
        """Get lat/lon coordinates for metadata."""
        lat_coord = None
        lon_coord = None
        for coord_name in data_arr.coords:
            if 'longitude' in coord_name.lower():
                lon_coord = coord_name
            if 'latitude' in coord_name.lower():
                lat_coord = coord_name
        ds_info['coordinates'] = [lon_coord, lat_coord]

        return ds_info['coordinates']

    def _available_new_datasets(self):
        possible_vars = list(self.nc.data_vars.items()) + list(self.nc.coords.items())
        for var_name, val in possible_vars:
            if val.ndim < 2:
                # only handle 2d variables and 3-D BT.
                # This brings in uncorrected BT(YM) as well... agh!
                continue

            if isinstance(val, xr.DataArray):
                # get specific brightness temperature band products
                if (var_name == 'BT' and
                        self.filetype_info['file_type'] in 'mirs_atms'):
                    freq = self.nc.coords.get('Freq')
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
                        }
                        yield True, ds_info

                else:
                    ds_info = {
                        'file_type': self.filetype_info['file_type'],
                        'name': var_name,
                    }
                    yield True, ds_info

    def available_datasets(self, configured_datasets=None):
        """Dynamically discover what variables can be loaded from this file.

        See :meth:`satpy.readers.file_handlers.BaseHandler.available_datasets`
        for more information.

        """
        yield from self._available_if_this_file_type(configured_datasets)
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

        # handle coordinates (and recursive fun)
        new_coords = {}
        # 'time' dimension causes issues in other processing
        if 'Freq' in data.coords:
            data = data.drop_vars('Freq')
        if item in data.coords:
            self.coords[item] = data
        for coord_name in data.coords.keys():
            if coord_name not in self.coords:
                self.coords[coord_name] = self[coord_name]
            new_coords[coord_name] = self.coords[coord_name]
        data.coords.update(new_coords)
        return data

    def get_shape(self, key, info):
        """Get the shape of the data."""
        return self.nlines, self.ncols
