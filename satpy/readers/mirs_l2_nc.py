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
import dask as dask

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

    def __init__(self, filename, filename_info, filetype_info,
                 auto_maskandscale=False, xarray_kwargs=None,
                 cache_var_size=0, cache_handle=False):
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

    def get_swath(self, dsid):
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

    @dask.delayed
    def apply_atms_limb_correction(self, datasets, dmean, coeffs, amean, nchx, nchanx):
        """Apply atms limb correction coefficients."""
        all_new_ds = []
        # ds_type = datasets[0].dtype
        ds_type = np.float32
        coeff_sum = np.zeros(datasets.shape[1], dtype=ds_type)
        for channel_idx in range(datasets.shape[0]):
            ds = datasets[channel_idx]
            new_ds = ds.copy().values
            all_new_ds.append(new_ds)
            for fov_idx in range(N_FOV):
                coeff_sum[:] = 0
                for k in range(nchx[channel_idx]):
                    coef = coeffs[channel_idx, fov_idx, nchanx[channel_idx, k]] * (
                            datasets[nchanx[channel_idx, k], :, fov_idx] -
                            amean[nchanx[channel_idx, k], fov_idx, channel_idx])
                    coeff_sum = coeff_sum + coef.values.astype(ds_type)

                new_ds[:, fov_idx] = coeff_sum + dmean[channel_idx]

        return all_new_ds

    def limb_correct_atms_bt(self, ds_info):
        """Gather data needed for limb correction."""
        idx = ds_info['channel_index']
        bt_data = self['BT']
        bt_data = bt_data.rename(new_name_or_name_dict=ds_info["name"])

        skip_limb_correction = True
        if self.sensor.lower() != "atms" or skip_limb_correction:
            LOG.info("Limb Correction will not be applied to non-ATMS BTs")
            bt_data = bt_data[:, :, idx]
            return bt_data

        LOG.info("Starting ATMS Limb Correction...")
        # transpose bt_data for correction
        bt_data = bt_data.transpose("Channel", "y", "x")

        deps = ds_info['dependencies']
        if len(deps) != 2:
            LOG.error("Expected 1 dependencies to create corrected BT product, got %d" % (len(deps),))
            raise ValueError("Expected 1 dependencies to create corrected BT product, got %d" % (len(deps),))

        surf_type_name = deps[1]
        surf_type_mask = self[surf_type_name]

        sea = self.read_atms_limb_correction_coeffs(LIMB_SEA_FILE)
        sea_bt = self.apply_atms_limb_correction(bt_data, *sea)
        self.sea_bt_data = sea_bt.compute()

        land = self.read_atms_limb_correction_coeffs(LIMB_LAND_FILE)
        land_bt = self.apply_atms_limb_correction(bt_data, *land)
        self.land_bt_data = land_bt.compute()

        LOG.info("Finishing limb correction")
        is_sea = (surf_type_mask == 0)
        bt_data = bt_data[idx, :, :]
        new_data = (self.sea_bt_data[idx].squeeze()*[is_sea]) +\
                   (self.land_bt_data[idx].squeeze()*[~is_sea])
        # return the same original swath object since we modified the data in place

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
            'area': self.get_swath(ds_info['name'])
        })

        return metadata

    def get_dataset(self, ds_id, ds_info):
        """Get datasets."""
        if 'dependencies' in ds_info.keys():
            data = self.limb_correct_atms_bt(ds_info)
            self.nc = self.nc.merge(data)
        else:
            data = self[ds_id]

        data.attrs = self.get_metadata(data, ds_info)

        return data

    def available_datasets(self, configured_datasets=None):
        """Update information for or add datasets provided by this file."""
        handled_variables = set()
        # update previously configured datasets
        for is_avail, ds_info in (configured_datasets or []):
            # some other file handler knows how to load this
            if is_avail is not None:
                yield is_avail, ds_info

            var_name = ds_info.get('file_key', ds_info['name'])
            matches = self.file_type_matches(ds_info['file_type'])
            # we can confidently say that we can provide this dataset and can
            # provide more info
            if matches and var_name in self.nc.keys():
                handled_variables.add(var_name)
                new_info = ds_info.copy()  # don't mess up the above yielded
                yield True, new_info
            elif is_avail is None:
                # if we didn't know how to handle this dataset and no one else did
                # then we should keep it going down the chain
                yield is_avail, ds_info

        # Provide new datasets
        for var_name, val in self.nc.items():
            if var_name in handled_variables:
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
                        new_name = "btemp_{}{}{}".format(normal_f, normal_p, str(
                            c2[normal_f + normal_p] if c[normal_f + normal_p] > 1 else ''))

                        desc_bt = ("Channel {} Brightness Temperature"
                                   " at {}GHz {}".format(idx, normal_f,
                                                         normal_p))
                        ds_info = {
                            'file_type': self.filetype_info['file_type'],
                            'name': new_name,
                            'description': desc_bt,
                            'units': 'K',
                            'channel_index': idx,
                            'frequency': "{}GHz".format(normal_f),
                            'polarization': normal_p,
                            'dependencies': ('BT', 'Sfc_type'),
                            'coordinates': ('longitude', 'latitude'),
                        }
                        yield True, ds_info

                else:
                    ds_info = {
                        'file_type': self.filetype_info['file_type'],
                        'name': var_name,
                        'coordinates': ('longitude', 'latitude'),
                    }
                    yield True, ds_info

    def __getitem__(self, item):
        """Wrap around `self.nc[item]`.

        Some datasets use a 32-bit float scaling factor like the 'x' and 'y'
        variables which causes inaccurate unscaled data values. This method
        forces the scale factor to a 64-bit float first.

        """
        data = self.nc[item]
        attrs = data.attrs
        factor = data.attrs.get('scale_factor')
        fill = data.attrs.get('_FillValue')
        if fill is not None:
            data = data.where(data != fill)
        if factor is not None:
            # make sure the factor is a 64-bit float
            # can't do this in place since data is most likely uint16
            # and we are making it a 64-bit float
            data = data * float(factor)
        data.attrs = attrs

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
