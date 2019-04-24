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
"""Generic Reader for GRIB2 files.

Currently this reader depends on the `pygrib` python package. The `eccodes`
package from ECMWF is preferred, but does not support python 3 at the time
of writing.

"""
import logging
import numpy as np
import xarray as xr
import dask.array as da
from pyproj import Proj
from pyresample import geometry
from datetime import datetime
from cfgrib import dataset

from satpy import DatasetID, CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler
import cfgrib

LOG = logging.getLogger(__name__)


CF_UNITS = {
    'none': '1',
}


class GRIBFileHandler(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info, *req_fh, **fh_kwargs):
        super(GRIBFileHandler, self).__init__(filename, filename_info, filetype_info)
        self._msg_datasets = {}
        self._start_time = None
        self._end_time = None
        self.allow_nd = fh_kwargs.get('allow_nd', False)
        self.grib_file = None

        try:
            if fh_kwargs.get('backend_kwargs', None) is not None:
                self.grib_file = xr.open_dataset(filename, engine='cfgrib', backend_kwargs=fh_kwargs['backend_kwargs'])
            else:
                self.grib_file = xr.open_dataset(filename, engine='cfgrib', backend_kwargs={'indexpath': ''})
        except KeyError:
            raise KeyError("Unknown argument in backend_kwargs: {}".format(fh_kwargs))
        except RuntimeError:
            raise IOError("Unknown GRIB file format: {}".format(self.filename))
        except ValueError:
            # use cfgirb message api
            gf = dataset.messages.FileStream(filename)
            names = []
            for message in gf:
                names.append(message['shortName'])
            names = list(dict.fromkeys(names))

            dts = []
            for name in names:
                try:
                    fh_kwargs['backend_kwargs']['filter_by_keys']['shortName'] = name
                    dts.append(xr.open_dataset(filename, engine='cfgrib', backend_kwargs=fh_kwargs['backend_kwargs']))
                except:
                    continue
            print(dts)
            self.grib_file = xr.merge(dts)
            print(self.grib_file)
        try:
            self._start_time = self._convert_datetime(self.grib_file.coords['valid_time'].values)
            self._end_time = self._convert_datetime(self.grib_file.coords['valid_time'].values)
            self._analyze_messages()
        except:
            self._start_time = None
            self._end_time = None

    # analyzes msgs per var
    def _analyze_messages(self):
        # opening them this way opens them by level
        # need to test with four vars??
        # open it the normal way
        # should only be 3 dimensions or less (lat, long, level)
        #print(self.grib_file)
        for var in self.grib_file.data_vars.keys():
            print(var)
                # if the dimensions of the dataset are two, then level should be None
                # else, there should be level for each dimension 
                # if there are more than 3 dimensions, allow_nd should be true
                # check number of dimensions
            # if doing per level, then has to be greater than 2 dimensions if want to double loop
            #if not self.allow_nd and len(self.grib_file.dims) > 2:
            if not self.allow_nd:
                print('got here')
                print(self.grib_file[var]) 
                if len(self.grib_file[var].dims) < 3:
                    level = None
                    msg_id = DatasetID(name=self.grib_file[var].name, level=level)
                    ds_info = {
                        'name': self.grib_file[var].name,
                        'level': level,
                        'typeOfLevel': self.grib_file[var].attrs['GRIB_typeOfLevel'],
                        'file_type': self.filetype_info['file_type'],
                    }
                    self._msg_datasets[msg_id] = ds_info
                else:
                    for val in self.grib_file[var]:
                        print('val: ', val)
                        if len(self.grib_file[var].dims) > 2: # lat long and assuming the type of level
                            level = int(val.coords[val.attrs['GRIB_typeOfLevel']].values)
                        else:
                            raise ValueError('Specify allow_nd to True in reader keyword arguments to allow multidimensional datasets')
                        msg_id = DatasetID(name=val.name, level=level)
                        ds_info = {
                            'name': val.name,
                            'level': level,
                            'typeOfLevel': val.attrs['GRIB_typeOfLevel'],
                            'file_type': self.filetype_info['file_type'],
                        }
                        self._msg_datasets[msg_id] = ds_info

            else: # open the dataset as nd bc it is greater than 2 dims
            # ex. just iterate over the vars and open them
                msg_id = DatasetID(name=self.grib_file[var].name,
                            level=None)

                ds_info = {
                    'name': self.grib_file[var].name,
                    'level': None,
                    'typeOfLevel': self.grib_file[var].attrs['GRIB_typeOfLevel'],
                    'file_type': self.filetype_info['file_type'],
                }

                self._msg_datasets[msg_id] = ds_info

    # FIXME not needed?
    def _create_dataset_ids(self, keys):
        from itertools import product
        ordered_keys = [k for k in keys.keys() if 'id_key' in keys[k]]
        for id_vals in product(*[keys[k]['values'] for k in ordered_keys]):
            id_keys = [keys[k]['id_key'] for k in ordered_keys]
            msg_info = dict(zip(ordered_keys, id_vals))
            ds_info = dict(zip(id_keys, id_vals))
            msg_id = DatasetID(**ds_info)
            ds_info = msg_id.to_dict()
            ds_info.update(msg_info)
            ds_info['file_type'] = self.filetype_info['file_type']
            self._msg_datasets[msg_id] = ds_info

    @staticmethod
    def _convert_datetime(msg, format="%Y-%m-%d %H:%M:%S"):
        u = np.datetime64(0, 's')
        o = np.timedelta64(1, 's')
        s = (msg - u) / o
        return datetime.utcfromtimestamp(s)

    @property
    def start_time(self):
        """Get start time of this entire file.

        Assumes the first message is the earliest message.

        """
        return self._start_time

    @property
    def end_time(self):
        """Get end time of this entire file.

        Assumes the last message is the latest message.

        """
        return self._end_time

    def available_datasets(self):
        """Automatically determine datasets provided by this file"""
        return self._msg_datasets.items()

    def _get_message(self, ds_info):
        #self.grib_file = xr.open_dataset(self.filename, engine='cfgrib', backend_kwargs=ds_info['fh_kwargs'])
        print(ds_info)
        if self.allow_nd or len(self.grib_file[ds_info['name']].dims) < 3:
            ret = self.grib_file[ds_info['name']]
            return ret
        else:
            # make sure the dims are in the right order
            l = []
            for k in self.grib_file[ds_info['name']].dims:
                if k == ds_info['typeOfLevel']:
                    l.insert(0, k)
                else:
                    l.append(k)
            self.grib_file.transpose(*l)
            ret = self.grib_file[ds_info['name']].loc[ds_info['level'], :, :]
            return ret


    def _area_def_from_msg(self, msg):
        proj_params = {}
        lats = msg['latitude'].values
        lons = msg['longitude'].values
        if msg.attrs['GRIB_gridType'] in ['regular_ll', 'regular_gg']:
            proj_params['proj'] = 'eqc'
            proj = Proj(**proj_params)
            min_lon = lons[0]
            max_lon = lons[-1]
            min_lat = lats[0]
            max_lat = lats[-1]
            if min_lat > max_lat:
                # lats aren't in the order we thought they were, flip them
                # we also need to flip the data in the data loading section
                min_lat, max_lat = max_lat, min_lat
            shape = (lats.shape[0], lons.shape[0])
            min_x, min_y = proj(min_lon, min_lat)
            max_x, max_y = proj(max_lon, max_lat)
            if max_x < min_x and 'over' not in proj_params:
                # wrap around
                proj_params['over'] = True
                proj = Proj(**proj_params)
                max_x, max_y = proj(max_lon, max_lat)
            pixel_size_x = (max_x - min_x) / (shape[1] - 1)
            pixel_size_y = (max_y - min_y) / (shape[0] - 1)
            extents = (
                min_x - pixel_size_x / 2.,
                min_y - pixel_size_y / 2.,
                max_x + pixel_size_x / 2.,
                max_y + pixel_size_y / 2.,
            )
    #    else:
    #        shape = lats.shape + lons.shape
    #        # take the corner points only
    #        lons = lons[([0, 0, -1, -1])]
    #        lats = lats[([0, 0, -1, -1])]
    #        # correct for longitudes over 180
    #        lons[lons > 180] -= 360
    #        #proj_params['proj'] = 
    #        proj = Proj(**proj_params)
    #        x, y = proj(lons, lats)
    #        if msg.attrs.get('jScansPositively', -1) == 1:
    #            min_x, min_y = x[0], y[0]
    #            max_x, max_y = x[3], y[3]
    #        else:
    #            min_x, min_y = x[2], y[2]
    #            max_x, max_y = x[1], y[1]
    #        half_x = abs((max_x - min_x) / (shape[1] - 1)) / 2.
    #        half_y = abs((max_y - min_y) / (shape[0] - 1)) / 2.
    #        extents = (min_x - half_x, min_y - half_y, max_x + half_x, max_y + half_y)

        return geometry.AreaDefinition(
            'on-the-fly grib area',
            'on-the-fly grib area',
            'on-the-fly grib area',
            proj_params,
            shape[1],
            shape[0],
            extents,
        )


    def get_area_def(self, dsid):
        """Get area definition for message.

        If latlong grid then convert to valid eqc grid.

        """
        msg = self._get_message(self._msg_datasets[dsid])
        try:
            return self._area_def_from_msg(msg)
        except (RuntimeError, KeyError):
            raise RuntimeError("Unknown GRIB projection information")

    def get_metadata(self, msg, ds_info):
        model_time = self._convert_datetime(msg['time'].values)
        start_time = self._convert_datetime(msg['valid_time'].values)
        end_time = start_time

        ds_info.update({
            'filename': self.filename,
            'name': msg.name,
            'typeOfLevel': msg.attrs['GRIB_typeOfLevel'],
            'units': msg.attrs['units'],
            'model_time': model_time,
            'valid_min': np.amin(msg.values),
            'valid_max': np.amax(msg.values),
            'start_time': start_time,
            'end_time': end_time,
            'platform_name': 'unknown'
        })
        return ds_info

    def get_dataset(self, dataset_id, ds_info):
        """Read a GRIB message into an xarray DataArray."""

        msg = self._get_message(ds_info)
        msg = msg.rename({'latitude': 'y', 'longitude': 'x'})
        new_dims = msg.dims 
        ds_info = self.get_metadata(msg, ds_info)
        fill = msg.attrs['GRIB_missingValue']
        data = msg.values.astype(np.float32)
        if msg.attrs.get('jScansPositively', -1) == 1:
            data = data[::1]
        
        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)
            data = da.from_array(data, chunks=CHUNK_SIZE)
        else:
            data[data == fill] = np.nan
            data = da.from_array(data, chunks=CHUNK_SIZE)
    
        if len(data.shape) < 3:
            return xr.DataArray(data, attrs=ds_info, dims=('y', 'x'))
        else:
            return xr.DataArray(data, attrs=ds_info, dims=new_dims)
