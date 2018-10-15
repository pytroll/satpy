# Copyright (c) 2017-2018 PyTroll Community

# Author(s):

#   Sauli Joro <sauli.joro@eumetsat.int>
#   Colin Duff <colin.duff@external.eumetsat.int>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""SEVIRI netcdf format reader. """

from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.msg_base import (SEVIRICalibrationHandler,
                                    CHANNEL_NAMES, CALIB, SATNUM,)
from satpy.readers.eum_base import (time_cds_short, time_cds,
                                    time_cds_expanded)
import xarray as xr
import dask.array as da

from satpy import CHUNK_SIZE
from pyresample import geometry

import numpy as np
import datetime


class NCSEVIRIFileHandler(BaseFileHandler, SEVIRICalibrationHandler):
    def __init__(self, filename, filename_info, filetype_info):
        super(NCSEVIRIFileHandler, self).__init__(filename, filename_info, filetype_info)
        self.nc = None
        self.mda = {}
        self._read_file()
        self.reference = datetime.datetime(1958, 1, 1)

    @property
    def start_time(self):
        delta = datetime.timedelta(days=int(self.nc.attrs['true_repeat_cycle_start_day']),
                                   milliseconds=int(self.nc.attrs['true_repeat_cycle_start_mi_sec']))
        mydate = self.reference + delta
        start_time_str = datetime.datetime.strftime(mydate, '%Y%m%d%H%M%S')+'Z'
        return start_time_str

    @property
    def end_time(self):
        delta = datetime.timedelta(days=int(self.nc.attrs['true_repeat_cycle_start_day']),
                                   milliseconds=int(self.nc.attrs['true_repeat_cycle_start_mi_sec']))
        mydate = self.reference + delta
        end_time_str = datetime.datetime.strftime(mydate, '%Y%m%d%H%M%S')+'Z'
        return end_time_str

    def _read_file(self):
        if self.nc is None:
            self.nc = xr.open_dataset(self.filename,
                                      decode_cf=False,
                                      mask_and_scale=True,
                                      chunks={'num_columns_vis_ir': CHUNK_SIZE,
                                              'num_rows_vis_ir': CHUNK_SIZE})

            self.nc = self.nc.rename({'num_columns_vis_ir': 'x', 'num_rows_vis_ir': 'y'})

        # Obtain some area definition attributes
        equatorial_radius = (self.nc.attrs['equatorial_radius'] * 1000.)
        polar_radius = (self.nc.attrs['north_polar_radius'] * 1000 + self.nc.attrs['south_polar_radius']*1000) * 0.5
        ssp_lon = self.nc.attrs['longitude_of_SSP']
        self.mda['projection_parameters'] = {'a': equatorial_radius,
                                             'b': polar_radius,
                                             'h': 35785831.00,
                                             'ssp_longitude': ssp_lon}

        self.mda['number_of_lines'] = int(self.nc.dims['y'])
        self.mda['number_of_columns'] = int(self.nc.dims['x'])

    def get_dataset(self, dataset_id, dataset_info):

        channel = dataset_id.name
        i = list(CHANNEL_NAMES.values()).index(channel)
        dataset = self.nc[dataset_info['nc_key']]

        # Temp array to store channel data before assignment to data set
        arr1 = xr.DataArray(self.nc[dataset_info['nc_key']],
                            dims=['y', 'x']).where(self.nc[dataset_info['nc_key']].values != 0).astype(np.float32)

        dataset.attrs.update(dataset_info)

        # Calibrate the data as needed
        # MPEF MSG calibration coeffiencts (gain and count)
        offset = dataset.attrs['add_offset']
        gain = dataset.attrs['scale_factor']
        self.platform_id = int(self.nc.attrs['satellite_id'])
        cal_type = self.nc['planned_chan_processing'].values[i]

        # Correct for the scan line order
        arr1 = da.flipud(arr1)

        if dataset_id.calibration == 'counts':
            return dataset

        if dataset_id.calibration in ['radiance', 'reflectance', 'brightness_temperature']:
            arr1 = self._convert_to_radiance(arr1, gain, offset)

        if dataset_id.calibration == 'reflectance':
            solar_irradiance = CALIB[int(self.platform_id)][channel]["F"]
            arr1 = self._vis_calibrate(arr1, solar_irradiance)

        elif dataset_id.calibration == 'brightness_temperature':
            arr1 = self._ir_calibrate(arr1, channel, cal_type)

        dataset.values = da.from_array(arr1, chunks=(CHUNK_SIZE, CHUNK_SIZE))
        return dataset

    def get_area_def(self, dataset_id):
        a = self.mda['projection_parameters']['a']
        b = self.mda['projection_parameters']['b']
        h = self.mda['projection_parameters']['h']
        lon_0 = self.mda['projection_parameters']['ssp_longitude']

        proj_dict = {'a': float(a),
                     'b': float(b),
                     'lon_0': float(lon_0),
                     'h': float(h),
                     'proj': 'geos',
                     'units': 'm'}

        if dataset_id.name == 'HRV':
            nlines = self.mda['hrv_number_of_lines']
            ncols = self.mda['hrv_number_of_columns']
        else:
            nlines = self.mda['number_of_lines']
            ncols = self.mda['number_of_columns']
        area = geometry.AreaDefinition(
             'some_area_name',
             "On-the-fly area",
             'geosmsg',
             proj_dict,
             ncols,
             nlines,
             self.get_area_extent(dataset_id))

        return area

    def get_area_extent(self, dsid):
        if dsid.name != 'HRV':

            # following calculations assume grid origin is south-east corner
            # section 7.2.4 of MSG Level 1.5 Image Data Format Description
            origins = {0: 'NW', 1: 'SW', 2: 'SE', 3: 'NE'}
            grid_origin = self.nc.attrs['vis_ir_grid_origin']
            grid_origin = int(grid_origin, 16)
            if grid_origin != 2:
                raise NotImplementedError(
                    'Grid origin not supported number: {}, {} corner'
                    .format(grid_origin, origins[grid_origin])
                )

            center_point = 3712/2

            north = self.nc.attrs['northern_line_planned']
            east = self.nc.attrs['eastern_line_planned']
            west = self.nc.attrs['western_line_planned']
            south = self.nc.attrs['southern_line_planned']

            column_step = self.nc.attrs['vis_ir_column_dir_grid_step'] * 1000.0

            line_step = self.nc.attrs['vis_ir_line_dir_grid_step'] * 1000.0
            # section 3.1.4.2 of MSG Level 1.5 Image Data Format Description
            earth_model = int(self.nc.attrs['type_of_earth_model'], 16)
            if earth_model == 2:
                ns_offset = 0  # north +ve
                we_offset = 0  # west +ve
            elif earth_model == 1:
                ns_offset = -0.5  # north +ve
                we_offset = 0.5  # west +ve
            else:
                raise NotImplementedError(
                    'unrecognised earth model: {}'.format(earth_model)
                )
            # section 3.1.5 of MSG Level 1.5 Image Data Format Description
            ll_c = (center_point - west - 0.5 + we_offset) * column_step
            ll_l = (south - center_point - 0.5 + ns_offset) * line_step
            ur_c = (center_point - east + 0.5 + we_offset) * column_step
            ur_l = (north - center_point + 0.5 + ns_offset) * line_step
            area_extent = (ll_c, ll_l, ur_c, ur_l)

        else:

            raise NotImplementedError('HRV not supported!')

        return area_extent


class NCSEVIRIHRVFileHandler():
    pass
