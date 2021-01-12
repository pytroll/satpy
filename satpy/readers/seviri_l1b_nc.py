#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2019 Satpy developers
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
"""SEVIRI netcdf format reader."""

from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.seviri_base import (SEVIRICalibrationHandler,
                                       CHANNEL_NAMES, SATNUM)
from satpy.readers.eum_base import get_service_mode
import xarray as xr

from satpy.readers._geos_area import get_area_definition, get_geos_area_naming
from satpy import CHUNK_SIZE

import datetime


class NCSEVIRIFileHandler(BaseFileHandler):
    """File handler for NC seviri files.

    **Calibration**

    See :mod:`satpy.readers.seviri_base`. Note that there is only one set of
    calibration coefficients available in the netCDF files and therefore there
    is no `calib_mode` argument.

    """

    def __init__(self, filename, filename_info, filetype_info,
                 ext_calib_coefs=None):
        """Init the file handler."""
        super(NCSEVIRIFileHandler, self).__init__(filename, filename_info, filetype_info)
        self.ext_calib_coefs = ext_calib_coefs or {}
        self.nc = None
        self.mda = {}
        self.reference = datetime.datetime(1958, 1, 1)
        self._read_file()

    @property
    def start_time(self):
        """Get the start time."""
        return self.deltaSt

    @property
    def end_time(self):
        """Get the end time."""
        return self.deltaEnd

    def _read_file(self):
        """Read the file."""
        if self.nc is None:

            self.nc = xr.open_dataset(self.filename,
                                      decode_cf=True,
                                      mask_and_scale=False,
                                      chunks=CHUNK_SIZE)

        # Obtain some area definition attributes
        equatorial_radius = (self.nc.attrs['equatorial_radius'] * 1000.)
        polar_radius = (self.nc.attrs['north_polar_radius'] * 1000 + self.nc.attrs['south_polar_radius'] * 1000) * 0.5
        ssp_lon = self.nc.attrs['longitude_of_SSP']
        self.mda['projection_parameters'] = {'a': equatorial_radius,
                                             'b': polar_radius,
                                             'h': 35785831.00,
                                             'ssp_longitude': ssp_lon}

        self.mda['number_of_lines'] = int(self.nc.dims['num_rows_vis_ir'])
        self.mda['number_of_columns'] = int(self.nc.dims['num_columns_vis_ir'])

        self.mda['hrv_number_of_lines'] = int(self.nc.dims['num_rows_hrv'])
        self.mda['hrv_number_of_columns'] = int(self.nc.dims['num_columns_hrv'])

        self.deltaSt = self.reference + datetime.timedelta(
            days=int(self.nc.attrs['true_repeat_cycle_start_day']),
            milliseconds=int(self.nc.attrs['true_repeat_cycle_start_mi_sec']))

        self.deltaEnd = self.reference + datetime.timedelta(
            days=int(self.nc.attrs['planned_repeat_cycle_end_day']),
            milliseconds=int(self.nc.attrs['planned_repeat_cycle_end_mi_sec']))

        self.north = int(self.nc.attrs['north_most_line'])
        self.east = int(self.nc.attrs['east_most_pixel'])
        self.west = int(self.nc.attrs['west_most_pixel'])
        self.south = int(self.nc.attrs['south_most_line'])

    def get_dataset(self, dataset_id, dataset_info):
        """Get the dataset."""
        channel = dataset_id['name']

        if (channel == 'HRV'):
            self.nc = self.nc.rename({'num_columns_hrv': 'x', 'num_rows_hrv': 'y'})
        else:
            # the first channel of a composite will rename the dimension variable
            # but the later channels will raise a value error as its already been renamed
            # we can just ignore these exceptions
            try:
                self.nc = self.nc.rename({'num_columns_vis_ir': 'x', 'num_rows_vis_ir': 'y'})
            except ValueError:
                pass

        dataset = self.nc[dataset_info['nc_key']]
        dataset.attrs.update(dataset_info)
        self.platform_id = int(self.nc.attrs['satellite_id'])

        # Correct for the scan line order
        dataset = dataset.sel(y=slice(None, None, -1))

        # Calibrate the data as needed
        dataset = self.calibrate(dataset, dataset_id)

        # Update dataset attributes
        dataset.attrs.update(self.nc[dataset_info['nc_key']].attrs)
        dataset.attrs.update(dataset_info)
        dataset.attrs['platform_name'] = "Meteosat-" + SATNUM[self.platform_id]
        dataset.attrs['sensor'] = 'seviri'
        dataset.attrs['orbital_parameters'] = {
            'projection_longitude': self.mda['projection_parameters']['ssp_longitude'],
            'projection_latitude': 0.,
            'projection_altitude': self.mda['projection_parameters']['h']}

        # remove attributes from original file which don't apply anymore
        strip_attrs = ["comment", "long_name", "nc_key", "scale_factor", "add_offset", "valid_min", "valid_max"]
        for a in strip_attrs:
            dataset.attrs.pop(a)

        return dataset

    def calibrate(self, dataset, dataset_id):
        """Calibrate the data."""
        channel = dataset_id['name']
        calibration = dataset_id['calibration']

        if dataset_id['calibration'] == 'counts':
            dataset.attrs['_FillValue'] = 0

        calib = SEVIRICalibrationHandler(
            platform_id=int(self.platform_id),
            channel_name=channel,
            coefs=self._get_calib_coefs(dataset, channel),
            calib_mode='NOMINAL',
            scan_time=self.start_time
        )

        return calib.calibrate(dataset, calibration)

    def _get_calib_coefs(self, dataset, channel):
        """Get coefficients for calibration from counts to radiance."""
        band_idx = list(CHANNEL_NAMES.values()).index(channel)
        offset = dataset.attrs['add_offset'].astype('float32')
        gain = dataset.attrs['scale_factor'].astype('float32')
        # Only one calibration available here
        return {
            'coefs': {
                'NOMINAL': {
                    'gain': gain,
                    'offset': offset
                },
                'EXTERNAL': self.ext_calib_coefs.get(channel, {})
            },
            'radiance_type': self.nc['planned_chan_processing'].values[band_idx]
        }

    def get_area_def(self, dataset_id):
        """Get the area def.

        Note that the AreaDefinition area extents returned by this function for NetCDF data will be slightly
        different compared to the area extents returned by the SEVIRI HRIT reader.
        This is due to slightly different pixel size values when calculated using the data available in the files. E.g.
        for the 3 km grid:

        ``NetCDF:  self.nc.attrs['vis_ir_column_dir_grid_step'] == 3000.4031658172607``
        ``HRIT: np.deg2rad(2.**16 / pdict['lfac']) * pdict['h'] == 3000.4032785810186``

        This results in the Native 3 km full-disk area extents being approx. 20 cm shorter in each direction.

        The method for calculating the area extents used by the HRIT reader (CFAC/LFAC mechanism) keeps the
        highest level of numeric precision and is used as reference by EUM. For this reason, the standard area
        definitions defined in the `areas.yaml` file correspond to the HRIT ones.

        """
        pdict = {}
        pdict['a'] = self.mda['projection_parameters']['a']
        pdict['b'] = self.mda['projection_parameters']['b']
        pdict['h'] = self.mda['projection_parameters']['h']
        pdict['ssp_lon'] = self.mda['projection_parameters']['ssp_longitude']

        area_naming_input_dict = {'platform_name': 'msg',
                                  'instrument_name': 'seviri',
                                  'resolution': int(dataset_id['resolution'])
                                  }
        area_naming = get_geos_area_naming({**area_naming_input_dict,
                                            **get_service_mode('seviri', pdict['ssp_lon'])})

        if dataset_id['name'] == 'HRV':
            pdict['nlines'] = self.mda['hrv_number_of_lines']
            pdict['ncols'] = self.mda['hrv_number_of_columns']
            pdict['a_name'] = area_naming['area_id']
            pdict['a_desc'] = area_naming['description']
            pdict['p_id'] = ""
        else:
            pdict['nlines'] = self.mda['number_of_lines']
            pdict['ncols'] = self.mda['number_of_columns']
            pdict['a_name'] = area_naming['area_id']
            pdict['a_desc'] = area_naming['description']
            pdict['p_id'] = ""

        area = get_area_definition(pdict, self.get_area_extent(dataset_id))

        return area

    def get_area_extent(self, dsid):
        """Get the area extent."""
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

        column_step = self.nc.attrs['vis_ir_column_dir_grid_step'] * 1000.0

        line_step = self.nc.attrs['vis_ir_line_dir_grid_step'] * 1000.0

        # check for Earth model as this affects the north-south and
        # west-east offsets
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
        ll_c = (center_point - self.west - 0.5 + we_offset) * column_step
        ll_l = (self.south - center_point - 0.5 + ns_offset) * line_step
        ur_c = (center_point - self.east + 0.5 + we_offset) * column_step
        ur_l = (self.north - center_point + 0.5 + ns_offset) * line_step
        area_extent = (ll_c, ll_l, ur_c, ur_l)

        return area_extent


class NCSEVIRIHRVFileHandler(BaseFileHandler, SEVIRICalibrationHandler):
    """HRV filehandler."""

    pass
