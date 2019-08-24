import logging
from datetime import datetime

import numpy as np
import xarray as xr

from pyresample import geometry
from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE

logger = logging.getLogger(__name__)

PLATFORM_NAMES = {
    'G16': 'GOES-16',
    'G17': 'GOES-17',
}

#class NC_GLM_L2_LCFA(BaseFileHandler): — add this with glmtools


class NC_GLM_L2_IMAGERY(BaseFileHandler):
    def __init__(self, filename, filename_info, filetype_info):
        super(NC_GLM_L2_IMAGERY, self).__init__(filename, filename_info, filetype_info)
        # xarray's default netcdf4 engine. 
        # It includes handling of the _Unsigned attribute, so that the default
        # mask_and_scale behavior is correct.
        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=True,
                                  chunks={'x': CHUNK_SIZE, 'y': CHUNK_SIZE})
        # self.nc = self.nc.rename({'t': 'time'})
        platform_shortname = filename_info['platform_shortname']
        self.platform_name = PLATFORM_NAMES.get(platform_shortname)
        self.sensor = 'glm'
        self.nlines, self.ncols = self.nc["DQF"].shape
        self.coords = {}

    def get_shape(self, key, info):
        """Get the shape of the data."""
        return self.nlines, self.ncols
    
    def get_area_def(self, key):
        """Get the area definition of the data at hand."""
        projection = self.nc["goes_imager_projection"]
        a = projection.attrs['semi_major_axis']
        h = projection.attrs['perspective_point_height']
        b = projection.attrs['semi_minor_axis']

        lon_0 = projection.attrs['longitude_of_projection_origin']
        sweep_axis = projection.attrs['sweep_angle_axis'][0]

        # x and y extents in m
        h = float(h)
        x = self['x']
        y = self['y']
        x_l = h * x[0]
        x_r = h * x[-1]
        y_l = h * y[-1]
        y_u = h * y[0]
        x_half = (x_r - x_l) / (self.ncols - 1) / 2.
        y_half = (y_u - y_l) / (self.nlines - 1) / 2.
        area_extent = (x_l - x_half, y_l - y_half, x_r + x_half, y_u + y_half)

        proj_dict = {'a': float(a),
                     'b': float(b),
                     'lon_0': float(lon_0),
                     'h': h,
                     'proj': 'geos',
                     'units': 'm',
                     'sweep': sweep_axis}

        # GLM imagery reuses the ABI imagery definitions.
        area = geometry.AreaDefinition(
            self.nc.attrs.get('orbital_slot', 'abi_geos'),
            self.nc.attrs.get('spatial_resolution', 'ABI L1B file area'),
            'abi_geos',
            proj_dict,
            self.ncols,
            self.nlines,
            np.asarray(area_extent))

        return area

    @property
    def start_time(self):
        return datetime.strptime(self.nc.attrs['time_coverage_start'], '%Y-%m-%dT%H:%M:%SZ')

    @property
    def end_time(self):
        return datetime.strptime(self.nc.attrs['time_coverage_end'], '%Y-%m-%dT%H:%M:%SZ')

    def __getitem__(self, item):
        """Wrapper around `self.nc[item]`. Unlike ABI, we like what xarray
        does with our data, so just pass it through.
        """
        logger.debug(item)
        # logger.debug(self.nc)
        return self.nc[item]

    def get_dataset(self, key, info):
        """Load a dataset."""
        logger.debug('Reading in get_dataset %s.', key.name)
        res = self[key.name]

        # convert to satpy standard units
        if res.attrs['units'] == '1':
            res *= 100
            res.attrs['units'] = '%'

        res.attrs.update({'platform_name': self.platform_name,
                          'sensor': self.sensor,
                          'satellite_latitude': float(self['nominal_satellite_subpoint_lat']),
                          'satellite_longitude': float(self['nominal_satellite_subpoint_lon']),
                          # 'satellite_altitude': float(self['nominal_satellite_height']),
                      })

        # Add orbital parameters
        projection = self.nc["goes_imager_projection"]
        res.attrs['orbital_parameters'] = {
            'projection_longitude': float(projection.attrs['longitude_of_projection_origin']),
            'projection_latitude': float(projection.attrs['latitude_of_projection_origin']),
            'projection_altitude': float(projection.attrs['perspective_point_height']),
            'satellite_nominal_latitude': float(self['nominal_satellite_subpoint_lat']),
            'satellite_nominal_longitude': float(self['nominal_satellite_subpoint_lon']),
            # 'satellite_nominal_altitude': float(self['nominal_satellite_height']),
            'yaw_flip': False,
        }

        res.attrs.update(key.to_dict())
        # remove attributes that could be confusing later
        res.attrs.pop('_FillValue', None)
        res.attrs.pop('scale_factor', None)
        res.attrs.pop('add_offset', None)
        res.attrs.pop('_Unsigned', None)
        res.attrs.pop('ancillary_variables', None)  # Can't currently load DQF
        # add in information from the filename that may be useful to the user
        # for key in ('observation_type', 'scene_abbr', 'scan_mode', 'platform_shortname'):
        for key in ('scene_abbr', 'scan_mode', 'platform_shortname'):
            res.attrs[key] = self.filename_info[key]
        # copy global attributes to metadata
        for key in ('scene_id', 'orbital_slot', 'instrument_ID', 'production_site', 'timeline_ID'):
            res.attrs[key] = self.nc.attrs.get(key)
        # only include these if they are present
        for key in ('fusion_args',):
            if key in self.nc.attrs:
                res.attrs[key] = self.nc.attrs[key]

        return res
        
    def __del__(self):
        try:
            self.nc.close()
        except (IOError, OSError, AttributeError):
            pass
    