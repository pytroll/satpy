from satpy.readers.hdf4_utils import HDF4FileHandler
from pyresample import geometry
import numpy as np


class VIIRSEDRFlood(HDF4FileHandler):
    @property
    def start_time(self):
        return self.filename_info['start_time']

    @property
    def end_time(self):
        return self.filename_info.get('end_time', self.start_time)

    @property
    def sensor_name(self):
        sensor = self['/attr/SensorIdentifyCode']
        if isinstance(sensor, np.ndarray):
            return str(sensor.astype(str))
        return sensor

    @property
    def platform_name(self):
        platform_name = self['/attr/Satellitename']
        if isinstance(platform_name, np.ndarray):
            return str(platform_name.astype(str))
        return platform_name

    def get_metadata(self, data, ds_info):
        metadata = {}
        metadata.update(data.attrs)
        metadata.update(ds_info)
        metadata.update({
            'sensor': self.sensor_name,
            'platform_name': self.platform_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
        })

        return metadata

    def get_dataset(self, ds_id, ds_info):
        data = self[ds_id.name]

        data.attrs = self.get_metadata(data, ds_info)

        fill = data.attrs.pop('_Fillvalue')
        offset = data.attrs.get('add_offset')
        scale_factor = data.attrs.get('scale_factor')

        data = data.where(data != fill)
        if scale_factor is not None and offset is not None:
            data *= scale_factor
            data += offset

        return data

    def get_area_def(self, ds_id):
        data = self[ds_id.name]

        proj_dict = {
            'proj': 'latlong',
            'datum': 'WGS84',
            'ellps': 'WGS84',
            'no_defs': True
        }

        area_extent = [data.attrs.get('ProjectionMinLongitude'), data.attrs.get('ProjectionMinLatitude'),
                       data.attrs.get('ProjectionMaxLongitude'), data.attrs.get('ProjectionMaxLatitude')]

        area = geometry.AreaDefinition(
            'viirs_flood_area',
            'name_of_proj',
            'id_of_proj',
            proj_dict,
            int(self.filename_info['dim0']),
            int(self.filename_info['dim1']),
            np.asarray(area_extent)
        )

        return area
