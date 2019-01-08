from datetime import datetime
from satpy.readers.hdf5_utils import HDF5FileHandler
from pyspectral.blackbody import blackbody_wn_rad2temp as rad2temp
import numpy as np
import xarray as xr
import dask.array as da
import logging


class VIRR_L1B(HDF5FileHandler):
    """VIRR_L1B reader."""

    def __init__(self, filename, filename_info, filetype_info):
        super(VIRR_L1B, self).__init__(filename, filename_info, filetype_info)
        logging.debug('day/night flag for {0}: {1}'.format(filename, self['/attr/Day Or Night Flag']))
        self.geolocation_prefix = filetype_info['geolocation_prefix']
        self.platform_id = filename_info['platform_id']
        self.l1b_prefix = 'Data/'
        self.wave_number = 'Emissive_Centroid_Wave_Number'
        if filename_info['platform_id'] == 'FY3B':
            self.l1b_prefix = ''
            self.wave_number = 'Emmisive_Centroid_Wave_Number'

    def get_dataset(self, dataset_id, ds_info):
        file_key = self.geolocation_prefix + ds_info.get('file_key', dataset_id.name)
        if self.platform_id == 'FY3B':
            file_key = file_key.replace('Data/', '')
        data = self.get(file_key)
        if data is None:
            logging.error('File key "{0}" could not be found in file {1}'.format(file_key, self.filename))
        band_index = ds_info.get('band_index')
        if band_index is not None:
            data = data[band_index]
            data = data.where((data >= self[file_key + '/attr/valid_range'][0]) &
                              (data <= self[file_key + '/attr/valid_range'][1]))
            if 'E' in dataset_id.name:
                slope = self[self.l1b_prefix + 'Emissive_Radiance_Scales'].data[:, band_index][:, np.newaxis]
                intercept = self[self.l1b_prefix + 'Emissive_Radiance_Offsets'].data[:, band_index][:, np.newaxis]
                radiance_data = rad2temp(self['/attr/' + self.wave_number][band_index] * 100,
                                         (data * slope + intercept) * 1e-5)
                data = xr.DataArray(da.from_array(radiance_data, data.chunks),
                                    coords=data.coords, dims=data.dims, name=data.name, attrs=data.attrs)
            elif 'R' in dataset_id.name:
                slope = self['/attr/RefSB_Cal_Coefficients'][0::2]
                intercept = self['/attr/RefSB_Cal_Coefficients'][1::2]
                data = data * slope[band_index] + intercept[band_index]
        else:
            data = data.where((data >= self[file_key + '/attr/valid_range'][0]) &
                              (data <= self[file_key + '/attr/valid_range'][1]))
            data = self[file_key + '/attr/Intercept'] + self[file_key + '/attr/Slope'] * data
        new_dims = {old: new for old, new in zip(data.dims, ('y', 'x'))}
        data = data.rename(new_dims)
        data.attrs.update({'platform_name': self['/attr/Satellite Name'],
                           'sensor': self['/attr/Sensor Identification Code']})
        data.attrs.update(ds_info)
        units = self.get(file_key + '/attr/units')
        if units is not None and str(units).lower() != 'none':
            data.attrs.update({'units': self.get(file_key + '/attr/units')})
        elif data.attrs.get('calibration') == 'reflectance':
            data.attrs.update({'units': '%'})
        else:
            data.attrs.update({'units': '1'})
        return data

    @property
    def start_time(self):
        start_time = self['/attr/Observing Beginning Date'] + 'T' + self['/attr/Observing Beginning Time'] + 'Z'
        return datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        end_time = self['/attr/Observing Ending Date'] + 'T' + self['/attr/Observing Ending Time'] + 'Z'
        return datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S.%fZ')
