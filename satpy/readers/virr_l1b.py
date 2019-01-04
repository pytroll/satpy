from datetime import datetime
import numpy as np
import xarray as xr
import dask.array as da
import logging
from satpy.readers.hdf5_utils import HDF5FileHandler
from pyspectral.blackbody import blackbody_wn_rad2temp as rad2temp


class VIRR_L1B(HDF5FileHandler):
    """VIRR_L1B reader."""

    def __init__(self, filename, filename_info, filetype_info):
        super(VIRR_L1B, self).__init__(filename, filename_info, filetype_info)
        logging.info('day/night flag for {0}: {1}'.format(filename, self['/attr/Day Or Night Flag']))
        for key, val in self.file_content.items():
            print(key + ':', val)
        print()
        self.geolocation_extension = filetype_info['geolocation_extension']
        self.platform_id = filename_info['platform_id']
        self.l1b_extension = 'Data/'
        self.wave_number = 'Emissive_Centroid_Wave_Number'
        if filename_info['platform_id'] == 'FY3B':
            self.l1b_extension = ''
            self.wave_number = 'Emmisive_Centroid_Wave_Number'

    def get_dataset(self, dataset_id, ds_info):
        file_key = self.geolocation_extension + ds_info.get('file_key', dataset_id.name)
        if self.platform_id == 'FY3B':
            file_key = file_key.replace('Data/', '')
        # print('DAY/NIGHT: ', self['/attr/Day Or Night Flag'])
        # print(file_key)
        # print(self.filename_info)
        # print(self.filetype_info)
        # print()
        data = self.get(file_key)
        if data is None:
            raise ValueError('{0} cannot be found')
        band_index = ds_info.get('band_index')
        if band_index is not None:
            data = data[band_index]
            new_dims = {old: new for old, new in zip(data.dims, ('y', 'x'))}
            data = data.rename(new_dims)
            data = data.where((data >= self[file_key + '/attr/valid_range'][0]) &
                              (data <= self[file_key + '/attr/valid_range'][1]))
            if 'E' in dataset_id.name:
                slope = self[self.l1b_extension + 'Emissive_Radiance_Scales'].data[:, band_index][:, np.newaxis]
                intercept = self[self.l1b_extension + 'Emissive_Radiance_Offsets'].data[:, band_index][:, np.newaxis]
                radiance_data = rad2temp(self['/attr/' + self.wave_number][band_index] * 100,
                                         (data * slope + intercept) * 1e-5)
                data = xr.DataArray(da.from_array(radiance_data, data.chunks),
                                    coords=data.coords, dims=data.dims, name=data.name, attrs=data.attrs)
            elif 'R' in dataset_id.name:
                slope = self['/attr/RefSB_Cal_Coefficients'][0::2]
                intercept = self['/attr/RefSB_Cal_Coefficients'][1::2]
                data = data * slope[band_index] + intercept[band_index]
        else:
            data = data.attrs['Intercept'] + data.attrs['Slope'] * data

        data.attrs.update({'platform_name': self['/attr/Satellite Name'],
                           'sensor': self['/attr/Sensor Identification Code']})
        data.attrs.update(ds_info)
        if self[file_key].attrs.get('units') is not None:
            data.attrs.update({'units': self[file_key].attrs['units'].decode("utf-8")})
        elif data.attrs.get('calibration') == 'reflectance':
            data.attrs.update({'units': '%'})
        else:
            data.attrs.update({'units': 1})
        print(data)
        return data

    @property
    def start_time(self):
        start_time = self['/attr/Observing Beginning Date'] + 'T' + self['/attr/Observing Beginning Time'] + 'Z'
        return datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        end_time = self['/attr/Observing Ending Date'] + 'T' + self['/attr/Observing Ending Time'] + 'Z'
        return datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S.%fZ')
