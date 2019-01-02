from datetime import datetime
import numpy as np
from pyresample.geometry import SwathDefinition
from satpy.readers.hdf5_utils import HDF5FileHandler


class VIRR_L1B(HDF5FileHandler):
    """VIRR_L1B reader."""

    def __init__(self, filename, filename_info, filetype_info):
        print('HERE!!!')
        super(VIRR_L1B, self).__init__(filename, filename_info, filetype_info)
        for key, val in self.file_content.items():
            print(key + ':', val)
        self.lons = None
        self.lats = None
        if filename_info['platform_id'] == 'FY3B':
            self.extension = ''
            self.ECWN = 'Emmisive_Centroid_Wave_Number'
            if filename_info['meta_file'] == 'L1B':
                self.lons = self['Longitude']
                self.lats = self['Latitude']
        elif filename_info['platform_id'] == 'FY3C':
            self.extension = 'Data/'
            self.ECWN = 'Emmisive_Centroid_Wave_Number'
            if filename_info['meta_file'] == 'GEOXX':
                self.lons = self['Geolocation/Longitude']
                self.lats = self['Geolocation/Latitude']
        # self.satellite_longitude = self['/attr/Orbit Point Longitude']
        # self.satellite_latitude = self['/attr/Orbit Point Latitude']

    def plank(self, wavenumber, radiance):
        return 1.4387752 * wavenumber / np.log(1.191042e-5 * wavenumber**3 / radiance + 1)

    def get_dataset(self, dataset_id, ds_info):
        # if self.filename_info['meta_file'] != 'L1B':
        #     return
        file_key = self.extension + ds_info.get('file_key', dataset_id.name)
        if self.filename_info['meta_file'] == 'GEOXX':
            file_key = file_key.replace('Data/', '')
        if self.filename_info['meta_file'] == 'L1B':
            file_key = file_key.replace('Geolocation/', '')
        data = self.get(file_key)
        if data is None:
            return
        band_index = ds_info.get('band_index')
        print(self.filename_info, file_key)
        if band_index is not None:
            slope = {'E': self[self.extension + 'Emissive_Radiance_Scales'].data,
                     'R': self['/attr/RefSB_Cal_Coefficients'][::2]}
            intercept = {'E': self[self.extension + 'Emissive_Radiance_Offsets'].data,
                         'R': self['/attr/RefSB_Cal_Coefficients'][1::2]}
            data = data[band_index]
            new_dims = {old: new for old, new in zip(data.dims, ('y', 'x'))}
            data = data.rename(new_dims)
            data = data.where((data >= self[file_key + '/attr/valid_range'][0]) &
                              (data <= self[file_key + '/attr/valid_range'][1]))
            if 'E' in dataset_id.name:
                slope = slope['E'][:, band_index][:, np.newaxis]
                intercept = intercept['E'][:, band_index][:, np.newaxis]
                data = self.plank(self['/attr/' + self.ECWN][band_index], data * slope + intercept)
            elif 'R' in dataset_id.name:
                data = data * slope['R'][band_index] + intercept['R'][band_index]
        else:
            data = self[file_key + '/attr/Intercept'] + self[file_key + '/attr/Slope'] * data

        data.attrs.update({'platform_name': self['/attr/Satellite Name'],
                           'sensor': self['/attr/Sensor Identification Code']})
        data.attrs.update(ds_info)
        if data.attrs.get('units') is None or data.attrs['units'] == 'none':
            data.attrs['units'] = '1'
        print(data)
        return data

    def get_area_def(self, dsid):
        lons = self.lons
        lats = self.lats
        if lons is None or lats is None:
            return
        lons = lons.where((lons >= lons.attrs['valid_range'][0]) & (lons <= lons.attrs['valid_range'][1]))
        # lons = lons.where(lons != np.nan, lons, -999.0)
        lats = lats.where((lats >= lats.attrs['valid_range'][0]) & (lats <= lats.attrs['valid_range'][1]))
        # lats = lats.where(lats != np.nan, lats, -999.0)
        area = SwathDefinition(lons=lons, lats=lats).compute_optimal_bb_area()

        x_size = self['/attr/End Pixel Number'] - (self['/attr/Begin Pixel Number'] - 1)
        y_size = self['/attr/End Line Number'] - (self['/attr/Begin Line Number'] - 1)
        area.y_size = y_size
        area.x_size = x_size
        area.shape = (y_size, x_size)
        print(area)
        return area

    @property
    def satellite_longitude(self):
        return self['/attr/Orbit Point Longitude'][0]

    @property
    def satellite_latitude(self):
        return self['/attr/Orbit Point Latitude'][0]

    @property
    def satellite_altitude(self):
        # 6.673 * 5.98 * 1e13 / self['/attr/MeanMotion']**2?
        return self['/attr/MeanAnomaly']

    @property
    def start_time(self):
        start_time = (self['/attr/Observing Beginning Date'] + 'T' + self['/attr/Observing Beginning Time'] + 'Z')
        return datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S.%fZ')

    @property
    def end_time(self):
        end_time = (self['/attr/Observing Ending Date'] + 'T' + self['/attr/Observing Ending Time'] + 'Z')
        return datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S.%fZ')
