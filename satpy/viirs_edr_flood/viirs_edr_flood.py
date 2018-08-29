from satpy.readers.hdf4_utils import HDF4FileHandler


class VIIRSEDRFlood(HDF4FileHandler):
    def get_metadata(self, data, ds_info):
        metadata = {}
        metadata.update(data.attrs)
        metadata.update(ds_info) 
        metadata.update({
            'sensor': self['/attr/SensorIdentifyCode'],
            'platform_name' : self['/attr/Satellitename'],
         #  'resolution' : self['/attr/Resolution'],
            'start_time' : self.start_time(),
            'end_time' : self.end_time()
        })

        return metadata
        
    def get_dataset(self, ds_id, ds_info):
        data = self[ds_id.name]
        
        if ds_id.resolution:
            data.attrs['resolution'] = ds_id.resolution

        data.attrs = self.get_metadata(data, ds_info)
        
        fill = data.attrs.get('_Fillvalue')
        offset = data.attrs.get('add_offset')
        scale_factor = data.attrs.get('scale_factor')

        data = data.where(data != fill)
        if scale_factor is not None and offset is not None:
            data *= scale_factor
            data += offset

        return data
        
    @property
    def start_time(self):
       return self.filename_info['start_time'] 

    @property
    def end_time(self):
        return self.filename_info.get('end_time', self.start_time)


              



