import logging
from datetime import timedelta
import numpy as np
import xarray as xr
import dask.array as da

from satpy.resample import get_area_def

import eccodes as ec

from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger('BufrProductClasses')


sub_sat_dict = {"E0000": 0.0, "E0415": 41.5, "E0095": 9.5}
seg_area_dict = {"E0000": 'seviri_0deg', "E0415": 'seviri_iodc', "E0095": 'seviri_rss'}


class MSGBUFRFileHandler(BaseFileHandler):
    """File handler for MSG BUFR data"""
    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        super(MSGBUFRFileHandler, self).__init__(filename,
                                                 filename_info,
                                                 filetype_info)
        self.platform_name = filename_info['satellite']
        self.subsat = filename_info['subsat']
        self.rc_start = filename_info['start_time']
        self.filename = filename
        self.mda = {}

        self.ssp_lon = sub_sat_dict[filename_info['subsat']]

        # use the following keys to determine the segment size
        # for future non mpef bufr files maybe we need to use an index here?
        segw = self.get_attribute(0, '#1#segmentSizeAtNadirInXDirection', 1)/3000
        segh = self.get_attribute(0, '#1#segmentSizeAtNadirInYDirection', 1)/3000

        # here we get the latiude and longitude arrays used for the segment geolocation
        lats = self.get_array(0, 'latitude')
        lons = self.get_array(0, 'longitude')

        # Use the subsat point to determine the area definition to use for the geo location
        # reset the wight and height based on the segment size
        adef = get_area_def(seg_area_dict[self.subsat])
        adef.height = int(3712/segh)
        adef.width = int(3712/segw)
        # convert the lons/lats to rows and columns
        self.rows, self.cols = adef.lonlat2colrow(lons, lats)

        # Some bufr products may return a list of segment sizes so we use the largest
        # one to then calculate the number of lines and columns
        if isinstance(segw, (list, np.ndarray)):
            self.Nx = int(np.ceil(3712.0/max(segw)))
        else:
            self.Nx = int(np.ceil(3712.0/segw))

        if isinstance(segh, (list, np.ndarray)):
            self.Ny = int(np.ceil(3712.0/max(segh)))
        else:
            self.Ny = int(np.ceil(3712.0/segh))

    @property
    def keys(self):
        """Get all of the keys present in the BUFR file"""
        fh = open(self.filename)

        bufr = ec.codes_bufr_new_from_file(fh)
        ec.codes_set(bufr, 'unpack', 1)

        key_arr = []
        iterid = ec.codes_bufr_keys_iterator_new(bufr)
        while ec.codes_bufr_keys_iterator_next(iterid):
            key_arr.append(ec.codes_bufr_keys_iterator_get_name(iterid))
        ec.codes_bufr_keys_iterator_delete(iterid)
        ec.codes_release(bufr)

        fh.close()
        return key_arr

    @property
    def start_time(self):
        return self.rc_start

    @property
    def end_time(self):
        return self.rc_start+timedelta(minutes=15)

    def get_array(self, index, key='None', mnbr=None):
        ''' Get ancilliary array data eg latitudes and longitudes '''
        try:
            key_arr = self.keys
        except AssertionError:
            raise AssertionError('Unable to determine bufr product keys')

        with open(self.filename) as fh:
            fh = open(self.filename)
            if key != 'None':
                parameter = key
            else:
                parameter = key_arr[index + 20]

            if mnbr is not None:
                bufr = ec.codes_bufr_new_from_file(fh)
                ec.codes_set(bufr, 'unpack', 1)
                arr = ec.codes_get_array(bufr, parameter)
                ec.codes_release(bufr)

            else:
                msgCount = 0
                while True:

                    bufr = ec.codes_bufr_new_from_file(fh)
                    if bufr is None:
                        break

                    ec.codes_set(bufr, 'unpack', 1)
                    # if is the first message initialise our final array with
                    # the number of subsets contained in the first message
                    if (msgCount == 0):
                        arr = np.zeros(ec.codes_get(bufr, 'numberOfSubsets'), np.float)
                        arr[:] = ec.codes_get_array(bufr, parameter, float)

                    else:
                        tmpArr = np.zeros(ec.codes_get(bufr, 'numberOfSubsets'), np.float)
                        tmpArr[:] = ec.codes_get_array(bufr, parameter, float)
                        arr = np.concatenate((arr, tmpArr))

                    msgCount = msgCount+1
                    ec.codes_release(bufr)

        fh.close()
        if arr.size == 1:
            arr = arr[0]

        return arr

    def get_attribute(self, index, key='None', mnbr=None):
        ''' Get BUFR attributes '''
        try:
            key_arr = self.keys
        except AssertionError:
            raise AssertionError('Unable to determine bufr product keys')

        fh = open(self.filename, "rb")

        if key != 'None':
            parameter = key
        else:
            parameter = key_arr[index + 22]

        if mnbr is not None:
            bufr = ec.codes_bufr_new_from_file(fh)
            ec.codes_set(bufr, 'unpack', 1)
            arr = ec.codes_get_array(bufr, parameter)
            ec.codes_release(bufr)

        fh.close()

        if arr.size == 1:
            arr = arr[0]

        return arr

    def get_dataset(self, dsid, info):
        ''' here we loop through the BUFR file and for the required key
        append the data vlues to the dataset area'''

        arr2 = np.empty((self.Ny, self.Nx)).astype(np.float)
        arr2.fill(np.nan)

        # try:
        #    key_arr = self.keys
        # except AssertionError:
        #    raise AssertionError('Unable to determine bufr product keys')

        with open(self.filename, "rb") as fh:
            fh = open(self.filename, "rb")

            key = info['key']

            parameter = key

            msgCount = 0
            while True:

                bufr = ec.codes_bufr_new_from_file(fh)
                if bufr is None:
                    break

                ec.codes_set(bufr, 'unpack', 1)
                # if is the first message initialise our final array with
                # the number of subsets contained in the first message
                if (msgCount == 0):
                    arr = np.zeros(ec.codes_get(bufr, 'numberOfSubsets'), np.float)
                    arr[:] = ec.codes_get_array(bufr, parameter, float)

                else:
                    tmpArr = np.zeros(ec.codes_get(bufr, 'numberOfSubsets'), np.float)
                    tmpArr[:] = ec.codes_get_array(bufr, parameter, float)
                    arr = np.concatenate((arr, tmpArr))

                msgCount = msgCount+1
                ec.codes_release(bufr)

        fh.close()

        if arr.size == 1:
            arr = arr[0]

        arr[arr <= 0] = np.nan

        arr2[self.cols, self.rows] = da.from_array(arr, chunks=(1000))

        xarr = xr.DataArray(arr2, dims=['y', 'x'])

        if xarr is None:
            dataset = None
        else:
            dataset = xarr

            dataset.attrs['units'] = info['units']
            dataset.attrs['wavelength'] = info['wavelength']
            dataset.attrs['standard_name'] = info['standard_name']
            dataset.attrs['platform_name'] = self.platform_name
            dataset.attrs['sensor'] = 'seviri'

        return dataset
