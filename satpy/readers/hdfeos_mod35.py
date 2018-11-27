import logging
from datetime import datetime

import numpy as np
from pyhdf.error import HDF4Error
from pyhdf.SD import SD, SDC

import dask.array as da
import xarray.ufuncs as xu
import xarray as xr
from satpy import CHUNK_SIZE
from satpy.dataset import DatasetID
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.hdf4_utils import from_sds

logger = logging.getLogger(__name__)

import time

class HDFEOSFileReader(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(HDFEOSFileReader, self).__init__(filename, filename_info, filetype_info)
        self.filename = filename
        try:
            self.sd = SD(str(self.filename), SDC.READ)
        except HDF4Error as err:
            raise ValueError("Could not load data from " + str(self.filename)
                             + ": " + str(err))
        self.metadata = self.read_mda(self.sd.attributes()['CoreMetadata.0'])
        self.metadata.update(self.read_mda(
            self.sd.attributes()['StructMetadata.0']))
        self.metadata.update(self.read_mda(
            self.sd.attributes()['ArchiveMetadata.0']))

    @property
    def start_time(self):
        date = (self.metadata['INVENTORYMETADATA']['RANGEDATETIME']['RANGEBEGINNINGDATE']['VALUE'] + ' ' +
                self.metadata['INVENTORYMETADATA']['RANGEDATETIME']['RANGEBEGINNINGTIME']['VALUE'])
        return datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')

    @property
    def end_time(self):
        date = (self.metadata['INVENTORYMETADATA']['RANGEDATETIME']['RANGEENDINGDATE']['VALUE'] + ' ' +
                self.metadata['INVENTORYMETADATA']['RANGEDATETIME']['RANGEENDINGTIME']['VALUE'])
        return datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')

    def read_mda(self, attribute):
        lines = attribute.split('\n')
        mda = {}
        current_dict = mda
        path = []
        for line in lines:
            if "gdas1" in line:
                continue
            if not line:
                continue
            if line == 'END':
                break
            keyval = line.split('=')
            if len(keyval) <= 1:
                continue
            key, val =  keyval
            key = key.strip()
            val = val.strip()
            try:
                val = eval(val)
            except NameError:
                pass
            if key in ['GROUP', 'OBJECT']:
                new_dict = {}
                path.append(val)
                current_dict[val] = new_dict
                current_dict = new_dict
            elif key in ['END_GROUP', 'END_OBJECT']:
                if val != path[-1]:
                    raise SyntaxError
                path = path[:-1]
                current_dict = mda
                for item in path:
                    current_dict = current_dict[item]
            elif key in ['CLASS', 'NUM_VAL']:
                pass
            else:
                current_dict[key] = val
        return mda


class HDFEOSGeoReader(HDFEOSFileReader):

    def __init__(self, filename, filename_info, filetype_info):
        HDFEOSFileReader.__init__(self, filename, filename_info, filetype_info)

        ds = self.metadata['INVENTORYMETADATA'][
            'COLLECTIONDESCRIPTIONCLASS']['SHORTNAME']['VALUE']
        if ds.endswith('D03'):
            self.resolution = 1000
        else:
            self.resolution = 5000
        self.cache = {}
        self.cache[250] = {}
        self.cache[250]['lons'] = None
        self.cache[250]['lats'] = None

        self.cache[500] = {}
        self.cache[500]['lons'] = None
        self.cache[500]['lats'] = None

        self.cache[1000] = {}
        self.cache[1000]['lons'] = None
        self.cache[1000]['lats'] = None

    def get_dataset(self, key, info, out=None, xslice=None, yslice=None):
        """Get the dataset designated by *key*."""
        if key.name in ['solar_zenith_angle', 'solar_azimuth_angle',
                        'satellite_zenith_angle', 'satellite_azimuth_angle']:

            if key.name == 'solar_zenith_angle':
                var = self.sd.select('SolarZenith')
            if key.name == 'solar_azimuth_angle':
                var = self.sd.select('SolarAzimuth')
            if key.name == 'satellite_zenith_angle':
                var = self.sd.select('SensorZenith')
            if key.name == 'satellite_azimuth_angle':
                var = self.sd.select('SensorAzimuth')

            data = xr.DataArray(from_sds(var, chunks=CHUNK_SIZE),
                                dims=['y', 'x']).astype(np.float32)
            data = data.where(data != var._FillValue)
            data = data * np.float32(var.scale_factor)

            data.attrs = info
            return data

        if key.name not in ['longitude', 'latitude']:
            return

        if (self.cache[key.resolution]['lons'] is None or
                self.cache[key.resolution]['lats'] is None):

            lons_id = DatasetID('longitude',
                                resolution=key.resolution)
            lats_id = DatasetID('latitude',
                                resolution=key.resolution)

            lons, lats = self.load(
                [lons_id, lats_id], interpolate=False, raw=True)
            if key.resolution != self.resolution:
                from geotiepoints.geointerpolator import GeoInterpolator
                lons, lats = self._interpolate([lons, lats],
                                               self.resolution,
                                               lons_id.resolution,
                                               GeoInterpolator)
                lons = np.ma.masked_invalid(np.ascontiguousarray(lons))
                lats = np.ma.masked_invalid(np.ascontiguousarray(lats))
            self.cache[key.resolution]['lons'] = lons
            self.cache[key.resolution]['lats'] = lats

        if key.name == 'latitude':
            data = self.cache[key.resolution]['lats'].filled(np.nan)
            data = xr.DataArray(da.from_array(data, chunks=(CHUNK_SIZE, CHUNK_SIZE)),
                                dims=['y', 'x'])
        else:
            data = self.cache[key.resolution]['lons'].filled(np.nan)
            data = xr.DataArray(da.from_array(data, chunks=(CHUNK_SIZE,
                                                            CHUNK_SIZE)),
                                dims=['y', 'x'])
        data.attrs = info
        return data

    def load(self, keys, interpolate=True, raw=False):
        """Load the data."""
        projectables = []
        for key in keys:
            dataset = self.sd.select(key.name.capitalize())
            fill_value = dataset.attributes()["_FillValue"]
            try:
                scale_factor = dataset.attributes()["scale_factor"]
            except KeyError:
                scale_factor = 1
            data = np.ma.masked_equal(dataset.get(), fill_value) * scale_factor

            # TODO: interpolate if needed
            if (key.resolution is not None and
                    key.resolution < self.resolution and
                    interpolate):
                data = self._interpolate(data, self.resolution, key.resolution)
            if not raw:
                data = data.filled(np.nan)
                data = xr.DataArray(da.from_array(data, chunks=(CHUNK_SIZE,
                                                                CHUNK_SIZE)),
                                    dims=['y', 'x'])
            projectables.append(data)

        return projectables

    @staticmethod
    def _interpolate(data, coarse_resolution, resolution, interpolator=None):
        if resolution == coarse_resolution:
            return data

        if interpolator is None:
            from geotiepoints.interpolator import Interpolator
            interpolator = Interpolator

        logger.debug("Interpolating from " + str(coarse_resolution)
                     + " to " + str(resolution))

        if isinstance(data, (tuple, list, set)):
            lines = data[0].shape[0]
        else:
            lines = data.shape[0]

        if coarse_resolution == 5000:
            coarse_cols = np.arange(2, 1354, 5)
            lines *= 5
            coarse_rows = np.arange(2, lines, 5)

        elif coarse_resolution == 1000:
            coarse_cols = np.arange(1354)
            coarse_rows = np.arange(lines)

        if resolution == 1000:
            fine_cols = np.arange(1354)
            fine_rows = np.arange(lines)
            chunk_size = 10
        elif resolution == 500:
            fine_cols = np.arange(1354 * 2) / 2.0
            fine_rows = (np.arange(lines * 2) - 0.5) / 2.0
            chunk_size = 20
        elif resolution == 250:
            fine_cols = np.arange(1354 * 4) / 4.0
            fine_rows = (np.arange(lines * 4) - 1.5) / 4.0
            chunk_size = 40

        along_track_order = 1
        cross_track_order = 3

        satint = interpolator(data,
                              (coarse_rows, coarse_cols),
                              (fine_rows, fine_cols),
                              along_track_order,
                              cross_track_order,
                              chunk_size=chunk_size)

        satint.fill_borders("y", "x")
        return satint.interpolate()


class HDFEOSBandReader(HDFEOSFileReader):

    #not meeded because resolution is not coded in filename
    res = {"1": 1000,
           "Q": 250,
           "H": 500}

    def __init__(self, filename, filename_info, filetype_info):
        HDFEOSFileReader.__init__(self, filename, filename_info, filetype_info)
        ds = self.metadata['INVENTORYMETADATA'][
            'COLLECTIONDESCRIPTIONCLASS']['SHORTNAME']['VALUE']

        #resolution not coded in shortname of product
        #in mod35 resolution is 1km, 250m mask is encoded in last 2 byte
        #self.resolution = self.res[ds[-3]]

    def get_dataset(self, key, info):
        """Read data from file and return the corresponding projectables."""

        platform_name = self.metadata['INVENTORYMETADATA']['ASSOCIATEDPLATFORMINSTRUMENTSENSOR'][
            'ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER']['ASSOCIATEDPLATFORMSHORTNAME']['VALUE']

        info.update({'platform_name': 'EOS-' + platform_name})
        info.update({'sensor': 'modis'})
        #print(info)

        index, startbit, endbit = info.get("bits", "{}".format(key))
        file_key = info.get("file_key", "{}".format(key))

        subdata = self.sd.select(file_key)

        var_attrs = subdata.attributes()

        array = xr.DataArray(from_sds(subdata, chunks=CHUNK_SIZE)[index,:,:],
                                 dims=['y', 'x']).astype(np.int32)


        # strip bits
        array = bits_stripping(startbit, endbit, array).astype(np.float32)

        array.attrs = info

        #valid_range = var_attrs['valid_range']


        return array

    # These have to be interpolated...
    def get_height(self):
        return self.data.select("Height")

    def get_sunz(self):
        return self.data.select("SolarZenith")

    def get_suna(self):
        return self.data.select("SolarAzimuth")

    def get_satz(self):
        return self.data.select("SensorZenith")

    def get_sata(self):
        return self.data.select("SensorAzimuth")


#right_shift = da.array.frompyfunc(np.right_shift)
#left_shift = da.array.frompyfunc(np.left_shift)


def bits_stripping(bit_start, bit_count, value):
    """Extract specified bit from bit representation of integer value.

    Parameters
    ----------
    bit_start : int
        Starting index of the bits to extract (first bit has index 0)
    bit_count : int
        Number of bits starting from bit_start to extract
    value : int
        Number from which to extract the bits

    Returns
    -------
    int
        Value of the extracted bits

    """

    bitmask = pow(2, bit_start + bit_count) - 1

    return np.right_shift(da.bitwise_and(value, bitmask), bit_start)
#    return da.bitwise_and(value, bitmask)


def BitShiftCombine(byte1, byte2):
    """
    Combine two 8 bit integers to one 16 bit integer by
    concatenating

    Parameters
    ----------
    byte1, byte2 : np.int
        8 bit integers two combine

    Returns
    -------
    np.uint16
        16 bit integer with byte1 as the first 8 bit and byte2 as the second 8 bit
    """
    # make sure byte1 is 16 bit
    twoByte = np.uint16(byte1)
    # shift bits to the left
    twoByte = left_shift(twoByte, 8)  # twoByte << 8
    # concatenate the two bytes
    twoByte = da.bitwise_or(twoByte, byte2).astype(np.uint16)  # casting = "no")

    return (twoByte)


def bit_strip_250m_mask(inputArray):
    """
    Create 250m cloud mask from the last 16 bits of the modis 48 bit array.

    Takes each array element decodes the bits and distributes
    them to new array with 4 times the resolution in x and y direction
    of the input array (each input array element is replaced by 16 elements
    so to speak).

    Parameters
    ----------
    inputArray : np.uint16
        input array which to transform


    Returns
    -------
    np array
        Array with increased resolution and binary mask based on bits of input array

    """

    # create 4x4 array with numbers from 1 to 16 (count of bits)
    pix = np.arange(0, 16).reshape((4, 4)).astype(np.uint16)
    # tile the 4x4 array the size of the input array dimensions
    pix = np.tile(pix, (inputArray.shape[0], inputArray.shape[1]))
    # "interpolate" input array with nearest neighbour with resolution increase 4
    data = np.repeat(np.repeat(inputArray, 4, axis=0), 4, axis=1)
    # decode the corresponding bits to get the 250m mask
    mask250 = bits_stripping(pix, 1, data)

    return (mask250)


if __name__ == '__main__':
    from satpy.utils import debug_on
    debug_on()
    br = HDFEOSBandReader(
        '/media/droplet_data/data/modis_lcrs_2005_2006/mod35/2005/MOD35_L2.A2005001.0740.006.2014340132609.hdf')
    gr = HDFEOSGeoReader(
        '/media/droplet_data/data/modis_lcrs_2005_2006/mod03/2005/MOD03.A2005001.0740.006.2012274114305.hdf')