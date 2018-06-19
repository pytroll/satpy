#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010-2014, 2017.

# SMHI,
# Folkborgsvägen 1,
# Norrköping,
# Sweden

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Ronald Scheirer <ronald.scheirer@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Interface to Modis level 1b format.
http://www.icare.univ-lille1.fr/wiki/index.php/MODIS_geolocation
http://www.sciencedirect.com/science?_ob=MiamiImageURL&_imagekey=B6V6V-4700BJP-\
3-27&_cdi=5824&_user=671124&_check=y&_orig=search&_coverDate=11%2F30%2F2002&vie\
w=c&wchp=dGLzVlz-zSkWz&md5=bac5bc7a4f08007722ae793954f1dd63&ie=/sdarticle.pdf
"""

import logging
from datetime import datetime

import numpy as np
from pyhdf.error import HDF4Error
from pyhdf.SD import SD

import dask.array as da
import xarray.ufuncs as xu
import xarray as xr
from satpy import CHUNK_SIZE
from satpy.dataset import DatasetID
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.hdf4_utils import from_sds

logger = logging.getLogger(__name__)


class HDFEOSFileReader(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(HDFEOSFileReader, self).__init__(filename, filename_info, filetype_info)
        try:
            self.sd = SD(str(self.filename))
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
            if not line:
                continue
            if line == 'END':
                break
            key, val = line.split('=')
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

    res = {"1": 1000,
           "Q": 250,
           "H": 500}

    def __init__(self, filename, filename_info, filetype_info):
        HDFEOSFileReader.__init__(self, filename, filename_info, filetype_info)

        ds = self.metadata['INVENTORYMETADATA'][
            'COLLECTIONDESCRIPTIONCLASS']['SHORTNAME']['VALUE']
        self.resolution = self.res[ds[-3]]

    def get_dataset(self, key, info):
        """Read data from file and return the corresponding projectables."""
        datadict = {
            1000: ['EV_250_Aggr1km_RefSB',
                   'EV_500_Aggr1km_RefSB',
                   'EV_1KM_RefSB',
                   'EV_1KM_Emissive'],
            500: ['EV_250_Aggr500_RefSB',
                  'EV_500_RefSB'],
            250: ['EV_250_RefSB']}

        platform_name = self.metadata['INVENTORYMETADATA']['ASSOCIATEDPLATFORMINSTRUMENTSENSOR'][
            'ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER']['ASSOCIATEDPLATFORMSHORTNAME']['VALUE']

        info.update({'platform_name': 'EOS-' + platform_name})
        info.update({'sensor': 'modis'})

        if self.resolution != key.resolution:
            return

        datasets = datadict[self.resolution]
        for dataset in datasets:
            subdata = self.sd.select(dataset)
            band_names = subdata.attributes()["band_names"].split(",")

            # get the relative indices of the desired channel
            try:
                index = band_names.index(key.name)
            except ValueError:
                continue
            uncertainty = self.sd.select(dataset + "_Uncert_Indexes")
            array = xr.DataArray(from_sds(subdata, chunks=CHUNK_SIZE)[index, :, :],
                                 dims=['y', 'x']).astype(np.float32)
            valid_range = subdata.attributes()['valid_range']
            array = array.where(array >= np.float32(valid_range[0]))
            array = array.where(array <= np.float32(valid_range[1]))
            array = array.where(from_sds(uncertainty, chunks=CHUNK_SIZE)[index, :, :] < 15)

            if dataset.endswith('Emissive'):
                projectable = calibrate_tb(array, subdata.attributes(), index, key.name)
            else:
                projectable = calibrate_refl(array, subdata.attributes(), index)
            projectable.attrs = info

            # if ((platform_name == 'Aqua' and key.name in ["6", "27", "36"]) or
            #         (platform_name == 'Terra' and key.name in ["29"])):
            #     height, width = projectable.shape
            #     row_indices = projectable.mask.sum(1) == width
            #     if row_indices.sum() != height:
            #         projectable.mask[row_indices, :] = True

            # Get the orbit number
            # if not satscene.orbit:
            #     mda = self.data.attributes()["CoreMetadata.0"]
            #     orbit_idx = mda.index("ORBITNUMBER")
            #     satscene.orbit = mda[orbit_idx + 111:orbit_idx + 116]

            # Get the geolocation
            # if resolution != 1000:
            #    logger.warning("Cannot load geolocation at this resolution (yet).")
            #    return

            # Trimming out dead sensor lines (detectors) on terra:
            # (in addition channel 27, 30, 34, 35, and 36 are nosiy)
            # if satscene.satname == "terra":
            #     for band in ["29"]:
            #         if not satscene[band].is_loaded() or satscene[band].data.mask.all():
            #             continue
            #         width = satscene[band].data.shape[1]
            #         height = satscene[band].data.shape[0]
            #         indices = satscene[band].data.mask.sum(1) < width
            #         if indices.sum() == height:
            #             continue
            #         satscene[band] = satscene[band].data[indices, :]
            #         satscene[band].area = geometry.SwathDefinition(
            #             lons=satscene[band].area.lons[indices, :],
            #             lats=satscene[band].area.lats[indices, :])
            return projectable

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


def calibrate_refl(array, attributes, index):
    """Calibration for reflective channels."""
    offset = np.float32(attributes["reflectance_offsets"][index])
    scale = np.float32(attributes["reflectance_scales"][index])

    array = (array - offset) * scale * 100

    return array


def calibrate_tb(array, attributes, index, band_name):
    """Calibration for the emissive channels."""
    offset = np.float32(attributes["radiance_offsets"][index])
    scale = np.float32(attributes["radiance_scales"][index])

    array = (array - offset) * scale

    # Planck constant (Joule second)
    h__ = np.float32(6.6260755e-34)

    # Speed of light in vacuum (meters per second)
    c__ = np.float32(2.9979246e+8)

    # Boltzmann constant (Joules per Kelvin)
    k__ = np.float32(1.380658e-23)

    # Derived constants
    c_1 = 2 * h__ * c__ * c__
    c_2 = (h__ * c__) / k__

    # Effective central wavenumber (inverse centimeters)
    cwn = np.array([
        2.641775E+3, 2.505277E+3, 2.518028E+3, 2.465428E+3,
        2.235815E+3, 2.200346E+3, 1.477967E+3, 1.362737E+3,
        1.173190E+3, 1.027715E+3, 9.080884E+2, 8.315399E+2,
        7.483394E+2, 7.308963E+2, 7.188681E+2, 7.045367E+2],
        dtype=np.float32)

    # Temperature correction slope (no units)
    tcs = np.array([
        9.993411E-1, 9.998646E-1, 9.998584E-1, 9.998682E-1,
        9.998819E-1, 9.998845E-1, 9.994877E-1, 9.994918E-1,
        9.995495E-1, 9.997398E-1, 9.995608E-1, 9.997256E-1,
        9.999160E-1, 9.999167E-1, 9.999191E-1, 9.999281E-1],
        dtype=np.float32)

    # Temperature correction intercept (Kelvin)
    tci = np.array([
        4.770532E-1, 9.262664E-2, 9.757996E-2, 8.929242E-2,
        7.310901E-2, 7.060415E-2, 2.204921E-1, 2.046087E-1,
        1.599191E-1, 8.253401E-2, 1.302699E-1, 7.181833E-2,
        1.972608E-2, 1.913568E-2, 1.817817E-2, 1.583042E-2],
        dtype=np.float32)

    # Transfer wavenumber [cm^(-1)] to wavelength [m]
    cwn = 1. / (cwn * 100)

    # Some versions of the modis files do not contain all the bands.
    emmissive_channels = ["20", "21", "22", "23", "24", "25", "27", "28", "29",
                          "30", "31", "32", "33", "34", "35", "36"]
    global_index = emmissive_channels.index(band_name)

    cwn = cwn[global_index]
    tcs = tcs[global_index]
    tci = tci[global_index]
    array = c_2 / (cwn * xu.log(c_1 / (1000000 * array * cwn ** 5) + 1))
    array = (array - tci) / tcs
    return array


if __name__ == '__main__':
    from satpy.utils import debug_on
    debug_on()
    br = HDFEOSBandReader(
        '/data/temp/Martin.Raspaud/MYD021km_A16220_130933_2016220132537.hdf')
    gr = HDFEOSGeoReader(
        '/data/temp/Martin.Raspaud/MYD03_A16220_130933_2016220132537.hdf')
