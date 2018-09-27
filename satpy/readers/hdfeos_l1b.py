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
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.hdf4_utils import from_sds

logger = logging.getLogger(__name__)


# TODO on interpolation:
# - factorize !!!
# - for 500m and 250m use all 1km pixels as base
# - go over to cartesian coordinates for tricky situation (eg poles, dateline)
# - test !!!

R = 6371.
# Aqua scan width and altitude in km
scan_width = 10.00017
H = 705.


def compute_phi(zeta):
    return np.arcsin(R * np.sin(zeta) / (R + H))


def compute_theta(zeta, phi):
    return zeta - phi


def compute_zeta(phi):
    return np.arcsin((R + H) * np.sin(phi) / R)


def compute_expansion_alignment(satz_a, satz_b, satz_c, satz_d):
    """All angles in radians."""
    zeta_a = satz_a
    zeta_b = satz_b

    phi_a = compute_phi(zeta_a)
    phi_b = compute_phi(zeta_b)
    theta_a = compute_theta(zeta_a, phi_a)
    theta_b = compute_theta(zeta_b, phi_b)
    phi = (phi_a + phi_b) / 2
    zeta = compute_zeta(phi)
    theta = compute_theta(zeta, phi)

    c_expansion = 4 * (((theta_a + theta_b) / 2 - theta) / (theta_a - theta_b))

    sin_beta_2 = scan_width / (2 * H)

    d = ((R + H) / R * np.cos(phi) - np.cos(zeta)) * sin_beta_2
    e = np.cos(zeta) - np.sqrt(np.cos(zeta) ** 2 - d ** 2)

    c_alignment = 4 * e * np.sin(zeta) / (theta_a - theta_b)

    return c_expansion, c_alignment


def get_corners(arr):
    arr_a = arr[:, :-1, :-1]
    arr_b = arr[:, :-1, 1:]
    arr_c = arr[:, 1:, 1:]
    arr_d = arr[:, 1:, :-1]
    return arr_a, arr_b, arr_c, arr_d


class ModisInterpolator():

    def __init__(self, cres, fres):
        if cres == 1000:
            self.cscan_len = 10
            self.cscan_width = 1
            self.cscan_full_width = 1354
        elif cres == 5000:
            self.cscan_len = 2
            self.cscan_width = 5
            self.cscan_full_width = 271

        if fres == 250:
            self.fscan_width = 4 * self.cscan_width
            self.fscan_full_width = 1354 * 4
            self.fscan_len = 4 * 10 // self.cscan_len
            self.get_coords = self._get_coords_1km
            self.expand_tiepoint_array = self._expand_tiepoint_array_1km
        elif fres == 500:
            self.fscan_width = 2 * self.cscan_width
            self.fscan_full_width = 1354 * 2
            self.fscan_len = 2 * 10 // self.cscan_len
            self.get_coords = self._get_coords_1km
            self.expand_tiepoint_array = self._expand_tiepoint_array_1km
        elif fres == 1000:
            self.fscan_width = 1 * self.cscan_width
            self.fscan_full_width = 1354
            self.fscan_len = 1 * 10 // self.cscan_len
            self.get_coords = self._get_coords_5km
            self.expand_tiepoint_array = self._expand_tiepoint_array_5km

    def _expand_tiepoint_array_1km(self, arr, lines, cols):
        arr = da.repeat(arr, lines, axis=1)
        arr = da.concatenate((arr[:, :lines//2, :], arr, arr[:, -(lines//2):, :]), axis=1)
        arr = da.repeat(arr.reshape((-1, self.cscan_full_width - 1)), cols, axis=1)
        return da.hstack((arr, arr[:, -cols:]))

    def _get_coords_1km(self, scans):
        y = (np.arange((self.cscan_len + 1) * self.fscan_len) % self.fscan_len) + .5
        y = y[self.fscan_len // 2:-(self.fscan_len // 2)]
        y[:self.fscan_len//2] = np.arange(-self.fscan_len/2 + .5, 0)
        y[-(self.fscan_len//2):] = np.arange(self.fscan_len + .5, self.fscan_len * 3 / 2)
        y = np.tile(y, scans)

        x = np.arange(self.fscan_full_width) % self.fscan_width
        x[-self.fscan_width:] = np.arange(self.fscan_width, self.fscan_width * 2)
        return x, y

    def _expand_tiepoint_array_5km(self, arr, lines, cols):
        arr = da.repeat(arr, lines * 2, axis=1)
        arr = da.repeat(arr.reshape((-1, self.cscan_full_width - 1)), cols, axis=1)
        return da.hstack((arr[:, :2], arr, arr[:, -2:]))

    def _get_coords_5km(self, scans):
        y = np.arange(self.fscan_len * self.cscan_len) - 2
        y = np.tile(y, scans)

        x = (np.arange(self.fscan_full_width) - 2) % self.fscan_width
        x[0] = -2
        x[1] = -1
        x[-2] = 5
        x[-1] = 6
        return x, y

    def interpolate(self, lon1, lat1, satz1):
        cscan_len = self.cscan_len
        cscan_full_width = self.cscan_full_width

        fscan_width = self.fscan_width
        fscan_len = self.fscan_len

        scans = lat1.shape[0] // cscan_len
        latattrs = lat1.attrs
        lonattrs = lon1.attrs
        dims = lat1.dims
        lat1 = lat1.data
        lon1 = lon1.data
        satz1 = satz1.data

        lat1 = lat1.reshape((-1, cscan_len, cscan_full_width))
        lon1 = lon1.reshape((-1, cscan_len, cscan_full_width))
        satz1 = satz1.reshape((-1, cscan_len, cscan_full_width))

        lats_a, lats_b, lats_c, lats_d = get_corners(lat1)
        lons_a, lons_b, lons_c, lons_d = get_corners(lon1)
        satz_a, satz_b, satz_c, satz_d = get_corners(da.deg2rad(satz1))
        c_exp, c_ali = compute_expansion_alignment(satz_a, satz_b, satz_c, satz_d)

        x, y = self.get_coords(scans)
        i_rs, i_rt = da.meshgrid(x, y)

        p_os = 0
        p_ot = 0

        s_s = (p_os + i_rs) * 1. / fscan_width
        s_t = (p_ot + i_rt) * 1. / fscan_len

        cols = fscan_width
        lines = fscan_len

        c_exp_full = self.expand_tiepoint_array(c_exp, lines, cols)
        c_ali_full = self.expand_tiepoint_array(c_ali, lines, cols)

        a_track = s_t
        a_scan = (s_s + s_s * (1 - s_s) * c_exp_full + s_t*(1 - s_t) * c_ali_full)

        lats_a = self.expand_tiepoint_array(lats_a, lines, cols)
        lats_b = self.expand_tiepoint_array(lats_b, lines, cols)
        lats_c = self.expand_tiepoint_array(lats_c, lines, cols)
        lats_d = self.expand_tiepoint_array(lats_d, lines, cols)
        lons_a = self.expand_tiepoint_array(lons_a, lines, cols)
        lons_b = self.expand_tiepoint_array(lons_b, lines, cols)
        lons_c = self.expand_tiepoint_array(lons_c, lines, cols)
        lons_d = self.expand_tiepoint_array(lons_d, lines, cols)

        lats_1 = (1 - a_scan) * lats_a + a_scan * lats_b
        lats_2 = (1 - a_scan) * lats_d + a_scan * lats_c
        lats = (1 - a_track) * lats_1 + a_track * lats_2

        lons_1 = (1 - a_scan) * lons_a + a_scan * lons_b
        lons_2 = (1 - a_scan) * lons_d + a_scan * lons_c
        lons = (1 - a_track) * lons_1 + a_track * lons_2

        return xr.DataArray(lons, attrs=lonattrs, dims=dims), xr.DataArray(lats, attrs=latattrs, dims=dims)


def modis_1km_to_250m(lon1, lat1, satz1):

    interp = ModisInterpolator(1000, 250)
    return interp.interpolate(lon1, lat1, satz1)


def modis_1km_to_500m(lon1, lat1, satz1):

    interp = ModisInterpolator(1000, 500)
    return interp.interpolate(lon1, lat1, satz1)


def modis_5km_to_1km(lon1, lat1, satz1):

    interp = ModisInterpolator(5000, 1000)
    return interp.interpolate(lon1, lat1, satz1)


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

    def get_dataset(self, key, info):
        """Get the dataset designated by *key*."""
        if key.name == 'solar_zenith_angle':
            data = self.load('SolarZenith')
        elif key.name == 'solar_azimuth_angle':
            data = self.load('SolarAzimuth')
        elif key.name == 'satellite_zenith_angle':
            data = self.load('SensorZenith')
        elif key.name == 'satellite_azimuth_angle':
            data = self.load('SensorAzimuth')
        elif key.name == 'longitude':
            data = self.load('Longitude')
        elif key.name == 'latitude':
            data = self.load('Latitude')
        else:
            return

        if key.resolution != self.resolution:
            # let's see if we have something in the cache
            try:
                data = self.cache[key.resolution][key.name]
                data.attrs.update(info)
                data.attrs['standard_name'] = data.attrs['name']
                return data
            except KeyError:
                self.cache.setdefault(key.resolution, {})

            # too bad, now we need to interpolate
            satz = self.load('SensorZenith')
            if key.name in ['longitude', 'latitude']:
                data_a = self.load('Longitude')
                data_b = self.load('Latitude')
            elif key.name in ['satellite_azimuth_angle', 'satellite_zenith_angle']:
                data_a = self.load('SensorAzimuth')
                data_b = self.load('SensorZenith') - 90
            elif key.name in ['solar_azimuth_angle', 'solar_zenith_angle']:
                data_a = self.load('SolarAzimuth')
                data_b = self.load('SolarZenith') - 90

            data_a, data_b = self._interpolate(data_a, data_b, satz,
                                               self.resolution, key.resolution)

            if key.name in ['longitude', 'latitude']:
                self.cache[key.resolution]['longitude'] = data_a
                self.cache[key.resolution]['latitude'] = data_b
            elif key.name in ['satellite_azimuth_angle', 'satellite_zenith_angle']:
                self.cache[key.resolution]['satellite_azimuth_angle'] = data_a
                self.cache[key.resolution]['satellite_zenith_angle'] = data_b + 90
            elif key.name in ['solar_azimuth_angle', 'solar_zenith_angle']:
                self.cache[key.resolution]['solar_azimuth_angle'] = data_a
                self.cache[key.resolution]['solar_zenith_angle'] = data_b + 90

            data = self.cache[key.resolution][key.name]

        data.attrs.update(info)
        data.attrs['standard_name'] = data.attrs['name']
        return data

    def load(self, file_key):
        """Load the data."""
        var = self.sd.select(file_key)
        data = xr.DataArray(from_sds(var, chunks=CHUNK_SIZE),
                            dims=['y', 'x']).astype(np.float32)
        data = data.where(data != var._FillValue)
        try:
            data = data * np.float32(var.scale_factor)
        except AttributeError:
            pass
        return data

    @staticmethod
    def _interpolate(clons, clats, csatz, coarse_resolution, resolution):
        if resolution == coarse_resolution:
            return clons, clats

        funs = {(5000, 1000): modis_5km_to_1km,
                (1000, 500): modis_1km_to_500m,
                (1000, 250): modis_1km_to_250m}

        try:
            fun = funs[(coarse_resolution, resolution)]
        except KeyError:
            raise NotImplementedError('Interpolation from {}m to {}m not implemented'.format(
                                      coarse_resolution, resolution))

        logger.debug("Interpolating from " + str(coarse_resolution)
                     + " to " + str(resolution))

        return fun(clons, clats, csatz)


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
            var_attrs = subdata.attributes()
            band_names = var_attrs["band_names"].split(",")

            # get the relative indices of the desired channel
            try:
                index = band_names.index(key.name)
            except ValueError:
                continue
            uncertainty = self.sd.select(dataset + "_Uncert_Indexes")
            array = xr.DataArray(from_sds(subdata, chunks=CHUNK_SIZE)[index, :, :],
                                 dims=['y', 'x']).astype(np.float32)
            valid_range = var_attrs['valid_range']
            array = array.where(array >= np.float32(valid_range[0]))
            array = array.where(array <= np.float32(valid_range[1]))
            array = array.where(from_sds(uncertainty, chunks=CHUNK_SIZE)[index, :, :] < 15)

            if key.calibration == 'brightness_temperature':
                projectable = calibrate_bt(array, var_attrs, index, key.name)
                info.setdefault('units', 'K')
                info.setdefault('standard_name', 'toa_brightness_temperature')
            elif key.calibration == 'reflectance':
                projectable = calibrate_refl(array, var_attrs, index)
                info.setdefault('units', '%')
                info.setdefault('standard_name',
                                'toa_bidirectional_reflectance')
            elif key.calibration == 'radiance':
                projectable = calibrate_radiance(array, var_attrs, index)
                info.setdefault('units', var_attrs.get('radiance_units'))
                info.setdefault('standard_name',
                                'toa_outgoing_radiance_per_unit_wavelength')
            elif key.calibration == 'counts':
                projectable = calibrate_counts(array, var_attrs, index)
                info.setdefault('units', 'counts')
                info.setdefault('standard_name', 'counts')  # made up
            else:
                raise ValueError("Unknown calibration for "
                                 "key: {}".format(key))
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


def calibrate_counts(array, attributes, index):
    """Calibration for counts channels."""
    offset = np.float32(attributes["corrected_counts_offsets"][index])
    scale = np.float32(attributes["corrected_counts_scales"][index])
    array = (array - offset) * scale
    return array


def calibrate_radiance(array, attributes, index):
    """Calibration for radiance channels."""
    offset = np.float32(attributes["radiance_offsets"][index])
    scale = np.float32(attributes["radiance_scales"][index])
    array = (array - offset) * scale
    return array


def calibrate_refl(array, attributes, index):
    """Calibration for reflective channels."""
    offset = np.float32(attributes["reflectance_offsets"][index])
    scale = np.float32(attributes["reflectance_scales"][index])
    # convert to reflectance and convert from 1 to %
    array = (array - offset) * scale * 100
    return array


def calibrate_bt(array, attributes, index, band_name):
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
