#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010-2017 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Modis level 1b hdf-eos format reader

Introduction
------------

The ``modis_l1b`` reader reads and calibrates Modis L1 image data in hdf-eos format. Files often have
a pattern similar to the following one:

.. parsed-literal::
    M[O/Y]D02[1/H/Q]KM.A[date].[time].[collection].[processing_time].hdf

Other patterns where "collection" and/or "proccessing_time" are missing might also work
(see the readers yaml file for details). Geolocation files (MOD03) are also supported.


Geolocation files
-----------------

For the 1km data (mod021km) geolocation files (mod03) are optional. If not given to the reader
1km geolocations will be interpolated from the 5km geolocation contained within the file.

For the 500m and 250m data geolocation files are needed.


References:
    - Modis gelocation description: http://www.icare.univ-lille1.fr/wiki/index.php/MODIS_geolocation
"""
import logging

import numpy as np

import xarray as xr
from satpy import CHUNK_SIZE
from satpy.readers.hdfeos_base import HDFEOSBaseFileReader, HDFEOSGeoReader
from satpy.readers.hdf4_utils import from_sds

logger = logging.getLogger(__name__)


class HDFEOSBandReader(HDFEOSBaseFileReader):
    """Handler for the regular band channels."""

    res = {"1": 1000,
           "Q": 250,
           "H": 500}

    def __init__(self, filename, filename_info, filetype_info):
        HDFEOSBaseFileReader.__init__(self, filename, filename_info, filetype_info)

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

            # Fill values:
            # Data Value Meaning
            # 65535 Fill Value (includes reflective band data at night mode
            # and completely missing L1A scans)
            # 65534 L1A DN is missing within a scan
            # 65533 Detector is saturated
            # 65532 Cannot compute zero point DN, e.g., SV is saturated
            # 65531 Detector is dead (see comments below)
            # 65530 RSB dn** below the minimum of the scaling range
            # 65529 TEB radiance or RSB dn** exceeds the maximum of the
            # scaling range
            # 65528 Aggregation algorithm failure
            # 65527 Rotation of Earth view Sector from nominal science
            # collection position
            # 65526 Calibration coefficient b1 could not be computed
            # 65525 Subframe is dead
            # 65524 Both sides of the PCLW electronics on simultaneously
            # 65501 - 65523 (reserved for future use)
            # 65500 NAD closed upper limit

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


class MixedHDFEOSReader(HDFEOSGeoReader, HDFEOSBandReader):
    """A file handler for the files that have both regular bands and geographical information in them."""

    def __init__(self, filename, filename_info, filetype_info):
        HDFEOSGeoReader.__init__(self, filename, filename_info, filetype_info)
        HDFEOSBandReader.__init__(self, filename, filename_info, filetype_info)

    def get_dataset(self, key, info):
        if key.name in HDFEOSGeoReader.DATASET_NAMES:
            return HDFEOSGeoReader.get_dataset(self, key, info)
        return HDFEOSBandReader.get_dataset(self, key, info)


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
    array = c_2 / (cwn * np.log(c_1 / (1000000 * array * cwn ** 5) + 1))
    array = (array - tci) / tcs
    return array
