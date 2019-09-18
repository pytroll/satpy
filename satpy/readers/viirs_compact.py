#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2019 Satpy developers
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
"""Compact viirs format.

.. note:: It should be possible to further enhance this reader by performing the
   interpolation of the angles and lon/lats at the native dask chunk level.

"""

import logging
from datetime import datetime, timedelta

import h5py
import numpy as np
import xarray as xr
import dask.array as da

from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.utils import np2str
from satpy.utils import angle2xyz, lonlat2xyz, xyz2angle, xyz2lonlat
from satpy import CHUNK_SIZE

try:
    import tables
except ImportError:
    tables = None

chans_dict = {"M01": "M1",
              "M02": "M2",
              "M03": "M3",
              "M04": "M4",
              "M05": "M5",
              "M06": "M6",
              "M07": "M7",
              "M08": "M8",
              "M09": "M9",
              "M10": "M10",
              "M11": "M11",
              "M12": "M12",
              "M13": "M13",
              "M14": "M14",
              "M15": "M15",
              "M16": "M16",
              "DNB": "DNB"}

logger = logging.getLogger(__name__)

c = 299792458  # m.s-1
h = 6.6260755e-34  # m2kg.s-1
k = 1.380658e-23  # m2kg.s-2.K-1

short_names = {'NPP': 'Suomi-NPP',
               'J01': 'NOAA-20',
               'J02': 'NOAA-21'}


class VIIRSCompactFileHandler(BaseFileHandler):
    """A file handler class for VIIRS compact format."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super(VIIRSCompactFileHandler, self).__init__(filename, filename_info,
                                                      filetype_info)
        self.h5f = h5py.File(self.filename, "r")
        self.finfo = filename_info
        self.lons = None
        self.lats = None
        if filetype_info['file_type'] == 'compact_m':
            self.ch_type = 'MOD'
        elif filetype_info['file_type'] == 'compact_dnb':
            self.ch_type = 'DNB'
        else:
            raise IOError('Compact Viirs file type not recognized.')

        geo_data = self.h5f["Data_Products"]["VIIRS-%s-GEO" % self.ch_type]["VIIRS-%s-GEO_Gran_0" % self.ch_type]
        self.min_lat = np.asscalar(geo_data.attrs['South_Bounding_Coordinate'])
        self.max_lat = np.asscalar(geo_data.attrs['North_Bounding_Coordinate'])
        self.min_lon = np.asscalar(geo_data.attrs['West_Bounding_Coordinate'])
        self.max_lon = np.asscalar(geo_data.attrs['East_Bounding_Coordinate'])

        self.switch_to_cart = ((abs(self.max_lon - self.min_lon) > 90)
                               or (max(abs(self.min_lat), abs(self.max_lat)) > 60))

        self.scans = self.h5f["All_Data"]["NumberOfScans"][0]
        self.geostuff = self.h5f["All_Data"]['VIIRS-%s-GEO_All' % self.ch_type]

        self.c_align = da.from_array(self.geostuff["AlignmentCoefficient"])[
            np.newaxis, np.newaxis, :, np.newaxis]
        self.c_exp = da.from_array(self.geostuff["ExpansionCoefficient"])[
            np.newaxis, np.newaxis, :, np.newaxis]

        for key in self.h5f["All_Data"].keys():
            if key.startswith("VIIRS") and key.endswith("SDR_All"):
                channel = key.split('-')[1]
                break

        # FIXME:  this supposes  there is  only one  tiepoint zone  in the
        # track direction
        self.scan_size = self.h5f["All_Data/VIIRS-%s-SDR_All" %
                                  channel].attrs["TiePointZoneSizeTrack"]
        self.scan_size = np.asscalar(self.scan_size)
        self.track_offset = self.h5f["All_Data/VIIRS-%s-SDR_All" %
                                     channel].attrs["PixelOffsetTrack"]
        self.scan_offset = self.h5f["All_Data/VIIRS-%s-SDR_All" %
                                    channel].attrs["PixelOffsetScan"]

        try:
            self.group_locations = self.geostuff[
                "TiePointZoneGroupLocationScanCompact"].value
        except KeyError:
            self.group_locations = [0]

        self.tpz_sizes = self.h5f["All_Data/VIIRS-%s-SDR_All" %
                                  channel].attrs["TiePointZoneSizeScan"]
        self.nb_tpzs = self.geostuff["NumberOfTiePointZonesScan"].value

        self.cache = {}

        self.mda = {}
        short_name = np2str(self.h5f.attrs['Platform_Short_Name'])
        self.mda['platform_name'] = short_names.get(short_name, short_name)
        self.mda['sensor'] = 'viirs'

    def get_dataset(self, key, info):
        """Load a dataset."""
        logger.debug('Reading %s.', key.name)
        if key.name in chans_dict:
            m_data = self.read_dataset(key, info)
        else:
            m_data = self.read_geo(key, info)
        m_data.attrs.update(info)
        return m_data

    def get_bounding_box(self):
        """Get the bounding box of the data."""
        for key in self.h5f["Data_Products"].keys():
            if key.startswith("VIIRS") and key.endswith("GEO"):
                lats = self.h5f["Data_Products"][key][
                    key + '_Gran_0'].attrs['G-Ring_Latitude']
                lons = self.h5f["Data_Products"][key][
                    key + '_Gran_0'].attrs['G-Ring_Longitude']
                break
        else:
            raise KeyError('Cannot find bounding coordinates!')
        return lons.ravel(), lats.ravel()

    @property
    def start_time(self):
        """Get the start time."""
        return self.finfo['start_time']

    @property
    def end_time(self):
        """Get the end time."""
        end_time = datetime.combine(self.start_time.date(),
                                    self.finfo['end_time'].time())
        if end_time < self.start_time:
            end_time += timedelta(days=1)
        return end_time

    def read_geo(self, key, info):
        """Read angles."""
        pairs = {('satellite_azimuth_angle', 'satellite_zenith_angle'):
                 ("SatelliteAzimuthAngle", "SatelliteZenithAngle"),
                 ('solar_azimuth_angle', 'solar_zenith_angle'):
                 ("SolarAzimuthAngle", "SolarZenithAngle"),
                 ('dnb_solar_azimuth_angle', 'dnb_solar_zenith_angle'):
                 ("SolarAzimuthAngle", "SolarZenithAngle"),
                 ('dnb_lunar_azimuth_angle', 'dnb_lunar_zenith_angle'):
                 ("LunarAzimuthAngle", "LunarZenithAngle"),
                 }

        for pair, fkeys in pairs.items():
            if key.name in pair:
                if (self.cache.get(pair[0]) is None
                        or self.cache.get(pair[1]) is None):
                    angles = self.angles(*fkeys)
                    self.cache[pair[0]], self.cache[pair[1]] = angles
                if key.name == pair[0]:
                    return xr.DataArray(self.cache[pair[0]], name=key.name,
                                        attrs=self.mda, dims=('y', 'x'))
                else:
                    return xr.DataArray(self.cache[pair[1]], name=key.name,
                                        attrs=self.mda, dims=('y', 'x'))

        if info.get('standard_name') in ['latitude', 'longitude']:
            if self.lons is None or self.lats is None:
                self.lons, self.lats = self.navigate()
            mda = self.mda.copy()
            mda.update(info)
            if info['standard_name'] == 'longitude':
                return xr.DataArray(self.lons, attrs=mda, dims=('y', 'x'))
            else:
                return xr.DataArray(self.lats, attrs=mda, dims=('y', 'x'))

        if key.name == 'dnb_moon_illumination_fraction':
            mda = self.mda.copy()
            mda.update(info)
            return xr.DataArray(da.from_array(self.geostuff["MoonIllumFraction"]),
                                attrs=info)

    def read_dataset(self, dataset_key, info):
        """Read a dataset."""
        h5f = self.h5f
        channel = chans_dict[dataset_key.name]
        chan_dict = dict([(key.split("-")[1], key)
                          for key in h5f["All_Data"].keys()
                          if key.startswith("VIIRS")])

        h5rads = h5f["All_Data"][chan_dict[channel]]["Radiance"]
        chunks = h5rads.chunks or CHUNK_SIZE
        rads = xr.DataArray(da.from_array(h5rads, chunks=chunks),
                            name=dataset_key.name,
                            dims=['y', 'x']).astype(np.float32)
        h5attrs = h5rads.attrs
        scans = h5f["All_Data"]["NumberOfScans"][0]
        rads = rads[:scans * 16, :]
        # if channel in ("M9", ):
        #     arr = rads[:scans * 16, :].astype(np.float32)
        #     arr[arr > 65526] = np.nan
        #     arr = np.ma.masked_array(arr, mask=arr_mask)
        # else:
        #     arr = np.ma.masked_greater(rads[:scans * 16, :].astype(np.float32),
        #                                65526)
        rads = rads.where(rads <= 65526)
        try:
            rads = xr.where(rads <= h5attrs['Threshold'],
                            rads * h5attrs['RadianceScaleLow'] +
                            h5attrs['RadianceOffsetLow'],
                            rads * h5attrs['RadianceScaleHigh'] +
                            h5attrs['RadianceOffsetHigh'])
        except (KeyError, AttributeError):
            logger.info("Missing attribute for scaling of %s.", channel)
            pass
        unit = "W m-2 sr-1 μm-1"
        if dataset_key.calibration == 'counts':
            raise NotImplementedError("Can't get counts from this data")
        if dataset_key.calibration in ['reflectance', 'brightness_temperature']:
            # do calibrate
            try:
                # First guess: VIS or NIR data
                a_vis = h5attrs['EquivalentWidth']
                b_vis = h5attrs['IntegratedSolarIrradiance']
                dse = h5attrs['EarthSunDistanceNormalised']
                rads *= 100 * np.pi * a_vis / b_vis * (dse**2)
                unit = "%"
            except KeyError:
                # Maybe it's IR data?
                try:
                    a_ir = h5attrs['BandCorrectionCoefficientA']
                    b_ir = h5attrs['BandCorrectionCoefficientB']
                    lambda_c = h5attrs['CentralWaveLength']
                    rads *= 1e6
                    rads = (h * c) / (k * lambda_c *
                                      np.log(1 +
                                             (2 * h * c ** 2) /
                                             ((lambda_c ** 5) * rads)))
                    rads *= a_ir
                    rads += b_ir
                    unit = "K"
                except KeyError:
                    logger.warning("Calibration failed.")

        elif dataset_key.calibration != 'radiance':
            raise ValueError("Calibration parameter should be radiance, "
                             "reflectance or brightness_temperature")
        rads = rads.clip(min=0)
        rads.attrs = self.mda
        rads.attrs['units'] = unit
        return rads

    def navigate(self):
        """Generate lon and lat datasets."""
        all_lon = da.from_array(self.geostuff["Longitude"])
        all_lat = da.from_array(self.geostuff["Latitude"])

        res = []
        param_start = 0
        for tpz_size, nb_tpz, start in zip(self.tpz_sizes, self.nb_tpzs,
                                           self.group_locations):

            lon = all_lon[:, start:start + nb_tpz + 1]
            lat = all_lat[:, start:start + nb_tpz + 1]

            c_align = self.c_align[:, :, param_start:param_start + nb_tpz, :]
            c_exp = self.c_exp[:, :, param_start:param_start + nb_tpz, :]

            param_start += nb_tpz

            expanded = []

            if self.switch_to_cart:
                arrays = lonlat2xyz(lon, lat)
            else:
                arrays = (lon, lat)

            for data in arrays:
                expanded.append(expand_array(
                    data, self.scans, c_align, c_exp, self.scan_size,
                    tpz_size, nb_tpz, self.track_offset, self.scan_offset))

            if self.switch_to_cart:
                res.append(xyz2lonlat(*expanded))
            else:
                res.append(expanded)
        lons, lats = zip(*res)
        return da.hstack(lons), da.hstack(lats)

    def angles(self, azi_name, zen_name):
        """Compute the angle datasets."""
        all_lat = da.from_array(self.geostuff["Latitude"])
        all_zen = da.from_array(self.geostuff[zen_name])
        all_azi = da.from_array(self.geostuff[azi_name])

        res = []

        param_start = 0
        for tpz_size, nb_tpz, start in zip(self.tpz_sizes, self.nb_tpzs,
                                           self.group_locations):
            lat = all_lat[:, start:start + nb_tpz + 1]
            zen = all_zen[:, start:start + nb_tpz + 1]
            azi = all_azi[:, start:start + nb_tpz + 1]

            c_align = self.c_align[:, :, param_start:param_start + nb_tpz, :]
            c_exp = self.c_exp[:, :, param_start:param_start + nb_tpz, :]

            param_start += nb_tpz

            if (np.max(azi) - np.min(azi) > 5) or (np.min(zen) < 10) or (
                    np.max(abs(lat)) > 80):
                expanded = []
                for data in angle2xyz(azi, zen):
                    expanded.append(expand_array(
                        data, self.scans, c_align, c_exp, self.scan_size,
                        tpz_size, nb_tpz, self.track_offset, self.scan_offset))

                azi, zen = xyz2angle(*expanded)
                res.append((azi, zen))
            else:
                expanded = []
                for data in (azi, zen):
                    expanded.append(expand_array(
                        data, self.scans, c_align, c_exp, self.scan_size,
                        tpz_size, nb_tpz, self.track_offset, self.scan_offset))
                res.append(expanded)

        azi, zen = zip(*res)
        return da.hstack(azi), da.hstack(zen)


def expand_array(data,
                 scans,
                 c_align,
                 c_exp,
                 scan_size=16,
                 tpz_size=16,
                 nties=200,
                 track_offset=0.5,
                 scan_offset=0.5):
    """Expand *data* according to alignment and expansion."""
    nties = np.asscalar(nties)
    tpz_size = np.asscalar(tpz_size)
    s_scan, s_track = da.meshgrid(np.arange(nties * tpz_size),
                                  np.arange(scans * scan_size))
    s_track = (s_track.reshape(scans, scan_size, nties, tpz_size) % scan_size
               + track_offset) / scan_size
    s_scan = (s_scan.reshape(scans, scan_size, nties, tpz_size) % tpz_size
              + scan_offset) / tpz_size

    a_scan = s_scan + s_scan * (1 - s_scan) * c_exp + s_track * (
        1 - s_track) * c_align
    a_track = s_track

    data_a = data[:scans * 2:2, np.newaxis, :-1, np.newaxis]
    data_b = data[:scans * 2:2, np.newaxis, 1:, np.newaxis]
    data_c = data[1:scans * 2:2, np.newaxis, 1:, np.newaxis]
    data_d = data[1:scans * 2:2, np.newaxis, :-1, np.newaxis]

    fdata = ((1 - a_track) *
             ((1 - a_scan) * data_a + a_scan * data_b) + a_track * (
                 (1 - a_scan) * data_d + a_scan * data_c))
    return fdata.reshape(scans * scan_size, nties * tpz_size)
