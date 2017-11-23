#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2014, 2015 Martin Raspaud

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Compact viirs format.
"""

import bz2
import glob
import logging
import os
from datetime import datetime, timedelta

import h5py
import numpy as np

from pyresample.geometry import SwathDefinition
from satpy.dataset import Dataset
from satpy.readers.file_handlers import BaseFileHandler
from satpy.utils import angle2xyz, lonlat2xyz, xyz2angle, xyz2lonlat

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

short_names = {'NPP': 'Suomi-NPP'}


class VIIRSCompactFileHandler(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(VIIRSCompactFileHandler, self).__init__(filename, filename_info,
                                                      filetype_info)
        self.h5f = h5py.File(self.filename, "r")
        self.finfo = filename_info
        self.lons = None
        self.lats = None

        self.scans = self.h5f["All_Data"]["NumberOfScans"][0]
        for key in self.h5f["All_Data"].keys():
            if key.startswith("VIIRS") and key.endswith("GEO_All"):
                self.geostuff = self.h5f["All_Data"][key]
                break

        self.c_align = self.geostuff["AlignmentCoefficient"].value[
            np.newaxis, np.newaxis, :, np.newaxis]
        self.c_exp = self.geostuff["ExpansionCoefficient"].value[
            np.newaxis, np.newaxis, :, np.newaxis]

        for key in self.h5f["All_Data"].keys():
            if key.startswith("VIIRS") and key.endswith("SDR_All"):
                channel = key.split('-')[1]
                break

        # FIXME:  this supposes  there is  only one  tiepoint zone  in the
        # track direction
        self.scan_size = self.h5f["All_Data/VIIRS-%s-SDR_All" %
                                  channel].attrs["TiePointZoneSizeTrack"][0]
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
        short_name = self.h5f.attrs['Platform_Short_Name'][0][0]
        self.mda['platform_name'] = short_names.get(short_name, short_name)
        self.mda['sensor'] = 'viirs'

    def get_dataset(self, key, info):
        """Load a dataset
        """

        logger.debug('Reading %s.', key.name)
        if key.name in chans_dict:
            m_data = self.read_dataset(key, info)
        else:
            m_data = self.read_geo(key, info)
        m_data.info.update(info)
        return m_data

    def get_bounding_box(self):
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
        return self.finfo['start_time']

    @property
    def end_time(self):
        end_time = datetime.combine(self.start_time.date(),
                                    self.finfo['end_time'].time())
        if end_time < self.start_time:
            end_time += timedelta(days=1)
        return end_time

    def read_geo(self, key, info):
        """Read angles.
        """
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
                if (self.cache.get(pair[0]) is None or
                        self.cache.get(pair[1]) is None):
                    angles = self.angles(*fkeys)
                    self.cache[pair[0]], self.cache[pair[1]] = angles
                if key.name == pair[0]:
                    return Dataset(self.cache[pair[0]],
                                   copy=False, name=key.name, **self.mda)
                else:
                    return Dataset(self.cache[pair[1]],
                                   copy=False, name=key.name, **self.mda)

        if info.get('standard_name') in ['latitude', 'longitude']:
            if self.lons is None or self.lats is None:
                self.lons, self.lats = self.navigate()
            mda = self.mda.copy()
            mda.update(info)
            if info['standard_name'] == 'longitude':
                return Dataset(self.lons, copy=False, id=key, **mda)
            else:
                return Dataset(self.lats, copy=False, id=key, **mda)

        if key.name == 'dnb_moon_illumination_fraction':
            mda = self.mda.copy()
            mda.update(info)
            return Dataset(self.geostuff["MoonIllumFraction"].value, **info)

    def read_dataset(self, dataset_key, info):
        h5f = self.h5f
        channel = chans_dict[dataset_key.name]
        chan_dict = dict([(key.split("-")[1], key)
                          for key in h5f["All_Data"].keys()
                          if key.startswith("VIIRS")])

        scans = h5f["All_Data"]["NumberOfScans"][0]
        res = []
        units = []
        arr_mask = np.ma.nomask

        rads = h5f["All_Data"][chan_dict[channel]]["Radiance"]

        if channel in ("M9", ):
            arr = rads[:scans * 16, :].astype(np.float32)
            arr[arr > 65526] = np.nan
            arr = np.ma.masked_array(arr, mask=arr_mask)
        else:
            arr = np.ma.masked_greater(rads[:scans * 16, :].astype(np.float32),
                                       65526)
        try:
            arr = np.ma.where(arr <= rads.attrs['Threshold'],
                              arr * rads.attrs['RadianceScaleLow'] +
                              rads.attrs['RadianceOffsetLow'],
                              arr * rads.attrs['RadianceScaleHigh'] +
                              rads.attrs['RadianceOffsetHigh'],)
            arr_mask = arr.mask
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
                a_vis = rads.attrs['EquivalentWidth']
                b_vis = rads.attrs['IntegratedSolarIrradiance']
                dse = rads.attrs['EarthSunDistanceNormalised']
                arr *= 100 * np.pi * a_vis / b_vis * (dse**2)
                unit = "%"
            except KeyError:
                # Maybe it's IR data?
                try:
                    a_ir = rads.attrs['BandCorrectionCoefficientA']
                    b_ir = rads.attrs['BandCorrectionCoefficientB']
                    lambda_c = rads.attrs['CentralWaveLength']
                    arr *= 1e6
                    arr = (h * c) / (k * lambda_c *
                                     np.log(1 +
                                            (2 * h * c ** 2) /
                                            ((lambda_c ** 5) * arr)))
                    arr *= a_ir
                    arr += b_ir
                    unit = "K"
                except KeyError:
                    logger.warning("Calibration failed.")

        elif dataset_key.calibration != 'radiance':
            raise ValueError("Calibration parameter should be radiance, "
                             "reflectance or brightness_temperature")
        arr[arr < 0] = 0

        return Dataset(arr, units=unit, copy=False, name=dataset_key.name, **self.mda)

    def navigate(self):

        all_lon = self.geostuff["Longitude"].value
        all_lat = self.geostuff["Latitude"].value

        res = []

        param_start = 0
        for tpz_size, nb_tpz, start in zip(self.tpz_sizes, self.nb_tpzs,
                                           self.group_locations):

            lon = all_lon[:, start:start + nb_tpz + 1]
            lat = all_lat[:, start:start + nb_tpz + 1]

            c_align = self.c_align[:, :, param_start:param_start + nb_tpz, :]
            c_exp = self.c_exp[:, :, param_start:param_start + nb_tpz, :]

            param_start += nb_tpz

            if (np.max(lon) - np.min(lon) > 90) or (np.max(abs(lat)) > 60):
                expanded = []
                for data in lonlat2xyz(lon, lat):
                    expanded.append(expand_array(
                        data, self.scans, c_align, c_exp, self.scan_size,
                        tpz_size, nb_tpz, self.track_offset, self.scan_offset))
                res.append(xyz2lonlat(*expanded))
            else:
                expanded = []
                for data in (lon, lat):
                    expanded.append(expand_array(
                        data, self.scans, c_align, c_exp, self.scan_size,
                        tpz_size, nb_tpz, self.track_offset, self.scan_offset))
                res.append(expanded)

        lons, lats = zip(*res)

        return np.hstack(lons), np.hstack(lats)

    def angles(self, azi_name, zen_name):

        all_lat = self.geostuff["Latitude"].value
        all_zen = self.geostuff[zen_name].value
        all_azi = self.geostuff[azi_name].value

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
        return np.hstack(azi), np.hstack(zen)


def read_dnb(h5f):

    scans = h5f.get_node("/All_Data/NumberOfScans").read()[0]
    res = []
    units = []

    rads_dset = h5f.get_node("/All_Data/VIIRS-DNB-SDR_All")
    arr = np.ma.masked_greater(rads_dset.Radiance.read()[:scans * 16, :], 1.0)
    unit = "W m-2 sr-1 μm-1"
    arr[arr < 0] = 0
    res.append(arr)
    units.append(unit)

    return res, units


def expand_array(data,
                 scans,
                 c_align,
                 c_exp,
                 scan_size=16,
                 tpz_size=16,
                 nties=200,
                 track_offset=0.5,
                 scan_offset=0.5):
    s_track, s_scan = np.mgrid[0:scans * scan_size, 0:nties * tpz_size]
    s_track = (s_track.reshape(scans, scan_size, nties, tpz_size) % scan_size +
               track_offset) / scan_size
    s_scan = (s_scan.reshape(scans, scan_size, nties, tpz_size) % tpz_size +
              scan_offset) / tpz_size

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


def navigate_dnb(h5f):

    scans = h5f.get_node("/All_Data/NumberOfScans").read()[0]
    geo_dset = h5f.get_node("/All_Data/VIIRS-DNB-GEO_All")
    all_c_align = geo_dset.AlignmentCoefficient.read()[
        np.newaxis, np.newaxis, :, np.newaxis]
    all_c_exp = geo_dset.ExpansionCoefficient.read()[np.newaxis, np.newaxis, :,
                                                     np.newaxis]
    all_lon = geo_dset.Longitude.read()
    all_lat = geo_dset.Latitude.read()

    res = []

    # FIXME: this supposes there is only one tiepoint zone in the
    # track direction
    scan_size = h5f.get_node_attr("/All_Data/VIIRS-DNB-SDR_All",
                                  "TiePointZoneSizeTrack")[0]
    track_offset = h5f.get_node_attr("/All_Data/VIIRS-DNB-SDR_All",
                                     "PixelOffsetTrack")[0]
    scan_offset = h5f.get_node_attr("/All_Data/VIIRS-DNB-SDR_All",
                                    "PixelOffsetScan")[0]

    try:
        group_locations = geo_dset.TiePointZoneGroupLocationScanCompact.read()
    except KeyError:
        group_locations = [0]
    param_start = 0
    for tpz_size, nb_tpz, start in \
        zip(h5f.get_node_attr("/All_Data/VIIRS-DNB-SDR_All",
                              "TiePointZoneSizeScan"),
            geo_dset.NumberOfTiePointZonesScan.read(),
            group_locations):
        lon = all_lon[:, start:start + nb_tpz + 1]
        lat = all_lat[:, start:start + nb_tpz + 1]
        c_align = all_c_align[:, :, param_start:param_start + nb_tpz, :]
        c_exp = all_c_exp[:, :, param_start:param_start + nb_tpz, :]
        param_start += nb_tpz
        nties = nb_tpz
        if (np.max(lon) - np.min(lon) > 90) or (np.max(abs(lat)) > 60):
            x, y, z = lonlat2xyz(lon, lat)
            x, y, z = (
                expand_array(x, scans, c_align, c_exp, scan_size, tpz_size,
                             nties, track_offset, scan_offset),
                expand_array(y, scans, c_align, c_exp, scan_size, tpz_size,
                             nties, track_offset, scan_offset), expand_array(
                                 z, scans, c_align, c_exp, scan_size, tpz_size,
                                 nties, track_offset, scan_offset))
            res.append(xyz2lonlat(x, y, z))
        else:
            res.append(
                (expand_array(lon, scans, c_align, c_exp, scan_size, tpz_size,
                              nties, track_offset, scan_offset),
                 expand_array(lat, scans, c_align, c_exp, scan_size, tpz_size,
                              nties, track_offset, scan_offset)))
    lons, lats = zip(*res)
    return np.hstack(lons), np.hstack(lats)
