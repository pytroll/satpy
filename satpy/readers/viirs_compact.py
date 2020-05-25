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

This is a reader for the Compact VIIRS format shipped on Eumetcast for the
VIIRS SDR. The format is compressed in multiple ways, notably by shipping only
tie-points for geographical data. The interpolation of this data is done using
dask operations, so it should be relatively performant.

For more information on this format, the reader can refer to the
`Compact VIIRS SDR Product Format User Guide` that can be found on this EARS_ page.

.. _EARS: https://www.eumetsat.int/website/home/Data/RegionalDataServiceEARS/EARSVIIRS/index.html

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
        self.min_lat = geo_data.attrs['South_Bounding_Coordinate'].item()
        self.max_lat = geo_data.attrs['North_Bounding_Coordinate'].item()
        self.min_lon = geo_data.attrs['West_Bounding_Coordinate'].item()
        self.max_lon = geo_data.attrs['East_Bounding_Coordinate'].item()

        self.switch_to_cart = ((abs(self.max_lon - self.min_lon) > 90)
                               or (max(abs(self.min_lat), abs(self.max_lat)) > 60))

        self.scans = self.h5f["All_Data"]["NumberOfScans"][0]
        self.geostuff = self.h5f["All_Data"]['VIIRS-%s-GEO_All' % self.ch_type]

        for key in self.h5f["All_Data"].keys():
            if key.startswith("VIIRS") and key.endswith("SDR_All"):
                channel = key.split('-')[1]
                break

        # FIXME:  this supposes  there is  only one  tiepoint zone  in the
        # track direction
        self.scan_size = self.h5f["All_Data/VIIRS-%s-SDR_All" %
                                  channel].attrs["TiePointZoneSizeTrack"].item()
        self.track_offset = self.h5f["All_Data/VIIRS-%s-SDR_All" %
                                     channel].attrs["PixelOffsetTrack"]
        self.scan_offset = self.h5f["All_Data/VIIRS-%s-SDR_All" %
                                    channel].attrs["PixelOffsetScan"]

        try:
            self.group_locations = self.geostuff[
                "TiePointZoneGroupLocationScanCompact"][()]
        except KeyError:
            self.group_locations = [0]

        self.tpz_sizes = da.from_array(self.h5f["All_Data/VIIRS-%s-SDR_All" % channel].attrs["TiePointZoneSizeScan"],
                                       chunks=1)
        if len(self.tpz_sizes.shape) == 2:
            if self.tpz_sizes.shape[1] != 1:
                raise NotImplementedError("Can't handle 2 dimensional tiepoint zones.")
            self.tpz_sizes = self.tpz_sizes.squeeze(1)
        self.nb_tpzs = self.geostuff["NumberOfTiePointZonesScan"]
        self.c_align = da.from_array(self.geostuff["AlignmentCoefficient"],
                                     chunks=tuple(self.nb_tpzs))
        self.c_exp = da.from_array(self.geostuff["ExpansionCoefficient"],
                                   chunks=tuple(self.nb_tpzs))
        self.nb_tpzs = da.from_array(self.nb_tpzs, chunks=1)
        self._expansion_coefs = None

        self.cache = {}

        self.mda = {}
        short_name = np2str(self.h5f.attrs['Platform_Short_Name'])
        self.mda['platform_name'] = short_names.get(short_name, short_name)
        self.mda['sensor'] = 'viirs'

    def __del__(self):
        """Close file handlers when we are done."""
        try:
            self.h5f.close()
        except OSError:
            pass

    def get_dataset(self, key, info):
        """Load a dataset."""
        logger.debug('Reading %s.', key.name)
        if key.name in chans_dict:
            m_data = self.read_dataset(key, info)
        else:
            m_data = self.read_geo(key, info)
        m_data.attrs.update(info)
        m_data.attrs['rows_per_scan'] = self.scan_size
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
        if self.lons is None or self.lats is None:
            self.lons, self.lats = self.navigate()
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

    def expand(self, data, coefs):
        """Perform the expansion in numpy domain."""
        data = data.reshape(data.shape[:-1])

        coefs = coefs.reshape(self.scans, self.scan_size, data.shape[1] - 1, -1, 4)

        coef_a = coefs[:, :, :, :, 0]
        coef_b = coefs[:, :, :, :, 1]
        coef_c = coefs[:, :, :, :, 2]
        coef_d = coefs[:, :, :, :, 3]

        data_a = data[:self.scans * 2:2, np.newaxis, :-1, np.newaxis]
        data_b = data[:self.scans * 2:2, np.newaxis, 1:, np.newaxis]
        data_c = data[1:self.scans * 2:2, np.newaxis, 1:, np.newaxis]
        data_d = data[1:self.scans * 2:2, np.newaxis, :-1, np.newaxis]

        fdata = (coef_a * data_a + coef_b * data_b + coef_d * data_d + coef_c * data_c)

        return fdata.reshape(self.scans * self.scan_size, -1)

    def expand_angle_and_nav(self, arrays):
        """Expand angle and navigation datasets."""
        res = []
        for array in arrays:
            res.append(da.map_blocks(self.expand, array[:, :, np.newaxis], self.expansion_coefs,
                                     dtype=array.dtype, drop_axis=2, chunks=self.expansion_coefs.chunks[:-1]))
        return res

    def get_coefs(self, c_align, c_exp, tpz_size, nb_tpz, v_track):
        """Compute the coeffs in numpy domain."""
        nties = nb_tpz.item()
        tpz_size = tpz_size.item()
        v_scan = (np.arange(nties * tpz_size) % tpz_size + self.scan_offset) / tpz_size
        s_scan, s_track = np.meshgrid(v_scan, v_track)
        s_track = s_track.reshape(self.scans, self.scan_size, nties, tpz_size)
        s_scan = s_scan.reshape(self.scans, self.scan_size, nties, tpz_size)

        c_align = c_align[np.newaxis, np.newaxis, :, np.newaxis]
        c_exp = c_exp[np.newaxis, np.newaxis, :, np.newaxis]

        a_scan = s_scan + s_scan * (1 - s_scan) * c_exp + s_track * (
            1 - s_track) * c_align
        a_track = s_track
        coef_a = (1 - a_track) * (1 - a_scan)
        coef_b = (1 - a_track) * a_scan
        coef_d = a_track * (1 - a_scan)
        coef_c = a_track * a_scan
        res = np.stack([coef_a, coef_b, coef_c, coef_d], axis=4).reshape(self.scans * self.scan_size, -1, 4)
        return res

    @property
    def expansion_coefs(self):
        """Compute the expansion coefficients."""
        if self._expansion_coefs is not None:
            return self._expansion_coefs
        v_track = (np.arange(self.scans * self.scan_size) % self.scan_size + self.track_offset) / self.scan_size
        self.tpz_sizes = self.tpz_sizes.persist()
        self.nb_tpzs = self.nb_tpzs.persist()
        col_chunks = (self.tpz_sizes * self.nb_tpzs).compute()
        self._expansion_coefs = da.map_blocks(self.get_coefs, self.c_align, self.c_exp, self.tpz_sizes, self.nb_tpzs,
                                              dtype=np.float64, v_track=v_track,  new_axis=[0, 2],
                                              chunks=(self.scans * self.scan_size,
                                                      tuple(col_chunks), 4))

        return self._expansion_coefs

    def navigate(self):
        """Generate the navigation datasets."""
        shape = self.geostuff['Longitude'].shape
        hchunks = (self.nb_tpzs + 1).compute()
        chunks = (shape[0], tuple(hchunks))
        lon = da.from_array(self.geostuff["Longitude"], chunks=chunks)
        lat = da.from_array(self.geostuff["Latitude"], chunks=chunks)
        if self.switch_to_cart:
            arrays = lonlat2xyz(lon, lat)
        else:
            arrays = (lon, lat)

        expanded = self.expand_angle_and_nav(arrays)
        if self.switch_to_cart:
            return xyz2lonlat(*expanded)

        return expanded

    def angles(self, azi_name, zen_name):
        """Generate the angle datasets."""
        shape = self.geostuff['Longitude'].shape
        hchunks = (self.nb_tpzs + 1).compute()
        chunks = (shape[0], tuple(hchunks))

        azi = self.geostuff[azi_name]
        zen = self.geostuff[zen_name]

        switch_to_cart = ((np.max(azi) - np.min(azi) > 5)
                          or (np.min(zen) < 10)
                          or (max(abs(self.min_lat), abs(self.max_lat)) > 80))

        azi = da.from_array(azi, chunks=chunks)
        zen = da.from_array(zen, chunks=chunks)

        if switch_to_cart:
            arrays = convert_from_angles(azi, zen)
        else:
            arrays = (azi, zen)

        expanded = self.expand_angle_and_nav(arrays)
        if switch_to_cart:
            return convert_to_angles(*expanded)

        return expanded


def convert_from_angles(azi, zen):
    """Convert the angles to cartesian coordinates."""
    x, y, z, = angle2xyz(azi, zen)
    # Conversion to ECEF is recommended by the provider, but no significant
    # difference has been seen.
    # x, y, z = (-np.sin(lon) * x + np.cos(lon) * y,
    #            -np.sin(lat) * np.cos(lon) * x - np.sin(lat) * np.sin(lon) * y + np.cos(lat) * z,
    #            np.cos(lat) * np.cos(lon) * x + np.cos(lat) * np.sin(lon) * y + np.sin(lat) * z)
    return x, y, z


def convert_to_angles(x, y, z):
    """Convert the cartesian coordinates to angles."""
    # Conversion to ECEF is recommended by the provider, but no significant
    # difference has been seen.
    # x, y, z = (-np.sin(lon) * x - np.sin(lat) * np.cos(lon) * y + np.cos(lat) * np.cos(lon) * z,
    #            np.cos(lon) * x - np.sin(lat) * np.sin(lon) * y + np.cos(lat) * np.sin(lon) * z,
    #            np.cos(lat) * y + np.sin(lat) * z)
    azi, zen = xyz2angle(x, y, z, acos=True)
    return azi, zen


def expand_arrays(arrays,
                  scans,
                  c_align,
                  c_exp,
                  scan_size=16,
                  tpz_size=16,
                  nties=200,
                  track_offset=0.5,
                  scan_offset=0.5):
    """Expand *data* according to alignment and expansion."""
    nties = nties.item()
    tpz_size = tpz_size.item()
    s_scan, s_track = da.meshgrid(da.arange(nties * tpz_size),
                                  da.arange(scans * scan_size))
    s_track = (s_track.reshape(scans, scan_size, nties, tpz_size) % scan_size
               + track_offset) / scan_size
    s_scan = (s_scan.reshape(scans, scan_size, nties, tpz_size) % tpz_size
              + scan_offset) / tpz_size

    a_scan = s_scan + s_scan * (1 - s_scan) * c_exp + s_track * (
        1 - s_track) * c_align
    a_track = s_track
    expanded = []
    coef_a = (1 - a_track) * (1 - a_scan)
    coef_b = (1 - a_track) * a_scan
    coef_d = a_track * (1 - a_scan)
    coef_c = a_track * a_scan
    for data in arrays:
        data_a = data[:scans * 2:2, np.newaxis, :-1, np.newaxis]
        data_b = data[:scans * 2:2, np.newaxis, 1:, np.newaxis]
        data_c = data[1:scans * 2:2, np.newaxis, 1:, np.newaxis]
        data_d = data[1:scans * 2:2, np.newaxis, :-1, np.newaxis]
        fdata = (coef_a * data_a + coef_b * data_b
                 + coef_d * data_d + coef_c * data_c)
        expanded.append(fdata.reshape(scans * scan_size, nties * tpz_size))
    return expanded
