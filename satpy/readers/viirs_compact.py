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
from ConfigParser import ConfigParser
from datetime import datetime, timedelta

import h5py
import numpy as np
from pyresample.geometry import SwathDefinition

from mpop import CONFIG_PATH
from satpy.projectable import Projectable
from satpy.readers.file_handlers import BaseFileHandler

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


class VIIRSCompactMFileHandler(BaseFileHandler):
    def __init__(self, filename, filename_info, filetype_info):
        super(VIIRSCompactMFileHandler, self).__init__(filename, filename_info,
                                                       filetype_info)
        self.h5f = h5py.File(self.filename, "r")
        self.finfo = filename_info

    def get_dataset(self, key, info):
        m_data = self.read_m(key)
        m_data.info.update(info)
        lons, lats = self.navigate_m(key)
        m_area_def = SwathDefinition(
            np.ma.masked_where(m_data.mask, lons),
            np.ma.masked_where(m_data.mask, lats))
        m_area_def.name = 'yo'
        m_data.info['area'] = m_area_def
        return m_data

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

    def read_m(self, key, calibrate=1):

        h5f = self.h5f
        channel = chans_dict[key.name]
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
                              arr * rads.attrs['RadianceScaleHigh'] + \
                              rads.attrs['RadianceOffsetHigh'],)
            arr_mask = arr.mask
        except KeyError:
            print "KeyError"
            pass
        unit = "W m-2 sr-1 μm-1"
        if calibrate == 0:
            raise NotImplementedError("Can't get counts from this data")
        if calibrate == 1:
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
                    arr = (h * c) / (k * lambda_c * \
                                     np.log(1 +
                                            (2 * h * c ** 2) /
                                            ((lambda_c ** 5) * arr)))
                    arr *= a_ir
                    arr += b_ir
                    unit = "K"
                except KeyError:
                    logger.warning("Calibration failed.")

        elif calibrate != 2:
            raise ValueError("Calibrate parameter should be 1 or 2")
        arr[arr < 0] = 0

        return Projectable(arr, units=unit)

    def navigate_m(self, key):
        h5f = self.h5f
        channel = chans_dict[key.name]

        if not channel.startswith("M"):
            raise ValueError("Unknow channel type for band %s", channel)

        scans = h5f["All_Data"]["NumberOfScans"][0]
        geostuff = h5f["All_Data"]["VIIRS-MOD-GEO_All"]
        all_c_align = geostuff["AlignmentCoefficient"].value[
            np.newaxis, np.newaxis, :, np.newaxis]
        all_c_exp = geostuff["ExpansionCoefficient"].value[
            np.newaxis, np.newaxis, :, np.newaxis]
        all_lon = geostuff["Longitude"].value
        all_lat = geostuff["Latitude"].value

        res = []

        # FIXME:  this supposes  there is  only one  tiepoint zone  in the
        # track direction
        scan_size = h5f["All_Data/VIIRS-%s-SDR_All" % \
                        channel].attrs["TiePointZoneSizeTrack"][0]
        track_offset = h5f["All_Data/VIIRS-%s-SDR_All" % \
                           channel].attrs["PixelOffsetTrack"]
        scan_offset = h5f["All_Data/VIIRS-%s-SDR_All" % \
                          channel].attrs["PixelOffsetScan"]

        try:
            group_locations = h5f["All_Data/VIIRS-MOD-GEO_All/"
                                  "TiePointZoneGroupLocationScanCompact"].value
        except KeyError:
            group_locations = [0]
        param_start = 0
        for tpz_size, nb_tpz, start in \
            zip(h5f["All_Data/VIIRS-%s-SDR_All" % \
                    channel].attrs["TiePointZoneSizeScan"],
                h5f["All_Data/VIIRS-MOD-GEO_All/NumberOfTiePointZonesScan"].value,
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
                                 nties, track_offset, scan_offset),
                    expand_array(z, scans, c_align, c_exp, scan_size, tpz_size,
                                 nties, track_offset, scan_offset))
                res.append(xyz2lonlat(x, y, z))
            else:
                res.append(
                    (expand_array(lon, scans, c_align, c_exp, scan_size,
                                  tpz_size, nties, track_offset, scan_offset),
                     expand_array(lat, scans, c_align, c_exp, scan_size,
                                  tpz_size, nties, track_offset, scan_offset)))
        lons, lats = zip(*res)
        return np.hstack(lons), np.hstack(lats)


def load(satscene, *args, **kwargs):
    del args

    files_to_load = []
    files_to_delete = []

    try:
        filename = kwargs.get("filename")
        logger.debug("reading %s", str(filename))
        if filename is not None:
            if isinstance(filename, (list, set, tuple)):
                files = filename
            else:
                files = [filename]
            files_to_load = []
            for filename in files:
                pathname, ext = os.path.splitext(filename)
                if ext == ".bz2":
                    zipfile = bz2.BZ2File(filename)
                    newname = os.path.join("/tmp", os.path.basename(pathname))
                    if not os.path.exists(newname):
                        with open(newname, "wb") as fp_:
                            fp_.write(zipfile.read())
                    zipfile.close()
                    files_to_load.append(newname)
                    files_to_delete.append(newname)
                else:
                    files_to_load.append(filename)
        else:
            time_start, time_end = kwargs.get("time_interval",
                                              (satscene.time_slot, None))

            conf = ConfigParser()
            conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
            options = {}
            for option, value in conf.items(
                    satscene.instrument_name + "-level2",
                    raw=True):
                options[option] = value

            template = os.path.join(options["dir"], options["filename"])

            second = timedelta(seconds=1)
            files_to_load = []

            if time_end is not None:
                time = time_start - second * 85
                files_to_load = []
                while time <= time_end:
                    fname = time.strftime(template)
                    flist = glob.glob(fname)
                    try:
                        files_to_load.append(flist[0])
                        time += second * 80
                    except IndexError:
                        pass
                    time += second

            else:
                files_to_load = glob.glob(time_start.strftime(template))

        chan_dict = {"M01": "M1",
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

        channels = [(chn, chan_dict[chn]) for chn in satscene.channels_to_load
                    if chn in chan_dict]
        try:
            channels_to_load, chans = zip(*channels)
        except ValueError:
            return

        m_chans = []
        dnb_chan = []
        for chn in chans:
            if chn.startswith('M'):
                m_chans.append(chn)
            elif chn.startswith('DNB'):
                dnb_chan.append(chn)
            else:
                raise ValueError("Reading of channel %s not implemented", chn)

        m_datas = []
        m_lonlats = []
        dnb_datas = []
        dnb_lonlats = []

        for fname in files_to_load:
            is_dnb = os.path.basename(fname).startswith('SVDNBC')
            logger.debug("Reading %s", fname)
            if is_dnb:
                if tables:
                    h5f = tables.open_file(fname, "r")
                else:
                    logger.warning("DNB data could not be read from %s, "
                                   "PyTables not available.", fname)
                    continue
            else:
                h5f = h5py.File(fname, "r")
            if m_chans and not is_dnb:
                try:
                    arr, m_units = read_m(h5f, m_chans)
                    m_datas.append(arr)
                    m_lonlats.append(navigate_m(h5f, m_chans[0]))
                except KeyError:
                    pass
            if dnb_chan and is_dnb and tables:
                try:
                    arr, dnb_units = read_dnb(h5f)
                    dnb_datas.append(arr)
                    dnb_lonlats.append(navigate_dnb(h5f))
                except KeyError:
                    pass
            h5f.close()

        if len(m_lonlats) > 0:
            m_lons = np.ma.vstack([lonlat[0] for lonlat in m_lonlats])
            m_lats = np.ma.vstack([lonlat[1] for lonlat in m_lonlats])
        if len(dnb_lonlats) > 0:
            dnb_lons = np.ma.vstack([lonlat[0] for lonlat in dnb_lonlats])
            dnb_lats = np.ma.vstack([lonlat[1] for lonlat in dnb_lonlats])

        m_i = 0
        dnb_i = 0
        for chn in channels_to_load:
            if m_datas and chn.startswith('M'):
                m_data = np.ma.vstack([dat[m_i] for dat in m_datas])
                satscene[chn] = m_data
                satscene[chn].info["units"] = m_units[m_i]
                m_i += 1
            if dnb_datas and chn.startswith('DNB'):
                dnb_data = np.ma.vstack([dat[dnb_i] for dat in dnb_datas])
                satscene[chn] = dnb_data
                satscene[chn].info["units"] = dnb_units[dnb_i]
                dnb_i += 1

        if m_datas:
            m_area_def = SwathDefinition(
                np.ma.masked_where(m_data.mask, m_lons),
                np.ma.masked_where(m_data.mask, m_lats))
        else:
            logger.warning("No M channel data available.")

        if dnb_datas:
            dnb_area_def = SwathDefinition(
                np.ma.masked_where(dnb_data.mask,
                                   dnb_lons), np.ma.masked_where(dnb_data.mask,
                                                                 dnb_lats))
        else:
            logger.warning("No DNB data available.")

        for chn in channels_to_load:
            if "DNB" not in chn and m_datas:
                satscene[chn].area = m_area_def

        if dnb_datas:
            for chn in dnb_chan:
                satscene[chn].area = dnb_area_def

    finally:
        for fname in files_to_delete:
            if os.path.exists(fname):
                os.remove(fname)


def read_m(h5f, channels, calibrate=1):

    chan_dict = dict([(key.split("-")[
        1], key) for key in h5f["All_Data"].keys() if key.startswith("VIIRS")])

    scans = h5f["All_Data"]["NumberOfScans"][0]
    res = []
    units = []
    arr_mask = np.ma.nomask

    for channel in channels:
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
                              arr * rads.attrs['RadianceScaleHigh'] + \
                              rads.attrs['RadianceOffsetHigh'],)
            arr_mask = arr.mask
        except KeyError:
            print "KeyError"
            pass
        unit = "W m-2 sr-1 μm-1"
        if calibrate == 0:
            raise NotImplementedError("Can't get counts from this data")
        if calibrate == 1:
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
                    arr = (h * c) / (k * lambda_c * \
                                     np.log(1 +
                                            (2 * h * c ** 2) /
                                            ((lambda_c ** 5) * arr)))
                    arr *= a_ir
                    arr += b_ir
                    unit = "K"
                except KeyError:
                    logger.warning("Calibration failed.")

        elif calibrate != 2:
            raise ValueError("Calibrate parameter should be 1 or 2")
        arr[arr < 0] = 0
        res.append(arr)
        units.append(unit)

    return res, units


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


def lonlat2xyz(lon, lat):
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x, y, z


def xyz2lonlat(x, y, z):
    lon = np.rad2deg(np.arctan2(y, x))
    lat = np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2)))
    return lon, lat


def navigate_m(h5f, channel):

    if not channel.startswith("M"):
        raise ValueError("Unknow channel type for band %s", channel)

    scans = h5f["All_Data"]["NumberOfScans"][0]
    geostuff = h5f["All_Data"]["VIIRS-MOD-GEO_All"]
    all_c_align = geostuff["AlignmentCoefficient"].value[
        np.newaxis, np.newaxis, :, np.newaxis]
    all_c_exp = geostuff["ExpansionCoefficient"].value[
        np.newaxis, np.newaxis, :, np.newaxis]
    all_lon = geostuff["Longitude"].value
    all_lat = geostuff["Latitude"].value

    res = []

    # FIXME:  this supposes  there is  only one  tiepoint zone  in the
    # track direction
    scan_size = h5f["All_Data/VIIRS-%s-SDR_All" % \
                    channel].attrs["TiePointZoneSizeTrack"][0]
    track_offset = h5f["All_Data/VIIRS-%s-SDR_All" % \
                       channel].attrs["PixelOffsetTrack"]
    scan_offset = h5f["All_Data/VIIRS-%s-SDR_All" % \
                      channel].attrs["PixelOffsetScan"]

    try:
        group_locations = h5f["All_Data/VIIRS-MOD-GEO_All/"
                              "TiePointZoneGroupLocationScanCompact"].value
    except KeyError:
        group_locations = [0]
    param_start = 0
    for tpz_size, nb_tpz, start in \
        zip(h5f["All_Data/VIIRS-%s-SDR_All" % \
                channel].attrs["TiePointZoneSizeScan"],
            h5f["All_Data/VIIRS-MOD-GEO_All/NumberOfTiePointZonesScan"].value,
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


if __name__ == '__main__':
    #filename = "/local_disk/data/satellite/polar/compact_viirs/SVMC_npp_d20140114_t1245125_e1246367_b11480_c20140114125427496143_eum_ops.h5"
    filename = "/local_disk/data/satellite/polar/compact_viirs/mymy.h5"
    h5f = h5py.File(filename, 'r')
    # ch1, ch2, ch3, ch4 = read(h5f, ["M5", "M4", "M2", "M12"])
    # img = GeoImage((ch1, ch2, ch3),
    #                None,
    #                None,
    #                fill_value=None,
    #                mode="RGB")

    # img.enhance(stretch="linear")
    # img.enhance(gamma=2.0)
    # img.show()

    lons, lats = navigate_m(h5f)
