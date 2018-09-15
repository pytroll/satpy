#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2010-2018 PyTroll Community

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Bybbroe <adam.dybbroe@smhi.se>
#   Sauli Joro <sauli.joro@eumetsat.int>

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

"""HDF5 format reader.

References:
    MSG Level 1.5 Image Data FormatDescription

TODO:
- HRV navigation

"""

import logging
from datetime import datetime, timedelta

import numpy as np

from pyresample import geometry
from satpy.readers.hdf5_utils import HDF5FileHandler

from satpy.readers.msg_base import SEVIRICalibrationHandler
from satpy.readers.msg_base import (CHANNEL_NAMES, CALIB, SATNUM)
from satpy.readers.eum_base import timecds2datetime

logger = logging.getLogger("hdf5_msg")


def make_time_cds_expanded(tcds_array):
    return (datetime(1958, 1, 1) +
            timedelta(days=int(tcds_array["days"]),
                      milliseconds=int(tcds_array["milliseconds"]),
                      microseconds=float(tcds_array["microseconds"] +
                                         tcds_array["nanoseconds"] / 1000.)))


def subdict(keys, value):
    """
    Takes a list and a value and if the list contains more than one key
    subdictionarys for each key will be created.

    Parameters
    ----------
    keys : list of str
        List of one or more strings
    value : any
        Value to assign to (sub)dictionary key

    Returns
    -------
    dict
        Dict or dict of dicts

    """
    tdict = {}
    key = keys[0]
    if len(keys) == 1:
        tdict[key] = value
    else:
        keys.remove(key)
        tdict[key] = subdict(keys, value)
    return tdict

import collections
from collections import defaultdict


def dict_merge(dct, merge_dct):
    """
    Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    
    Parameters
    ----------
    dct : dict
        dict onto which the merge is executed
    merge_dct : dct
        dct merged into dct
        
    Returns
    -------
    None
    """
    for k, v in merge_dct.iteritems():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


def rec2dict(arr):
    """
    Converts an array of attributes to a dictionary.

    Parameters
    ----------
    arr : ndarray
        DESCR array from hdf5 MSG data file

    Returns
    -------
    dict

    """
    res = {}
    for dtuple in arr:
        fullkey = dtuple[0].split("-")
        key = fullkey[0]
        data = dtuple[1]
        ndict = subdict(fullkey, data)
        dict_merge(res, ndict)
    return res



class HRITMSGPrologueFileHandler(HRITFileHandler):

    """MSG HRIT format reader
    """

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super(HRITMSGPrologueFileHandler, self).__init__(filename, filename_info,
                                                         filetype_info,
                                                         (msg_hdr_map,
                                                          msg_variable_length_headers,
                                                          msg_text_headers))

        self.prologue = {}
        self.read_prologue()

        service = filename_info["service"]
        if service == "":
            self.mda["service"] = "0DEG"
        else:
            self.mda["service"] = service

    def read_prologue(self):
        """Read the prologue metadata."""
        with open(self.filename, "rb") as fp_:
            fp_.seek(self.mda["total_header_length"])
            data = np.fromfile(fp_, dtype=prologue, count=1)[0]

            self.prologue.update(recarray2dict(data))

            try:
                impf = np.fromfile(fp_, dtype=impf_configuration, count=1)[0]
            except IndexError:
                logger.info("No IMPF configuration field found in prologue.")
            else:
                self.prologue.update(recarray2dict(impf))

        self.process_prologue()

    def process_prologue(self):
        """Reprocess prologue to correct types."""
        pacqtime = self.prologue["ImageAcquisition"]["PlannedAcquisitionTime"]

        start = pacqtime["TrueRepeatCycleStart"]
        psend = pacqtime["PlannedForwardScanEnd"]
        prend = pacqtime["PlannedRepeatCycleEnd"]

        start = make_time_cds_expanded(start)
        psend = make_time_cds_expanded(psend)
        prend = make_time_cds_expanded(prend)

        pacqtime["TrueRepeatCycleStart"] = start
        pacqtime["PlannedForwardScanEnd"] = psend
        pacqtime["PlannedRepeatCycleEnd"] = prend



class HRITMSGEpilogueFileHandler(HRITFileHandler):

    """MSG HRIT format reader
    """

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super(HRITMSGEpilogueFileHandler, self).__init__(filename, filename_info,
                                                         filetype_info,
                                                         (msg_hdr_map,
                                                          msg_variable_length_headers,
                                                          msg_text_headers))
        self.epilogue = {}
        self.read_epilogue()

        service = filename_info["service"]
        if service == "":
            self.mda["service"] = "0DEG"
        else:
            self.mda["service"] = service

    def read_epilogue(self):
        """Read the prologue metadata."""
        with open(self.filename, "rb") as fp_:
            fp_.seek(self.mda["total_header_length"])
            data = np.fromfile(fp_, dtype=epilogue, count=1)[0]

            self.epilogue.update(recarray2dict(data))

        pacqtime = self.epilogue["ImageProductionStats"][
            "ActualScanningSummary"]

        start = pacqtime["ForwardScanStart"]
        end = pacqtime["ForwardScanEnd"]

        start = make_time_cds_short(start)
        end = make_time_cds_short(end)

        pacqtime["ForwardScanEnd"] = end
        pacqtime["ForwardScanStart"] = start


class HDF5MSGFileHandler(HDF5FileHandler, SEVIRICalibrationHandler):

    """MSG HDF5 format reader
    """

    def __init__(self, filename, filename_info, filetype_info,
                 prologue, epilogue):
        """Initialize the reader."""
        super(HDF5MSGFileHandler, self).__init__(filename, filename_info,
                                                 filetype_info,
                                                 (msg_hdr_map,
                                                  msg_variable_length_headers,
                                                  msg_text_headers))

        self.prologue = prologue.prologue
        self.epilogue = epilogue.epilogue
        self._filename_info = filename_info

        self._get_header()

    def get_metadata(self, ds_id, ds_info):
        """
        Get the metadata for specific dataset listed in yaml config


        :param ds_id:
        :param ds_info:
        :return:
        """
        pass


    def _get_header(self):
        """Read the header info, and fill the metadata dictionary"""

        self.mda = defaultdict(dict)

        for k, d in self.file_content.items():
            if isinstance(d, h5py.Dataset) and k.endswith("DESCR") or k.endswith("SUBSET"):
                group = k.split("/")[-2]
                key = k.split("/")[-1]
                dset = self[k]
                self.mda[group][key] = rec2dict(dset)

        self.calib_coeffs = np.array(self["U-MARF/MSG/Level1.5/RadiometricProcessing/Level15ImageCalibration_ARRAY"])

        earth_model = self.mda["GeometricProcessing"]["GeometricProcessing_DESCR"]["EarthModel"]
        self.mda["offset_corrected"] = earth_model["TypeOfEarthModel"] == 1
        b = (earth_model["NorthPolarRadius"] + earth_model["SouthPolarRadius"]) / 2.0 * 1000
        self.mda["projection_parameters"]["a"] = earth_model["EquatorialRadius"] * 1000
        self.mda["projection_parameters"]["b"] = b
        ssp = self.mda["ImageDescription"]["ImageDescription_DESCR"]["ProjectionDescription"]["LongitudeOfSSP"]
        self.mda["projection_parameters"]["SSP_longitude"] = ssp
        self.mda["projection_parameters"]["SSP_latitude"] = 0.0
        self.platform_id = self.mda["ImageProductionStats"]["ImageProductionStats_DESCR"]["SatelliteID"]
        self.platform_name = "Meteosat-" + SATNUM[self.platform_id]
        self.mda["platform_name"] = self.platform_name
        service = self._filename_info["service"]
        if service == "":
            self.mda["service"] = "0DEG"
        else:
            self.mda["service"] = service
        self.channel_name = CHANNEL_NAMES[self.mda["spectral_channel_id"]]

    @property
    def start_time(self):
        time = self.mda["ImageProductionStats"]["ImageProductionStats_DESCR"]["ActualScanningSummary"]["ForwardScanStart"]

        return timecds2datetime({k.capitalize(): v for k,v in time.items()})

    @property
    def end_time(self):
        time = self.mda["ImageProductionStats"]["ImageProductionStats_DESCR"]["ActualScanningSummary"]["ForwardScanEnd"]

        return timecds2datetime({k.capitalize(): v for k,v in time.items()})

    def get_xy_from_linecol(self, line, col, offsets, factors):
        """Get the intermediate coordinates from line & col.

        Intermediate coordinates are actually the instruments scanning angles.
        """
        loff, coff = offsets
        lfac, cfac = factors
        x__ = (col - coff) / cfac * 2**16
        y__ = - (line - loff) / lfac * 2**16

        return x__, y__

    def get_area_extent(self, size, offsets, factors, platform_height):
        """Get the area extent of the file."""
        nlines, ncols = size
        h = platform_height

        loff, coff = offsets
        loff -= nlines
        offsets = loff, coff
        # count starts at 1
        cols = 1 - 0.5
        lines = 1 - 0.5
        ll_x, ll_y = self.get_xy_from_linecol(-lines, cols, offsets, factors)

        cols += ncols
        lines += nlines
        ur_x, ur_y = self.get_xy_from_linecol(-lines, cols, offsets, factors)

        aex = (np.deg2rad(ll_x) * h, np.deg2rad(ll_y) * h,
               np.deg2rad(ur_x) * h, np.deg2rad(ur_y) * h)

        #offset corrected?
        if not self.mda["offset_corrected"]:
            xadj = 1500
            yadj = 1500
            aex = (aex[0] + xadj, aex[1] + yadj,
                   aex[2] + xadj, aex[3] + yadj)

        return aex

    def get_area_def(self, dsid):
        """Get the area definition of the band."""
        if dsid.name != "HRV":
            return super(HRITMSGFileHandler, self).get_area_def(dsid)

        cfac = np.int32(self.mda["cfac"])
        lfac = np.int32(self.mda["lfac"])
        loff = np.float32(self.mda["loff"])


        b = self.mda["projection_parameters"]["b"]
        a = self.mda["projection_parameters"]["a"]
        h = self.mda["projection_parameters"]["h"]
        lon_0 = self.mda["projection_parameters"]["SSP_longitude"]



        nlines = int(self.mda["number_of_lines"])
        ncols = int(self.mda["number_of_columns"])

        segment_number = self.mda["segment_sequence_number"]

        current_first_line = (segment_number -
                              self.mda["planned_start_segment_number"]) * nlines
        bounds = self.epilogue["ImageProductionStats"]["ActualL15CoverageHRV"]

        upper_south_line = bounds[
            "LowerNorthLineActual"] - current_first_line - 1
        upper_south_line = min(max(upper_south_line, 0), nlines)

        lower_coff = (5566 - bounds["LowerEastColumnActual"] + 1)
        upper_coff = (5566 - bounds["UpperEastColumnActual"] + 1)

        lower_area_extent = self.get_area_extent((upper_south_line, ncols),
                                                 (loff, lower_coff),
                                                 (lfac, cfac),
                                                 h)

        upper_area_extent = self.get_area_extent((nlines - upper_south_line,
                                                  ncols),
                                                 (loff - upper_south_line,
                                                  upper_coff),
                                                 (lfac, cfac),
                                                 h)

        proj_dict = {"a": float(a),
                     "b": float(b),
                     "lon_0": float(lon_0),
                     "h": float(h),
                     "proj": "geos",
                     "units": "m"}

        lower_area = geometry.AreaDefinition(
            "some_area_name",
            "On-the-fly area",
            "geosmsg",
            proj_dict,
            ncols,
            upper_south_line,
            lower_area_extent)

        upper_area = geometry.AreaDefinition(
            "some_area_name",
            "On-the-fly area",
            "geosmsg",
            proj_dict,
            ncols,
            nlines - upper_south_line,
            upper_area_extent)

        area = geometry.StackedAreaDefinition(lower_area, upper_area)

        self.area = area.squeeze()
        return area

    def get_dataset(self, dataset_id, ds_info):
        #res = super(HRITMSGFileHandler, self).get_dataset(key, info)
        ds_path = ds_info.get("file_key", "{}".format(dataset_id))
        res = self.["U-MARF/MSG/Level1.5/DATA/" + ds_path + "/IMAGE_DATA"]

        res = self.calibrate(res, key.calibration)
        res.attrs["units"] = info["units"]
        res.attrs["wavelength"] = info["wavelength"]
        res.attrs["standard_name"] = info["standard_name"]
        res.attrs["platform_name"] = self.platform_name
        res.attrs["sensor"] = "seviri"
        res.attrs["satellite_longitude"] = self.mda["projection_parameters"]["SSP_longitude"]
        res.attrs["satellite_latitude"] = self.mda["projection_parameters"]["SSP_latitude"]
        res.attrs["satellite_altitude"] = self.mda["projection_parameters"]["h"]
        return res

    def calibrate(self, data, calibration):
        """Calibrate the data."""
        tic = datetime.now()

        channel_id = int(self.mda[ds_path]["LineSideInfo_DESCR"]["ChannelID"])
        channel_name = CHANNEL_NAMES[channel_id]

        if calibration == "counts":
            res = data
        elif calibration in ["radiance", "reflectance", "brightness_temperature"]:
            gain = self.calib_coeffs["Cal_Slope"][channel_id - 1]
            offset = self.calib_coeffs["Cal_Offset"][channel_id - 1]
            data = data.where(data > 0)
            res = self._convert_to_radiance(data.astype(np.float32), gain, offset)
            #line_mask = self.mda["image_segment_line_quality"]["line_validity"] >= 2
            #line_mask &= self.mda["image_segment_line_quality"]["line_validity"] <= 3
            #line_mask &= self.mda["image_segment_line_quality"]["line_radiometric_quality"] == 4
            #line_mask &= self.mda["image_segment_line_quality"]["line_geometric_quality"] == 4
            #res *= np.choose(line_mask, [1, np.nan])[:, np.newaxis].astype(np.float32)

        if calibration == "reflectance":
            solar_irradiance = CALIB[self.platform_id][channel_name]["F"]
            res = self._vis_calibrate(res, solar_irradiance)

        elif calibration == "brightness_temperature":
            cal_type = self.mda["ImageDescription"]["ImageDescription_DESCR"]["Level 1_5 ImageProduction"]["PlannedChanProcessing"][channel_id - 1]
            res = self._ir_calibrate(res, channel_name, cal_type)

        logger.debug("Calibration time " + str(datetime.now() - tic))
        return res


def show(data, negate=False):
    """Show the stretched data.
    """
    from PIL import Image as pil
    data = np.array((data - data.min()) * 255.0 /
                    (data.max() - data.min()), np.uint8)
    if negate:
        data = 255 - data
    img = pil.fromarray(data)
    img.show()
