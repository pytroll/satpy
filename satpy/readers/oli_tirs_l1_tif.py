#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Satpy developers
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
"""Landsat OLI/TIRS Level 1 reader.

Details of the data format can be found here:
  https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1822_Landsat8-9-OLI-TIRS-C2-L1-DFCB-v6.pdf
  https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product

NOTE: The scene geometry data (SZA, VZA, SAA, VAA) is retrieved from the L1 TIFF files, which are derived from Band 04.
The geometry differs between bands, so if you need precise geometry you should calculate this from the metadata instead.

"""

import logging
from datetime import datetime, timezone

import defusedxml.ElementTree as ET
import numpy as np
import xarray as xr

from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)

PLATFORMS = {"08": "Landsat-8",
             "09": "Landsat-9"}

OLI_BANDLIST = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"]
TIRS_BANDLIST = ["B10", "B11"]
PAN_BANDLIST = ["B8"]
ANGLIST = ["satellite_azimuth_angle",
           "satellite_zenith_angle",
           "solar_azimuth_angle",
           "solar_zenith_angle"]

ANGLIST_CHAN = ["sza", "saa", "vaa", "vza"]

BANDLIST = OLI_BANDLIST + TIRS_BANDLIST


class OLITIRSCHReader(BaseFileHandler):
    """File handler for Landsat L1 files (tif)."""

    @staticmethod
    def get_btype(file_type):
        """Return the band type from the file type."""
        pos = file_type.rfind("_")
        if pos == -1:
            raise ValueError(f"Invalid file type: {file_type}")
        else:
            return file_type[pos+1:]

    @property
    def start_time(self):
        """Return start time."""
        return self._mda.start_time

    @property
    def end_time(self):
        """Return end time."""
        return self._mda.end_time

    def __init__(self, filename, filename_info, filetype_info, mda, **kwargs):
        """Initialize the reader."""
        super().__init__(filename, filename_info, filetype_info)

        # Check we have landsat data
        if filename_info["platform_type"] != "L":
            raise ValueError("This reader only supports Landsat data")

        # Get the channel name
        self.channel = self.get_btype(filetype_info["file_type"])

        # Data can be VIS, TIR or Combined. This flag denotes what the granule contains (O, T or C respectively).
        self.chan_selector = filename_info["data_type"]

        self._obs_date = filename_info["observation_date"]
        self._mda = mda

        # Retrieve some per-band useful metadata
        self.bsat = self._mda.band_saturation
        self.calinfo = self._mda.band_calibration
        self.platform_name = PLATFORMS[filename_info["spacecraft_id"]]

    def get_dataset(self, key, info):
        """Load a dataset."""
        if self.channel != key["name"] and self.channel not in ANGLIST_CHAN:
            raise ValueError(f"Requested channel {key['name']} does not match the reader channel {self.channel}")

        if key["name"] in OLI_BANDLIST and self.chan_selector not in ["O", "C"]:
            raise ValueError(f"Requested channel {key['name']} is not available in this granule")
        if key["name"] in TIRS_BANDLIST and self.chan_selector not in ["T", "C"]:
            raise ValueError(f"Requested channel {key['name']} is not available in this granule")

        logger.debug("Reading %s.", key["name"])

        data = xr.open_dataarray(self.filename, engine="rasterio",
                                 chunks={"band": 1,
                                         "y": "auto",
                                         "x": "auto"},
                                 mask_and_scale=False).squeeze()


        # The fill value for Landsat is '0', for calibration simplicity convert it to np.nan
        data.data = xr.where(data.data == 0, np.float32(np.nan), data.data)

        attrs = data.attrs.copy()
        # Add useful metadata to the attributes.
        attrs["perc_cloud_cover"] = self._mda.cloud_cover
        # Add platform / sensor attributes
        attrs["platform_name"] = self.platform_name
        attrs["sensor"] = "OLI_TIRS"
        # Apply attrs from YAML
        attrs["standard_name"] = info["standard_name"]
        attrs["units"] = info["units"]

        # Only OLI bands have a saturation flag
        if key["name"] in OLI_BANDLIST:
            attrs["saturated"] = self.bsat[key["name"]]

        # Rename to Satpy convention
        data = data.rename({"band": "bands"})

        data.attrs.update(attrs)

        # Calibrate if we're using a band rather than a QA or geometry dataset
        if key["name"] in BANDLIST:
            data = self.calibrate(data, key["calibration"])
        if key["name"] in ANGLIST:
            data.data = data.data * 0.01

        return data

    def calibrate(self, data, calibration):
        """Calibrate the data from counts into the desired units."""
        if calibration == "counts":
            return data

        if calibration in ["radiance", "brightness_temperature"]:
            data.data = data.data * self.calinfo[self.channel][0] + self.calinfo[self.channel][1]
            if calibration == "radiance":
                data.data = data.data.astype(np.float32)
                return data

        if calibration == "reflectance":
            if int(self.channel[1:]) < 10:
                data.data = data.data * self.calinfo[self.channel][2] + self.calinfo[self.channel][3]
                data.data = data.data.astype(np.float32) * 100
                return data

        if calibration == "brightness_temperature":
            if self.channel[1:] in ["10", "11"]:
                data.data = (self.calinfo[self.channel][3] / np.log((self.calinfo[self.channel][2] / data.data) + 1))
                data.data = data.data.astype(np.float32)
                return data

    def get_area_def(self, dsid):
        """Get area definition of the image from the metadata."""
        return self._mda.build_area_def(dsid["name"])


class OLITIRSMDReader(BaseFileHandler):
    """File handler for Landsat L1 files (tif)."""
    def __init__(self, filename, filename_info, filetype_info):
        """Init the reader."""
        super().__init__(filename, filename_info, filetype_info)
        # Check we have landsat data
        if filename_info["platform_type"] != "L":
            raise ValueError("This reader only supports Landsat data")
        self.platform_name = PLATFORMS[filename_info["spacecraft_id"]]
        self._obs_date = filename_info["observation_date"]
        self.root = ET.parse(self.filename)
        self.process_level = filename_info["process_level_correction"]
        import bottleneck  # noqa
        import geotiepoints  # noqa


    @property
    def center_time(self):
        """Return center time."""
        return datetime.strptime(self.root.find(".//IMAGE_ATTRIBUTES/SCENE_CENTER_TIME").text[:-2],
                                 "%H:%M:%S.%f").replace(tzinfo=timezone.utc)

    @property
    def start_time(self):
        """Return start time.

        This is actually the scene center time, as we don't have the start time.
        It is constructed from the observation date (from the filename) and the center time (from the metadata).
        """
        return datetime(self._obs_date.year, self._obs_date.month, self._obs_date.day,
                        self.center_time.hour, self.center_time.minute, self.center_time.second,
                        tzinfo=timezone.utc)

    @property
    def end_time(self):
        """Return end time.

        This is actually the scene center time, as we don't have the end time.
        It is constructed from the observation date (from the filename) and the center time (from the metadata).
        """
        return datetime(self._obs_date.year, self._obs_date.month, self._obs_date.day,
                        self.center_time.hour, self.center_time.minute, self.center_time.second,
                        tzinfo=timezone.utc)

    @property
    def cloud_cover(self):
        """Return estimated granule cloud cover percentage."""
        return float(self.root.find(".//IMAGE_ATTRIBUTES/CLOUD_COVER").text)

    def _get_satflag(self, band):
        """Return saturation flag for a band."""
        flag = self.root.find(f".//IMAGE_ATTRIBUTES/SATURATION_BAND_{band}").text
        if flag == "Y":
            return True
        return False

    @property
    def band_saturation(self):
        """Return per-band saturation flag."""
        bdict = {}
        for i in range(1, 10):
            bdict[f"B{i:01d}"] = self._get_satflag(i)

        return bdict

    def _get_band_radcal(self, band):
        """Get the radiance scale and offset values."""
        rad_gain = float(self.root.find(f".//LEVEL1_RADIOMETRIC_RESCALING/RADIANCE_MULT_BAND_{band}").text)
        rad_add = float(self.root.find(f".//LEVEL1_RADIOMETRIC_RESCALING/RADIANCE_ADD_BAND_{band}").text)
        return rad_gain, rad_add

    def _get_band_viscal(self, band):
        """Return visible channel calibration info."""
        rad_gain, rad_add = self._get_band_radcal(band)
        ref_gain = float(self.root.find(f".//LEVEL1_RADIOMETRIC_RESCALING/REFLECTANCE_MULT_BAND_{band}").text)
        ref_add = float(self.root.find(f".//LEVEL1_RADIOMETRIC_RESCALING/REFLECTANCE_ADD_BAND_{band}").text)
        return rad_gain, rad_add, ref_gain, ref_add

    def _get_band_tircal(self, band):
        """Return thermal channel calibration info."""
        rad_gain, rad_add = self._get_band_radcal(band)
        bt_k1 = float(self.root.find(f".//LEVEL1_THERMAL_CONSTANTS/K1_CONSTANT_BAND_{band}").text)
        bt_k2 = float(self.root.find(f".//LEVEL1_THERMAL_CONSTANTS/K2_CONSTANT_BAND_{band}").text)
        return rad_gain, rad_add, bt_k1, bt_k2

    @property
    def band_calibration(self):
        """Return per-band saturation flag."""
        bdict = {}
        for i in range(1, 10):
            bdict[f"B{i:01d}"] = self._get_band_viscal(i)
        for i in range(10, 12):
            bdict[f"B{i:02d}"] = self._get_band_tircal(i)

        return bdict

    def earth_sun_distance(self):
        """Return Earth-Sun distance."""
        return float(self.root.find(".//IMAGE_ATTRIBUTES/EARTH_SUN_DISTANCE").text)

    def build_area_def(self, bname):
        """Build area definition from metadata."""
        from pyresample.geometry import AreaDefinition

        # Here we assume that the thermal bands have the same resolution as the reflective bands,
        # with only the panchromatic band (b08) having a different resolution.
        if bname in PAN_BANDLIST:
            pixoff = float(self.root.find(".//PROJECTION_ATTRIBUTES/GRID_CELL_SIZE_PANCHROMATIC").text) / 2.
            x_size = float(self.root.find(".//PROJECTION_ATTRIBUTES/PANCHROMATIC_SAMPLES").text)
            y_size = float(self.root.find(".//PROJECTION_ATTRIBUTES/PANCHROMATIC_LINES").text)
        else:
            pixoff = float(self.root.find(".//PROJECTION_ATTRIBUTES/GRID_CELL_SIZE_REFLECTIVE").text) / 2.
            x_size = float(self.root.find(".//PROJECTION_ATTRIBUTES/REFLECTIVE_SAMPLES").text)
            y_size = float(self.root.find(".//PROJECTION_ATTRIBUTES/REFLECTIVE_LINES").text)

        # Get remaining geoinfo from file
        datum = self.root.find(".//PROJECTION_ATTRIBUTES/DATUM").text
        utm_zone = int(self.root.find(".//PROJECTION_ATTRIBUTES/UTM_ZONE").text)
        utm_str = f"{utm_zone}N"

        # We need to subtract / add half a pixel from the corner to get the correct extent (pixel centers)
        ext_p1 = float(self.root.find(".//PROJECTION_ATTRIBUTES/CORNER_UL_PROJECTION_X_PRODUCT").text) - pixoff
        ext_p2 = float(self.root.find(".//PROJECTION_ATTRIBUTES/CORNER_LR_PROJECTION_Y_PRODUCT").text) - pixoff
        ext_p3 = float(self.root.find(".//PROJECTION_ATTRIBUTES/CORNER_LR_PROJECTION_X_PRODUCT").text) + pixoff
        ext_p4 = float(self.root.find(".//PROJECTION_ATTRIBUTES/CORNER_UL_PROJECTION_Y_PRODUCT").text) + pixoff

        # Create area definition
        pcs_id = f"{datum} / UTM zone {utm_str}"
        proj4_dict = {"proj": "utm", "zone": utm_zone, "datum": datum, "units": "m", "no_defs": None, "type": "crs"}
        area_extent = (ext_p1, ext_p2, ext_p3, ext_p4)

        # Return the area extent
        return AreaDefinition("geotiff_area", pcs_id, pcs_id, proj4_dict, x_size, y_size, area_extent)
