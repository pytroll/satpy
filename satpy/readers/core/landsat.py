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
"""Landsat reader.

Details of the data format can be found here:
  https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1822_Landsat8-9-OLI-TIRS_C2_L1_DataFormatControlBook-v7.pdf
  https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1414_Landsat7ETM-C2-L1-DFCB-v3.pdf
  https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1415_Landsat4-5-TM-C2-L1-DFCB-v3.pdf
  https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1416_LandsatMSS-C2-L1-DFCB-v3.pdf
  https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1328_Landsat8-9_OLI-TIRS-C2-L2_DFCB-v7.pdf
  https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1337_Landsat7ETM-C2-L2-DFCB-v6.pdf
  https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1336_Landsat4-5-TM-C2-L2-DFCB-v4.pdf
  https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1927_Landsat7-Data-Users-Handbook-v3.pdf
  https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product
  https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products

NOTE: The scene geometry data (SZA, VZA, SAA, VAA) is retrieved from the L1 TIFF files, which are derived from Band 04.
The geometry differs between bands, so if you need precise geometry you should calculate this from the metadata instead.

"""

import logging
from datetime import datetime, timezone

import dask.array as da
import defusedxml.ElementTree as ET
import numpy as np
import rioxarray  # noqa: F401  # need by xarray with the engine rasterio
import xarray as xr

from satpy.readers.core.file_handlers import BaseFileHandler
from satpy.readers.core.remote import open_file_or_filename

logger = logging.getLogger(__name__)

PLATFORMS = {"01": "Landsat-1",
             "02": "Landsat-2",
             "03": "Landsat-3",
             "04": "Landsat-4",
             "05": "Landsat-5",
             "07": "Landsat-7",
             "08": "Landsat-8",
             "09": "Landsat-9"}

PAN_BANDLIST = ["B8"]

ANGLIST = ["satellite_azimuth_angle",
           "satellite_zenith_angle",
           "solar_azimuth_angle",
           "solar_zenith_angle"]

ANGLIST_CHAN = ["sza", "saa", "vaa", "vza"]
QALIST_CHAN = ["qa", "qa_radsat", "qa_aerosol", "qa_atmos_opacity", "qa_cloud", "qa_st"]


class BaseLandsatReader(BaseFileHandler):
    """Basic file handler for Landsat files (tif)."""

    @staticmethod
    def get_btype(file_type):
        """Return the band type from the file type."""
        if "_B" in file_type:
            pos = file_type.rfind("_B")
        else:
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

    @property
    def spectral_bands(self):
        """Landsat bands."""
        raise NotImplementedError()

    @property
    def thermal_bands(self):
        """Landsat thermal bands."""
        raise NotImplementedError()

    @property
    def sensor(self):
        """Sensor name."""
        raise NotImplementedError()

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
        self.collection = filename_info["collection_id"]

    def get_dataset(self, key, info):
        """Load a dataset."""
        if self.channel != key["name"] and self.channel not in ANGLIST_CHAN and key["name"] not in QALIST_CHAN:
            raise ValueError(f"Requested channel {key['name']} does not match the reader channel {self.channel}")

        # OLI-TIRS sensor data sometimes can contain only OLI or only TIRS data
        if self.sensor == "OLI_TIRS" and key["name"] in self.spectral_bands and self.chan_selector not in ["O", "C"]:
            raise ValueError(f"Requested channel {key['name']} is not available in this granule")
        if self.sensor == "OLI_TIRS" and key["name"] in self.thermal_bands and self.chan_selector not in ["T", "C"]:
            raise ValueError(f"Requested channel {key['name']} is not available in this granule")

        logger.debug("Reading %s.", key["name"])

        # xarray use the engine 'rasterio' to open the file, but
        #   its actually rioxarray used in the backend.
        #   however, error is not explicit enough (see https://github.com/pydata/xarray/issues/7831)
        data = xr.open_dataarray(open_file_or_filename(self.filename), engine="rasterio",
                                 chunks={"band": 1,
                                         "y": "auto",
                                         "x": "auto"},
                                 mask_and_scale=False).squeeze()

        # For calibration simplicity convert the fill value to np.nan
        bands_9999 = ["TRAD", "DRAD", "URAD", "ATRAN", "EMIS", "EMSD", "CDIST", "qa_st", "qa_atmos_opacity"]
        if self.collection == "02" and key["name"] in bands_9999:
            # The fill value for several Landsat L2 bands is '-9999'
            data.data = da.where(data.data == -9999, np.float32(np.nan), data.data)
        else:
            # The fill value for Landsat is '0'
            data.data = da.where(data.data == 0, np.float32(np.nan), data.data)

        attrs = data.attrs.copy()
        # Add useful metadata to the attributes.
        attrs["perc_cloud_cover"] = self._mda.cloud_cover
        # Add platform / sensor attributes
        attrs["platform_name"] = self.platform_name
        attrs["sensor"] = self.sensor
        # Apply attrs from YAML
        if "standard_name" in info:
            attrs["standard_name"] = info["standard_name"]
        if "units" in info:
            attrs["units"] = info["units"]

        # Only OLI bands have a saturation flag
        if key["name"] in self.bsat:
            attrs["saturated"] = self.bsat[key["name"]]

        # Rename to Satpy convention
        data = data.rename({"band": "bands"})

        data.attrs.update(attrs)

        # Calibrate if we're using a band rather than a QA or geometry dataset
        if "calibration" in key:
            data = self.calibrate(data, key["calibration"])
        if key["name"] in ANGLIST:
            data.data = data.data * 0.01

        return data

    def calibrate(self, data, calibration):
        """Calibrate the data from counts into the desired units."""
        raise NotImplementedError()

    def get_area_def(self, dsid):
        """Get area definition of the image from the metadata."""
        return self._mda.build_area_def(dsid["name"])


class BaseLandsatL1Reader(BaseLandsatReader):
    """Basic file handler for Landsat L1 files (tif)."""

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
            data.data = data.data * self.calinfo[self.channel][2] + self.calinfo[self.channel][3]
            data.data = data.data.astype(np.float32) * 100
            return data

        if calibration == "brightness_temperature":
            data.data = (self.calinfo[self.channel][3] / np.log((self.calinfo[self.channel][2] / data.data) + 1))
            data.data = data.data.astype(np.float32)
            return data


class BaseLandsatL2Reader(BaseLandsatReader):
    """Basic file handler for Landsat L2 files (tif)."""

    def calibrate(self, data, calibration):
        """Calibrate the data from counts into the desired units."""
        if calibration == "counts":
            return data

        if calibration in ["reflectance", "brightness_temperature"]:
            data.data = data.data * self.calinfo[self.channel][0] + self.calinfo[self.channel][1]
            if calibration == "reflectance":
                data.data = data.data * 100
            data.data = data.data.astype(np.float32)
            return data

        if calibration == "radiance":
            data.data = data.data * 0.001
            data.data = data.data.astype(np.float32)
            return data

        if calibration in ["emissivity", "atmospheric_transmittance"]:
            data.data = data.data * 0.0001
            data.data = data.data.astype(np.float32)
            return data

        if calibration in ["qa_brightness_temperature", "cloud_distance"]:
            data.data = data.data * 0.01
            data.data = data.data.astype(np.float32)
            return data


class OLITIRSCHReader(BaseLandsatL1Reader):
    """File handler for Landsat OLI-TIRS L1 files (tif)."""

    @property
    def spectral_bands(self):
        """Landsat bands."""
        return ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"]

    @property
    def thermal_bands(self):
        """Landsat thermal bands."""
        return ["B10", "B11"]

    @property
    def sensor(self):
        """Sensor name."""
        return "OLI_TIRS"


class OLITIRSL2CHReader(BaseLandsatL2Reader):
    """File handler for Landsat OLI-TIRS L2 files (tif)."""

    @property
    def spectral_bands(self):
        """Landsat bands."""
        return ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]

    @property
    def thermal_bands(self):
        """Landsat thermal bands."""
        return ["B10"]

    @property
    def sensor(self):
        """Sensor name."""
        return "OLI_TIRS"


class ETMCHReader(BaseLandsatL1Reader):
    """File handler for Landsat ETM+ L1 files (tif)."""

    @property
    def spectral_bands(self):
        """Landsat bands."""
        return ["B1", "B2", "B3", "B4", "B5", "B7", "B8"]

    @property
    def thermal_bands(self):
        """Landsat thermal bands."""
        return ["B6_VCID_1", "B6_VCID_2"]

    @property
    def sensor(self):
        """Sensor name."""
        return "ETM+"


class ETML2CHReader(BaseLandsatL2Reader):
    """File handler for Landsat ETM+ L2 files (tif)."""

    @property
    def spectral_bands(self):
        """Landsat bands."""
        return ["B1", "B2", "B3", "B4", "B5", "B7"]

    @property
    def thermal_bands(self):
        """Landsat thermal bands."""
        return ["B6"]

    @property
    def sensor(self):
        """Sensor name."""
        return "ETM+"


class TMCHReader(BaseLandsatL1Reader):
    """File handler for Landsat TM L1 files (tif)."""

    @property
    def spectral_bands(self):
        """Landsat bands."""
        return ["B1", "B2", "B3", "B4", "B5", "B7"]

    @property
    def thermal_bands(self):
        """Landsat thermal bands."""
        return ["B6"]

    @property
    def sensor(self):
        """Sensor name."""
        return "TM"


class TML2CHReader(BaseLandsatL2Reader):
    """File handler for Landsat TM L2 files (tif)."""

    @property
    def spectral_bands(self):
        """Landsat bands."""
        return ["B1", "B2", "B3", "B4", "B5", "B7"]

    @property
    def thermal_bands(self):
        """Landsat thermal bands."""
        return ["B6"]

    @property
    def sensor(self):
        """Sensor name."""
        return "TM"


class MSSCHReader(BaseLandsatL1Reader):
    """File handler for Landsat MSS L1 files (tif)."""

    @property
    def spectral_bands(self):
        """Landsat bands."""
        if self.platform_name in ["Landsat-1", "Landsat-2", "Landsat-3"]:
            return ["B4", "B5", "B6", "B7"]
        elif self.platform_name in ["Landsat-4", "Landsat-5"]:
            return ["B1", "B2", "B3", "B4"]

    @property
    def thermal_bands(self):
        """Landsat thermal bands."""
        return []

    @property
    def sensor(self):
        """Sensor name."""
        return "MSS"

    def available_datasets(self, configured_datasets=None):
        """Set up wavelength to B4 band."""
        # update previously configured datasets
        for is_avail, ds_info in (configured_datasets or []):
            # some other file handler knows how to load this
            # don't override what they've done
            if is_avail is not None:
                yield is_avail, ds_info

            matches = self.file_type_matches(ds_info["file_type"])
            if matches:
                if ds_info.get("name") == "B4":
                    # Modify the dataset's wavelength dynamically
                    new_info = ds_info.copy()
                    if self.platform_name in ["Landsat-1", "Landsat-2", "Landsat-3"]:
                        new_info["wavelength"] = [0.5, 0.55, 0.6]  # Green
                    elif self.platform_name in ["Landsat-4", "Landsat-5"]:
                        new_info["wavelength"] = [0.8, 0.95, 1.1]  # NIR
                    yield True, new_info
                else:
                    yield True, ds_info

            elif is_avail is None:
                # we don't know what to do with this
                # see if another future file handler does
                yield is_avail, ds_info


class BaseLandsatMDReader(BaseFileHandler):
    """Metadata file handler for Landsat files (tif)."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init the reader."""
        super().__init__(filename, filename_info, filetype_info)
        # Check we have landsat data
        if filename_info["platform_type"] != "L":
            raise ValueError("This reader only supports Landsat data")
        self.platform_name = PLATFORMS[filename_info["spacecraft_id"]]
        self._obs_date = filename_info["observation_date"]
        self.root = ET.parse(open_file_or_filename(self.filename))
        self.process_level = filename_info["process_level_correction"]
        import bottleneck  # noqa
        import geotiepoints  # noqa

    def get_cal_params(self, top_key, key_1, key_2):
        """Read the requested calibration parameters."""
        gain_flags = {}
        add_flags = {}

        for elem in self.root.findall(f".//{top_key}/*"):
            if elem.tag.startswith(f"{key_1}_BAND_"):
                band_num = elem.tag.replace(f"{key_1}_BAND_", "")
                if band_num.startswith("ST_"):
                    band_num = band_num.replace("ST_", "")
                else:
                    band_num = "B" + band_num
                gain_flags[band_num] = float(elem.text)

            if elem.tag.startswith(f"{key_2}_BAND_"):
                band_num = elem.tag.replace(f"{key_2}_BAND_", "")
                if band_num.startswith("ST_"):
                    band_num = band_num.replace("ST_", "")
                else:
                    band_num = "B" + band_num
                add_flags[band_num] = float(elem.text)

        return {key: tuple([gain_flags[key], add_flags[key]]) for key in gain_flags}

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

    @property
    def band_saturation(self):
        """Return per-band saturation flag."""
        flags = {}
        for elem in self.root.findall(".//IMAGE_ATTRIBUTES/*"):
            if elem.tag.startswith("SATURATION_BAND_"):
                band_num = elem.tag.replace("SATURATION_BAND_", "B")
                flags[band_num] = (elem.text == "Y")
        return flags

    @property
    def band_calibration(self):
        """Return per-band saturation flag."""
        raise NotImplementedError()

    def earth_sun_distance(self):
        """Return Earth-Sun distance."""
        return float(self.root.find(".//IMAGE_ATTRIBUTES/EARTH_SUN_DISTANCE").text)

    def build_area_def(self, bname):
        """Build area definition from metadata."""
        from pyresample.geometry import AreaDefinition

        # Here we assume that the thermal bands have the same resolution as the reflective bands,
        # with only the panchromatic band (b08) having a different resolution.
        if bname in PAN_BANDLIST and self.platform_name in ["Landsat-7", "Landsat-8", "Landsat-9"]:
            pixoff = float(self.root.find(".//PROJECTION_ATTRIBUTES/GRID_CELL_SIZE_PANCHROMATIC").text) / 2.
            x_size = float(self.root.find(".//PROJECTION_ATTRIBUTES/PANCHROMATIC_SAMPLES").text)
            y_size = float(self.root.find(".//PROJECTION_ATTRIBUTES/PANCHROMATIC_LINES").text)
        else:
            pixoff = float(self.root.find(".//PROJECTION_ATTRIBUTES/GRID_CELL_SIZE_REFLECTIVE").text) / 2.
            x_size = float(self.root.find(".//PROJECTION_ATTRIBUTES/REFLECTIVE_SAMPLES").text)
            y_size = float(self.root.find(".//PROJECTION_ATTRIBUTES/REFLECTIVE_LINES").text)

        # Get remaining geoinfo from file
        datum = self.root.find(".//PROJECTION_ATTRIBUTES/DATUM").text

        # Reading utm zone or get specific crs for arctic and antarctic
        if self.root.find(".//PROJECTION_ATTRIBUTES/UTM_ZONE") is not None:
            utm_zone = self.root.find(".//PROJECTION_ATTRIBUTES/UTM_ZONE").text
            pcs_id = f"{datum} / UTM zone {utm_zone}N"
            proj_code = f"EPSG:326{utm_zone.zfill(2)}"
        else:
            lat_ts = self.root.find(".//PROJECTION_ATTRIBUTES/TRUE_SCALE_LAT").text
            if lat_ts == "-71.00000":
                # Antarctic
                proj_code = "EPSG:3031"
            if lat_ts == "71.00000":
                # Arctic
                proj_code = "EPSG:3995"
            pcs_id = f"{datum} / EPSG: {proj_code[5:]}N"

        # We need to subtract / add half a pixel from the corner to get the correct extent (pixel centers)
        ext_p1 = float(self.root.find(".//PROJECTION_ATTRIBUTES/CORNER_UL_PROJECTION_X_PRODUCT").text) - pixoff
        ext_p2 = float(self.root.find(".//PROJECTION_ATTRIBUTES/CORNER_LR_PROJECTION_Y_PRODUCT").text) - pixoff
        ext_p3 = float(self.root.find(".//PROJECTION_ATTRIBUTES/CORNER_LR_PROJECTION_X_PRODUCT").text) + pixoff
        ext_p4 = float(self.root.find(".//PROJECTION_ATTRIBUTES/CORNER_UL_PROJECTION_Y_PRODUCT").text) + pixoff

        # Create area definition
        area_extent = (ext_p1, ext_p2, ext_p3, ext_p4)

        # Return the area extent
        return AreaDefinition(f"EPSG_{proj_code[5:]}", pcs_id, pcs_id, proj_code, x_size, y_size, area_extent)


class LandsatL1MDReader(BaseLandsatMDReader):
    """Metadata file handler for Landsat L1 files (tif)."""

    @property
    def band_calibration(self):
        """Return per-band calibration parameters."""
        radcal = self.get_cal_params("LEVEL1_RADIOMETRIC_RESCALING", "RADIANCE_MULT", "RADIANCE_ADD")
        viscal = self.get_cal_params("LEVEL1_RADIOMETRIC_RESCALING", "REFLECTANCE_MULT", "REFLECTANCE_ADD")
        tircal = self.get_cal_params("LEVEL1_THERMAL_CONSTANTS", "K1_CONSTANT", "K2_CONSTANT")
        topcal = viscal | tircal
        return {key: tuple([*radcal[key], *topcal[key]]) for key in radcal}


class LandsatL2MDReader(BaseLandsatMDReader):
    """Metadata file handler for Landsat L2 files (tif)."""

    @property
    def band_calibration(self):
        """Return per-band calibration parameters."""
        viscal = self.get_cal_params("LEVEL2_SURFACE_REFLECTANCE_PARAMETERS", "REFLECTANCE_MULT", "REFLECTANCE_ADD")
        tircal = self.get_cal_params("LEVEL2_SURFACE_TEMPERATURE_PARAMETERS", "TEMPERATURE_MULT", "TEMPERATURE_ADD")
        return viscal | tircal
