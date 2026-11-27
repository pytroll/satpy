#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
#
# This file is part of satpy. Written by Meteo France in august 2024.
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

"""Interface to GEOSTATIONNARY L1B NETCDF data from ICARE (Lille).

Introduction
------------

The ``geos_netcdficare`` reader reads some geostationnary netcdf build by
Meteo France and stored at Icare.

The brightness temperature and albedo are calibrated.

That has been stored by the ICARE Data and Services Center
Data can be accessed via: http://www.icare.univ-lille1.fr

This reader concerns the following netcdfs :

. msg with a longitude near 0° :
Mmultic3kmNC4_msg03_20231113_111500.nc
Mmultic1kmNC4_msg03_20231113_111500.nc

. Msg rapid scan with a longitude near 9.5° :
Mrsmultic3kmNC4_msg03_20231113_111500.nc
Mrsmultic1kmNC4_msg03_20231113_111500.nc

. Msg with a longitude near 42° :
Imultic3kmNC4_msg03_20231113_111500.nc
Imultic1kmNC4_msg03_20231113_111500.nc

. Himawari :
Jmultic2kmNC4_hima09_20231113_111500.nc
Jmultic1kmNC4_hima09_20231113_111500.nc
Jmultic500mNC4_hima09_20231113_111500.nc

. Goesr near -137° :
Wmultic2kmNC4_goes16_202406281000.nc.
The better resolution are not built at Lannion, only at Tahiti.

. Goesr in -75° :
Emultic2kmNC4_goes16_202406281000.nc
Emultic1kmNC4_goes16_202406281000.nc
Emultic500mNC4_goes16_202406281000.nc

. Mtg :
Mmultic2km_mtgi1_20240104_090000.nc
Mmultic1km_mtgi1_20240104_090000.nc
Mmultic500m_mtgi1_20240104_090000.nc


Example:
--------
Here is an example how to read the data in satpy:

    from satpy import Scene
    import glob

    filenames = glob.glob('data/*2019-03-01T12-00-00*.hdf')
    scn = Scene(filenames = filenames, reader = 'hima_netcdficare')
    scn.load(['true_color'])    # scn.load(['VIS006'])

    my_area = AreaDefinition(
        'my_area', 'zone', 'my_area',
        '+proj=latlong +lon_0=0 +a=6378169 +b=6356583 +h=35785831 +x_0=0
            +y_0=0 +pm=0',
        8500, 4000,
        [-180., -80., 180., 80],
        nprocs=16)

    natscn = scn.resample(my_area, resampler='nearest')
    natscn.save_dataset(composite_name, filename = filename_image_out)

EXECUTION TIME :
    50 seconds for a 2 kms goesr airmass rgb disk.
DATE OF CREATION :
    2024 16th october.
LAST VERSIONS :

AUTHOR :
    Meteo France.

"""

import datetime as dt

import numpy as np

from satpy.readers._geos_area import get_area_definition, get_area_extent

import xarray as xr

from satpy.readers.file_handlers import BaseFileHandler

import logging
logger = logging.getLogger('netcdficare')

# Planck :
C1 = 1.1910427e-5
C2 = 1.4387752


class NETCDF_ICARE(BaseFileHandler) :
    # Cf readers/file_handlers.py.

    def __init__(self, filename, filename_info, filetype_info) :
        """Init the file handler."""

        super().__init__(filename, filename_info, filetype_info)

        self.nc = xr.open_dataset(
            self.filename, decode_cf=True, mask_and_scale=False,
            chunks={"xc": "auto", "yc": "auto"})
        self.metadata = {}

        self.metadata["start_time"] = self.get_endOrStartTime(
            "time_coverage_start")
        self.metadata["end_time"] = self.get_endOrStartTime(
            "time_coverage_end")

        # message = "Reading: " + filename
        # message += " start: " + format(self.start_time)
        # message += " end: " + format(self.end_time)
        # logger.info(message)

        self.netcdfCommonAttributReading()
        # __init__()

    def netcdfCommonAttributReading(self) :
        self.sensor = self.sensor_name()
        # seviri

        self.platform = self.platform_name()
        # Meteosat-10

        self.res()
        # Resolution : 3000.4 m

        self.actualLongitude = self.satlon()
        self.projectionLongitude = self.projlon()

        self.zone = self.nc.attrs["Area_of_acquisition"]
        # globe.

    def sensor_name(self) :
        """Get the sensor name seviri, fci, abi, ahi.
        """
        variable = self.nc["satellite"]
        platform = variable.attrs["id"]        # msg1, msg01, MSG1...
        platform = platform[:3]            # msg, MSG
        platform = platform.lower()        # msg

        pdict = {}
        pdict["msg"] = "seviri"
        pdict["mtg"] = "fci"
        pdict["goe"] = "abi"
        pdict["him"] = "ahi"

        if platform in pdict :
            sensor = pdict[platform]
        else :
            message = "Unsupported satellite platform : "
            message += self.platform
            raise NotImplementedError(message)
        return sensor

    def platform_name(self) :
        # Icare and météo france use non-standard platform names.
        # Change is needed for pyspectral :
        # pyspectral/rsr_seviri_Meteosat-10.h5 in the call
        # Calculator(platform_name, sensor, name).

        variable = self.nc["satellite"]
        platform = variable.attrs["id"]    # msg1, msg01, MSG1...

        pdict = {}
        pdict["msg1"] = "Meteosat-08"
        pdict["msg01"] = "Meteosat-08"
        pdict["MSG1"] = "Meteosat-08"
        pdict["msg2"] = "Meteosat-09"
        pdict["msg02"] = "Meteosat-09"
        pdict["MSG2"] = "Meteosat-09"
        pdict["msg3"] = "Meteosat-10"
        pdict["msg03"] = "Meteosat-10"
        pdict["MSG3"] = "Meteosat-10"
        pdict["msg4"] = "Meteosat-11"
        pdict["msg04"] = "Meteosat-11"
        pdict["MSG4"] = "Meteosat-11"
        pdict["mtgi1"] = "Meteosat-12"
        pdict["mtg1"] = "Meteosat-12"
        pdict["MTG01"] = "Meteosat-12"
        pdict["goes16"] = "GOES-16"
        pdict["goes17"] = "GOES-17"
        pdict["goes18"] = "GOES-18"
        pdict["goes19"] = "GOES-19"
        pdict["hima08"] = "Himawari-8"
        pdict["hima09"] = "Himawari-9"

        if platform in pdict :
            platform = pdict[platform]
        else :
            message = "Unsupported satellite platform : " + platform
            raise NotImplementedError(message)
        return platform
        # platform_name()

    def satlon(self) :
        """Get the satellite longitude."""
        variable = self.nc["satellite"]
        actualLongitude = variable.attrs["lon"]
        return actualLongitude

    def projlon(self):
        """Get the projection longitude."""
        variable = self.nc["geos"]
        projectionLongitude = variable.attrs["longitude_of_projection_origin"]
        return projectionLongitude

    @property
    def projection(self):
        """Get the projection."""
        return "geos"

    def res(self) :
        """Get the resolution.
        The resolution can be read in the attribute geotransform
        of the following variables :
        GeosCoordinateSystem500m, GeosCoordinateSystem_h,
        GeosCoordinateSystem1km, GeosCoordinateSystem2km,
        GeosCoordinateSystem.
        cfac, lfac, coff, loff can be read in the variables ImageNavigationxxx.
        """

        if "GeosCoordinateSystem500m" in self.nc :
            # Mtg, himawari, goesr.
            variable = self.nc["GeosCoordinateSystem500m"]
            Xvariable = self.nc["X500m"]
            Yvariable = self.nc["Y500m"]
            navigationString = "ImageNavigation500m"

        elif "GeosCoordinateSystem_h" in self.nc :
            # Hrv from msg.
            variable = self.nc["GeosCoordinateSystem_h"]
            Xvariable = self.nc["X_h"]
            Yvariable = self.nc["Y_h"]
            navigationString = "ImageNavigation_h"

        elif "GeosCoordinateSystem1km" in self.nc :
            # Mtg, himawari, goesr.
            variable = self.nc["GeosCoordinateSystem1km"]
            Xvariable = self.nc["X1km"]
            Yvariable = self.nc["Y1km"]
            navigationString = "ImageNavigation1km"

        elif "GeosCoordinateSystem2km" in self.nc :
            # Mtg, himawari, goesr.
            variable = self.nc["GeosCoordinateSystem2km"]
            Xvariable = self.nc["X2km"]
            Yvariable = self.nc["Y2km"]
            navigationString = "ImageNavigation2km"

        elif "GeosCoordinateSystem" in self.nc :
            # Msg in 3 kms.
            variable = self.nc["GeosCoordinateSystem"]
            Xvariable = self.nc["X"]
            Yvariable = self.nc["Y"]
            navigationString = "ImageNavigation"

        else :
            message = "Variables GeosCoordinateSystemXX not founded."
            raise NotImplementedError(message)

        geotransform = variable.attrs["GeoTransform"]
        # geotransform =  -5570254, 3000.40604, 0, 5570254, 0, -3000.40604

        chunksGeotransform = geotransform.split(", ")
        self.resolution = float(chunksGeotransform[1])
        # 3000.40604

        self.X = Xvariable[:]
        self.nbpix = self.X.shape[0]
        self.Y = Yvariable[:]
        self.nblig = self.Y.shape[0]

        variable = self.nc[navigationString]
        self.cfac = float(variable.attrs["CFAC"])
        self.lfac = float(variable.attrs["LFAC"])
        self.coff = float(variable.attrs["COFF"])
        self.loff = float(variable.attrs["LOFF"])
        # res()

    def get_endOrStartTime(self, AttributeName) :
        """Get the end or the start time. Global attribute of the netcdf.
        AttributName : "time_coverage_start", "time_coverage_end"
        """
        attr = self.nc.attrs[AttributeName]
        # YYYY-MM-DDTHH:MM:SSZNNN or YYYY-MM-DDTHH:MM:SSZ
        # In some versions milliseconds are present, sometimes not.
        lengthString = len(attr)
        if lengthString == 22 :
            # Goesr : 2024-06-28T10:00:21.1Z
            stacq = dt.datetime.strptime(attr, "%Y-%m-%dT%H:%M:%S.%fZ")
        elif lengthString == 20 :
            # Mtg.
            stacq = dt.datetime.strptime(attr, "%Y-%m-%dT%H:%M:%SZ")
        else :
            # Msg, hima.
            stacq = dt.datetime.strptime(attr, "%Y-%m-%dT%H:%M:%SZ%f")
        return stacq
        # get_endOrStartTime()

    @property
    def start_time(self) :
        return(self.metadata["start_time"])

    @property
    def end_time(self) :
        return(self.metadata["end_time"])

    @property
    def alt(self) :
        """Get the altitude."""
        variable = self.nc["satellite"]
        altitude = variable.attrs["dst"]  # 36000000 meters.
        altitude += 6378169.        # equatorial radius of the earth.
        return altitude

    def prepare_metadata(self, variable) :
        """Get the metadata for a channel variable.
        Add the global attributes."""
        mda = {}

        attributs = variable.attrs
        for name in attributs :
            mda.update({name: attributs.get(name)})

        mda.update({
            "start_time": self.start_time,
            "end_time": self.end_time,
            "platform_name": self.platform,
            "sensor": self.sensor,
            "zone": self.zone,
            "projection_altitude": self.alt,
            "cfac": self.cfac,
            "lfac": self.lfac,
            "coff": self.coff,
            "loff": self.loff,
            "resolution": self.resolution,
            "satellite_actual_longitude": self.actualLongitude,
            "projection_longitude": self.projectionLongitude,
            "projection_type": self.projection
            })

        mda.update(self.orbital_param())
        return mda
        # prepare_metadata().

    def buildChannelCorrespondanceName(self) :
        pdict = {}

        # For mtg.
        # vis_04 is the name in satpy.
        # VIS004 is the icare/meteofrance netcdf name.
        pdict["vis_04"] = "VIS004"
        pdict["vis_05"] = "VIS005"
        pdict["vis_06"] = "VIS006"
        pdict["vis_08"] = "VIS008"
        pdict["vis_09"] = "VIS009"
        pdict["nir_13"] = "IR_013"
        pdict["nir_16"] = "IR_016"
        pdict["nir_22"] = "IR_022"
        pdict["ir_38"] = "IR_038"
        pdict["wv_63"] = "WV_063"
        pdict["wv_73"] = "WV_073"
        pdict["ir_87"] = "IR_087"
        pdict["ir_97"] = "IR_097"
        pdict["ir_105"] = "IR_105"
        pdict["ir_123"] = "IR_123"
        pdict["ir_133"] = "IR_133"

        # For msg, the satpy and icare channel names are the same.
        pdict["VIS006"] = "VIS006"
        pdict["VIS008"] = "VIS008"
        pdict["HRV"] = "HRV"
        pdict["IR_016"] = "IR_016"
        pdict["IR_039"] = "IR_039"
        pdict["WV_062"] = "WV_062"
        pdict["WV_073"] = "WV_073"
        pdict["IR_087"] = "IR_087"
        pdict["IR_097"] = "IR_097"
        pdict["IR_108"] = "IR_108"
        pdict["IR_120"] = "IR_120"
        pdict["IR_134"] = "IR_134"

        # For the goesr satellites :
        pdict["C01"] = "VIS_004"
        pdict["C02"] = "VIS_006"
        pdict["C03"] = "VIS_008"
        pdict["C04"] = "VIS_014"
        pdict["C05"] = "VIS_016"
        pdict["C06"] = "VIS_022"
        pdict["C07"] = "IR_039"
        pdict["C08"] = "IR_062"
        pdict["C09"] = "IR_069"
        pdict["C10"] = "IR_073"
        pdict["C11"] = "IR_085"
        pdict["C12"] = "IR_096"
        pdict["C13"] = "IR_103"
        pdict["C14"] = "IR_114"
        pdict["C15"] = "IR_123"
        pdict["C16"] = "IR_133"

        # For himawari.
        # BO1 : name in satpy. VIS004 : name in icare/meteofrance netcdf.
        pdict["B01"] = "VIS004"
        pdict["B02"] = "VIS005"
        pdict["B03"] = "VIS006"
        pdict["B04"] = "VIS008"
        pdict["B05"] = "IR_016"
        pdict["B06"] = "IR_022"
        pdict["B07"] = "IR_038"
        pdict["B08"] = "WV_062"
        pdict["B09"] = "WV_069"
        pdict["B10"] = "WV_073"
        pdict["B11"] = "IR_085"
        pdict["B12"] = "IR_096"
        pdict["B13"] = "IR_104"
        pdict["B14"] = "IR_112"
        pdict["B15"] = "IR_123"
        pdict["B16"] = "IR_132"
        return pdict
        # buildChannelCorrespondanceName()

    def _get_dsname(self, ds_id) :
        """Return the correct dataset name based on requested band.
        ds_id = DataID(name='vis_08',
            wavelength=WavelengthRange(...),
            resolution=2000, calibration=<calibration.reflectance>,
            modifiers=())
        """
        pdict = self.buildChannelCorrespondanceName()

        satpyName = ds_id["name"]
        if satpyName in pdict :
            icareName = pdict[satpyName]
        else :
            message = "Soft not adaptated for this channel : ds_id = "
            message += satpyName
            raise NotImplementedError(message)

        return icareName
        # _get_dsname()

    def channelAttributs(self, ds_get_name) :
        if ds_get_name not in self.nc :
            message = "Channel " + ds_get_name + "not founded "
            message += "in the netcdf."
            raise NotImplementedError(message)

        self.mda = {}
        self.scale_factor = {}
        self.offset = {}
        self.alpha = {}
        self.beta = {}
        self.nuc = {}
        self.bandfactor = {}
        self.backtocountVariable = {}

        variable = self.nc[ds_get_name]
        attributs = variable.attrs

        self.scale_factor[ds_get_name] = attributs["scale_factor"]
        self.offset[ds_get_name] = attributs["add_offset"]

        if "nuc" in attributs :
            # Brightness temperature.
            self.alpha[ds_get_name] = attributs["alpha"]
            self.beta[ds_get_name] = attributs["beta"]
            self.nuc[ds_get_name] = attributs["nuc"]

            backtocountName = "Temp_to_Native_count_" + ds_get_name

        elif "bandfactor" in attributs :
            # Albedo.
            self.bandfactor[ds_get_name] = attributs["bandfactor"]
            backtocountName = "Albedo_to_Native_count_" + ds_get_name

        else :
            message = "Nuc or bandfactor not founded in the attributs"
            message += " of " + ds_get_name
            raise NotImplementedError(message)

        self.backtocountVariable[ds_get_name] = self.nc[backtocountName]
        # (65536). Correspondence from 0 to 65535 towards
        # the original spatial agency counts.

        self.mda[ds_get_name] = self.prepare_metadata(variable)
        # channelAttributs(ds_get_name)

    def comebacktoNativeData(self, ds_get_name) :
        """ Come back to the original counts of the hrit.
        ds_get_name : meteo france name of a channel : IR_108. """

        variable = self.nc[ds_get_name]
        # Variable is between -9000 to 4000 (temperature)
        # or between 0 to 10000 (albedo).

        offset = self.offset[ds_get_name]        # 0 or 273.15
        variable += 32768                # 0 to 65535

        if offset == 0. :
            # Albedo.
            name = "Albedo_to_Native_count_" + ds_get_name
        else :
            name = "Temp_to_Native_count_" + ds_get_name
            """ Temp_to_Native_count_IR_062 """

        backtocountVariable = self.nc[name]        # (65536).
        # Correspondence from 0 to 65535
        # towards the original spatial agency counts.

        arrayTableConversion = xr.DataArray.to_numpy(backtocountVariable)

        tableContents = arrayTableConversion[variable[:]]
        """ Come back to the original counts of the hrit.
        tableau : 0 to 4095 if native datas coded with 12 bits. """

        variable[:] = tableContents
        return(variable)
        # comebacktoNativeData(self, ds_get_name)

    def comebacktoRadiance(self, ds_get_name) :
        """ Come back to the radiance.
        ds_get_name : meteo france name of a channel : IR_108. """

        variable = self.nc[ds_get_name]
        # Variable is between -9000 to 4000 (temperature)
        # or between 0 to 10000 (albedo).

        scale_factor = self.scale_factor[ds_get_name]        # 0.01
        offset = self.offset[ds_get_name]        # 0 or 273.15

        if offset == 0. :
            # Visible channel.
            bandfactor = self.bandfactor[ds_get_name]

            # Variable is an albedo from 0 to 10000.
            variable = variable * scale_factor / 100. * bandfactor
            # => variable is a reflectance between 0 and 1.
            # radiance in mWm-2sr-1(cm-1)-1
        else :
            # Brightness temperature.
            nuc = self.nuc[ds_get_name]
            alpha = self.alpha[ds_get_name]
            beta = self.beta[ds_get_name]

            variable = variable * scale_factor + offset
            # variable becomes Kelvin.

            variable = variable * alpha + beta
            resul1 = C1 * np.power(nuc, 3.)
            resul2 = C2 * nuc
            val2 = np.exp(resul2 / variable) - 1.
            variable = resul1 / val2
            # Radiance in mWm-2sr-1(cm-1)-1
        return(variable)
        # comebacktoRadiance(self, ds_get_name)

    def get_dataset(self, ds_id, ds_info) :
        """Get the dataset.
        ds_id["calibration"] = key["calibration"] =
            ["brightness_temperature", "reflectance", "radiance", "counts"]
        """
        ds_get_name = self._get_dsname(ds_id)
        # ds_get_name is the meteo France Icare name of the channel : IR_096.

        self.channelAttributs(ds_get_name)

        mda = self.mda[ds_get_name]
        mda.update(ds_info)

        calibration = ds_id["calibration"]

        if calibration == "counts" :
            # Come back to the original counts of the hrit...
            variable = self.comebacktoNativeData(ds_get_name)

        elif calibration == "radiance" :
            # Come back to the radiance.
            variable = self.comebacktoRadiance(ds_get_name)

        elif calibration == "brightness_temperature" :
            variable = self.nc[ds_get_name]
            # WV_062 calibration.brightness_temperature, from -9000 to 4000
            scale_factor = self.scale_factor[ds_get_name]
            offset = self.offset[ds_get_name]
            if offset != 273.15 :
                message = "Soft not intended for a reflectance "
                message += "with a wave length more than 3 microns. "
                message += ds_get_name + " offset = " + str(offset)
                raise NotImplementedError(message)

            variable = variable * scale_factor + offset
            # variable becomes Kelvin.

        elif calibration == "reflectance" :
            variable = self.nc[ds_get_name]
            # VIS006 calibration.reflectance, from 0 to 10000
            scale_factor = self.scale_factor[ds_get_name]
            offset = self.offset[ds_get_name]
            if offset != 0. :
                message = "Soft not intended "
                message += "for a brightness temperature "
                message += "with a wave length less than 3 microns. "
                message += ds_get_name + " offset = " + str(offset)
                raise NotImplementedError(message)

            variable = variable * scale_factor
            # variable becomes an albedo between 0 and 100.

        else :
            message = "Calibration mode not expected : " + calibration
            raise NotImplementedError(message)

        variable = variable.rename(
            {variable.dims[0] : "y", variable.dims[1] : "x"})
        variable.attrs.update(mda)
        return variable
        # get_dataset()

    def orbital_param(self) :
        orb_param_dict = {
            "orbital_parameters": {
                "satellite_actual_longitude": self.actualLongitude,
                "satellite_actual_latitude": 0.,
                "satellite_actual_altitude": self.alt,
                "satellite_nominal_longitude": self.projectionLongitude,
                "satellite_nominal_latitude": 0,
                "satellite_nominal_altitude": self.alt,
                "projection_longitude": self.projectionLongitude,
                "projection_latitude": 0.,
                "projection_altitude": self.alt,
                }
            }
        return orb_param_dict

    def channelType(self, pdict, pdictResoAdesc, pdictResoPid, satellite) :
        strNbpix = str(self.nbpix)
        if strNbpix in pdictResoAdesc :
            pdict["a_desc"] = pdictResoAdesc[strNbpix]
            pdict["p_id"] = pdictResoPid[strNbpix]
        else :
            message = "Resolution " + str(self.nbpix)
            message += " not expected for " + satellite
            raise NotImplementedError(message)

        return(pdict)

    def get_area_def(self, ds_id) :
        """Get the area def."""

        pdict = {}
        pdict["cfac"] = np.int32(self.cfac)
        pdict["lfac"] = np.int32(self.lfac)
        pdict["coff"] = np.float32(self.coff)
        pdict["loff"] = np.float32(self.loff)

        pdict["a"] = 6378169
        pdict["b"] = 6356583.8
        pdict["h"] = self.alt - pdict["a"]
        # 36000000 mètres.
        pdict["ssp_lon"] = self.projectionLongitude
        pdict["ncols"] = self.nblig
        pdict["nlines"] = self.nbpix
        pdict["sweep"] = "y"

        # Force scandir to SEVIRI default, not known from file
        pdict["scandir"] = "S2N"
        pdict["a_name"] = "geosmsg"

        pdictResoAdesc = {}
        pdictResoPid = {}

        if self.sensor == "seviri" :
            # msg.
            pdict["scandir"] = "N2S"
            pdict["a_name"] = "geosmsg"

            pdictResoAdesc["3712"] = "MSG/SEVIRI low resolution channel area"
            pdictResoPid["3712"] = "msg_lowres"
            pdictResoAdesc["11136"] = "MSG/SEVIRI HRV channel area"
            pdictResoPid["11136"] = "msg_hires"

            pdict = self.channelType(
                pdict, pdictResoAdesc, pdictResoPid, "msg")

        elif self.sensor == "fci" :
            # mtg.
            pdict["scandir"] = "N2S"
            pdict["a_name"] = "geosmtg"

            pdictResoAdesc["5568"] = "MTG 2km channel area"
            pdictResoPid["5568"] = "mtg_lowres"
            pdictResoAdesc["11136"] = "MTG 1km channel area"
            pdictResoPid["11136"] = "mtg_midres"
            pdictResoAdesc["22272"] = "MTG 500m channel area"
            pdictResoPid["22272"] = "mtg_hires"

            pdict = self.channelType(
                pdict, pdictResoAdesc, pdictResoPid, "mtg")

        elif self.sensor == "ahi" :
            # Himawari.
            pdict["scandir"] = "N2S"
            pdict["a_name"] = "geoshima"

            pdictResoAdesc["5500"] = "HIMA 2km channel area"
            pdictResoPid["5500"] = "hima_lowres"
            pdictResoAdesc["11000"] = "HIMA 1km channel area"
            pdictResoPid["11000"] = "hima_midres"
            pdictResoAdesc["22000"] = "HIMA 500m channel area"
            pdictResoPid["22000"] = "hima_hires"

            pdict = self.channelType(
                pdict, pdictResoAdesc, pdictResoPid, "hima")

        elif self.sensor == "abi" :
            # Goesr.
            pdict["scandir"] = "N2S"
            pdict["a_name"] = "geosgoesr"
            pdict["sweep"] = "x"

            pdictResoAdesc["5424"] = "GOESR 2km channel area"
            pdictResoPid["5424"] = "goesr_lowres"
            pdictResoAdesc["10848"] = "GOESR 1km channel area"
            pdictResoPid["10848"] = "goesr_midres"
            pdictResoAdesc["21696"] = "GOESR 500m channel area"
            pdictResoPid["21696"] = "goesr_hires"

            pdict = self.channelType(
                pdict, pdictResoAdesc, pdictResoPid, "goesr")

        else :
            message = "Sensor " + self.sensor + " not expected."
            raise NotImplementedError(message)

        aex = get_area_extent(pdict)
        area = get_area_definition(pdict, aex)

        return area
        # get_area_def()
