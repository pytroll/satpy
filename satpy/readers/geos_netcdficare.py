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

The brightness tempera ture and albedo are calibrated.

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
	scn.load(['true_color'])	# scn.load(['VIS006'])

	my_area = AreaDefinition(
		'my_area', 'zone', 'my_area',
		'+proj=latlong +lon_0=0 +a=6378169 +b=6356583 +h=35785831 +x_0=0
			+y_0=0 +pm=0',
		8500, 4000,
		[-180., -80., 180., 80],
		nprocs=16)

	natscn = scn.resample(my_area, resampler='nearest')
	natscn.save_dataset(composite_name, filename = filename_image_out)

"""

import datetime as dt

import numpy as np

from satpy.readers._geos_area import get_area_definition, get_area_extent

# from netCDF4 import Dataset
# from xarray import DataArray
import xarray as xr

from satpy.readers.file_handlers import BaseFileHandler

# from satpy._compat import cached_property
# from satpy.utils import get_dask_chunk_size_in_bytes
import dask
from satpy.readers import open_file_or_filename
# import math


class NETCDF_ICARE(BaseFileHandler) :
	# Cf readers/file_handlers.py.

	def __init__(self, filename, filename_info, filetype_info) :
		"""Init the file handler."""

		super().__init__(filename, filename_info, filetype_info)

		# chunk_bytes = self._chunk_bytes_for_resolution()
		chunk_bytes = '128 MiB'
		# chunk_bytes = '64 MiB'

		# with dask.config.set(**{'array.slicing.split_large_chunks': True}) :
		with dask.config.set({"array.chunk-size": chunk_bytes}) :
			f_obj = open_file_or_filename(self.filename)
			self.nc = xr.open_dataset(
				f_obj, decode_cf=True, mask_and_scale=False,
				chunks={"xc": "auto", "yc": "auto"})

		self.metadata = {}

		self.metadata["start_time"] = self.get_endOrStartTime(
			"time_coverage_start")
		self.metadata["end_time"] = self.get_endOrStartTime(
			"time_coverage_end")

		print(
			"Reading: {}".format(filename),
			" start: {} ".format(self.start_time),
			" end: {}".format(self.end_time))
		self._cache = {}

		self.sensor_name()
		self.res()
		self.longitudeReelle = self.satlon()
		self.longitudedeprojection = self.projlon()

		self.zone = self.nc.attrs["Area_of_acquisition"]
		# globe, europe.

		# Reading the needed attributes.
		self.initialisation_dataset()
		# __init__()

	def sensor_name(self) :
		"""Get the sensor name.
		The sensor and platform names are stored together, eg: MSG1/SEVIRI
		"""
		variable = self.nc["satellite"]
		self.plateforme = variable.attrs["id"]

		if "goes" in self.plateforme :
			self.sensor = "abi"
		elif ("msg" in self.plateforme) or ("MSG" in self.plateforme) :
			self.sensor = "seviri"
		elif ("mtg" in self.plateforme) or ("MTG" in self.plateforme) :
			self.sensor = "fci"
		elif "hima" in self.plateforme :
			self.sensor = "ahi"
		else :
			raise NameError("Unsupported satellite platform : " + self.plateforme)
			
		# Icare and météo france use non-standard platform names.
		# Change is needed for pyspectral :
		# pyspectral/rsr_seviri_Meteosat-10.h5 in the call
		# Calculator(platform_name, sensor, name).
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

		if self.plateforme in pdict :
			self.plateforme = pdict[self.plateforme]
		else :
			print(
				"Unsupported satellite platform : " +
				self.plateforme)
			exit(1)

		# sensor_name()

	def satlon(self) :
		"""Get the satellite longitude."""
		variable = self.nc["satellite"]
		longitudeReelle = variable.attrs["lon"]
		return longitudeReelle

	def projlon(self):
		"""Get the projection longitude."""
		variable = self.nc["geos"]
		longitudedeprojection = variable.attrs["longitude_of_projection_origin"]
		return longitudedeprojection

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
			variableX = self.nc["X500m"]
			variableY = self.nc["Y500m"]
			chaineNavigation = "ImageNavigation500m"

		elif "GeosCoordinateSystem_h" in self.nc :
			# Hrv from msg.
			variable = self.nc["GeosCoordinateSystem_h"]
			variableX = self.nc["X_h"]
			variableY = self.nc["Y_h"]
			chaineNavigation = "ImageNavigation_h"

		elif "GeosCoordinateSystem1km" in self.nc :
			# Mtg, himawari, goesr.
			variable = self.nc["GeosCoordinateSystem1km"]
			variableX = self.nc["X1km"]
			variableY = self.nc["Y1km"]
			chaineNavigation = "ImageNavigation1km"

		elif "GeosCoordinateSystem2km" in self.nc :
			# Mtg, himawari, goesr.
			variable = self.nc["GeosCoordinateSystem2km"]
			variableX = self.nc["X2km"]
			variableY = self.nc["Y2km"]
			chaineNavigation = "ImageNavigation2km"

		elif "GeosCoordinateSystem" in self.nc :
			# Msg in 3 kms.
			variable = self.nc["GeosCoordinateSystem"]
			variableX = self.nc["X"]
			variableY = self.nc["Y"]
			chaineNavigation = "ImageNavigation"

		else :
			print("Variables GeosCoordinateSystemXX not founded.")
			exit(1)

		geotransform = variable.attrs["GeoTransform"]

		# print("geotransform = ", geotransform)
		# geotransform =  -5570254, 3000.40604, 0, 5570254, 0, -3000.40604
		morceaux = geotransform.split(", ")
		self.resolution = float(morceaux[1])
		# print("resolution = ", self.resolution) # 3000.40604

		self.X = variableX[:]
		self.nbpix = self.X.shape[0]
		self.Y = variableY[:]
		self.nblig = self.Y.shape[0]

		variable = self.nc[chaineNavigation]
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
		longueurchaine = len(attr)
		if longueurchaine == 22 :
			# Goesr : 2024-06-28T10:00:21.1Z
			stacq = dt.datetime.strptime(attr, "%Y-%m-%dT%H:%M:%S.%fZ")
		elif longueurchaine == 20 :
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
		altitude += 6378169.		# equatorial radius of the earth.
		return altitude

	def preparer_metadata(self, variable) :
		"""Get the metadata."""
		mda = {}

		attributs = variable.attrs
		for name in attributs :
			mda.update({name: attributs.get(name)})

		mda.update({
			"start_time": self.start_time,
			"end_time": self.end_time,
			"platform_name": self.plateforme,
			"sensor": self.sensor,
			"zone": self.zone,
			"projection_altitude": self.alt,
			"cfac": self.cfac,
			"lfac": self.lfac,
			"coff": self.coff,
			"loff": self.loff,
			"resolution": self.resolution,
			"satellite_actual_longitude": self.longitudeReelle,
			"projection_longitude": self.longitudedeprojection,
			"projection_type": self.projection
			})

		# Placer ici les paramètres orbitaux, une seule fois ?
		mda.update(self.orbital_param())

		return mda
		# preparer_metadata().

	def _get_dsname(self, ds_id) :
		"""Return the correct dataset name based on requested band.
		ds_id = DataID(name='vis_08',
			wavelength=WavelengthRange(...),
			resolution=2000, calibration=<calibration.reflectance>,
			modifiers=())
		"""
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

		satpyName = ds_id["name"]
		if satpyName in pdict :
			icareName = pdict[satpyName]
		else :
			print(
				"Soft not adaptated for this channel : ds_id = ", 
				satpyName)
			exit(1)

		return icareName
		# _get_dsname()

	def initialisation_dataset(self) :
		listeToutesVariables = {
			"VIS004", "VIS_004", "VIS005", "VIS_005", "VIS006",
			"VIS_006", "VIS006", "VIS_008", "VIS008", "VIS_009",
			"VIS009", "IR_013", "IR_016", "IR_022", "IR_038",
			"IR_039", "IR_062", "WV_062", "WV_063", "IR_069",
			"WV_069", "IR_073", "WV_073", "IR_085", "IR_087",
			"IR_096", "IR_097", "IR_103", "IR_104", "IR_105",
			"IR_108", "IR_112", "IR_120", "IR_123", "IR_132",
			"IR_133", "IR_134", "HRV"
			}

		self.mda = {}
		self.scale_factor = {}
		self.offset = {}
		self.alpha = {}
		self.beta = {}
		self.nuc = {}
		self.bandfactor = {}
		self.variableComptes = {}

		# Loop over the all possible ds_get_name of every satellites :
		for ds_get_name in listeToutesVariables :
			if ds_get_name in self.nc :

				variable = self.nc[ds_get_name]
				attributs = variable.attrs

				self.scale_factor[ds_get_name] = attributs["scale_factor"]
				self.offset[ds_get_name] = attributs["add_offset"]

				if "nuc" in attributs :
					# Brightness temperature.
					self.alpha[ds_get_name]	= attributs["alpha"]
					self.beta[ds_get_name]	= attributs["beta"]
					self.nuc[ds_get_name]	= attributs["nuc"]

					nomRetourComptes = "Temp_to_Native_count_" + ds_get_name

				elif "bandfactor" in attributs :
					# Albedo.
					self.bandfactor[ds_get_name] = attributs["bandfactor"]

					nomRetourComptes = "Albedo_to_Native_count_" + ds_get_name

				else :
					print(
						"Nuc or bandfactor not founded int the attributs of ",
						ds_get_name)
					exit(1)

				self.variableComptes[ds_get_name] = self.nc[nomRetourComptes]
				# (65536). Correspondence from 0 to 65535 towards 
				# the original spatial agency counts.

				self.mda[ds_get_name] = self.preparer_metadata(variable)
		# initialisation_dataset()

	def comebacktoNativeData(self, ds_get_name) :
		""" Come back to the original counts of the hrit.
		ds_get_name : meteo france name of a channel : IR_108. """

		variable = self.nc[ds_get_name]
		# Variable is between -9000 to 4000 (temperature) 
		# or between 0 to 10000 (albedo).

		offset = self.offset[ds_get_name]		# 0 or 273.15
		variable += 32768				# 0 to 65535

		if offset == 0. :
			# Albedo.
			nom = "Albedo_to_Native_count_" + ds_get_name
		else :
			nom = "Temp_to_Native_count_" + ds_get_name
			""" Temp_to_Native_count_IR_062 """

		variableComptes = self.nc[nom]		# (65536).
		# Correspondence from 0 to 65535 towards the original spatial agency counts.

		arrayTableConversion = xr.DataArray.to_numpy(variableComptes)

		tableau = arrayTableConversion[variable[:]]
		""" Come back to the original counts of the hrit.
		tableau : 0 to 4095 if native datas coded with 12 bits. """

		variable[:] = tableau
		return(variable)
		# comebacktoNativeData(self, ds_get_name)

	def comebacktoRadiance(self, ds_get_name) :
		# Come back to the radiance.

		variable = self.nc[ds_get_name]
		# Variable is between -9000 to 4000 (temperature)
		# or between 0 to 10000 (albedo).

		scale_factor = self.scale_factor[ds_get_name]		# 0.01
		offset = self.offset[ds_get_name]		# 0 or 273.15

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
			# variable is in Kelvin.
			variable = variable * alpha + beta

			c1 = 1.1910427e-5		# Planck
			c2 = 1.4387752			# Planck
			resul1 = c1 * np.power(nuc, 3.)
			resul2 = c2 * nuc
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
		# "IR_096"

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
				print(ds_get_name, ". offset = ", offset)
				message = "Soft not intended for a reflectance "
				message += "with a wave length more than 3 microns."
				print(message)
				exit(1)

			variable = variable * scale_factor + offset
			# variable is in Kelvin.

		elif calibration == "reflectance" :
			variable = self.nc[ds_get_name]
			# VIS006 calibration.reflectance, from 0 to 10000
			scale_factor = self.scale_factor[ds_get_name]
			offset = self.offset[ds_get_name]
			if offset != 0. :
				print(ds_get_name, ". offset = ", offset)
				message = "Soft not intended "
				message += "for a brightness temperature "
				message += "with a wave length less than 3 microns."
				print(message)
				exit(1)

			variable = variable * scale_factor
			# variable is an albedo between 0 and 100.

		else :
			print("Calibration mode not expected : ", calibration)
			exit(1)

		variable = variable.rename({variable.dims[0] : "y", variable.dims[1] : "x"})
		variable.attrs.update(mda)
		return variable
		# get_dataset()

	def orbital_param(self) :
		orb_param_dict = {
			"orbital_parameters": {
				"satellite_actual_longitude": self.longitudeReelle,
				"satellite_actual_latitude": 0.,
				"satellite_actual_altitude": self.alt,
				"satellite_nominal_longitude": self.longitudedeprojection,
				"satellite_nominal_latitude": 0,
				"satellite_nominal_altitude": self.alt,
				"projection_longitude": self.longitudedeprojection,
				"projection_latitude": 0.,
				"projection_altitude": self.alt,
				}
			}

		return orb_param_dict

	def resolutionSeviri(self, pdict) :
		if self.nbpix == 3712 :
			pdict["a_desc"] = "MSG/SEVIRI low resolution channel area"
			pdict["p_id"] = "msg_lowres"
		elif self.nbpix == 11136 :
			pdict["a_desc"] = "MSG/SEVIRI HRV channel area"
			pdict["p_id"] = "msg_hires"
		else :
			print("ERROR : not expected resolution for msg : ", self.nbpix)
			exit(1)
		return(pdict)

	def resolutionFci(self, pdict) :
		if self.nbpix == 5568 :
			pdict["a_desc"] = "MTG 2km channel area"
			pdict["p_id"] = "mtg_lowres"
		elif self.nbpix == 11136 :
			pdict["a_desc"] = "MTG 1km channel area"
			pdict["p_id"] = "mtg_midres"
		elif self.nbpix == 22272 :
			pdict["a_desc"] = "MTG 500m channel area"
			pdict["p_id"] = "mtg_hires"
		else :
			print("ERROR : not expected resolution for mtg : ", self.nbpix)
			exit(1)
		return(pdict)

	def resolutionAhi(self, pdict) :
		if self.nbpix == 5500 :
			pdict["a_desc"] = "HIMA 2km channel area"
			pdict["p_id"] = "hima_lowres"
		elif self.nbpix == 11000 :
			pdict["a_desc"] = "HIMA 1km channel area"
			pdict["p_id"] = "hima_midres"
		elif self.nbpix == 22000 :
			pdict["a_desc"] = "HIMA 500m channel area"
			pdict["p_id"] = "hima_hires"
		else :
			print(
				"ERROR : not expected resolution for hima : ",
				self.nbpix)
			exit(1)
		return(pdict)

	def resolutionAbi(self, pdict) :
		if self.nbpix == 5424 :
			pdict["a_desc"] = "GOESR 2km channel area"
			pdict["p_id"] = "goesr_lowres"
		elif self.nbpix == 10848 :
			pdict["a_desc"] = "GOESR 1km channel area"
			pdict["p_id"] = "goesr_midres"
		elif self.nbpix == 21696 :
			pdict["a_desc"] = "GOESR 500m channel area"
			pdict["p_id"] = "goesr_hires"
		else :
			print(
				"ERROR : not expected resolution for goesr : ",
				self.nbpix)
			exit(1)
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
		pdict["ssp_lon"] = self.longitudedeprojection
		pdict["ncols"] = self.nblig
		pdict["nlines"] = self.nbpix
		pdict["sweep"] = "y"

		# Force scandir to SEVIRI default, not known from file
		pdict["scandir"] = "S2N"
		pdict["a_name"] = "geosmsg"

		if self.sensor == "seviri" :
			# msg.
			pdict["scandir"] = "N2S"
			pdict["a_name"] = "geosmsg"
			pdict = self.resolutionSeviri(pdict)

		elif self.sensor == "fci" :
			# mtg.
			pdict["scandir"] = "N2S"
			pdict["a_name"] = "geosmtg"
			pdict = self.resolutionFci(pdict)

		elif self.sensor == "ahi" :
			# Himawari.
			pdict["scandir"] = "N2S"
			pdict["a_name"] = "geoshima"
			pdict = self.resolutionAhi(pdict)

		elif self.sensor == "abi" :
			# Goesr.
			pdict["scandir"] = "N2S"
			pdict["a_name"] = "geosgoesr"
			pdict["sweep"] = "x"
			pdict = self.resolutionAbi(pdict)

		else :
			print("ERROR : " + self.sensor + " not expected.")
			exit(1)

		aex = get_area_extent(pdict)
		area = get_area_definition(pdict, aex)

		# print("area = ", area)
		return area
		# get_area_def()
