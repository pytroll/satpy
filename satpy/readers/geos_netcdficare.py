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

The ``geos_netcdficare`` reader reads some geostationnary netcdf build by Meteo France and stored at Icare.
The brightness tempeture and albedo are calibrated.

That has been stored by the ICARE Data and Services Center
Data can be accessed via: http://www.icare.univ-lille1.fr

This reader concerns the following netcdfs :

. msg with a longitude near 0° :
Mmultic3kmNC4_msg03_20231113_111500.nc Mmultic1kmNC4_msg03_20231113_111500.nc

. Msg rapid scan with a longitude near 9.5° :
Mrsmultic3kmNC4_msg03_20231113_111500.nc Mrsmultic1kmNC4_msg03_20231113_111500.nc

. Msg with a longitude near 42° :
Imultic3kmNC4_msg03_20231113_111500.nc Imultic1kmNC4_msg03_20231113_111500.nc

. Himawari :
Jmultic2kmNC4_hima09_20231113_111500.nc Jmultic1kmNC4_hima09_20231113_111500.nc Jmultic500mNC4_hima09_20231113_111500.nc

. Goesr near -137° :
Wmultic2kmNC4_goes16_202406281000.nc. The better resolution are not built at Lannion, only at Tahiti.

. Goesr in -75° :
Emultic2kmNC4_goes16_202406281000.nc Emultic1kmNC4_goes16_202406281000.nc Emultic500mNC4_goes16_202406281000.nc

. Mtg :
Mmultic2km_mtgi1_20240104_090000.nc Mmultic1km_mtgi1_20240104_090000.nc Mmultic500m_mtgi1_20240104_090000.nc


Example:
--------
Here is an example how to read the data in satpy:

.. code-block:: python

	from satpy import Scene
	import glob

	filenames = glob.glob('data/*2019-03-01T12-00-00*.hdf')
	scn = Scene(filenames = filenames, reader = 'hima_netcdficare')
	scn.load(['true_color'])	# scn.load(['VIS006'])

	my_area = AreaDefinition('my_area', 'zone', 'my_area',
						 '+proj=latlong +lon_0=0 +a=6378169 +b=6356583 +h=35785831 +x_0=0 +y_0=0 +pm=0',
						 8500, 4000,
						 [-180., -80., 180., 80],
						 nprocs=16)

	natscn = scn.resample(my_area, resampler='nearest')
	natscn.save_dataset(composite_name, filename = filename_image_out)

Output:

.. code-block:: none

	<xarray.DataArray 'array-a1d52b7e19ec5a875e2f038df5b60d7e' (y: 3712, x: 3712)>
	dask.array<add, shape=(3712, 3712), dtype=float32, chunksize=(1024, 1024), chunktype=numpy.ndarray>
	Coordinates:
		crs	  object +proj=geos +a=6378169.0 +b=6356583.8 +lon_0=0.0 +h=35785831.0 +units=m +type=crs
	  * y		(y) float64 5.566e+06 5.563e+06 5.56e+06 ... -5.566e+06 -5.569e+06
	  * x		(x) float64 -5.566e+06 -5.563e+06 -5.56e+06 ... 5.566e+06 5.569e+06
	Attributes:
		start_time:		   2004-12-29 12:15:00
		end_time:			 2004-12-29 12:27:44
		area:				 Area ID: geosmsg\nDescription: MSG/SEVIRI low resol...
		name:				 IR_108
		resolution:		   3000.403165817
		calibration:		  brightness_temperature
		polarization:		 None
		level:				None
		modifiers:			()
		ancillary_variables:  []


"""

import datetime as dt

import numpy as np

from satpy.readers._geos_area import get_area_definition, get_area_extent

from netCDF4 import Dataset
from xarray import DataArray
import xarray as xr		

from satpy.readers.file_handlers import BaseFileHandler

from satpy._compat import cached_property
# from satpy.utils import get_dask_chunk_size_in_bytes
import	dask	
from satpy.readers import open_file_or_filename	
import math

class NETCDF_ICARE(BaseFileHandler) :
	# Cf readers/file_handlers.py.

	def __init__(self, filename, filename_info, filetype_info) :
		"""Init the file handler."""

		super().__init__(filename, filename_info, filetype_info) # cache_var_size=0, cache_handle = False)
		# self.ncfileIn = Dataset(filename, mode = "r", clobber=True, diskless=False, persist=False, format='NETCDF4')

		# self.filename = filename
		# self.filetype_info = filetype_info

		# chunk_bytes = self._chunk_bytes_for_resolution()
		chunk_bytes = '128 MiB'
		# chunk_bytes = '64 MiB'

		with dask.config.set({"array.chunk-size": chunk_bytes}) :
			# with dask.config.set(**{'array.slicing.split_large_chunks': True}) :
			f_obj = open_file_or_filename(self.filename)
			self.nc = xr.open_dataset(f_obj,
						decode_cf=True,
						mask_and_scale=False,
						chunks={"xc": "auto", "yc": "auto"})

		self.metadata = {}

		self.metadata["start_time"] = self.get_start_time()
		self.metadata["end_time"] = self.get_end_time()

		print("Reading: {}".format(filename), " start: {} ".format(self.start_time), \
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


	def _chunk_bytes_for_resolution(self) -> int:
		"""Get a best-guess optimal chunk size for resolution-based chunking.

		First a chunk size is chosen for the provided Dask setting `array.chunk-size`
		and then aligned with a hardcoded on-disk chunk size of 226. This is then
		adjusted to match the current resolution.

		This should result in 500 meter data having 4 times as many pixels per
		dask array chunk (2 in each dimension) as 1km data and 8 times as many
		as 2km data. As data is combined or upsampled geographically the arrays
		should not need to be rechunked. Care is taken to make sure that array
		chunks are aligned with on-disk file chunks at all resolutions, but at
		the cost of flexibility due to a hardcoded on-disk chunk size of 226
		elements per dimension.

		"""
		num_high_res_elems_per_dim = math.sqrt(get_dask_chunk_size_in_bytes() / 4)  # 32-bit floats
		# assume on-disk chunk size of 226
		# this is true for all CSPP Geo GRB output (226 for all sectors) and full disk from other sources
		# 250 has been seen for AWS/CLASS CONUS, Mesoscale 1, and Mesoscale 2 files
		# we align this with 4 on-disk chunks at 500m, so it will be 2 on-disk chunks for 1km, and 1 for 2km
		high_res_elems_disk_aligned = round(max(num_high_res_elems_per_dim / (4 * 226), 1)) * (4 * 226)
		low_res_factor = int(self.filetype_info.get("resolution", 2000) // 500)
		res_elems_per_dim = int(high_res_elems_disk_aligned / low_res_factor)

		print("num_high_res_elems_per_dim = ", num_high_res_elems_per_dim, 	\
			" res_elems_per_dim = ", res_elems_per_dim)
		# num_high_res_elems_per_dim =  5792.618751480198  res_elems_per_dim =  1356
		return (res_elems_per_dim ** 2) * 2  # 16-bit integers on disk


	def sensor_name(self) :
		"""Get the sensor name."""
		# the sensor and platform names are stored together, eg: MSG1/SEVIRI
		variable = self.nc["satellite"]
		self.plateforme = variable.attrs["id"]

		if "goes" in self.plateforme :
			self.sensor = "abi"
		elif "msg" in self.plateforme :
			self.sensor = "seviri"
		elif "MSG" in self.plateforme :
			self.sensor = "seviri"
		elif "mtg" in self.plateforme :
			self.sensor = "fci"
		elif "hima" in self.plateforme :
			self.sensor = "ahi"
		elif "MTG" in self.plateforme :
			self.sensor = "fci"
		else :
			raise NameError("Unsupported satellite platform : " + self.plateforme)
			
		# Icare and météo france use non-standard platform names. Change is needed for pyspectral.
		# pyspectral/rsr_seviri_Meteosat-10.h5 and not rsr_seviri_msg3_seviri.h5 in the call
		# Calculator(metadata["platform_name"], metadata["sensor"], metadata["name"]).
		if self.plateforme == "msg1" or self.plateforme == "msg01" or self.plateforme == "MSG1" :
			self.plateforme = "Meteosat-08"
		elif self.plateforme == "msg2" or self.plateforme == "msg02" or self.plateforme == "MSG2" :
			self.plateforme = "Meteosat-09"
		elif self.plateforme == "msg3" or self.plateforme == "msg03" or self.plateforme == "MSG3" :
			self.plateforme = "Meteosat-10"
		elif self.plateforme == "msg4" or self.plateforme == "msg04" or self.plateforme == "MSG4" :
			self.plateforme = "Meteosat-11"
		elif self.plateforme == "mtgi1" or self.plateforme == "mtg1" or self.plateforme == "MTG01" :
			self.plateforme = "Meteosat-12"
		elif self.plateforme == "goes16" :
			self.plateforme = "GOES-16"
		elif self.plateforme == "goes17" :
			self.plateforme = "GOES-17"
		elif self.plateforme == "goes18" :
			self.plateforme = "GOES-18"
		elif self.plateforme == "goes19" :
			self.plateforme = "GOES-19"
		elif self.plateforme == "hima08" :
			self.plateforme = "Himawari-8"
		elif self.plateforme == "hima09" :
			self.plateforme = "Himawari-9"
		# sensor_name()



	def satlon(self):
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
		"""Get the resolution."""
		# The resolution can be read in the attribute geotransform of the follonwing variables :
		# GeosCoordinateSystem500m, GeosCoordinateSystem_h, 
		# GeosCoordinateSystem1km, GeosCoordinateSystem2km,
		# GeosCoordinateSystem.
		# cfac, lfac, coff, loff can be read in the variables ImageNavigationxxx.
		trouve = False
		try :
			# Mtg, himawari, goesr.
			variable = self.nc["GeosCoordinateSystem500m"]
			variableX = self.nc["X500m"]
			variableY = self.nc["Y500m"]
			chaineNavigation = "ImageNavigation500m"
			trouve = True
		except KeyError :
			None
		if not trouve :
			try :
				# Hrv from msg.
				variable = self.nc["GeosCoordinateSystem_h"]
				variableX = self.nc["X_h"]
				variableY = self.nc["Y_h"]
				chaineNavigation = "ImageNavigation_h"
				trouve = True
			except KeyError :
				None
		if not trouve :
			try :
				# Mtg, himawari, goesr.
				variable = self.nc["GeosCoordinateSystem1km"]
				variableX = self.nc["X1km"]
				variableY = self.nc["Y1km"]
				chaineNavigation = "ImageNavigation1km"
				trouve = True
			except KeyError :
				None
		if not trouve :
			try :
				# Mtg, himawari, goesr.
				variable = self.nc["GeosCoordinateSystem2km"]
				variableX = self.nc["X2km"]
				variableY = self.nc["Y2km"]
				chaineNavigation = "ImageNavigation2km"
				trouve = True
			except KeyError :
				None

		if not trouve :
			try :
				# Msg in 3 kms.
				variable = self.nc["GeosCoordinateSystem"]
				variableX = self.nc["X"]
				variableY = self.nc["Y"]
				chaineNavigation = "ImageNavigation"
				trouve = True
			except KeyError :
				None

		if not trouve :
			print("Variables GeosCoordinateSystemXX not founded.")
			exit(1)

		geotransform = variable.attrs["GeoTransform"]

		# print("geotransform = ", geotransform)
		# geotransform =  -5570254, 3000.40604, 0, 5570254, 0, -3000.40604
		morceaux = geotransform.split(", ")
		self.resolution = float(morceaux[1])	# 3000.40604
		# print("resolution = ", self.resolution)

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


	def get_end_time(self) :
		"""Get the end time. Global attribute of the netcdf."""
		attr = self.nc.attrs["time_coverage_end"]
		# YYYY-MM-DDTHH:MM:SSZNNN
		# In some versions milliseconds are present, sometimes not.
		# For the goesr : 2024-06-28T10:00:21.1Z
		longueurchaine = len(attr)
		if longueurchaine == 22 :
			# Goesr.
			stacq = dt.datetime.strptime(attr, "%Y-%m-%dT%H:%M:%S.%fZ")
		elif longueurchaine == 20 :
			# Mtg.
			stacq = dt.datetime.strptime(attr, "%Y-%m-%dT%H:%M:%SZ")
		else :
			# Msg, hima.
			stacq = dt.datetime.strptime(attr, "%Y-%m-%dT%H:%M:%SZ%f")
		return stacq
		# get_end_time()

	def get_start_time(self) :
		"""Get the start time. Global attribute of the netcdf."""
		attr = self.nc.attrs["time_coverage_start"]

		# YYYY-MM-DDTHH:MM:SSZ
		# In some versions milliseconds are present, sometimes not.
		longueurchaine = len(attr)
		if longueurchaine == 22 :
			# Goesr.
			stacq = dt.datetime.strptime(attr, "%Y-%m-%dT%H:%M:%S.%fZ")
		elif longueurchaine == 20 :
			# Mtg.
			stacq = dt.datetime.strptime(attr, "%Y-%m-%dT%H:%M:%SZ")
		else :
			# Msg, hima.
			stacq = dt.datetime.strptime(attr, "%Y-%m-%dT%H:%M:%SZ%f")
		return stacq
		# get_start_time()

	@property
	def start_time(self) :
		return(self.metadata["start_time"])
	@property
	def end_time(self) :
		return(self.metadata["end_time"])


	@property
	def alt(self):
		"""Get the altitude."""
		variable = self.nc["satellite"]
		altitude = variable.attrs["dst"]
		# 36000000 meters.
		altitude += 6378169.	# equatorial radius of the earth.
		return altitude


	# -----------------------------------------------------	#
	# -----------------------------------------------------	#
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
		"""Return the correct dataset name based on requested band."""
		# ds_id =  DataID(name='vis_08', wavelength=WavelengthRange(min=0.815, central=0.865, max=0.915, unit='µm'),
		#		resolution=2000, calibration=<calibration.reflectance>, modifiers=())
		
		# For mtg :
		if ds_id["name"] == 'vis_04' :		# Name in satpy.
			ds_get_name = "VIS004"		# Name in icare/meteofrance netcdf.
		elif ds_id["name"] == 'vis_05' :
			ds_get_name = "VIS005"
		elif ds_id["name"] == 'vis_06' :
			ds_get_name = "VIS006"
		elif ds_id["name"] == 'vis_08' :
			ds_get_name = "VIS008"
		elif ds_id["name"] == 'vis_09' :
			ds_get_name = "VIS009"
		elif ds_id["name"] == 'nir_13' :
			ds_get_name = "IR_013"
		elif ds_id["name"] == 'nir_16' :
			ds_get_name = "IR_016"
		elif ds_id["name"] == 'nir_22' :
			ds_get_name = "IR_022"
		elif ds_id["name"] == 'ir_38' :
			ds_get_name = "IR_038"
		elif ds_id["name"] == 'wv_63' :
			ds_get_name = "WV_063"
		elif ds_id["name"] == 'wv_73' :
			ds_get_name = "WV_073"
		elif ds_id["name"] == 'ir_87' :
			ds_get_name = "IR_087"
		elif ds_id["name"] == 'ir_97' :
			ds_get_name = "IR_097"
		elif ds_id["name"] == 'ir_105' :
			ds_get_name = "IR_105"
		elif ds_id["name"] == 'ir_123' :
			ds_get_name = "IR_123"
		elif ds_id["name"] == 'ir_133' :
			ds_get_name = "IR_133"

		# For msg, the satpy and icare channel names are the same.
		elif ds_id["name"] == 'IR_039' or ds_id["name"] == 'IR_016' or ds_id["name"] == 'VIS008' \
			or ds_id["name"] == 'IR_087' or ds_id["name"] == 'IR_097' or ds_id["name"] == 'IR_108' \
			or ds_id["name"] == 'IR_120' or ds_id["name"] == 'IR_134' or ds_id["name"] == 'VIS006' \
			or ds_id["name"] == 'WV_062' or ds_id["name"] == 'WV_073' or ds_id["name"] == 'HRV' :
			ds_get_name = ds_id["name"]

		# For the goesr :
		elif ds_id["name"] == 'C01' :
			ds_get_name = "VIS_004"
		elif ds_id["name"] == 'C02' :
			ds_get_name = "VIS_006"
		elif ds_id["name"] == 'C03' :
			ds_get_name = "VIS_008"
		elif ds_id["name"] == 'C04' :
			ds_get_name = "VIS_014"
		elif ds_id["name"] == 'C05' :
			ds_get_name = "VIS_016"
		elif ds_id["name"] == 'C06' :
			ds_get_name = "VIS_022"
		elif ds_id["name"] == 'C07' :
			ds_get_name = "IR_039"
		elif ds_id["name"] == 'C08' :
			ds_get_name = "IR_062"
		elif ds_id["name"] == 'C09' :
			ds_get_name = "IR_069"
		elif ds_id["name"] == 'C10' :
			ds_get_name = "IR_073"
		elif ds_id["name"] == 'C11' :
			ds_get_name = "IR_085"
		elif ds_id["name"] == 'C12' :
			ds_get_name = "IR_096"
		elif ds_id["name"] == 'C13' :
			ds_get_name = "IR_103"
		elif ds_id["name"] == 'C14' :
			ds_get_name = "IR_114"
		elif ds_id["name"] == 'C15' :
			ds_get_name = "IR_123"
		elif ds_id["name"] == 'C16' :
			ds_get_name = "IR_133"

		# For himawari.
		elif ds_id["name"] == 'B01' :		# Name in satpy.
			ds_get_name = "VIS004"		# Name in icare/meteofrance netcdf.
		elif ds_id["name"] == 'B02' :
			ds_get_name = "VIS005"
		elif ds_id["name"] == 'B03' :
			ds_get_name = "VIS006"
		elif ds_id["name"] == 'B04' :
			ds_get_name = "VIS008"
		elif ds_id["name"] == 'B05' :
			ds_get_name = "IR_016"
		elif ds_id["name"] == 'B06' :
			ds_get_name = "IR_022"
		elif ds_id["name"] == 'B07' :
			ds_get_name = "IR_038"
		elif ds_id["name"] == 'B08' :
			ds_get_name = "WV_062"
		elif ds_id["name"] == 'B09' :
			ds_get_name = "WV_069"
		elif ds_id["name"] == 'B10' :
			ds_get_name = "WV_073"
		elif ds_id["name"] == 'B11' :
			ds_get_name = "IR_085"
		elif ds_id["name"] == 'B12' :
			ds_get_name = "IR_096"
		elif ds_id["name"] == 'B13' :
			ds_get_name = "IR_104"
		elif ds_id["name"] == 'B14' :
			ds_get_name = "IR_112"
		elif ds_id["name"] == 'B15' :
			ds_get_name = "IR_123"
		elif ds_id["name"] == 'B16' :
			ds_get_name = "IR_132"

		else :
			print("Soft not adaptated for this channel : ds_id = ", ds_id["name"])
			exit(1)

		return ds_get_name
		# _get_dsname()


	def initialisation_dataset(self) :
		listeToutesVariables = {"VIS004", "VIS_004", "VIS005", "VIS_005", "VIS006", "VIS_006", "VIS006", \
		"VIS_008", "VIS008", "VIS_009", "VIS009", "IR_013", "IR_016", "IR_022", "IR_038", "IR_039",	\
		"IR_062", "WV_062", "WV_063", "IR_069", "WV_069", "IR_073", "WV_073", "IR_085", "IR_087",	\
		"IR_096", "IR_097", "IR_103", "IR_104", "IR_105", "IR_108", "IR_112", "IR_120", "IR_123", 	\
		"IR_132", "IR_133", "IR_134", "HRV"}

		self.mda = {}
		self.scale_factor = {}
		self.offset = {}
		self.alpha = {}
		self.beta = {}
		self.bandfactor = {}
		self.nuc = {}
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

					nomRetourComptes = "Temp_to_Native_count_" + ds_get_name

				elif "bandfactor" in attributs :
					# Albedo.
					self.bandfactor[ds_get_name] = attributs["bandfactor"]

					nomRetourComptes = "Albedo_to_Native_count_" + ds_get_name

				else :
					print("Nuc or bandfactor not founded int the attributs of ", ds_get_name)
					exit(1)

				self.variableComptes[ds_get_name] = self.nc[nomRetourComptes]
				# (65536). Correspondence from 0 to 65535 towards the original spatial agency counts.

				self.mda[ds_get_name] = self.preparer_metadata(variable)
		# initialisation_dataset()


	def get_dataset(self, ds_id, ds_info) :
		"""Get the dataset.
		ds_id["calibration"]	= key["calibration"] 
					= ["brightness_temperature", "reflectance", "radiance", "counts"]
		"""
		ds_get_name = self._get_dsname(ds_id)	# "IR_096"

		variable = self.nc[ds_get_name]

		scale_factor = self.scale_factor[ds_get_name]
		offset = self.offset[ds_get_name]

		# print(ds_get_name, ds_id["calibration"], " min = ", np.min(variable[:].values), 
		#	" max = ", np.max(variable[:].values), "dims = ", variable.dims)
		# WV_062 calibration.brightness_temperature, from -9000 to 4000
		# VIS006 calibration.reflectance, from 0 to 10000

		mda = self.mda[ds_get_name]
		mda.update(ds_info)

		calibration = ds_id["calibration"]
		if calibration == "counts" :
			# Come back to the original counts of the hrit...

			# Variable is between -9000 to 4000 (temperature) 
			# or between 0 to 10000 (albedo).

			variable += 32768	# 0 to 65535

			if offset == 0. :
				# Albedo.
				nom = "Albedo_to_Native_count_" + ds_get_name
			else :
				nom = "Temp_to_Native_count_" + ds_get_name

			variableComptes = self.variableComptes[ds_get_name]
			# (65536). Correspondence from 0 to 65535 towards the original spatial agency counts.

			# Come back to the original counts of the hrit...
			sortie = variablesComptes[AlbedoBT]

		elif calibration == "radiance" :
			# Come back to the radiance.

			if offset == 0. : 
				# Visible channel.
				bandfactor = self.bandfactor[ds_get_name]

				# Variable is an albedo from 0 to 10000.
				variable = variable * scale_factor/ 100. * bandfactor
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

		elif calibration == "brightness_temperature" :
			if offset != 273.15 :
				print(ds_get_name, ". offset = ", offset)
				print("Soft not intended for a reflectance with a wave length more than 3 microns.")
				exit(1)

			variable = variable * scale_factor + offset
			# variable is in Kelvin.

		elif calibration == "reflectance" :
			if offset != 0. :
				print(ds_get_name, ". offset = ", offset)
				message = "Soft not intended for a brightness temperature "
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
			}}

		return orb_param_dict

	def get_area_def(self, ds_id) :
		"""Get the area def."""

		pdict = {}
		pdict["cfac"] = np.int32(self.cfac)
		pdict["lfac"] = np.int32(self.lfac)
		pdict["coff"] = np.float32(self.coff)
		pdict["loff"] = np.float32(self.loff)

		pdict["a"] = 6378169
		pdict["b"] = 6356583.8
		pdict["h"] = self.alt - pdict["a"]	# 36000000 mètres.
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
			if self.nbpix == 3712 :
				pdict["a_desc"] = "MSG/SEVIRI low resolution channel area"
				pdict["p_id"] = "msg_lowres"
			elif self.nbpix == 11136 :
				pdict["a_desc"] = "MSG/SEVIRI HRV channel area"
				pdict["p_id"] = "msg_hires"
			else :
				print("ERROR : not expected resolution for msg : ", self.nbpix)
				exit(1)

		elif self.sensor == "fci" :
			# mtg.
			pdict["scandir"] = "N2S"
			pdict["a_name"] = "geosmtg"
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
				
		elif self.sensor == "ahi" :
			# Himawari.
			pdict["scandir"] = "N2S"
			pdict["a_name"] = "geoshima"
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
				print("ERROR : not expected resolution for hima : ", self.nbpix)
				exit(1)

		elif self.sensor == "abi" :
			# Goesr.
			pdict["scandir"] = "N2S"
			pdict["a_name"] = "geosgoesr"
			pdict["sweep"] = "x"
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
				print("ERROR : not expected resolution for goesr : ", self.nbpix)
				exit(1)
		else :
			print("ERROR : " + self.sensor + " not expected.")
			exit(1)

		aex = get_area_extent(pdict)
		area = get_area_definition(pdict, aex)

		# print("area = ", area)
		return area
		# get_area_def()



