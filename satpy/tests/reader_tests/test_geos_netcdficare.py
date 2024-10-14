#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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

"""Tests for the Icare MeteoFrance netcdfs reader,
	satpy/readers/geos_netcdficare.py.

SYNTAXE :
	pytest

	Before, copy the satpy/tests/reader_tests/test_geos_netcdfcare.py file
	into the pytest directory pytest_projec.

EXECUTION TIME :
	20 seconds.
DATE OF CREATION :
	2024 11th october.
LAST VERSIONS :

AUTHOR :
	Meteo France.
"""

import os

import numpy as np

from satpy.scene import Scene
from satpy import find_files_and_readers

from datetime import datetime
from netCDF4 import Dataset


class TestGeosNetcdfIcareReader() :
	# Test of the geos_netcdficare reader.
	# This reader has been build for the Icare Meteo France netcdfs.

	def test_geos_netcdficare(self, tmp_path) :
		""" A dummy netcdf is built. 
		A scene self.scn for the convection product for the same date
		is built. We check that the scene parameters are the same 
		as thoses in the dummy netcdf.
		This procedure is called by pytest.
		"""

		self.init(tmp_path)

		startTime = self.scn.start_time
		startTimeString = startTime.strftime('%Y-%m-%dT%H:%M:%S')
		# 2024-06-28T10:00:40
		assert startTimeString == self.expectedStartTime

		endTime = self.scn.end_time
		endTimeString = endTime.strftime('%Y-%m-%dT%H:%M:%S')
		# 2024-06-28T10:12:41
		assert endTimeString == self.expectedEndTime

		sensor = self.scn.sensor_names
		for isensor in sensor :
			capteur = isensor
		assert capteur == self.expectedSensor

		platform = "erreur"
		altitude = -1.
		longitude = 999.

		for data_arr in self.values :
			# values come from the scene.
			# print("data_arr.attrs = ", data_arr.attrs)
			if "platform_name" in data_arr.attrs :
				platform = data_arr.attrs["platform_name"]
			if "orbital_parameters" in data_arr.attrs :
				subAttr = data_arr.attrs["orbital_parameters"]
				if "satellite_actual_altitude" in subAttr :
					altitude = subAttr["satellite_actual_altitude"]
			if "satellite_actual_longitude" in data_arr.attrs :
				longitude = data_arr.attrs["satellite_actual_longitude"]

		longitude = float(int(longitude * 10.)) / 10.
		assert platform == self.expectedPlatform
		assert longitude == self.expectedLongitude
		assert altitude == self.expectedAltitude

		xr = self.scn.to_xarray_dataset()
		# print("xr = ", xr)
		matrice = xr["convection"]
		nblin = matrice.shape[1]
		nbpix = matrice.shape[2]
		assert nblin == self.expectedNblin
		assert nbpix == self.expectedNbpix

		cfac = xr.attrs["cfac"]
		assert cfac == self.expectedCfac
		lfac = xr.attrs["lfac"]
		assert lfac == self.expectedLfac
		coff = xr.attrs["coff"]
		assert coff == self.expectedCoff
		loff = xr.attrs["loff"]
		assert loff == self.expectedLoff

		satpyId = xr.attrs["_satpy_id"]
		# DataID(name='convection', resolution=3000.403165817)
		# Cf satpy/dataset/dataid.py.

		resolution = satpyId.get("resolution")
		resolution = float(int(resolution * 10.)) / 10.
		assert resolution == self.expectedResolution

		# A picture of convection composite will be displayed.
		self.scn.show("convection")
		print("The picture should be pink.")
		# test_geos_netcdficare(self, tmp_path)

	def init(self, tmp_path) :
		"""
		A fake netcdf is built.
		A scene is built with the reader to be tested, applied to this netcdf.
		Called by test_geos_netcdficare().
		"""
		self.netcdfName = tmp_path / "Mmultic3kmNC4_msg03_202406281000.nc"
		self.filepath = tmp_path

		self.buildNetcdf(self.netcdfName)

		# We will check that the parameters written in the dummy netcdf can be read.
		self.expectedStartTime = "2024-06-28T10:00:09"
		self.expectedEndTime = "2024-06-28T10:12:41"
		actualAltitude = 35786691 + 6378169		# 42164860.0
		actualLongitude = 0.1
		self.expectedPlatform = "Meteosat-10"
		self.expectedSensor = "seviri"
		self.expectedAltitude = actualAltitude
		self.expectedLongitude = actualLongitude
		self.expectedCfac = 1.3642337E7
		self.expectedLfac = 1.3642337E7
		self.expectedCoff = 1857.0
		self.expectedLoff = 1857.0
		self.expectedResolution = 3000.4
		self.expectedNblin = 3712
		self.expectedNbpix = 3712

		# To build a scene at the date 20240628_100000,
		# a netcdf corresponding to msg_netcdficare 
		# is looked for in the filepath directory.
		yaml_file = 'msg_netcdficare'
		myfiles = find_files_and_readers(
			base_dir=self.filepath, start_time=datetime(2024, 6, 28, 10, 0),
			end_time=datetime(2024, 6, 28, 10, 0), reader=yaml_file)
		print("Found myfiles = ", myfiles)
		# {'msg_netcdficare': ['/tmp/Mmultic3kmNC4_msg03_202406281000.nc']}

		self.scn = Scene(filenames=myfiles, reader=yaml_file)

		print(self.scn.available_dataset_names())
		# ['IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120',
		# 'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073']

		print(self.scn.available_composite_names())
		""" Static compositor  {'_satpy_id': DataID(name='_night_background'),
		'standard_name': 'night_background',
		'prerequisites': [],
		'optional_prerequisites': []}
		Static compositor  {'_satpy_id': DataID(name='_night_background_hires'),
		'standard_name': 'night_background', 'prerequisites': [],
		'optional_prerequisites': []}
		['airmass', 'ash', 'cloud_phase_distinction', 'cloud_phase_distinction_raw',
		'cloudtop', 'cloudtop_daytime', 'colorized_ir_clouds', 'convection'...
		"""

		self.scn.load(['convection'])
		self.values = self.scn.values()
		# init()

	def buildNetcdf(self, ncName) :
		"""
		ncName : tmp_path / Mmultic3kmNC4_msg03_202406281000.nc
		A dummy icare Meteo France netcdf is built here.
		Called by init().
		"""
		if os.path.exists(ncName) :
			os.remove(ncName)
		ncfileOut = Dataset(
			ncName, mode="w", clobber=True,
			diskless=False, persist=False, format='NETCDF4')

		ncfileOut.createDimension(u'ny', 3712)
		ncfileOut.createDimension(u'nx', 3712)
		ncfileOut.createDimension(u'numerical_count', 65536)
		ncfileOut.setncattr("time_coverage_start", "2024-06-28T10:00:09Z383")
		ncfileOut.setncattr("time_coverage_end", "2024-06-28T10:12:41Z365")
		ncfileOut.setncattr("Area_of_acquisition", "globe")

		fill_value = -32768
		var = ncfileOut.createVariable(
			"satellite", "c", zlib=True, complevel=4,
			shuffle=True, fletcher32=False, contiguous=False,
			chunksizes=None, endian='native', least_significant_digit=None)

		var.setncattr("id", "msg03")
		var.setncattr("dst", 35786691.)
		var.setncattr("lon", float(0.1))

		var = ncfileOut.createVariable(
			"geos", "c", zlib=True, complevel=4, shuffle=True,
			fletcher32=False, contiguous=False, chunksizes=None,
			endian='native', least_significant_digit=None)
		var.setncattr("longitude_of_projection_origin", 0.)

		var = ncfileOut.createVariable(
			"GeosCoordinateSystem", "c", zlib=True, complevel=4,
			shuffle=True, fletcher32=False, contiguous=False,
			chunksizes=None, endian='native', least_significant_digit=None)
		var.setncattr(
			"GeoTransform",
			"-5570254, 3000.40604, 0, 5570254, 0, -3000.40604")

		var = ncfileOut.createVariable(
			"ImageNavigation", "c", zlib=True, complevel=4,
			shuffle=True, fletcher32=False, contiguous=False,
			chunksizes=None, endian='native', least_significant_digit=None)
		var.setncattr("CFAC", 1.3642337E7)
		var.setncattr("LFAC", 1.3642337E7)
		var.setncattr("COFF", 1857.0)
		var.setncattr("LOFF", 1857.0)

		var = ncfileOut.createVariable(
			"X", 'float32', u'nx', zlib=True, complevel=4,
			shuffle=True, fletcher32=False, contiguous=False,
			chunksizes=None, endian='native', least_significant_digit=None)
		x0 = -5570254.
		dx = 3000.40604
		var[:] = np.array(([(x0 + dx * i) for i in range(3712)]))

		var = ncfileOut.createVariable(
			"Y", 'float32', u'ny', zlib=True, complevel=4,
			shuffle=True, fletcher32=False, contiguous=False,
			chunksizes=None, endian='native', least_significant_digit=None)
		y0 = 5570254.
		dy = -3000.40604
		var[:] = np.array(([(y0 + dy * i) for i in range(3712)]))

		for channel in {"VIS006", "VIS008", "IR_016"} :
			var = ncfileOut.createVariable(
				channel, 'short', ('ny', 'nx'), zlib=True, complevel=4,
				shuffle=True, fletcher32=False, contiguous=False,
				chunksizes=None, endian='native',
				least_significant_digit=None, fill_value=fill_value)
			var[:] = np.array(([[i * 2 for i in range(3712)] for j in range(3712)]))
			# Hundredths of albedo between 0 and 10000.
			var.setncattr("scale_factor", 0.01)
			var.setncattr("add_offset", 0.)
			var.setncattr("bandfactor", 20.76)
			var.setncattr("_CoordinateSystems", "GeosCoordinateSystem")

			var = ncfileOut.createVariable(
				"Albedo_to_Native_count_" + channel, 'short',
				'numerical_count', zlib=True, complevel=4, shuffle=True,
				fletcher32=False, contiguous=False, chunksizes=None,
				endian='native', least_significant_digit=None,
				fill_value=-9999)
			var[:] = np.array(([-9999 for i in range(65536)]))
			# In order to come back to the native datas on 10, 12 or 16 bits.

		for channel in {
			"IR_039", "WV_062", "WV_073", "IR_087", "IR_097",
			"IR_108", "IR_120", "IR_134"} :
			var = ncfileOut.createVariable(
				channel, 'short', ('ny', 'nx'), zlib=True, complevel=4,
				shuffle=True, fletcher32=False, contiguous=False,
				chunksizes=None, endian='native',
				least_significant_digit=None, fill_value=fill_value)
			var[:] = np.array(
				([[-9000 + j * 4 for i in range(3712)] for j in range(3712)]))
			# Hundredths of celcius degrees.
			var.setncattr("scale_factor", 0.01)
			var.setncattr("add_offset", 273.15)
			var.setncattr("nuc", 1600.548)
			var.setncattr("alpha", 0.9963)
			var.setncattr("beta", 2.185)
			var.setncattr("_CoordinateSystems", "GeosCoordinateSystem")

			var = ncfileOut.createVariable(
				"Temp_to_Native_count_" + channel, 'short',
				'numerical_count', zlib=True, complevel=4, shuffle=True,
				fletcher32=False, contiguous=False, chunksizes=None,
				endian='native', least_significant_digit=None,
				fill_value=-9999)
			var[:] = np.array(([-9999 for i in range(65536)]))
			# In order to come back to the native datas on 10, 12 or 16 bits.
		ncfileOut.close
		# buildNetcdf()

	# class TestGeosNetcdfIcareReader.
