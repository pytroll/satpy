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
    4 minutes.
DATE OF CREATION :
    2024 11th october.
LAST VERSIONS :

AUTHOR :
    Météo France.
"""

import os

import numpy as np

from satpy.scene import Scene
from satpy import find_files_and_readers

from datetime import datetime
from netCDF4 import Dataset

import logging
logger = logging.getLogger('netcdficare')


class TestGeosNetcdfIcareReader() :
    # Test of the geos_netcdficare reader.
    # This reader has been build for the Icare Meteo France netcdfs.

    def test_mtg_netcdficare(self, tmp_path) :
        """ A dummy netcdf is built.
        A scene self.scn for the nir_16 product for the same date
        is built. We check that the scene parameters are the same
        as thoses in the dummy netcdf.
        This procedure is called by pytest.
        """
        self.initMtg(tmp_path)
        self.scn.load(['nir_16'])
        self.values = self.scn.values()
        self.checkingSceneParameter("nir_16")
        # test_mtg_netcdficare()

    def test_msg_netcdficare(self, tmp_path) :
        self.initMsgHrv(tmp_path)
        self.scn.load(['HRV'])
        self.values = self.scn.values()
        self.checkingSceneParameter("HRV")

        self.initMsg(tmp_path)
        self.scn.load(['convection'])
        self.values = self.scn.values()
        self.checkingSceneParameter("convection")
        # test_msg_netcdficare()

    def test_hima_netcdficare(self, tmp_path) :
        self.initHima(tmp_path)
        self.scn.load(['B10'])
        self.values = self.scn.values()
        self.checkingSceneParameter("B10")
        # test_hima_netcdficare()

    def test_goesr_netcdficare(self, tmp_path) :
        self.initGoesr(tmp_path)
        self.scn.load(['C02'])
        self.values = self.scn.values()
        self.checkingSceneParameter("C02")
        # test_goesr_netcdficare()

    # -----------------------------------------------------    #
    # typeImage : convection, airmass...
    # -----------------------------------------------------    #
    def checkingSceneParameter(self, typeImage) :
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

        platform = "error"
        altitude = -1.
        longitude = 999.

        for data_arr in self.values :
            # values come from the scene.
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
        matrice = xr[typeImage]
        nbdim = len(matrice.shape)
        if nbdim == 3 :
            # RGB.
            nblin = matrice.shape[1]
            nbpix = matrice.shape[2]
        elif nbdim == 2 :
            # PGM.
            nblin = matrice.shape[0]
            nbpix = matrice.shape[1]
        else :
            print("Dimension of shape not expected : ", nbdim)
            exit(1)

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
        # checkingSceneParameter()

    def initMsg(self, tmp_path) :
        """
        A fake netcdf is built.
        A scene is built with the reader to be tested, applied to this netcdf.
        Called by test_msg_netcdficare().
        """
        self.netcdfName = tmp_path / "Mmultic3kmNC4_msg03_202406281000.nc"
        self.filepath = tmp_path

        listIR = {
            "IR_039", "WV_062", "WV_073", "IR_087", "IR_097",
            "IR_108", "IR_120", "IR_134"
            }

        self.buildNetcdf(
            self.netcdfName, 3712, "msg03", x0=-5570254, dx=3000.40604,
            y0=5570254, cfac=1.3642337E7, coff=1857.0,
            listVisible={"VIS006", "VIS008", "IR_016"}, listIR=listIR,
            coordinateSystemName="GeosCoordinateSystem",
            nomImageNavigation="ImageNavigation",
            nomX="X", nomY="Y", time_coverage_end="2024-06-28T10:12:41Z365")

        # We will check that the parameters written in the dummy netcdf
        # can be read.
        self.expectedStartTime = "2024-06-28T10:00:09"
        self.expectedEndTime = "2024-06-28T10:12:41"
        actualAltitude = 35786691 + 6378169        # 42164860.0
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
        logger.info("Found myfiles = ", myfiles)
        # {'msg_netcdficare': ['/tmp/Mmultic3kmNC4_msg03_202406281000.nc']}

        self.scn = Scene(filenames=myfiles, reader=yaml_file)
        # initMsg()

    def initMsgHrv(self, tmp_path) :
        """
        A fake netcdf is built.
        A scene is built with the reader to be tested, applied to this netcdf.
        Called by test_msg_netcdficare().
        """
        self.netcdfName = tmp_path / "Mmultic1kmNC4_msg03_202406281000.nc"
        self.filepath = tmp_path

        self.buildNetcdf(
            self.netcdfName, 11136, "msg03", x0=-5571254., dx=1000.135,
            y0=5570254., cfac=40927014, coff=5566.0, listVisible={"HRV"},
            listIR={}, coordinateSystemName="GeosCoordinateSystem_h",
            nomImageNavigation="ImageNavigation_h", nomX="X_h", nomY="Y_h",
            time_coverage_end="2024-06-28T10:12:41Z365")

        # We will check that the parameters written in the dummy netcdf
        # can be read.
        self.expectedStartTime = "2024-06-28T10:00:09"
        self.expectedEndTime = "2024-06-28T10:12:41"
        actualAltitude = 35786691 + 6378169        # 42164860.0
        actualLongitude = 0.1
        self.expectedPlatform = "Meteosat-10"
        self.expectedSensor = "seviri"
        self.expectedAltitude = actualAltitude
        self.expectedLongitude = actualLongitude
        self.expectedCfac = 40927014
        self.expectedLfac = 40927014
        self.expectedCoff = 5566.0
        self.expectedLoff = 5566.0
        self.expectedResolution = 1000.1
        self.expectedNblin = 11136
        self.expectedNbpix = 11136

        # To build a scene at the date 20240628_100000,
        # a netcdf corresponding to msg_netcdficare
        # is looked for, in the filepath directory.
        yaml_file = 'msg_netcdficare'
        myfiles = find_files_and_readers(
            base_dir=self.filepath, start_time=datetime(2024, 6, 28, 10, 0),
            end_time=datetime(2024, 6, 28, 10, 0), reader=yaml_file)
        # logger.info("Found myfiles = ", myfiles)
        # {'msg_netcdficare': ['/tmp/Mmultic3kmNC4_msg03_202406281000.nc']}

        self.scn = Scene(filenames=myfiles, reader=yaml_file)
        # initMsgHrv()

    def initHima(self, tmp_path) :
        """
        A fake netcdf is built.
        A scene is built with the reader to be tested, applied to this netcdf.
        Called by test_hima_netcdficare().
        """
        self.netcdfName = tmp_path / "Jmultic2kmNC4_hima09_202406281000.nc"
        self.filepath = tmp_path

        listVisible = {
            "VIS004", "VIS005", "VIS006", "VIS008", "IR_016", "IR_022"
            }
        listIR = {
            "IR_038", "WV_062", "WV_069", "WV_073", "IR_085", "IR_096",
            "IR_104", "IR_112", "IR_123", "IR_132"
            }
        self.buildNetcdf(
            self.netcdfName, 5500, "hima09", x0=-5500000, dx=2000.0000047,
            y0=5500000, cfac=20466275., coff=2750.5,
            listVisible=listVisible, listIR=listIR,
            coordinateSystemName="GeosCoordinateSystem2km",
            nomImageNavigation="ImageNavigation2km",
            nomX="X2km", nomY="Y2km",
            time_coverage_end="2024-06-28T10:08:41Z365")

        # We will check that the parameters written in the dummy netcdf
        # can be read.
        self.expectedStartTime = "2024-06-28T10:00:09"
        self.expectedEndTime = "2024-06-28T10:08:41"
        actualAltitude = 35786691 + 6378169        # 42164860.0
        actualLongitude = 0.1
        self.expectedPlatform = "Himawari-9"
        self.expectedSensor = "ahi"
        self.expectedAltitude = actualAltitude
        self.expectedLongitude = actualLongitude
        self.expectedCfac = 20466275.
        self.expectedLfac = 20466275.
        self.expectedCoff = 2750.5
        self.expectedLoff = 2750.5
        self.expectedResolution = 2000.
        self.expectedNblin = 5500
        self.expectedNbpix = 5500

        # To build a scene at the date 20240628_100000,
        # a netcdf corresponding to msg_netcdficare
        # is looked for in the filepath directory.
        yaml_file = 'hima_netcdficare'
        myfiles = find_files_and_readers(
            base_dir=self.filepath, start_time=datetime(2024, 6, 28, 10, 0),
            end_time=datetime(2024, 6, 28, 10, 0), reader=yaml_file)
        # logger.info("Found myfiles = ", myfiles)
        # {'msg_netcdficare': ['/tmp/Mmultic3kmNC4_msg03_202406281000.nc']}

        self.scn = Scene(filenames=myfiles, reader=yaml_file)
        # initHima()

    def initMtg(self, tmp_path) :
        """
        A fake netcdf is built.
        A scene is built with the reader to be tested, applied to this netcdf.
        Called by test_mtg_netcdficare().
        """
        self.netcdfName = tmp_path / "Mmultic1kmNC4_mtgi1_202406281000.nc"
        self.filepath = tmp_path

        # self.buildNetcdf(
        #    self.netcdfName, 5568, "mtgi1",
        #    x0 = -5568000, dx = 2000.0000047, y0 = 5568000, cfac = 13642337, coff = 2784.5,
        #    listVisible= {"VIS004", "VIS005", "VIS006", "VIS008", "VIS009",
        #        "IR_013", "IR_016", "IR_022"},
        #    listIR={"IR_038", "WV_062", "WV_073", "IR_087",
        #        "IR_097", "IR_105", "IR_123", "IR_133"},
        #    coordinateSystemName = "GeosCoordinateSystem2km",
        #    nomImageNavigation = "ImageNavigation2km",
        #    nomX = "X2km", nomY = "Y2km",
        #    time_coverage_end = "2024-06-28T10:08:41Z365")

        listVisible = {
            "VIS004", "VIS005", "VIS006", "VIS008",
            "VIS009", "IR_013", "IR_016", "IR_022"
            }

        self.buildNetcdf(
            self.netcdfName, 11136, "mtgi1",
            x0=-5568000, dx=1000.0000047, y0=5568000,
            cfac=4.093316350596011E7, coff=5568.5,
            listVisible=listVisible, listIR={"IR_038", "IR_105"},
            coordinateSystemName="GeosCoordinateSystem1km",
            nomImageNavigation="ImageNavigation1km",
            nomX="X1km", nomY="Y1km", time_coverage_end="2024-06-28T10:08:41Z365")

        # We will check that the parameters written in the dummy netcdf
        # can be read.
        self.expectedStartTime = "2024-06-28T10:00:09"
        self.expectedEndTime = "2024-06-28T10:08:41"
        actualAltitude = 35786691 + 6378169        # 42164860.0
        actualLongitude = 0.1
        self.expectedPlatform = "Meteosat-12"
        self.expectedSensor = "fci"
        self.expectedAltitude = actualAltitude
        self.expectedLongitude = actualLongitude
        self.expectedCfac = 4.093316350596011E7    # 13642337.
        self.expectedLfac = 4.093316350596011E7    # 13642337.
        self.expectedCoff = 5568.5    # 2784.5
        self.expectedLoff = 5568.5    # 2784.5
        self.expectedResolution = 1000.
        self.expectedNblin = 11136    # 5568
        self.expectedNbpix = 11136    # 5568

        # To build a scene at the date 20240628_100000,
        # a netcdf corresponding to msg_netcdficare
        # is looked for in the filepath directory.
        yaml_file = 'mtg_netcdficare'
        myfiles = find_files_and_readers(
            base_dir=self.filepath, start_time=datetime(2024, 6, 28, 10, 0),
            end_time=datetime(2024, 6, 28, 10, 0), reader=yaml_file)
        # logger.info("Found myfiles = ", myfiles)
        # {'msg_netcdficare': ['/tmp/Mmultic3kmNC4_msg03_202406281000.nc']}

        self.scn = Scene(filenames=myfiles, reader=yaml_file)
        # initMtg()

    def initGoesr(self, tmp_path) :
        """
        A fake netcdf is built.
        A scene is built with the reader to be tested, applied to this netcdf.
        Called by test_goesr_netcdficare().
        """
        self.netcdfName = tmp_path / "Emultic2kmNC4_goes16_202406281000.nc"
        self.filepath = tmp_path

        listVisible = {
            "VIS_004", "VIS_006", "VIS_008", "VIS_014", "VIS_016", "VIS_022"
            }
        listIR = {
            "IR_039", "IR_062", "IR_069", "IR_073", "IR_085",
            "IR_096", "IR_103", "IR_114", "IR_123", "IR_133"
            }
        self.buildNetcdf(
            self.netcdfName, 5424, "goes16",
            x0=-5434894.8, dx=2004.017288, y0=5434894.8,
            cfac=20425338.9, coff=2712.5,
            listVisible=listVisible, listIR=listIR,
            coordinateSystemName="GeosCoordinateSystem2km",
            nomImageNavigation="ImageNavigation2km",
            nomX="X2km", nomY="Y2km",
            time_coverage_end="2024-06-28T10:08:41.1Z")

        # We will check that the parameters written in the dummy netcdf
        # can be read.
        self.expectedStartTime = "2024-06-28T10:00:09"
        self.expectedEndTime = "2024-06-28T10:08:41"
        actualAltitude = 35786691 + 6378169        # 42164860.0
        actualLongitude = 0.1
        self.expectedPlatform = "GOES-16"
        self.expectedSensor = "abi"
        self.expectedAltitude = actualAltitude
        self.expectedLongitude = actualLongitude
        self.expectedCfac = 20425338.9
        self.expectedLfac = 20425338.9
        self.expectedCoff = 2712.5
        self.expectedLoff = 2712.5
        self.expectedResolution = 2000.
        self.expectedNblin = 5424
        self.expectedNbpix = 5424

        # To build a scene at the date 20240628_100000,
        # a netcdf corresponding to msg_netcdficare
        # is looked for in the filepath directory.
        yaml_file = 'goesr_netcdficare'
        myfiles = find_files_and_readers(
            base_dir=self.filepath, start_time=datetime(2024, 6, 28, 10, 0),
            end_time=datetime(2024, 6, 28, 10, 0), reader=yaml_file)

        self.scn = Scene(filenames=myfiles, reader=yaml_file)
        # initGoesr()

    def buildNetcdf(
                self, ncName, nbpix=3712, nomSatellite="msg03",
                x0=-5570254, dx=3000.40604, y0=5570254,
                cfac=1.3642337E7, coff=1857.0,
                listVisible={}, listIR={},
                coordinateSystemName="GeosCoordinateSystem",
                nomImageNavigation="ImageNavigation",
                nomX="X", nomY="Y",
                time_coverage_end="2024-06-28T10:12:41Z365") :
        """
        ncName : tmp_path / Mmultic3kmNC4_msg03_202406281000.nc
        A dummy icare Meteo France netcdf is built here.
        Called by initMsg...
        listVisible = {"VIS006", "VIS008", "IR_016"}
        listIR = {"IR_039", "WV_062", "WV_073", "IR_087", "IR_097", "IR_108",
            "IR_120", "IR_134"}
        """
        if os.path.exists(ncName) :
            os.remove(ncName)
        ncfileOut = Dataset(
            ncName, mode="w", clobber=True,
            diskless=False, persist=False, format='NETCDF4')

        ncfileOut.createDimension(u'ny', nbpix)
        ncfileOut.createDimension(u'nx', nbpix)
        ncfileOut.createDimension(u'numerical_count', 65536)
        ncfileOut.setncattr("time_coverage_start", "2024-06-28T10:00:09Z383")
        ncfileOut.setncattr("time_coverage_end", time_coverage_end)
        ncfileOut.setncattr("Area_of_acquisition", "globe")

        fill_value = -32768
        var = ncfileOut.createVariable(
            "satellite", "c", zlib=True, complevel=4,
            shuffle=True, fletcher32=False, contiguous=False,
            chunksizes=None, endian='native', least_significant_digit=None)

        var.setncattr("id", nomSatellite)
        var.setncattr("dst", 35786691.)
        var.setncattr("lon", float(0.1))

        var = ncfileOut.createVariable(
            "geos", "c", zlib=True, complevel=4, shuffle=True,
            fletcher32=False, contiguous=False, chunksizes=None,
            endian='native', least_significant_digit=None)
        var.setncattr("longitude_of_projection_origin", 0.)

        var = ncfileOut.createVariable(
            coordinateSystemName, "c", zlib=True, complevel=4,
            shuffle=True, fletcher32=False, contiguous=False,
            chunksizes=None, endian='native', least_significant_digit=None)

        stringGeotransform = str(x0) + ", " + str(dx) + ", 0, "
        stringGeotransform += str(-x0) + ", 0, " + str(-dx)
        # -5570254, 3000.40604, 0, 5570254, 0, -3000.40604
        var.setncattr("GeoTransform", stringGeotransform)

        var = ncfileOut.createVariable(
            nomImageNavigation, "c", zlib=True, complevel=4,
            shuffle=True, fletcher32=False, contiguous=False,
            chunksizes=None, endian='native', least_significant_digit=None)
        var.setncattr("CFAC", cfac)
        var.setncattr("LFAC", cfac)
        var.setncattr("COFF", coff)
        var.setncattr("LOFF", coff)

        var = ncfileOut.createVariable(
            nomX, 'float32', u'nx', zlib=True, complevel=4,
            shuffle=True, fletcher32=False, contiguous=False,
            chunksizes=None, endian='native', least_significant_digit=None)
        var[:] = np.array(([(x0 + dx * i) for i in range(nbpix)]))

        var = ncfileOut.createVariable(
            nomY, 'float32', u'ny', zlib=True, complevel=4,
            shuffle=True, fletcher32=False, contiguous=False,
            chunksizes=None, endian='native', least_significant_digit=None)
        y0 = -x0
        dy = -dx
        var[:] = np.array(([(y0 + dy * i) for i in range(nbpix)]))

        self.visibleChannelsCreation(
            ncfileOut, fill_value, listVisible, nbpix, coordinateSystemName)
        self.infrarougeChannelsCreation(
            ncfileOut, fill_value, listIR, nbpix, coordinateSystemName)
        ncfileOut.close
        # buildNetcdf()

    def visibleChannelsCreation(
            self, ncfileOut, fill_value, listVisible, nbpix, coordinateSystemName) :

        for channel in listVisible :
            var = ncfileOut.createVariable(
                channel, 'short', ('ny', 'nx'), zlib=True, complevel=4,
                shuffle=True, fletcher32=False, contiguous=False,
                chunksizes=None, endian='native',
                least_significant_digit=None, fill_value=fill_value)
            var[:] = np.array(
                ([[i * 2 for i in range(nbpix)] for j in range(nbpix)]))
            # Hundredths of albedo between 0 and 10000.
            var.setncattr("scale_factor", 0.01)
            var.setncattr("add_offset", 0.)
            var.setncattr("bandfactor", 20.76)
            var.setncattr("_CoordinateSystems", coordinateSystemName)

            var = ncfileOut.createVariable(
                "Albedo_to_Native_count_" + channel, 'short',
                'numerical_count', zlib=True, complevel=4, shuffle=True,
                fletcher32=False, contiguous=False, chunksizes=None,
                endian='native', least_significant_digit=None,
                fill_value=-9999)
            var[:] = np.array(([-9999 for i in range(65536)]))
            # In order to come back to the native datas on 10, 12 or 16 bits.

    def infrarougeChannelsCreation(
            self, ncfileOut, fill_value, listIR, nbpix, coordinateSystemName) :

        for channel in listIR :
            var = ncfileOut.createVariable(
                channel, 'short', ('ny', 'nx'), zlib=True, complevel=4,
                shuffle=True, fletcher32=False, contiguous=False,
                chunksizes=None, endian='native',
                least_significant_digit=None, fill_value=fill_value)
            var[:] = np.array(
                    ([[-9000 + j * 4 for i in range(nbpix)] for j in range(nbpix)])
                    )
            # Hundredths of celcius degrees.
            var.setncattr("scale_factor", 0.01)
            var.setncattr("add_offset", 273.15)
            var.setncattr("nuc", 1600.548)
            var.setncattr("alpha", 0.9963)
            var.setncattr("beta", 2.185)
            var.setncattr("_CoordinateSystems", coordinateSystemName)

            var = ncfileOut.createVariable(
                "Temp_to_Native_count_" + channel, 'short',
                'numerical_count', zlib=True, complevel=4, shuffle=True,
                fletcher32=False, contiguous=False, chunksizes=None,
                endian='native', least_significant_digit=None,
                fill_value=-9999)
            var[:] = np.array(([-9999 for i in range(65536)]))
            # In order to come back to the native datas on 10, 12 or 16 bits.

    # class TestGeosNetcdfIcareReader.
