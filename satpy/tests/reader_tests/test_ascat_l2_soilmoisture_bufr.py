#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2021 Satpy developers
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
"""Unittesting the ASCAT SCATTEROMETER SOIL MOISTURE BUFR reader."""

import os
import sys
import unittest
from datetime import datetime

import numpy as np

# TDB: this test is based on test_seviri_l2_bufr.py and test_iasi_l2.py

# This is a test for ASCAT SoilMoisture product message, take from a real
# bufr file distributed over EUMETCAST


def create_message():
    """Create fake message for testing."""
    nlat = 10
    nlon = 10
    samples = nlat*nlon
    lat, lon = np.meshgrid(np.linspace(63, 65, nlat), np.linspace(-30, -20, nlon))
    lat = np.round(np.ravel(lat), 4)
    lon = np.round(np.ravel(lon), 4)
    rstate = np.random.RandomState(0)
    surfaceSoilMoisture = np.round(rstate.rand(samples)*100, 1)
    surfaceSoilMoisture[0] = -1e+100
    retmsg = {
        'inputDelayedDescriptorReplicationFactor': [8],
        'edition': 4,
        'masterTableNumber': 0,
        'bufrHeaderCentre': 254,
        'bufrHeaderSubCentre': 0,
        'updateSequenceNumber': 0,
        'dataCategory': 12,
        'internationalDataSubCategory': 255,
        'dataSubCategory': 190,
        'masterTablesVersionNumber': 13,
        'localTablesVersionNumber': 0,
        'typicalYear': 2020,
        'typicalMonth': 12,
        'typicalDay': 21,
        'typicalHour': 9,
        'typicalMinute': 33,
        'typicalSecond': 0,
        'numberOfSubsets': samples,
        'observedData': 1,
        'compressedData': 1,
        'unexpandedDescriptors': 312061,
        'centre': 254,
        'subCentre': 0,
        '#1#softwareIdentification': 1000,
        'satelliteIdentifier': 4,
        'satelliteInstruments': 190,
        'year': 2020,
        'month': 12,
        'day': 21,
        'hour': 9,
        'minute': 33,
        'second': np.linspace(0, 59, samples),
        'latitude': lat,
        'longitude': lon,
        'surfaceSoilMoisture': surfaceSoilMoisture,
        'soilMoistureQuality': np.zeros(samples),
    }
    return retmsg


MSG = create_message()

# the notional filename that would contain the above test message data
FILENAME = 'W_XX-EUMETSAT-TEST,SOUNDING+SATELLITE,METOPA+ASCAT_C_EUMC_20201221093300_73545_eps_o_125_ssm_l2.bin'
# the information that would be extracted from the above filename according to the pattern in the .yaml
FILENAME_INFO = {
    'reception_location': 'TEST',
    'platform': 'METOPA',
    'instrument': 'ASCAT',
    'start_time': '20201221093300',
    'perigee': '73545',
    'species': '125_ssm',
    'level': 'l2'
}

# file type info for the above file that is defined in the .yaml
FILETYPE_INFO = {
    'file_type': 'ascat_l2_soilmoisture_bufr',
    'file_reader': 'AscatSoilMoistureBufr'
}


def save_test_data(path):
    """Save the test file to the indicated directory."""
    import eccodes as ec
    filepath = os.path.join(path, FILENAME)
    with open(filepath, "wb") as f:
        for m in [MSG]:
            buf = ec.codes_bufr_new_from_samples('BUFR4_local_satellite')
            for key in m:
                val = m[key]
                if np.isscalar(val):
                    ec.codes_set(buf, key, val)
                else:
                    ec.codes_set_array(buf, key, val)
            ec.codes_set(buf, 'pack', 1)
            ec.codes_write(buf, f)
            ec.codes_release(buf)
    return filepath


class TesitAscatL2SoilmoistureBufr(unittest.TestCase):
    """Test ASCAT Soil Mosture loader."""

    def setUp(self):
        """Create temporary file to perform tests with."""
        import tempfile

        from satpy.readers.ascat_l2_soilmoisture_bufr import AscatSoilMoistureBufr
        self.base_dir = tempfile.mkdtemp()
        self.fname = save_test_data(self.base_dir)
        self.fname_info = FILENAME_INFO
        self.ftype_info = FILETYPE_INFO
        self.reader = AscatSoilMoistureBufr(self.fname, self.fname_info, self.ftype_info)

    def tearDown(self):
        """Remove the temporary directory created for a test."""
        try:
            import shutil
            shutil.rmtree(self.base_dir, ignore_errors=True)
        except OSError:
            pass

    @unittest.skipIf(sys.platform.startswith('win'), "'eccodes' not supported on Windows")
    def test_scene(self):
        """Test scene creation."""
        from satpy import Scene
        fname = os.path.join(self.base_dir, FILENAME)
        scn = Scene(reader='ascat_l2_soilmoisture_bufr', filenames=[fname])
        self.assertTrue('scatterometer' in scn.sensor_names)
        self.assertTrue(datetime(2020, 12, 21, 9, 33, 0) == scn.start_time)
        self.assertTrue(datetime(2020, 12, 21, 9, 33, 59) == scn.end_time)

    @unittest.skipIf(sys.platform.startswith('win'), "'eccodes' not supported on Windows")
    def test_scene_load_available_datasets(self):
        """Test that all datasets are available."""
        from satpy import Scene
        fname = os.path.join(self.base_dir, FILENAME)
        scn = Scene(reader='ascat_l2_soilmoisture_bufr', filenames=[fname])
        self.assertTrue('surface_soil_moisture' in scn.available_dataset_names())
        scn.load(scn.available_dataset_names())
        loaded = [dataset.name for dataset in scn]
        self.assertTrue(sorted(loaded) == sorted(scn.available_dataset_names()))

    @unittest.skipIf(sys.platform.startswith('win'), "'eccodes' not supported on Windows")
    def test_scene_dataset_values(self):
        """Test loading data."""
        from satpy import Scene
        fname = os.path.join(self.base_dir, FILENAME)
        scn = Scene(reader='ascat_l2_soilmoisture_bufr', filenames=[fname])
        for name in scn.available_dataset_names():
            scn.load([name])
            loaded_values = scn[name].values
            fill_value = scn[name].attrs['fill_value']
            # replace nans in data loaded from file with the fill value defined in the .yaml
            # to make them comparable
            loaded_values_nan_filled = np.nan_to_num(loaded_values, nan=fill_value)
            key = scn[name].attrs['key']
            original_values = MSG[key]
            # this makes each assertion below a separate test from unittest's point of view
            # (note: if all subtests pass, they will count as one test)
            with self.subTest(msg="Test failed for dataset: "+name):
                self.assertTrue(np.allclose(original_values, loaded_values_nan_filled))
