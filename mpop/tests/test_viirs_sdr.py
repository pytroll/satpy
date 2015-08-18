#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module for testing the mpop.channel module.
"""

import unittest
import mock
from datetime import datetime
import numpy as np


class FakeHDF5MetaData(object):
    def __init__(self, filename, **kwargs):
        self.filename = filename
        self.d = {
            "AggregateBeginningDate": "20150101",
            "AggregateBeginningTime": "101112.5Z",
            "AggregateEndingDate": "20150102",
            "AggregateEndingTime": "111210.6Z",
            "G-Ring_Longitude": np.array([0.0, 0.1, 0.2, 0.3]),
            "G-Ring_Latitude": np.array([0.0, 0.1, 0.2, 0.3]),
            "AggregateBeginningOrbitNumber": "5",
            "AggregateEndingOrbitNumber": "6",
            "Instrument_Short_Name": "VIIRS",
            "Platform_Short_Name": "NPP",
            "N_Geo_Ref": "test_geo.h5",
            # Data:
            "Radiance": np.arange(10.0 * 300, dtype=np.float32).reshape((10, 300)),
            "Radiance/shape": (10, 300),
            "RadianceFactors": np.array([2.0, 1.0], dtype=np.float32),
            "Reflectance": np.arange(10.0 * 300, dtype=np.float32).reshape((10, 300)),
            "Reflectance/shape": (10, 300),
            "ReflectanceFactors": np.array([2.0, 1.0], dtype=np.float32),
            "BT": np.arange(10.0 * 300, dtype=np.float32).reshape((10, 300)),
            "BT/shape": (10, 300),
            "BTFactors": np.array([2.0, 1.0], dtype=np.float32),
            "FakeData": np.arange(10.0 * 300, dtype=np.float32).reshape((10, 300)),
            "FakeData/shape": (10, 300),
            "Longitude": np.arange(10.0 * 300, dtype=np.float32).reshape((10, 300)),
            "Latitude": np.arange(10.0 * 300, dtype=np.float32).reshape((10, 300)),
        }

    def __getitem__(self, item):
        return self.d[item]


class TestSDRFileReader(unittest.TestCase):
    """Test the SDRFileReader class used by the VIIRS SDR Reader.
    """
    def setUp(self):
        from mpop.readers.viirs_sdr import FileKey
        file_keys = [
            FileKey("beginning_date", "AggregateBeginningDate"),
            FileKey("beginning_time", "AggregateBeginningTime"),
            FileKey("ending_date", "AggregateEndingDate"),
            FileKey("ending_time", "AggregateEndingTime"),
            FileKey("gring_longitude", "G-Ring_Longitude"),
            FileKey("gring_latitude", "G-Ring_Latitude"),
            FileKey("beginning_orbit_number", "AggregateBeginningOrbitNumber"),
            FileKey("ending_orbit_number", "AggregateEndingOrbitNumber"),
            FileKey("instrument_short_name", "Instrument_Short_Name"),
            FileKey("platform_short_name", "Platform_Short_Name"),
            FileKey("geo_file_reference", "N_Geo_Ref"),
            FileKey("radiance", "Radiance", scaling_factors="radiance_factors", units="W m-2 sr-1"),
            FileKey("radiance_factors", "RadianceFactors"),
            FileKey("reflectance", "Reflectance", scaling_factors="reflectance_factors", units="%"),
            FileKey("reflectance_factors", "ReflectanceFactors"),
            FileKey("brightness_temperature", "BT", scaling_factors="bt_factors", units="K"),
            FileKey("bt_factors", "BTFactors"),
            FileKey("unknown_data", "FakeData"),
            FileKey("longitude", "Longitude"),
            FileKey("latitude", "Latitude"),
        ]
        self.file_keys = dict((x.name, x) for x in file_keys)

    def test_init_basic(self):
        from mpop.readers.viirs_sdr import SDRFileReader
        patcher = mock.patch.object(SDRFileReader, '__bases__', (FakeHDF5MetaData,))
        with patcher:
            patcher.is_local = True
            file_reader = SDRFileReader("fake_file_type", "test.h5", file_keys=self.file_keys)
            self.assertEqual(file_reader.start_time, datetime(2015, 1, 1, 10, 11, 12, 500000))
            self.assertEqual(file_reader.end_time, datetime(2015, 1, 2, 11, 12, 10, 600000))
            self.assertRaises(ValueError, file_reader._parse_npp_datetime, "19580102", "120000.0Z")

    def test_get_funcs(self):
        from mpop.readers.viirs_sdr import SDRFileReader
        patcher = mock.patch.object(SDRFileReader, '__bases__', (FakeHDF5MetaData,))
        with patcher:
            patcher.is_local = True
            file_reader = SDRFileReader("fake_file_type", "test.h5", file_keys=self.file_keys)
            gring_lon, gring_lat = file_reader.get_ring_lonlats()
            begin_orbit = file_reader.get_begin_orbit_number()
            self.assertIsInstance(begin_orbit, int)
            end_orbit = file_reader.get_end_orbit_number()
            self.assertIsInstance(end_orbit, int)
            instrument_name = file_reader.get_sensor_name()
            self.assertEqual(instrument_name, "VIIRS")
            platform_name = file_reader.get_platform_name()
            self.assertEqual(platform_name, "NPP")
            geo_ref = file_reader.get_geofilename()

    def test_data_shape(self):
        from mpop.readers.viirs_sdr import SDRFileReader
        patcher = mock.patch.object(SDRFileReader, '__bases__', (FakeHDF5MetaData,))
        with patcher:
            patcher.is_local = True
            file_reader = SDRFileReader("fake_file_type", "test.h5", file_keys=self.file_keys)
            self.assertEquals(file_reader["reflectance/shape"], (10, 300))

    def test_file_units(self):
        from mpop.readers.viirs_sdr import SDRFileReader
        patcher = mock.patch.object(SDRFileReader, '__bases__', (FakeHDF5MetaData,))
        with patcher:
            patcher.is_local = True
            file_reader = SDRFileReader("fake_file_type", "test.h5", file_keys=self.file_keys)
            file_units = file_reader.get_file_units("reflectance")
            self.assertEqual(file_units, "fraction")
            file_units = file_reader.get_file_units("radiance")
            self.assertEqual(file_units, "W cm-2 sr-1")
            file_units = file_reader.get_file_units("brightness_temperature")
            self.assertEqual(file_units, "K")
            file_units = file_reader.get_file_units("unknown_data")
            self.assertIs(file_units, None)
            file_units = file_reader.get_file_units("longitude")
            self.assertEqual(file_units, "degrees")
            file_units = file_reader.get_file_units("latitude")
            self.assertEqual(file_units, "degrees")



class OldTestViirsSDRReader(object):
    """Class for testing the VIIRS SDR reader class.
    """
    
    def test_get_swath_segment(self):
        """
        Test choosing swath segments based on datatime interval
        """
        
        filenames = [
            "SVM15_npp_d20130312_t1034305_e1035546_b07108_c20130312110058559507_cspp_dev.h5", 
            "SVM15_npp_d20130312_t1035559_e1037201_b07108_c20130312110449303310_cspp_dev.h5",
            "SVM15_npp_d20130312_t1037213_e1038455_b07108_c20130312110755391459_cspp_dev.h5",
            "SVM15_npp_d20130312_t1038467_e1040109_b07108_c20130312111106961103_cspp_dev.h5",
            "SVM15_npp_d20130312_t1040121_e1041363_b07108_c20130312111425464510_cspp_dev.h5",
            "SVM15_npp_d20130312_t1041375_e1043017_b07108_c20130312111720550253_cspp_dev.h5",
            "SVM15_npp_d20130312_t1043029_e1044271_b07108_c20130312112246726129_cspp_dev.h5",
            "SVM15_npp_d20130312_t1044283_e1045525_b07108_c20130312113037160389_cspp_dev.h5",
            "SVM15_npp_d20130312_t1045537_e1047179_b07108_c20130312114330237590_cspp_dev.h5",
            "SVM15_npp_d20130312_t1047191_e1048433_b07108_c20130312120148075096_cspp_dev.h5",
            "SVM15_npp_d20130312_t1048445_e1050070_b07108_c20130312120745231147_cspp_dev.h5",
            ]
       

        #
        # Test search for multiple granules
        result = [
            "SVM15_npp_d20130312_t1038467_e1040109_b07108_c20130312111106961103_cspp_dev.h5",
            "SVM15_npp_d20130312_t1040121_e1041363_b07108_c20130312111425464510_cspp_dev.h5",
            "SVM15_npp_d20130312_t1041375_e1043017_b07108_c20130312111720550253_cspp_dev.h5",
            "SVM15_npp_d20130312_t1043029_e1044271_b07108_c20130312112246726129_cspp_dev.h5",
            "SVM15_npp_d20130312_t1044283_e1045525_b07108_c20130312113037160389_cspp_dev.h5",
            ]

        tstart = datetime(2013, 3, 12, 10, 39)
        tend = datetime(2013, 3, 12, 10, 45)

        

        sublist = _get_swathsegment(filenames, tstart, tend)

        self.assert_(sublist == result)


        #
        # Test search for single granule
        tslot = datetime(2013, 3, 12, 10, 45)

        result_file = [
            "SVM15_npp_d20130312_t1044283_e1045525_b07108_c20130312113037160389_cspp_dev.h5",
            ]
        
        single_file = _get_swathsegment(filenames, tslot)

        self.assert_(result_file == single_file)


def suite():
    """The test suite for test_viirs_sdr.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestSDRFileReader))
    
    return mysuite
