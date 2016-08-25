#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module for testing the satpy.readers.viirs_sdr module.
"""

import sys
if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest
import mock
from datetime import datetime
import numpy as np

DEFAULT_FILE_DTYPE = np.uint16
DEFAULT_FILE_SHAPE = (10, 300)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)
DEFAULT_FILE_FACTORS = np.array([2.0, 1.0], dtype=np.float32)
DEFAULT_LAT_DATA = np.linspace(45, 65, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LAT_DATA = np.repeat([DEFAULT_LAT_DATA], DEFAULT_FILE_SHAPE[0], axis=0)
DEFAULT_LON_DATA = np.linspace(5, 45, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LON_DATA = np.repeat([DEFAULT_LON_DATA], DEFAULT_FILE_SHAPE[0], axis=0)


def _setup_file_keys():
    from satpy.readers.viirs_sdr import FileKey
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
        FileKey("geo_file_reference", "N_GEO_Ref"),
        FileKey("radiance", "Radiance", factor="radiance_factors", units="W m-2 sr-1"),
        FileKey("radiance_factors", "RadianceFactors"),
        FileKey("reflectance", "Reflectance", factor="reflectance_factors", units="%"),
        FileKey("reflectance_factors", "ReflectanceFactors"),
        FileKey("brightness_temperature", "BrightnessTemperature", factor="bt_factors", units="K"),
        FileKey("bt_factors", "BrightnessTemperatureFactors"),
        FileKey("unknown_data", "FakeData", factor="bad_factors"),
        FileKey("unknown_data2", "FakeData", file_units="fake", dtype="int64"),
        FileKey("unknown_data3", "FakeDataFloat", factor="nonexistent"),
        FileKey("bad_factors", "BadFactors"),
        FileKey("longitude", "Longitude"),
        FileKey("latitude", "Latitude"),
    ]
    return dict((x.name, x) for x in file_keys)


class FakeHDF5MetaData(object):
    def __init__(self, filename, **kwargs):
        self.filename = filename
        offset = kwargs.pop("offset", 0)
        begin_times = ["10{0:02d}12.5Z".format(x) for x in range(5)]
        end_times = ["11{0:02d}10.6Z".format(x) for x in range(5)]

        self.d = {
            "AggregateBeginningDate": "20150101",
            "AggregateBeginningTime": begin_times[offset],
            "AggregateEndingDate": "20150101",
            "AggregateEndingTime": end_times[offset],
            "G-Ring_Longitude": np.array([0.0, 0.1, 0.2, 0.3]),
            "G-Ring_Latitude": np.array([0.0, 0.1, 0.2, 0.3]),
            "AggregateBeginningOrbitNumber": "{0:d}".format(offset),
            "AggregateEndingOrbitNumber": "{0:d}".format(offset + 1),
            "Instrument_Short_Name": "VIIRS",
            "Platform_Short_Name": "NPP",
            "N_GEO_Ref": "GITCO_npp_d20120225_t{0:02d}07061_e2359590_b01708_c20120226002502222157_noaa_ops.h5".format(offset),
            "BadFactors": np.array([-999.0, -999.0, -999.0, -999.0], dtype=np.float32),
        }

        for k in ["Radiance", "Reflectance", "BrightnessTemperature", "FakeDataset"]:
            self.d[k] = DEFAULT_FILE_DATA.copy()
            self.d[k + "/shape"] = DEFAULT_FILE_SHAPE
            self.d[k + "Factors"] = DEFAULT_FILE_FACTORS.copy()
        for k in ["Latitude"]:
            self.d[k] = np.linspace(45, 65, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
            self.d[k] = np.repeat([self.d[k]], DEFAULT_FILE_SHAPE[0], axis=0)
            self.d[k + "/shape"] = DEFAULT_FILE_SHAPE
        for k in ["Longitude"]:
            self.d[k] = np.linspace(5, 45, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
            self.d[k] = np.repeat([self.d[k]], DEFAULT_FILE_SHAPE[0], axis=0)
            self.d[k + "/shape"] = DEFAULT_FILE_SHAPE
        for k in ["FakeData"]:
            self.d[k] = DEFAULT_FILE_DATA.copy()
            self.d[k + "/shape"] = DEFAULT_FILE_SHAPE
        for k in ["FakeDataFloat"]:
            self.d[k] = DEFAULT_FILE_DATA.copy().astype(np.float32)
            self.d[k + "/shape"] = DEFAULT_FILE_SHAPE

        self.d.update(kwargs)

    def __getitem__(self, item):
        key = item.rsplit("/", 1)
        if key[-1] == "shape":
            key = key[0].rsplit("/", 1)[-1] + "/shape"
        else:
            key = key[-1]
        return self.d[key]


class TestHDF5MetaData(unittest.TestCase):
    def test_init_doesnt_exist(self):
        from satpy.readers.viirs_sdr import HDF5FileHandler
        self.assertRaises(IOError, HDF5FileHandler, "test_asdflkajsd.h5")

    @mock.patch("h5py.File")
    @mock.patch("os.path.exists")
    def test_init_basic(self, os_exists_mock, h5py_file_mock):
        import h5py
        from satpy.readers.viirs_sdr import HDF5FileHandler
        os_exists_mock.return_value = True
        f_handle = h5py_file_mock.return_value
        f_handle.attrs = {
            "test_int": np.array([1]),
            "test_str": np.array("VIIRS"),
            "test_arr": np.arange(5),
        }
        # f_handle.visititems.side_effect = lambda f: f()
        h = HDF5FileHandler("fake.h5")
        self.assertTrue(h5py_file_mock.called)
        self.assertTrue(f_handle.visititems.called)
        self.assertTrue(f_handle.close.called)
        self.assertEqual(f_handle.visititems.call_args, ((h.collect_metadata,),))

    @mock.patch("h5py.File")
    @mock.patch("os.path.exists")
    def test_collect_metadata(self, os_exists_mock, h5py_file_mock):
        import h5py
        from satpy.readers.viirs_sdr import HDF5FileHandler
        os_exists_mock.return_value = True
        f_handle = h5py_file_mock.return_value
        h = HDF5FileHandler("fake.h5")
        with mock.patch.object(h, "_collect_attrs") as collect_attrs_patch:
            obj_mock = mock.Mock()
            h.collect_metadata("fake", obj_mock)
            self.assertTrue(collect_attrs_patch.called)
            self.assertEqual(collect_attrs_patch.call_args, (("fake", obj_mock.attrs),))

        with mock.patch.object(h, "_collect_attrs") as collect_attrs_patch:
            obj_mock = mock.Mock(spec=h5py.Dataset)
            obj_mock.shape = DEFAULT_FILE_SHAPE
            h.collect_metadata("fake", obj_mock)
            self.assertTrue(collect_attrs_patch.called)
            self.assertEqual(collect_attrs_patch.call_args, (("fake", obj_mock.attrs),))
            self.assertIn("fake", h.file_content)
            self.assertIn("fake/shape", h.file_content)
            self.assertEqual(h.file_content["fake/shape"], DEFAULT_FILE_SHAPE)

    @mock.patch("h5py.File")
    @mock.patch("os.path.exists")
    def test_getitem(self, os_exists_mock, h5py_file_mock):
        import h5py
        from satpy.readers.viirs_sdr import HDF5FileHandler
        os_exists_mock.return_value = True
        f_handle = h5py_file_mock.return_value
        fake_dataset = f_handle["fake"].value
        h = HDF5FileHandler("fake.h5")
        h.file_content["fake"] = mock.Mock(spec=h5py.Dataset)
        h.file_content["fake_other"] = 5
        data = h["fake"]
        self.assertEqual(data, fake_dataset)
        data = h["fake_other"]
        self.assertEqual(data, 5)


class TestSDRFileReader(unittest.TestCase):
    """Test the SDRFileReader class used by the VIIRS SDR Reader.
    """
    def setUp(self):
        self.file_keys = _setup_file_keys()

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_init_basic(self):
        from satpy.readers.viirs_sdr import SDRFileReader
        file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
        self.assertEqual(file_reader.start_time, datetime(2015, 1, 1, 10, 0, 12, 500000))
        self.assertEqual(file_reader.end_time, datetime(2015, 1, 1, 11, 0, 10, 600000))
        self.assertRaises(ValueError, file_reader._parse_datetime, "19580102", "120000.0Z")

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_get_funcs(self):
        from satpy.readers.viirs_sdr import SDRFileReader
        file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
        gring_lon, gring_lat = file_reader.ring_lonlats
        begin_orbit = file_reader.begin_orbit_number
        self.assertIsInstance(begin_orbit, int)
        self.assertEqual(begin_orbit, 0)
        end_orbit = file_reader.end_orbit_number
        self.assertIsInstance(end_orbit, int)
        self.assertEqual(end_orbit, 1)
        instrument_name = file_reader.sensor_name
        self.assertEqual(instrument_name, "VIIRS")
        platform_name = file_reader.platform_name
        self.assertEqual(platform_name, "NPP")
        geo_ref = file_reader.geofilename

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_data_shape(self):
        from satpy.readers.viirs_sdr import SDRFileReader
        file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
        self.assertEquals(file_reader["reflectance/shape"], (10, 300))

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_file_units(self):
        from satpy.readers.viirs_sdr import SDRFileReader
        file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
        file_units = file_reader.get_file_units("reflectance")
        self.assertEqual(file_units, "1")
        file_units = file_reader.get_file_units("radiance")
        self.assertEqual(file_units, "W cm-2 sr-1")
        file_units = file_reader.get_file_units("brightness_temperature")
        self.assertEqual(file_units, "K")
        file_units = file_reader.get_file_units("unknown_data")
        self.assertIs(file_units, None)
        file_units = file_reader.get_file_units("unknown_data2")
        self.assertIs(file_units, "fake")
        file_units = file_reader.get_file_units("longitude")
        self.assertEqual(file_units, "degrees")
        file_units = file_reader.get_file_units("latitude")
        self.assertEqual(file_units, "degrees")

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_units(self):
        from satpy.readers.viirs_sdr import SDRFileReader
        file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
        units = file_reader.get_units("reflectance")
        self.assertEqual(units, "%")
        units = file_reader.get_units("radiance")
        self.assertEqual(units, "W m-2 sr-1")
        units = file_reader.get_units("brightness_temperature")
        self.assertEqual(units, "K")
        units = file_reader.get_units("unknown_data")
        self.assertIs(units, None)
        units = file_reader.get_units("unknown_data2")
        self.assertIs(units, "fake")
        units = file_reader.get_units("longitude")
        self.assertEqual(units, "degrees")
        units = file_reader.get_units("latitude")
        self.assertEqual(units, "degrees")

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_getting_raw_data(self):
        from satpy.readers.viirs_sdr import SDRFileReader
        file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
        for k in ["radiance", "reflectance", "brightness_temperature",
                  "unknown_data", "unknown_data2"]:
            data = file_reader[k]
            self.assertTrue(data.dtype == DEFAULT_FILE_DTYPE)
            np.testing.assert_array_equal(data, DEFAULT_FILE_DATA)

        for k in ["longitude"]:
            data = file_reader[k]
            self.assertTrue(data.dtype == DEFAULT_FILE_DTYPE)
            np.testing.assert_array_equal(data, DEFAULT_LON_DATA)

        for k in ["latitude"]:
            data = file_reader[k]
            self.assertTrue(data.dtype == DEFAULT_FILE_DTYPE)
            np.testing.assert_array_equal(data, DEFAULT_LAT_DATA)

        for k in ["unknown_data3"]:
            data = file_reader[k]
            self.assertTrue(data.dtype == np.float32)
            np.testing.assert_array_equal(data, DEFAULT_FILE_DATA)

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_get_swath_data_noscale(self):
        from satpy.readers.viirs_sdr import SDRFileReader
        file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
        for k in ["longitude", "latitude"]:
            # these shouldn't have any change to their file data
            # normally unknown_data would, but the scaling factors are bad in this test file
            data, mask = file_reader.get_swath_data(k)
            self.assertTrue(data.dtype == np.float32)
            self.assertTrue(mask.dtype == np.bool)
            if k == "longitude":
                np.testing.assert_array_equal(data, DEFAULT_LON_DATA)
            else:
                np.testing.assert_array_equal(data, DEFAULT_LAT_DATA)

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_get_swath_data_badscale(self):
        from satpy.readers.viirs_sdr import SDRFileReader
        file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
        for k in ["unknown_data"]:
            # these shouldn't have any change to the file data
            # normally there would be, but the scaling factors are bad in this test file so all data is masked
            data, mask = file_reader.get_swath_data(k)
            self.assertTrue(data.dtype == np.float32)
            self.assertTrue(mask.dtype == np.bool)
            np.testing.assert_array_equal(data, DEFAULT_FILE_DATA)
            # bad fill values should result in bad science data
            np.testing.assert_array_equal(mask, True)

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_get_swath_data_scale(self):
        from satpy.readers.viirs_sdr import SDRFileReader
        file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
        valid_data = DEFAULT_FILE_DATA * DEFAULT_FILE_FACTORS[0] + DEFAULT_FILE_FACTORS[1]
        for k in ["radiance", "reflectance", "brightness_temperature"]:
            data, mask = file_reader.get_swath_data(k)
            self.assertTrue(data.dtype == np.float32)
            self.assertTrue(mask.dtype == np.bool)
            if k == "radiance":
                np.testing.assert_array_equal(data, valid_data * 10000.0)
            elif k == "reflectance":
                np.testing.assert_array_equal(data, valid_data * 100.0)
            else:
                np.testing.assert_array_equal(data, valid_data)

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_get_swath_data_noscale_dtype(self):
        """Test that we can get swath data that converts to a non-float output dtype and don't have any scaling.
        """
        from satpy.readers.viirs_sdr import SDRFileReader
        file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
        for k in ["unknown_data2"]:
            data, mask = file_reader.get_swath_data(k)
            self.assertTrue(data.dtype == np.int64)
            self.assertTrue(mask.dtype == np.bool)
            np.testing.assert_array_equal(data, DEFAULT_FILE_DATA)

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_get_swath_data_missing_scale_no_dtype(self):
        """Test that we can get swath data that can have scaling, but doesn't have any in the file and doesn't
        require any conversion for dtype.
        """
        from satpy.readers.viirs_sdr import SDRFileReader
        file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
        for k in ["unknown_data3"]:
            data, mask = file_reader.get_swath_data(k)
            self.assertTrue(data.dtype == np.float32)
            self.assertTrue(mask.dtype == np.bool)
            np.testing.assert_array_equal(data, DEFAULT_FILE_DATA)

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_get_swath_data_inplace(self):
        """Test that we can get most file keys and write the data inplace in arrays provided.
        """
        from satpy.readers.viirs_sdr import SDRFileReader
        file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
        data_out = np.zeros(DEFAULT_FILE_SHAPE, dtype=np.float32)
        mask_out = np.zeros(DEFAULT_FILE_SHAPE, dtype=np.bool)
        for k in ["radiance", "reflectance", "brightness_temperature",
                  "longitude", "latitude", "unknown_data"]:
            data, mask = file_reader.get_swath_data(k, data_out=data_out, mask_out=mask_out)
            self.assertTrue(len(np.nonzero(data != 0)[0]) > 0)
            self.assertTrue(data.dtype == np.float32)
            self.assertTrue(mask.dtype == np.bool)
            data_out[:] = 0
            mask_out[:] = False

        data_out = np.zeros(DEFAULT_FILE_SHAPE, dtype=np.int64)
        for k in ["unknown_data2"]:
            data, mask = file_reader.get_swath_data(k, data_out=data_out, mask_out=mask_out)
            self.assertTrue(len(np.nonzero(data != 0)[0]) > 0)
            self.assertTrue(data.dtype == np.int64)
            self.assertTrue(mask.dtype == np.bool)
            data_out[:] = 0
            mask_out[:] = False


class TestSDRMultiFileReader(unittest.TestCase):
    def setUp(self):
        self.file_keys = _setup_file_keys()

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_init_basic(self):
        from satpy.readers.viirs_sdr import SDRFileReader
        from satpy.readers.viirs_sdr import MultiFileReader
        file_readers = [SDRFileReader("fake_file_type", "test.h5", self.file_keys, offset=x) for x in range(5)]
        file_reader = MultiFileReader("fake_file_type", file_readers, self.file_keys)

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_properties(self):
        from satpy.readers.viirs_sdr import SDRFileReader
        from satpy.readers.viirs_sdr import MultiFileReader
        file_readers = [
            SDRFileReader("fake_file_type", "test{0:02d}.h5".format(x), self.file_keys, offset=x) for x in range(5)]
        file_reader = MultiFileReader("fake_file_type", file_readers, self.file_keys)
        fns = ["test00.h5", "test01.h5", "test02.h5", "test03.h5", "test04.h5"]
        geo_fns = ["GITCO_npp_d20120225_t{0:02d}07061_e2359590_b01708_c20120226002502222157_noaa_ops.h5".format(x) for x in range(5)]
        self.assertListEqual(file_reader.filenames, fns)
        self.assertEqual(file_reader.start_time, datetime(2015, 1, 1, 10, 0, 12, 500000))
        self.assertEqual(file_reader.end_time, datetime(2015, 1, 1, 11, 4, 10, 600000))
        self.assertEqual(file_reader.begin_orbit_number, 0)
        self.assertEqual(file_reader.end_orbit_number, 5)
        self.assertEqual(file_reader.platform_name, "NPP")
        self.assertEqual(file_reader.sensor_name, "VIIRS")
        self.assertListEqual(file_reader.geofilenames, geo_fns)

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_get_units(self):
        from satpy.readers.viirs_sdr import SDRFileReader
        from satpy.readers.viirs_sdr import MultiFileReader
        file_readers = [
            SDRFileReader("fake_file_type", "test{0:02d}.h5".format(x), self.file_keys, offset=x) for x in range(5)]
        file_reader = MultiFileReader("fake_file_type", file_readers, self.file_keys)
        # shouldn't need to thoroughly test this because its tested in the single file reader tests
        self.assertEqual(file_reader.get_units("radiance"), "W m-2 sr-1")

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_get_swath_data(self):
        from satpy.readers.viirs_sdr import SDRFileReader
        from satpy.readers.viirs_sdr import MultiFileReader
        file_readers = [
            SDRFileReader("fake_file_type", "test{0:02d}.h5".format(x), self.file_keys, offset=x) for x in range(5)]
        file_reader = MultiFileReader("fake_file_type", file_readers, self.file_keys)
        data = file_reader.get_swath_data("brightness_temperature")
        # make sure its a masked array
        self.assertTrue(hasattr(data, "mask"))
        self.assertTrue(hasattr(data, "data"))
        valid_shape = (DEFAULT_FILE_SHAPE[0]*5, DEFAULT_FILE_SHAPE[1])
        self.assertEqual(data.shape, valid_shape)
        valid_mask = np.zeros(valid_shape, dtype=np.bool)
        valid_data = np.concatenate(tuple(DEFAULT_FILE_DATA.astype(np.float32) for x in range(5)))
        valid_data = np.ma.masked_array(valid_data, valid_mask) * 2.0 + 1.0
        np.testing.assert_array_equal(data, valid_data)

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_get_swath_data_to_disk(self):
        from satpy.readers.viirs_sdr import SDRFileReader
        from satpy.readers.viirs_sdr import MultiFileReader
        file_readers = [
            SDRFileReader("fake_file_type", "test{0:02d}.h5".format(x), self.file_keys, offset=x) for x in range(5)]
        file_reader = MultiFileReader("fake_file_type", file_readers, self.file_keys)
        self.assertRaises(NotImplementedError, file_reader.get_swath_data, "radiance", filename="test.dat")


class TestVIIRSSDRReader(unittest.TestCase):
    def setUp(self):
        example_filenames = {}
        for file_type in ["SVI01", "SVI04", "SVM01", "GITCO", "GMTCO"]:
            fn_pat = "{0}_npp_d2012022{1:d}_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5"
            fns = [fn_pat.format(file_type, x) for x in range(5)]
            example_filenames[file_type] = fns
        self.example_filenames = example_filenames

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_init_no_args(self):
        from satpy.readers.viirs_sdr import VIIRSSDRReader
        self.assertRaises(ValueError, VIIRSSDRReader)

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_init_only_files_not_matching(self):
        from satpy.readers.viirs_sdr import VIIRSSDRReader
        input_files = ["test_{0:d}.h5".format(x) for x in range(5)]
        self.assertRaises(ValueError, VIIRSSDRReader, filenames=input_files)

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_init_only_files_no_geo_files(self):
        from satpy.readers.viirs_sdr import VIIRSSDRReader
        reader = VIIRSSDRReader(filenames=self.example_filenames["SVI01"])

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_init_start_time_not_matching(self):
        from satpy.readers.viirs_sdr import VIIRSSDRReader
        filenames = self.example_filenames["SVI01"] + self.example_filenames["GITCO"]
        self.assertRaises(IOError, VIIRSSDRReader, filenames=filenames, start_time=datetime(2012, 3, 1))

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_init_start_time_end_time_not_matching(self):
        from satpy.readers.viirs_sdr import VIIRSSDRReader
        filenames = self.example_filenames["SVI01"] + self.example_filenames["GITCO"]
        self.assertRaises(IOError, VIIRSSDRReader, filenames=filenames,
                          start_time=datetime(2012, 3, 1, 11, 0, 0),
                          end_time=datetime(2012, 3, 1, 12, 0, 0))

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_init_start_time_matching(self):
        from satpy.readers.viirs_sdr import VIIRSSDRReader
        filenames = self.example_filenames["SVI01"] + self.example_filenames["GITCO"]
        reader = VIIRSSDRReader(filenames=filenames, start_time=datetime(2015, 1, 1, 11, 0, 0))

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_init_start_time_end_time_matching(self):
        from satpy.readers.viirs_sdr import VIIRSSDRReader
        filenames = self.example_filenames["SVI01"] + self.example_filenames["GITCO"]
        # when files are at the beginning of the range
        reader = VIIRSSDRReader(filenames=filenames,
                          start_time=datetime(2015, 1, 1, 11, 0, 0),
                          end_time=datetime(2015, 1, 1, 12, 0, 0))
        # when files are at the end of the range
        reader = VIIRSSDRReader(filenames=filenames,
                                start_time=datetime(2015, 1, 1, 10, 0, 0),
                                end_time=datetime(2015, 1, 1, 11, 0, 0))
        # when files fall within the whole range
        reader = VIIRSSDRReader(filenames=filenames,
                                start_time=datetime(2015, 1, 1, 9, 0, 0),
                                end_time=datetime(2015, 1, 1, 13, 0, 0))

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_init_varying_number_files(self):
        from satpy.readers.viirs_sdr import VIIRSSDRReader
        filenames = self.example_filenames["SVI01"] + self.example_filenames["GITCO"][1:]
        self.assertRaises(IOError, VIIRSSDRReader, filenames=filenames,
                                start_time=datetime(2015, 1, 1, 11, 0, 0),
                                end_time=datetime(2015, 1, 1, 12, 0, 0))

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_load_unknown_names(self):
        from satpy.readers.viirs_sdr import VIIRSSDRReader
        filenames = self.example_filenames["SVI01"] + self.example_filenames["GITCO"]
        reader = VIIRSSDRReader(filenames=filenames,
                                start_time=datetime(2015, 1, 1, 11, 0, 0),
                                end_time=datetime(2015, 1, 1, 12, 0, 0))
        datasets = reader.load(["fake", "fake2"])
        self.assertEqual(datasets, {})

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_load_basic(self):
        from satpy.readers.viirs_sdr import VIIRSSDRReader
        filenames = self.example_filenames["SVI01"] + \
                    self.example_filenames["SVI04"] + \
                    self.example_filenames["SVM01"] + \
                    self.example_filenames["GITCO"] + \
                    self.example_filenames["GMTCO"]
        reader = VIIRSSDRReader(filenames=filenames,
                                start_time=datetime(2015, 1, 1, 11, 0, 0),
                                end_time=datetime(2015, 1, 1, 12, 0, 0))

        datasets = reader.load(["I01", "I04", "M01"], unused=1, unused2=2)

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_load_navigation_not_found(self):
        from satpy.readers.viirs_sdr import VIIRSSDRReader
        filenames = self.example_filenames["SVI01"] + \
                    self.example_filenames["GMTCO"]
        reader = VIIRSSDRReader(filenames=filenames,
                                start_time=datetime(2015, 1, 1, 11, 0, 0),
                                end_time=datetime(2015, 1, 1, 12, 0, 0))

        with mock.patch.object(reader, "identify_file_types") as ift_mock:
            ift_mock.return_value = {}
            self.assertRaises(RuntimeError, reader.load_navigation, "gitco", "svi01")

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_load_navigation(self):
        from satpy.readers.viirs_sdr import VIIRSSDRReader
        filenames = self.example_filenames["SVI01"] + \
                    self.example_filenames["GMTCO"]
        reader = VIIRSSDRReader(filenames=filenames,
                                start_time=datetime(2015, 1, 1, 11, 0, 0),
                                end_time=datetime(2015, 1, 1, 12, 0, 0))

        area = reader.load_navigation("gitco", dep_file_type="svi01")

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_get_dataset_info_no_cal(self):
        from satpy.readers.viirs_sdr import VIIRSSDRReader
        filenames = self.example_filenames["SVI01"] + \
                    self.example_filenames["GMTCO"]
        reader = VIIRSSDRReader(filenames=filenames,
                                start_time=datetime(2015, 1, 1, 11, 0, 0),
                                end_time=datetime(2015, 1, 1, 12, 0, 0))

        info = reader._get_dataset_info("I01", ["reflectance"])

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_get_dataset_info_no_cal_level(self):
        from satpy.readers.viirs_sdr import VIIRSSDRReader
        filenames = self.example_filenames["SVI01"] + \
                    self.example_filenames["GMTCO"]
        reader = VIIRSSDRReader(filenames=filenames,
                                start_time=datetime(2015, 1, 1, 11, 0, 0),
                                end_time=datetime(2015, 1, 1, 12, 0, 0))

        # the channels calibration level isn't configured
        info = reader._get_dataset_info("I01", calibration=["counts"])

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_get_dataset_info_cal_equals(self):
        from satpy.readers.viirs_sdr import VIIRSSDRReader
        filenames = self.example_filenames["SVI01"] + \
                    self.example_filenames["GMTCO"]
        reader = VIIRSSDRReader(filenames=filenames,
                                start_time=datetime(2015, 1, 1, 11, 0, 0),
                                end_time=datetime(2015, 1, 1, 12, 0, 0))
        reader.datasets["I01"]["calibration"] = ["reflectance", "radiance"]
        reader.datasets["I01"]["file_type"] = ["svi01", "svi01_2"]
        reader.datasets["I01"]["file_key"] = ["reflectance", "radiance"]
        reader.datasets["I01"]["navigation"] = ["gitco", "gitco2"]

        info = reader._get_dataset_info("I01", calibration=["reflectance"])
        self.assertEqual(info["calibration"], "reflectance")
        self.assertEqual(info["file_type"], "svi01")
        self.assertEqual(info["file_key"], "reflectance")
        self.assertEqual(info["navigation"], "gitco")

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_get_dataset_info_cal_found(self):
        from satpy.readers.viirs_sdr import VIIRSSDRReader
        filenames = self.example_filenames["SVI01"] + \
                    self.example_filenames["GMTCO"]
        reader = VIIRSSDRReader(filenames=filenames,
                                start_time=datetime(2015, 1, 1, 11, 0, 0),
                                end_time=datetime(2015, 1, 1, 12, 0, 0))
        reader.datasets["I01"]["calibration"] = ["reflectance", "radiance"]
        reader.datasets["I01"]["file_type"] = ["svi01", "svi01_2"]
        reader.datasets["I01"]["file_key"] = ["reflectance", "radiance"]
        reader.datasets["I01"]["navigation"] = ["gitco", "gitco2"]
        reader.file_readers["svi01_2"] = reader.file_readers["svi01"]
        reader.file_types["svi01_2"] = reader.file_types["svi01"]
        reader.navigations["gitco2"] = reader.navigations["gitco"]

        info = reader._get_dataset_info("I01", calibration=["radiance"])
        self.assertEqual(info["calibration"], "radiance")
        self.assertEqual(info["file_type"], "svi01_2")
        self.assertEqual(info["file_key"], "radiance")
        self.assertEqual(info["navigation"], "gitco2")

    @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
    def test_get_dataset_info_file_type_not_configured(self):
        from satpy.readers.viirs_sdr import VIIRSSDRReader
        filenames = self.example_filenames["SVI01"] + \
                    self.example_filenames["GMTCO"]
        reader = VIIRSSDRReader(filenames=filenames,
                                start_time=datetime(2015, 1, 1, 11, 0, 0),
                                end_time=datetime(2015, 1, 1, 12, 0, 0))
        reader.datasets["I01"]["calibration"] = ["reflectance"]
        reader.datasets["I01"]["file_type"] = ["svi01"]
        reader.datasets["I01"]["file_key"] = ["reflectance"]
        reader.datasets["I01"]["navigation"] = ["gitco"]

        info = reader._get_dataset_info("I01", calibration=["radiance"])
        self.assertEqual(info["calibration"], "reflectance")
        self.assertEqual(info["file_type"], "svi01")
        self.assertEqual(info["file_key"], "reflectance")
        self.assertEqual(info["navigation"], "gitco")


def suite():
    """The test suite for test_viirs_sdr.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestSDRFileReader))
    mysuite.addTest(loader.loadTestsFromTestCase(TestVIIRSSDRReader))
    mysuite.addTest(loader.loadTestsFromTestCase(TestHDF5MetaData))
    mysuite.addTest(loader.loadTestsFromTestCase(TestSDRMultiFileReader))
    
    return mysuite
