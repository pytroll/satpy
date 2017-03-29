#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module for testing the satpy.readers.viirs_sdr module.
"""

import os
import sys
if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest
import mock
from datetime import datetime
import numpy as np
from satpy.readers.hdf5_utils import HDF5FileHandler

DEFAULT_FILE_DTYPE = np.uint16
DEFAULT_FILE_SHAPE = (10, 300)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)
DEFAULT_FILE_FACTORS = np.array([2.0, 1.0], dtype=np.float32)
DEFAULT_LAT_DATA = np.linspace(45, 65, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LAT_DATA = np.repeat([DEFAULT_LAT_DATA], DEFAULT_FILE_SHAPE[0], axis=0)
DEFAULT_LON_DATA = np.linspace(5, 45, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LON_DATA = np.repeat([DEFAULT_LON_DATA], DEFAULT_FILE_SHAPE[0], axis=0)


class FakeHDF5FileHandler(HDF5FileHandler):
    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        super(HDF5FileHandler, self).__init__(filename, filename_info, filetype_info)
        offset = kwargs.get('offset', 0)
        begin_times = ["10{0:02d}12.5Z".format(x) for x in range(5)]
        end_times = ["11{0:02d}10.6Z".format(x) for x in range(5)]

        prefix1 = 'Data_Products/{file_group}'.format(**filetype_info)
        prefix2 = '{prefix}/{file_group}_Aggr'.format(prefix=prefix1, **filetype_info)
        prefix3 = 'All_Data/{file_group}_All'.format(**filetype_info)
        file_content = {
            "{prefix2}/attr/AggregateBeginningDate": "20120225",
            "{prefix2}/attr/AggregateBeginningTime": begin_times[offset],
            "{prefix2}/attr/AggregateEndingDate": "20120225",
            "{prefix2}/attr/AggregateEndingTime": end_times[offset],
            "{prefix2}/attr/G-Ring_Longitude": np.array([0.0, 0.1, 0.2, 0.3]),
            "{prefix2}/attr/G-Ring_Latitude": np.array([0.0, 0.1, 0.2, 0.3]),
            "{prefix2}/attr/AggregateBeginningOrbitNumber": "{0:d}".format(offset),
            "{prefix2}/attr/AggregateEndingOrbitNumber": "{0:d}".format(offset + 1),

            "{prefix1}/attr/Instrument_Short_Name": "VIIRS",
            "/attr/Platform_Short_Name": "NPP",
            "/attr/N_GEO_Ref": "GITCO_npp_d20120225_t{0:02d}07061_e2359590_b01708_c20120226002502222157_noaa_ops.h5".format(offset),
        }
        self.file_content = {}
        for k, v in file_content.items():
            self.file_content[k.format(prefix1=prefix1, prefix2=prefix2)] = v

        for k in ["Radiance", "Reflectance", "BrightnessTemperature"]:
            k = prefix3 + "/" + k
            self.file_content[k] = DEFAULT_FILE_DATA.copy()
            self.file_content[k + "/shape"] = DEFAULT_FILE_SHAPE
            self.file_content[k + "Factors"] = DEFAULT_FILE_FACTORS.copy()
        for k in ["Latitude"]:
            k = prefix3 + "/" + k
            self.file_content[k] = np.linspace(45, 65, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
            self.file_content[k] = np.repeat([self.file_content[k]], DEFAULT_FILE_SHAPE[0], axis=0)
            self.file_content[k + "/shape"] = DEFAULT_FILE_SHAPE
        for k in ["Longitude"]:
            k = prefix3 + "/" + k
            self.file_content[k] = np.linspace(5, 45, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
            self.file_content[k] = np.repeat([self.file_content[k]], DEFAULT_FILE_SHAPE[0], axis=0)
            self.file_content[k + "/shape"] = DEFAULT_FILE_SHAPE

        self.file_content.update(kwargs)


class TestVIIRSSDRFileHandler(unittest.TestCase):
    def setUp(self):
        self.p = mock.patch('satpy.readers.hdf5_utils.HDF5FileHandler', FakeHDF5FileHandler)
        self.fake_hdf5 = self.p.start()

    def tearDown(self):
        self.p.stop()

    def test_init(self):
        from satpy.readers.viirs_sdr import VIIRSSDRFileHandler
        handler = VIIRSSDRFileHandler('fake.h5', {}, {'file_group': 'VIIRS-I1-SDR'})
        self.assertTrue(hasattr(handler, 'file_content'))
        self.assertTrue(hasattr(handler, 'filename'))
        self.assertTrue(hasattr(handler, 'filename_info'))
        self.assertTrue(hasattr(handler, 'filetype_info'))

    def test_start_end_time(self):
        from satpy.readers.viirs_sdr import VIIRSSDRFileHandler
        handler = VIIRSSDRFileHandler('fake.h5', {}, {'file_group': 'VIIRS-I1-SDR'})
        self.assertEquals(handler.start_time, datetime(2012, 2, 25, 10, 0, 12, 500000))
        self.assertEquals(handler.end_time, datetime(2012, 2, 25, 11, 0, 10, 600000))

    def test_start_end_orbit(self):
        from satpy.readers.viirs_sdr import VIIRSSDRFileHandler
        handler = VIIRSSDRFileHandler('fake.h5', {}, {'file_group': 'VIIRS-I1-SDR'})
        self.assertEquals(handler.start_orbit_number, 0)
        self.assertEquals(handler.end_orbit_number, 1)

    def test_platform_name(self):
        from satpy.readers.viirs_sdr import VIIRSSDRFileHandler
        handler = VIIRSSDRFileHandler('fake.h5', {}, {'file_group': 'VIIRS-I1-SDR'})
        self.assertEquals(handler.platform_name, 'Suomi-NPP')

    def test_sensor_name(self):
        from satpy.readers.viirs_sdr import VIIRSSDRFileHandler
        handler = VIIRSSDRFileHandler('fake.h5', {}, {'file_group': 'VIIRS-I1-SDR'})
        self.assertEquals(handler.sensor_name, 'viirs')

    def test_get_shape_rad(self):
        from satpy.readers.viirs_sdr import VIIRSSDRFileHandler
        from satpy.dataset import DatasetID
        handler = VIIRSSDRFileHandler('fake.h5', {}, {'file_group': 'VIIRS-I1-SDR'})
        i = DatasetID(name='I01', wavelength=(0.600, 0.640, 0.680), resolution=371, calibration='radiance', modifiers=('sunz_corrected',))
        ds_info = {
            'standard_name': 'toa_outgoing_radiance_per_unit_wavelength',
            'units': 'W m-2 um-1 sr-1',
        }
        shape = handler.get_shape(i, ds_info)
        self.assertTupleEqual(shape, (10, 300))

    def test_get_shape_refl(self):
        from satpy.readers.viirs_sdr import VIIRSSDRFileHandler
        from satpy.dataset import DatasetID
        handler = VIIRSSDRFileHandler('fake.h5', {}, {'file_group': 'VIIRS-I1-SDR'})
        i = DatasetID(name='I01', wavelength=(0.600, 0.640, 0.680), resolution=371, calibration='reflectance', modifiers=('sunz_corrected',))
        ds_info = {
            'standard_name': 'toa_bidirectional_reflectance',
            'units': '%',
        }
        shape = handler.get_shape(i, ds_info)
        self.assertTupleEqual(shape, (10, 300))

    def test_get_shape_bt(self):
        from satpy.readers.viirs_sdr import VIIRSSDRFileHandler
        from satpy.dataset import DatasetID
        handler = VIIRSSDRFileHandler('fake.h5', {}, {'file_group': 'VIIRS-I4-SDR'})
        i = DatasetID(name='I04', wavelength=(3.580, 3.740, 3.900), resolution=371, calibration='brightness_temperature', modifiers=('sunz_corrected',))
        ds_info = {
            'standard_name': 'toa_brightness_temperature',
            'units': 'K',
        }
        shape = handler.get_shape(i, ds_info)
        self.assertTupleEqual(shape, (10, 300))

    def test_radiance_no_out(self):
        from satpy.readers.viirs_sdr import VIIRSSDRFileHandler
        from satpy.dataset import DatasetID, Dataset
        handler = VIIRSSDRFileHandler('fake.h5', {}, {'file_group': 'VIIRS-I1-SDR'})
        i = DatasetID(name='I01', wavelength=(0.600, 0.640, 0.680), resolution=371, calibration='radiance', modifiers=('sunz_corrected',))
        ds_info = {
            'standard_name': 'toa_outgoing_radiance_per_unit_wavelength',
            'units': 'W m-2 um-1 sr-1',
        }
        ds = handler.get_dataset(i, ds_info)
        self.assertIsInstance(ds, Dataset)
        self.assertDictContainsSubset(ds_info, ds.info)

    def test_radiance_out(self):
        from satpy.readers.viirs_sdr import VIIRSSDRFileHandler
        from satpy.readers.yaml_reader import Shuttle
        from satpy.dataset import DatasetID, Dataset
        handler = VIIRSSDRFileHandler('fake.h5', {}, {'file_group': 'VIIRS-I1-SDR'})
        i = DatasetID(name='I01', wavelength=(0.600, 0.640, 0.680), resolution=371, calibration='radiance', modifiers=('sunz_corrected',))
        ds_info = {
            'standard_name': 'toa_outgoing_radiance_per_unit_wavelength',
            'units': 'W m-2 um-1 sr-1',
        }
        overall_shape = (10, 300)
        data = np.empty(overall_shape,
                        dtype=ds_info.get('dtype', np.float32))
        mask = np.ma.make_mask_none(overall_shape)
        info = {}
        s = Shuttle(data=data, mask=mask, info=info)
        ds = handler.get_dataset(i, ds_info, out=s)
        self.assertIsInstance(ds, Dataset)
        self.assertDictContainsSubset(ds_info, ds.info)
        self.assertTupleEqual(np.byte_bounds(ds), np.byte_bounds(data))
        self.assertTupleEqual(np.byte_bounds(ds.mask), np.byte_bounds(mask))

    def test_dnb_radiance_no_out(self):
        from satpy.readers.viirs_sdr import VIIRSSDRFileHandler
        from satpy.dataset import DatasetID, Dataset
        handler = VIIRSSDRFileHandler('fake.h5', {}, {'file_group': 'VIIRS-DNB-SDR'})
        i = DatasetID(name='DNB', wavelength=(0.500, 0.700, 0.900), resolution=743, calibration='radiance')
        ds_info = {
            'standard_name': 'toa_outgoing_radiance_per_unit_wavelength',
            'units': 'W m-2 sr-1',
            'file_units': 'W cm-2 sr-1',
        }
        ds = handler.get_dataset(i, ds_info)
        self.assertIsInstance(ds, Dataset)
        self.assertDictContainsSubset(ds_info, ds.info)

    def test_dnb_radiance_out(self):
        from satpy.readers.viirs_sdr import VIIRSSDRFileHandler
        from satpy.readers.yaml_reader import Shuttle
        from satpy.dataset import DatasetID, Dataset
        handler = VIIRSSDRFileHandler('fake.h5', {}, {'file_group': 'VIIRS-DNB-SDR'})
        i = DatasetID(name='DNB', wavelength=(0.500, 0.700, 0.900), resolution=743, calibration='radiance')
        ds_info = {
            'standard_name': 'toa_outgoing_radiance_per_unit_wavelength',
            'units': 'W m-2 sr-1',
            'file_units': 'W cm-2 sr-1',
        }
        overall_shape = (10, 300)
        data = np.empty(overall_shape,
                        dtype=ds_info.get('dtype', np.float32))
        mask = np.ma.make_mask_none(overall_shape)
        info = {}
        s = Shuttle(data=data, mask=mask, info=info)
        ds = handler.get_dataset(i, ds_info, out=s)
        self.assertIsInstance(ds, Dataset)
        self.assertDictContainsSubset(ds_info, ds.info)
        self.assertTupleEqual(np.byte_bounds(ds), np.byte_bounds(data))
        self.assertTupleEqual(np.byte_bounds(ds.mask), np.byte_bounds(mask))

    def test_reflectance_no_out(self):
        from satpy.readers.viirs_sdr import VIIRSSDRFileHandler
        from satpy.dataset import DatasetID, Dataset
        handler = VIIRSSDRFileHandler('fake.h5', {}, {'file_group': 'VIIRS-I1-SDR'})
        i = DatasetID(name='I01', wavelength=(0.600, 0.640, 0.680), resolution=371, calibration='reflectance', modifiers=('sunz_corrected',))
        ds_info = {
            'standard_name': 'toa_bidirectional_reflectance',
            'units': '%',
        }
        ds = handler.get_dataset(i, ds_info)
        self.assertIsInstance(ds, Dataset)
        self.assertDictContainsSubset(ds_info, ds.info)

    def test_reflectance_out(self):
        from satpy.readers.viirs_sdr import VIIRSSDRFileHandler
        from satpy.readers.yaml_reader import Shuttle
        from satpy.dataset import DatasetID, Dataset
        handler = VIIRSSDRFileHandler('fake.h5', {}, {'file_group': 'VIIRS-I1-SDR'})
        i = DatasetID(name='I01', wavelength=(0.600, 0.640, 0.680), resolution=371, calibration='reflectance', modifiers=('sunz_corrected',))
        ds_info = {
            'standard_name': 'toa_bidirectional_reflectance',
            'units': '%',
        }
        overall_shape = (10, 300)
        data = np.empty(overall_shape,
                        dtype=ds_info.get('dtype', np.float32))
        mask = np.ma.make_mask_none(overall_shape)
        info = {}
        s = Shuttle(data=data, mask=mask, info=info)
        ds = handler.get_dataset(i, ds_info, out=s)
        self.assertIsInstance(ds, Dataset)
        self.assertDictContainsSubset(ds_info, ds.info)
        self.assertTupleEqual(np.byte_bounds(ds), np.byte_bounds(data))
        self.assertTupleEqual(np.byte_bounds(ds.mask), np.byte_bounds(mask))

    def test_bt_no_out(self):
        from satpy.readers.viirs_sdr import VIIRSSDRFileHandler
        from satpy.dataset import DatasetID, Dataset
        handler = VIIRSSDRFileHandler('fake.h5', {}, {'file_group': 'VIIRS-I4-SDR'})
        i = DatasetID(name='I04', wavelength=(3.580, 3.740, 3.900), resolution=371, calibration='brightness_temperature', modifiers=('sunz_corrected',))
        ds_info = {
            'standard_name': 'toa_brightness_temperature',
            'units': 'K',
        }
        ds = handler.get_dataset(i, ds_info)
        self.assertIsInstance(ds, Dataset)
        self.assertDictContainsSubset(ds_info, ds.info)

    def test_bt_out(self):
        from satpy.readers.viirs_sdr import VIIRSSDRFileHandler
        from satpy.readers.yaml_reader import Shuttle
        from satpy.dataset import DatasetID, Dataset
        handler = VIIRSSDRFileHandler('fake.h5', {}, {'file_group': 'VIIRS-I4-SDR'})
        i = DatasetID(name='I04', wavelength=(3.580, 3.740, 3.900), resolution=371, calibration='brightness_temperature', modifiers=('sunz_corrected',))
        ds_info = {
            'standard_name': 'toa_brightness_temperature',
            'units': 'K',
        }
        overall_shape = (10, 300)
        data = np.empty(overall_shape,
                        dtype=ds_info.get('dtype', np.float32))
        mask = np.ma.make_mask_none(overall_shape)
        info = {}
        s = Shuttle(data=data, mask=mask, info=info)
        ds = handler.get_dataset(i, ds_info, out=s)
        self.assertIsInstance(ds, Dataset)
        self.assertDictContainsSubset(ds_info, ds.info)
        self.assertTupleEqual(np.byte_bounds(ds), np.byte_bounds(data))
        self.assertTupleEqual(np.byte_bounds(ds.mask), np.byte_bounds(mask))


class TestVIIRSSDRReader(unittest.TestCase):
    yaml_file = "viirs_sdr.yaml"

    def setUp(self):
        from satpy.config import config_search_paths
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        self.p = mock.patch('satpy.readers.hdf5_utils.HDF5FileHandler', FakeHDF5FileHandler)
        self.fake_hdf5 = self.p.start()

    def tearDown(self):
        self.p.stop()

    def test_init(self):
        """Test basic init with no extra parameters."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

    def test_init_start_time_beyond(self):
        """Test basic init with start_time after the provided files."""
        from satpy.readers import load_reader
        from datetime import datetime
        r = load_reader(self.reader_configs,
                        start_time=datetime(2012, 2, 26))
        loadables = r.select_files_from_pathnames([
            'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        self.assertTrue(len(loadables), 0)

    def test_init_end_time_beyond(self):
        """Test basic init with end_time before the provided files."""
        from satpy.readers import load_reader
        from datetime import datetime
        r = load_reader(self.reader_configs,
                        end_time=datetime(2012, 2, 24))
        loadables = r.select_files_from_pathnames([
            'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        self.assertTrue(len(loadables), 0)

    def test_init_start_end_time(self):
        """Test basic init with end_time before the provided files."""
        from satpy.readers import load_reader
        from datetime import datetime
        r = load_reader(self.reader_configs,
                        start_time=datetime(2012, 2, 24),
                        end_time=datetime(2012, 2, 26))
        loadables = r.select_files_from_pathnames([
            'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

# class TestHDF5MetaData(unittest.TestCase):
#     def test_init_doesnt_exist(self):
#         from satpy.readers.viirs_sdr import HDF5FileHandler
#         self.assertRaises(IOError, HDF5FileHandler, "test_asdflkajsd.h5")
#
#     @mock.patch("h5py.File")
#     @mock.patch("os.path.exists")
#     def test_init_basic(self, os_exists_mock, h5py_file_mock):
#         import h5py
#         from satpy.readers.viirs_sdr import HDF5FileHandler
#         os_exists_mock.return_value = True
#         f_handle = h5py_file_mock.return_value
#         f_handle.attrs = {
#             "test_int": np.array([1]),
#             "test_str": np.array("VIIRS"),
#             "test_arr": np.arange(5),
#         }
#         # f_handle.visititems.side_effect = lambda f: f()
#         h = HDF5FileHandler("fake.h5")
#         self.assertTrue(h5py_file_mock.called)
#         self.assertTrue(f_handle.visititems.called)
#         self.assertTrue(f_handle.close.called)
#         self.assertEqual(f_handle.visititems.call_args, ((h.collect_metadata,),))
#
#     @mock.patch("h5py.File")
#     @mock.patch("os.path.exists")
#     def test_collect_metadata(self, os_exists_mock, h5py_file_mock):
#         import h5py
#         from satpy.readers.viirs_sdr import HDF5FileHandler
#         os_exists_mock.return_value = True
#         f_handle = h5py_file_mock.return_value
#         h = HDF5FileHandler("fake.h5")
#         with mock.patch.object(h, "_collect_attrs") as collect_attrs_patch:
#             obj_mock = mock.Mock()
#             h.collect_metadata("fake", obj_mock)
#             self.assertTrue(collect_attrs_patch.called)
#             self.assertEqual(collect_attrs_patch.call_args, (("fake", obj_mock.attrs),))
#
#         with mock.patch.object(h, "_collect_attrs") as collect_attrs_patch:
#             obj_mock = mock.Mock(spec=h5py.Dataset)
#             obj_mock.shape = DEFAULT_FILE_SHAPE
#             h.collect_metadata("fake", obj_mock)
#             self.assertTrue(collect_attrs_patch.called)
#             self.assertEqual(collect_attrs_patch.call_args, (("fake", obj_mock.attrs),))
#             self.assertIn("fake", h.file_content)
#             self.assertIn("fake/shape", h.file_content)
#             self.assertEqual(h.file_content["fake/shape"], DEFAULT_FILE_SHAPE)
#
#     @mock.patch("h5py.File")
#     @mock.patch("os.path.exists")
#     def test_getitem(self, os_exists_mock, h5py_file_mock):
#         import h5py
#         from satpy.readers.viirs_sdr import HDF5FileHandler
#         os_exists_mock.return_value = True
#         f_handle = h5py_file_mock.return_value
#         fake_dataset = f_handle["fake"].value
#         h = HDF5FileHandler("fake.h5")
#         h.file_content["fake"] = mock.Mock(spec=h5py.Dataset)
#         h.file_content["fake_other"] = 5
#         data = h["fake"]
#         self.assertEqual(data, fake_dataset)
#         data = h["fake_other"]
#         self.assertEqual(data, 5)
#
#
# class TestSDRFileReader(unittest.TestCase):
#     """Test the SDRFileReader class used by the VIIRS SDR Reader.
#     """
#     def setUp(self):
#         self.file_keys = _setup_file_keys()
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_init_basic(self):
#         from satpy.readers.viirs_sdr import SDRFileReader
#         file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
#         self.assertEqual(file_reader.start_time, datetime(2015, 1, 1, 10, 0, 12, 500000))
#         self.assertEqual(file_reader.end_time, datetime(2015, 1, 1, 11, 0, 10, 600000))
#         self.assertRaises(ValueError, file_reader._parse_datetime, "19580102", "120000.0Z")
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_get_funcs(self):
#         from satpy.readers.viirs_sdr import SDRFileReader
#         file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
#         gring_lon, gring_lat = file_reader.ring_lonlats
#         begin_orbit = file_reader.begin_orbit_number
#         self.assertIsInstance(begin_orbit, int)
#         self.assertEqual(begin_orbit, 0)
#         end_orbit = file_reader.end_orbit_number
#         self.assertIsInstance(end_orbit, int)
#         self.assertEqual(end_orbit, 1)
#         instrument_name = file_reader.sensor_name
#         self.assertEqual(instrument_name, "VIIRS")
#         platform_name = file_reader.platform_name
#         self.assertEqual(platform_name, "NPP")
#         geo_ref = file_reader.geofilename
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_data_shape(self):
#         from satpy.readers.viirs_sdr import SDRFileReader
#         file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
#         self.assertEquals(file_reader["reflectance/shape"], (10, 300))
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_file_units(self):
#         from satpy.readers.viirs_sdr import SDRFileReader
#         file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
#         file_units = file_reader.get_file_units("reflectance")
#         self.assertEqual(file_units, "1")
#         file_units = file_reader.get_file_units("radiance")
#         self.assertEqual(file_units, "W cm-2 sr-1")
#         file_units = file_reader.get_file_units("brightness_temperature")
#         self.assertEqual(file_units, "K")
#         file_units = file_reader.get_file_units("unknown_data")
#         self.assertIs(file_units, None)
#         file_units = file_reader.get_file_units("unknown_data2")
#         self.assertIs(file_units, "fake")
#         file_units = file_reader.get_file_units("longitude")
#         self.assertEqual(file_units, "degrees")
#         file_units = file_reader.get_file_units("latitude")
#         self.assertEqual(file_units, "degrees")
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_units(self):
#         from satpy.readers.viirs_sdr import SDRFileReader
#         file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
#         units = file_reader.get_units("reflectance")
#         self.assertEqual(units, "%")
#         units = file_reader.get_units("radiance")
#         self.assertEqual(units, "W m-2 sr-1")
#         units = file_reader.get_units("brightness_temperature")
#         self.assertEqual(units, "K")
#         units = file_reader.get_units("unknown_data")
#         self.assertIs(units, None)
#         units = file_reader.get_units("unknown_data2")
#         self.assertIs(units, "fake")
#         units = file_reader.get_units("longitude")
#         self.assertEqual(units, "degrees")
#         units = file_reader.get_units("latitude")
#         self.assertEqual(units, "degrees")
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_getting_raw_data(self):
#         from satpy.readers.viirs_sdr import SDRFileReader
#         file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
#         for k in ["radiance", "reflectance", "brightness_temperature",
#                   "unknown_data", "unknown_data2"]:
#             data = file_reader[k]
#             self.assertTrue(data.dtype == DEFAULT_FILE_DTYPE)
#             np.testing.assert_array_equal(data, DEFAULT_FILE_DATA)
#
#         for k in ["longitude"]:
#             data = file_reader[k]
#             self.assertTrue(data.dtype == DEFAULT_FILE_DTYPE)
#             np.testing.assert_array_equal(data, DEFAULT_LON_DATA)
#
#         for k in ["latitude"]:
#             data = file_reader[k]
#             self.assertTrue(data.dtype == DEFAULT_FILE_DTYPE)
#             np.testing.assert_array_equal(data, DEFAULT_LAT_DATA)
#
#         for k in ["unknown_data3"]:
#             data = file_reader[k]
#             self.assertTrue(data.dtype == np.float32)
#             np.testing.assert_array_equal(data, DEFAULT_FILE_DATA)
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_get_swath_data_noscale(self):
#         from satpy.readers.viirs_sdr import SDRFileReader
#         file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
#         for k in ["longitude", "latitude"]:
#             # these shouldn't have any change to their file data
#             # normally unknown_data would, but the scaling factors are bad in this test file
#             data, mask = file_reader.get_swath_data(k)
#             self.assertTrue(data.dtype == np.float32)
#             self.assertTrue(mask.dtype == np.bool)
#             if k == "longitude":
#                 np.testing.assert_array_equal(data, DEFAULT_LON_DATA)
#             else:
#                 np.testing.assert_array_equal(data, DEFAULT_LAT_DATA)
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_get_swath_data_badscale(self):
#         from satpy.readers.viirs_sdr import SDRFileReader
#         file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
#         for k in ["unknown_data"]:
#             # these shouldn't have any change to the file data
#             # normally there would be, but the scaling factors are bad in this test file so all data is masked
#             data, mask = file_reader.get_swath_data(k)
#             self.assertTrue(data.dtype == np.float32)
#             self.assertTrue(mask.dtype == np.bool)
#             np.testing.assert_array_equal(data, DEFAULT_FILE_DATA)
#             # bad fill values should result in bad science data
#             np.testing.assert_array_equal(mask, True)
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_get_swath_data_scale(self):
#         from satpy.readers.viirs_sdr import SDRFileReader
#         file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
#         valid_data = DEFAULT_FILE_DATA * DEFAULT_FILE_FACTORS[0] + DEFAULT_FILE_FACTORS[1]
#         for k in ["radiance", "reflectance", "brightness_temperature"]:
#             data, mask = file_reader.get_swath_data(k)
#             self.assertTrue(data.dtype == np.float32)
#             self.assertTrue(mask.dtype == np.bool)
#             if k == "radiance":
#                 np.testing.assert_array_equal(data, valid_data * 10000.0)
#             elif k == "reflectance":
#                 np.testing.assert_array_equal(data, valid_data * 100.0)
#             else:
#                 np.testing.assert_array_equal(data, valid_data)
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_get_swath_data_noscale_dtype(self):
#         """Test that we can get swath data that converts to a non-float output dtype and don't have any scaling.
#         """
#         from satpy.readers.viirs_sdr import SDRFileReader
#         file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
#         for k in ["unknown_data2"]:
#             data, mask = file_reader.get_swath_data(k)
#             self.assertTrue(data.dtype == np.int64)
#             self.assertTrue(mask.dtype == np.bool)
#             np.testing.assert_array_equal(data, DEFAULT_FILE_DATA)
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_get_swath_data_missing_scale_no_dtype(self):
#         """Test that we can get swath data that can have scaling, but doesn't have any in the file and doesn't
#         require any conversion for dtype.
#         """
#         from satpy.readers.viirs_sdr import SDRFileReader
#         file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
#         for k in ["unknown_data3"]:
#             data, mask = file_reader.get_swath_data(k)
#             self.assertTrue(data.dtype == np.float32)
#             self.assertTrue(mask.dtype == np.bool)
#             np.testing.assert_array_equal(data, DEFAULT_FILE_DATA)
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_get_swath_data_inplace(self):
#         """Test that we can get most file keys and write the data inplace in arrays provided.
#         """
#         from satpy.readers.viirs_sdr import SDRFileReader
#         file_reader = SDRFileReader("fake_file_type", "test.h5", self.file_keys)
#         data_out = np.zeros(DEFAULT_FILE_SHAPE, dtype=np.float32)
#         mask_out = np.zeros(DEFAULT_FILE_SHAPE, dtype=np.bool)
#         for k in ["radiance", "reflectance", "brightness_temperature",
#                   "longitude", "latitude", "unknown_data"]:
#             data, mask = file_reader.get_swath_data(k, data_out=data_out, mask_out=mask_out)
#             self.assertTrue(len(np.nonzero(data != 0)[0]) > 0)
#             self.assertTrue(data.dtype == np.float32)
#             self.assertTrue(mask.dtype == np.bool)
#             data_out[:] = 0
#             mask_out[:] = False
#
#         data_out = np.zeros(DEFAULT_FILE_SHAPE, dtype=np.int64)
#         for k in ["unknown_data2"]:
#             data, mask = file_reader.get_swath_data(k, data_out=data_out, mask_out=mask_out)
#             self.assertTrue(len(np.nonzero(data != 0)[0]) > 0)
#             self.assertTrue(data.dtype == np.int64)
#             self.assertTrue(mask.dtype == np.bool)
#             data_out[:] = 0
#             mask_out[:] = False
#
#
# class TestSDRMultiFileReader(unittest.TestCase):
#     def setUp(self):
#         self.file_keys = _setup_file_keys()
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_init_basic(self):
#         from satpy.readers.viirs_sdr import SDRFileReader
#         from satpy.readers.viirs_sdr import MultiFileReader
#         file_readers = [SDRFileReader("fake_file_type", "test.h5", self.file_keys, offset=x) for x in range(5)]
#         file_reader = MultiFileReader("fake_file_type", file_readers, self.file_keys)
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_properties(self):
#         from satpy.readers.viirs_sdr import SDRFileReader
#         from satpy.readers.viirs_sdr import MultiFileReader
#         file_readers = [
#             SDRFileReader("fake_file_type", "test{0:02d}.h5".format(x), self.file_keys, offset=x) for x in range(5)]
#         file_reader = MultiFileReader("fake_file_type", file_readers, self.file_keys)
#         fns = ["test00.h5", "test01.h5", "test02.h5", "test03.h5", "test04.h5"]
#         geo_fns = ["GITCO_npp_d20120225_t{0:02d}07061_e2359590_b01708_c20120226002502222157_noaa_ops.h5".format(x) for x in range(5)]
#         self.assertListEqual(file_reader.filenames, fns)
#         self.assertEqual(file_reader.start_time, datetime(2015, 1, 1, 10, 0, 12, 500000))
#         self.assertEqual(file_reader.end_time, datetime(2015, 1, 1, 11, 4, 10, 600000))
#         self.assertEqual(file_reader.begin_orbit_number, 0)
#         self.assertEqual(file_reader.end_orbit_number, 5)
#         self.assertEqual(file_reader.platform_name, "NPP")
#         self.assertEqual(file_reader.sensor_name, "VIIRS")
#         self.assertListEqual(file_reader.geofilenames, geo_fns)
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_get_units(self):
#         from satpy.readers.viirs_sdr import SDRFileReader
#         from satpy.readers.viirs_sdr import MultiFileReader
#         file_readers = [
#             SDRFileReader("fake_file_type", "test{0:02d}.h5".format(x), self.file_keys, offset=x) for x in range(5)]
#         file_reader = MultiFileReader("fake_file_type", file_readers, self.file_keys)
#         # shouldn't need to thoroughly test this because its tested in the single file reader tests
#         self.assertEqual(file_reader.get_units("radiance"), "W m-2 sr-1")
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_get_swath_data(self):
#         from satpy.readers.viirs_sdr import SDRFileReader
#         from satpy.readers.viirs_sdr import MultiFileReader
#         file_readers = [
#             SDRFileReader("fake_file_type", "test{0:02d}.h5".format(x), self.file_keys, offset=x) for x in range(5)]
#         file_reader = MultiFileReader("fake_file_type", file_readers, self.file_keys)
#         data = file_reader.get_swath_data("brightness_temperature")
#         # make sure its a masked array
#         self.assertTrue(hasattr(data, "mask"))
#         self.assertTrue(hasattr(data, "data"))
#         valid_shape = (DEFAULT_FILE_SHAPE[0]*5, DEFAULT_FILE_SHAPE[1])
#         self.assertEqual(data.shape, valid_shape)
#         valid_mask = np.zeros(valid_shape, dtype=np.bool)
#         valid_data = np.concatenate(tuple(DEFAULT_FILE_DATA.astype(np.float32) for x in range(5)))
#         valid_data = np.ma.masked_array(valid_data, valid_mask) * 2.0 + 1.0
#         np.testing.assert_array_equal(data, valid_data)
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_get_swath_data_to_disk(self):
#         from satpy.readers.viirs_sdr import SDRFileReader
#         from satpy.readers.viirs_sdr import MultiFileReader
#         file_readers = [
#             SDRFileReader("fake_file_type", "test{0:02d}.h5".format(x), self.file_keys, offset=x) for x in range(5)]
#         file_reader = MultiFileReader("fake_file_type", file_readers, self.file_keys)
#         self.assertRaises(NotImplementedError, file_reader.get_swath_data, "radiance", filename="test.dat")
#
#
# class TestVIIRSSDRReader(unittest.TestCase):
#     def setUp(self):
#         example_filenames = {}
#         for file_type in ["SVI01", "SVI04", "SVM01", "GITCO", "GMTCO"]:
#             fn_pat = "{0}_npp_d2012022{1:d}_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5"
#             fns = [fn_pat.format(file_type, x) for x in range(5)]
#             example_filenames[file_type] = fns
#         self.example_filenames = example_filenames
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_init_no_args(self):
#         from satpy.readers.viirs_sdr import VIIRSSDRReader
#         self.assertRaises(ValueError, VIIRSSDRReader)
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_init_only_files_not_matching(self):
#         from satpy.readers.viirs_sdr import VIIRSSDRReader
#         input_files = ["test_{0:d}.h5".format(x) for x in range(5)]
#         self.assertRaises(ValueError, VIIRSSDRReader, filenames=input_files)
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_init_only_files_no_geo_files(self):
#         from satpy.readers.viirs_sdr import VIIRSSDRReader
#         reader = VIIRSSDRReader(filenames=self.example_filenames["SVI01"])
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_init_start_time_not_matching(self):
#         from satpy.readers.viirs_sdr import VIIRSSDRReader
#         filenames = self.example_filenames["SVI01"] + self.example_filenames["GITCO"]
#         self.assertRaises(IOError, VIIRSSDRReader, filenames=filenames, start_time=datetime(2012, 3, 1))
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_init_start_time_end_time_not_matching(self):
#         from satpy.readers.viirs_sdr import VIIRSSDRReader
#         filenames = self.example_filenames["SVI01"] + self.example_filenames["GITCO"]
#         self.assertRaises(IOError, VIIRSSDRReader, filenames=filenames,
#                           start_time=datetime(2012, 3, 1, 11, 0, 0),
#                           end_time=datetime(2012, 3, 1, 12, 0, 0))
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_init_start_time_matching(self):
#         from satpy.readers.viirs_sdr import VIIRSSDRReader
#         filenames = self.example_filenames["SVI01"] + self.example_filenames["GITCO"]
#         reader = VIIRSSDRReader(filenames=filenames, start_time=datetime(2015, 1, 1, 11, 0, 0))
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_init_start_time_end_time_matching(self):
#         from satpy.readers.viirs_sdr import VIIRSSDRReader
#         filenames = self.example_filenames["SVI01"] + self.example_filenames["GITCO"]
#         # when files are at the beginning of the range
#         reader = VIIRSSDRReader(filenames=filenames,
#                           start_time=datetime(2015, 1, 1, 11, 0, 0),
#                           end_time=datetime(2015, 1, 1, 12, 0, 0))
#         # when files are at the end of the range
#         reader = VIIRSSDRReader(filenames=filenames,
#                                 start_time=datetime(2015, 1, 1, 10, 0, 0),
#                                 end_time=datetime(2015, 1, 1, 11, 0, 0))
#         # when files fall within the whole range
#         reader = VIIRSSDRReader(filenames=filenames,
#                                 start_time=datetime(2015, 1, 1, 9, 0, 0),
#                                 end_time=datetime(2015, 1, 1, 13, 0, 0))
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_init_varying_number_files(self):
#         from satpy.readers.viirs_sdr import VIIRSSDRReader
#         filenames = self.example_filenames["SVI01"] + self.example_filenames["GITCO"][1:]
#         self.assertRaises(IOError, VIIRSSDRReader, filenames=filenames,
#                                 start_time=datetime(2015, 1, 1, 11, 0, 0),
#                                 end_time=datetime(2015, 1, 1, 12, 0, 0))
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_load_unknown_names(self):
#         from satpy.readers.viirs_sdr import VIIRSSDRReader
#         filenames = self.example_filenames["SVI01"] + self.example_filenames["GITCO"]
#         reader = VIIRSSDRReader(filenames=filenames,
#                                 start_time=datetime(2015, 1, 1, 11, 0, 0),
#                                 end_time=datetime(2015, 1, 1, 12, 0, 0))
#         datasets = reader.load(["fake", "fake2"])
#         self.assertEqual(datasets, {})
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_load_basic(self):
#         from satpy.readers.viirs_sdr import VIIRSSDRReader
#         filenames = self.example_filenames["SVI01"] + \
#                     self.example_filenames["SVI04"] + \
#                     self.example_filenames["SVM01"] + \
#                     self.example_filenames["GITCO"] + \
#                     self.example_filenames["GMTCO"]
#         reader = VIIRSSDRReader(filenames=filenames,
#                                 start_time=datetime(2015, 1, 1, 11, 0, 0),
#                                 end_time=datetime(2015, 1, 1, 12, 0, 0))
#
#         datasets = reader.load(["I01", "I04", "M01"], unused=1, unused2=2)
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_load_navigation_not_found(self):
#         from satpy.readers.viirs_sdr import VIIRSSDRReader
#         filenames = self.example_filenames["SVI01"] + \
#                     self.example_filenames["GMTCO"]
#         reader = VIIRSSDRReader(filenames=filenames,
#                                 start_time=datetime(2015, 1, 1, 11, 0, 0),
#                                 end_time=datetime(2015, 1, 1, 12, 0, 0))
#
#         with mock.patch.object(reader, "identify_file_types") as ift_mock:
#             ift_mock.return_value = {}
#             self.assertRaises(RuntimeError, reader.load_navigation, "gitco", "svi01")
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_load_navigation(self):
#         from satpy.readers.viirs_sdr import VIIRSSDRReader
#         filenames = self.example_filenames["SVI01"] + \
#                     self.example_filenames["GMTCO"]
#         reader = VIIRSSDRReader(filenames=filenames,
#                                 start_time=datetime(2015, 1, 1, 11, 0, 0),
#                                 end_time=datetime(2015, 1, 1, 12, 0, 0))
#
#         area = reader.load_navigation("gitco", dep_file_type="svi01")
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_get_dataset_info_no_cal(self):
#         from satpy.readers.viirs_sdr import VIIRSSDRReader
#         filenames = self.example_filenames["SVI01"] + \
#                     self.example_filenames["GMTCO"]
#         reader = VIIRSSDRReader(filenames=filenames,
#                                 start_time=datetime(2015, 1, 1, 11, 0, 0),
#                                 end_time=datetime(2015, 1, 1, 12, 0, 0))
#
#         info = reader._get_dataset_info("I01", ["reflectance"])
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_get_dataset_info_no_cal_level(self):
#         from satpy.readers.viirs_sdr import VIIRSSDRReader
#         filenames = self.example_filenames["SVI01"] + \
#                     self.example_filenames["GMTCO"]
#         reader = VIIRSSDRReader(filenames=filenames,
#                                 start_time=datetime(2015, 1, 1, 11, 0, 0),
#                                 end_time=datetime(2015, 1, 1, 12, 0, 0))
#
#         # the channels calibration level isn't configured
#         info = reader._get_dataset_info("I01", calibration=["counts"])
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_get_dataset_info_cal_equals(self):
#         from satpy.readers.viirs_sdr import VIIRSSDRReader
#         filenames = self.example_filenames["SVI01"] + \
#                     self.example_filenames["GMTCO"]
#         reader = VIIRSSDRReader(filenames=filenames,
#                                 start_time=datetime(2015, 1, 1, 11, 0, 0),
#                                 end_time=datetime(2015, 1, 1, 12, 0, 0))
#         reader.datasets["I01"]["calibration"] = ["reflectance", "radiance"]
#         reader.datasets["I01"]["file_type"] = ["svi01", "svi01_2"]
#         reader.datasets["I01"]["file_key"] = ["reflectance", "radiance"]
#         reader.datasets["I01"]["navigation"] = ["gitco", "gitco2"]
#
#         info = reader._get_dataset_info("I01", calibration=["reflectance"])
#         self.assertEqual(info["calibration"], "reflectance")
#         self.assertEqual(info["file_type"], "svi01")
#         self.assertEqual(info["file_key"], "reflectance")
#         self.assertEqual(info["navigation"], "gitco")
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_get_dataset_info_cal_found(self):
#         from satpy.readers.viirs_sdr import VIIRSSDRReader
#         filenames = self.example_filenames["SVI01"] + \
#                     self.example_filenames["GMTCO"]
#         reader = VIIRSSDRReader(filenames=filenames,
#                                 start_time=datetime(2015, 1, 1, 11, 0, 0),
#                                 end_time=datetime(2015, 1, 1, 12, 0, 0))
#         reader.datasets["I01"]["calibration"] = ["reflectance", "radiance"]
#         reader.datasets["I01"]["file_type"] = ["svi01", "svi01_2"]
#         reader.datasets["I01"]["file_key"] = ["reflectance", "radiance"]
#         reader.datasets["I01"]["navigation"] = ["gitco", "gitco2"]
#         reader.file_readers["svi01_2"] = reader.file_readers["svi01"]
#         reader.file_types["svi01_2"] = reader.file_types["svi01"]
#         reader.navigations["gitco2"] = reader.navigations["gitco"]
#
#         info = reader._get_dataset_info("I01", calibration=["radiance"])
#         self.assertEqual(info["calibration"], "radiance")
#         self.assertEqual(info["file_type"], "svi01_2")
#         self.assertEqual(info["file_key"], "radiance")
#         self.assertEqual(info["navigation"], "gitco2")
#
#     @mock.patch('satpy.readers.viirs_sdr.HDF5MetaData', FakeHDF5MetaData)
#     def test_get_dataset_info_file_type_not_configured(self):
#         from satpy.readers.viirs_sdr import VIIRSSDRReader
#         filenames = self.example_filenames["SVI01"] + \
#                     self.example_filenames["GMTCO"]
#         reader = VIIRSSDRReader(filenames=filenames,
#                                 start_time=datetime(2015, 1, 1, 11, 0, 0),
#                                 end_time=datetime(2015, 1, 1, 12, 0, 0))
#         reader.datasets["I01"]["calibration"] = ["reflectance"]
#         reader.datasets["I01"]["file_type"] = ["svi01"]
#         reader.datasets["I01"]["file_key"] = ["reflectance"]
#         reader.datasets["I01"]["navigation"] = ["gitco"]
#
#         info = reader._get_dataset_info("I01", calibration=["radiance"])
#         self.assertEqual(info["calibration"], "reflectance")
#         self.assertEqual(info["file_type"], "svi01")
#         self.assertEqual(info["file_key"], "reflectance")
#         self.assertEqual(info["navigation"], "gitco")


def suite():
    """The test suite for test_viirs_sdr.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestVIIRSSDRFileHandler))
    mysuite.addTest(loader.loadTestsFromTestCase(TestVIIRSSDRReader))

    return mysuite
