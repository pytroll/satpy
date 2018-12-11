import datetime
import logging
import sys


import numpy as np
import xarray as xr

from satpy import DatasetID


if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


class GOESNCFileHandlerTest(unittest.TestCase):

    longMessage = True

    @mock.patch('satpy.readers.nc_goes.xr')
    def setUp(self, xr_):
        # Disable logging
        logging.getLogger("urllib3").setLevel(logging.WARNING)

        from satpy.readers.nc_goes import GOESNCFileHandler, CALIB_COEFS

        self.coefs = CALIB_COEFS['GOES-15']
        self.all_coefs = CALIB_COEFS
        self.channels = sorted(self.coefs.keys())
        self.ir_channels = sorted([ch for ch in self.channels
                                   if not GOESNCFileHandler._is_vis(ch)])
        self.vis_channels = sorted([ch for ch in self.channels
                                    if GOESNCFileHandler._is_vis(ch)])

        # Mock file access to return a fake dataset. Choose a medium count value
        # (100) to avoid elements being masked due to invalid
        # radiance/reflectance/BT
        nrows = ncols = 300
        self.counts = 100 * 32 * np.ones((1, nrows, ncols))  # emulate 10-bit
        self.lon = np.zeros((nrows, ncols))  # Dummy
        self.lat = np.repeat(np.linspace(-150, 150, nrows), ncols).reshape(
            nrows, ncols)  # Includes invalid values to be masked
        self.time = datetime.datetime(2018, 8, 16, 16, 7)

        xr_.open_dataset.return_value = xr.Dataset(
            {'data': xr.DataArray(data=self.counts, dims=('time', 'yc', 'xc')),
             'lon': xr.DataArray(data=self.lon,  dims=('yc', 'xc')),
             'lat': xr.DataArray(data=self.lat, dims=('yc', 'xc')),
             'time': xr.DataArray(data=np.array([self.time],
                                                dtype='datetime64[ms]'),
                                  dims=('time',)),
             'bands': xr.DataArray(data=np.array([1]))},
            attrs={'Satellite Sensor': 'G-15'})

        # Instantiate reader using the mocked open_dataset() method
        self.reader = GOESNCFileHandler(filename='dummy', filename_info={},
                                        filetype_info={})

    def test_get_dataset_coords(self):
        """Test whether coordinates returned by get_dataset() are correct"""
        lon = self.reader.get_dataset(key=DatasetID(name='longitude',
                                                    calibration=None),
                                      info={})
        lat = self.reader.get_dataset(key=DatasetID(name='latitude',
                                                    calibration=None),
                                      info={})
        # ... this only compares the valid (unmasked) elements
        self.assertTrue(np.all(lat.to_masked_array() == self.lat),
                        msg='get_dataset() returns invalid latitude')
        self.assertTrue(np.all(lon.to_masked_array() == self.lon),
                        msg='get_dataset() returns invalid longitude')

    def test_get_dataset_counts(self):
        """Test whether counts returned by get_dataset() are correct"""
        for ch in self.channels:
            counts = self.reader.get_dataset(
                key=DatasetID(name=ch, calibration='counts'), info={})
            # ... this only compares the valid (unmasked) elements
            self.assertTrue(np.all(self.counts/32. == counts.to_masked_array()),
                            msg='get_dataset() returns invalid counts for '
                                'channel {}'.format(ch))

    def test_get_dataset_masks(self):
        """Test whether data and coordinates are masked consistently"""
        # Requires that no element has been masked due to invalid
        # radiance/reflectance/BT (see setUp()).
        lon = self.reader.get_dataset(key=DatasetID(name='longitude',
                                                    calibration=None),
                                      info={})
        lon_mask = lon.to_masked_array().mask
        for ch in self.channels:
            for calib in ('counts', 'radiance', 'reflectance',
                          'brightness_temperature'):
                try:
                    data = self.reader.get_dataset(
                        key=DatasetID(name=ch, calibration=calib), info={})
                except ValueError:
                    continue
                data_mask = data.to_masked_array().mask
                self.assertTrue(np.all(data_mask == lon_mask),
                                msg='get_dataset() returns inconsistently '
                                    'masked {} in channel {}'.format(calib, ch))

    def test_get_dataset_invalid(self):
        """Test handling of invalid calibrations"""
        # VIS -> BT
        args = dict(key=DatasetID(name='00_7',
                                  calibration='brightness_temperature'),
                    info={})
        self.assertRaises(ValueError, self.reader.get_dataset, **args)

        # IR -> Reflectance
        args = dict(key=DatasetID(name='10_7',
                                  calibration='reflectance'),
                    info={})
        self.assertRaises(ValueError, self.reader.get_dataset, **args)

        # Unsupported calibration
        args = dict(key=DatasetID(name='10_7',
                                  calibration='invalid'),
                    info={})
        self.assertRaises(ValueError, self.reader.get_dataset, **args)

    def test_get_nadir_pixel(self):
        """Test identification of the nadir pixel"""
        from satpy.readers.nc_goes import FULL_DISC

        earth_mask = np.array([[0, 0, 0, 0],
                               [0, 1, 0, 0],
                               [1, 1, 1, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 0]])
        nadir_row, nadir_col = self.reader._get_nadir_pixel(
            earth_mask=earth_mask, sector=FULL_DISC)
        self.assertEqual((nadir_row, nadir_col), (2, 1),
                         msg='Incorrect nadir pixel')

    def test_get_earth_mask(self):
        """Test identification of earth/space pixels"""
        lat = xr.DataArray([-100, -90, -45, 0, 45, 90, 100])
        expected = np.array([0, 1, 1, 1, 1, 1, 0])
        mask = self.reader._get_earth_mask(lat)
        self.assertTrue(np.all(mask == expected),
                        msg='Incorrect identification of earth/space pixel')

    def test_is_yaw_flip(self):
        """Test yaw flip identification"""
        lat_asc = xr.DataArray([[1, 1, 1],
                                [2, 2, 2],
                                [3, 3, 3]])
        lat_dsc = xr.DataArray([[3, 3, 3],
                                [2, 2, 3],
                                [1, 1, 1]])
        self.assertEqual(self.reader._is_yaw_flip(lat_asc, delta=1), True,
                         msg='Yaw flip not identified')
        self.assertEqual(self.reader._is_yaw_flip(lat_dsc, delta=1), False,
                         msg='Yaw flip false alarm')

    def test_calibrate(self):
        """Test whether the correct calibration methods are called"""
        for ch in self.channels:
            if self.reader._is_vis(ch):
                calibs = {'radiance': '_viscounts2radiance',
                          'reflectance': '_calibrate_vis'}
            else:
                calibs = {'radiance': '_ircounts2radiance',
                          'brightness_temperature': '_calibrate_ir'}
            for calib, method in calibs.items():
                with mock.patch.object(self.reader, method) as target_func:
                    self.reader.calibrate(counts=self.reader.nc['data'],
                                          calibration=calib, channel=ch)
                    target_func.assert_called()

    def test_viscounts2radiance(self):
        """Test conversion from VIS counts to radiance"""
        # Reference data is for detector #1
        slope = self.coefs['00_7']['slope'][0]
        offset = self.coefs['00_7']['offset'][0]
        counts = xr.DataArray([0, 100, 200, 500, 1000, 1023])
        rad_expected = xr.DataArray(
            [0., 41.54896, 100.06862,
             275.6276, 568.2259, 581.685422])
        rad = self.reader._viscounts2radiance(counts=counts, slope=slope,
                                              offset=offset)
        self.assertTrue(np.allclose(rad.data, rad_expected.data, atol=1E-6),
                        msg='Incorrect conversion from VIS counts to '
                            'radiance')

    def test_ircounts2radiance(self):
        """Test conversion from IR counts to radiance"""
        # Test counts
        counts = xr.DataArray([0, 100, 500, 1000, 1023])

        # Reference Radiance from NOAA lookup tables (same for detectors 1 and
        # 2, see [IR])
        rad_expected = {
            '03_9': np.array([0, 0.140, 1.899, 4.098, 4.199]),
            '06_5': np.array([0, 1.825, 12.124, 24.998, 25.590]),
            '10_7': np.array([0, 16.126, 92.630, 188.259, 192.658]),
            '13_3': np.array([0, 15.084, 87.421, 177.842, 182.001])
        }

        # The input counts are exact, but the accuracy of the output radiance is
        # limited to 3 digits
        atol = 1E-3

        for ch in sorted(rad_expected.keys()):
            coefs = self.coefs[ch]
            rad = self.reader._ircounts2radiance(
                counts=counts, scale=coefs['scale'], offset=coefs['offset'])
            self.assertTrue(np.allclose(rad.data, rad_expected[ch], atol=atol),
                            msg='Incorrect conversion from IR counts to '
                                'radiance in channel {}'.format(ch))

    def test_calibrate_vis(self):
        """Test VIS calibration"""
        rad = xr.DataArray([0, 1, 10, 100, 500])
        refl_expected = xr.DataArray([0., 0.188852, 1.88852, 18.8852, 94.426])
        refl = self.reader._calibrate_vis(radiance=rad,
                                          k=self.coefs['00_7']['k'])
        self.assertTrue(np.allclose(refl.data, refl_expected.data, atol=1E-6),
                        msg='Incorrect conversion from radiance to '
                            'reflectance')

    def test_calibrate_ir(self):
        """Test IR calibration"""
        # Test radiance values and corresponding BT from NOAA lookup tables
        # rev. H (see [IR]).
        rad = {
            '03_9': xr.DataArray([0, 0.1, 2, 3.997, 4.199]),
            '06_5': xr.DataArray([0, 0.821, 12.201, 25.590, 100]),
            '10_7': xr.DataArray([0, 11.727, 101.810, 189.407, 192.658]),
            '13_3': xr.DataArray([0, 22.679, 90.133, 182.001, 500])
        }
        bt_expected = {
            '03_9': np.array([[np.nan, 253.213, 319.451, 339.983, np.nan],
                              [np.nan, 253.213, 319.451, 339.983, np.nan]]),
            '06_5': np.array([[np.nan, 200.291, 267.860, 294.988, np.nan],
                              [np.nan, 200.308, 267.879, 295.008, np.nan]]),
            '10_7': np.array([[np.nan, 200.105, 294.437, 339.960, np.nan],
                              [np.nan, 200.097, 294.429, 339.953, np.nan]]),
            '13_3': np.array([[np.nan, 200.006, 267.517, 321.986, np.nan],
                              [np.nan, 200.014, 267.524, 321.990, np.nan]])
        }  # first row is for detector 1, second for detector 2.

        # The accuracy of the input radiance is limited to 3 digits so that
        # the results differ slightly.
        atol = {'03_9': 0.04, '06_5': 0.03, '10_7': 0.01, '13_3': 0.01}

        for ch in sorted(rad.keys()):
            coefs = self.coefs[ch]
            for det in [0, 1]:
                bt = self.reader._calibrate_ir(radiance=rad[ch],
                                               coefs={'a': coefs['a'][det],
                                                      'b': coefs['b'][det],
                                                      'n': coefs['n'][det],
                                                      'btmin': coefs['btmin'],
                                                      'btmax': coefs['btmax']})
                self.assertTrue(
                    np.allclose(bt.data, bt_expected[ch][det], equal_nan=True,
                                atol=atol[ch]),
                    msg='Incorrect conversion from radiance to brightness '
                        'temperature in channel {} detector {}'.format(ch, det))

    def test_start_time(self):
        """Test dataset start time stamp"""
        self.assertEqual(self.reader.start_time, self.time)

    def test_end_time(self):
        """Test dataset end time stamp"""
        from satpy.readers.nc_goes import (SCAN_DURATION, FULL_DISC,
                                           UNKNOWN_SECTOR)
        expected = {
            UNKNOWN_SECTOR: self.time,
            FULL_DISC: self.time + SCAN_DURATION[FULL_DISC]
        }
        for sector, end_time in expected.items():
            self.reader.sector = sector
            self.assertEqual(self.reader.end_time, end_time)

    def test_get_sector(self):
        """Test sector identification"""
        from satpy.readers.nc_goes import (FULL_DISC, NORTH_HEMIS_EAST,
                                           SOUTH_HEMIS_EAST, NORTH_HEMIS_WEST,
                                           SOUTH_HEMIS_WEST, UNKNOWN_SECTOR)
        shapes_vis = {
            (10800, 20754): FULL_DISC,
            (7286, 13900): NORTH_HEMIS_EAST,
            (2301, 13840): SOUTH_HEMIS_EAST,
            (5400, 13200): NORTH_HEMIS_WEST,
            (4300, 11090): SOUTH_HEMIS_WEST,
            (123, 456): UNKNOWN_SECTOR
        }
        shapes_ir = {
            (2700, 5200): FULL_DISC,
            (1850, 3450): NORTH_HEMIS_EAST,
            (600, 3500): SOUTH_HEMIS_EAST,
            (1310, 3300): NORTH_HEMIS_WEST,
            (1099, 2800): SOUTH_HEMIS_WEST,
            (123, 456): UNKNOWN_SECTOR
        }
        shapes = shapes_ir.copy()
        shapes.update(shapes_vis)
        for (nlines, ncols), sector_ref in shapes.items():
            if (nlines, ncols) in shapes_vis:
                channel = '00_7'
            else:
                channel = '10_7'
            sector = self.reader._get_sector(channel=channel, nlines=nlines,
                                             ncols=ncols)
            self.assertEqual(sector, sector_ref,
                             msg='Incorrect sector identification')


def suite():
    """Test suite for GOES netCDF reader"""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(GOESNCFileHandlerTest))
    return mysuite


if __name__ == '__main__':
    unittest.main()
