#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009-2019 Satpy developers
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

from unittest import TestCase, main, TestLoader, TestSuite
import numpy as np
try:
    from unittest import mock
except ImportError:  # python 2
    import mock

GAC_PATTERN = 'NSS.GHRR.{platform_id:2s}.D{start_time:%y%j.S%H%M}.E{end_time:%H%M}.B{orbit_number:05d}{end_orbit_last_digits:02d}.{station:2s}'  # noqa

EXAMPLE_FILENAMES = ['NSS.GHRR.NA.D79184.S1150.E1337.B0008384.WI',
                     'NSS.GHRR.NA.D79184.S2350.E0137.B0008384.WI',
                     'NSS.GHRR.NA.D80021.S0927.E1121.B0295354.WI',
                     'NSS.GHRR.NA.D80021.S1120.E1301.B0295455.WI',
                     'NSS.GHRR.NA.D80021.S1256.E1450.B0295556.GC',
                     'NSS.GHRR.NE.D83208.S1219.E1404.B0171819.WI',
                     'NSS.GHRR.NG.D88002.S0614.E0807.B0670506.WI',
                     'NSS.GHRR.TN.D79183.S1258.E1444.B0369697.GC',
                     'NSS.GHRR.TN.D80003.S1147.E1332.B0630506.GC',
                     'NSS.GHRR.TN.D80003.S1328.E1513.B0630507.GC',
                     'NSS.GHRR.TN.D80003.S1509.E1654.B0630608.GC']


class TestGACLACFile(TestCase):
    """Test the GACLAC file handler."""

    def setUp(self):
        """Patch pygac imports."""
        self.pygac = mock.MagicMock()
        self.fhs = mock.MagicMock()
        modules = {
            'pygac': self.pygac,
            'pygac.gac_klm': self.pygac.gac_klm,
            'pygac.gac_pod': self.pygac.gac_pod,
        }

        self.module_patcher = mock.patch.dict('sys.modules', modules)
        self.module_patcher.start()

    def tearDown(self):
        """Unpatch the pygac imports."""
        self.module_patcher.stop()

    def test_gaclacfile(self):
        """Test the methods of the GACLACFile class."""
        from satpy.readers.avhrr_l1b_gaclac import GACLACFile
        from trollsift import parse
        from pygac.gac_klm import GACKLMReader
        from pygac.gac_pod import GACPODReader
        from satpy.dataset import DatasetID
        filename = np.random.choice(EXAMPLE_FILENAMES)
        filename_info = parse(GAC_PATTERN, filename)
        fh = GACLACFile(filename, filename_info, {})

        self.assertLess(fh.start_time, fh.end_time, "Start time must preceed end time.")
        if fh.sensor == 'avhrr-3':
            self.assertIs(fh.reader_class, GACKLMReader)
        else:
            self.assertIs(fh.reader_class, GACPODReader)

        key = DatasetID('1')
        info = {'name': '1', 'standard_name': 'reflectance'}

        ch_ones = np.ones((10, 10))
        acq_ones = np.ones((10, ))
        GACPODReader.return_value.get_calibrated_channels.return_value.__getitem__.return_value = ch_ones
        GACPODReader.return_value.get_times.return_value = acq_ones
        res = fh.get_dataset(key, info)
        np.testing.assert_allclose(res.data, ch_ones)
        self.assertIs(res.coords['acq_time'].data, acq_ones)

        for item in ['solar_zenith_angle', 'sensor_zenith_angle',
                     # 'solar_azimuth_angle', 'sensor_azimuth_angle',
                     'sun_sensor_azimuth_difference_angle']:
            key = DatasetID(item)
            info = {'name': item}

            angle_ones = np.ones((10, 10))
            acq_ones = np.ones((10, ))

            GACPODReader.return_value.get_angles.return_value = (angle_ones, ) * 5
            GACPODReader.return_value.get_times.return_value = acq_ones
            GACPODReader.return_value.get_tle_lines.return_value = 'tle1', 'tle2'
            res = fh.get_dataset(key, info)
            np.testing.assert_allclose(res.data, angle_ones)
            self.assertIs(res.coords['acq_time'].data, acq_ones)
            self.assertDictEqual(res.attrs['orbital_parameters'], {'tle': ('tle1', 'tle2')})

        key = DatasetID('longitude')
        info = {'name': 'longitude', 'unit': 'degrees_east'}

        lon_ones = np.ones((10, 10))
        lat_ones = np.ones((10, 10))
        acq_ones = np.ones((10, ))
        GACPODReader.return_value.lons = None

        def fill_lonlat():
            GACPODReader.return_value.lons = lon_ones
            GACPODReader.return_value.lats = lat_ones

        GACPODReader.return_value.get_lonlat.side_effect = fill_lonlat
        GACPODReader.return_value.get_times.return_value = acq_ones
        res = fh.get_dataset(key, info)
        np.testing.assert_allclose(res.data, lon_ones)
        self.assertEqual(res.attrs['unit'], 'degrees_east')
        self.assertIs(res.coords['acq_time'].data, acq_ones)

        key = DatasetID('latitude')
        info = {'name': 'latitude', 'unit': 'degrees_north'}

        res = fh.get_dataset(key, info)
        np.testing.assert_allclose(res.data, lat_ones)
        self.assertEqual(res.attrs['unit'], 'degrees_north')
        self.assertIs(res.coords['acq_time'].data, acq_ones)
        GACPODReader.return_value.get_lonlat.assert_called_once()


def suite():
    """The test suite."""
    loader = TestLoader()
    mysuite = TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestGACLACFile))

    return mysuite


if __name__ == '__main__':
    main()
