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
"""Module for testing the satpy.readers.sar-c_safe module."""
import unittest
import unittest.mock as mock

from satpy.dataset import DataQuery
import dask.array as da
import xarray as xr
import numpy as np


class TestSAFEGRD(unittest.TestCase):
    """Test the SAFE GRD file handler."""

    @mock.patch('rasterio.open')
    def setUp(self, mocked_rio_open):
        """Set up the test case."""
        from satpy.readers.sar_c_safe import SAFEGRD
        filename_info = {'mission_id': 'S1A', 'dataset_name': 'foo', 'start_time': 0, 'end_time': 0,
                         'polarization': 'vv'}
        filetype_info = 'bla'
        self.noisefh = mock.MagicMock()
        self.noisefh.get_noise_correction.return_value = xr.DataArray(np.zeros((2, 2)))
        self.calfh = mock.MagicMock()
        self.calfh.get_calibration_constant.return_value = 1
        self.calfh.get_calibration.return_value = xr.DataArray(np.ones((2, 2)))

        self.test_fh = SAFEGRD('S1A_IW_GRDH_1SDV_20190201T024655_20190201T024720_025730_02DC2A_AE07.SAFE/measurement/s1a-iw-grd'  # noqa
                               '-vv-20190201t024655-20190201t024720-025730-02dc2a-001.tiff',
                               filename_info, filetype_info, self.calfh, self.noisefh)
        self.mocked_rio_open = mocked_rio_open

    def test_instantiate(self):
        """Test initialization of file handlers."""
        assert(self.test_fh._polarization == 'vv')
        assert(self.test_fh.calibration == self.calfh)
        assert(self.test_fh.noise == self.noisefh)
        self.mocked_rio_open.assert_called()

    def test_read_calibrated_natural(self):
        """Test the calibration routines."""
        calibration = mock.MagicMock()
        calibration.name = "sigma_nought"
        with mock.patch('satpy.readers.sar_c_safe.xr.open_rasterio') as fake_read_band:
            fake_read_band.return_value = xr.DataArray(da.from_array(np.array([[0, 1], [2, 3]])))
            xarr = self.test_fh.get_dataset(DataQuery(name="measurement", polarization="vv",
                                                      calibration=calibration, quantity='natural'), info=dict())
            np.testing.assert_allclose(xarr, [[np.nan, 2], [5, 10]])

    def test_read_calibrated_dB(self):
        """Test the calibration routines."""
        calibration = mock.MagicMock()
        calibration.name = "sigma_nought"
        with mock.patch('satpy.readers.sar_c_safe.xr.open_rasterio') as fake_read_band:
            fake_read_band.return_value = xr.DataArray(da.from_array(np.array([[0, 1], [2, 3]])))
            xarr = self.test_fh.get_dataset(DataQuery(name="measurement", polarization="vv",
                                                      calibration=calibration, quantity='dB'), info=dict())
            np.testing.assert_allclose(xarr, [[np.nan, 3.0103], [6.9897, 10]])

    def test_read_lon_lats(self):
        """Test reading lons and lats."""

        class FakeGCP:

            def __init__(self, *args):
                self.row, self.col, self.x, self.y, self.z = args

        gcps = [FakeGCP(0, 0, 0, 0, 0),
                FakeGCP(0, 3, 1, 0, 0),
                FakeGCP(3, 0, 0, 1, 0),
                FakeGCP(3, 3, 1, 1, 0),
                FakeGCP(0, 7, 2, 0, 0),
                FakeGCP(3, 7, 2, 1, 0),
                FakeGCP(7, 7, 2, 2, 0),
                FakeGCP(7, 3, 1, 2, 0),
                FakeGCP(7, 0, 0, 2, 0),
                FakeGCP(0, 15, 3, 0, 0),
                FakeGCP(3, 15, 3, 1, 0),
                FakeGCP(7, 15, 3, 2, 0),
                FakeGCP(15, 15, 3, 3, 0),
                FakeGCP(15, 7, 2, 3, 0),
                FakeGCP(15, 3, 1, 3, 0),
                FakeGCP(15, 0, 0, 3, 0),
                ]

        crs = dict(init='epsg:4326')

        self.mocked_rio_open.return_value.gcps = [gcps, crs]
        self.mocked_rio_open.return_value.shape = [16, 16]

        query = DataQuery(name="longitude", polarization="vv")
        xarr = self.test_fh.get_dataset(query, info=dict())
        expected = np.array([[3.79492915e-16, 5.91666667e-01, 9.09722222e-01,
                              1.00000000e+00, 9.08333333e-01, 6.80555556e-01,
                              3.62500000e-01, 8.32667268e-17, -3.61111111e-01,
                              -6.75000000e-01, -8.95833333e-01, -9.77777778e-01,
                              -8.75000000e-01, -5.41666667e-01, 6.80555556e-02,
                              1.00000000e+00],
                             [1.19166667e+00, 1.32437500e+00, 1.36941964e+00,
                              1.34166667e+00, 1.25598214e+00, 1.12723214e+00,
                              9.70282738e-01, 8.00000000e-01, 6.31250000e-01,
                              4.78898810e-01, 3.57812500e-01, 2.82857143e-01,
                              2.68898810e-01, 3.30803571e-01, 4.83437500e-01,
                              7.41666667e-01],
                             [1.82638889e+00, 1.77596726e+00, 1.72667765e+00,
                              1.67757937e+00, 1.62773172e+00, 1.57619402e+00,
                              1.52202558e+00, 1.46428571e+00, 1.40203373e+00,
                              1.33432894e+00, 1.26023065e+00, 1.17879819e+00,
                              1.08909084e+00, 9.90167942e-01, 8.81088790e-01,
                              7.60912698e-01],
                             [2.00000000e+00, 1.99166667e+00, 1.99305556e+00,
                              2.00000000e+00, 2.00833333e+00, 2.01388889e+00,
                              2.01250000e+00, 2.00000000e+00, 1.97222222e+00,
                              1.92500000e+00, 1.85416667e+00, 1.75555556e+00,
                              1.62500000e+00, 1.45833333e+00, 1.25138889e+00,
                              1.00000000e+00],
                             [1.80833333e+00, 2.01669643e+00, 2.18011267e+00,
                              2.30119048e+00, 2.38253827e+00, 2.42676446e+00,
                              2.43647747e+00, 2.41428571e+00, 2.36279762e+00,
                              2.28462160e+00, 2.18236607e+00, 2.05863946e+00,
                              1.91605017e+00, 1.75720663e+00, 1.58471726e+00,
                              1.40119048e+00],
                             [1.34722222e+00, 1.89627976e+00, 2.29940830e+00,
                              2.57341270e+00, 2.73509779e+00, 2.80126842e+00,
                              2.78872945e+00, 2.71428571e+00, 2.59474206e+00,
                              2.44690334e+00, 2.28757440e+00, 2.13356009e+00,
                              2.00166525e+00, 1.90869473e+00, 1.87145337e+00,
                              1.90674603e+00],
                             [7.12500000e-01, 1.67563988e+00, 2.36250177e+00,
                              2.80892857e+00, 3.05076318e+00, 3.12384850e+00,
                              3.06402742e+00, 2.90714286e+00, 2.68903770e+00,
                              2.44555485e+00, 2.21253720e+00, 2.02582766e+00,
                              1.92126913e+00, 1.93470451e+00, 2.10197669e+00,
                              2.45892857e+00],
                             [5.55111512e-16, 1.40000000e+00, 2.38095238e+00,
                              3.00000000e+00, 3.31428571e+00, 3.38095238e+00,
                              3.25714286e+00, 3.00000000e+00, 2.66666667e+00,
                              2.31428571e+00, 2.00000000e+00, 1.78095238e+00,
                              1.71428571e+00, 1.85714286e+00, 2.26666667e+00,
                              3.00000000e+00],
                             [-6.94444444e-01, 1.11458333e+00, 2.36631944e+00,
                              3.13888889e+00, 3.51041667e+00, 3.55902778e+00,
                              3.36284722e+00, 3.00000000e+00, 2.54861111e+00,
                              2.08680556e+00, 1.69270833e+00, 1.44444444e+00,
                              1.42013889e+00, 1.69791667e+00, 2.35590278e+00,
                              3.47222222e+00],
                             [-1.27500000e+00, 8.64613095e-01, 2.33016227e+00,
                              3.21785714e+00, 3.62390731e+00, 3.64452239e+00,
                              3.37591199e+00, 2.91428571e+00, 2.35585317e+00,
                              1.79682398e+00, 1.33340774e+00, 1.06181406e+00,
                              1.07825255e+00, 1.47893282e+00, 2.36006448e+00,
                              3.81785714e+00],
                             [-1.64583333e+00, 6.95312500e-01, 2.28404018e+00,
                              3.22916667e+00, 3.63950893e+00, 3.62388393e+00,
                              3.29110863e+00, 2.75000000e+00, 2.10937500e+00,
                              1.47805060e+00, 9.64843750e-01, 6.78571429e-01,
                              7.28050595e-01, 1.22209821e+00, 2.26953125e+00,
                              3.97916667e+00],
                             [-1.71111111e+00, 6.51904762e-01, 2.23951247e+00,
                              3.16507937e+00, 3.54197279e+00, 3.48356009e+00,
                              3.10320862e+00, 2.51428571e+00, 1.83015873e+00,
                              1.16419501e+00, 6.29761905e-01, 3.40226757e-01,
                              4.08956916e-01, 9.49319728e-01, 2.07468254e+00,
                              3.89841270e+00],
                             [-1.37500000e+00, 7.79613095e-01, 2.20813846e+00,
                              3.01785714e+00, 3.31605017e+00, 3.20999858e+00,
                              2.80698342e+00, 2.21428571e+00, 1.53918651e+00,
                              8.88966837e-01, 3.70907738e-01, 9.22902494e-02,
                              1.60395408e-01, 6.82504252e-01, 1.76589782e+00,
                              3.51785714e+00],
                             [-5.41666667e-01, 1.12366071e+00, 2.20147747e+00,
                              2.77976190e+00, 2.94649235e+00, 2.78964711e+00,
                              2.39720451e+00, 1.85714286e+00, 1.25744048e+00,
                              6.86075680e-01, 2.31026786e-01, -1.97278912e-02,
                              2.17899660e-02, 4.43558673e-01, 1.33355655e+00,
                              2.77976190e+00],
                             [8.84722222e-01, 1.72927083e+00, 2.23108879e+00,
                              2.44305556e+00, 2.41805060e+00, 2.20895337e+00,
                              1.86864335e+00, 1.45000000e+00, 1.00590278e+00,
                              5.89231151e-01, 2.52864583e-01, 4.96825397e-02,
                              3.25644841e-02, 2.54389881e-01, 7.68038194e-01,
                              1.62638889e+00],
                             [3.00000000e+00, 2.64166667e+00, 2.30853175e+00,
                              2.00000000e+00, 1.71547619e+00, 1.45436508e+00,
                              1.21607143e+00, 1.00000000e+00, 8.05555556e-01,
                              6.32142857e-01, 4.79166667e-01, 3.46031746e-01,
                              2.32142857e-01, 1.36904762e-01, 5.97222222e-02,
                              0.00000000e+00]])
        np.testing.assert_allclose(xarr.values, expected)
