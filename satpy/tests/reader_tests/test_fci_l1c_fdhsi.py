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
"""Tests for the 'fci_l1c_fdhsi' reader."""

from __future__ import (division, absolute_import, print_function,
                        unicode_literals)

import sys
import os

import numpy as np
import xarray as xr
import dask.array as da
import unittest
import numpy.testing

from satpy.tests.reader_tests.test_netcdf_utils import FakeNetCDF4FileHandler

try:
    from unittest import mock  # Python 3.3 or newer
except ImportError:
    import mock  # Python 2.7


class FakeNetCDF4FileHandler2(FakeNetCDF4FileHandler):
    def _get_test_calib_for_channel_ir(self, chroot, meas):
        from pyspectral.blackbody import (
                H_PLANCK as h,
                K_BOLTZMANN as k,
                C_SPEED as c)
        xrda = xr.DataArray
        data = {}
        data[meas + "/radiance_unit_conversion_coefficient"] = xrda(1)
        data[chroot + "/central_wavelength_actual"] = xrda(10)
        data[meas + "/radiance_to_bt_conversion_coefficient_a"] = xrda(1000)
        data[meas + "/radiance_to_bt_conversion_coefficient_b"] = xrda(1)
        data[meas + "/radiance_to_bt_conversion_constant_c1"] = xrda(2*h*c**2)
        data[meas + "/radiance_to_bt_conversion_constant_c2"] = xrda(h*c/k)
        return data

    def _get_test_calib_for_channel_vis(self, chroot, meas):
        xrda = xr.DataArray
        data = {}
        data[meas + "/channel_effective_solar_irradiance"] = xrda(50)
        return data

    def _get_test_content_for_channel(self, pat, ch):
        xrda = xr.DataArray
        nrows = 200
        ncols = 11136
        chroot = "data/{:s}"
        meas = chroot + "/measured"
        rad = meas + "/effective_radiance"
        pos = meas + "/{:s}_position_{:s}"
        shp = rad + "/shape"
        data = {}
        ch_str = pat.format(ch)
        ch_path = rad.format(ch_str)
        d = xr.DataArray(
                da.ones((nrows, ncols), dtype="uint16", chunks=1024),
                dims=("y", "x"),
                attrs={
                    "valid_range": [0, 4095],
                    "scale_factor": 5,
                    "add_offset": 10,
                    "units": "mW.m-2.sr-1.(cm-1)-1",
                    }
                )
        data[ch_path] = d
        data[pos.format(ch_str, "start", "row")] = xrda(0)
        data[pos.format(ch_str, "start", "column")] = xrda(0)
        data[pos.format(ch_str, "end", "row")] = xrda(nrows)
        data[pos.format(ch_str, "end", "column")] = xrda(ncols)
        if pat.startswith("ir") or pat.startswith("wv"):
            data.update(self._get_test_calib_for_channel_ir(chroot.format(ch_str),
                        meas.format(ch_str)))
        elif pat.startswith("vis") or pat.startswith("nir"):
            data.update(self._get_test_calib_for_channel_vis(chroot.format(ch_str),
                        meas.format(ch_str)))
        data[shp.format(ch_str)] = (nrows, ncols)
        return data

    def _get_test_content_all_channels(self):
        chan_patterns = {
                "vis_{:>02d}": (4, 5, 6, 8, 9),
                "nir_{:>02d}": (13, 16, 22),
                "ir_{:>02d}": (38, 87, 97, 105, 123, 133),
                "wv_{:>02d}": (63, 73),
                }
        data = {}
        for pat in chan_patterns.keys():
            for ch_num in chan_patterns[pat]:
                data.update(self._get_test_content_for_channel(pat, ch_num))
        return data

    def _get_test_content_areadef(self):
        data = {}
        proc = "state/processor"
        for (lb, no) in (
                ("earth_equatorial_radius", 6378137),
                ("earth_polar_radius", 6356752),
                ("reference_altitude", 35786000),
                ("projection_origin_longitude", 0)):
            data[proc + "/" + lb] = xr.DataArray(no)
        return data

    def get_test_content(self, filename, filename_info, filetype_info):
        # mock global attributes
        # - root groups global
        # - other groups global
        # mock data variables
        # mock dimensions
        #
        # ... but only what satpy is using ...

        D = {}
        D.update(self._get_test_content_all_channels())
        D.update(self._get_test_content_areadef())
        return D


class FakeNetCDF4FileHandler3(FakeNetCDF4FileHandler2):
    """Mock bad data
    """
    def _get_test_calib_for_channel_ir(self, chroot, meas):
        from netCDF4 import default_fillvals
        v = xr.DataArray(default_fillvals["f4"])
        data = {}
        data[meas + "/radiance_unit_conversion_coefficient"] = v
        data[chroot + "/central_wavelength_actual"] = v
        data[meas + "/radiance_to_bt_conversion_coefficient_a"] = v
        data[meas + "/radiance_to_bt_conversion_coefficient_b"] = v
        data[meas + "/radiance_to_bt_conversion_constant_c1"] = v
        data[meas + "/radiance_to_bt_conversion_constant_c2"] = v
        return data


class TestFCIL1CFDHSIReader(unittest.TestCase):
    yaml_file = "fci_l1c_fdhsi.yaml"

    _alt_handler = FakeNetCDF4FileHandler2

    def setUp(self):
        """Wrap NetCDF4 FileHandler with our own fake handler
        """

        # implementation strongly inspired by test_viirs_l1b.py
        from satpy.config import config_search_paths
        from satpy.readers.fci_l1c_fdhsi import FCIFDHSIFileHandler

        self.reader_configs = config_search_paths(
                os.path.join("readers", self.yaml_file))
        self.p = mock.patch.object(
                FCIFDHSIFileHandler,
                "__bases__",
                (self._alt_handler,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the NetCDF4 file handler
        """
        # implementation strongly inspired by test_viirs_l1b.py
        self.p.stop()


class TestFCIL1CFDHSIReaderGoodData(TestFCIL1CFDHSIReader):
    """Test FCI L1C FDHSI reader
    """

    # TODO:
    # - test special case for extended range IR38
    # - test geolocation

    _alt_handler = FakeNetCDF4FileHandler2

    def test_file_pattern(self):
        """Test file pattern matching
        """
        from satpy.readers import load_reader

        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114442_GTT_DEV_"
            "20170410113934_20170410113942_N__C_0070_0068.nc",
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114451_GTT_DEV_"
            "20170410113942_20170410113951_N__C_0070_0069.nc",
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114500_GTT_DEV_"
            "20170410113951_20170410114000_N__C_0070_0070.nc",
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-HRFI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_19700101000000_GTT_DEV_"
            "19700000000000_19700000000000_N__C_0042_0070.nc",
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-TRAIL--L2P-NC4E_C_EUMT_20170410114600_GTT_DEV_"
            "20170410113000_20170410114000_N__C_0070_0071.nc",
        ]

        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        # only 4 out of 6 above should match
        self.assertTrue(4, len(files))

    _chans = {"solar": ["vis_04", "vis_05", "vis_06", "vis_08", "vis_09",
                        "nir_13", "nir_16", "nir_22"],
              "terran": ["ir_38", "wv_63", "wv_73", "ir_87", "ir_97", "ir_105",
                         "ir_123", "ir_133"]}

    def test_load_counts(self):
        """Test loading with counts
        """
        from satpy import DatasetID
        from satpy.readers import load_reader

        # testing two filenames to test correctly combined
        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114442_GTT_DEV_"
            "20170410113934_20170410113942_N__C_0070_0068.nc",
        ]

        reader = load_reader(self.reader_configs)
        loadables = reader.select_files_from_pathnames(filenames)
        reader.create_filehandlers(loadables)
        res = reader.load(
                [DatasetID(name=name, calibration="counts") for name in
                    self._chans["solar"] + self._chans["terran"]])
        self.assertEqual(16, len(res))
        for ch in self._chans["solar"] + self._chans["terran"]:
            self.assertEqual(res[ch].shape, (200*2, 11136))
            self.assertEqual(res[ch].dtype, np.uint16)
            self.assertEqual(res[ch].attrs["calibration"], "counts")
            self.assertEqual(res[ch].attrs["units"], "1")
            numpy.testing.assert_array_equal(res[ch], 1)

    def test_load_radiance(self):
        """Test loading with radiance
        """
        from satpy import DatasetID
        from satpy.readers import load_reader

        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
        ]

        reader = load_reader(self.reader_configs)
        loadables = reader.select_files_from_pathnames(filenames)
        reader.create_filehandlers(loadables)
        res = reader.load(
                [DatasetID(name=name, calibration="radiance") for name in
                    self._chans["solar"] + self._chans["terran"]])
        self.assertEqual(16, len(res))
        for ch in self._chans["solar"] + self._chans["terran"]:
            self.assertEqual(res[ch].shape, (200, 11136))
            self.assertEqual(res[ch].dtype, np.float64)
            self.assertEqual(res[ch].attrs["calibration"], "radiance")
            self.assertEqual(res[ch].attrs["units"], 'mW.m-2.sr-1.(cm-1)-1')
            numpy.testing.assert_array_equal(res[ch], 15)

    def test_load_reflectance(self):
        """Test loading with reflectance
        """
        from satpy import DatasetID
        from satpy.readers import load_reader

        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
        ]

        reader = load_reader(self.reader_configs)
        loadables = reader.select_files_from_pathnames(filenames)
        reader.create_filehandlers(loadables)
        res = reader.load(
                [DatasetID(name=name, calibration="reflectance") for name in
                    self._chans["solar"]])
        self.assertEqual(8, len(res))
        for ch in self._chans["solar"]:
            self.assertEqual(res[ch].shape, (200, 11136))
            self.assertEqual(res[ch].dtype, np.float64)
            self.assertEqual(res[ch].attrs["calibration"], "reflectance")
            self.assertEqual(res[ch].attrs["units"], "%")
            numpy.testing.assert_array_equal(res[ch], 15 / 50 * 100)

    def test_load_bt(self):
        """Test loading with bt
        """
        from satpy import DatasetID
        from satpy.readers import load_reader

        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
        ]

        reader = load_reader(self.reader_configs)
        loadables = reader.select_files_from_pathnames(filenames)
        reader.create_filehandlers(loadables)
        res = reader.load(
                [DatasetID(name=name, calibration="brightness_temperature") for
                    name in self._chans["terran"]])
        self.assertEqual(8, len(res))
        for ch in self._chans["terran"]:
            self.assertEqual(res[ch].shape, (200, 11136))
            self.assertEqual(res[ch].dtype, np.float64)
            self.assertEqual(res[ch].attrs["calibration"],
                             "brightness_temperature")
            self.assertEqual(res[ch].attrs["units"], "K")
            numpy.testing.assert_array_almost_equal(
                    res[ch],
                    181.917084)


class TestFCIL1CFDHSIReaderBadData(TestFCIL1CFDHSIReader):
    _alt_handler = FakeNetCDF4FileHandler3

    @unittest.skipIf(
            sys.version_info < (3, 4),
            "skipping log message testing on old Python version "
            "that doesn't have TestCase.assertLogs")
    def test_handling_bad_data_ir(self):
        """Test handling of bad data
        """
        from satpy import DatasetID
        from satpy.readers import load_reader

        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
        ]

        reader = load_reader(self.reader_configs)
        loadables = reader.select_files_from_pathnames(filenames)
        reader.create_filehandlers(loadables)
        with self.assertLogs(
                'satpy.readers.fci_l1c_fdhsi',
                level="ERROR") as cm:
            reader.load([DatasetID(
                    name="ir_123",
                    calibration="brightness_temperature")])
        self.assertRegex(cm.output[0], "cannot produce brightness temperatur")


def suite():
    """The test suite
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestFCIL1CFDHSIReaderGoodData))
    mysuite.addTest(loader.loadTestsFromTestCase(TestFCIL1CFDHSIReaderBadData))
    return mysuite
