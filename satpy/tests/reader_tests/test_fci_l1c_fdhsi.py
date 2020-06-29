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

import os
import numpy as np
import xarray as xr
import dask.array as da
import numpy.testing
import pytest
import logging
from unittest import mock
from satpy.tests.reader_tests.test_netcdf_utils import FakeNetCDF4FileHandler


class FakeNetCDF4FileHandler2(FakeNetCDF4FileHandler):
    """Class for faking the NetCDF4 Filehandler."""

    def _get_test_calib_for_channel_ir(self, chroot, meas):
        from pyspectral.blackbody import (
                H_PLANCK as h,
                K_BOLTZMANN as k,
                C_SPEED as c)
        xrda = xr.DataArray
        data = {}
        data[meas + "/radiance_to_bt_conversion_coefficient_wavenumber"] = xrda(955)
        data[meas + "/radiance_to_bt_conversion_coefficient_a"] = xrda(1)
        data[meas + "/radiance_to_bt_conversion_coefficient_b"] = xrda(0.4)
        data[meas + "/radiance_to_bt_conversion_constant_c1"] = xrda(1e11*2*h*c**2)
        data[meas + "/radiance_to_bt_conversion_constant_c2"] = xrda(1e2*h*c/k)
        return data

    def _get_test_calib_for_channel_vis(self, chroot, meas):
        xrda = xr.DataArray
        data = {}
        data["state/celestial/earth_sun_distance"] = xrda(149597870.7)
        data[meas + "/channel_effective_solar_irradiance"] = xrda(50)
        return data

    def _get_test_content_for_channel(self, pat, ch):
        xrda = xr.DataArray
        nrows = 200
        ncols = 11136
        chroot = "data/{:s}"
        meas = chroot + "/measured"
        rad = meas + "/effective_radiance"
        qual = meas + "/pixel_quality"
        pos = meas + "/{:s}_position_{:s}"
        shp = rad + "/shape"
        x = meas + "/x"
        y = meas + "/y"
        data = {}
        ch_str = pat.format(ch)
        ch_path = rad.format(ch_str)

        common_attrs = {
                "scale_factor": 5,
                "add_offset": 10,
                "long_name": "Effective Radiance",
                "units": "mW.m-2.sr-1.(cm-1)-1",
                "ancillary_variables": "pixel_quality"
                }
        if ch == 38:
            fire_line = da.ones((1, ncols), dtype="uint16", chunks=1024) * 5000
            data_without_fires = da.ones((nrows-1, ncols), dtype="uint16", chunks=1024)
            d = xrda(
                da.concatenate([fire_line, data_without_fires], axis=0),
                dims=("y", "x"),
                attrs={
                    "valid_range": [0, 8191],
                    "warm_scale_factor": 2,
                    "warm_add_offset": -300,
                    **common_attrs
                }
            )
        else:
            d = xrda(
                da.ones((nrows, ncols), dtype="uint16", chunks=1024),
                dims=("y", "x"),
                attrs={
                    "valid_range": [0, 4095],
                    "warm_scale_factor": 1,
                    "warm_add_offset": 0,
                    **common_attrs
                    }
                )

        data[ch_path] = d
        data[x.format(ch_str)] = xrda(
                da.arange(1, ncols+1, dtype="uint16"),
                dims=("x",),
                attrs={
                    "scale_factor": -5.58877772833e-05,
                    "add_offset": 0.155619515845,
                    }
                )
        data[y.format(ch_str)] = xrda(
                da.arange(1, nrows+1, dtype="uint16"),
                dims=("y",),
                attrs={
                    "scale_factor": -5.58877772833e-05,
                    "add_offset": 0.155619515845,
                    }
                )
        data[qual.format(ch_str)] = xrda(
                da.arange(nrows*ncols, dtype="uint8").reshape(nrows, ncols) % 128,
                dims=("y", "x"))

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
        proj = "data/mtg_geos_projection"

        attrs = {
                "sweep_angle_axis": "x",
                "perspective_point_height": "35786400",
                "semi_major_axis": "6378137",
                "semi_minor_axis": "6356752",
                "longitude_of_projection_origin": "0",
                "inverse_flattening": "298.257223563",
                "units": "m"}
        data[proj] = xr.DataArray(
                0,
                dims=(),
                attrs=attrs)

        # also set attributes cached, as this may be how they are accessed with
        # the NetCDF4FileHandler
        for (k, v) in attrs.items():
            data[proj + "/attr/" + k] = v

        return data

    def _get_global_attributes(self):
        data = {}
        attrs = {"platform": "MTI1"}
        for (k, v) in attrs.items():
            data["/attr/" + k] = v
        return data

    def get_test_content(self, filename, filename_info, filetype_info):
        """Get the content of the test data."""
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
        D.update(self._get_global_attributes())
        return D


class FakeNetCDF4FileHandler3(FakeNetCDF4FileHandler2):
    """Mock bad data."""

    def _get_test_calib_for_channel_ir(self, chroot, meas):
        from netCDF4 import default_fillvals
        v = xr.DataArray(default_fillvals["f4"])
        data = {}
        data[meas + "/radiance_to_bt_conversion_coefficient_wavenumber"] = v
        data[meas + "/radiance_to_bt_conversion_coefficient_a"] = v
        data[meas + "/radiance_to_bt_conversion_coefficient_b"] = v
        data[meas + "/radiance_to_bt_conversion_constant_c1"] = v
        data[meas + "/radiance_to_bt_conversion_constant_c2"] = v
        return data

    def _get_test_calib_for_channel_vis(self, chroot, meas):
        data = super()._get_test_calib_for_channel_vis(chroot, meas)
        from netCDF4 import default_fillvals
        v = xr.DataArray(default_fillvals["f4"])
        data[meas + "/channel_effective_solar_irradiance"] = v
        return data


@pytest.fixture
def reader_configs():
    """Return reader configs for FCI."""

    from satpy.config import config_search_paths
    return config_search_paths(
        os.path.join("readers", "fci_l1c_fdhsi.yaml"))


class TestFCIL1CFDHSIReader:
    """Initialize the unittest TestCase for the FCI L1C FDHSI Reader."""

    yaml_file = "fci_l1c_fdhsi.yaml"

    _alt_handler = FakeNetCDF4FileHandler2

    @pytest.fixture(autouse=True, scope="class")
    def fake_handler(self):
        """Wrap NetCDF4 FileHandler with our own fake handler."""
        # implementation strongly inspired by test_viirs_l1b.py
        from satpy.readers.fci_l1c_fdhsi import FCIFDHSIFileHandler
        p = mock.patch.object(
                FCIFDHSIFileHandler,
                "__bases__",
                (self._alt_handler,))
        with p:
            p.is_local = True
            yield p


class TestFCIL1CFDHSIReaderGoodData(TestFCIL1CFDHSIReader):
    """Test FCI L1C FDHSI reader."""

    # TODO:
    # - test geolocation

    _alt_handler = FakeNetCDF4FileHandler2

    def test_file_pattern(self, reader_configs):
        """Test file pattern matching."""
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
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-TRAIL--L2P-NC4E_C_EUMT_20170410114600_GTT_DEV_"
            "20170410113000_20170410114000_N__C_0070_0071.nc",
        ]

        reader = load_reader(reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        # only 4 out of 5 above should match
        assert len(files) == 4

    _chans = {"solar": ["vis_04", "vis_05", "vis_06", "vis_08", "vis_09",
                        "nir_13", "nir_16", "nir_22"],
              "terran": ["ir_38", "wv_63", "wv_73", "ir_87", "ir_97", "ir_105",
                         "ir_123", "ir_133"]}

    def test_load_counts(self, reader_configs):
        """Test loading with counts."""
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

        reader = load_reader(reader_configs)
        loadables = reader.select_files_from_pathnames(filenames)
        reader.create_filehandlers(loadables)
        res = reader.load(
                [DatasetID(name=name, calibration="counts") for name in
                    self._chans["solar"] + self._chans["terran"]])
        assert 16 == len(res)
        for ch in self._chans["solar"] + self._chans["terran"]:
            assert res[ch].shape == (200*2, 11136)
            assert res[ch].dtype == np.uint16
            assert res[ch].attrs["calibration"] == "counts"
            assert res[ch].attrs["units"] == "1"
            if ch == 'ir_38':
                numpy.testing.assert_array_equal(res[ch][~0], 1)
                numpy.testing.assert_array_equal(res[ch][0], 5000)
            else:
                numpy.testing.assert_array_equal(res[ch], 1)

    def test_load_radiance(self, reader_configs):
        """Test loading with radiance."""
        from satpy import DatasetID
        from satpy.readers import load_reader

        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
        ]

        reader = load_reader(reader_configs)
        loadables = reader.select_files_from_pathnames(filenames)
        reader.create_filehandlers(loadables)
        res = reader.load(
                [DatasetID(name=name, calibration="radiance") for name in
                    self._chans["solar"] + self._chans["terran"]])
        assert 16 == len(res)
        for ch in self._chans["solar"] + self._chans["terran"]:
            assert res[ch].shape == (200, 11136)
            assert res[ch].dtype == np.float64
            assert res[ch].attrs["calibration"] == "radiance"
            assert res[ch].attrs["units"] == 'mW.m-2.sr-1.(cm-1)-1'
            if ch == 'ir_38':
                numpy.testing.assert_array_equal(res[ch][~0], 15)
                numpy.testing.assert_array_equal(res[ch][0], 9700)
            else:
                numpy.testing.assert_array_equal(res[ch], 15)

    def test_load_reflectance(self, reader_configs):
        """Test loading with reflectance."""
        from satpy import DatasetID
        from satpy.readers import load_reader

        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
        ]

        reader = load_reader(reader_configs)
        loadables = reader.select_files_from_pathnames(filenames)
        reader.create_filehandlers(loadables)
        res = reader.load(
                [DatasetID(name=name, calibration="reflectance") for name in
                    self._chans["solar"]])
        assert 8 == len(res)
        for ch in self._chans["solar"]:
            assert res[ch].shape == (200, 11136)
            assert res[ch].dtype == np.float64
            assert res[ch].attrs["calibration"] == "reflectance"
            assert res[ch].attrs["units"] == "%"
            numpy.testing.assert_array_equal(res[ch], 100 * 15 * 1 * np.pi / 50)

    def test_load_bt(self, reader_configs, caplog):
        """Test loading with bt."""
        from satpy import DatasetID
        from satpy.readers import load_reader
        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
        ]

        reader = load_reader(reader_configs)
        loadables = reader.select_files_from_pathnames(filenames)
        reader.create_filehandlers(loadables)
        with caplog.at_level(logging.WARNING):
            res = reader.load(
                    [DatasetID(name=name, calibration="brightness_temperature") for
                        name in self._chans["terran"]])
            assert caplog.text == ""
        for ch in self._chans["terran"]:
            assert res[ch].shape == (200, 11136)
            assert res[ch].dtype == np.float64
            assert res[ch].attrs["calibration"] == "brightness_temperature"
            assert res[ch].attrs["units"] == "K"

            if ch == 'ir_38':
                numpy.testing.assert_array_almost_equal(res[ch][~0], 209.68274099)
                numpy.testing.assert_array_almost_equal(res[ch][0], 1888.851296)
            else:
                numpy.testing.assert_array_almost_equal(res[ch], 209.68274099)

    def test_load_composite(self):
        """Test that composites are loadable."""
        # when dedicated composites for FCI FDHSI are implemented in satpy,
        # this method should probably move to a dedicated class and module
        # in the tests.compositor_tests package

        from satpy.composites import CompositorLoader
        cl = CompositorLoader()
        (comps, mods) = cl.load_compositors(["fci"])
        assert len(comps["fci"]) > 0
        assert len(mods["fci"]) > 0

    def test_load_quality_only(self, reader_configs):
        """Test that loading quality only works."""
        from satpy.readers import load_reader

        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
        ]

        reader = load_reader(reader_configs)
        loadables = reader.select_files_from_pathnames(filenames)
        reader.create_filehandlers(loadables)
        res = reader.load(["ir_123_pixel_quality"])
        assert res["ir_123_pixel_quality"].attrs["name"] == "ir_123_pixel_quality"

    def test_platform_name(self, reader_configs):
        """Test that platform name is exposed.

        Test that the FCI reader exposes the platform name.  Corresponds
        to GH issue 1014.
        """
        from satpy.readers import load_reader

        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
        ]

        reader = load_reader(reader_configs)
        loadables = reader.select_files_from_pathnames(filenames)
        reader.create_filehandlers(loadables)
        res = reader.load(["ir_123"])
        assert res["ir_123"].attrs["platform_name"] == "MTG-I1"

    def test_excs(self, reader_configs, caplog):
        """Test that exceptions are raised where expected."""
        from satpy import DatasetID
        from satpy.readers import load_reader

        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
        ]

        reader = load_reader(reader_configs)
        loadables = reader.select_files_from_pathnames(filenames)
        fhs = reader.create_filehandlers(loadables)

        with pytest.raises(ValueError):
            fhs["fci_l1c_fdhsi"][0].get_dataset(DatasetID(name="invalid"), {})
        with pytest.raises(ValueError):
            fhs["fci_l1c_fdhsi"][0]._get_dataset_quality(DatasetID(name="invalid"),
                                                         {})
        with caplog.at_level(logging.ERROR):
            fhs["fci_l1c_fdhsi"][0].get_dataset(
                    DatasetID(name="ir_123", calibration="unknown"),
                    {"units": "unknown"})
            assert "unknown calibration key" in caplog.text


class TestFCIL1CFDHSIReaderBadData(TestFCIL1CFDHSIReader):
    """Test the FCI L1C FDHSI Reader for bad data input."""

    _alt_handler = FakeNetCDF4FileHandler3

    def test_handling_bad_data_ir(self, reader_configs, caplog):
        """Test handling of bad IR data."""
        from satpy import DatasetID
        from satpy.readers import load_reader

        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
        ]

        reader = load_reader(reader_configs)
        loadables = reader.select_files_from_pathnames(filenames)
        reader.create_filehandlers(loadables)
        with caplog.at_level("ERROR"):
            reader.load([DatasetID(
                    name="ir_123",
                    calibration="brightness_temperature")])
            assert "cannot produce brightness temperature" in caplog.text

    def test_handling_bad_data_vis(self, reader_configs, caplog):
        """Test handling of bad VIS data."""
        from satpy import DatasetID
        from satpy.readers import load_reader

        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
        ]

        reader = load_reader(reader_configs)
        loadables = reader.select_files_from_pathnames(filenames)
        reader.create_filehandlers(loadables)
        with caplog.at_level("ERROR"):
            reader.load([DatasetID(
                    name="vis_04",
                    calibration="reflectance")])
            assert "cannot produce reflectance" in caplog.text
