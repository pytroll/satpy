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
"""Tests for the 'fci_l1c_nc' reader."""

import logging
import os
from unittest import mock

import dask.array as da
import numpy as np
import numpy.testing
import pytest
import xarray as xr

from satpy.tests.reader_tests.test_netcdf_utils import FakeNetCDF4FileHandler


class FakeNetCDF4FileHandler2(FakeNetCDF4FileHandler):
    """Class for faking the NetCDF4 Filehandler."""

    def _get_test_calib_for_channel_ir(self, chroot, meas):
        from pyspectral.blackbody import C_SPEED as c
        from pyspectral.blackbody import H_PLANCK as h
        from pyspectral.blackbody import K_BOLTZMANN as k
        xrda = xr.DataArray
        data = {}
        data[meas + "/radiance_to_bt_conversion_coefficient_wavenumber"] = xrda(955)
        data[meas + "/radiance_to_bt_conversion_coefficient_a"] = xrda(1)
        data[meas + "/radiance_to_bt_conversion_coefficient_b"] = xrda(0.4)
        data[meas + "/radiance_to_bt_conversion_constant_c1"] = xrda(1e11 * 2 * h * c ** 2)
        data[meas + "/radiance_to_bt_conversion_constant_c2"] = xrda(1e2 * h * c / k)
        return data

    def _get_test_calib_for_channel_vis(self, chroot, meas):
        xrda = xr.DataArray
        data = {}
        data["state/celestial/earth_sun_distance"] = xrda(da.repeat(da.array([149597870.7]), 6000))
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
        index_map = meas + "/index_map"
        rad_conv_coeff = meas + "/radiance_unit_conversion_coefficient"
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
            data_without_fires = da.ones((nrows - 1, ncols), dtype="uint16", chunks=1024)
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
            da.arange(1, ncols + 1, dtype="uint16"),
            dims=("x",),
            attrs={
                "scale_factor": -5.58877772833e-05,
                "add_offset": 0.155619515845,
            }
        )
        data[y.format(ch_str)] = xrda(
            da.arange(1, nrows + 1, dtype="uint16"),
            dims=("y",),
            attrs={
                "scale_factor": -5.58877772833e-05,
                "add_offset": 0.155619515845,
            }
        )
        data[qual.format(ch_str)] = xrda(
            da.arange(nrows * ncols, dtype="uint8").reshape(nrows, ncols) % 128,
            dims=("y", "x"))
        # add dummy data for index map starting from 1
        data[index_map.format(ch_str)] = xrda(
            (da.arange(nrows * ncols, dtype="uint16").reshape(nrows, ncols) % 6000) + 1,
            dims=("y", "x"))

        data[rad_conv_coeff.format(ch_str)] = xrda(1234.56)
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
        for pat in chan_patterns:
            for ch_num in chan_patterns[pat]:
                data.update(self._get_test_content_for_channel(pat, ch_num))
        return data

    def _get_test_content_areadef(self):
        data = {}

        proj = "data/mtg_geos_projection"

        attrs = {
            "sweep_angle_axis": "y",
            "perspective_point_height": "35786400.0",
            "semi_major_axis": "6378137.0",
            "longitude_of_projection_origin": "0.0",
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

    def _get_test_content_aux_data(self):
        from satpy.readers.fci_l1c_nc import AUX_DATA
        xrda = xr.DataArray
        data = {}
        indices_dim = 6000
        for key, value in AUX_DATA.items():
            # skip population of earth_sun_distance as this is already defined for reflectance calculation
            if key == 'earth_sun_distance':
                continue
            data[value] = xrda(da.arange(indices_dim, dtype="float32"), dims=("index"))

        # compute the last data entry to simulate the FCI caching
        data[list(AUX_DATA.values())[-1]] = data[list(AUX_DATA.values())[-1]].compute()

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
        D.update(self._get_test_content_aux_data())
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
    from satpy._config import config_search_paths
    return config_search_paths(
        os.path.join("readers", "fci_l1c_nc.yaml"))


def _get_reader_with_filehandlers(filenames, reader_configs):
    from satpy.readers import load_reader
    reader = load_reader(reader_configs)
    loadables = reader.select_files_from_pathnames(filenames)
    reader.create_filehandlers(loadables)
    return reader


class TestFCIL1cNCReader:
    """Initialize the unittest TestCase for the FCI L1c NetCDF Reader."""

    yaml_file = "fci_l1c_nc.yaml"

    _alt_handler = FakeNetCDF4FileHandler2

    @pytest.fixture(autouse=True, scope="class")
    def fake_handler(self):
        """Wrap NetCDF4 FileHandler with our own fake handler."""
        # implementation strongly inspired by test_viirs_l1b.py
        from satpy.readers.fci_l1c_nc import FCIL1cNCFileHandler
        p = mock.patch.object(
                FCIL1cNCFileHandler,
                "__bases__",
                (self._alt_handler,))
        with p:
            p.is_local = True
            yield p


class TestFCIL1cNCReaderGoodData(TestFCIL1cNCReader):
    """Test FCI L1c NetCDF reader."""

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
        from satpy.tests.utils import make_dataid

        # testing two filenames to test correctly combined
        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114442_GTT_DEV_"
            "20170410113934_20170410113942_N__C_0070_0068.nc",
        ]

        reader = _get_reader_with_filehandlers(filenames, reader_configs)
        res = reader.load(
            [make_dataid(name=name, calibration="counts") for name in
             self._chans["solar"] + self._chans["terran"]], pad_data=False)
        assert 16 == len(res)
        for ch in self._chans["solar"] + self._chans["terran"]:
            assert res[ch].shape == (200 * 2, 11136)
            assert res[ch].dtype == np.uint16
            assert res[ch].attrs["calibration"] == "counts"
            assert res[ch].attrs["units"] == "count"
            if ch == 'ir_38':
                numpy.testing.assert_array_equal(res[ch][~0], 1)
                numpy.testing.assert_array_equal(res[ch][0], 5000)
            else:
                numpy.testing.assert_array_equal(res[ch], 1)

    def test_load_radiance(self, reader_configs):
        """Test loading with radiance."""
        from satpy.tests.utils import make_dataid

        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
        ]

        reader = _get_reader_with_filehandlers(filenames, reader_configs)
        res = reader.load(
            [make_dataid(name=name, calibration="radiance") for name in
             self._chans["solar"] + self._chans["terran"]], pad_data=False)
        assert 16 == len(res)
        for ch in self._chans["solar"] + self._chans["terran"]:
            assert res[ch].shape == (200, 11136)
            assert res[ch].dtype == np.float64
            assert res[ch].attrs["calibration"] == "radiance"
            assert res[ch].attrs["units"] == 'mW m-2 sr-1 (cm-1)-1'
            assert res[ch].attrs["radiance_unit_conversion_coefficient"] == 1234.56
            if ch == 'ir_38':
                numpy.testing.assert_array_equal(res[ch][~0], 15)
                numpy.testing.assert_array_equal(res[ch][0], 9700)
            else:
                numpy.testing.assert_array_equal(res[ch], 15)

    def test_load_reflectance(self, reader_configs):
        """Test loading with reflectance."""
        from satpy.tests.utils import make_dataid

        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
        ]

        reader = _get_reader_with_filehandlers(filenames, reader_configs)
        res = reader.load(
            [make_dataid(name=name, calibration="reflectance") for name in
             self._chans["solar"]], pad_data=False)
        assert 8 == len(res)
        for ch in self._chans["solar"]:
            assert res[ch].shape == (200, 11136)
            assert res[ch].dtype == np.float64
            assert res[ch].attrs["calibration"] == "reflectance"
            assert res[ch].attrs["units"] == "%"
            numpy.testing.assert_array_almost_equal(res[ch], 100 * 15 * 1 * np.pi / 50)

    def test_load_bt(self, reader_configs, caplog):
        """Test loading with bt."""
        from satpy.tests.utils import make_dataid

        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
        ]

        reader = _get_reader_with_filehandlers(filenames, reader_configs)
        with caplog.at_level(logging.WARNING):
            res = reader.load(
                [make_dataid(name=name, calibration="brightness_temperature") for
                 name in self._chans["terran"]], pad_data=False)
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

    def test_load_index_map(self, reader_configs):
        """Test loading of index_map."""
        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc"
        ]

        reader = _get_reader_with_filehandlers(filenames, reader_configs)
        res = reader.load(
            [name + '_index_map' for name in
             self._chans["solar"] + self._chans["terran"]], pad_data=False)
        assert 16 == len(res)
        for ch in self._chans["solar"] + self._chans["terran"]:
            assert res[ch + '_index_map'].shape == (200, 11136)
            numpy.testing.assert_array_equal(res[ch + '_index_map'][1, 1], 5138)

    def test_load_aux_data(self, reader_configs):
        """Test loading of auxiliary data."""
        from satpy.readers.fci_l1c_nc import AUX_DATA

        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc"
        ]

        reader = _get_reader_with_filehandlers(filenames, reader_configs)
        res = reader.load(['vis_04_' + key for key in AUX_DATA.keys()],
                          pad_data=False)
        for aux in ['vis_04_' + key for key in AUX_DATA.keys()]:

            assert res[aux].shape == (200, 11136)
            if aux == 'vis_04_earth_sun_distance':
                numpy.testing.assert_array_equal(res[aux][1, 1], 149597870.7)
            else:
                numpy.testing.assert_array_equal(res[aux][1, 1], 5137)

    def test_load_composite(self):
        """Test that composites are loadable."""
        # when dedicated composites for FCI are implemented in satpy,
        # this method should probably move to a dedicated class and module
        # in the tests.compositor_tests package

        from satpy.composites.config_loader import load_compositor_configs_for_sensors
        comps, mods = load_compositor_configs_for_sensors(['fci'])
        assert len(comps["fci"]) > 0
        assert len(mods["fci"]) > 0

    def test_load_quality_only(self, reader_configs):
        """Test that loading quality only works."""
        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
        ]

        reader = _get_reader_with_filehandlers(filenames, reader_configs)
        res = reader.load(
            [name + '_pixel_quality' for name in
             self._chans["solar"] + self._chans["terran"]], pad_data=False)
        assert 16 == len(res)
        for ch in self._chans["solar"] + self._chans["terran"]:
            assert res[ch + '_pixel_quality'].shape == (200, 11136)
            numpy.testing.assert_array_equal(res[ch + '_pixel_quality'][1, 1], 1)
            assert res[ch + '_pixel_quality'].attrs["name"] == ch + '_pixel_quality'

    def test_platform_name(self, reader_configs):
        """Test that platform name is exposed.

        Test that the FCI reader exposes the platform name.  Corresponds
        to GH issue 1014.
        """
        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
        ]

        reader = _get_reader_with_filehandlers(filenames, reader_configs)
        res = reader.load(["ir_123"], pad_data=False)
        assert res["ir_123"].attrs["platform_name"] == "MTG-I1"

    def test_excs(self, reader_configs):
        """Test that exceptions are raised where expected."""
        from satpy.tests.utils import make_dataid
        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
        ]

        reader = _get_reader_with_filehandlers(filenames, reader_configs)

        with pytest.raises(ValueError):
            reader.file_handlers["fci_l1c_fdhsi"][0].get_dataset(make_dataid(name="invalid"), {})
        with pytest.raises(ValueError):
            reader.file_handlers["fci_l1c_fdhsi"][0].get_dataset(
                make_dataid(name="ir_123", calibration="unknown"),
                {"units": "unknown"})

    def test_area_definition_computation(self, reader_configs):
        """Test that the geolocation computation is correct."""
        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
        ]

        reader = _get_reader_with_filehandlers(filenames, reader_configs)
        res = reader.load(['ir_105', 'vis_06'], pad_data=False)

        # test that area_ids are harmonisation-conform <platform>_<instrument>_<service>_<resolution>
        assert res['vis_06'].attrs['area'].area_id == 'mtg_fci_fdss_1km'
        assert res['ir_105'].attrs['area'].area_id == 'mtg_fci_fdss_2km'

        area_def = res['ir_105'].attrs['area']
        # test area extents computation
        np.testing.assert_array_almost_equal(np.array(area_def.area_extent),
                                             np.array([-5568062.23065902, 5168057.7600648,
                                                       16704186.692027, 5568062.23065902]))

        # check that the projection is read in properly
        assert area_def.crs.coordinate_operation.method_name == 'Geostationary Satellite (Sweep Y)'
        assert area_def.crs.coordinate_operation.params[0].value == 0.0  # projection origin longitude
        assert area_def.crs.coordinate_operation.params[1].value == 35786400.0  # projection height
        assert area_def.crs.ellipsoid.semi_major_metre == 6378137.0
        assert area_def.crs.ellipsoid.inverse_flattening == 298.257223563
        assert area_def.crs.ellipsoid.is_semi_minor_computed


class TestFCIL1cNCReaderBadData(TestFCIL1cNCReader):
    """Test the FCI L1c NetCDF Reader for bad data input."""

    _alt_handler = FakeNetCDF4FileHandler3

    def test_handling_bad_data_ir(self, reader_configs, caplog):
        """Test handling of bad IR data."""
        from satpy.tests.utils import make_dataid

        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
        ]

        reader = _get_reader_with_filehandlers(filenames, reader_configs)
        with caplog.at_level("ERROR"):
            reader.load([make_dataid(
                name="ir_123",
                calibration="brightness_temperature")], pad_data=False)
            assert "cannot produce brightness temperature" in caplog.text

    def test_handling_bad_data_vis(self, reader_configs, caplog):
        """Test handling of bad VIS data."""
        from satpy.tests.utils import make_dataid

        filenames = [
            "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
            "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
            "20170410113925_20170410113934_N__C_0070_0067.nc",
        ]

        reader = _get_reader_with_filehandlers(filenames, reader_configs)
        with caplog.at_level("ERROR"):
            reader.load([make_dataid(
                name="vis_04",
                calibration="reflectance")], pad_data=False)
            assert "cannot produce reflectance" in caplog.text
