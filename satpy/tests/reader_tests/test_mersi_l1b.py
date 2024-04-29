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
"""Tests for the 'mersi2_l1b' reader."""
import os
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from satpy.tests.reader_tests.test_hdf5_utils import FakeHDF5FileHandler


def _get_calibration(num_scans, ftype):
    calibration = {
        f"Calibration/{ftype}_Cal_Coeff":
            xr.DataArray(
                da.ones((19, 3), chunks=1024),
                attrs={"Slope": np.array([1.] * 19), "Intercept": np.array([0.] * 19)},
                dims=("_bands", "_coeffs")),
        "Calibration/IR_Cal_Coeff":
            xr.DataArray(
                da.ones((6, 4, num_scans), chunks=1024),
                attrs={"Slope": np.array([1.] * 6), "Intercept": np.array([0.] * 6)},
                dims=("_bands", "_coeffs", "_scans")),
    }
    return calibration


def _get_250m_data(num_scans, rows_per_scan, num_cols, old_fy3ab_form=False):
    # Set some default attributes
    fill_value_name = "_FillValue" if old_fy3ab_form else "FillValue"
    key_prefix = "" if old_fy3ab_form else "Data/"

    def_attrs = {fill_value_name: 65535,
                 "valid_range": [0, 4095],
                 "Slope": np.array([1.] * 1), "Intercept": np.array([0.] * 1)
                 }
    nounits_attrs = {**def_attrs, **{"units": "NO"}}
    radunits_attrs = {**def_attrs, **{"units": "mW/ (m2 cm-1 sr)"}}

    data = {
        f"{key_prefix}EV_250_RefSB_b1":
            xr.DataArray(
                da.ones((num_scans * rows_per_scan, num_cols), chunks=1024, dtype=np.uint16),
                attrs=nounits_attrs,
                dims=("_rows", "_cols")),
        f"{key_prefix}EV_250_RefSB_b2":
            xr.DataArray(
                da.ones((num_scans * rows_per_scan, num_cols), chunks=1024, dtype=np.uint16),
                attrs=nounits_attrs,
                dims=("_rows", "_cols")),
        f"{key_prefix}EV_250_RefSB_b3":
            xr.DataArray(
                da.ones((num_scans * rows_per_scan, num_cols), chunks=1024, dtype=np.uint16),
                attrs=nounits_attrs,
                dims=("_rows", "_cols")),
        f"{key_prefix}EV_250_RefSB_b4":
            xr.DataArray(
                da.ones((num_scans * rows_per_scan, num_cols), chunks=1024, dtype=np.uint16),
                attrs=nounits_attrs,
                dims=("_rows", "_cols")),
        f"{key_prefix}EV_250_Emissive_b24":
            xr.DataArray(
                da.ones((num_scans * rows_per_scan, num_cols), chunks=1024, dtype=np.uint16),
                attrs=radunits_attrs,
                dims=("_rows", "_cols")),
        f"{key_prefix}EV_250_Emissive_b25":
            xr.DataArray(
                da.ones((num_scans * rows_per_scan, num_cols), chunks=1024, dtype=np.uint16),
                attrs=radunits_attrs,
                dims=("_rows", "_cols")),
        f"{key_prefix}EV_250_Emissive":
            xr.DataArray(
                da.ones((num_scans * rows_per_scan, num_cols), chunks=1024, dtype=np.uint16),
                attrs=nounits_attrs,
                dims=("_rows", "_cols")),
    }
    return data


def _get_500m_data(num_scans, rows_per_scan, num_cols):
    data = {
        "Data/EV_Reflectance":
            xr.DataArray(
                da.ones((5, num_scans * rows_per_scan, num_cols), chunks=1024,
                        dtype=np.uint16),
                attrs={
                    "Slope": np.array([1.] * 5), "Intercept": np.array([0.] * 5),
                    "FillValue": 65535,
                    "units": "NO",
                    "valid_range": [0, 4095],
                    "long_name": b"500m Earth View Science Data",
                },
                dims=("_ref_bands", "_rows", "_cols")),
        "Data/EV_Emissive":
            xr.DataArray(
                da.ones((3, num_scans * rows_per_scan, num_cols), chunks=1024,
                        dtype=np.uint16),
                attrs={
                    "Slope": np.array([1.] * 3), "Intercept": np.array([0.] * 3),
                    "FillValue": 65535,
                    "units": "mW/ (m2 cm-1 sr)",
                    "valid_range": [0, 25000],
                    "long_name": b"500m Emissive Bands Earth View "
                                 b"Science Data",
                },
                dims=("_ir_bands", "_rows", "_cols")),
    }
    return data


def _get_1km_data(num_scans, rows_per_scan, num_cols, old_fy3ab_form=False, mersi1=False):
    fill_value_name = "_FillValue" if old_fy3ab_form else "FillValue"
    key_prefix = "" if old_fy3ab_form else "Data/"
    radunits = "NO" if mersi1 else "mW/ (m2 cm-1 sr)"

    data = {
        "Data/EV_1KM_LL":
            xr.DataArray(
                da.ones((num_scans * rows_per_scan, num_cols), chunks=1024,
                        dtype=np.uint16),
                attrs={
                    "Slope": np.array([1.]), "Intercept": np.array([0.]),
                    "FillValue": 65535,
                    "units": "NO",
                    "valid_range": [0, 4095],
                    "long_name": b"1km Earth View Science Data",
                },
                dims=("_rows", "_cols")),
        f"{key_prefix}EV_1KM_RefSB":
            xr.DataArray(
                da.ones((15, num_scans * rows_per_scan, num_cols), chunks=1024,
                        dtype=np.uint16),
                attrs={
                    "Slope": np.array([1.] * 15), "Intercept": np.array([0.] * 15),
                    fill_value_name: 65535,
                    "units": "NO",
                    "valid_range": [0, 4095],
                    "long_name": b"1km Earth View Science Data",
                },
                dims=("_ref_bands", "_rows", "_cols")),
        "Data/EV_1KM_Emissive":
            xr.DataArray(
                da.ones((4, num_scans * rows_per_scan, num_cols), chunks=1024,
                        dtype=np.uint16),
                attrs={
                    "Slope": np.array([1.] * 4), "Intercept": np.array([0.] * 4),
                    "FillValue": 65535,
                    "units": "mW/ (m2 cm-1 sr)",
                    "valid_range": [0, 25000],
                    "long_name": b"1km Emissive Bands Earth View "
                                 b"Science Data",
                },
                dims=("_ir_bands", "_rows", "_cols")),
        f"{key_prefix}EV_250_Aggr.1KM_RefSB":
            xr.DataArray(
                da.ones((4, num_scans * rows_per_scan, num_cols), chunks=1024,
                        dtype=np.uint16),
                attrs={
                    "Slope": np.array([1.] * 4), "Intercept": np.array([0.] * 4),
                    fill_value_name: 65535,
                    "units": "NO",
                    "valid_range": [0, 4095],
                    "long_name": b"250m Reflective Bands Earth View "
                                 b"Science Data Aggregated to 1 km"
                },
                dims=("_ref250_bands", "_rows", "_cols")),
        f"{key_prefix}EV_250_Aggr.1KM_Emissive":
            xr.DataArray(
                da.ones((num_scans * rows_per_scan, num_cols), chunks=1024,
                        dtype=np.uint16),
                attrs={
                    "Slope": np.array([1.]), "Intercept": np.array([0.]),
                    fill_value_name: 65535,
                    "units": radunits,
                    "valid_range": [0, 4095],
                    "long_name": b"250m Emissive Bands Earth View "
                                 b"Science Data Aggregated to 1 km"
                },
                dims=("_rows", "_cols")) if mersi1 else
                xr.DataArray(
                    da.ones((4, num_scans * rows_per_scan, num_cols), chunks=1024,
                            dtype=np.uint16),
                    attrs={
                        "Slope": np.array([1.] * 2), "Intercept": np.array([0.] * 2),
                        "FillValue": 65535,
                        "units": "mW/ (m2 cm-1 sr)",
                        "valid_range": [0, 4095],
                        "long_name": b"250m Emissive Bands Earth View "
                                     b"Science Data Aggregated to 1 km"
                    },
                    dims=("_ir250_bands", "_rows", "_cols"))
    }
    return data


def _get_250m_ll_data(num_scans, rows_per_scan, num_cols):
    # Set some default attributes
    def_attrs = {"FillValue": 65535,
                 "valid_range": [0, 4095],
                 "Slope": np.array([1.]), "Intercept": np.array([0.]),
                 "long_name": b"250m Earth View Science Data",
                 "units": "mW/ (m2 cm-1 sr)",
                 }
    data = {
        "Data/EV_250_Emissive_b6":
            xr.DataArray(
                da.ones((num_scans * rows_per_scan, num_cols), chunks=1024, dtype=np.uint16),
                attrs=def_attrs,
                dims=("_rows", "_cols")),
        "Data/EV_250_Emissive_b7":
            xr.DataArray(
                da.ones((num_scans * rows_per_scan, num_cols), chunks=1024, dtype=np.uint16),
                attrs=def_attrs,
                dims=("_rows", "_cols")),
    }
    return data


def _get_geo_data(num_scans, rows_per_scan, num_cols, prefix):
    geo = {
        prefix + "Longitude":
            xr.DataArray(
                da.ones((num_scans * rows_per_scan, num_cols), chunks=1024),
                attrs={
                    "Slope": np.array([1.] * 1), "Intercept": np.array([0.] * 1),
                    "units": "degree",
                    "valid_range": [-90, 90],
                },
                dims=("_rows", "_cols")),
        prefix + "Latitude":
            xr.DataArray(
                da.ones((num_scans * rows_per_scan, num_cols), chunks=1024),
                attrs={
                    "Slope": np.array([1.] * 1), "Intercept": np.array([0.] * 1),
                    "units": "degree",
                    "valid_range": [-180, 180],
                },
                dims=("_rows", "_cols")),
        prefix + "SensorZenith":
            xr.DataArray(
                da.ones((num_scans * rows_per_scan, num_cols), chunks=1024),
                attrs={
                    "Slope": np.array([.01] * 1), "Intercept": np.array([0.] * 1),
                    "units": "degree",
                    "valid_range": [0, 28000],
                },
                dims=("_rows", "_cols")),
    }
    return geo


def make_test_data(dims):
    """Make test data."""
    return xr.DataArray(da.from_array(np.ones([dim for dim in dims], dtype=np.float32) * 10, [dim for dim in dims]))


class FakeHDF5FileHandler2(FakeHDF5FileHandler):
    """Swap-in HDF5 File Handler."""

    num_scans = 2
    num_cols = 2048

    @property
    def _rows_per_scan(self):
        return self.filetype_info.get("rows_per_scan", 10)

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        global_attrs = {
            "/attr/Observing Beginning Date": "2019-01-01",
            "/attr/Observing Ending Date": "2019-01-01",
            "/attr/Observing Beginning Time": "18:27:39.720",
            "/attr/Observing Ending Time": "18:38:36.728",
        }
        fy3a_attrs = {
            "/attr/VIR_Cal_Coeff": np.array([0.0, 1.0, 0.0] * 19),
        }
        fy3b_attrs = {
            "/attr/VIS_Cal_Coeff": np.array([0.0, 1.0, 0.0] * 19),
        }

        global_attrs, ftype = self._set_sensor_attrs(global_attrs)
        self._add_tbb_coefficients(global_attrs)
        data = self._get_data_file_content()

        test_content = {}
        test_content.update(global_attrs)
        if "fy3a_mersi1" in self.filetype_info["file_type"]:
            test_content.update(data[0])
            test_content.update(data[1])
        else:
            test_content.update(data)
        if "fy3a_mersi1" in self.filetype_info["file_type"]:
            test_content.update(fy3a_attrs)
        elif "fy3b_mersi1" in self.filetype_info["file_type"]:
            test_content.update(fy3b_attrs)
        if not self.filetype_info["file_type"].startswith(("fy3a_mersi1", "fy3b_mersi1")):
            test_content.update(_get_calibration(self.num_scans, ftype))
        return test_content

    def _set_sensor_attrs(self, global_attrs):
        if "fy3a_mersi1" in self.filetype_info["file_type"]:
            global_attrs["/attr/Satellite Name"] = "FY-3A"
            global_attrs["/attr/Sensor Identification Code"] = "MERSI"
            ftype = "VIS"
        elif "fy3b_mersi1" in self.filetype_info["file_type"]:
            global_attrs["/attr/Satellite Name"] = "FY-3B"
            global_attrs["/attr/Sensor Identification Code"] = "MERSI"
            ftype = "VIS"
        elif "fy3c_mersi1" in self.filetype_info["file_type"]:
            global_attrs["/attr/Satellite Name"] = "FY-3C"
            global_attrs["/attr/Sensor Identification Code"] = "MERSI"
            ftype = "VIS"
        elif "mersi2_l1b" in self.filetype_info["file_type"]:
            global_attrs["/attr/Satellite Name"] = "FY-3D"
            global_attrs["/attr/Sensor Identification Code"] = "MERSI"
            ftype = "VIS"
        elif "mersi_ll" in self.filetype_info["file_type"]:
            global_attrs["/attr/Satellite Name"] = "FY-3E"
            global_attrs["/attr/Sensor Identification Code"] = "MERSI LL"
            ftype = "VIS"
        elif "mersi_rm" in self.filetype_info["file_type"]:
            global_attrs["/attr/Satellite Name"] = "FY-3G"
            global_attrs["/attr/Sensor Identification Code"] = "MERSI RM"
            ftype = "RSB"
        return global_attrs, ftype

    def _get_data_file_content(self):
        if "fy3a_mersi1" in self.filetype_info["file_type"]:
            return self._add_band_data_file_content(), self._add_geo_data_file_content()
        else:
            if "_geo" in self.filetype_info["file_type"]:
                return self._add_geo_data_file_content()
            else:
                return self._add_band_data_file_content()

    def _add_geo_data_file_content(self):
        num_scans = self.num_scans
        rows_per_scan = self._rows_per_scan
        return _get_geo_data(num_scans, rows_per_scan,
                             self._num_cols_for_file_type,
                             self._geo_prefix_for_file_type)

    def _add_band_data_file_content(self):
        num_cols = self._num_cols_for_file_type
        num_scans = self.num_scans
        rows_per_scan = self._rows_per_scan
        is_mersi1 = self.filetype_info["file_type"].startswith(("fy3a_mersi1", "fy3b_mersi1", "fy3c_mersi1"))
        is_fy3ab_mersi1 = self.filetype_info["file_type"].startswith(("fy3a_mersi1", "fy3b_mersi1"))
        is_mersi2 = self.filetype_info["file_type"].startswith("mersi2_")
        is_mersill = self.filetype_info["file_type"].startswith("mersi_ll")
        is_1km = "_1000" in self.filetype_info["file_type"]
        is_250m = "_250" in self.filetype_info["file_type"]

        if is_1km:
            return _get_1km_data(num_scans, rows_per_scan, num_cols, old_fy3ab_form=is_fy3ab_mersi1, mersi1=is_mersi1)
        elif is_250m:
            if is_mersi1 or is_mersi2:
                return _get_250m_data(num_scans, rows_per_scan, num_cols, old_fy3ab_form=is_fy3ab_mersi1)
            elif is_mersill:
                return _get_250m_ll_data(num_scans, rows_per_scan, num_cols)
            else:
                return
        else:
            return _get_500m_data(num_scans, rows_per_scan, num_cols)

    def _add_tbb_coefficients(self, global_attrs):
        if not self.filetype_info["file_type"].startswith("mersi2_"):
            return

        if "_1000" in self.filetype_info["file_type"]:
            global_attrs["/attr/TBB_Trans_Coefficient_A"] = np.array([1.0] * 6)
            global_attrs["/attr/TBB_Trans_Coefficient_B"] = np.array([0.0] * 6)
        else:
            global_attrs["/attr/TBB_Trans_Coefficient_A"] = np.array([0.0] * 6)
            global_attrs["/attr/TBB_Trans_Coefficient_B"] = np.array([0.0] * 6)

    @property
    def _num_cols_for_file_type(self):
        return self.num_cols if "1000" in self.filetype_info["file_type"] else self.num_cols * 2

    @property
    def _geo_prefix_for_file_type(self):
        if self.filetype_info["file_type"].startswith(("fy3a_mersi1", "fy3b_mersi1")):
            return ""
        else:
            if "1000" in self.filetype_info["file_type"]:
                return "Geolocation/"
            elif "500" in self.filetype_info["file_type"]:
                return "Geolocation/"
            else:
                return ""


def _test_helper(res, band_list, exp_cal, exp_unit, exp_shape):
    """Remove test code duplication."""
    for band in band_list:
        assert res[band].attrs["calibration"] == exp_cal
        assert res[band].attrs["units"] == exp_unit
        assert res[band].shape == exp_shape


class MERSIL1BTester:
    """Test MERSI1/2/LL/RM L1B Reader."""

    def setup_method(self):
        """Wrap HDF5 file handler with our own fake handler."""
        from satpy._config import config_search_paths
        from satpy.readers.mersi_l1b import MERSIL1B
        self.reader_configs = config_search_paths(os.path.join("readers", self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(MERSIL1B, "__bases__", (FakeHDF5FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def teardown_method(self):
        """Stop wrapping the HDF5 file handler."""
        self.p.stop()


class MERSI1L1BTester(MERSIL1BTester):
    """Test MERSI1 L1B Reader."""

    yaml_file = ""
    filenames_1000m = []
    filenames_250m = []
    filenames_all = []

    def test_all_resolutions(self):
        """Test loading data when all resolutions are available."""
        from satpy.dataset.data_dict import get_key
        from satpy.readers import load_reader
        from satpy.tests.utils import make_dataid
        filenames = self.filenames_all
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert len(files) == len(filenames)
        reader.create_filehandlers(files)
        # Make sure we have some files
        assert reader.file_handlers

        # Verify that we have multiple resolutions for:
        #     - Bands 1-4 (visible)
        #     - Bands 5 (IR)
        available_datasets = reader.available_dataset_ids
        for band_name in ("1", "2", "3", "4", "5"):
            num_results = 2
            ds_id = make_dataid(name=band_name, resolution=250)
            res = get_key(ds_id, available_datasets,
                          num_results=num_results, best=False)
            assert num_results == len(res)
            ds_id = make_dataid(name=band_name, resolution=1000)
            res = get_key(ds_id, available_datasets,
                          num_results=num_results, best=False)
            assert num_results == len(res)

        res = reader.load(["1", "2", "3", "4", "5", "6", "7", "8"])
        assert len(res) == 8
        _test_helper(res, ["1", "2", "3", "4"], "reflectance", "%", (2 * 40, 2048 * 2))
        assert res["5"].shape == (2 * 40, 2048 * 2)
        assert res["5"].attrs["calibration"] == "brightness_temperature"
        assert res["5"].attrs["units"] == "K"
        assert res["6"].shape == (2 * 10, 2048)
        assert res["6"].attrs["calibration"] == "reflectance"
        assert res["6"].attrs["units"] == "%"

    def test_counts_calib(self):
        """Test loading data at counts calibration."""
        from satpy.readers import load_reader
        from satpy.tests.utils import make_dataid
        filenames = self.filenames_all
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert len(files) == len(filenames)
        reader.create_filehandlers(files)
        # Make sure we have some files
        assert reader.file_handlers

        ds_ids = []
        for band_name in ["1", "2", "3", "4", "5", "6", "19", "20"]:
            ds_ids.append(make_dataid(name=band_name, calibration="counts"))
        ds_ids.append(make_dataid(name="satellite_zenith_angle"))
        res = reader.load(ds_ids)
        assert len(res) == 9
        _test_helper(res, ["1", "2", "3", "4", "5"], "counts", "1", (2 * 40, 2048 * 2))
        _test_helper(res, ["6", "19", "20"], "counts", "1", (2 * 10, 2048))

    def test_1km_resolutions(self):
        """Test loading data when only 1km resolutions are available."""
        from satpy.dataset.data_dict import get_key
        from satpy.readers import load_reader
        from satpy.tests.utils import make_dataid
        filenames = self.filenames_1000m
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert len(files) == len(filenames)
        reader.create_filehandlers(files)
        # Make sure we have some files
        assert reader.file_handlers

        # Verify that we have multiple resolutions for:
        #     - Bands 1-4 (visible)
        #     - Bands 5 (IR)
        available_datasets = reader.available_dataset_ids
        for band_name in ("1", "2", "3", "4", "5"):
            num_results = 2
            ds_id = make_dataid(name=band_name, resolution=250)
            with pytest.raises(KeyError):
                get_key(ds_id, available_datasets, num_results=num_results, best=False)
            ds_id = make_dataid(name=band_name, resolution=1000)
            res = get_key(ds_id, available_datasets,
                          num_results=num_results, best=False)
            assert num_results == len(res)

        res = reader.load(["1", "2", "3", "4", "5", "6", "7", "8"])
        assert len(res) == 8
        _test_helper(res, ["1", "2", "3", "4", "6", "7", "8"], "reflectance", "%", (2 * 10, 2048))
        assert res["5"].shape == (2 * 10, 2048)
        assert res["5"].attrs["calibration"] == "brightness_temperature"
        assert res["5"].attrs["units"] == "K"

    def test_250_resolutions(self):
        """Test loading data when only 250m resolutions are available."""
        from satpy.dataset.data_dict import get_key
        from satpy.readers import load_reader
        from satpy.tests.utils import make_dataid
        filenames = self.filenames_250m
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert len(files) == len(filenames)
        reader.create_filehandlers(files)
        # Make sure we have some files
        assert reader.file_handlers

        # Verify that we have multiple resolutions for:
        #     - Bands 1-4 (visible)
        #     - Bands 5 (IR)
        available_datasets = reader.available_dataset_ids
        for band_name in ("1", "2", "3", "4", "5"):
            num_results = 2
            ds_id = make_dataid(name=band_name, resolution=250)
            res = get_key(ds_id, available_datasets,
                          num_results=num_results, best=False)
            assert num_results == len(res)
            ds_id = make_dataid(name=band_name, resolution=1000)
            with pytest.raises(KeyError):
                get_key(ds_id, available_datasets, num_results=num_results, best=False)

        res = reader.load(["1", "2", "3", "4", "5", "6", "7"])
        assert len(res) == 5
        with pytest.raises(KeyError):
            res.__getitem__("6")
        with pytest.raises(KeyError):
            res.__getitem__("7")
        _test_helper(res, ["1", "2", "3", "4"], "reflectance", "%", (2 * 40, 2048 * 2))
        assert res["5"].shape == (2 * 40, 2048 * 2)
        assert res["5"].attrs["calibration"] == "brightness_temperature"
        assert res["5"].attrs["units"] == "K"


class TestFY3AMERSI1L1B(MERSI1L1BTester):
    """Test the FY3A MERSI1 L1B reader."""

    yaml_file = "fy3a_mersi1_l1b.yaml"
    filenames_1000m = ["FY3A_MERSI_GBAL_L1_20090601_1200_1000M_MS.hdf"]
    filenames_250m = ["FY3A_MERSI_GBAL_L1_20090601_1200_0250M_MS.hdf"]
    filenames_all = filenames_1000m + filenames_250m


class TestFY3BMERSI1L1B(MERSI1L1BTester):
    """Test the FY3A MERSI1 L1B reader."""

    yaml_file = "fy3b_mersi1_l1b.yaml"
    filenames_1000m = ["FY3B_MERSI_GBAL_L1_20110824_1850_1000M_MS.hdf"]
    filenames_250m = ["FY3B_MERSI_GBAL_L1_20110824_1850_0250M_MS.hdf", "FY3B_MERSI_GBAL_L1_20110824_1850_GEOXX_MS.hdf"]
    filenames_all = filenames_1000m + filenames_250m


class TestFY3CMERSI1L1B(MERSI1L1BTester):
    """Test the FY3A MERSI1 L1B reader."""

    yaml_file = "fy3c_mersi1_l1b.yaml"
    filenames_1000m = ["FY3C_MERSI_GBAL_L1_20131002_1835_1000M_MS.hdf", "FY3C_MERSI_GBAL_L1_20131002_1835_GEO1K_MS.hdf"]
    filenames_250m = ["FY3C_MERSI_GBAL_L1_20131002_1835_0250M_MS.hdf", "FY3C_MERSI_GBAL_L1_20131002_1835_GEOQK_MS.hdf"]
    filenames_all = filenames_1000m + filenames_250m



class TestMERSI2L1B(MERSIL1BTester):
    """Test the FY3D MERSI2 L1B reader."""

    yaml_file = "mersi2_l1b.yaml"
    filenames_1000m = ["tf2019071182739.FY3D-X_MERSI_1000M_L1B.HDF", "tf2019071182739.FY3D-X_MERSI_GEO1K_L1B.HDF"]
    filenames_250m = ["tf2019071182739.FY3D-X_MERSI_0250M_L1B.HDF", "tf2019071182739.FY3D-X_MERSI_GEOQK_L1B.HDF"]
    filenames_all = filenames_1000m + filenames_250m

    def test_all_resolutions(self):
        """Test loading data when all resolutions are available."""
        from satpy.dataset.data_dict import get_key
        from satpy.readers import load_reader
        from satpy.tests.utils import make_dataid
        filenames = self.filenames_all
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert 4 == len(files)
        reader.create_filehandlers(files)
        # Make sure we have some files
        assert reader.file_handlers

        # Verify that we have multiple resolutions for:
        #     - Bands 1-4 (visible)
        #     - Bands 24-25 (IR)
        available_datasets = reader.available_dataset_ids
        for band_name in ("1", "2", "3", "4", "24", "25"):
            if band_name in ("24", "25"):
                # don't know how to get radiance for IR bands
                num_results = 2
            else:
                num_results = 3
            ds_id = make_dataid(name=band_name, resolution=250)
            res = get_key(ds_id, available_datasets,
                          num_results=num_results, best=False)
            assert num_results == len(res)
            ds_id = make_dataid(name=band_name, resolution=1000)
            res = get_key(ds_id, available_datasets,
                          num_results=num_results, best=False)
            assert num_results == len(res)

        res = reader.load(["1", "2", "3", "4", "5", "20", "24", "25"])
        assert len(res) == 8
        _test_helper(res, ["1", "2", "3", "4"], "reflectance", "%", (2 * 40, 2048 * 2))
        _test_helper(res, ["24", "25"], "brightness_temperature", "K", (2 * 40, 2048 * 2))
        assert res["5"].shape == (2 * 10, 2048)
        assert res["5"].attrs["calibration"] == "reflectance"
        assert res["5"].attrs["units"] == "%"
        assert res["20"].shape == (2 * 10, 2048)
        assert res["20"].attrs["calibration"] == "brightness_temperature"
        assert res["20"].attrs["units"] == "K"

    def test_counts_calib(self):
        """Test loading data at counts calibration."""
        from satpy.readers import load_reader
        from satpy.tests.utils import make_dataid
        filenames = self.filenames_all
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert 4 == len(files)
        reader.create_filehandlers(files)
        # Make sure we have some files
        assert reader.file_handlers

        ds_ids = []
        for band_name in ["1", "2", "3", "4", "5", "20", "24", "25"]:
            ds_ids.append(make_dataid(name=band_name, calibration="counts"))
        ds_ids.append(make_dataid(name="satellite_zenith_angle"))
        res = reader.load(ds_ids)
        assert len(res) == 9
        _test_helper(res, ["1", "2", "3", "4", "24", "25"], "counts", "1", (2 * 40, 2048 * 2))
        _test_helper(res, ["5", "20"], "counts", "1", (2 * 10, 2048))

    def test_rad_calib(self):
        """Test loading data at radiance calibration."""
        from satpy.readers import load_reader
        from satpy.tests.utils import make_dataid
        filenames = self.filenames_all
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert 4 == len(files)
        reader.create_filehandlers(files)
        # Make sure we have some files
        assert reader.file_handlers

        ds_ids = []
        for band_name in ["1", "2", "3", "4", "5"]:
            ds_ids.append(make_dataid(name=band_name, calibration="radiance"))
        res = reader.load(ds_ids)
        assert len(res) == 5
        _test_helper(res, ["1", "2", "3", "4"], "radiance", "mW/ (m2 cm-1 sr)", (2 * 40, 2048 * 2))
        assert res["5"].shape == (2 * 10, 2048)
        assert res["5"].attrs["calibration"] == "radiance"
        assert res["5"].attrs["units"] == "mW/ (m2 cm-1 sr)"

    def test_1km_resolutions(self):
        """Test loading data when only 1km resolutions are available."""
        from satpy.dataset.data_dict import get_key
        from satpy.readers import load_reader
        from satpy.tests.utils import make_dataid
        filenames = self.filenames_1000m
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert 2 == len(files)
        reader.create_filehandlers(files)
        # Make sure we have some files
        assert reader.file_handlers

        # Verify that we have multiple resolutions for:
        #     - Bands 1-4 (visible)
        #     - Bands 24-25 (IR)
        available_datasets = reader.available_dataset_ids
        for band_name in ("1", "2", "3", "4", "24", "25"):
            if band_name in ("24", "25"):
                # don't know how to get radiance for IR bands
                num_results = 2
            else:
                num_results = 3
            ds_id = make_dataid(name=band_name, resolution=250)
            with pytest.raises(KeyError):
                get_key(ds_id, available_datasets, num_results=num_results, best=False)
            ds_id = make_dataid(name=band_name, resolution=1000)
            res = get_key(ds_id, available_datasets,
                          num_results=num_results, best=False)
            assert num_results == len(res)

        res = reader.load(["1", "2", "3", "4", "5", "20", "24", "25"])
        assert len(res) == 8
        _test_helper(res, ["1", "2", "3", "4"], "reflectance", "%", (2 * 10, 2048))
        _test_helper(res, ["24", "25"], "brightness_temperature", "K", (2 * 10, 2048))
        assert res["5"].shape == (2 * 10, 2048)
        assert res["5"].attrs["calibration"] == "reflectance"
        assert res["5"].attrs["units"] == "%"
        assert res["20"].shape == (2 * 10, 2048)
        assert res["20"].attrs["calibration"] == "brightness_temperature"
        assert res["20"].attrs["units"] == "K"

    def test_250_resolutions(self):
        """Test loading data when only 250m resolutions are available."""
        from satpy.dataset.data_dict import get_key
        from satpy.readers import load_reader
        from satpy.tests.utils import make_dataid
        filenames = self.filenames_250m
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert 2 == len(files)
        reader.create_filehandlers(files)
        # Make sure we have some files
        assert reader.file_handlers

        # Verify that we have multiple resolutions for:
        #     - Bands 1-4 (visible)
        #     - Bands 24-25 (IR)
        available_datasets = reader.available_dataset_ids
        for band_name in ("1", "2", "3", "4", "24", "25"):
            if band_name in ("24", "25"):
                # don't know how to get radiance for IR bands
                num_results = 2
            else:
                num_results = 3
            ds_id = make_dataid(name=band_name, resolution=250)
            res = get_key(ds_id, available_datasets,
                          num_results=num_results, best=False)
            assert num_results == len(res)
            ds_id = make_dataid(name=band_name, resolution=1000)
            with pytest.raises(KeyError):
                get_key(ds_id, available_datasets, num_results=num_results, best=False)

        res = reader.load(["1", "2", "3", "4", "5", "20", "24", "25"])
        assert len(res) == 6
        with pytest.raises(KeyError):
            res.__getitem__("5")
        with pytest.raises(KeyError):
            res.__getitem__("20")
        # _test_helper(res)
        assert res["24"].shape == (2 * 40, 2048 * 2)
        assert res["24"].attrs["calibration"] == "brightness_temperature"
        assert res["24"].attrs["units"] == "K"
        assert res["25"].shape == (2 * 40, 2048 * 2)
        assert res["25"].attrs["calibration"] == "brightness_temperature"
        assert res["25"].attrs["units"] == "K"


class TestMERSILLL1B(MERSIL1BTester):
    """Test the FY3E MERSI-LL L1B reader."""

    yaml_file = "mersi_ll_l1b.yaml"
    filenames_1000m = ["FY3E_MERSI_GRAN_L1_20230410_1910_1000M_V0.HDF", "FY3E_MERSI_GRAN_L1_20230410_1910_GEO1K_V0.HDF"]
    filenames_250m = ["FY3E_MERSI_GRAN_L1_20230410_1910_0250M_V0.HDF", "FY3E_MERSI_GRAN_L1_20230410_1910_GEOQK_V0.HDF"]
    filenames_all = filenames_1000m + filenames_250m

    def test_all_resolutions(self):
        """Test loading data when all resolutions are available."""
        from satpy.dataset.data_dict import get_key
        from satpy.readers import load_reader
        from satpy.tests.utils import make_dataid
        filenames = self.filenames_all
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert 4 == len(files)
        reader.create_filehandlers(files)
        # Make sure we have some files
        assert reader.file_handlers

        # Verify that we have multiple resolutions for:
        #     - Bands 1-4 (visible)
        #     - Bands 24-25 (IR)
        available_datasets = reader.available_dataset_ids
        for band_name in ("6", "7"):
            num_results = 2
            ds_id = make_dataid(name=band_name, resolution=250)
            res = get_key(ds_id, available_datasets,
                          num_results=num_results, best=False)
            assert num_results == len(res)
            ds_id = make_dataid(name=band_name, resolution=1000)
            res = get_key(ds_id, available_datasets,
                          num_results=num_results, best=False)
            assert num_results == len(res)

        res = reader.load(["1", "2", "4", "7"])
        assert len(res) == 4
        assert res["4"].shape == (2 * 10, 2048)
        assert res["1"].attrs["calibration"] == "radiance"
        assert res["1"].attrs["units"] == "mW/ (m2 cm-1 sr)"
        assert res["2"].shape == (2 * 10, 2048)
        assert res["2"].attrs["calibration"] == "brightness_temperature"
        assert res["2"].attrs["units"] == "K"
        assert res["7"].shape == (2 * 40, 2048 * 2)
        assert res["7"].attrs["calibration"] == "brightness_temperature"
        assert res["7"].attrs["units"] == "K"

    def test_rad_calib(self):
        """Test loading data at radiance calibration."""
        from satpy.readers import load_reader
        from satpy.tests.utils import make_dataid
        filenames = self.filenames_all
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert 4 == len(files)
        reader.create_filehandlers(files)
        # Make sure we have some files
        assert reader.file_handlers

        ds_ids = []
        for band_name in ["1", "3", "4", "6", "7"]:
            ds_ids.append(make_dataid(name=band_name, calibration="radiance"))
        res = reader.load(ds_ids)
        assert len(res) == 5
        _test_helper(res, ["1", "3", "4"], "radiance", "mW/ (m2 cm-1 sr)", (2 * 10, 2048))
        _test_helper(res, ["6", "7"], "radiance", "mW/ (m2 cm-1 sr)", (2 * 40, 2048 * 2))

    def test_1km_resolutions(self):
        """Test loading data when only 1km resolutions are available."""
        from satpy.dataset.data_dict import get_key
        from satpy.readers import load_reader
        from satpy.tests.utils import make_dataid
        filenames = self.filenames_1000m
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert 2 == len(files)
        reader.create_filehandlers(files)
        # Make sure we have some files
        assert reader.file_handlers

        # Verify that we have multiple resolutions for:
        #     - Band 6-7 (IR)
        #     - Bands 24-25 (IR)
        available_datasets = reader.available_dataset_ids
        for band_name in ("1", "2", "3", "4", "6", "7"):
            if band_name == "1":
                # don't know how to get anything apart from radiance for LL band
                num_results = 1
            else:
                num_results = 2
            ds_id = make_dataid(name=band_name, resolution=250)
            with pytest.raises(KeyError):
                get_key(ds_id, available_datasets, num_results=num_results, best=False)
            ds_id = make_dataid(name=band_name, resolution=1000)
            res = get_key(ds_id, available_datasets,
                          num_results=num_results, best=False)
            if band_name == "1":
                assert num_results == len([res])
            else:
                assert num_results == len(res)

        res = reader.load(["1", "2", "3", "5", "6", "7"])
        assert len(res) == 6
        assert res["1"].shape == (2 * 10, 2048)
        assert res["1"].attrs["calibration"] == "radiance"
        assert res["1"].attrs["units"] == "mW/ (m2 cm-1 sr)"
        _test_helper(res, ["2", "3", "5", "6", "7"], "brightness_temperature", "K", (2 * 10, 2048))

    def test_250_resolutions(self):
        """Test loading data when only 250m resolutions are available."""
        from satpy.dataset.data_dict import get_key
        from satpy.readers import load_reader
        from satpy.tests.utils import make_dataid
        filenames = self.filenames_250m
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert 2 == len(files)
        reader.create_filehandlers(files)
        # Make sure we have some files
        assert reader.file_handlers

        # Verify that we have multiple resolutions for:
        #     - Bands 6-7
        available_datasets = reader.available_dataset_ids
        for band_name in ("6", "7"):
            num_results = 2
            ds_id = make_dataid(name=band_name, resolution=250)
            res = get_key(ds_id, available_datasets,
                          num_results=num_results, best=False)
            assert num_results == len(res)
            ds_id = make_dataid(name=band_name, resolution=1000)
            with pytest.raises(KeyError):
                get_key(ds_id, available_datasets, num_results=num_results, best=False)

        res = reader.load(["1", "6", "7"])
        assert 2 == len(res)
        with pytest.raises(KeyError):
            res.__getitem__("1")
        assert (2 * 40, 2048 * 2) == res["6"].shape
        assert "brightness_temperature" == res["6"].attrs["calibration"]
        assert "K" == res["6"].attrs["units"]
        assert (2 * 40, 2048 * 2) == res["7"].shape
        assert "brightness_temperature" == res["7"].attrs["calibration"]
        assert "K" == res["7"].attrs["units"]


class TestMERSIRML1B(MERSIL1BTester):
    """Test the FY3E MERSI-RM L1B reader."""

    yaml_file = "mersi_rm_l1b.yaml"
    filenames_500m = ["FY3G_MERSI_GRAN_L1_20230410_1910_0500M_V1.HDF",
                      "FY3G_MERSI_GRAN_L1_20230410_1910_GEOHK_V1.HDF",
                      ]

    def test_500m_resolution(self):
        """Test loading data when all resolutions are available."""
        from satpy.readers import load_reader
        filenames = self.filenames_500m
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert 2 == len(files)
        reader.create_filehandlers(files)
        # Make sure we have some files
        assert reader.file_handlers

        res = reader.load(["1", "2", "4", "7"])
        assert len(res) == 4
        assert res["4"].shape == (2 * 10, 4096)
        assert res["1"].attrs["calibration"] == "reflectance"
        assert res["1"].attrs["units"] == "%"
        assert res["2"].shape == (2 * 10, 4096)
        assert res["2"].attrs["calibration"] == "reflectance"
        assert res["2"].attrs["units"] == "%"
        assert res["7"].shape == (20, 2048 * 2)
        assert res["7"].attrs["calibration"] == "brightness_temperature"
        assert res["7"].attrs["units"] == "K"

    def test_rad_calib(self):
        """Test loading data at radiance calibration."""
        from satpy.readers import load_reader
        from satpy.tests.utils import make_dataid
        filenames = self.filenames_500m
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert 2 == len(files)
        reader.create_filehandlers(files)
        # Make sure we have some files
        assert reader.file_handlers

        band_names = ["1", "3", "4", "6", "7"]
        ds_ids = []
        for band_name in band_names:
            ds_ids.append(make_dataid(name=band_name, calibration="radiance"))
        res = reader.load(ds_ids)
        assert len(res) == 5
        for band_name in band_names:
          assert res[band_name].shape == (20, 4096)
          assert res[band_name].attrs["calibration"] == "radiance"
          assert res[band_name].attrs["units"] == "mW/ (m2 cm-1 sr)"
