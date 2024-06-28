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
        "Calibration/Solar_Irradiance":
            xr.DataArray(
                da.ones((19, ), chunks=1024),
                attrs={"Slope": np.array([1.] * 19), "Intercept": np.array([0.] * 19)},
                dims=("_bands")),
        "Calibration/Solar_Irradiance_LL":
            xr.DataArray(
                da.ones((1, ), chunks=1024),
                attrs={"Slope": np.array([1.]), "Intercept": np.array([0.])},
                dims=("_bands")),
        "Calibration/IR_Cal_Coeff":
            xr.DataArray(
                da.ones((6, 4, num_scans), chunks=1024),
                attrs={"Slope": np.array([1.] * 6), "Intercept": np.array([0.] * 6)},
                dims=("_bands", "_coeffs", "_scans")),
    }
    return calibration


def _get_250m_data(num_scans, rows_per_scan, num_cols, filetype_info):
    # Set some default attributes
    is_fy3ab_mersi1 = filetype_info["file_type"].startswith(("fy3a_mersi1", "fy3b_mersi1"))

    fill_value_name = "_FillValue" if is_fy3ab_mersi1 else "FillValue"
    key_prefix = "" if is_fy3ab_mersi1 else "Data/"

    def_attrs = {fill_value_name: 65535,
                 "valid_range": [0, 4095],
                 "Slope": np.array([1.] * 1), "Intercept": np.array([0.] * 1)
                 }
    nounits_attrs = {**def_attrs, **{"units": "NO"}}
    radunits_attrs = {**def_attrs, **{"units": "mW/ (m2 cm-1 sr)"}}
    valid_range_none_attrs = radunits_attrs.copy()
    valid_range_none_attrs["valid_range"] = None

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
                attrs=valid_range_none_attrs,
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


def _get_1km_data(num_scans, rows_per_scan, num_cols, filetype_info):
    is_mersi1 = filetype_info["file_type"].startswith(("fy3a_mersi1", "fy3b_mersi1", "fy3c_mersi1"))
    is_fy3ab_mersi1 = filetype_info["file_type"].startswith(("fy3a_mersi1", "fy3b_mersi1"))

    fill_value_name = "_FillValue" if is_fy3ab_mersi1 else "FillValue"
    key_prefix = "" if is_fy3ab_mersi1 else "Data/"
    radunits = "NO" if is_mersi1 else "mW/ (m2 cm-1 sr)"

    data = {"Data/EV_1KM_LL":
            xr.DataArray(da.ones((num_scans * rows_per_scan, num_cols), chunks=1024, dtype=np.uint16),
                attrs={"Slope": np.array([1.]), "Intercept": np.array([0.]),
                       "FillValue": 65535,
                       "units": "NO",
                       "valid_range": [0, 4095],
                       "long_name": b"1km Earth View Science Data"},
                dims=("_rows", "_cols")),
            f"{key_prefix}EV_1KM_RefSB":
            xr.DataArray(da.ones((15, num_scans * rows_per_scan, num_cols), chunks=1024, dtype=np.uint16),
                attrs={"Slope": np.array([1.] * 15), "Intercept": np.array([0.] * 15),
                       fill_value_name: 65535,
                       "units": "NO",
                       "valid_range": [0, 4095],
                       "long_name": b"1km Earth View Science Data"},
                dims=("_ref_bands", "_rows", "_cols")),
        "Data/EV_1KM_Emissive":
            xr.DataArray(da.ones((4, num_scans * rows_per_scan, num_cols), chunks=1024, dtype=np.uint16),
                attrs={"Slope": np.array([1.] * 4), "Intercept": np.array([0.] * 4),
                       "FillValue": 65535,
                       "units": "mW/ (m2 cm-1 sr)",
                       "valid_range": [0, 25000],
                       "long_name": b"1km Emissive Bands Earth View Science Data"},
                dims=("_ir_bands", "_rows", "_cols")),
        f"{key_prefix}EV_250_Aggr.1KM_RefSB":
            xr.DataArray(da.ones((4, num_scans * rows_per_scan, num_cols), chunks=1024, dtype=np.uint16),
                attrs={"Slope": np.array([1.] * 4), "Intercept": np.array([0.] * 4),
                       fill_value_name: 65535,
                       "units": "NO",
                       "valid_range": [0, 4095],
                       "long_name": b"250m Reflective Bands Earth View Science Data Aggregated to 1 km"},
                dims=("_ref250_bands", "_rows", "_cols")),
        f"{key_prefix}EV_250_Aggr.1KM_Emissive":
            xr.DataArray(da.ones((num_scans * rows_per_scan, num_cols), chunks=1024, dtype=np.uint16),
                attrs={"Slope": np.array([1.]), "Intercept": np.array([0.]),
                       fill_value_name: 65535,
                       "units": radunits,
                       "valid_range": [0, 4095],
                       "long_name": b"250m Emissive Bands Earth View Science Data Aggregated to 1 km"},
                dims=("_rows", "_cols")) if is_mersi1 else
                xr.DataArray(da.ones((2, num_scans * rows_per_scan, num_cols), chunks=1024, dtype=np.uint16),
                    attrs={"Slope": np.array([1.] * 2), "Intercept": np.array([0.] * 2),
                           "FillValue": 65535,
                           "units": "mW/ (m2 cm-1 sr)",
                           "valid_range": [0, 4095],
                           "long_name": b"250m Emissive Bands Earth View Science Data Aggregated to 1 km"},
                    dims=("_ir250_bands", "_rows", "_cols")),
        f"{key_prefix}SensorZenith":
                xr.DataArray(
                    da.ones((num_scans * rows_per_scan, num_cols), chunks=1024),
                    attrs={
                        "Slope": np.array([.01] * 1), "Intercept": np.array([0.] * 1),
                        "units": "degree",
                        "valid_range": [0, 28000],
                    },
                    dims=("_rows", "_cols")),
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
        fy3d_attrs = {
            "/attr/Solar_Irradiance": np.array([1.0] * 19),
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
        elif "mersi2" in self.filetype_info["file_type"]:
            test_content.update(fy3d_attrs)
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
            ftype = "LL"
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
        is_mersill = self.filetype_info["file_type"].startswith("mersi_ll")
        is_1km = "_1000" in self.filetype_info["file_type"]
        is_250m = "_250" in self.filetype_info["file_type"]

        if is_1km:
            return _get_1km_data(num_scans, rows_per_scan, num_cols, self.filetype_info)
        elif is_250m:
            if is_mersill:
                return _get_250m_ll_data(num_scans, rows_per_scan, num_cols)
            else:
                return _get_250m_data(num_scans, rows_per_scan, num_cols, self.filetype_info)
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


def _assert_bands_mda_as_exp(res, band_list, exp_result):
    """Remove test code duplication."""
    exp_cal = exp_result[0]
    exp_unit = exp_result[1]
    exp_shape = exp_result[2]
    for band in band_list:
        assert res[band].attrs["calibration"] == exp_cal
        assert res[band].attrs["units"] == exp_unit
        assert res[band].shape == exp_shape


def _test_find_files_and_readers(reader_config, filenames):
    """Test file and reader search."""
    from satpy.readers import load_reader
    reader = load_reader(reader_config)
    files = reader.select_files_from_pathnames(filenames)
    # Make sure we have some files
    reader.create_filehandlers(files)
    assert len(files) == len(filenames)
    assert reader.file_handlers
    return reader


def _test_multi_resolutions(available_datasets, band_list, test_resolution, cal_results_number):
    """Test some bands have multiple resolutions."""
    for band_name in band_list:
        from satpy.dataset.data_dict import get_key
        from satpy.tests.utils import make_dataid
        ds_id = make_dataid(name=band_name, resolution=250)
        if test_resolution == "1000":
            with pytest.raises(KeyError):
                get_key(ds_id, available_datasets, num_results=cal_results_number, best=False)
        else:

            res = get_key(ds_id, available_datasets, num_results=cal_results_number, best=False)
            assert len(res) == cal_results_number

        ds_id = make_dataid(name=band_name, resolution=1000)
        if test_resolution == "250":
            with pytest.raises(KeyError):
                get_key(ds_id, available_datasets, num_results=cal_results_number, best=False)
        else:

            res = get_key(ds_id, available_datasets, num_results=cal_results_number, best=False)
            assert len(res) == cal_results_number


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


class MERSI12llL1BTester(MERSIL1BTester):
    """Test MERSI1/2/LL L1B Reader."""

    yaml_file: str = ""
    filenames_1000m: list= []
    filenames_250m: list = []
    filenames_all: list = []
    vis_250_bands: list = []
    ir_250_bands: list = []
    vis_1000_bands: list = []
    ir_1000_bands: list = []
    bands_1000: list = []
    bands_250: list = []

    def test_all_resolutions(self):
        """Test loading data when all resolutions or specific one are available."""
        resolution_list = ["all", "250", "1000"]
        file_list = [self.filenames_all, self.filenames_250m, self.filenames_1000m]

        for resolution in resolution_list:
            filenames = file_list[resolution_list.index(resolution)]
            reader = _test_find_files_and_readers(self.reader_configs, filenames)

            # Verify that we have multiple resolutions for:
            # ---------MERSI-1---------
            #     - Bands 1-4 (visible)
            #     - Bands 5 (IR)
            # ---------MERSI-2---------
            #     - Bands 1-4 (visible)
            #     - Bands 24-25 (IR)
            # ---------MERSI-LL---------
            #     - Bands 6-7 (IR)
            available_datasets = reader.available_dataset_ids
            # Only MERSI-2/LL VIS has radiance calibration
            vis_num_results = 3 if self.yaml_file in ["mersi2_l1b.yaml", "mersi_ll_l1b.yaml"] else 2
            ir_num_results = 3
            _test_multi_resolutions(available_datasets, self.vis_250_bands, resolution, vis_num_results)
            _test_multi_resolutions(available_datasets, self.ir_250_bands, resolution, ir_num_results)

            res = reader.load(self.bands_1000 + self.bands_250)
            if resolution != "250":
                assert len(res) == len(self.bands_1000 + self.bands_250)
            else:
                assert len(res) == len(self.bands_250)
                for band in self.bands_1000:
                    with pytest.raises(KeyError):
                        res.__getitem__(band)

            if resolution in ["all", "250"]:
                _assert_bands_mda_as_exp(res, self.vis_250_bands, ("reflectance", "%", (2 * 40, 2048 * 2)))
                _assert_bands_mda_as_exp(res, self.ir_250_bands, ("brightness_temperature", "K", (2 * 40, 2048 * 2)))

                if resolution == "all":
                    _assert_bands_mda_as_exp(res, self.vis_1000_bands, ("reflectance", "%", (2 * 10, 2048)))
                    _assert_bands_mda_as_exp(res, self.ir_1000_bands, ("brightness_temperature", "K", (2 * 10, 2048)))
            else:
                _assert_bands_mda_as_exp(res, self.vis_250_bands, ("reflectance", "%", (2 * 10, 2048)))
                _assert_bands_mda_as_exp(res, self.vis_1000_bands, ("reflectance", "%", (2 * 10, 2048)))
                _assert_bands_mda_as_exp(res, self.ir_250_bands, ("brightness_temperature", "K", (2 * 10, 2048)))
                _assert_bands_mda_as_exp(res, self.ir_1000_bands, ("brightness_temperature", "K", (2 * 10, 2048)))

    def test_counts_calib(self):
        """Test loading data at counts calibration."""
        from satpy.tests.utils import make_dataid
        filenames = self.filenames_all
        reader = _test_find_files_and_readers(self.reader_configs, filenames)

        ds_ids = []
        for band_name in self.bands_1000 + self.bands_250:
            ds_ids.append(make_dataid(name=band_name, calibration="counts"))
        ds_ids.append(make_dataid(name="satellite_zenith_angle"))
        res = reader.load(ds_ids)
        assert len(res) == len(self.bands_1000) + len(self.bands_250) + 1
        _assert_bands_mda_as_exp(res, self.bands_250, ("counts", "1", (2 * 40, 2048 * 2)))
        _assert_bands_mda_as_exp(res, self.bands_1000, ("counts", "1", (2 * 10, 2048)))

    def test_rad_calib(self):
        """Test loading data at radiance calibration. For MERSI-2/LL VIS/IR and MERSI-1 IR."""
        from satpy.tests.utils import make_dataid
        filenames = self.filenames_all
        reader = _test_find_files_and_readers(self.reader_configs, filenames)

        ds_ids = []
        test_bands = self.bands_1000 + self.bands_250 if self.yaml_file in ["mersi2_l1b.yaml", "mersi_ll_l1b.yaml"] \
            else self.ir_250_bands + self.ir_1000_bands

        for band_name in test_bands:
            ds_ids.append(make_dataid(name=band_name, calibration="radiance"))
        res = reader.load(ds_ids)
        assert len(res) == len(test_bands)
        if self.yaml_file in ["mersi2_l1b.yaml", "mersi_ll_l1b.yaml"]:
            _assert_bands_mda_as_exp(res, self.bands_250, ("radiance", "mW/ (m2 cm-1 sr)", (2 * 40, 2048 * 2)))
            _assert_bands_mda_as_exp(res, self.bands_1000, ("radiance", "mW/ (m2 cm-1 sr)", (2 * 10, 2048)))
        else:
            _assert_bands_mda_as_exp(res, self.ir_250_bands, ("radiance", "mW/ (m2 cm-1 sr)", (2 * 40, 2048 * 2)))
            _assert_bands_mda_as_exp(res, self.ir_1000_bands, ("radiance", "mW/ (m2 cm-1 sr)", (2 * 10, 2048)))


class TestFY3AMERSI1L1B(MERSI12llL1BTester):
    """Test the FY3A MERSI1 L1B reader."""

    yaml_file = "fy3a_mersi1_l1b.yaml"
    filenames_1000m = ["FY3A_MERSI_GBAL_L1_20090601_1200_1000M_MS.hdf"]
    filenames_250m = ["FY3A_MERSI_GBAL_L1_20090601_1200_0250M_MS.hdf"]
    filenames_all = filenames_1000m + filenames_250m
    vis_250_bands = ["1", "2", "3", "4"]
    ir_250_bands = ["5"]
    vis_1000_bands = ["6", "7", "8", "11", "15", "19", "20"]
    ir_1000_bands = []
    bands_1000 = vis_1000_bands + ir_1000_bands
    bands_250 = vis_250_bands + ir_250_bands


class TestFY3BMERSI1L1B(MERSI12llL1BTester):
    """Test the FY3B MERSI1 L1B reader."""

    yaml_file = "fy3b_mersi1_l1b.yaml"
    filenames_1000m = ["FY3B_MERSI_GBAL_L1_20110824_1850_1000M_MS.hdf"]
    filenames_250m = ["FY3B_MERSI_GBAL_L1_20110824_1850_0250M_MS.hdf", "FY3B_MERSI_GBAL_L1_20110824_1850_GEOXX_MS.hdf"]
    filenames_all = filenames_1000m + filenames_250m
    vis_250_bands = ["1", "2", "3", "4"]
    ir_250_bands = ["5"]
    vis_1000_bands = ["6", "7", "8", "11", "15", "19", "20"]
    ir_1000_bands = []
    bands_1000 = vis_1000_bands + ir_1000_bands
    bands_250 = vis_250_bands + ir_250_bands


class TestFY3CMERSI1L1B(MERSI12llL1BTester):
    """Test the FY3C MERSI1 L1B reader."""

    yaml_file = "fy3c_mersi1_l1b.yaml"
    filenames_1000m = ["FY3C_MERSI_GBAL_L1_20131002_1835_1000M_MS.hdf", "FY3C_MERSI_GBAL_L1_20131002_1835_GEO1K_MS.hdf"]
    filenames_250m = ["FY3C_MERSI_GBAL_L1_20131002_1835_0250M_MS.hdf", "FY3C_MERSI_GBAL_L1_20131002_1835_GEOQK_MS.hdf"]
    filenames_all = filenames_1000m + filenames_250m
    vis_250_bands = ["1", "2", "3", "4"]
    ir_250_bands = ["5"]
    vis_1000_bands = ["6", "7", "8", "11", "15", "19", "20"]
    ir_1000_bands = []
    bands_1000 = vis_1000_bands + ir_1000_bands
    bands_250 = vis_250_bands + ir_250_bands


class TestFY3DMERSI2L1B(MERSI12llL1BTester):
    """Test the FY3D MERSI2 L1B reader."""

    yaml_file = "mersi2_l1b.yaml"
    filenames_1000m = ["tf2019071182739.FY3D-X_MERSI_1000M_L1B.HDF", "tf2019071182739.FY3D-X_MERSI_GEO1K_L1B.HDF"]
    filenames_250m = ["tf2019071182739.FY3D-X_MERSI_0250M_L1B.HDF", "tf2019071182739.FY3D-X_MERSI_GEOQK_L1B.HDF"]
    filenames_all = filenames_1000m + filenames_250m
    vis_250_bands = ["1", "2", "3", "4"]
    ir_250_bands = ["24", "25"]
    vis_1000_bands = ["5", "8", "9", "11", "15", "17", "19"]
    ir_1000_bands = ["20", "21", "23"]
    bands_1000 = vis_1000_bands + ir_1000_bands
    bands_250 = vis_250_bands + ir_250_bands


class TestFY3EMERSIllL1B(MERSI12llL1BTester):
    """Test the FY3D MERSI2 L1B reader."""

    yaml_file = "mersi_ll_l1b.yaml"
    filenames_1000m = ["FY3E_MERSI_GRAN_L1_20230410_1910_1000M_V0.HDF", "FY3E_MERSI_GRAN_L1_20230410_1910_GEO1K_V0.HDF"]
    filenames_250m = ["FY3E_MERSI_GRAN_L1_20230410_1910_0250M_V0.HDF", "FY3E_MERSI_GRAN_L1_20230410_1910_GEOQK_V0.HDF"]
    filenames_all = filenames_1000m + filenames_250m
    vis_250_bands = []
    ir_250_bands = ["6", "7"]
    vis_1000_bands = ["1"]
    ir_1000_bands = ["2", "3", "5"]
    bands_1000 = vis_1000_bands + ir_1000_bands
    bands_250 = vis_250_bands + ir_250_bands


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
