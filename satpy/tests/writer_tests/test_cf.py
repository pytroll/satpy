#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2019 Satpy developers
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
"""Tests for the CF writer."""

import datetime as dt
import os
import tempfile
import warnings

import numpy as np
import pytest
import xarray as xr
from packaging.version import Version

from satpy import Scene
from satpy.tests.utils import make_dsq
from satpy.writers.cf_writer import _get_backend_versions

try:
    from pyproj import CRS
except ImportError:
    CRS = None

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path
# - request


class TempFile:
    """A temporary filename class."""

    def __init__(self, suffix=".nc"):
        """Initialize."""
        self.filename = None
        self.suffix = suffix

    def __enter__(self):
        """Enter."""
        self.handle, self.filename = tempfile.mkstemp(suffix=self.suffix)
        os.close(self.handle)
        return self.filename

    def __exit__(self, *args):
        """Exit."""
        os.remove(self.filename)


class TestCFWriter:
    """Test case for CF writer."""

    def test_init(self):
        """Test initializing the CFWriter class."""
        from satpy.writers import configs_for_writer
        from satpy.writers.cf_writer import CFWriter

        CFWriter(config_files=list(configs_for_writer("cf"))[0])

    def test_save_array(self):
        """Test saving an array to netcdf/cf."""
        scn = Scene()
        start_time = dt.datetime(2018, 5, 30, 10, 0)
        end_time = dt.datetime(2018, 5, 30, 10, 15)
        scn["test-array"] = xr.DataArray([1, 2, 3],
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time,
                                                    prerequisites=[make_dsq(name="hej")]))
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer="cf")
            with xr.open_dataset(filename) as f:
                np.testing.assert_array_equal(f["test-array"][:], [1, 2, 3])
                expected_prereq = ("DataQuery(name='hej')")
                assert f["test-array"].attrs["prerequisites"] == expected_prereq

    def test_save_array_coords(self):
        """Test saving array with coordinates."""
        scn = Scene()
        start_time = dt.datetime(2018, 5, 30, 10, 0)
        end_time = dt.datetime(2018, 5, 30, 10, 15)
        coords = {
            "x": np.arange(3),
            "y": np.arange(1),
        }
        if CRS is not None:
            proj_str = ("+proj=geos +lon_0=-95.0 +h=35786023.0 "
                        "+a=6378137.0 +b=6356752.31414 +sweep=x "
                        "+units=m +no_defs")
            coords["crs"] = CRS.from_string(proj_str)
        scn["test-array"] = xr.DataArray([[1, 2, 3]],
                                         dims=("y", "x"),
                                         coords=coords,
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time,
                                                    prerequisites=[make_dsq(name="hej")]))
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer="cf")
            with xr.open_dataset(filename) as f:
                np.testing.assert_array_equal(f["test-array"][:], [[1, 2, 3]])
                np.testing.assert_array_equal(f["x"][:], [0, 1, 2])
                np.testing.assert_array_equal(f["y"][:], [0])
                assert "crs" not in f
                assert "_FillValue" not in f["x"].attrs
                assert "_FillValue" not in f["y"].attrs
                expected_prereq = ("DataQuery(name='hej')")
                assert f["test-array"].attrs["prerequisites"] == expected_prereq

    def test_save_dataset_a_digit(self):
        """Test saving an array to netcdf/cf where dataset name starting with a digit."""
        scn = Scene()
        scn["1"] = xr.DataArray([1, 2, 3])
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer="cf")
            with xr.open_dataset(filename) as f:
                np.testing.assert_array_equal(f["CHANNEL_1"][:], [1, 2, 3])

    def test_save_dataset_a_digit_prefix(self):
        """Test saving an array to netcdf/cf where dataset name starting with a digit with prefix."""
        scn = Scene()
        scn["1"] = xr.DataArray([1, 2, 3])
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer="cf", numeric_name_prefix="TEST")
            with xr.open_dataset(filename) as f:
                np.testing.assert_array_equal(f["TEST1"][:], [1, 2, 3])

    def test_save_dataset_a_digit_prefix_include_attr(self):
        """Test saving an array to netcdf/cf where dataset name starting with a digit with prefix include orig name."""
        scn = Scene()
        scn["1"] = xr.DataArray([1, 2, 3])
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer="cf", include_orig_name=True, numeric_name_prefix="TEST")
            with xr.open_dataset(filename) as f:
                np.testing.assert_array_equal(f["TEST1"][:], [1, 2, 3])
                assert f["TEST1"].attrs["original_name"] == "1"

    def test_save_dataset_a_digit_no_prefix_include_attr(self):
        """Test saving an array to netcdf/cf dataset name starting with a digit with no prefix include orig name."""
        scn = Scene()
        scn["1"] = xr.DataArray([1, 2, 3])
        with TempFile() as filename:
            with pytest.warns(UserWarning, match=r"Invalid NetCDF dataset name"):
                scn.save_datasets(filename=filename, writer="cf", include_orig_name=True, numeric_name_prefix="")
            with xr.open_dataset(filename) as f:
                np.testing.assert_array_equal(f["1"][:], [1, 2, 3])
                assert "original_name" not in f["1"].attrs

    def test_ancillary_variables(self):
        """Test ancillary_variables cited each other."""
        from satpy.tests.utils import make_dataid
        scn = Scene()
        start_time = dt.datetime(2018, 5, 30, 10, 0)
        end_time = dt.datetime(2018, 5, 30, 10, 15)
        da = xr.DataArray([1, 2, 3],
                          attrs=dict(start_time=start_time,
                          end_time=end_time,
                          prerequisites=[make_dataid(name="hej")]))
        scn["test-array-1"] = da
        scn["test-array-2"] = da.copy()
        scn["test-array-1"].attrs["ancillary_variables"] = [scn["test-array-2"]]
        scn["test-array-2"].attrs["ancillary_variables"] = [scn["test-array-1"]]
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer="cf")
            with xr.open_dataset(filename) as f:
                assert f["test-array-1"].attrs["ancillary_variables"] == "test-array-2"
                assert f["test-array-2"].attrs["ancillary_variables"] == "test-array-1"

    def test_groups(self):
        """Test creating a file with groups."""
        tstart = dt.datetime(2019, 4, 1, 12, 0)
        tend = dt.datetime(2019, 4, 1, 12, 15)

        data_visir = [[1, 2], [3, 4]]
        y_visir = [1, 2]
        x_visir = [1, 2]
        time_vis006 = [1, 2]
        time_ir_108 = [3, 4]

        data_hrv = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        y_hrv = [1, 2, 3]
        x_hrv = [1, 2, 3]
        time_hrv = [1, 2, 3]

        scn = Scene()
        scn["VIS006"] = xr.DataArray(data_visir,
                                     dims=("y", "x"),
                                     coords={"y": y_visir, "x": x_visir, "acq_time": ("y", time_vis006)},
                                     attrs={"name": "VIS006", "start_time": tstart, "end_time": tend})
        scn["IR_108"] = xr.DataArray(data_visir,
                                     dims=("y", "x"),
                                     coords={"y": y_visir, "x": x_visir, "acq_time": ("y", time_ir_108)},
                                     attrs={"name": "IR_108", "start_time": tstart, "end_time": tend})
        scn["HRV"] = xr.DataArray(data_hrv,
                                  dims=("y", "x"),
                                  coords={"y": y_hrv, "x": x_hrv, "acq_time": ("y", time_hrv)},
                                  attrs={"name": "HRV", "start_time": tstart, "end_time": tend})

        with TempFile() as filename:
            with pytest.warns(UserWarning, match=r"Cannot pretty-format"):
                scn.save_datasets(filename=filename, writer="cf",
                                  groups={"visir": ["IR_108", "VIS006"], "hrv": ["HRV"]},
                                  pretty=True)

            nc_root = xr.open_dataset(filename)
            assert "history" in nc_root.attrs
            assert set(nc_root.variables.keys()) == set()

            nc_visir = xr.open_dataset(filename, group="visir")
            nc_hrv = xr.open_dataset(filename, group="hrv")
            assert set(nc_visir.variables.keys()) == {"VIS006", "IR_108",
                                                      "y", "x", "VIS006_acq_time", "IR_108_acq_time"}
            assert set(nc_hrv.variables.keys()) == {"HRV", "y", "x", "acq_time"}
            for tst, ref in zip([nc_visir["VIS006"], nc_visir["IR_108"], nc_hrv["HRV"]],
                                [scn["VIS006"], scn["IR_108"], scn["HRV"]]):
                np.testing.assert_array_equal(tst.data, ref.data)
            nc_root.close()
            nc_visir.close()
            nc_hrv.close()

        # Different projection coordinates in one group are not supported
        with TempFile() as filename:
            with pytest.raises(ValueError, match="Datasets .* must have identical projection coordinates..*"):
                scn.save_datasets(datasets=["VIS006", "HRV"], filename=filename, writer="cf")

    def test_single_time_value(self):
        """Test setting a single time value."""
        scn = Scene()
        start_time = dt.datetime(2018, 5, 30, 10, 0)
        end_time = dt.datetime(2018, 5, 30, 10, 15)
        test_array = np.array([[1, 2], [3, 4]])
        scn["test-array"] = xr.DataArray(test_array,
                                         dims=["x", "y"],
                                         coords={"time": np.datetime64("2018-05-30T10:05:00", "ns")},
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time))
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer="cf", encoding={"time": {"units": "seconds since 2018-01-01"}})
            with xr.open_dataset(filename, decode_cf=True) as f:
                np.testing.assert_array_equal(f["time"], scn["test-array"]["time"])
                bounds_exp = np.array([[start_time, end_time]], dtype="datetime64[m]")
                np.testing.assert_array_equal(f["time_bnds"], bounds_exp)

    def test_time_coordinate_on_a_swath(self):
        """Test that time dimension is not added on swath data with time already as a coordinate."""
        scn = Scene()
        test_array = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        times = np.array(["2018-05-30T10:05:00", "2018-05-30T10:05:01",
                          "2018-05-30T10:05:02", "2018-05-30T10:05:03"], dtype="datetime64[ns]")
        scn["test-array"] = xr.DataArray(test_array,
                                         dims=["y", "x"],
                                         coords={"time": ("y", times)},
                                         attrs=dict(start_time=times[0], end_time=times[-1]))
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer="cf", pretty=True,
                              encoding={"time": {"units": "seconds since 2018-01-01"}})
            with xr.open_dataset(filename, decode_cf=True) as f:
                np.testing.assert_array_equal(f["time"], scn["test-array"]["time"])

    def test_bounds(self):
        """Test setting time bounds."""
        scn = Scene()
        start_time = dt.datetime(2018, 5, 30, 10, 0)
        end_time = dt.datetime(2018, 5, 30, 10, 15)
        test_array = np.array([[1, 2], [3, 4]]).reshape(2, 2, 1)
        scn["test-array"] = xr.DataArray(test_array,
                                         dims=["x", "y", "time"],
                                         coords={"time": [np.datetime64("2018-05-30T10:05:00", "ns")]},
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time))
        with TempFile() as filename:
            with warnings.catch_warnings():
                # The purpose is to use the default time encoding, silence the warning
                warnings.filterwarnings("ignore", category=UserWarning,
                                        message=r"Times can't be serialized faithfully to int64 with requested units")
                scn.save_datasets(filename=filename, writer="cf")
            # Check decoded time coordinates & bounds
            with xr.open_dataset(filename, decode_cf=True) as f:
                bounds_exp = np.array([[start_time, end_time]], dtype="datetime64[m]")
                np.testing.assert_array_equal(f["time_bnds"], bounds_exp)
                assert f["time"].attrs["bounds"] == "time_bnds"

            # Check raw time coordinates & bounds
            with xr.open_dataset(filename, decode_cf=False) as f:
                np.testing.assert_almost_equal(f["time_bnds"], [[-0.0034722, 0.0069444]])

        # User-specified time encoding should have preference
        with TempFile() as filename:
            time_units = "seconds since 2018-01-01"
            scn.save_datasets(filename=filename, encoding={"time": {"units": time_units}},
                              writer="cf")
            with xr.open_dataset(filename, decode_cf=False) as f:
                np.testing.assert_array_equal(f["time_bnds"], [[12909600, 12910500]])

    def test_bounds_minimum(self):
        """Test minimum bounds."""
        scn = Scene()
        start_timeA = dt.datetime(2018, 5, 30, 10, 0)  # expected to be used
        end_timeA = dt.datetime(2018, 5, 30, 10, 20)
        start_timeB = dt.datetime(2018, 5, 30, 10, 3)
        end_timeB = dt.datetime(2018, 5, 30, 10, 15)  # expected to be used
        test_arrayA = np.array([[1, 2], [3, 4]]).reshape(2, 2, 1)
        test_arrayB = np.array([[1, 2], [3, 5]]).reshape(2, 2, 1)
        scn["test-arrayA"] = xr.DataArray(test_arrayA,
                                          dims=["x", "y", "time"],
                                          coords={"time": [np.datetime64("2018-05-30T10:05:00", "ns")]},
                                          attrs=dict(start_time=start_timeA,
                                                     end_time=end_timeA))
        scn["test-arrayB"] = xr.DataArray(test_arrayB,
                                          dims=["x", "y", "time"],
                                          coords={"time": [np.datetime64("2018-05-30T10:05:00", "ns")]},
                                          attrs=dict(start_time=start_timeB,
                                                     end_time=end_timeB))
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer="cf",
                              encoding={"time": {"units": "seconds since 2018-01-01"}})
            with xr.open_dataset(filename, decode_cf=True) as f:
                bounds_exp = np.array([[start_timeA, end_timeB]], dtype="datetime64[m]")
                np.testing.assert_array_equal(f["time_bnds"], bounds_exp)

    def test_bounds_missing_time_info(self):
        """Test time bounds generation in case of missing time."""
        scn = Scene()
        start_timeA = dt.datetime(2018, 5, 30, 10, 0)
        end_timeA = dt.datetime(2018, 5, 30, 10, 15)
        test_arrayA = np.array([[1, 2], [3, 4]]).reshape(2, 2, 1)
        test_arrayB = np.array([[1, 2], [3, 5]]).reshape(2, 2, 1)
        scn["test-arrayA"] = xr.DataArray(test_arrayA,
                                          dims=["x", "y", "time"],
                                          coords={"time": [np.datetime64("2018-05-30T10:05:00", "ns")]},
                                          attrs=dict(start_time=start_timeA,
                                                     end_time=end_timeA))
        scn["test-arrayB"] = xr.DataArray(test_arrayB,
                                          dims=["x", "y", "time"],
                                          coords={"time": [np.datetime64("2018-05-30T10:05:00", "ns")]})
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer="cf",
                              encoding={"time": {"units": "seconds since 2018-01-01"}})
            with xr.open_dataset(filename, decode_cf=True) as f:
                bounds_exp = np.array([[start_timeA, end_timeA]], dtype="datetime64[m]")
                np.testing.assert_array_equal(f["time_bnds"], bounds_exp)

    def test_unlimited_dims_kwarg(self):
        """Test specification of unlimited dimensions."""
        scn = Scene()
        start_time = dt.datetime(2018, 5, 30, 10, 0)
        end_time = dt.datetime(2018, 5, 30, 10, 15)
        test_array = np.array([[1, 2], [3, 4]])
        scn["test-array"] = xr.DataArray(test_array,
                                         dims=["x", "y"],
                                         coords={"time": np.datetime64("2018-05-30T10:05:00", "ns")},
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time))
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer="cf", unlimited_dims=["time"],
                              encoding={"time": {"units": "seconds since 2018-01-01"}})
            with xr.open_dataset(filename) as f:
                assert set(f.encoding["unlimited_dims"]) == {"time"}

    def test_header_attrs(self):
        """Check global attributes are set."""
        scn = Scene()
        start_time = dt.datetime(2018, 5, 30, 10, 0)
        end_time = dt.datetime(2018, 5, 30, 10, 15)
        scn["test-array"] = xr.DataArray([1, 2, 3],
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time))
        with TempFile() as filename:
            header_attrs = {"sensor": "SEVIRI",
                            "orbit": 99999,
                            "none": None,
                            "list": [1, 2, 3],
                            "set": {1, 2, 3},
                            "dict": {"a": 1, "b": 2},
                            "nested": {"outer": {"inner1": 1, "inner2": 2}},
                            "bool": True,
                            "bool_": np.bool_(True)}
            scn.save_datasets(filename=filename,
                              header_attrs=header_attrs,
                              flatten_attrs=True,
                              writer="cf")
            with xr.open_dataset(filename) as f:
                assert "history" in f.attrs
                assert f.attrs["sensor"] == "SEVIRI"
                assert f.attrs["orbit"] == 99999
                np.testing.assert_array_equal(f.attrs["list"], [1, 2, 3])
                assert f.attrs["set"] == "{1, 2, 3}"
                assert f.attrs["dict_a"] == 1
                assert f.attrs["dict_b"] == 2
                assert f.attrs["nested_outer_inner1"] == 1
                assert f.attrs["nested_outer_inner2"] == 2
                assert f.attrs["bool"] == "true"
                assert f.attrs["bool_"] == "true"
                assert "none" not in f.attrs.keys()

    def test_load_module_with_old_pyproj(self):
        """Test that cf_writer can still be loaded with pyproj 1.9.6."""
        import importlib
        import sys

        import pyproj  # noqa 401
        old_version = sys.modules["pyproj"].__version__
        sys.modules["pyproj"].__version__ = "1.9.6"
        try:
            importlib.reload(sys.modules["satpy.writers.cf_writer"])
        finally:
            # Tear down
            sys.modules["pyproj"].__version__ = old_version
            importlib.reload(sys.modules["satpy.writers.cf_writer"])

    def test_global_attr_default_history_and_Conventions(self):
        """Test saving global attributes history and Conventions."""
        scn = Scene()
        start_time = dt.datetime(2018, 5, 30, 10, 0)
        end_time = dt.datetime(2018, 5, 30, 10, 15)
        scn["test-array"] = xr.DataArray([[1, 2, 3]],
                                         dims=("y", "x"),
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time,
                                                    prerequisites=[make_dsq(name="hej")]))
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer="cf")
            with xr.open_dataset(filename) as f:
                assert f.attrs["Conventions"] == "CF-1.7"
                assert "Created by pytroll/satpy on" in f.attrs["history"]

    def test_global_attr_history_and_Conventions(self):
        """Test saving global attributes history and Conventions."""
        scn = Scene()
        start_time = dt.datetime(2018, 5, 30, 10, 0)
        end_time = dt.datetime(2018, 5, 30, 10, 15)
        scn["test-array"] = xr.DataArray([[1, 2, 3]],
                                         dims=("y", "x"),
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time,
                                                    prerequisites=[make_dsq(name="hej")]))
        header_attrs = {}
        header_attrs["history"] = ("TEST add history",)
        header_attrs["Conventions"] = "CF-1.7, ACDD-1.3"
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer="cf", header_attrs=header_attrs)
            with xr.open_dataset(filename) as f:
                assert f.attrs["Conventions"] == "CF-1.7, ACDD-1.3"
                assert "TEST add history\n" in f.attrs["history"]
                assert "Created by pytroll/satpy on" in f.attrs["history"]


class TestNetcdfEncodingKwargs:
    """Test netCDF compression encodings."""

    @pytest.fixture
    def scene(self):
        """Create a fake scene."""
        scn = Scene()
        attrs = {
            "start_time": dt.datetime(2018, 5, 30, 10, 0),
            "end_time": dt.datetime(2018, 5, 30, 10, 15)
        }
        scn["test-array"] = xr.DataArray([1., 2, 3], attrs=attrs)
        return scn

    @pytest.fixture(params=[True, False])
    def compression_on(self, request):
        """Get compression options."""
        return request.param

    @pytest.fixture
    def encoding(self, compression_on):
        """Get encoding."""
        enc = {
            "test-array": {
                "dtype": "int8",
                "scale_factor": 0.1,
                "add_offset": 0.0,
                "_FillValue": 3,
            }
        }
        if compression_on:
            comp_params = _get_compression_params(complevel=7)
            enc["test-array"].update(comp_params)
        return enc

    @pytest.fixture
    def filename(self, tmp_path):
        """Get output filename."""
        return str(tmp_path / "test.nc")

    @pytest.fixture
    def complevel_exp(self, compression_on):
        """Get expected compression level."""
        if compression_on:
            return 7
        return 0

    @pytest.fixture
    def expected(self, complevel_exp):
        """Get expectated file contents."""
        return {
            "data": [10, 20, 30],
            "scale_factor": 0.1,
            "fill_value": 3,
            "dtype": np.int8,
            "complevel": complevel_exp
        }

    def test_encoding_kwarg(self, scene, encoding, filename, expected):
        """Test 'encoding' keyword argument."""
        scene.save_datasets(filename=filename, encoding=encoding, writer="cf")
        self._assert_encoding_as_expected(filename, expected)

    def _assert_encoding_as_expected(self, filename, expected):
        with xr.open_dataset(filename, mask_and_scale=False) as f:
            np.testing.assert_array_equal(f["test-array"][:], expected["data"])
            assert f["test-array"].attrs["scale_factor"] == expected["scale_factor"]
            assert f["test-array"].attrs["_FillValue"] == expected["fill_value"]
            assert f["test-array"].dtype == expected["dtype"]
            assert f["test-array"].encoding["complevel"] == expected["complevel"]

    def test_warning_if_backends_dont_match(self, scene, filename, monkeypatch):
        """Test warning if backends don't match."""
        import netCDF4
        with monkeypatch.context() as m:
            m.setattr(netCDF4, "__version__", "1.5.0")
            m.setattr(netCDF4, "__netcdf4libversion__", "4.9.1")
            with pytest.warns(UserWarning, match=r"Backend version mismatch"):
                scene.save_datasets(filename=filename, writer="cf")

    def test_no_warning_if_backends_match(self, scene, filename, monkeypatch):
        """Make sure no warning is issued if backends match."""
        import netCDF4
        with monkeypatch.context() as m:
            m.setattr(netCDF4, "__version__", "1.6.0")
            m.setattr(netCDF4, "__netcdf4libversion__", "4.9.0")
            m.setattr(xr, "__version__", "2022.12.0")
            with warnings.catch_warnings():
                scene.save_datasets(filename=filename, writer="cf")
                warnings.simplefilter("error")


class TestEncodingAttribute(TestNetcdfEncodingKwargs):
    """Test CF writer with 'encoding' dataset attribute."""

    @pytest.fixture
    def scene_with_encoding(self, scene, encoding):
        """Create scene with a dataset providing the 'encoding' attribute."""
        scene["test-array"].encoding = encoding["test-array"]
        return scene

    def test_encoding_attribute(self, scene_with_encoding, filename, expected):
        """Test 'encoding' dataset attribute."""
        scene_with_encoding.save_datasets(filename=filename, writer="cf")
        self._assert_encoding_as_expected(filename, expected)


def _get_compression_params(complevel):
    params = {"complevel": complevel}
    if _should_use_compression_keyword():
        params["compression"] = "zlib"
    else:
        params["zlib"] = True
    return params


def _should_use_compression_keyword():
    # xarray currently ignores the "compression" keyword, see
    # https://github.com/pydata/xarray/issues/7388. There's already an open
    # PR, so we assume that this will be fixed in the next minor release
    # (current release is 2023.02). If not, tests will fail and remind us.
    versions = _get_backend_versions()
    return (
        versions["libnetcdf"] >= Version("4.9.0") and
        versions["xarray"] >= Version("2024.1")
    )
