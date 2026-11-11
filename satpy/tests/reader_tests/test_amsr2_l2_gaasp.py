"""Tests for the 'amsr2_l2_gaasp' reader."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr

MBT_FILENAME = "AMSR2-MBT_v2r2_GW1_s202008120558310_e202008120607010_c202008120637340.nc"
PRECIP_FILENAME = "AMSR2-PRECIP_v2r2_GW1_s202008120558310_e202008120607010_c202008120637340.nc"
OCEAN_FILENAME = "AMSR2-OCEAN_v2r2_GW1_s202008120558310_e202008120607010_c202008120637340.nc"
SEAICE_NH_FILENAME = "AMSR2-SEAICE-NH_v2r2_GW1_s202008120558310_e202008120607010_c202008120637340.nc"
SEAICE_SH_FILENAME = "AMSR2-SEAICE-SH_v2r2_GW1_s202008120558310_e202008120607010_c202008120637340.nc"
SNOW_FILENAME = "AMSR2-SNOW_v2r2_GW1_s202008120558310_e202008120607010_c202008120637340.nc"
SOIL_FILENAME = "AMSR2-SOIL_v2r2_GW1_s202008120558310_e202008120607010_c202008120637340.nc"

EXAMPLE_FILENAMES = [
    MBT_FILENAME,
    PRECIP_FILENAME,
    OCEAN_FILENAME,
    SEAICE_NH_FILENAME,
    SEAICE_SH_FILENAME,
    SNOW_FILENAME,
    SOIL_FILENAME,
]

SEAICE_FILL = -9999
SEAICE_NO_DATA_FLAG = 120
# Kinds of pixels in a SEAICE file as (ice concentration, Flags)
SEAICE_PIXELS = {
    "good": (80, 0),
    "weather": (20, 8),
    "sst_missing": (0, 4),
    "filled": (60, 32),
}


def _get_shared_global_attrs(filename):
    attrs = {
        "time_coverage_start": "2020-08-12T05:58:31.0Z",
        "time_coverage_end": "2020-08-12T06:07:01.0Z",
        "platform_name": "GCOM-W1",
        "instrument_name": "AMSR2",
    }
    return attrs


def _create_two_res_gaasp_dataset(filename):
    """Represent files with two resolution of variables in them (ex. OCEAN)."""
    lon_var_hi = xr.DataArray(
        da.zeros((10, 10), dtype=np.float32),
        dims=("Number_of_Scans", "Number_of_hi_rez_FOVs"),
        attrs={"standard_name": "longitude"},
    )
    lat_var_hi = xr.DataArray(
        da.zeros((10, 10), dtype=np.float32),
        dims=("Number_of_Scans", "Number_of_hi_rez_FOVs"),
        attrs={"standard_name": "latitude"},
    )
    lon_var_lo = xr.DataArray(
        da.zeros((10, 10), dtype=np.float32),
        dims=("Number_of_Scans", "Number_of_low_rez_FOVs"),
        attrs={"standard_name": "longitude"},
    )
    lat_var_lo = xr.DataArray(
        da.zeros((10, 10), dtype=np.float32),
        dims=("Number_of_Scans", "Number_of_low_rez_FOVs"),
        attrs={"standard_name": "latitude"},
    )
    swath_var1 = xr.DataArray(
        da.zeros((10, 10), dtype=np.float32),
        dims=("Number_of_Scans", "Number_of_hi_rez_FOVs"),
        coords={"some_longitude_hi": lon_var_hi, "some_latitude_hi": lat_var_hi},
        attrs={"_FillValue": -9999.0, "scale_factor": 0.5, "add_offset": 2.0},
    )
    swath_var2 = xr.DataArray(
        da.zeros((10, 10), dtype=np.float32),
        dims=("Number_of_Scans", "Number_of_low_rez_FOVs"),
        coords={"some_longitude_lo": lon_var_lo, "some_latitude_lo": lat_var_lo},
        attrs={"_FillValue": -9999.0},
    )
    swath_int_var = xr.DataArray(
        da.zeros((10, 10), dtype=np.uint16),
        dims=("Number_of_Scans", "Number_of_low_rez_FOVs"),
        attrs={"_FillValue": 100, "comment": "Some comment"},
    )
    not_xy_dim_var = xr.DataArray(da.zeros((10, 5), dtype=np.float32), dims=("Number_of_Scans", "Time_Dimension"))
    time_var = xr.DataArray(da.zeros((5,), dtype=np.float32), dims=("Time_Dimension",))
    ds_vars = {
        "swath_var_hi": swath_var1,
        "swath_var_low": swath_var2,
        "swath_var_low_int": swath_int_var,
        "some_longitude_hi": lon_var_hi,
        "some_latitude_hi": lat_var_hi,
        "some_longitude_lo": lon_var_lo,
        "some_latitude_lo": lat_var_lo,
        "not_xy_dim_var": not_xy_dim_var,
        "time_var": time_var,
    }
    attrs = _get_shared_global_attrs(filename)
    ds = xr.Dataset(ds_vars, attrs=attrs)
    return ds


def _create_gridded_gaasp_dataset(filename):
    """Represent files with gridded products."""
    grid_var = xr.DataArray(
        da.zeros((10, 10), dtype=np.float32),
        dims=("Number_of_Y_Dimension", "Number_of_X_Dimension"),
        attrs={"_FillValue": -9999.0, "scale_factor": 0.5, "add_offset": 2.0},
    )
    latency_var = xr.DataArray(
        da.zeros((10, 10), dtype=np.int32),
        dims=("Number_of_Y_Dimension", "Number_of_X_Dimension"),
        attrs={
            "units": "seconds",
            "_FillValue": -9999,
        },
    )
    time_var = xr.DataArray(da.zeros((5,), dtype=np.float32), dims=("Time_Dimension",))
    ds_vars = {
        "grid_var": grid_var,
        "latency_var": latency_var,
        "time_var": time_var,
    }
    attrs = _get_shared_global_attrs(filename)
    return xr.Dataset(ds_vars, attrs=attrs)


def _create_one_res_gaasp_dataset(filename):
    """Represent files with one resolution of variables in them (ex. SOIL)."""
    lon_var_lo = xr.DataArray(
        da.zeros((10, 10), dtype=np.float32),
        dims=("Number_of_Scans", "Number_of_low_rez_FOVs"),
        attrs={"standard_name": "longitude"},
    )
    lat_var_lo = xr.DataArray(
        da.zeros((10, 10), dtype=np.float32),
        dims=("Number_of_Scans", "Number_of_low_rez_FOVs"),
        attrs={"standard_name": "latitude"},
    )
    swath_var2 = xr.DataArray(
        da.zeros((10, 10), dtype=np.float32),
        dims=("Number_of_Scans", "Number_of_low_rez_FOVs"),
        coords={"some_longitude_lo": lon_var_lo, "some_latitude_lo": lat_var_lo},
        attrs={"_FillValue": -9999.0, "scale_factor": 0.5, "add_offset": 2.0},
    )
    swath_int_var = xr.DataArray(
        da.zeros((10, 10), dtype=np.uint16),
        dims=("Number_of_Scans", "Number_of_low_rez_FOVs"),
        attrs={"_FillValue": 100, "comment": "Some comment"},
    )
    time_var = xr.DataArray(da.zeros((5,), dtype=np.float32), dims=("Time_Dimension",))
    ds_vars = {
        "swath_var": swath_var2,
        "swath_var_int": swath_int_var,
        "some_longitude_lo": lon_var_lo,
        "some_latitude_lo": lat_var_lo,
        "time_var": time_var,
    }
    attrs = _get_shared_global_attrs(filename)
    return xr.Dataset(ds_vars, attrs=attrs)


def _create_fake_gaasp_file(fake_path: Path, **kwargs):
    """Create a Dataset similar to reading an actual file with xarray.open_dataset."""
    filename = fake_path.name
    if filename in [MBT_FILENAME, PRECIP_FILENAME, OCEAN_FILENAME]:
        ds = _create_two_res_gaasp_dataset(filename)
    elif filename in [SEAICE_NH_FILENAME, SEAICE_SH_FILENAME]:
        ds = _create_gridded_gaasp_dataset(filename)
    else:
        ds = _create_one_res_gaasp_dataset(filename)
    ds.to_netcdf(fake_path, )


def _create_fake_ocean_wind_speed_file(filename, **kwargs):
    """Create a Dataset similar to reading an actual OCEAN file with wind speed data."""
    ds = _create_two_res_gaasp_dataset(filename)
    wspd_data = da.arange(10 * 10, dtype=np.float32).reshape((10, 10))
    wspd_data_arr = xr.DataArray(
        wspd_data,
        dims=("Number_of_Scans", "Number_of_low_rez_FOVs"),
        coords={"some_longitude_lo": ds["some_longitude_lo"], "some_latitude_lo": ds["some_latitude_lo"]},
        attrs={"_FillValue": -9999.0},
    )
    ds["WSPD"] = wspd_data_arr
    wspd_qc_data = np.zeros((10, 10), dtype=np.int32).reshape((10, 10))
    wspd_qc_data[-1, -1] = 1
    wspd_qc_data_arr = xr.DataArray(
        da.from_array(wspd_qc_data),
        dims=("Number_of_Scans", "Number_of_low_rez_FOVs"),
        attrs={"_FillValue": -9999, "units": "1"},
    )
    ds["WSPD_QC"] = wspd_qc_data_arr
    ds.to_netcdf(filename)


def _create_seaice_var(pixel_values, no_data_value, dtype, **attrs):
    """Create a variable with the pixels in the first row and "no data" pixels everywhere else."""
    data = np.full((10, 10), no_data_value, dtype=dtype)
    data[0, :len(pixel_values)] = pixel_values
    return xr.DataArray(
        da.from_array(data),
        dims=("Number_of_Y_Dimension", "Number_of_X_Dimension"),
        attrs={"_FillValue": SEAICE_FILL, **attrs},
    )


def _create_fake_seaice_file(filename):
    """Create a Dataset similar to reading an actual SEAICE file with each kind of pixel in it."""
    ds = _create_gridded_gaasp_dataset(filename)
    ice_values, flag_values = zip(*SEAICE_PIXELS.values())
    ds["NASA_Team_2_Ice_Concentration"] = _create_seaice_var(ice_values, SEAICE_FILL, np.int32, units="1")
    ds["NASA_Team_2_Multiyear_Ice"] = _create_seaice_var(ice_values, SEAICE_FILL, np.int32, units="1")
    ds["Flags"] = _create_seaice_var(flag_values, SEAICE_NO_DATA_FLAG, np.int16, units="1", comment="0: good")
    ds.to_netcdf(filename)


def _expected_seaice_valid_mask(pixel_names):
    valid_mask = np.zeros((10, 10), dtype=bool)
    valid_mask[0, :len(SEAICE_PIXELS)] = [pixel_name in pixel_names for pixel_name in SEAICE_PIXELS]
    return valid_mask


class TestGAASPReader:
    """Tests for the GAASP reader."""

    yaml_file = "amsr2_l2_gaasp.yaml"

    def setup_method(self):
        """Wrap pygrib to read fake data."""
        from satpy._config import config_search_paths

        self.reader_configs = config_search_paths(os.path.join("readers", self.yaml_file))

    @pytest.mark.parametrize(
        ("filenames", "expected_loadables"),
        [
            (EXAMPLE_FILENAMES, 7),
            ([MBT_FILENAME], 1),
            ([PRECIP_FILENAME], 1),
            ([OCEAN_FILENAME], 1),
            ([SEAICE_NH_FILENAME], 1),
            ([SEAICE_SH_FILENAME], 1),
            ([SNOW_FILENAME], 1),
            ([SOIL_FILENAME], 1),
        ],
    )
    def test_reader_creation(self, tmp_path, filenames, expected_loadables):
        """Test basic initialization."""
        from satpy.readers.core.loading import load_reader

        filenames = [tmp_path / filename for filename in filenames]
        for fn in filenames:
            _create_fake_gaasp_file(fn)
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames(filenames)
        assert len(loadables) == expected_loadables
        r.create_filehandlers(loadables)
        # make sure we have some files
        assert r.file_handlers

    @pytest.mark.parametrize(
        ("filenames", "expected_datasets"),
        [
            (
                EXAMPLE_FILENAMES,
                [
                    "swath_var_hi",
                    "swath_var_low",
                    "swath_var_low_int",
                    "swath_var",
                    "swath_var_int",
                    "grid_var_NH",
                    "grid_var_SH",
                    "latency_var_NH",
                    "latency_var_SH",
                ],
            ),
            ([MBT_FILENAME], ["swath_var_hi", "swath_var_low", "swath_var_low_int"]),
            ([PRECIP_FILENAME], ["swath_var_hi", "swath_var_low", "swath_var_low_int"]),
            ([OCEAN_FILENAME], ["swath_var_hi", "swath_var_low", "swath_var_low_int"]),
            ([SEAICE_NH_FILENAME], ["grid_var_NH", "latency_var_NH"]),
            ([SEAICE_SH_FILENAME], ["grid_var_SH", "latency_var_SH"]),
            ([SNOW_FILENAME], ["swath_var", "swath_var_int"]),
            ([SOIL_FILENAME], ["swath_var", "swath_var_int"]),
        ],
    )
    def test_available_datasets(self, tmp_path, filenames, expected_datasets):
        """Test that variables are dynamically discovered."""
        from satpy.readers.core.loading import load_reader

        filenames = [tmp_path / filename for filename in filenames]
        for fn in filenames:
            _create_fake_gaasp_file(fn)
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames(filenames)
        r.create_filehandlers(loadables)
        avails = list(r.available_dataset_names)
        for var_name in expected_datasets:
            assert var_name in avails
        assert "not_xy_dim_var" not in expected_datasets

    @staticmethod
    def _check_area(data_id, data_arr):
        from pyresample.geometry import AreaDefinition, SwathDefinition

        area = data_arr.attrs["area"]
        if "grid_var" in data_id["name"] or "latency_var" in data_id["name"]:
            assert isinstance(area, AreaDefinition)
        else:
            assert isinstance(area, SwathDefinition)

    @staticmethod
    def _check_fill(data_id, data_arr):
        if "int" in data_id["name"]:
            assert data_arr.attrs["_FillValue"] == 100
            assert np.issubdtype(data_arr.dtype, np.integer)
        else:
            assert "_FillValue" not in data_arr.attrs
            if data_id["name"].startswith("latency"):
                # timedelta only fits in 64-bit float
                assert data_arr.dtype.type == np.float64
            elif np.issubdtype(data_arr.dtype, np.floating):
                # we started with float32, it should stay that way
                assert data_arr.dtype.type == np.float32

    @staticmethod
    def _check_attrs(data_arr):
        attrs = data_arr.attrs
        assert "scale_factor" not in attrs
        assert "add_offset" not in attrs
        assert attrs["platform_name"] == "GCOM-W1"
        assert attrs["sensor"] == "amsr2"
        assert attrs["start_time"] == dt.datetime(2020, 8, 12, 5, 58, 31)
        assert attrs["end_time"] == dt.datetime(2020, 8, 12, 6, 7, 1)

    @pytest.mark.parametrize(
        ("filenames", "loadable_ids"),
        [
            (
                EXAMPLE_FILENAMES,
                [
                    "swath_var_hi",
                    "swath_var_low",
                    "swath_var_low_int",
                    "swath_var",
                    "swath_var_int",
                    "grid_var_NH",
                    "grid_var_SH",
                    "latency_var_NH",
                    "latency_var_SH",
                ],
            ),
            ([MBT_FILENAME], ["swath_var_hi", "swath_var_low", "swath_var_low_int"]),
            ([PRECIP_FILENAME], ["swath_var_hi", "swath_var_low", "swath_var_low_int"]),
            ([OCEAN_FILENAME], ["swath_var_hi", "swath_var_low", "swath_var_low_int"]),
            ([SEAICE_NH_FILENAME], ["grid_var_NH", "latency_var_NH"]),
            ([SEAICE_SH_FILENAME], ["grid_var_SH", "latency_var_SH"]),
            ([SNOW_FILENAME], ["swath_var", "swath_var_int"]),
            ([SOIL_FILENAME], ["swath_var", "swath_var_int"]),
        ],
    )
    def test_basic_load(self, tmp_path, filenames, loadable_ids):
        """Test that variables are loaded properly."""
        from satpy.readers.core.loading import load_reader

        filenames = [tmp_path / filename for filename in filenames]
        for fn in filenames:
            _create_fake_gaasp_file(fn)
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames(filenames)
        r.create_filehandlers(loadables)
        loaded_data_arrs = r.load(loadable_ids)
        assert loaded_data_arrs
        for data_id, data_arr in loaded_data_arrs.items():
            self._check_area(data_id, data_arr)
            self._check_fill(data_id, data_arr)
            self._check_attrs(data_arr)

    @pytest.mark.parametrize("filter_wspd", [False, True])
    def test_wind_speed_filtering(self, tmp_path, filter_wspd):
        """Test that wind speed can be filtered."""
        from satpy.readers.core.loading import load_reader

        _create_fake_ocean_wind_speed_file(tmp_path / OCEAN_FILENAME)
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([tmp_path / OCEAN_FILENAME])
        r.create_filehandlers(loadables, fh_kwargs={"filter_wind_speed": filter_wspd})
        loaded_data_arrs = r.load(["WSPD"])
        assert loaded_data_arrs
        data_arr = loaded_data_arrs["WSPD"]
        data_id = data_arr.attrs["_satpy_id"]
        self._check_area(data_id, data_arr)
        self._check_fill(data_id, data_arr)
        self._check_attrs(data_arr)
        has_nan = np.isnan(data_arr.data.compute()[-1, -1])
        if filter_wspd:
            assert has_nan
        else:
            assert not has_nan

    def _load_with_filters(self, file_path, names, **fh_kwargs):
        from satpy.readers.core.loading import load_reader

        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([file_path])
        r.create_filehandlers(loadables, fh_kwargs=fh_kwargs)
        return r.load(names)

    @pytest.mark.parametrize(
        ("filename", "var_suffix"),
        [
            (SEAICE_NH_FILENAME, "_NH"),
            (SEAICE_SH_FILENAME, "_SH"),
        ],
    )
    @pytest.mark.parametrize(
        ("filter_sea_ice", "expected_pixels"),
        [
            (False, list(SEAICE_PIXELS)),
            (True, ["good"]),
        ],
    )
    def test_sea_ice_filtering(self, tmp_path, filename, var_suffix, filter_sea_ice, expected_pixels):
        """Test that sea ice concentration can be filtered."""
        _create_fake_seaice_file(tmp_path / filename)
        var_name = "NASA_Team_2_Ice_Concentration" + var_suffix
        loaded_data_arrs = self._load_with_filters(tmp_path / filename, [var_name], filter_sea_ice=filter_sea_ice)
        data_arr = loaded_data_arrs[var_name]
        assert isinstance(data_arr.data, da.Array)
        self._check_attrs(data_arr)
        data = data_arr.data.compute()
        np.testing.assert_array_equal(~np.isnan(data), _expected_seaice_valid_mask(expected_pixels))
        assert data[0, 0] == SEAICE_PIXELS["good"][0]

    def test_sea_ice_filtering_only_applies_to_ice_concentration(self, tmp_path):
        """Test that other variables in the SEAICE files are not filtered."""
        _create_fake_seaice_file(tmp_path / SEAICE_NH_FILENAME)
        names = ["NASA_Team_2_Multiyear_Ice_NH", "Flags_NH"]
        unfiltered = self._load_with_filters(tmp_path / SEAICE_NH_FILENAME, names)
        filtered = self._load_with_filters(tmp_path / SEAICE_NH_FILENAME, names, filter_sea_ice=True)
        for name in names:
            np.testing.assert_array_equal(filtered[name].data.compute(), unfiltered[name].data.compute())

    def test_sea_ice_filter_accepted_by_non_seaice_files(self, tmp_path):
        """Test that all file types accept the sea ice filter as it is given to every file handler."""
        _create_fake_ocean_wind_speed_file(tmp_path / OCEAN_FILENAME)
        loaded_data_arrs = self._load_with_filters(tmp_path / OCEAN_FILENAME, ["WSPD"], filter_sea_ice=True)
        assert not np.isnan(loaded_data_arrs["WSPD"].data.compute()).any()
