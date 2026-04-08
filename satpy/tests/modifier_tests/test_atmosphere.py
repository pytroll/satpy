# Copyright (c) 2026 Satpy developers
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

"""Tests for modifiers in modifiers/atmosphere.py."""
import datetime as dt
import unittest
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition


class TestPSPRayleighReflectance:
    """Test the pyspectral-based Rayleigh correction modifier."""

    def _make_data_area(self):
        """Create test area definition and data."""
        rows = 3
        cols = 5
        area = AreaDefinition(
            "some_area_name", "On-the-fly area", "geosabii",
            {"a": "6378137.0", "b": "6356752.31414", "h": "35786023.0", "lon_0": "-89.5", "proj": "geos", "sweep": "x",
             "units": "m"},
            cols, rows,
            (-5434894.954752679, -5434894.964451744, 5434894.964451744, 5434894.954752679))

        data = np.zeros((rows, cols)) + 25
        data[1, :] += 25
        data[2, :] += 50
        data = da.from_array(data, chunks=2)
        return area, data

    def _create_test_data(self, name, wavelength, resolution):
        area, dnb = self._make_data_area()
        input_band = xr.DataArray(dnb,
                                  dims=("y", "x"),
                                  attrs={
                                      "platform_name": "Himawari-8",
                                      "calibration": "reflectance", "units": "%", "wavelength": wavelength,
                                      "name": name, "resolution": resolution, "sensor": "ahi",
                                      "start_time": "2017-09-20 17:30:40.800000",
                                      "end_time": "2017-09-20 17:41:17.500000",
                                      "area": area, "ancillary_variables": [],
                                      "orbital_parameters": {
                                          "satellite_nominal_longitude": -89.5,
                                          "satellite_nominal_latitude": 0.0,
                                          "satellite_nominal_altitude": 35786023.4375,
                                      },
                                  })

        red_band = xr.DataArray(dnb,
                                dims=("y", "x"),
                                attrs={
                                    "platform_name": "Himawari-8",
                                    "calibration": "reflectance", "units": "%", "wavelength": (0.62, 0.64, 0.66),
                                    "name": "B03", "resolution": 500, "sensor": "ahi",
                                    "start_time": "2017-09-20 17:30:40.800000",
                                    "end_time": "2017-09-20 17:41:17.500000",
                                    "area": area, "ancillary_variables": [],
                                    "orbital_parameters": {
                                        "satellite_nominal_longitude": -89.5,
                                        "satellite_nominal_latitude": 0.0,
                                        "satellite_nominal_altitude": 35786023.4375,
                                    },
                                })
        fake_angle_data = da.ones_like(dnb, dtype=np.float32) * 90.0
        angle1 = xr.DataArray(fake_angle_data,
                              dims=("y", "x"),
                              attrs={
                                  "platform_name": "Himawari-8",
                                  "calibration": "reflectance", "units": "%", "wavelength": wavelength,
                                  "name": "satellite_azimuth_angle", "resolution": resolution, "sensor": "ahi",
                                  "start_time": "2017-09-20 17:30:40.800000",
                                  "end_time": "2017-09-20 17:41:17.500000",
                                  "area": area, "ancillary_variables": [],
                              })
        return input_band, red_band, angle1, angle1, angle1, angle1

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize(
        ("name", "wavelength", "resolution", "aerosol_type", "reduce_lim_low", "reduce_lim_high", "reduce_strength"),
        [
            ("B01", (0.45, 0.47, 0.49), 1000, "rayleigh_only", 70, 95, 1),
            ("B02", (0.49, 0.51, 0.53), 1000, "rayleigh_only", 70, 95, 1),
            ("B03", (0.62, 0.64, 0.66), 500, "rayleigh_only", 70, 95, 1),
            ("B01", (0.45, 0.47, 0.49), 1000, "rayleigh_only", -95, -70, -1),
        ]
    )
    def test_rayleigh_corrector(
            self, tmp_path, name, wavelength, resolution, aerosol_type,
            reduce_lim_low, reduce_lim_high, reduce_strength, dtype):
        """Test PSPRayleighReflectance with fake data."""
        from pyspectral.testing import mock_rayleigh

        from satpy.modifiers.atmosphere import PSPRayleighReflectance

        ray_cor = PSPRayleighReflectance(name=name, atmosphere="us-standard", aerosol_types=aerosol_type,
                                         reduce_lim_low=reduce_lim_low, reduce_lim_high=reduce_lim_high,
                                         reduce_strength=reduce_strength)
        assert ray_cor.attrs["name"] == name
        assert ray_cor.attrs["atmosphere"] == "us-standard"
        assert ray_cor.attrs["aerosol_types"] == aerosol_type
        assert ray_cor.attrs["reduce_lim_low"] == reduce_lim_low
        assert ray_cor.attrs["reduce_lim_high"] == reduce_lim_high
        assert ray_cor.attrs["reduce_strength"] == reduce_strength

        input_band, red_band, *_ = self._create_test_data(name, wavelength, resolution)
        with mock_rayleigh(rayleigh_dir=tmp_path):
            res = ray_cor([input_band.astype(dtype), red_band.astype(dtype)])

        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.dtype == dtype
        data = res.values
        assert data.shape == (3, 5)
        assert data.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("as_optionals", [False, True])
    def test_rayleigh_with_angles(self, tmp_path, as_optionals, dtype):
        """Test PSPRayleighReflectance with angles provided."""
        from pyspectral.testing import mock_rayleigh

        from satpy.modifiers.atmosphere import PSPRayleighReflectance

        aerosol_type = "rayleigh_only"
        ray_cor = PSPRayleighReflectance(name="B01", atmosphere="us-standard", aerosol_types=aerosol_type)
        prereqs, opt_prereqs = self._get_angles_prereqs_and_opts(as_optionals, dtype)
        with mock.patch("satpy.modifiers.atmosphere.get_angles") as get_angles, mock_rayleigh(rayleigh_dir=tmp_path):
            res = ray_cor(prereqs, opt_prereqs)
        get_angles.assert_not_called()

        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.dtype == dtype
        data = res.values
        assert data.shape == (3, 5)
        assert data.dtype == dtype

    def _get_angles_prereqs_and_opts(self, as_optionals, dtype):
        wavelength = (0.45, 0.47, 0.49)
        resolution = 1000
        input_band, red_band, *angles = self._create_test_data("B01", wavelength, resolution)
        prereqs = [input_band.astype(dtype), red_band.astype(dtype)]
        opt_prereqs = []
        angles = [a.astype(dtype) for a in angles]
        if as_optionals:
            opt_prereqs = angles
        else:
            prereqs += angles
        return prereqs, opt_prereqs


class TestPSPAtmosphericalCorrection(unittest.TestCase):
    """Test the pyspectral-based atmospheric correction modifier."""

    def test_call(self):
        """Test atmospherical correction."""
        from pyresample.geometry import SwathDefinition

        from satpy.modifiers import PSPAtmosphericalCorrection

        # Patch methods
        lons = np.zeros((5, 5))
        lons[1, 1] = np.inf
        lons = da.from_array(lons, chunks=5)
        lats = np.zeros((5, 5))
        lats[1, 1] = np.inf
        lats = da.from_array(lats, chunks=5)
        area = SwathDefinition(lons, lats)
        stime = dt.datetime(2020, 1, 1, 12, 0, 0)
        orb_params = {
            "satellite_actual_altitude": 12345678,
            "nadir_longitude": 0.0,
            "nadir_latitude": 0.0,
        }
        band = xr.DataArray(da.zeros((5, 5)),
                            attrs={"area": area,
                                   "start_time": stime,
                                   "name": "name",
                                   "platform_name": "platform",
                                   "sensor": "sensor",
                                   "orbital_parameters": orb_params},
                            dims=("y", "x"))

        # Perform atmospherical correction
        psp = PSPAtmosphericalCorrection(name="dummy")
        res = psp(projectables=[band])
        res.compute()
