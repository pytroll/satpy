# Copyright (c) 2025 Satpy developers
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
"""Module for testing the satpy.readers.pace_oci_l1b_nc module."""

import os
import tempfile
from contextlib import suppress
from datetime import datetime

import numpy as np
import pytest
from netCDF4 import Dataset

from satpy import Scene

rng = np.random.default_rng()

n_scan = 10
n_pix = 100
n_blu = 10
n_red = 6
n_swir = 3

blu_wvl = np.array([400.0, 401.0, 402.0, 403.0, 404.0, 405.0, 406.0, 407.0, 408.0, 409.0])
blu_irr = np.array([1102.0, 1103.0, 1104.0, 1105.0, 1106.0, 1107.0, 1108.0, 1109.0, 1110.0, 1111.0])

red_wvl = np.array([600.0, 601.0, 602.0, 603.0, 604.0, 605.0])
red_irr = np.array([102, 104, 107, 99, 102, 59.])

swir_wvl = np.array([1000.0, 1100.0, 1200.0])
swir_bw = np.array([10.0, 20.0, 30.0])
swir_irr = np.array([331.0, 332.0, 333.0])

blu_cdata = rng.uniform(size=(n_blu, n_scan, n_pix), low=0, high=1.04)
red_cdata = rng.uniform(size=(n_red, n_scan, n_pix), low=0, high=1.04)
swir_cdata = rng.uniform(size=(n_swir, n_scan, n_pix), low=0, high=1.04)

sza_data = rng.uniform(size=(n_scan, n_pix), low=0, high=88)
lon_data = rng.uniform(size=(n_scan, n_pix), low=-180, high=180)
lat_data = rng.uniform(size=(n_scan, n_pix), low=-90, high=90)

esd = 0.9989


@pytest.fixture
def temp_nc_file():
    """Create a temporary netCDF file for the tests."""
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "PACE_OCI.20250320T143701.L1B.V3.nc")

    with Dataset(temp_file, "w", format="NETCDF4") as nc:
        nc.earth_sun_distance_correction = esd

        nc.createDimension("scans", n_scan)
        nc.createDimension("pixels", n_pix)
        nc.createDimension("blue_bands", n_blu)
        nc.createDimension("red_bands", n_red)
        nc.createDimension("swir_bands", n_swir)

        # Global attributes
        nc.time_coverage_start = "2023-01-01T00:29:15.000Z"
        nc.time_coverage_end = "2023-01-01T00:33:11.020Z"

        # Variables
        sensor_band_parameters = nc.createGroup("sensor_band_parameters")
        sensor_band_parameters.createVariable("blue_solar_irradiance", "f4", dimensions=("blue_bands",))
        sensor_band_parameters.createVariable("blue_wavelength", "f4", dimensions=("blue_bands",))

        sensor_band_parameters.createVariable("red_solar_irradiance", "f4", dimensions=("red_bands",))
        sensor_band_parameters.createVariable("red_wavelength", "f4", dimensions=("red_bands",))

        sensor_band_parameters.createVariable("SWIR_solar_irradiance", "f4", dimensions=("swir_bands",))
        sensor_band_parameters.createVariable("SWIR_wavelength", "f4", dimensions=("swir_bands",))
        sensor_band_parameters.createVariable("SWIR_bandpass", "f4", dimensions=("swir_bands",))

        observation_data = nc.createGroup("observation_data")
        observation_data.createVariable("rhot_blue", "f4", ("blue_bands", "scans", "pixels"))
        observation_data.createVariable("rhot_red", "f4", ("red_bands", "scans", "pixels"))
        observation_data.createVariable("rhot_SWIR", "f4", ("swir_bands", "scans", "pixels"))

        geolocation_data = nc.createGroup("geolocation_data")
        geolocation_data.createVariable("solar_zenith", "f4", ("scans", "pixels"))
        geolocation_data.createVariable("latitude", "f4", ("scans", "pixels"))
        geolocation_data.createVariable("longitude", "f4", ("scans", "pixels"))

        # Assign values
        sensor_band_parameters["blue_solar_irradiance"][:] = blu_irr
        sensor_band_parameters["blue_wavelength"][:] = blu_wvl

        sensor_band_parameters["red_solar_irradiance"][:] = red_irr
        sensor_band_parameters["red_wavelength"][:] = red_wvl

        sensor_band_parameters["SWIR_solar_irradiance"][:] = swir_irr
        sensor_band_parameters["SWIR_wavelength"][:] = swir_wvl
        sensor_band_parameters["SWIR_bandpass"][:] = swir_bw

        observation_data["rhot_blue"][:] = blu_cdata
        observation_data["rhot_red"][:] = red_cdata
        observation_data["rhot_SWIR"][:] = swir_cdata

        geolocation_data["solar_zenith"][:] = sza_data
        geolocation_data["longitude"][:] = lon_data
        geolocation_data["latitude"][:] = lat_data

        geolocation_data["latitude"].setncattr("long_name", "Latitudes of pixel locations")
        geolocation_data["latitude"].setncattr("valid_min", "-90.0")
        geolocation_data["latitude"].setncattr("valid_max", "90.0")
        geolocation_data["latitude"].setncattr("units", "degrees_north")
        geolocation_data["latitude"].setncattr("coordinates", "latitude longitude")

        geolocation_data["longitude"].setncattr("long_name", "Longitudes of pixel locations")
        geolocation_data["longitude"].setncattr("valid_min", "-180.0")
        geolocation_data["longitude"].setncattr("valid_max", "180.0")
        geolocation_data["longitude"].setncattr("units", "degrees_east")
        geolocation_data["longitude"].setncattr("coordinates", "latitude longitude")

    yield temp_file

    with suppress(OSError):
        os.remove(temp_file)
        os.rmdir(temp_dir)


@pytest.fixture
def scn(temp_nc_file):
    """Create a Scene object for the tests."""
    return Scene(reader="pace_oci_l1b_nc", filenames=[temp_nc_file])


def test_start_time_property(scn):
    """Test the start_time property."""
    assert scn.start_time == datetime(2023, 1, 1, 0, 29, 15)


def test_end_time_property(scn):
    """Test the end_time property."""
    assert scn.end_time == datetime(2023, 1, 1, 0, 33, 11, 20000)


def test_available_datasets(scn):
    """Test the available_datasets method to check we load correct number dynamically."""
    datasets = list(scn.available_dataset_ids())
    assert len(datasets) == 68


def test_get_dataset_refl(scn):
    """Test getting a dataset with reflectance calibration."""
    cname_b = "chan_blue_400"
    cname_s = "chan_swir_1000"
    scn.load([cname_b, cname_s], calibration="reflectance")
    np.testing.assert_allclose(scn[cname_b], blu_cdata[0, :, :] * 100., rtol=1e-6)
    np.testing.assert_allclose(scn[cname_s], swir_cdata[0, :, :] * 100., rtol=1e-6)

    assert scn[cname_b].attrs["standard_name"] == "toa_bidirectional_reflectance"
    assert scn[cname_s].attrs["standard_name"] == "toa_bidirectional_reflectance"
    assert scn[cname_b].attrs["units"] == "%"
    assert scn[cname_s].attrs["units"] == "%"


def test_get_dataset_radi(scn):
    """Test getting a dataset with radiance calibration."""
    cname_b = "chan_blue_400"
    cname_s = "chan_swir_1000"
    scn.load([cname_b, cname_s], calibration="radiance")

    rad_blu = (blu_cdata[0, :, :] * blu_irr[0] * np.cos(np.radians(sza_data))) / (esd * np.pi)
    rad_swi = (swir_cdata[0, :, :] * swir_irr[0] * np.cos(np.radians(sza_data))) / (esd * np.pi)

    np.testing.assert_allclose(scn[cname_b], rad_blu, rtol=5e-6)
    np.testing.assert_allclose(scn[cname_s], rad_swi, rtol=5e-6)

    assert scn[cname_b].attrs["standard_name"] == "toa_outgoing_radiance_per_unit_wavelength"
    assert scn[cname_s].attrs["standard_name"] == "toa_outgoing_radiance_per_unit_wavelength"
    assert scn[cname_b].attrs["units"] == "W m-2 sr-1 um-1"
    assert scn[cname_s].attrs["units"] == "W m-2 sr-1 um-1"


def test_load_geo(scn):
    """Test loading geolocation data."""
    scn.load(["latitude", "longitude", "solar_zenith_angle"])
    np.testing.assert_allclose(scn["latitude"], lat_data)
    np.testing.assert_allclose(scn["longitude"], lon_data)
    np.testing.assert_allclose(scn["solar_zenith_angle"], sza_data)
