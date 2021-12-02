# Copyright (c) 2021 Satpy developers
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

"""Tests related to parallax correction."""


import dask.array as da
import numpy as np
import pytest
from pyresample import create_area_def


@pytest.fixture
def fake_area_5x5_wide():
    """Get a 5×5 fake widely spaced area to use for parallax correction testing."""
    return create_area_def(
        "fribullus_xax",
        "epsg:4326",
        units="degrees",
        area_extent=[-10, -10, 10, 10],
        shape=(5, 5))


@pytest.fixture
def fake_area_5x5_tight():
    """Get a 5×5 fake tightly spaced area to use for parallax correction testing."""
    return create_area_def(
        "fribullus_xax",
        "epsg:4326",
        units="degrees",
        area_extent=[0, 40, 0.1, 40.1],
        shape=(5, 5))


@pytest.fixture
def fake_area_10x10_tight():
    """Get a 10×10 fake tightly spaced area to use for parallax correction testing.

    This area is an extension of fake_area_5x5_tight.
    """
    return create_area_def(
        "fribullux xax",
        "epsg:4326",
        units="degrees",
        area_extent=[-0.04, 39.96, 0.16, 40.16],
        shape=(10, 10))


def test_forward_parallax_ssp():
    """Test that at SSP, parallax correction does nothing."""
    from ...modifiers.parallax import forward_parallax
    sat_lat = sat_lon = lon = lat = 0.
    height = 5000.
    sat_alt = 30_000_000.
    corr_lon, corr_lat = forward_parallax(
        sat_lon, sat_lat, sat_alt, lon, lat, height)
    assert corr_lon == corr_lat == 0


def test_forward_parallax_clearsky():
    """Test parallax correction for clearsky case (returns NaN)."""
    from ...modifiers.parallax import forward_parallax
    sat_lat = sat_lon = 0
    lat = np.linspace(-20, 20, 25).reshape(5, 5)
    lon = np.linspace(-20, 20, 25).reshape(5, 5).T
    height = np.full((5, 5), np.nan)  # no CTH --> clearsky
    sat_alt = 35_000.  # km
    (corr_lon, corr_lat) = forward_parallax(
        sat_lon, sat_lat, sat_alt, lon, lat, height)
    # clearsky becomes NaN
    assert np.isnan(corr_lon).all()
    assert np.isnan(corr_lat).all()


def test_forward_parallax_cloudy():
    """Test parallax correction for fully cloudy scene."""
    from ...modifiers.parallax import forward_parallax
    sat_lat = sat_lon = 0
    lat = np.linspace(-20, 20, 25).reshape(5, 5)
    lon = np.linspace(-20, 20, 25).reshape(5, 5).T
    height = np.full((5, 5), 10)  # constant high clouds at 10 km
    sat_alt = 35_000.
    (corr_lon, corr_lat) = forward_parallax(
        sat_lon, sat_lat, sat_alt, lon, lat, height)
    # should be equal only at SSP
    delta_lon = corr_lon - lon
    delta_lat = corr_lat - lat
    assert delta_lat[2, 2] == delta_lon[2, 2] == 0
    assert (delta_lat == 0).sum() == 1
    assert (delta_lon == 0).sum() == 1
    # should always get closer to SSP
    assert (abs(corr_lon) <= abs(lon)).all()
    assert (abs(corr_lat) <= abs(lat)).all()
    # should be larger the further we get from SSP
    assert (delta_lon[2, 1:] < delta_lon[2, :-1]).all()
    assert (delta_lat[1:, 1] < delta_lat[:-1, 1]).all()
    # reference value from Simon Proud
    np.testing.assert_allclose(
        corr_lat[4, 4], 19.955884)  # FIXME confirm reference value
    np.testing.assert_allclose(
        corr_lon[4, 4], 19.950061)  # FIXME confirm reference value


def test_forward_parallax_mixed():
    """Test parallax correction for mixed cloudy case."""
    from ...modifiers.parallax import forward_parallax

    sat_lon = sat_lat = 0
    sat_alt = 35_785_831.0
    lon = da.array([[-20, -10, 0, 10, 20]]*5)
    lat = da.array([[-20, -10, 0, 10, 20]]*5).T
    alt = da.array([
        [np.nan, np.nan, 5., 6., np.nan],
        [np.nan, 6., 7., 7., 7.],
        [np.nan, 7., 8., 9., np.nan],
        [np.nan, 7., 7., 7., np.nan],
        [np.nan, 4., 3., np.nan, np.nan]])
    (corrected_lon, corrected_lat) = forward_parallax(
        sat_lon, sat_lat, sat_alt, lon, lat, alt)
    assert corrected_lon.shape == lon.shape
    assert corrected_lat.shape == lat.shape
    # lon/lat should be nan for clear-sky pixels
    assert np.isnan(corrected_lon[np.isnan(alt)]).all()
    assert np.isnan(corrected_lat[np.isnan(alt)]).all()
    # otherwise no nans
    assert np.isfinite(corrected_lon[~np.isnan(alt)]).all()
    assert np.isfinite(corrected_lat[~np.isnan(alt)]).all()


def test_init_parallaxcorrection(fake_area_5x5_wide):
    """Test that ParallaxCorrection class can be instantiated."""
    from ...modifiers.parallax import ParallaxCorrection
    ParallaxCorrection(fake_area_5x5_wide)


def test_correct_area(fake_area_5x5_tight, fake_area_10x10_tight):
    """Test that ParallaxCorrection can correct an Areadefinition."""
    from ...modifiers.parallax import ParallaxCorrection
    from ..utils import make_fake_scene
    corrector = ParallaxCorrection(fake_area_5x5_tight)

    sc = make_fake_scene(
            {"CTH_constant": np.full((10, 10), 10000),
             "CTH_clear": np.full((10, 10), np.nan)},
            daskify=False,
            area=fake_area_10x10_tight,
            common_attrs={
                "orbital_parameters": {
                    "satellite_actual_altitude": 35_000_000,
                    "satellite_actual_longitude": 0.0,
                    "satellite_actual_latitude": 0.0},
                "units": "m"
                    })
    # check that lat/lons remain the same for clearsky
    new_area = corrector(sc["CTH_clear"])
    np.testing.assert_allclose(
            new_area.get_lonlats(),
            fake_area_5x5_tight.get_lonlats())
    new_area = corrector(sc["CTH_constant"])
    assert new_area.shape == fake_area_5x5_tight.shape
    old_lonlats = fake_area_5x5_tight.get_lonlats()
    new_lonlats = new_area.get_lonlats()
    # no change at SSP where old_lat = new_lat = 0
    assert (old_lonlats[0][2, 2] == old_lonlats[1][2, 2]
            == new_lonlats[0][2, 2] == new_lonlats[1][2, 2])
    # no change where we had NaN
    np.testing.assert_array_equal(
        old_lonlats[0][(clear := np.isnan(sc["CTH_mixed"].values))],
        new_lonlats[0][clear])
