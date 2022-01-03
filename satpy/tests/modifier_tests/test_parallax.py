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
from pyproj import Geod
from pyresample import create_area_def


def _get_fake_areas(center, sizes, resolution):
    """Get multiple square areas with the same center.

    Returns multiple square areas centered at the same location

    Args:
        center (Tuple[float, float]): Center of all areass
        sizes (List[int]): Sizes of areas
        resolution (float): Resolution of fake area.

    Returns:
        List of areas.
    """
    return [create_area_def(
        "fribullus_xax",
        "epsg:4326",
        units="degrees",
        resolution=resolution,
        center=center,
        shape=(size, size))
        for size in sizes]


def _get_attrs(lat, lon, height=35_000_000):
    """Get attributes for datasets in fake scene."""
    return {
        "orbital_parameters": {
            "satellite_actual_altitude": height,
            "satellite_actual_longitude": lon,
            "satellite_actual_latitude": lat},
        "units": "m"
        }


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
def cloud(request):
    """Give me a cloud.

    Args (via request fixture):
        int: size of
    """
    (size, outer_lo, outer_hi, outer_val, inner_lo, inner_hi, inner_val) = request.param
    cth = np.full((size, size), np.nan)
    cth[outer_lo:outer_hi, outer_lo:outer_hi] = outer_val
    cth[inner_lo:inner_hi, inner_lo:inner_hi] = inner_val
    return cth


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


@pytest.mark.parametrize("lat,lon", [(0, 0), (0, 40), (0, 179.9)])
@pytest.mark.parametrize("resolution", [0.01, 0.5, 10])
def test_forward_parallax_cloudy_ssp(lat, lon, resolution):
    """Test parallax correction for fully cloudy scene at SSP."""
    from ...modifiers.parallax import forward_parallax

    N = 5
    lats = np.linspace(lat-N*resolution, lat+N*resolution, 25).reshape(N, N)
    lons = np.linspace(lon-N*resolution, lon+N*resolution, 25).reshape(N, N).T
    height = np.full((N, N), 10)  # constant high clouds at 10 km
    sat_alt = 35_000.
    (corr_lon, corr_lat) = forward_parallax(
        lon, lat, sat_alt, lons, lats, height)
    # confirm movements behave as expected
    geod = Geod(ellps="sphere")
    # need to use np.tile here as geod.inv doesn't seem to broadcast (not
    # when turning lon/lat in arrays of size (1, 1) either)
    corr_dist = geod.inv(np.tile(lon, [N, N]), np.tile(lat, [N, N]), corr_lon, corr_lat)[2]
    corr_delta = geod.inv(corr_lon, corr_lat, lons, lats)[2]
    uncorr_dist = geod.inv(np.tile(lon, [N, N]), np.tile(lat, [N, N]), lons, lats)[2]
    # should be equal at SSP and nowhere else
    np.testing.assert_allclose(corr_delta[2, 2], 0, atol=1e-9)
    assert np.isclose(corr_delta, 0, atol=1e-9).sum() == 1
    # should always get closer to SSP
    assert corr_dist[2, 2] == 0.0
    assert uncorr_dist[2, 2] == 0.0
    assert (corr_dist <= uncorr_dist).all()
    # should be larger the further we get from SSP
    assert (np.diff(corr_delta[N//2, :N//2+1]) < 0).all()
    assert (np.diff(corr_delta[N//2, N//2:]) > 0).all()
    assert (np.diff(corr_delta[N//2:, N//2]) > 0).all()
    assert (np.diff(corr_delta[:N//2+1, N//2]) < 0).all()
    assert (np.diff(np.diag(corr_delta)[:N//2+1]) < 0).all()
    assert (np.diff(np.diag(corr_delta)[N//2:]) > 0).all()


def test_forward_parallax_cloudy_slant():
    """Test parallax correction for fully cloudy scene (not SSP)."""
    from ...modifiers.parallax import forward_parallax
    sat_lat = sat_lon = 0
    lat = np.linspace(-20, 20, 25).reshape(5, 5)
    lon = np.linspace(-20, 20, 25).reshape(5, 5).T
    height = np.full((5, 5), 10)  # constant high clouds at 10 km
    sat_alt = 35_000.
    (corr_lon, corr_lat) = forward_parallax(
        sat_lon, sat_lat, sat_alt, lon, lat, height)
    # reference value from Simon Proud
    np.testing.assert_allclose(
        corr_lat[4, 4], 19.955, rtol=5e-4)
    np.testing.assert_allclose(
        corr_lon[4, 4], 19.960, rtol=5e-4)


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


@pytest.mark.parametrize("center", [(0, 0), (80, -10), (-180, 5)])
@pytest.mark.parametrize("sizes", [[5, 9]])
@pytest.mark.parametrize("resolution", [0.05, 1, 10])
def test_init_parallaxcorrection(center, sizes, resolution):
    """Test that ParallaxCorrection class can be instantiated."""
    from ...modifiers.parallax import ParallaxCorrection
    fake_area = _get_fake_areas(center, sizes, resolution)[0]
    ParallaxCorrection(fake_area)


@pytest.mark.parametrize("sat_lat,sat_lon,ar_lat,ar_lon",
                         [(0, 0, 0, 0), (0, 0, 40, 0)])
@pytest.mark.parametrize("resolution", [0.01, 0.5, 10])
def test_correct_area_clearsky(sat_lat, sat_lon, ar_lat, ar_lon, resolution):
    """Test that ParallaxCorrection doesn't change clearsky geolocation."""
    from ...modifiers.parallax import ParallaxCorrection
    from ..utils import make_fake_scene
    small = 5
    large = 9
    (fake_area_small, fake_area_large) = _get_fake_areas(
            (ar_lon, ar_lat), [small, large], resolution)
    corrector = ParallaxCorrection(fake_area_small)

    sc = make_fake_scene(
            {"CTH_clear": np.full((large, large), np.nan)},
            daskify=False,
            area=fake_area_large,
            common_attrs=_get_attrs(sat_lat, sat_lon, 35_000_000))

    new_area = corrector(sc["CTH_clear"])
    np.testing.assert_allclose(
            new_area.get_lonlats(),
            fake_area_small.get_lonlats())


@pytest.mark.parametrize("lat,lon",
                         [(0, 0), (0, 40), (0, 180),
                          (90, 0)])  # relevant for Арктика satellites
@pytest.mark.parametrize("resolution", [0.01, 0.5, 10])
def test_correct_area_ssp(lat, lon, resolution):
    """Test that ParallaxCorrection doesn't touch SSP."""
    from ...modifiers.parallax import ParallaxCorrection
    from ..utils import make_fake_scene
    small = 5
    large = 9
    (fake_area_small, fake_area_large) = _get_fake_areas(
            (lon, lat), [small, large], resolution)
    corrector = ParallaxCorrection(fake_area_small)

    sc = make_fake_scene(
            {"CTH_constant": np.full((large, large), 10000)},
            daskify=False,
            area=fake_area_large,
            common_attrs=_get_attrs(lat, lon, 35_000_000))
    new_area = corrector(sc["CTH_constant"])
    assert new_area.shape == fake_area_small.shape
    old_lonlats = fake_area_small.get_lonlats()
    new_lonlats = new_area.get_lonlats()
    np.testing.assert_allclose(
            old_lonlats[0][2, 2],
            new_lonlats[0][2, 2],
            atol=1e-9)
    np.testing.assert_allclose(
            old_lonlats[0][2, 2],
            lon,
            atol=1e-9)
    np.testing.assert_allclose(
            old_lonlats[1][2, 2],
            new_lonlats[1][2, 2],
            atol=1e-9)
    np.testing.assert_allclose(
            old_lonlats[1][2, 2],
            lat,
            atol=1e-9)


def test_correct_area_partlycloudy():
    """Test ParallaxCorrection for partly cloudy situation."""
    from ...modifiers.parallax import ParallaxCorrection
    from ..utils import make_fake_scene
    small = 5
    large = 9
    (fake_area_small, fake_area_large) = _get_fake_areas(
            (0, 50), [small, large], 0.1)
    corrector = ParallaxCorrection(fake_area_small)

    sc = make_fake_scene(
           {"CTH": np.array([
                [np.nan, np.nan, 5., 6., 7., 6., 5., np.nan, np.nan],
                [np.nan, 6., 7., 7., 7., np.nan, np.nan, np.nan, np.nan],
                [np.nan, 7., 8., 9., np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, 7., 7., 7., np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, 4., 3., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, 5., 8., 8., 8., 6, np.nan, np.nan],
                [np.nan, 9., 9., 9., 9., 9., 9., 9., np.nan],
                [np.nan, 9., 9., 9., 9., 9., 9., 9., np.nan],
                [np.nan, 9., 9., 9., 9., 9., 9., 9., np.nan],
                ])},
           daskify=False,
           area=fake_area_large,
           common_attrs=_get_attrs(0, 0, 40_000_000))
    new_area = corrector(sc["CTH"])
    assert new_area.shape == fake_area_small.shape

    # TODO: add more tests here


@pytest.mark.parametrize("cloud", [(9, 2, 8, 5, 3, 6, 8)], indirect=["cloud"])
def test_cloud(cloud):
    """Test using cloud."""
    pass
