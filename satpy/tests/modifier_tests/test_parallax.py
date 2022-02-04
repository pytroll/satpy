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

import datetime
import logging
import math
import unittest.mock

import dask.array as da
import numpy as np
import pyorbital.tlefile
import pyresample.kd_tree
import pytest
import xarray as xr
from pyproj import Geod
from pyresample import create_area_def
from pyresample.bilinear import NumpyBilinearResampler


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


def _get_attrs(lat, lon, height=35_000):
    """Get attributes for datasets in fake scene."""
    return {
        "orbital_parameters": {
            "satellite_actual_altitude": height,  # in km above surface
            "satellite_actual_longitude": lon,
            "satellite_actual_latitude": lat},
        "units": "m"  # does not apply to orbital parameters, I think!
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
    """Return an array representing a square cloud.

    Return an array representing a square cloud with a larger lower and a
    smaller higher part.

    Args (via request fixture):
        int: size of array
        int: Index of start of outer cloud.
        int: Index of end of outer cloud.
        int: Value (CTH) of outer cloud.
        int: Index of start of inner cloud.
        int: Index of start of outer cloud.
        int: Value (CTH) of inner cloud.
    """
    (size, outer_lo, outer_hi, outer_val, inner_lo, inner_hi, inner_val) = request.param
    cth = np.full((size, size), np.nan)
    cth[outer_lo:outer_hi, outer_lo:outer_hi] = outer_val
    cth[inner_lo:inner_hi, inner_lo:inner_hi] = inner_val
    return cth


@pytest.fixture
def resampler(request):
    """Return a resampler function."""
    resampler_name = request.param
    if resampler_name == "nearest":
        return pyresample.kd_tree.resample_nearest
    elif resampler_name == "bilinear":
        def resample_bilinear(source_area, what, base_area, radius_of_influence):
            sampler = NumpyBilinearResampler(
                    source_area, base_area, radius_of_influence)
            return sampler.resample(what)
        return resample_bilinear
    else:
        raise ValueError("resampler name must be 'nearest' or 'bilinear', "
                         f"got {resampler_name!s}")


def test_forward_parallax_ssp():
    """Test that at SSP, parallax correction does nothing."""
    from ...modifiers.parallax import forward_parallax
    sat_lat = sat_lon = lon = lat = 0.
    height = 5000.  # m
    sat_alt = 30_000_000.  # m
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
    sat_alt = 35_000_000.  # m above surface
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
    height = np.full((N, N), 10_000)  # constant high clouds at 10 km
    sat_alt = 35_000_000.  # satellite at 35 Mm
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
    assert (uncorr_dist - corr_dist >= -1e-8).all()
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
    height = np.full((5, 5), 10_000)  # constant high clouds at 10 km
    sat_alt = 35_000_000.  # satellite at 35 Mm
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
    sat_alt = 35_785_831.0  # m
    lon = da.array([[-20, -10, 0, 10, 20]]*5)
    lat = da.array([[-20, -10, 0, 10, 20]]*5).T
    alt = da.array([
        [np.nan, np.nan, 5000., 6000., np.nan],
        [np.nan, 6000., 7000., 7000., 7000.],
        [np.nan, 7000., 8000., 9000., np.nan],
        [np.nan, 7000., 7000., 7000., np.nan],
        [np.nan, 4000., 3000., np.nan, np.nan]])
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


def test_forward_parallax_horizon():
    """Test that exception is raised if satellites exactly at the horizon.

    Test the rather unlikely case of a satellite elevation of exactly 0
    """
    from ...modifiers.parallax import forward_parallax
    sat_lat = sat_lon = lon = lat = 0.
    height = 5000.
    sat_alt = 30_000_000.
    with unittest.mock.patch("satpy.modifiers.parallax.get_observer_look") as smpg:
        smpg.return_value = (0, 0)
        with pytest.raises(NotImplementedError):
            forward_parallax(sat_lon, sat_lat, sat_alt, lon, lat, height)


@pytest.mark.parametrize("center", [(0, 0), (80, -10), (-180, 5)])
@pytest.mark.parametrize("sizes", [[5, 9]])
@pytest.mark.parametrize("resolution", [0.05, 1, 10])
def test_init_parallaxcorrection(center, sizes, resolution):
    """Test that ParallaxCorrection class can be instantiated."""
    from ...modifiers.parallax import ParallaxCorrection
    fake_area = _get_fake_areas(center, sizes, resolution)[0]
    pc = ParallaxCorrection(fake_area)
    assert pc.base_area == fake_area
    assert pc.resampler_args["radius_of_influence"] == 50_000
    pc = ParallaxCorrection(
            fake_area,
            resampler_args={"radius_of_influence": 25_000})
    assert pc.resampler_args["radius_of_influence"] == 25_000


@pytest.fixture
def xfail_selected_clearsky_combis(request):
    """Mark certain parameter combinations as failing.

    Clearsky parallax correction fails for some combinations of parameters.
    This fixture helps to mark only those combinations as failing.
    """
    # solution inspired by https://stackoverflow.com/q/64349115/974555
    resolution = request.getfixturevalue("resolution")
    resampler = request.getfixturevalue("resampler")

    if (resampler.__name__ == "resample_bilinear" and
            math.isclose(resolution, 0.01)):
        request.node.add_marker(pytest.mark.xfail(
             reason="parallax correction may fail with bilinear"))


@pytest.mark.parametrize("sat_lat,sat_lon,ar_lat,ar_lon",
                         [(0, 0, 0, 0), (0, 0, 40, 0)])
@pytest.mark.parametrize("resolution", [0.01, 0.5, 10])
@pytest.mark.parametrize("resampler", ["nearest", "bilinear"],
                         indirect=["resampler"])
@pytest.mark.usefixtures('xfail_selected_clearsky_combis')
def test_correct_area_clearsky(sat_lat, sat_lon, ar_lat, ar_lon, resolution,
                               resampler, caplog):
    """Test that ParallaxCorrection doesn't change clearsky geolocation."""
    from ...modifiers.parallax import ParallaxCorrection
    from ..utils import make_fake_scene
    small = 5
    large = 9
    (fake_area_small, fake_area_large) = _get_fake_areas(
            (ar_lon, ar_lat), [small, large], resolution)
    corrector = ParallaxCorrection(fake_area_small, resampler)

    sc = make_fake_scene(
            {"CTH_clear": np.full((large, large), np.nan)},
            daskify=False,
            area=fake_area_large,
            common_attrs=_get_attrs(sat_lat, sat_lon, 35_000))

    with caplog.at_level(logging.DEBUG):
        new_area = corrector(sc["CTH_clear"])
    assert "Calculating parallax correction using heights from CTH_clear" in caplog.text
    np.testing.assert_allclose(
            new_area.get_lonlats(),
            fake_area_small.get_lonlats())


@pytest.fixture
def xfail_selected_ssp_combis(request):
    """Mark certain parameter combinations as failing.

    SSP parallax correction fails for some combinations of parameters.
    This fixture helps to mark only those combinations as failing.
    """
    # solution inspired by https://stackoverflow.com/q/64349115/974555
    lon = request.getfixturevalue("lon")
    lat = request.getfixturevalue("lat")
    resampler = request.getfixturevalue("resampler")

    if (resampler.__name__ == "resample_bilinear" and
            (lon == 180 or lat == 90)):
        request.node.add_marker(pytest.mark.xfail(
             reason="parallax correction may fail with bilinear"))


@pytest.mark.parametrize("lat,lon",
                         [(0, 0), (0, 40), (0, 180),
                          (90, 0)])  # relevant for Арктика satellites
@pytest.mark.parametrize("resolution", [0.01, 0.5, 10])
@pytest.mark.parametrize(
        "resampler",
        ["nearest", "bilinear"],
        indirect=["resampler"])
@pytest.mark.usefixtures('xfail_selected_ssp_combis')
def test_correct_area_ssp(lat, lon, resolution, resampler):
    """Test that ParallaxCorrection doesn't touch SSP."""
    from ...modifiers.parallax import ParallaxCorrection
    from ..utils import make_fake_scene
    small = 5
    large = 9
    (fake_area_small, fake_area_large) = _get_fake_areas(
            (lon, lat), [small, large], resolution)
    corrector = ParallaxCorrection(fake_area_small, resampler)

    sc = make_fake_scene(
            {"CTH_constant": np.full((large, large), 10000)},
            daskify=False,
            area=fake_area_large,
            common_attrs=_get_attrs(lat, lon, 35_000))
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


@pytest.mark.parametrize("daskify", [False, True])
@pytest.mark.parametrize(
        "resampler",
        ["nearest",
         pytest.param("bilinear", marks=pytest.mark.xfail(
             reason="parallax correction inaccurate with bilinear"))],
        indirect=["resampler"])
def test_correct_area_partlycloudy(daskify, resampler):
    """Test ParallaxCorrection for partly cloudy situation."""
    from ...modifiers.parallax import ParallaxCorrection
    from ..utils import make_fake_scene
    small = 5
    large = 9
    (fake_area_small, fake_area_large) = _get_fake_areas(
            (0, 50), [small, large], 0.1)
    (fake_area_lons, fake_area_lats) = fake_area_small.get_lonlats()
    corrector = ParallaxCorrection(fake_area_small, resampler)

    sc = make_fake_scene(
           {"CTH": np.array([
                [np.nan, np.nan, 5000., 6000., 7000., 6000., 5000., np.nan, np.nan],
                [np.nan, 6000., 7000., 7000., 7000., np.nan, np.nan, np.nan, np.nan],
                [np.nan, 7000., 8000., 9000., np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, 7000., 7000., 7000., np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, 4000., 3000., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, 5000., 8000., 8000., 8000., 6000., np.nan, np.nan],
                [np.nan, 9000., 9000., 9000., 9000., 9000., 9000., 9000., np.nan],
                [np.nan, 9000., 9000., 9000., 9000., 9000., 9000., 9000., np.nan],
                [np.nan, 9000., 9000., 9000., 9000., 9000., 9000., 9000., np.nan],
                ])},
           daskify=daskify,
           area=fake_area_large,
           common_attrs=_get_attrs(0, 0, 40_000))
    new_area = corrector(sc["CTH"])
    assert new_area.shape == fake_area_small.shape
    (new_lons, new_lats) = new_area.get_lonlats()
    assert fake_area_lons[3, 4] != new_lons[3, 4]

    np.testing.assert_allclose(
        new_lons,
        np.array([
            [-0.19999939, -0.09999966, 0.0, 0.1, 0.2],
            [-0.19999947, -0.09999973, 0.0, 0.1, 0.2],
            [-0.19999977, -0.1, 0.0, 0.1, 0.2],
            [-0.19999962, -0.0999997, 0.0, 0.0999997, 0.19999955],
            [-0.19999932, -0.09999966, 0.0, 0.09999966, 0.19999932]]),
        rtol=1e-5)
    np.testing.assert_allclose(
        new_lats,
        np.array([
            [50.19991371, 50.19990292, 50.2, 50.2, 50.2],
            [50.09992476, 50.09992476, 50.1, 50.1, 50.1],
            [49.99996787, 50.0, 50.0, 50.0, 50.0],
            [49.89994664, 49.89991462, 49.89991462, 49.89991462, 49.89993597],
            [49.79990429, 49.79990429, 49.79990429, 49.79990429, 49.79990429]]),
        rtol=1e-6)


@pytest.mark.parametrize("res1,res2", [(0.01, 1), (1, 0.01)])
def test_correct_area_clearsky_different_resolutions(res1, res2):
    """Test clearsky correction when areas have different resolutions."""
    from ...modifiers.parallax import ParallaxCorrection
    from ..utils import make_fake_scene
    areas_res1 = _get_fake_areas((0, 0), [5, 9], res1)
    areas_res2 = _get_fake_areas((0, 0), [5, 9], res2)
    fake_area_small = areas_res1[0]
    fake_area_large = areas_res2[1]

    with pytest.warns(None) as record:
        sc = make_fake_scene(
                {"CTH_clear": np.full((9, 9), np.nan)},
                daskify=False,
                area=fake_area_large,
                common_attrs=_get_attrs(0, 0, 35_000))
    assert len(record) == 0

    corrector = ParallaxCorrection(fake_area_small)
    new_area = corrector(sc["CTH_clear"])
    np.testing.assert_allclose(
            new_area.get_lonlats(),
            fake_area_small.get_lonlats())


@pytest.mark.xfail(reason="awaiting pyresample fixes")
def test_correct_area_cloudy_no_overlap():
    """Test cloudy correction when areas have no overlap."""
    from ...modifiers.parallax import MissingHeightError, ParallaxCorrection
    from ..utils import make_fake_scene
    areas_00 = _get_fake_areas((0, 40), [5, 9], 0.1)
    areas_shift = _get_fake_areas((90, 20), [5, 9], 0.1)
    fake_area_small = areas_00[0]
    fake_area_large = areas_shift[1]

    sc = make_fake_scene(
            {"CTH_constant": np.full((9, 9), 10000)},
            daskify=False,
            area=fake_area_large,
            common_attrs=_get_attrs(0, 0, 35_000))

    corrector = ParallaxCorrection(fake_area_small)
    with pytest.raises(MissingHeightError):
        corrector(sc["CTH_constant"])


@pytest.mark.xfail(reason="awaiting pyresample fixes")
def test_correct_area_cloudy_partly_shifted():
    """Test cloudy correction when areas overlap only partly."""
    from ...modifiers.parallax import IncompleteHeightWarning, ParallaxCorrection
    from ..utils import make_fake_scene
    areas_00 = _get_fake_areas((0, 40), [5, 9], 0.1)
    areas_shift = _get_fake_areas((0.5, 40), [5, 9], 0.1)
    fake_area_small = areas_00[0]
    fake_area_large = areas_shift[1]

    sc = make_fake_scene(
            {"CTH_constant": np.full((9, 9), 10000)},
            daskify=False,
            area=fake_area_large,
            common_attrs=_get_attrs(0, 0, 35_000))

    corrector = ParallaxCorrection(fake_area_small)

    with pytest.warns(IncompleteHeightWarning):
        new_area = corrector(sc["CTH_constant"])
    assert new_area.shape == fake_area_small.shape


def test_correct_area_cloudy_same_area():
    """Test cloudy correction when areas are the same."""
    from ...modifiers.parallax import ParallaxCorrection
    from ..utils import make_fake_scene
    area = _get_fake_areas((0, 0), [9], 0.1)[0]

    sc = make_fake_scene(
            {"CTH_constant": np.full((9, 9), 10000)},
            daskify=False,
            area=area,
            common_attrs=_get_attrs(0, 0, 35_000))

    corrector = ParallaxCorrection(area)
    corrector(sc["CTH_constant"])


def test_correct_area_no_orbital_parameters(caplog):
    """Test ParallaxCorrection when CTH has no orbital parameters.

    Some CTH products, such as NWCSAF-GEO, do not include information
    on satellite location directly.  Rather, they include platform name,
    sensor, start time, and end time, that we have to use instead.
    """
    from ...modifiers.parallax import ParallaxCorrection
    from ..utils import make_fake_scene
    small = 5
    large = 9
    (fake_area_small, fake_area_large) = _get_fake_areas(
            (0, 0), [small, large], 0.05)
    corrector = ParallaxCorrection(fake_area_small)

    sc = make_fake_scene(
            {"CTH_clear": np.full((large, large), np.nan)},
            daskify=False,
            area=fake_area_large,
            common_attrs={
                "platform_name": "Meteosat-42",
                "sensor": "irives",
                "start_time": datetime.datetime(3021, 11, 30, 12, 24, 17),
                "end_time": datetime.datetime(3021, 11, 30, 12, 27, 22)})
    with unittest.mock.patch("pyorbital.tlefile.read") as plr:
        plr.return_value = pyorbital.tlefile.Tle(
                "Meteosat-42",
                line1="1 40732U 15034A   22011.84285506  .00000004  00000+0  00000+0 0  9995",
                line2="2 40732   0.2533 325.0106 0000976 118.8734 330.4058  1.00272123 23817")
        with caplog.at_level(logging.WARNING):
            new_area = corrector(sc["CTH_clear"])
    assert "Orbital parameters missing from metadata." in caplog.text
    np.testing.assert_allclose(
            new_area.get_lonlats(),
            fake_area_small.get_lonlats())


@pytest.mark.parametrize("cloud", [(9, 2, 8, 5, 3, 6, 8)], indirect=["cloud"])
def test_cloud(cloud):
    """Test using cloud."""
    pass


def test_parallax_modifier_interface():
    """Test the modifier interface."""
    from ...modifiers.parallax import ParallaxCorrectionModifier
    (area_small, area_large) = _get_fake_areas((0, 0), [5, 9], 0.1)
    fake_bt = xr.DataArray(
            np.linspace(220, 230, 25).reshape(5, 5),
            dims=("y", "x"),
            attrs={"area": area_small, **_get_attrs(0, 0, 35_000)})
    cth_clear = xr.DataArray(
            np.full((9, 9), np.nan),
            dims=("y", "x"),
            attrs={"area": area_large, **_get_attrs(0, 0, 35_000)})
    modif = ParallaxCorrectionModifier(
            name="parallax_corrected_dataset",
            prerequisites=[fake_bt, cth_clear],
            optional_prerequisites=[],
            search_radius=25_000)
    res = modif([fake_bt, cth_clear], optional_datasets=[])
    np.testing.assert_allclose(res, fake_bt)


def test_parallax_modifier_interface_with_cloud():
    """Test the modifier interface with a cloud.

    Test corresponds to a real bug encountered when using CTH data
    from NWCSAF-GEO, which created strange speckles in Africa (see
    https://github.com/pytroll/satpy/pull/1904#issuecomment-1011161623
    for an example).  Create fake CTH corresponding to NWCSAF-GEO area and
    BT corresponding to full disk SEVIRI, and test that no strange speckles
    occur.
    """
    from ...modifiers.parallax import ParallaxCorrectionModifier

    w_cth = 25
    h_cth = 15
    proj_dict = {'a': '6378137', 'h': '35785863', 'proj': 'geos', 'units': 'm'}
    fake_area_cth = pyresample.create_area_def(
            area_id="test-area",
            projection=proj_dict,
            area_extent=(-2296808.75, 2785874.75, 2293808.25, 5570249.0),
            shape=(h_cth, w_cth))

    sz_bt = 20
    fake_area_bt = pyresample.create_area_def(
            "test-area-2",
            projection=proj_dict,
            area_extent=(-5567248.0742, -5513240.8172, 5513240.8172, 5567248.0742),
            shape=(sz_bt, sz_bt))

    (lons_cth, lats_cth) = fake_area_cth.get_lonlats()
    fake_cth_data = np.where(
            np.isfinite(lons_cth) & np.isfinite(lats_cth),
            15000,
            np.nan)

    (lons_bt, lats_bt) = fake_area_bt.get_lonlats()
    fake_bt_data = np.where(
            np.isfinite(lons_bt) & np.isfinite(lats_bt),
            np.linspace(200, 300, lons_bt.size).reshape(lons_bt.shape),
            np.nan)

    attrs = _get_attrs(0, 0)

    fake_bt = xr.DataArray(
            fake_bt_data,
            dims=("y", "x"),
            attrs={**attrs, "area": fake_area_bt})
    fake_cth = xr.DataArray(
            fake_cth_data,
            dims=("y", "x"),
            attrs={**attrs, "area": fake_area_cth})

    modif = ParallaxCorrectionModifier(
            name="parallax_corrected_dataset",
            prerequisites=[fake_bt, fake_cth],
            optional_prerequisites=[],
            search_radius=25_000)

    res = modif([fake_bt, fake_cth], optional_datasets=[])

    # with a constant cloud, a monotonically increasing BT should still
    # do so after parallax correction
    assert not (res.diff("x") < 0).any()


@pytest.mark.parametrize("cth", [7500, 15000])
def test_modifier_interface_cloud_moves_to_observer(cth):
    """Test that a cloud moves to the observer.

    With the modifier interface, use a high resolution area and test that
    pixels are moved in the direction of the observer and not away from it.
    """
    from ...modifiers.parallax import ParallaxCorrectionModifier

    # make a fake area rather far north with a rather high resolution,
    # with a cloud in the middle and somewhat consistent brightness
    # temperatures

    area_føroyar = pyresample.create_area_def(
            "føroyar", 4087,
            area_extent=[-861785.8867075047, 6820719.391005835,
                         -686309.8124887547, 6954386.383193335],
            resolution=500)

    w_cloud = 20
    h_cloud = 3

    # location of cloud in uncorrected data
    lat_min_i = 155
    lat_max_i = lat_min_i + h_cloud
    lon_min_i = 140
    lon_max_i = lon_min_i + w_cloud

    # location of cloud in corrected data
    dest_lat_min_i = 167
    dest_lat_max_i = 170
    dest_lon_min_i = 182
    dest_lon_max_i = 202

    fake_bt_data = np.linspace(
            270, 330, math.prod(area_føroyar.shape), dtype="f8").reshape(
                    area_føroyar.shape).round(2)
    fake_cth_data = np.full(area_føroyar.shape, np.nan, dtype="f8")
    fake_bt_data[lon_min_i:lon_max_i, lat_min_i:lat_max_i] = np.linspace(
            180, 220, w_cloud*h_cloud).reshape(w_cloud, h_cloud).round(2)
    fake_cth_data[lon_min_i:lon_max_i, lat_min_i:lat_max_i] = cth

    attrs = _get_attrs(0, 0)

    fake_bt = xr.DataArray(
            fake_bt_data,
            dims=("y", "x"),
            attrs={**attrs, "area": area_føroyar})

    fake_cth = xr.DataArray(
            fake_cth_data,
            dims=("y", "x"),
            attrs={**attrs, "area": area_føroyar})

    modif = ParallaxCorrectionModifier(
            name="parallax_corrected_dataset",
            prerequisites=[fake_bt, fake_cth],
            optional_prerequisites=[],
            debug_mode=True)

    res = modif([fake_bt, fake_cth], optional_datasets=[])

    assert fake_bt.attrs["area"] == area_føroyar  # should not be changed
    assert res.attrs["area"] == fake_bt.attrs["area"]
    # confirm old cloud area now fill value
    # that only happens for small fill values...
    assert np.isnan(res[lon_min_i:lon_max_i, lat_min_i:lat_max_i]).all()
    # confirm rest of the area does not have fill values
    idx = np.ones_like(fake_bt.data, dtype="?")
    idx[lon_min_i:lon_max_i, lat_min_i:lat_max_i] = False
    assert np.isfinite(res.data[idx]).all()
    # confirm that rest of area pixel values did not change, except where
    # cloud arrived or originated
    delta = res - fake_bt
    delta[lon_min_i:lon_max_i, lat_min_i:lat_max_i] = 0
    delta[dest_lon_min_i:dest_lon_max_i, dest_lat_min_i:dest_lat_max_i] = 0
    assert (delta == 0).all()
    # verify that cloud moved south
    assert (res.attrs["area"].get_lonlats()[1][dest_lon_min_i:dest_lon_max_i,
                                               dest_lat_min_i:dest_lat_max_i] <
            fake_bt.attrs["area"].get_lonlats()[1][lon_min_i:lon_max_i,
                                                   lat_min_i:lat_max_i]).all()
    # verify that all pixels at the new cloud location are indeed cloudy
    assert (res[dest_lon_min_i:dest_lon_max_i, dest_lat_min_i:dest_lat_max_i]
            < 250).all()
