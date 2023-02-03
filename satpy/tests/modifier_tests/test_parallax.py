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
import os
import unittest.mock

import dask.array as da
import dask.config
import numpy as np
import pyorbital.tlefile
import pyresample.kd_tree
import pytest
import xarray as xr
from pyproj import Geod
from pyresample import create_area_def

import satpy.resample

from ...writers import get_enhanced_image

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path
# - caplog
# - request


@pytest.fixture
def fake_tle():
    """Produce fake Two Line Element (TLE) object from pyorbital."""
    return pyorbital.tlefile.Tle(
        "Meteosat-42",
        line1="1 40732U 15034A   22011.84285506  .00000004  00000+0  00000+0 0 9995",
        line2="2 40732   0.2533 325.0106 0000976 118.8734 330.4058  1.00272123 23817")


def _get_fake_areas(center, sizes, resolution, code=4326):
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
        code,
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


class TestForwardParallax:
    """Test the forward parallax function with various inputs."""

    def test_get_parallax_corrected_lonlats_ssp(self):
        """Test that at SSP, parallax correction does nothing."""
        from ...modifiers.parallax import get_parallax_corrected_lonlats
        sat_lat = sat_lon = lon = lat = 0.
        height = 5000.  # m
        sat_alt = 30_000_000.  # m
        corr_lon, corr_lat = get_parallax_corrected_lonlats(
            sat_lon, sat_lat, sat_alt, lon, lat, height)
        assert corr_lon == corr_lat == 0

    def test_get_parallax_corrected_lonlats_clearsky(self):
        """Test parallax correction for clearsky case (returns NaN)."""
        from ...modifiers.parallax import get_parallax_corrected_lonlats
        sat_lat = sat_lon = 0
        lat = np.linspace(-20, 20, 25).reshape(5, 5)
        lon = np.linspace(-20, 20, 25).reshape(5, 5).T
        height = np.full((5, 5), np.nan)  # no CTH --> clearsky
        sat_alt = 35_000_000.  # m above surface
        (corr_lon, corr_lat) = get_parallax_corrected_lonlats(
            sat_lon, sat_lat, sat_alt, lon, lat, height)
        # clearsky becomes NaN
        assert np.isnan(corr_lon).all()
        assert np.isnan(corr_lat).all()

    @pytest.mark.parametrize("lat,lon", [(0, 0), (0, 40), (0, 179.9)])
    @pytest.mark.parametrize("resolution", [0.01, 0.5, 10])
    def test_get_parallax_corrected_lonlats_cloudy_ssp(self, lat, lon, resolution):
        """Test parallax correction for fully cloudy scene at SSP."""
        from ...modifiers.parallax import get_parallax_corrected_lonlats

        N = 5
        lats = np.linspace(lat-N*resolution, lat+N*resolution, 25).reshape(N, N)
        lons = np.linspace(lon-N*resolution, lon+N*resolution, 25).reshape(N, N).T
        height = np.full((N, N), 10_000)  # constant high clouds at 10 km
        sat_alt = 35_000_000.  # satellite at 35 Mm
        (corr_lon, corr_lat) = get_parallax_corrected_lonlats(
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

    def test_get_parallax_corrected_lonlats_cloudy_slant(self):
        """Test parallax correction for fully cloudy scene (not SSP)."""
        from ...modifiers.parallax import get_parallax_corrected_lonlats
        sat_lat = sat_lon = 0
        lat = np.linspace(-20, 20, 25).reshape(5, 5)
        lon = np.linspace(-20, 20, 25).reshape(5, 5).T
        height = np.full((5, 5), 10_000)  # constant high clouds at 10 km
        sat_alt = 35_000_000.  # satellite at 35 Mm
        (corr_lon, corr_lat) = get_parallax_corrected_lonlats(
            sat_lon, sat_lat, sat_alt, lon, lat, height)
        # reference value from Simon Proud
        np.testing.assert_allclose(
            corr_lat[4, 4], 19.955, rtol=5e-4)
        np.testing.assert_allclose(
            corr_lon[4, 4], 19.960, rtol=5e-4)

    def test_get_parallax_corrected_lonlats_mixed(self):
        """Test parallax correction for mixed cloudy case."""
        from ...modifiers.parallax import get_parallax_corrected_lonlats

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
        (corrected_lon, corrected_lat) = get_parallax_corrected_lonlats(
            sat_lon, sat_lat, sat_alt, lon, lat, alt)
        assert corrected_lon.shape == lon.shape
        assert corrected_lat.shape == lat.shape
        # lon/lat should be nan for clear-sky pixels
        assert np.isnan(corrected_lon[np.isnan(alt)]).all()
        assert np.isnan(corrected_lat[np.isnan(alt)]).all()
        # otherwise no nans
        assert np.isfinite(corrected_lon[~np.isnan(alt)]).all()
        assert np.isfinite(corrected_lat[~np.isnan(alt)]).all()

    def test_get_parallax_corrected_lonlats_horizon(self):
        """Test that exception is raised if satellites exactly at the horizon.

        Test the rather unlikely case of a satellite elevation of exactly 0
        """
        from ...modifiers.parallax import get_parallax_corrected_lonlats
        sat_lat = sat_lon = lon = lat = 0.
        height = 5000.
        sat_alt = 30_000_000.
        with unittest.mock.patch("satpy.modifiers.parallax.get_observer_look") as smpg:
            smpg.return_value = (0, 0)
            with pytest.raises(NotImplementedError):
                get_parallax_corrected_lonlats(sat_lon, sat_lat, sat_alt, lon, lat, height)

    def test_get_surface_parallax_displacement(self):
        """Test surface parallax displacement."""
        from ...modifiers.parallax import get_surface_parallax_displacement

        val = get_surface_parallax_displacement(
                0, 0, 36_000_000, 0, 10, 10_000)
        np.testing.assert_allclose(val, 2141.2404451757875)


class TestParallaxCorrectionClass:
    """Test that the ParallaxCorrection class is behaving sensibly."""

    @pytest.mark.parametrize("center", [(0, 0), (80, -10), (-180, 5)])
    @pytest.mark.parametrize("sizes", [[5, 9]])
    @pytest.mark.parametrize("resolution", [0.05, 1, 10])
    def test_init_parallaxcorrection(self, center, sizes, resolution):
        """Test that ParallaxCorrection class can be instantiated."""
        from ...modifiers.parallax import ParallaxCorrection
        fake_area = _get_fake_areas(center, sizes, resolution)[0]
        pc = ParallaxCorrection(fake_area)
        assert pc.base_area == fake_area

    @pytest.mark.parametrize("sat_pos,ar_pos",
                             [((0, 0), (0, 0)), ((0, 0), (40, 0))])
    @pytest.mark.parametrize("resolution", [0.01, 0.5, 10])
    def test_correct_area_clearsky(self, sat_pos, ar_pos, resolution, caplog):
        """Test that ParallaxCorrection doesn't change clearsky geolocation."""
        from ...modifiers.parallax import ParallaxCorrection
        from ..utils import make_fake_scene
        (sat_lat, sat_lon) = sat_pos
        (ar_lat, ar_lon) = ar_pos
        small = 5
        large = 9
        (fake_area_small, fake_area_large) = _get_fake_areas(
                (ar_lon, ar_lat), [small, large], resolution)
        corrector = ParallaxCorrection(fake_area_small)

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

    @pytest.mark.parametrize("lat,lon",
                             [(0, 0), (0, 40), (0, 180),
                              (90, 0)])  # relevant for Арктика satellites
    @pytest.mark.parametrize("resolution", [0.01, 0.5, 10])
    def test_correct_area_ssp(self, lat, lon, resolution):
        """Test that ParallaxCorrection doesn't touch SSP."""
        from ...modifiers.parallax import ParallaxCorrection
        from ..utils import make_fake_scene
        codes = {
                (0, 0): 4326,
                (0, 40): 4326,
                (0, 180): 3575,
                (90, 0): 3575}
        small = 5
        large = 9
        (fake_area_small, fake_area_large) = _get_fake_areas(
                (lon, lat), [small, large], resolution,
                code=codes[(lat, lon)])
        corrector = ParallaxCorrection(fake_area_small)

        sc = make_fake_scene(
                {"CTH_constant": np.full((large, large), 10000)},
                daskify=False,
                area=fake_area_large,
                common_attrs=_get_attrs(lat, lon, 35_000))
        new_area = corrector(sc["CTH_constant"])
        assert new_area.shape == fake_area_small.shape
        old_lonlats = fake_area_small.get_lonlats()
        new_lonlats = new_area.get_lonlats()
        if lat != 90:  # don't check SSP longitude if lat=90
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
    def test_correct_area_partlycloudy(self, daskify):
        """Test ParallaxCorrection for partly cloudy situation."""
        from ...modifiers.parallax import ParallaxCorrection
        from ..utils import make_fake_scene
        small = 5
        large = 9
        (fake_area_small, fake_area_large) = _get_fake_areas(
                (0, 50), [small, large], 0.1)
        (fake_area_lons, fake_area_lats) = fake_area_small.get_lonlats()
        corrector = ParallaxCorrection(fake_area_small)

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
                [np.nan, np.nan, 0.0, 0.1, 0.2],
                [-0.20078652, -0.10044222, 0.0, 0.1, 0.2],
                [-0.20068529, -0.10034264, 0.0, 0.1, 0.2],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [-0.20048537, -0.10038778, 0., 0.10038778, 0.20058219]]),
            rtol=1e-5)
        np.testing.assert_allclose(
            new_lats,
            np.array([
                [np.nan, np.nan, 50.2, 50.2, 50.2],
                [50.2110675, 50.22493181, 50.1, 50.1, 50.1],
                [50.09680357, 50.09680346, 50.0, 50.0, 50.0],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [49.86860622, 49.9097198, 49.90971976, 49.9097198, 49.88231496]]),
            rtol=1e-6)

    @pytest.mark.parametrize("res1,res2", [(0.08, 0.3), (0.3, 0.08)])
    def test_correct_area_clearsky_different_resolutions(self, res1, res2):
        """Test clearsky correction when areas have different resolutions."""
        from ...modifiers.parallax import ParallaxCorrection
        from ..utils import make_fake_scene

        # areas with different resolutions, but same coverage

        area1 = create_area_def(
            "fribullus_xax",
            4326,
            units="degrees",
            resolution=res1,
            area_extent=[-1, -1, 1, 1])

        area2 = create_area_def(
            "fribullus_xax",
            4326,
            units="degrees",
            resolution=res2,
            area_extent=[-1, -1, 1, 1])

        with pytest.warns(None) as record:
            sc = make_fake_scene(
                    {"CTH_clear": np.full(area1.shape, np.nan)},
                    daskify=False,
                    area=area1,
                    common_attrs=_get_attrs(0, 0, 35_000))
        assert len(record) == 0

        corrector = ParallaxCorrection(area2)
        new_area = corrector(sc["CTH_clear"])
        np.testing.assert_allclose(
                new_area.get_lonlats(),
                area2.get_lonlats())

    @pytest.mark.xfail(reason="awaiting pyresample fixes")
    def test_correct_area_cloudy_no_overlap(self, ):
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
    def test_correct_area_cloudy_partly_shifted(self, ):
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

    def test_correct_area_cloudy_same_area(self, ):
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

    def test_correct_area_no_orbital_parameters(self, caplog, fake_tle):
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
            plr.return_value = fake_tle
            with caplog.at_level(logging.WARNING):
                new_area = corrector(sc["CTH_clear"])
        assert "Orbital parameters missing from metadata." in caplog.text
        np.testing.assert_allclose(
                new_area.get_lonlats(),
                fake_area_small.get_lonlats())


class TestParallaxCorrectionModifier:
    """Test that the parallax correction modifier works correctly."""

    def test_parallax_modifier_interface(self):
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
                cth_radius_of_influence=48_000,
                dataset_radius_of_influence=49_000)
        res = modif([fake_bt, cth_clear], optional_datasets=[])
        np.testing.assert_allclose(res, fake_bt)
        with unittest.mock.patch("satpy.modifiers.parallax.resample_dataset") as smp:
            smp.side_effect = satpy.resample.resample_dataset
            modif([fake_bt, cth_clear], optional_datasets=[])
            assert smp.call_args_list[0].kwargs["radius_of_influence"] == 48_000
            assert smp.call_args_list[1].kwargs["radius_of_influence"] == 49_000

    def test_parallax_modifier_interface_with_cloud(self):
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

    @pytest.fixture
    def test_area(self, request):
        """Produce test area for parallax correction unit tests.

        Produce test area for the modifier-interface parallax correction unit
        tests.
        """
        extents = {
            "foroyar": [-861785.8867075047, 6820719.391005835, -686309.8124887547, 6954386.383193335],
            "ouagadougou": [-232482.90622750926, 1328206.360136668,
                            -114074.70310250926, 1422810.852324168],
            }
        where = request.param
        return pyresample.create_area_def(where, 4087, area_extent=extents[where], resolution=500)

    def _get_fake_cloud_datasets(self, test_area, cth, use_dask):
        """Return datasets for BT and CTH for fake cloud."""
        w_cloud = 20
        h_cloud = 5

        # location of cloud in uncorrected data
        lat_min_i = 155
        lat_max_i = lat_min_i + h_cloud
        lon_min_i = 140
        lon_max_i = lon_min_i + w_cloud

        fake_bt_data = np.linspace(
                270, 330, math.prod(test_area.shape), dtype="f8").reshape(
                        test_area.shape).round(2)
        fake_cth_data = np.full(test_area.shape, np.nan, dtype="f8")
        fake_bt_data[lat_min_i:lat_max_i, lon_min_i:lon_max_i] = np.linspace(
                180, 220, w_cloud*h_cloud).reshape(h_cloud, w_cloud).round(2)
        fake_cth_data[lat_min_i:lat_max_i, lon_min_i:lon_max_i] = cth

        if use_dask:
            fake_bt_data = da.array(fake_bt_data)
            fake_cth_data = da.array(fake_cth_data)

        attrs = _get_attrs(0, 0)

        fake_bt = xr.DataArray(
                fake_bt_data,
                dims=("y", "x"),
                attrs={**attrs, "area": test_area})

        fake_cth = xr.DataArray(
                fake_cth_data,
                dims=("y", "x"),
                attrs={**attrs, "area": test_area})

        cma = np.zeros(shape=fake_bt.shape, dtype="?")
        cma[lat_min_i:lat_max_i, lon_min_i:lon_max_i] = True

        return (fake_bt, fake_cth, cma)

    @pytest.mark.parametrize("test_area", ["foroyar", "ouagadougou"], indirect=["test_area"])
    def test_modifier_interface_fog_no_shift(self, test_area):
        """Test that fog isn't masked or shifted."""
        from ...modifiers.parallax import ParallaxCorrectionModifier

        (fake_bt, fake_cth, _) = self._get_fake_cloud_datasets(test_area, 50, use_dask=False)

        modif = ParallaxCorrectionModifier(
                name="parallax_corrected_dataset",
                prerequisites=[fake_bt, fake_cth],
                optional_prerequisites=[],
                debug_mode=True)

        res = modif([fake_bt, fake_cth], optional_datasets=[])

        assert np.isfinite(res).all()
        np.testing.assert_allclose(res, fake_bt)

    @pytest.mark.parametrize("cth", [7500, 15000])
    @pytest.mark.parametrize("use_dask", [True, False])
    @pytest.mark.parametrize("test_area", ["foroyar", "ouagadougou"], indirect=["test_area"])
    def test_modifier_interface_cloud_moves_to_observer(self, cth, use_dask, test_area):
        """Test that a cloud moves to the observer.

        With the modifier interface, use a high resolution area and test that
        pixels are moved in the direction of the observer and not away from it.
        """
        from ...modifiers.parallax import ParallaxCorrectionModifier

        (fake_bt, fake_cth, cma) = self._get_fake_cloud_datasets(test_area, cth, use_dask=use_dask)

        # location of cloud in corrected data
        # this may no longer be rectangular!
        dest_mask = np.zeros(shape=test_area.shape, dtype="?")
        cloud_location = {
                "foroyar": {
                    7500: (197, 202, 152, 172),
                    15000: (239, 244, 165, 184)},
                "ouagadougou": {
                    7500: (159, 164, 140, 160),
                    15000: (163, 168, 141, 161)}}
        (x_lo, x_hi, y_lo, y_hi) = cloud_location[test_area.name][cth]
        dest_mask[x_lo:x_hi, y_lo:y_hi] = True

        modif = ParallaxCorrectionModifier(
                name="parallax_corrected_dataset",
                prerequisites=[fake_bt, fake_cth],
                optional_prerequisites=[],
                debug_mode=True)

        res = modif([fake_bt, fake_cth], optional_datasets=[])

        assert fake_bt.attrs["area"] == test_area  # should not be changed
        assert res.attrs["area"] == fake_bt.attrs["area"]
        # confirm old cloud area now fill value
        # except where it overlaps with new cloud
        assert np.isnan(res.data[cma & (~dest_mask)]).all()
        # confirm rest of the area does not have fill values
        assert np.isfinite(res.data[~cma]).all()
        # confirm that rest of area pixel values did not change, except where
        # cloud arrived or originated
        delta = res - fake_bt
        assert (delta.data[~(cma | dest_mask)] == 0).all()
        # verify that cloud moved south.  Pointwise comparison might not work because
        # cloud may shrink.
        assert ((res.attrs["area"].get_lonlats()[1][dest_mask]).mean() <
                fake_bt.attrs["area"].get_lonlats()[1][cma].mean())
        # verify that all pixels at the new cloud location are indeed cloudy
        assert (res.data[dest_mask] < 250).all()


_test_yaml_code = """
sensor_name: visir

modifiers:
  parallax_corrected:
    modifier: !!python/name:satpy.modifiers.parallax.ParallaxCorrectionModifier
    prerequisites:
      - name: "ctth_alti"

composites:
  parallax_corrected_VIS006:
    compositor: !!python/name:satpy.composites.SingleBandCompositor
    prerequisites:
      - name: VIS006
        modifiers: [parallax_corrected]
"""


class TestParallaxCorrectionSceneLoad:
    """Test that scene load interface works as expected."""

    @pytest.fixture
    def yaml_code(self):
        """Return YAML code for parallax_corrected_VIS006."""
        return _test_yaml_code

    @pytest.fixture
    def conf_file(self, yaml_code, tmp_path):
        """Produce a fake configuration file."""
        conf_file = tmp_path / "test.yaml"
        with conf_file.open(mode="wt", encoding="ascii") as fp:
            fp.write(yaml_code)
        return conf_file

    @pytest.fixture
    def fake_scene(self, yaml_code):
        """Produce fake scene and prepare fake composite config."""
        from satpy import Scene
        from satpy.dataset.dataid import WavelengthRange
        from satpy.tests.utils import make_dataid

        area = _get_fake_areas((0, 0), [5], 1)[0]
        sc = Scene()
        sc["VIS006"] = xr.DataArray(
            np.linspace(0, 99, 25).reshape(5, 5),
            dims=("y", "x"),
            attrs={
                "_satpy_id": make_dataid(
                    name="VIS006",
                    wavelength=WavelengthRange(min=0.56, central=0.635, max=0.71, unit="µm"),
                    resolution=3000,
                    calibration="reflectance",
                    modifiers=()),
                "modifiers": (),
                "sensor": "seviri",
                "area": area})
        sc["ctth_alti"] = xr.DataArray(
            np.linspace(0, 99, 25).reshape(5, 5),
            dims=("y", "x"),
            attrs={
                "_satpy_id": make_dataid(
                    name="ctth_alti",
                    resolution=3000,
                    modifiers=()),
                "modifiers": (),
                "sensor": {"seviri"},
                "platform_name": "Meteosat-11",
                "start_time": datetime.datetime(2022, 4, 12, 9, 0),
                "area": area})
        return sc

    def test_double_load(self, fake_scene, conf_file, fake_tle):
        """Test that loading corrected and uncorrected works correctly.

        When the modifier ``__call__`` method fails to call
        ``self.apply_modifier_info(new, old)`` and the original and
        parallax-corrected dataset are requested at the same time, the
        DataArrays differ but the underlying dask arrays have object identity,
        which in turn leads to both being parallax corrected.  This unit test
        confirms that there is no such object identity.
        """
        with unittest.mock.patch(
                "satpy.composites.config_loader.config_search_paths") as sccc, \
             unittest.mock.patch("pyorbital.tlefile.read") as plr:
            sccc.return_value = [os.fspath(conf_file)]
            plr.return_value = fake_tle
            fake_scene.load(["parallax_corrected_VIS006", "VIS006"])
            assert fake_scene["VIS006"] is not fake_scene["parallax_corrected_VIS006"]
        assert fake_scene["VIS006"].data is not fake_scene["parallax_corrected_VIS006"].data

    @pytest.mark.xfail(reason="awaiting pyresample fixes")
    def test_no_compute(self, fake_scene, conf_file):
        """Test that no computation occurs."""
        from satpy.tests.utils import CustomScheduler
        with unittest.mock.patch(
                "satpy.composites.config_loader.config_search_paths") as sccc, \
             dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            sccc.return_value = [os.fspath(conf_file)]
            fake_scene.load(["parallax_corrected_VIS006"])

    def test_enhanced_image(self, fake_scene, conf_file, fake_tle):
        """Test that image enhancement is the same."""
        with unittest.mock.patch(
                "satpy.composites.config_loader.config_search_paths") as sccc, \
             unittest.mock.patch("pyorbital.tlefile.read") as plr:
            sccc.return_value = [os.fspath(conf_file)]
            plr.return_value = fake_tle
            fake_scene.load(["parallax_corrected_VIS006", "VIS006"])
        im1 = get_enhanced_image(fake_scene["VIS006"])
        im2 = get_enhanced_image(fake_scene["parallax_corrected_VIS006"])
        assert im1.data.attrs["enhancement_history"] == im2.data.attrs["enhancement_history"]
