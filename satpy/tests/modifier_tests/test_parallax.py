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
from pyresample.geometry import SwathDefinition


def test_parallax_correct_ssp():
    """Test that at SSP, parallax correction does nothing."""
    from ...modifiers.parallax import parallax_correct
    sat_lat = sat_lon = lon = lat = 0.
    height = 5000.
    sat_alt = 30_000_000.
    corr_lon, corr_lat = parallax_correct(
        sat_lon, sat_lat, sat_alt, lon, lat, height)
    assert corr_lon == corr_lat == 0


def test_parallax_correct_clearsky():
    """Test parallax correction for clearsky case (should do nothing)."""
    from ...modifiers.parallax import parallax_correct
    sat_lat = sat_lon = 0
    lat = np.linspace(-20, 20, 25).reshape(5, 5)
    lon = np.linspace(-20, 20, 25).reshape(5, 5).T
    height = np.full((5, 5), np.nan)  # no CTH --> clearsky
    sat_alt = 35_000_000.
    (corr_lon, corr_lat) = parallax_correct(
        sat_lon, sat_lat, sat_alt, lon, lat, height)
    np.testing.assert_array_equal(corr_lon, lon)
    np.testing.assert_array_equal(corr_lat, lat)


def test_parallax_correct_cloudy():
    """Test parallax correction for fully cloudy scene."""
    from ...modifiers.parallax import parallax_correct
    sat_lat = sat_lon = 0
    lat = np.linspace(-20, 20, 25).reshape(5, 5)
    lon = np.linspace(-20, 20, 25).reshape(5, 5).T
    height = np.full((5, 5), 10_000)  # constant high clouds
    sat_alt = 35_000_000.
    (corr_lon, corr_lat) = parallax_correct(
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
        corr_lat[4, 4], 19.9)  # FIXME replace reference
    np.testing.assert_allclose(
        corr_lon[4, 4], 19.9)  # FIXME replace reference


def test_parallax_correct_mixed():
    """Test parallax correction for mixed cloudy case."""
    from ...modifiers.parallax import parallax_correct

    sat_lon = sat_lat = 0
    sat_alt = 35_785_831.0
    lon = da.array([[-20, -10, 0, 10, 20]]*5)
    lat = da.array([[-20, -10, 0, 10, 20]]*5).T
    alt = da.array([
        [np.nan, np.nan, 5000, 6000, np.nan],
        [np.nan, 6000, 7000, 7000, 7000],
        [np.nan, 7000, 8000, 9000, np.nan],
        [np.nan, 7000, 7000, 7000, np.nan],
        [np.nan, 4000, 3000, np.nan, np.nan]])
    (corrected_lon, corrected_lat) = parallax_correct(
        sat_lon, sat_lat, sat_alt, lon, lat, alt)
    assert corrected_lon.shape == lon.shape
    assert corrected_lat.shape == lat.shape
    # lon/lat should not change for clear-sky pixels
    np.testing.assert_array_equal(lon[np.isnan(alt)], corrected_lon[np.isnan(alt)])
    np.testing.assert_array_equal(lat[np.isnan(alt)], corrected_lat[np.isnan(alt)])


def test_parallax_correction_other():
    """Test parallax correction.

    Test non-existing implementation.
    """
    from ...modifiers.geometry import parallax_correction
    from ..utils import make_fake_scene

    sc = make_fake_scene(
            {"CTH": np.array([[np.nan, np.nan, 5000, 6000, np.nan],
                              [np.nan, 6000, 7000, 7000, 7000],
                              [np.nan, 7000, 8000, 9000, np.nan],
                              [np.nan, 7000, 7000, 7000, np.nan],
                              [np.nan, 4000, 3000, np.nan, np.nan]]),
             "IR108": np.array([[290, 290, 240, 230, 290],
                                [290, 230, 220, 220, 220],
                                [290, 220, 210, 200, 290],
                                [290, 220, 220, 220, 290],
                                [290, 250, 260, 290, 290]])},
            daskify=False,
            area=True)

    new_sc = parallax_correction(sc)
    assert new_sc.keys() == sc.keys()
    assert isinstance(new_sc["CTH"].attrs["area"], SwathDefinition)
