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
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Parallax correction.

Routines related to parallax correction using datasets involving height, such
as cloud top height.
"""

from datetime import datetime

import numpy as np
import xarray as xr
from pyorbital.orbital import A as EARTH_RADIUS
from pyorbital.orbital import get_observer_look
from pyresample.geometry import SwathDefinition
from scipy.signal import convolve

from satpy.utils import get_satpos, lonlat2xyz, xyz2lonlat

from . import projection


def parallax_correct(sat_lon, sat_lat, sat_alt, lon, lat, height):
    """Correct parallax.

    Using satellite position, correct the parallax for coordinates lon, lat,
    and height.

    Args:
        sat_lon: Satellite longitude
        sat_lat: Satellite latitude
        sat_alt: Satellite altitude
        lon: Dataset longitudes
        lat: Dataset latitudes
        height: Dataset altitudes

    Returns:
        (corrected_lon, corrected_lat)
    """
    X_sat = np.hstack(lonlat2xyz(sat_lon, sat_lat)) * sat_alt
    X = np.stack(lonlat2xyz(lon, lat), axis=-1) * EARTH_RADIUS
    # the datetime doesn't actually affect the result but is required
    # so we use a placeholder
    (_, elevation) = get_observer_look(
            sat_lon, sat_lat, sat_alt,
            datetime(2000, 1, 1), lon, lat, EARTH_RADIUS)
    # TODO: handle cases where this could divide by 0
    parallax_distance = height / np.sin(np.deg2rad(elevation))

    X_d = X - X_sat
    sat_distance = np.sqrt((X_d*X_d).sum(axis=-1))
    dist_shape = X_d.shape[:-1] + (1,)  # force correct array broadcasting
    X_top = X - X_d*(parallax_distance/sat_distance).reshape(dist_shape)

    (corrected_lon, corrected_lat) = xyz2lonlat(
        X_top[..., 0], X_top[..., 1], X_top[..., 2])
    return (corrected_lon, corrected_lat)


class ParallaxCorrection:
    """Class for parallax corrections.

    Initialise using a base area.
    """

    def __init__(self, base_area):
        """Initialise parallax correction class.

        See class docstring for documentation.
        """
        self.base_area = base_area
        self.source_area = None
        self.resampler = None  # we prepare this lazily

    def __call__(self, cth_dataset):
        """Apply parallax correction to dataset.

        Args:
            cth_dataset: Dataset containing cloud top heights (or other heights
                to be corrected).

        Returns:
            pyresample.geometry.SwathDefinition: Swathdefinition with correct
                lat/lon coordinates.
        """
        return self.corrected_area(cth_dataset)

    def corrected_area(self, cth_dataset):
        """Corrected area.

        Calculate the corrected SwathDefinition for dataset.
        """
        area = cth_dataset.area
        (sat_lon, sat_lat, sat_alt) = get_satpos(cth_dataset)
        cth_dataset = self._preprocess_cth(cth_dataset)
        (pixel_lon, pixel_lat) = area.get_lonlats()

        # Pixel coordinates according to parallax correction
        (corr_lon, corr_lat) = parallax_correct(
            sat_lon, sat_lat, sat_alt,
            np.array(pixel_lon), np.array(pixel_lat), np.array(cth_dataset)
        )

        corr_lon = xr.DataArray(corr_lon)
        corr_lat = xr.DataArray(corr_lat)
        corr_area = SwathDefinition(corr_lon, corr_lat)

        # But we are not actually moving pixels, rather we want a
        # coordinate transformation. With this transformation we approximately
        # invert the pixel coordinate transformation, giving the lon and lat
        # where we should retrieve a value for a given pixel.
        (proj_lon, proj_lat) = self._invert_lonlat(
                pixel_lon, pixel_lat, corr_area)
        proj_lon = xr.DataArray(proj_lon)
        proj_lat = xr.DataArray(proj_lat)

        return SwathDefinition(proj_lon, proj_lat)

    def correct_points(self, cth_dataset, lons, lats):
        """To be documented."""
        area = cth_dataset.area
        (sat_lon, sat_lat, sat_alt) = get_satpos(cth_dataset)
        cth_dataset = self._preprocess_cth(cth_dataset)
        grid_proj = projection.GridProjection(area)
        (y, x) = grid_proj(lons, lats)

        x0 = np.floor(x).astype(int)
        x1 = x0+1
        y0 = np.floor(y).astype(int)
        y1 = y0+1
        valid = (x0 >= 0) & (x1 < area.width) & \
            (y0 >= 0) & (y1 < area.height)
        x0 = xr.DataArray(x0[valid], dims=("y",))
        x1 = xr.DataArray(x1[valid], dims=("y",))
        y0 = xr.DataArray(y0[valid], dims=("y",))
        y1 = xr.DataArray(y1[valid], dims=("y",))

        h00 = np.array(cth_dataset[y0, x0])
        h01 = np.array(cth_dataset[y0, x1])
        h10 = np.array(cth_dataset[y1, x0])
        h11 = np.array(cth_dataset[y1, x1])
        dx = x[valid]-x0
        h0 = h00 + dx*(h01-h00)
        h1 = h10 + dx*(h11-h10)
        h = h0 + (y[valid]-y0)*(h1-h0)
        h = np.array(h)

        mask = ~valid
        corr_lon = np.ma.MaskedArray(lons, mask=mask)
        corr_lat = np.ma.MaskedArray(lats, mask=mask)
        (corr_lon[valid], corr_lat[valid]) = parallax_correct(
            sat_lon, sat_lat, sat_alt, lons[valid], lats[valid], h
        )

        return corr_lon, corr_lat

    def _preprocess_cth(self, cth_dataset):
        """To be documented."""
        units = cth_dataset.units
        cth = cth_dataset.copy().fillna(0.0)
        if units == 'm':  # convert to km
            cth = cth * 1e-3
        return cth

    def _invert_lonlat(self, pixel_lon, pixel_lat, source_area):
        (source_lon, source_lat) = source_area.get_lonlats()
        grid_projection = projection.GridProjection(self.base_area)
        (y, x) = grid_projection(source_lon, source_lat)
        x = x.round().astype(int)
        y = y.round().astype(int)

        valid = (x >= 0) & (x < self.base_area.width) & \
            (y >= 0) & (y < self.base_area.height)
        num_in_area = np.count_nonzero(valid)
        s = np.sqrt(np.prod(self.base_area.shape) / num_in_area) / 1.18
        r = int(round(s*4))
        (kx, ky) = np.mgrid[-r:r+1, -r:r+1]
        k = np.exp(-0.5*(kx**2+ky**2)/s**2)
        k /= k.sum()

        # reevaluate for the expanded area
        valid = ((x >= -r) & (x < self.base_area.width + r) &
                 (y >= -r) & (y < self.base_area.height + r))
        x = x[valid] + r
        y = y[valid]+r
        pixel_lon = pixel_lon[valid]
        pixel_lat = pixel_lat[valid]
        expanded_shape = (self.base_area.shape[0]+2*r, self.base_area.shape[1]+2*r)

        mask = np.zeros(expanded_shape)
        mask[y, x] = 1
        weight_sum = convolve(mask, k, mode='same')

        def inv_coord(pixel_coord):
            c = np.zeros(expanded_shape)
            c[y, x] = pixel_coord
            c = convolve(c, k, mode='same') / weight_sum
            return c[r:-r, r:-r].copy().astype(np.float32)

        inv_lon = inv_coord(pixel_lon)
        inv_lat = inv_coord(pixel_lat)

        return (inv_lon.astype(np.float32), inv_lat.astype(np.float32))
