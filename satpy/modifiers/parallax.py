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

The geolocation of (geostationary) satellite imagery is calculated
on the assumption of a clear view from the satellite to the surface.
When a cloud blocks the view of the surface, the geolocation is not
accurate for the cloud top.  This module contains routines to project
the geolocation onto the cloud top.

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


def forward_parallax(sat_lon, sat_lat, sat_alt, lon, lat, height):
    """Calculate forward parallax effect.

    Calculate the forward parallax effect.  When a satellite instrument
    observes the Earth, the geolocation assumes it sees the Earth surface at
    the geoid (elevation zero).  In reality, the ray may stop short of the
    geoid as it observes, for example, a cloud or elevated ground.  This
    function calculates the forward parallax effect.  If the view of a pixel at
    location (lat, lon) is blocked by a cloud at height h, we calculate the
    location of this blocking.

    Calculate parallax correction based on satellite position and
    (cloud top) height coordinates in geodetic (unprojected) coordinates.
    This function calculates the latitude and longitude belonging to the
    cloud top, based on the location of the satellite and the location
    of the cloud.

    For scenes that are only partly cloudy, the user might set the cloud top
    height for clear-sky pixels to NaN.  This function will return a corrected
    lat/lon as NaN as well.  The user can use the original lat/lon for those
    pixels or use the higher level :class:`ParallaxCorrection` class.

    This function assumes a spherical Earth.

    Args:
        sat_lon (number): Satellite longitude in geodetic coordinates [°]
        sat_lat (number): Satellite latitude in geodetic coordinates [°]
        sat_alt (number): Satellite altitude above the Earth surface [km]
        lon (array or number): Longitudes of pixel or pixels to be corrected,
            in geodetic coordinates [°]
        lat (array or number): Latitudes of pixel/pixels to be corrected, in
            geodetic coordinates [°]
        height (array or number): Heights of pixels on which the correction
            will be based.  Typically this is the cloud based height. [km]

    Returns:
        (corrected_lon, corrected_lat): New geolocation for the longitude and
            latitude that were to be corrected, in geodetic coordinates. [°]
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

    This class contains higher-level functionality to wrap the parallal
    correction calculations in :func:`forward_parallax`.  The class is
    initialised using a base area, which is the area for which a corrected
    geolocation will be calculated.  The resulting object is a callable.
    Calling the object with an array of (cloud top) heights returns a
    :class:`~pyresample.geometry.SwathDefinition` describing the new ,
    corrected geolocation.  This ``SwathDefinition`` can then be used for
    resampling a satpy Scene, yielding a corrected geolocation for all datasets
    in the Scene.  For example::

      >>> global_scene = satpy.Scene(reader="seviri_l1b_hrit", filenames=files_sat)
      >>> global_scene.load(['IR_087','IR_120'])
      >>> global_nwc = satpy.Scene(filenames=files_nwc)
      >>> global_nwc.load(['ctth'])
      >>> area_def = satpy.resample.get_area_def(area)
      >>> parallax_correction = ParallaxCorrection(area_def)
      >>> plax_corr_area = forward_parallax(global_nwc["ctth"])
      >>> local_scene = global_scene.resample(plax_corr_area)
      >>> local_nwc = global_nwc.resample(plax_corr_area)
      >>> local_scene[...].attrs["area"] = area_def
      >>> local_nwc[...].attrs["area"] = area_def

    Note that the ``ctth`` dataset must contain geolocation metadata, such as
    set in the ``orbital_parameters`` dataset attribute by many readers.
    """

    def __init__(self, base_area):
        """Initialise parallax correction class.

        Args:
            base_area (pyresample.AreaDefinition): Area for which calculated
                geolocation will be calculated.
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
            pyresample.geometry.SwathDefinition: Swathdefinition with corrected
                lat/lon coordinates.
        """
        return self.corrected_area(cth_dataset)

    def corrected_area(self, cth_dataset):
        """Corrected area.

        Calculate the corrected SwathDefinition for dataset.

        Returns a parallax corrected swathdefinition of the base area.
        """
        area = cth_dataset.area
        (sat_lon, sat_lat, sat_alt) = get_satpos(cth_dataset)
        cth_dataset = self._preprocess_cth(cth_dataset)
        (pixel_lon, pixel_lat) = area.get_lonlats()

        # Pixel coordinates according to parallax correction
        (corr_lon, corr_lat) = forward_parallax(
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

    def _preprocess_cth(self, cth_dataset):
        """To be documented."""
        units = cth_dataset.units
        cth = cth_dataset.copy().fillna(0.0)
        if units == 'm':  # convert to km
            cth = cth * 1e-3
        return cth

    def _invert_lonlat(self, pixel_lon, pixel_lat, source_area):
        """Invert the lon/lat coordinate transformation.

        When a satellite observes a cloud, a reprojection onto the cloud has
        already happened.
        """
        (source_lon, source_lat) = source_area.get_lonlats()
        grid_projection = projection.GridProjection(self.base_area)
        (y, x) = grid_projection(source_lon, source_lat)
        x = x.round().astype(int)
        y = y.round().astype(int)

        # set of points it starts with on an irregular "grid"
        # needs to go from irregular "grid" to regular grid
        # kernel density interpolation for regridding from irregular grid into
        # to regular one
        # radial basis interpolation
        # calculates estimate of needed size of convolution kernel for density
        # estimation
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
