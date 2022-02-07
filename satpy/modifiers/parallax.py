# Copyright (c) 2021-2022 Satpy developers
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

The geolocation of (geostationary) satellite imagery is calculated by
agencies or in satpy readers with the assumption of a clear view from
the satellite to the geoid.  When a cloud blocks the view of the Earth
surface on the surface is above sea level, the geolocation is not accurate
for the cloud or mountain top.  This module contains routines to correct
imagery such that pixels are shifted or interpolated to correct for this
parallax effect.

See also the :doc:`../modifiers` page in the documentation for an introduction to
parallax correction as a modifier in Satpy.
"""

import datetime
import inspect
import logging
import warnings

import dask.array as da
import numpy as np
import xarray as xr
from pyorbital.orbital import A as EARTH_RADIUS
from pyorbital.orbital import get_observer_look
from pyresample.bucket import BucketResampler
from pyresample.geometry import SwathDefinition

from satpy.modifiers import ModifierBase
from satpy.resample import resample_dataset
from satpy.utils import get_satpos, lonlat2xyz, xyz2lonlat

logger = logging.getLogger(__name__)


class MissingHeightError(ValueError):
    """Raised when heights do not overlap with area to be corrected."""


class IncompleteHeightWarning(UserWarning):
    """Raised when heights only partially overlap with area to be corrected."""


def forward_parallax(sat_lon, sat_lat, sat_alt, lon, lat, height):
    """Calculate forward parallax effect.

    Calculate the forward parallax effect.  When a satellite instrument
    observes the Earth, the geolocation assumes it sees the Earth surface
    at the geoid (elevation zero).  In reality, the field of view may
    stop short of the geoid as it observes, for example, a cloud or
    elevated ground.  This function calculates the forward parallax
    effect.  If the view of a pixel at location (lat, lon) is blocked
    by a cloud at height h, we calculate the location of this blocking.

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
        sat_alt (number): Satellite altitude above the Earth surface [m]
        lon (array or number): Longitudes of pixel or pixels to be corrected,
            in geodetic coordinates [°]
        lat (array or number): Latitudes of pixel/pixels to be corrected, in
            geodetic coordinates [°]
        height (array or number): Heights of pixels on which the correction
            will be based.  Typically this is the cloud top height. [m]

    Returns:
        tuple[float, float]: New geolocation
            New geolocation ``(lon, lat)`` for the longitude and
            latitude that were to be corrected, in geodetic coordinates. [°]
    """
    # Be careful with units here.  Heights may be either in m or km, and may
    # refer to either the Earth's surface on the Earth's centre.  Cloud top
    # height is usually reported in metres above the Earth's surface, rarely in
    # km.  Satellite altitude may be reported in either m or km, but orbital
    # parameters may be in relation the the Earths centre.  The Earth radius
    # from pyresample is reported in km.

    x_sat = np.hstack(lonlat2xyz(sat_lon, sat_lat)) * sat_alt
    x = np.stack(lonlat2xyz(lon, lat), axis=-1) * EARTH_RADIUS*1e3  # km → m
    # the datetime doesn't actually affect the result but is required
    # so we use a placeholder
    (_, elevation) = get_observer_look(
            sat_lon, sat_lat, sat_alt/1e3,  # m → km (wanted by get_observer_look)
            datetime.datetime(2000, 1, 1), lon, lat, 0)
    if np.isscalar(elevation) and elevation == 0:
        raise NotImplementedError(
                "Parallax correction not implemented for "
                "satellite elevation 0")
    if np.isscalar(elevation) and elevation < 0:
        raise ValueError(
                "Satellite is below the horizon.  Cannot calculate parallax "
                "correction.")
    parallax_distance = height / np.sin(np.deg2rad(elevation))

    x_d = x - x_sat
    sat_distance = np.sqrt((x_d*x_d).sum(axis=-1))
    dist_shape = x_d.shape[:-1] + (1,)  # force correct array broadcasting
    x_top = x - x_d*(parallax_distance/sat_distance).reshape(dist_shape)

    (corrected_lon, corrected_lat) = xyz2lonlat(
        x_top[..., 0], x_top[..., 1], x_top[..., 2])
    return (corrected_lon, corrected_lat)


class ParallaxCorrection:
    """Class for parallax corrections.

    This class contains higher-level functionality to wrap the parallal
    correction calculations in :func:`forward_parallax`.  The class is
    initialised using a base area, which is the area for which a corrected
    geolocation will be calculated.  The resulting object is a callable.
    Calling the object with an array of (cloud top) heights returns a
    :class:`~pyresample.geometry.SwathDefinition` describing the new ,
    corrected geolocation.  The cloud top height should cover at least the
    area for which the corrected geolocation will be calculated.

    Note that the ``ctth`` dataset must contain satellite location
    metadata, such as set in the ``orbital_parameters`` dataset attribute
    that is set by many Satpy readers.

    This procedure can be configured as a modifier using the
    :class:`ParallaxCorrectionModifier` class.  However, the modifier can only
    be applied to one dataset at the time, which may not provide optimal
    performance.
    """

    def __init__(self, base_area,
                 debug_mode=False):
        """Initialise parallax correction class.

        Args:
            base_area (:class:`~pyresample.AreaDefinition`): Area for which calculated
                geolocation will be calculated.
            debug_mode (bool): Store diagnostic information in
                self.diagnostics.  This attribute always apply to the most
                recently applied operation only.
        """
        self.base_area = base_area
        self.debug_mode = debug_mode
        self.diagnostics = {}

    def __call__(self, cth_dataset):
        """Apply parallax correction to dataset.

        Args:
            cth_dataset: Dataset containing cloud top heights (or other heights
                to be corrected).

        Returns:
            :class:'~pyresample.geometry.SwathDefinition`: Swathdefinition with corrected
                lat/lon coordinates.
        """
        self.diagnostics.clear()
        return self.corrected_area(cth_dataset)

    def corrected_area(self, cth_dataset):
        """Corrected area.

        Calculate the corrected SwathDefinition for dataset.

        Returns a parallax corrected swathdefinition of the base area.
        """
        logger.debug("Calculating parallax correction using heights from "
                     f"{cth_dataset.attrs.get('name', cth_dataset.name)!s}, "
                     f"with base area {self.base_area.name!s}.")
        try:
            (sat_lon, sat_lat, sat_alt_km) = get_satpos(cth_dataset)
        except KeyError:
            logger.warning(
                    "Orbital parameters missing from metadata. "
                    "Calculating from TLE using skyfield and astropy.")
            (sat_lon, sat_lat, sat_alt_km) = _get_satpos_alt(cth_dataset)
        sat_alt_m = sat_alt_km * 1000
        self._check_overlap(cth_dataset)

        cth_dataset = self._prepare_cth_dataset(cth_dataset)

        (pixel_lon, pixel_lat) = self.base_area.get_lonlats()
        # calculate the shift/error due to the parallax effect
        (shifted_lon, shifted_lat) = forward_parallax(
                sat_lon, sat_lat, sat_alt_m,
                pixel_lon, pixel_lat, cth_dataset.data)

        # lons and lats passed to SwathDefinition must be data-arrays with
        # dimensions,  see https://github.com/pytroll/satpy/issues/1434
        # and https://github.com/pytroll/satpy/issues/1997
        shifted_lon = xr.DataArray(shifted_lon, dims=("y", "x"))
        shifted_lat = xr.DataArray(shifted_lat, dims=("y", "x"))
        shifted_area = SwathDefinition(shifted_lon, shifted_lat)

        # But we are not actually moving pixels, rather we want a
        # coordinate transformation. With this transformation we approximately
        # invert the pixel coordinate transformation, giving the lon and lat
        # where we should retrieve a value for a given pixel.
        (proj_lon, proj_lat) = self._invert_lonlat(
                pixel_lon, pixel_lat, shifted_area)
        # compute here, or I can't use it for resampling later
        proj_lon = xr.DataArray(proj_lon.compute(), dims=("y", "x"))
        proj_lat = xr.DataArray(proj_lat.compute(), dims=("y", "x"))

        return SwathDefinition(proj_lon, proj_lat)

    def _prepare_cth_dataset(self, cth_dataset):
        """Prepare CTH dataset.

        Set cloud top height to zero wherever lat/lon are valid but CTH is
        undefined.  Then resample onto the base area.
        """
        # for calculating the parallax effect, set cth to 0 where it is
        # undefined, unless pixels have no valid lat/lon
        # NB: 0 may be below the surface... could be a problem for high
        # resolution imagery in mountainous or high elevation terrain
        # NB: how tolerant of xarray & dask is this?
        cth_dataset = resample_dataset(
                cth_dataset, self.base_area, resampler="nearest",
                radius_of_influence=50000)
        (pixel_lon, pixel_lat) = cth_dataset.attrs["area"].get_lonlats(chunks=1024)
        cth_dataset = cth_dataset.where(np.isfinite(pixel_lon) & np.isfinite(pixel_lat))
        cth_dataset = cth_dataset.where(cth_dataset.notnull(), 0)
        return cth_dataset

    def _check_overlap(self, cth_dataset):
        """Ensure cth_dataset is usable for parallax correction.

        Checks the coverage of ``cth_dataset`` compared to the ``base_area``.  If
        the entirety of ``base_area`` is covered by ``cth_dataset``, do
        nothing.  If only part of ``base_area`` is covered by ``cth_dataset``,
        raise a `IncompleteHeightWarning`.  If none of ``base_area`` is covered
        by ``cth_dataset``, raise a `MissingHeightError`.
        """
        warnings.warn(
            "Overlap checking not impelemented. Waiting for "
            "fix for https://github.com/pytroll/pyresample/issues/329")

    def _invert_lonlat(self, pixel_lon, pixel_lat, source_area):
        """Invert the lon/lat coordinate transformation.

        When a satellite observes a cloud, a reprojection onto the cloud has
        already happened.
        """
        (source_lon, source_lat) = source_area.get_lonlats()
        lon_diff = source_lon - pixel_lon
        lat_diff = source_lat - pixel_lat
        # We use the bucket resampler here, because parallax correction
        # inevitably means there will be 2 source pixels ending up in the same
        # destination pixel.  We want to choose the biggest shift (max abs in
        # lat_diff and lon_diff), because the biggest shift corresponds to the
        # highest clouds, and if we move a 10 km cloud over a 2 km one, we
        # should retain the 10 km.
        #
        # some things to keep in mind:
        # - even with a constant cloud height, 3 source pixels may end up in
        #   the same destination pixel, because pixels get larger in the
        #   direction of the satellite.  This means clouds may shrink as they
        #   approach the satellite.
        # - the x-shift is a function of y and the y-shift is a function of x,
        #   so a cloud that was rectangular at the start may no longer be
        #   rectangular at the end
        br = BucketResampler(self.base_area,
                             da.array(source_lon), da.array(source_lat))
        inv_lat_diff = br.get_abs_max(lat_diff)
        inv_lon_diff = br.get_abs_max(lon_diff)

        (base_lon, base_lat) = self.base_area.get_lonlats()
        inv_lon = base_lon - inv_lon_diff
        inv_lat = base_lat - inv_lat_diff
        if self.debug_mode:
            self.diagnostics["source_lon"] = source_lon
            self.diagnostics["source_lat"] = source_lat
            self.diagnostics["inv_lon"] = inv_lon
            self.diagnostics["inv_lat"] = inv_lat
            self.diagnostics["base_lon"] = base_lon
            self.diagnostics["base_lat"] = base_lat
            self.diagnostics["inv_lon_diff"] = inv_lon_diff
            self.diagnostics["inv_lat_diff"] = inv_lat_diff
            self.diagnostics["pixel_lon"] = pixel_lon
            self.diagnostics["pixel_lat"] = pixel_lat
            self.diagnostics["lon_diff"] = lon_diff
            self.diagnostics["lat_diff"] = lat_diff
            self.diagnostics["source_area"] = source_area
            self.diagnostics["count"] = br.get_count()
        return (inv_lon, inv_lat)


class ParallaxCorrectionModifier(ModifierBase):
    """Modifier for parallax correction.

    Apply parallax correction as a modifier.  Uses the
    :class:`ParallaxCorrection` class, which in turn uses the
    :func:`forward_parallax` function.  See the documentation there for
    details on the behaviour.

    To use this, add in your ``etc/modifiers/visir.yaml`` something like::

        sensor_name: visir

        modifiers:
          parallax_corrected:
            modifier: !!python/name:satpy.modifiers.parallax.ParallaxCorrectionModifier
            prerequisites:
              - "ctth_alti"
            resampler_args:
              radius_of_influence: 50000

        composites:

          parallax_corrected_VIS006:
            compositor: !!python/name:satpy.composites.GenericCompositor
            prerequisites:
              - name: VIS006
                modifiers: [parallax_corrected]
            standard_name: VIS006

    Here, ``ctth_alti`` is CTH provided by the ``nwcsaf-geo`` reader, so to use it
    one would have to pass both on scene creation::

        sc = Scene({"seviri_l1b_hrit": files_l1b, "nwcsaf-geo": files_l2})
        sc.load(["parallax_corrected_VIS006"])

    Alternately, you can use the lower-level API directly with the
    :class:`ParallaxCorrection` class, which may be more efficient if multiple
    datasets need to be corrected.  RGB Composites cannot be modified in this way
    (i.e. you can't replace "VIS006" by "natural_color").  To get a parallax
    corrected RGB composite, create a new composite where each input has the
    modifier applied.
    """

    def __call__(self, projectables, optional_datasets=None, **info):
        """Apply parallax correction.

        The argument ``projectables`` needs to contain the dataset to be
        projected and the height to use for the correction.
        """
        (to_be_corrected, cth) = projectables
        base_area = to_be_corrected.attrs["area"]
        corrector = self._get_corrector(base_area)
        plax_corr_area = corrector(cth)
        res1 = resample_dataset(
                to_be_corrected, plax_corr_area,
                radius_of_influence=2500, fill_value=np.nan)
        res1.attrs["area"] = to_be_corrected.attrs["area"]

        return res1

    def _get_corrector(self, base_area):
        # only pass on those attributes that are arguments by
        # ParallaxCorrection.__init__
        sig = inspect.signature(ParallaxCorrection.__init__)
        kwargs = {}
        for k in sig.parameters.keys() & self.attrs.keys():
            kwargs[k] = self.attrs[k]
        corrector = ParallaxCorrection(base_area, **kwargs)
        return corrector


def _get_satpos_alt(cth_dataset):
    """Get satellite position if no orbital parameters in metadata.

    Some cloud top height datasets lack orbital parameter information in
    metadata.  Here, orbital parameters are calculated based on the platform
    name and start time, via Two Line Element (TLE) information.

    Needs pyorbital, skyfield, and astropy to be installed.
    """
    from pyorbital.orbital import tlefile
    from skyfield.api import EarthSatellite, load
    from skyfield.toposlib import wgs84

    name = cth_dataset.attrs["platform_name"]
    tle = tlefile.read(name)
    es = EarthSatellite(tle.line1, tle.line2, name)
    ts = load.timescale()
    gc = es.at(ts.from_datetime(
        cth_dataset.attrs["start_time"].replace(tzinfo=datetime.timezone.utc)))
    (lat, lon) = wgs84.latlon_of(gc)
    height = wgs84.height_of(gc).to("km")
    return (lon.degrees, lat.degrees, height.value)
