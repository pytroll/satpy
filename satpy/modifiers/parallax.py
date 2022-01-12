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

import numpy as np
import xarray as xr
from pyorbital.orbital import A as EARTH_RADIUS
from pyorbital.orbital import get_observer_look
from pyresample.geometry import SwathDefinition
from pyresample.kd_tree import resample_nearest

from satpy import Scene
from satpy.modifiers import ModifierBase
from satpy.utils import get_satpos, lonlat2xyz, xyz2lonlat

logger = logging.getLogger(__name__)


class MissingHeightError(ValueError):
    """Raised when heights do not overlap with area to be corrected."""


class IncompleteHeightWarning(UserWarning):
    """Raised when heights only partially overlap with area to be corrected."""


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
        tuple[float, float]: New geolocation
            New geolocation ``(lon, lat)`` for the longitude and
            latitude that were to be corrected, in geodetic coordinates. [°]
    """
    x_sat = np.hstack(lonlat2xyz(sat_lon, sat_lat)) * sat_alt
    x = np.stack(lonlat2xyz(lon, lat), axis=-1) * EARTH_RADIUS
    # the datetime doesn't actually affect the result but is required
    # so we use a placeholder
    (_, elevation) = get_observer_look(
            sat_lon, sat_lat, sat_alt,
            datetime.datetime(2000, 1, 1), lon, lat, EARTH_RADIUS)
    if np.isscalar(elevation) and elevation == 0:
        raise NotImplementedError(
                "Parallax correction not implemented for "
                "satellite elevation 0")
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
    corrected geolocation.  This ``SwathDefinition`` can then be used for
    resampling a satpy Scene, yielding a corrected geolocation for all datasets
    in the Scene.  For example::

      >>> global_scene = satpy.Scene(reader="seviri_l1b_hrit", filenames=files_sat)
      >>> global_scene.load(['IR_087','IR_120'])
      >>> global_nwc = satpy.Scene(filenames=files_nwc)
      >>> global_nwc.load(['ctth'])
      >>> area_def = satpy.resample.get_area_def(area)
      >>> parallax_correction = ParallaxCorrection(area_def)
      >>> plax_corr_area = parallax_correction(global_nwc["ctth"])
      >>> local_scene = global_scene.resample(plax_corr_area)
      >>> local_nwc = global_nwc.resample(plax_corr_area)
      >>> local_scene[...].attrs["area"] = area_def
      >>> local_nwc[...].attrs["area"] = area_def

    Note that the ``ctth`` dataset must contain geolocation metadata, such as
    set in the ``orbital_parameters`` dataset attribute by many readers.

    This produce can be configured as a modifier using the
    :class:`ParallaxCorrectionModifier` class.  However, the modifier can only
    be applied to one dataset at the time, which may not provide optimal
    performance.
    """

    def __init__(self, base_area,
                 resampler=resample_nearest, search_radius=50_000):
        """Initialise parallax correction class.

        Args:
            base_area (:class:`~pyresample.AreaDefinition`): Area for which calculated
                geolocation will be calculated.
            resampler (function): Function to use for resampling.  Must
                have same interface as
                :func:`~pyresample.kd_tree.resample_nearest`.  Note that using
                the bilinear resampler included with Satpy yields incorrect results
                near the poles and the antimeridian for the purposes of
                parallax correction.  The default, the nearest neighbour
                resampling, yiedls correct results.
            search_radius (number): Search radius to use with resampler.
        """
        self.base_area = base_area
        self.resampler = resampler
        self.search_radius = search_radius

    def __call__(self, cth_dataset):
        """Apply parallax correction to dataset.

        Args:
            cth_dataset: Dataset containing cloud top heights (or other heights
                to be corrected).

        Returns:
            :class:'~pyresample.geometry.SwathDefinition`: Swathdefinition with corrected
                lat/lon coordinates.
        """
        return self.corrected_area(cth_dataset)

    def corrected_area(self, cth_dataset):
        """Corrected area.

        Calculate the corrected SwathDefinition for dataset.

        Returns a parallax corrected swathdefinition of the base area.
        """
        logger.debug("Calculating parallax correction using "
                     f"{cth_dataset.attrs.get('name', cth_dataset.name)!s}")
        area = cth_dataset.area
        try:
            (sat_lon, sat_lat, sat_alt) = get_satpos(cth_dataset)
        except KeyError:
            logger.warning(
                    "Orbital parameters missing from metadata. "
                    "Calculating from TLE using skyfield and astropy.")
            (sat_lon, sat_lat, sat_alt) = _get_satpos_alt(cth_dataset)
        cth_dataset = self._preprocess_cth(cth_dataset)
        self._check_overlap(cth_dataset)
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

    def _preprocess_cth(self, cth_dataset):
        """To be documented."""
        cth = cth_dataset.copy().fillna(0.0)
        if "units" in cth_dataset.attrs.keys():
            units = cth_dataset.attrs["units"]
            if units == 'm':  # convert to km
                cth = cth * 1e-3
        return cth

    def _invert_lonlat(self, pixel_lon, pixel_lat, source_area):
        """Invert the lon/lat coordinate transformation.

        When a satellite observes a cloud, a reprojection onto the cloud has
        already happened.
        """
        (source_lon, source_lat) = source_area.get_lonlats()
        lon_diff = source_lon - pixel_lon
        lat_diff = source_lat - pixel_lat
        inv_lon_diff = self.resampler(
                source_area, lon_diff, self.base_area,
                self.search_radius)
        inv_lat_diff = self.resampler(
                source_area, lat_diff, self.base_area,
                self.search_radius)

        (base_lon, base_lat) = self.base_area.get_lonlats()
        inv_lon = base_lon + inv_lon_diff
        inv_lat = base_lat + inv_lat_diff
        return (inv_lon, inv_lat)


class ParallaxCorrectionModifier(ModifierBase):
    """Modifier for parallax correction.

    Apply parallax correction as a modifier.  Uses the
    :class:`ParallaxCorrection` class, which in turn uses the
    :func:`forward_parallax` function.  See the documentation there for
    details on the behaviour.

    To use this, add in your ``etc/modifiers/visir.yaml`` something like::

        modifiers:
          parallax_corrected:
            modifier: !!python/name:satpy.modifiers.parallax.ParallaxCorrectionModifier
            prerequisites:
            - name: CTH
            resampler: !!python/name:pyresample.kd_tree.resample_nearest
            search_radius: 50000

    Here, ``resampler`` and ``search_radius`` are optional.

    Alternately, you can use the lower-level API directly with the
    :class:`ParallaxCorrection` class, which may be more efficient if multiple
    datasets need to be resampled.
    """

    def __call__(self, projectables, optional_datasets=None, **info):
        """Apply parallax correction.

        The argument ``projectables`` needs to contain the dataset to be
        projected and the height to use for the correction.
        """
        # NB: Can I avoid creating a scene object here?
        (to_be_corrected, cth) = projectables
        base_area = to_be_corrected.attrs["area"]
        kwargs = {}
        # only pass on those attributes that are arguments by
        # ParallaxCorrection.__init__
        sig = inspect.signature(ParallaxCorrection.__init__)
        for k in sig.parameters.keys() & self.attrs.keys():
            kwargs[k] = self.attrs[k]
        corrector = ParallaxCorrection(base_area, **kwargs)
        plax_corr_area = corrector(cth)
        global_scene = Scene()
        global_scene["dataset"] = to_be_corrected
        local_scene = global_scene.resample(plax_corr_area)
        local_scene["dataset"].attrs["area"] = base_area
        return local_scene["dataset"]


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

    tle = tlefile.read(name := cth_dataset.attrs["platform_name"])
    es = EarthSatellite(tle.line1, tle.line2, name)
    ts = load.timescale()
    gc = es.at(ts.from_datetime(
        cth_dataset.attrs["start_time"].replace(tzinfo=datetime.timezone.utc)))
    (lat, lon) = wgs84.latlon_of(gc)
    height = wgs84.height_of(gc).to("km")
    return (lon.degrees, lat.degrees, height.value)
