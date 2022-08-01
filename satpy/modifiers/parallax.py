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
surface or the surface is above sea level, the geolocation is not accurate
for the cloud or mountain top.  This module contains routines to correct
imagery such that pixels are shifted or interpolated to correct for this
parallax effect.

Parallax correction is currently only supported for (cloud top) height
that arrives on an :class:`~pyresample.geometry.AreaDefinition`, such
as is standard for geostationary satellites.  Parallax correction with
data described by a :class:`~pyresample.geometry.SwathDefinition`,
such as is common for polar satellites, is not (yet) supported.

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
from pyproj import Geod
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


def get_parallax_corrected_lonlats(sat_lon, sat_lat, sat_alt, lon, lat, height):
    """Calculate parallax corrected lon/lats.

    Satellite geolocation generally assumes an unobstructed view of a smooth
    Earth surface.  In reality, this view may be obstructed by clouds or
    mountains.

    If the view of a pixel at location (lat, lon) is blocked by a cloud
    at height h, this function calculates the (lat, lon) coordinates
    of the cloud above/in front of the invisible surface.

    For scenes that are only partly cloudy, the user might set the cloud top
    height for clear-sky pixels to NaN.  This function will return a corrected
    lat/lon as NaN as well.  The user can use the original lat/lon for those
    pixels or use the higher level :class:`ParallaxCorrection` class.

    This function assumes a spherical Earth.

    .. note::

        Be careful with units!  This code expects ``sat_alt`` and
        ``height`` to be in meter above the Earth's surface.  You may
        have to convert your input correspondingly.  Cloud Top Height
        is usually reported in meters above the Earth's surface, rarely
        in km.  Satellite altitude may be reported in either m or km, but
        orbital parameters are usually in relation to the Earth's centre.
        The Earth radius from pyresample is reported in km.

    Args:
        sat_lon (number): Satellite longitude in geodetic coordinates [degrees]
        sat_lat (number): Satellite latitude in geodetic coordinates [degrees]
        sat_alt (number): Satellite altitude above the Earth surface [m]
        lon (array or number): Longitudes of pixel or pixels to be corrected,
            in geodetic coordinates [degrees]
        lat (array or number): Latitudes of pixel/pixels to be corrected, in
            geodetic coordinates [degrees]
        height (array or number): Heights of pixels on which the correction
            will be based.  Typically this is the cloud top height. [m]

    Returns:
        tuple[float, float]: Corrected geolocation
            Corrected geolocation ``(lon, lat)`` in geodetic coordinates for
            the pixel(s) to be corrected. [degrees]
    """
    elevation = _get_satellite_elevation(sat_lon, sat_lat, sat_alt, lon, lat)
    parallax_distance = _calculate_slant_cloud_distance(height, elevation)
    shifted_xyz = _get_parallax_shift_xyz(
            sat_lon, sat_lat, sat_alt, lon, lat, parallax_distance)

    return xyz2lonlat(
        shifted_xyz[..., 0], shifted_xyz[..., 1], shifted_xyz[..., 2])


def get_surface_parallax_displacement(
        sat_lon, sat_lat, sat_alt, lon, lat, height):
    """Calculate surface parallax displacement.

    Calculate the displacement due to parallax error.  Input parameters are
    identical to :func:`get_parallax_corrected_lonlats`.

    Returns:
        number or array: parallax displacement in meter
    """
    (corr_lon, corr_lat) = get_parallax_corrected_lonlats(sat_lon, sat_lat, sat_alt, lon, lat, height)
    # Get parallax displacement
    geod = Geod(ellps="sphere")
    _, _, parallax_dist = geod.inv(corr_lon, corr_lat, lon, lat)
    return parallax_dist


def _get_parallax_shift_xyz(sat_lon, sat_lat, sat_alt, lon, lat, parallax_distance):
    """Calculate the parallax shift in cartesian coordinates.

    From satellite position and cloud position, get the parallax shift in
    cartesian coordinates:

    Args:
        sat_lon (number): Satellite longitude in geodetic coordinates [degrees]
        sat_lat (number): Satellite latitude in geodetic coordinates [degrees]
        sat_alt (number): Satellite altitude above the Earth surface [m]
        lon (array or number): Longitudes of pixel or pixels to be corrected,
            in geodetic coordinates [degrees]
        lat (array or number): Latitudes of pixel/pixels to be corrected, in
            geodetic coordinates [degrees]
        parallax_distance (array or number): Cloud to ground distance with parallax
            effect [m].

    Returns:
        Parallax shift in cartesian coordinates in meter.
    """
    sat_xyz = np.hstack(lonlat2xyz(sat_lon, sat_lat)) * sat_alt
    cth_xyz = np.stack(lonlat2xyz(lon, lat), axis=-1) * EARTH_RADIUS*1e3  # km → m
    delta_xyz = cth_xyz - sat_xyz
    sat_distance = np.sqrt((delta_xyz*delta_xyz).sum(axis=-1))
    dist_shape = delta_xyz.shape[:-1] + (1,)  # force correct array broadcasting
    return cth_xyz - delta_xyz*(parallax_distance/sat_distance).reshape(dist_shape)


def _get_satellite_elevation(sat_lon, sat_lat, sat_alt, lon, lat):
    """Get satellite elevation.

    Get the satellite elevation from satellite lon/lat/alt for positions
    lon/lat.
    """
    placeholder_date = datetime.datetime(2000, 1, 1)  # no impact on get_observer_look?
    (_, elevation) = get_observer_look(
            sat_lon, sat_lat, sat_alt/1e3,  # m → km (wanted by get_observer_look)
            placeholder_date, lon, lat, 0)
    return elevation


def _calculate_slant_cloud_distance(height, elevation):
    """Calculate slant cloud to ground distance.

    From (cloud top) height and satellite elevation, calculate the
    slant cloud-to-ground distance along the line of sight of the satellite.
    """
    if np.isscalar(elevation) and elevation == 0:
        raise NotImplementedError(
                "Parallax correction not implemented for "
                "satellite elevation 0")
    if np.isscalar(elevation) and elevation < 0:
        raise ValueError(
                "Satellite is below the horizon.  Cannot calculate parallax "
                "correction.")
    return height / np.sin(np.deg2rad(elevation))


class ParallaxCorrection:
    """Parallax correction calculations.

    This class contains higher-level functionality to wrap the parallax
    correction calculations in :func:`get_parallax_corrected_lonlats`.  The class is
    initialised using a base area, which is the area for which a corrected
    geolocation will be calculated.  The resulting object is a callable.
    Calling the object with an array of (cloud top) heights returns a
    :class:`~pyresample.geometry.SwathDefinition` describing the new ,
    corrected geolocation.  The cloud top height should cover at least the
    area for which the corrected geolocation will be calculated.

    Note that the ``ctth`` dataset must contain satellite location
    metadata, such as set in the ``orbital_parameters`` dataset attribute
    that is set by many Satpy readers.  It is essential that the datasets to be
    corrected are coming from the same platform as the provided cloud top
    height.

    A note on the algorithm and the implementation.  Parallax correction
    is inherently an inverse problem.  The reported geolocation in
    satellite data files is the true location plus the parallax error.
    Therefore, this class first calculates the true geolocation (using
    :func:`get_parallax_corrected_lonlats`), which gives a shifted longitude and
    shifted latitude on an irregular grid.  The difference between
    the original and the shifted grid is the parallax error or shift.
    The magnitude of this error can be estimated with
    :func:`get_surface_parallax_displacement`.
    With this difference, we need to invert the parallax correction to
    calculate the corrected geolocation.  Due to parallax correction,
    high clouds shift a lot, low clouds shift a little, and cloud-free
    pixels shift not at all.  The shift may result in zero, one,
    two, or more source pixel onto a destination pixel.  Physically,
    this corresponds to the situation where a narrow but high cloud is
    viewed at a large angle.  The cloud may occupy two or more pixels when
    viewed at a large angle, but only one when viewed straight from above.
    To accurately reproduce this perspective, the parallax correction uses
    the :class:`~pyresample.bucket.BucketResampler` class, specifically
    the :meth:`~pyresample.bucket.BucketResampler.get_abs_max` method, to
    retain only the largest absolute shift (corresponding to the highest
    cloud) within each pixel.  Any other resampling method at this step
    would yield incorrect results.  When cloud moves over clear-sky, the
    clear-sky pixel is unshifted and the shift is located exactly in the
    centre of the grid box, so nearest-neighbour resampling would lead to
    such shifts being deselected.  Other resampling methods would average
    large shifts with small shifts, leading to unpredictable results.
    Now the reprojected shifts can be applied to the original lat/lon,
    returning a new :class:`~pyresample.geometry.SwathDefinition`.
    This is is the object returned by :meth:`corrected_area`.

    This procedure can be configured as a modifier using the
    :class:`ParallaxCorrectionModifier` class.  However, the modifier can only
    be applied to one dataset at the time, which may not provide optimal
    performance, although dask should reuse identical calculations between
    multiple channels.

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

    def __call__(self, cth_dataset, **kwargs):
        """Apply parallax correction to dataset.

        Args:
            cth_dataset: Dataset containing cloud top heights (or other heights
                to be corrected).

        Returns:
            :class:'~pyresample.geometry.SwathDefinition`: Swathdefinition with corrected
                lat/lon coordinates.
        """
        self.diagnostics.clear()
        return self.corrected_area(cth_dataset, **kwargs)

    def corrected_area(self, cth_dataset,
                       cth_resampler="nearest",
                       cth_radius_of_influence=50000,
                       lonlat_chunks=1024):
        """Return the parallax corrected SwathDefinition.

        Using the cloud top heights provided in ``cth_dataset``, calculate the
        :class:`pyresample.geometry.SwathDefinition` that estimates the
        geolocation for each pixel if it had been viewed from straight above
        (without parallax error).  The cloud top height will first be resampled
        onto the area passed upon class initialisation in :meth:`__init__`.
        Pixels that are invisible after parallax correction are not retained
        but get geolocation NaN.

        Args:
            cth_dataset (:class:`~xarray.DataArray`): Cloud top height in
                meters.  The variable attributes must contain an ``area``
                attribute describing the geolocation in a pyresample-aware way,
                and they must contain satellite orbital parameters.  The
                dimensions must be ``(y, x)``.  For best performance, this
                should be a dask-based :class:`~xarray.DataArray`.
            cth_resampler (string, optional): Resampler to use when resampling the
                (cloud top) height to the base area.  Defaults to "nearest".
            cth_radius_of_influence (number, optional): Radius of influence to use when
                resampling the (cloud top) height to the base area.  Defaults
                to 50000.
            lonlat_chunks (int, optional): Chunking to use when calculating lon/lats.
                Probably the default (1024) should be fine.

        Returns:
            :class:`~pyresample.geometry.SwathDefinition` describing parallax
            corrected geolocation.
        """
        logger.debug("Calculating parallax correction using heights from "
                     f"{cth_dataset.attrs.get('name', cth_dataset.name)!s}, "
                     f"with base area {self.base_area.name!s}.")
        (sat_lon, sat_lat, sat_alt_m) = _get_satpos_from_cth(cth_dataset)
        self._check_overlap(cth_dataset)

        cth_dataset = self._prepare_cth_dataset(
                cth_dataset, resampler=cth_resampler,
                radius_of_influence=cth_radius_of_influence,
                lonlat_chunks=lonlat_chunks)

        (base_lon, base_lat) = self.base_area.get_lonlats(chunks=lonlat_chunks)
        # calculate the shift/error due to the parallax effect
        (corrected_lon, corrected_lat) = get_parallax_corrected_lonlats(
                sat_lon, sat_lat, sat_alt_m,
                base_lon, base_lat, cth_dataset.data)

        shifted_area = self._get_swathdef_from_lon_lat(corrected_lon, corrected_lat)

        # But we are not actually moving pixels, rather we want a
        # coordinate transformation. With this transformation we approximately
        # invert the pixel coordinate transformation, giving the lon and lat
        # where we should retrieve a value for a given pixel.
        (proj_lon, proj_lat) = self._get_corrected_lon_lat(
                base_lon, base_lat, shifted_area)

        return self._get_swathdef_from_lon_lat(proj_lon, proj_lat)

    @staticmethod
    def _get_swathdef_from_lon_lat(lon, lat):
        """Return a SwathDefinition from lon/lat.

        Turn ndarrays describing lon/lat into xarray with dimensions y, x, then
        use these to create a :class:`~pyresample.geometry.SwathDefinition`.
        """
        # lons and lats passed to SwathDefinition must be data-arrays with
        # dimensions,  see https://github.com/pytroll/satpy/issues/1434
        # and https://github.com/pytroll/satpy/issues/1997
        return SwathDefinition(
            xr.DataArray(lon, dims=("y", "x")),
            xr.DataArray(lat, dims=("y", "x")))

    def _prepare_cth_dataset(
            self, cth_dataset, resampler="nearest", radius_of_influence=50000,
            lonlat_chunks=1024):
        """Prepare CTH dataset.

        Set cloud top height to zero wherever lat/lon are valid but CTH is
        undefined.  Then resample onto the base area.
        """
        # for calculating the parallax effect, set cth to 0 where it is
        # undefined, unless pixels have no valid lat/lon
        # NB: 0 may be below the surface... could be a problem for high
        # resolution imagery in mountainous or high elevation terrain
        # NB: how tolerant of xarray & dask is this?
        resampled_cth_dataset = resample_dataset(
                cth_dataset, self.base_area, resampler=resampler,
                radius_of_influence=radius_of_influence)
        (pixel_lon, pixel_lat) = resampled_cth_dataset.attrs["area"].get_lonlats(
                chunks=lonlat_chunks)
        masked_resampled_cth_dataset = resampled_cth_dataset.where(
                np.isfinite(pixel_lon) & np.isfinite(pixel_lat))
        masked_resampled_cth_dataset = masked_resampled_cth_dataset.where(
                masked_resampled_cth_dataset.notnull(), 0)
        return masked_resampled_cth_dataset

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

    def _get_corrected_lon_lat(self, base_lon, base_lat, shifted_area):
        """Calculate the corrected lon/lat based from the shifted area.

        After calculating the shifted area based on
        :func:`get_parallax_corrected_lonlats`,
        we invert the parallax error and estimate where those pixels came from.
        For details on the algorithm, see the class docstring.
        """
        (corrected_lon, corrected_lat) = shifted_area.get_lonlats(chunks=1024)
        lon_diff = corrected_lon - base_lon
        lat_diff = corrected_lat - base_lat
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
        bur = BucketResampler(self.base_area,
                              da.array(corrected_lon), da.array(corrected_lat))
        inv_lat_diff = bur.get_abs_max(lat_diff)
        inv_lon_diff = bur.get_abs_max(lon_diff)

        inv_lon = base_lon - inv_lon_diff
        inv_lat = base_lat - inv_lat_diff
        if self.debug_mode:
            self.diagnostics["corrected_lon"] = corrected_lon
            self.diagnostics["corrected_lat"] = corrected_lat
            self.diagnostics["inv_lon"] = inv_lon
            self.diagnostics["inv_lat"] = inv_lat
            self.diagnostics["inv_lon_diff"] = inv_lon_diff
            self.diagnostics["inv_lat_diff"] = inv_lat_diff
            self.diagnostics["base_lon"] = base_lon
            self.diagnostics["base_lat"] = base_lat
            self.diagnostics["lon_diff"] = lon_diff
            self.diagnostics["lat_diff"] = lat_diff
            self.diagnostics["shifted_area"] = shifted_area
            self.diagnostics["count"] = xr.DataArray(
                bur.get_count(), dims=("y", "x"), attrs={"area": self.base_area})
        return (inv_lon, inv_lat)


class ParallaxCorrectionModifier(ModifierBase):
    """Modifier for parallax correction.

    Apply parallax correction as a modifier.  Uses the
    :class:`ParallaxCorrection` class, which in turn uses the
    :func:`get_parallax_corrected_lonlats` function.  See the documentation there for
    details on the behaviour.

    To use this, add to ``composites/visir.yaml`` within ``SATPY_CONFIG_PATH``
    something like::

        sensor_name: visir

        modifiers:
          parallax_corrected:
            modifier: !!python/name:satpy.modifiers.parallax.ParallaxCorrectionModifier
            prerequisites:
              - "ctth_alti"
            dataset_radius_of_influence: 50000

        composites:

          parallax_corrected_VIS006:
            compositor: !!python/name:satpy.composites.SingleBandCompositor
            prerequisites:
              - name: VIS006
                modifiers: [parallax_corrected]

    Here, ``ctth_alti`` is CTH provided by the ``nwcsaf-geo`` reader, so to use it
    one would have to pass both on scene creation::

        sc = Scene({"seviri_l1b_hrit": files_l1b, "nwcsaf-geo": files_l2})
        sc.load(["parallax_corrected_VIS006"])

    The modifier takes optional global parameters, all of which are optional.
    They affect various steps in the algorithm.  Setting them may impact
    performance:

    cth_resampler
        Resampler to use when resampling (cloud top) height to the base area.
        Defaults to "nearest".
    cth_radius_of_influence
        Radius of influence to use when resampling the (cloud top) height to
        the base area.  Defaults to 50000.
    lonlat_chunks
        Chunk size to use when obtaining longitudes and latitudes from the area
        definition.  Defaults to 1024.  If you set this to None, then parallax
        correction will involve premature calculation.  Changing this may or
        may not make parallax correction slower or faster.
    dataset_radius_of_influence
        Radius of influence to use when resampling the dataset onto the
        swathdefinition describing the parallax-corrected area.  Defaults to
        50000.  This always uses nearest neighbour resampling.

    Alternately, you can use the lower-level API directly with the
    :class:`ParallaxCorrection` class, which may be more efficient if multiple
    datasets need to be corrected.  RGB Composites cannot be modified in this way
    (i.e. you can't replace "VIS006" by "natural_color").  To get a parallax
    corrected RGB composite, create a new composite where each input has the
    modifier applied.  The parallax calculation should only occur once, because
    calculations are happening via dask and dask should reuse the calculation.
    """

    def __call__(self, projectables, optional_datasets=None, **info):
        """Apply parallax correction.

        The argument ``projectables`` needs to contain the dataset to be
        projected and the height to use for the correction.
        """
        (to_be_corrected, cth) = projectables
        base_area = to_be_corrected.attrs["area"]
        corrector = self._get_corrector(base_area)
        plax_corr_area = corrector(
                cth,
                cth_resampler=self.attrs.get("cth_resampler", "nearest"),
                cth_radius_of_influence=self.attrs.get("cth_radius_of_influence", 50_000),
                lonlat_chunks=self.attrs.get("lonlat_chunks", 1024),
                )
        res = resample_dataset(
                to_be_corrected, plax_corr_area,
                radius_of_influence=self.attrs.get("dataset_radius_of_influence", 50_000),
                fill_value=np.nan)
        res.attrs["area"] = to_be_corrected.attrs["area"]
        self.apply_modifier_info(to_be_corrected, res)

        return res

    def _get_corrector(self, base_area):
        # only pass on those attributes that are arguments by
        # ParallaxCorrection.__init__
        sig = inspect.signature(ParallaxCorrection.__init__)
        kwargs = {}
        for k in sig.parameters.keys() & self.attrs.keys():
            kwargs[k] = self.attrs[k]
        corrector = ParallaxCorrection(base_area, **kwargs)
        return corrector


def _get_satpos_from_cth(cth_dataset):
    """Obtain satellite position from CTH dataset, height in meter.

    From a CTH dataset, obtain the satellite position lon, lat, altitude/m,
    either directly from orbital parameters, or, when missing, from the
    platform name using pyorbital and skyfield.
    """
    (sat_lon, sat_lat, sat_alt_km) = get_satpos(
            cth_dataset, use_tle=True)
    return (sat_lon, sat_lat, sat_alt_km * 1000)
