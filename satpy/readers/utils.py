#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2019 Satpy developers
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
"""Helper functions for satpy readers."""

import bz2
import logging
import os
import shutil
import tempfile
import warnings
from contextlib import closing, contextmanager
from io import BytesIO
from shutil import which
from subprocess import PIPE, Popen  # nosec

import numpy as np
import pyproj
import xarray as xr
from pyresample.geometry import AreaDefinition

from satpy import CHUNK_SIZE, config

LOGGER = logging.getLogger(__name__)


def np2str(value):
    """Convert an `numpy.string_` to str.

    Args:
        value (ndarray): scalar or 1-element numpy array to convert

    Raises:
        ValueError: if value is array larger than 1-element or it is not of
                    type `numpy.string_` or it is not a numpy array

    """
    if hasattr(value, 'dtype') and \
            issubclass(value.dtype.type, (np.str_, np.string_, np.object_)) \
            and value.size == 1:
        value = value.item()
        if not isinstance(value, str):
            # python 3 - was scalar numpy array of bytes
            # otherwise python 2 - scalar numpy array of 'str'
            value = value.decode()
        return value
    else:
        raise ValueError("Array is not a string type or is larger than 1")


def _get_geostationary_height(geos_area):
    params = geos_area.crs.coordinate_operation.params
    h_param = [p for p in params if 'satellite height' in p.name.lower()][0]
    return h_param.value


def _get_geostationary_reference_longitude(geos_area):
    params = geos_area.crs.coordinate_operation.params
    lon_0_params = [p for p in params if 'longitude of natural origin' in p.name.lower()]
    if not lon_0_params:
        return 0
    elif len(lon_0_params) != 1:
        raise ValueError("Not sure how to get reference longitude "
                         "information from AreaDefinition.")
    return lon_0_params[0].value


def _get_geostationary_semi_axes(geos_area):
    from pyresample.utils import proj4_radius_parameters
    return proj4_radius_parameters(geos_area.crs)


def get_geostationary_angle_extent(geos_area):
    """Get the max earth (vs space) viewing angles in x and y."""
    # TODO: take into account sweep_axis_angle parameter
    a, b = _get_geostationary_semi_axes(geos_area)
    h = _get_geostationary_height(geos_area)
    req = float(a) / 1000
    rp = float(b) / 1000
    h = float(h) / 1000 + req

    # compute some constants
    aeq = 1 - req**2 / (h ** 2)
    ap_ = 1 - rp**2 / (h ** 2)

    # generate points around the north hemisphere in satellite projection
    # make it a bit smaller so that we stay inside the valid area
    xmax = np.arccos(np.sqrt(aeq))
    ymax = np.arccos(np.sqrt(ap_))
    return xmax, ymax


def get_geostationary_mask(area, chunks=None):
    """Compute a mask of the earth's shape as seen by a geostationary satellite.

    Args:
        area (pyresample.geometry.AreaDefinition) : Corresponding area
                                                    definition
        chunks (int or tuple): Chunk size for the 2D array that is generated.

    Returns:
        Boolean mask, True inside the earth's shape, False outside.

    """
    # Compute projection coordinates at the earth's limb
    h = _get_geostationary_height(area)
    xmax, ymax = get_geostationary_angle_extent(area)
    xmax *= h
    ymax *= h

    # Compute projection coordinates at the centre of each pixel
    x, y = area.get_proj_coords(chunks=chunks or CHUNK_SIZE)

    # Compute mask of the earth's elliptical shape
    return ((x / xmax) ** 2 + (y / ymax) ** 2) <= 1


def _lonlat_from_geos_angle(x, y, geos_area):
    """Get lons and lats from x, y in projection coordinates."""
    a, b = _get_geostationary_semi_axes(geos_area)
    h = _get_geostationary_height(geos_area)
    lon_0 = _get_geostationary_reference_longitude(geos_area)
    h__ = float(h + a) / 1000
    b__ = (a / float(b)) ** 2

    sd = np.sqrt((h__ * np.cos(x) * np.cos(y)) ** 2 -
                 (np.cos(y)**2 + b__ * np.sin(y)**2) *
                 (h__**2 - (float(a) / 1000)**2))
    # sd = 0

    sn = (h__ * np.cos(x) * np.cos(y) - sd) / (np.cos(y)**2 + b__ * np.sin(y)**2)
    s1 = h__ - sn * np.cos(x) * np.cos(y)
    s2 = sn * np.sin(x) * np.cos(y)
    s3 = -sn * np.sin(y)
    sxy = np.sqrt(s1**2 + s2**2)

    lons = np.rad2deg(np.arctan2(s2, s1)) + lon_0
    lats = np.rad2deg(-np.arctan2(b__ * s3, sxy))

    return lons, lats


def get_geostationary_bounding_box(geos_area, nb_points=50):
    """Get the bbox in lon/lats of the valid pixels inside *geos_area*.

    Args:
      geos_area: The geostationary area to analyse.
      nb_points: Number of points on the polygon

    """
    xmax, ymax = get_geostationary_angle_extent(geos_area)
    h = _get_geostationary_height(geos_area)

    # generate points around the north hemisphere in satellite projection
    # make it a bit smaller so that we stay inside the valid area
    x = np.cos(np.linspace(-np.pi, 0, nb_points // 2)) * (xmax - 0.001)
    y = -np.sin(np.linspace(-np.pi, 0, nb_points // 2)) * (ymax - 0.001)

    # clip the projection coordinates to fit the area extent of geos_area
    ll_x, ll_y, ur_x, ur_y = (np.array(geos_area.area_extent) /
                              float(h))

    x = np.clip(np.concatenate([x, x[::-1]]), min(ll_x, ur_x), max(ll_x, ur_x))
    y = np.clip(np.concatenate([y, -y]), min(ll_y, ur_y), max(ll_y, ur_y))

    return _lonlat_from_geos_angle(x, y, geos_area)


def get_sub_area(area, xslice, yslice):
    """Apply slices to the area_extent and size of the area."""
    new_area_extent = ((area.pixel_upper_left[0] +
                        (xslice.start - 0.5) * area.pixel_size_x),
                       (area.pixel_upper_left[1] -
                        (yslice.stop - 0.5) * area.pixel_size_y),
                       (area.pixel_upper_left[0] +
                        (xslice.stop - 0.5) * area.pixel_size_x),
                       (area.pixel_upper_left[1] -
                        (yslice.start - 0.5) * area.pixel_size_y))

    return AreaDefinition(area.area_id, area.name,
                          area.proj_id, area.crs,
                          xslice.stop - xslice.start,
                          yslice.stop - yslice.start,
                          new_area_extent)


def unzip_file(filename, prefix=None):
    """Unzip the file ending with 'bz2'. Initially with pbzip2 if installed or bz2.

    Args:
        filename: The file to unzip.
        prefix (str, optional): If file is one of many segments of data, prefix random filename
        for correct sorting. This is normally the segment number.

    Returns:
        Temporary filename path for decompressed file or None.

    """
    if os.fspath(filename).endswith('bz2'):
        fdn, tmpfilepath = tempfile.mkstemp(prefix=prefix,
                                            dir=config["tmp_dir"])
        LOGGER.info("Using temp file for BZ2 decompression: %s", tmpfilepath)
        # try pbzip2
        pbzip = which('pbzip2')
        # Run external pbzip2
        if pbzip is not None:
            n_thr = os.environ.get('OMP_NUM_THREADS')
            if n_thr:
                runner = [pbzip,
                          '-dc',
                          '-p'+str(n_thr),
                          filename]
            else:
                runner = [pbzip,
                          '-dc',
                          filename]
            p = Popen(runner, stdout=PIPE, stderr=PIPE)  # nosec
            stdout = BytesIO(p.communicate()[0])
            status = p.returncode
            if status != 0:
                raise IOError("pbzip2 error '%s', failed, status=%d"
                              % (filename, status))
            with closing(os.fdopen(fdn, 'wb')) as ofpt:
                try:
                    stdout.seek(0)
                    shutil.copyfileobj(stdout, ofpt)
                except IOError:
                    import traceback
                    traceback.print_exc()
                    LOGGER.info("Failed to read bzipped file %s",
                                str(filename))
                    os.remove(tmpfilepath)
                    raise
            return tmpfilepath

        # Otherwise, fall back to the original method
        bz2file = bz2.BZ2File(filename)
        with closing(os.fdopen(fdn, 'wb')) as ofpt:
            try:
                ofpt.write(bz2file.read())
            except IOError:
                import traceback
                traceback.print_exc()
                LOGGER.info("Failed to read bzipped file %s", str(filename))
                os.remove(tmpfilepath)
                return None
        return tmpfilepath

    return None


@contextmanager
def unzip_context(filename):
    """Context manager for decompressing a .bz2 file on the fly.

    Uses `unzip_file`. Removes the uncompressed file on exit of the context manager.

    Returns: the filename of the uncompressed file or of the original file if it was not
    compressed.

    """
    unzipped = unzip_file(filename)
    if unzipped is not None:
        yield unzipped
        os.remove(unzipped)
    else:
        yield filename


@contextmanager
def generic_open(filename, *args, **kwargs):
    """Context manager for opening either a regular file or a bzip2 file.

    Returns a file-like object.
    """
    if os.fspath(filename).endswith('.bz2'):
        fp = bz2.open(filename, *args, **kwargs)
    else:
        try:
            fp = filename.open(*args, **kwargs)
        except AttributeError:
            fp = open(filename, *args, **kwargs)

    yield fp

    fp.close()


def bbox(img):
    """Find the bounding box around nonzero elements in the given array.

    Copied from https://stackoverflow.com/a/31402351/5703449 .

    Returns:
        rowmin, rowmax, colmin, colmax

    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def get_earth_radius(lon, lat, a, b):
    """Compute radius of the earth ellipsoid at the given longitude and latitude.

    Args:
        lon: Geodetic longitude (degrees)
        lat: Geodetic latitude (degrees)
        a: Semi-major axis of the ellipsoid (meters)
        b: Semi-minor axis of the ellipsoid (meters)

    Returns:
        Earth Radius (meters)

    """
    geocent = pyproj.Proj(proj='geocent', a=a, b=b, units='m')
    latlong = pyproj.Proj(proj='latlong', a=a, b=b, units='m')
    x, y, z = pyproj.transform(latlong, geocent, lon, lat, 0.)
    return np.sqrt(x**2 + y**2 + z**2)


def reduce_mda(mda, max_size=100):
    """Recursively remove arrays with more than `max_size` elements from the given metadata dictionary."""
    reduced = {}
    for key, val in mda.items():
        if isinstance(val, dict):
            reduced[key] = reduce_mda(val, max_size)
        elif not (isinstance(val, np.ndarray) and val.size > max_size):
            reduced[key] = val
    return reduced


def get_user_calibration_factors(band_name, correction_dict):
    """Retrieve radiance correction factors from user-supplied dict."""
    if band_name in correction_dict:
        try:
            slope = correction_dict[band_name]['slope']
            offset = correction_dict[band_name]['offset']
        except KeyError:
            raise KeyError("Incorrect correction factor dictionary. You must "
                           "supply 'slope' and 'offset' keys.")
    else:
        # If coefficients not present, warn user and use slope=1, offset=0
        warnings.warn("WARNING: You have selected radiance correction but "
                      " have not supplied coefficients for channel " +
                      band_name)
        return 1., 0.

    return slope, offset


def apply_rad_correction(data, slope, offset):
    """Apply GSICS-like correction factors to radiance data."""
    data = (data - offset) / slope
    return data


def get_array_date(scn_data, utc_date=None):
    """Get start time from a channel data array."""
    if utc_date is None:
        try:
            utc_date = scn_data.attrs['start_time']
        except KeyError:
            try:
                utc_date = scn_data.attrs['scheduled_time']
            except KeyError:
                raise KeyError('Scene has no start_time '
                               'or scheduled_time attribute.')
    return utc_date


def apply_earthsun_distance_correction(reflectance, utc_date=None):
    """Correct reflectance data to account for changing Earth-Sun distance."""
    from pyorbital.astronomy import sun_earth_distance_correction
    utc_date = get_array_date(reflectance, utc_date)
    sun_earth_dist = sun_earth_distance_correction(utc_date)

    reflectance.attrs['sun_earth_distance_correction_applied'] = True
    reflectance.attrs['sun_earth_distance_correction_factor'] = sun_earth_dist
    with xr.set_options(keep_attrs=True):
        reflectance = reflectance * sun_earth_dist * sun_earth_dist
    return reflectance


def remove_earthsun_distance_correction(reflectance, utc_date=None):
    """Remove the sun-earth distance correction."""
    from pyorbital.astronomy import sun_earth_distance_correction
    utc_date = get_array_date(reflectance, utc_date)
    sun_earth_dist = sun_earth_distance_correction(utc_date)

    reflectance.attrs['sun_earth_distance_correction_applied'] = False
    reflectance.attrs['sun_earth_distance_correction_factor'] = sun_earth_dist
    with xr.set_options(keep_attrs=True):
        reflectance = reflectance / (sun_earth_dist * sun_earth_dist)
    return reflectance
