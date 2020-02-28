#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009-2019 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# satpy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Module defining various utilities."""

import logging
import os
import re
import warnings
import numpy as np
import configparser

_is_logging_on = False
TRACE_LEVEL = 5


class OrderedConfigParser(object):
    """Intercepts read and stores ordered section names.

    Cannot use inheritance and super as ConfigParser use old style classes.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the instance."""
        self.config_parser = configparser.ConfigParser(*args, **kwargs)

    def __getattr__(self, name):
        """Get the attribute."""
        return getattr(self.config_parser, name)

    def read(self, filename):
        """Read config file."""
        try:
            conf_file = open(filename, 'r')
            config = conf_file.read()
            config_keys = re.findall(r'\[.*\]', config)
            self.section_keys = [key[1:-1] for key in config_keys]
        except IOError as e:
            # Pass if file not found
            if e.errno != 2:
                raise

        return self.config_parser.read(filename)

    def sections(self):
        """Get sections from config file."""
        try:
            return self.section_keys
        except:  # noqa: E722
            return self.config_parser.sections()


def ensure_dir(filename):
    """Check if the dir of f exists, otherwise create it."""
    directory = os.path.dirname(filename)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)


def debug_on():
    """Turn debugging logging on."""
    logging_on(logging.DEBUG)


def trace_on():
    """Turn trace logging on."""
    logging_on(TRACE_LEVEL)


def logging_on(level=logging.WARNING):
    """Turn logging on."""
    global _is_logging_on

    if not _is_logging_on:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("[%(levelname)s: %(asctime)s :"
                                               " %(name)s] %(message)s",
                                               '%Y-%m-%d %H:%M:%S'))
        console.setLevel(level)
        logging.getLogger('').addHandler(console)
        _is_logging_on = True

    log = logging.getLogger('')
    log.setLevel(level)
    for h in log.handlers:
        h.setLevel(level)


def logging_off():
    """Turn logging off."""
    logging.getLogger('').handlers = [logging.NullHandler()]


def get_logger(name):
    """Return logger with null handler added if needed."""
    if not hasattr(logging.Logger, 'trace'):
        logging.addLevelName(TRACE_LEVEL, 'TRACE')

        def trace(self, message, *args, **kwargs):
            if self.isEnabledFor(TRACE_LEVEL):
                # Yes, logger takes its '*args' as 'args'.
                self._log(TRACE_LEVEL, message, args, **kwargs)

        logging.Logger.trace = trace

    log = logging.getLogger(name)
    return log


def in_ipynb():
    """Check if we are in a jupyter notebook."""
    try:
        return 'ZMQ' in get_ipython().__class__.__name__
    except NameError:
        return False


# Spherical conversions


def lonlat2xyz(lon, lat):
    """Convert lon lat to cartesian."""
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x, y, z


def xyz2lonlat(x, y, z, asin=False):
    """Convert cartesian to lon lat."""
    lon = np.rad2deg(np.arctan2(y, x))
    if asin:
        lat = np.rad2deg(np.arcsin(z))
    else:
        lat = np.rad2deg(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))
    return lon, lat


def angle2xyz(azi, zen):
    """Convert azimuth and zenith to cartesian."""
    azi = np.deg2rad(azi)
    zen = np.deg2rad(zen)
    x = np.sin(zen) * np.sin(azi)
    y = np.sin(zen) * np.cos(azi)
    z = np.cos(zen)
    return x, y, z


def xyz2angle(x, y, z, acos=False):
    """Convert cartesian to azimuth and zenith."""
    azi = np.rad2deg(np.arctan2(x, y))
    if acos:
        zen = np.rad2deg(np.arccos(z))
    else:
        zen = 90 - np.rad2deg(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))
    return azi, zen


def proj_units_to_meters(proj_str):
    """Convert projection units from kilometers to meters."""
    proj_parts = proj_str.split()
    new_parts = []
    for itm in proj_parts:
        key, val = itm.split('=')
        key = key.strip('+')
        if key in ['a', 'b', 'h']:
            val = float(val)
            if val < 6e6:
                val *= 1000.
                val = '%.3f' % val

        if key == 'units' and val == 'km':
            continue

        new_parts.append('+%s=%s' % (key, val))

    return ' '.join(new_parts)


def _get_sunz_corr_li_and_shibata(cos_zen):
    return 24.35 / (2. * cos_zen + np.sqrt(498.5225 * cos_zen**2 + 1))


def sunzen_corr_cos(data, cos_zen, limit=88., max_sza=95.):
    """Perform Sun zenith angle correction.

    The correction is based on the provided cosine of the zenith
    angle (``cos_zen``).  The correction is limited
    to ``limit`` degrees (default: 88.0 degrees).  For larger zenith
    angles, the correction is the same as at the ``limit`` if ``max_sza``
    is `None`. The default behavior is to gradually reduce the correction
    past ``limit`` degrees up to ``max_sza`` where the correction becomes
    0. Both ``data`` and ``cos_zen`` should be 2D arrays of the same shape.

    """
    # Convert the zenith angle limit to cosine of zenith angle
    limit_rad = np.deg2rad(limit)
    limit_cos = np.cos(limit_rad)
    max_sza_rad = np.deg2rad(max_sza) if max_sza is not None else max_sza

    # Cosine correction
    corr = 1. / cos_zen
    if max_sza is not None:
        # gradually fall off for larger zenith angle
        grad_factor = (np.arccos(cos_zen) - limit_rad) / (max_sza_rad - limit_rad)
        # invert the factor so maximum correction is done at `limit` and falls off later
        grad_factor = 1. - np.log(grad_factor + 1) / np.log(2)
        # make sure we don't make anything negative
        grad_factor = grad_factor.clip(0.)
    else:
        # Use constant value (the limit) for larger zenith angles
        grad_factor = 1.
    corr = corr.where(cos_zen > limit_cos, grad_factor / limit_cos)
    # Force "night" pixels to 0 (where SZA is invalid)
    corr = corr.where(cos_zen.notnull(), 0)

    return data * corr


def atmospheric_path_length_correction(data, cos_zen, limit=88., max_sza=95.):
    """Perform Sun zenith angle correction.

    This function uses the correction method proposed by
    Li and Shibata (2006): https://doi.org/10.1175/JAS3682.1

    The correction is limited to ``limit`` degrees (default: 88.0 degrees). For
    larger zenith angles, the correction is the same as at the ``limit`` if
    ``max_sza`` is `None`. The default behavior is to gradually reduce the
    correction past ``limit`` degrees up to ``max_sza`` where the correction
    becomes 0. Both ``data`` and ``cos_zen`` should be 2D arrays of the same
    shape.

    """
    # Convert the zenith angle limit to cosine of zenith angle
    limit_rad = np.deg2rad(limit)
    limit_cos = np.cos(limit_rad)
    max_sza_rad = np.deg2rad(max_sza) if max_sza is not None else max_sza

    # Cosine correction
    corr = _get_sunz_corr_li_and_shibata(cos_zen)
    # Use constant value (the limit) for larger zenith angles
    corr_lim = _get_sunz_corr_li_and_shibata(limit_cos)

    if max_sza is not None:
        # gradually fall off for larger zenith angle
        grad_factor = (np.arccos(cos_zen) - limit_rad) / (max_sza_rad - limit_rad)
        # invert the factor so maximum correction is done at `limit` and falls off later
        grad_factor = 1. - np.log(grad_factor + 1) / np.log(2)
        # make sure we don't make anything negative
        grad_factor = grad_factor.clip(0.)
    else:
        # Use constant value (the limit) for larger zenith angles
        grad_factor = 1.
    corr = corr.where(cos_zen > limit_cos, grad_factor * corr_lim)
    # Force "night" pixels to 0 (where SZA is invalid)
    corr = corr.where(cos_zen.notnull(), 0)

    return data * corr


def get_satpos(dataset):
    """Get satellite position from dataset attributes.

    Preferences are:

    * Longitude & Latitude: Nadir, actual, nominal, projection
    * Altitude: Actual, nominal, projection

    A warning is issued when projection values have to be used because nothing else is available.

    Returns:
        Geodetic longitude, latitude, altitude

    """
    try:
        orb_params = dataset.attrs['orbital_parameters']

        # Altitude
        try:
            alt = orb_params['satellite_actual_altitude']
        except KeyError:
            try:
                alt = orb_params['satellite_nominal_altitude']
            except KeyError:
                alt = orb_params['projection_altitude']
                warnings.warn('Actual satellite altitude not available, using projection altitude instead.')

        # Longitude & Latitude
        try:
            lon = orb_params['nadir_longitude']
            lat = orb_params['nadir_latitude']
        except KeyError:
            try:
                lon = orb_params['satellite_actual_longitude']
                lat = orb_params['satellite_actual_latitude']
            except KeyError:
                try:
                    lon = orb_params['satellite_nominal_longitude']
                    lat = orb_params['satellite_nominal_latitude']
                except KeyError:
                    lon = orb_params['projection_longitude']
                    lat = orb_params['projection_latitude']
                    warnings.warn('Actual satellite lon/lat not available, using projection centre instead.')
    except KeyError:
        # Legacy
        lon = dataset.attrs['satellite_longitude']
        lat = dataset.attrs['satellite_latitude']
        alt = dataset.attrs['satellite_altitude']

    return lon, lat, alt
