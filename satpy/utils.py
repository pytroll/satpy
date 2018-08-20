#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009-2018 PyTroll developers
#
# Author(s):
#
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>
#   Esben S. Nielsen <esn@dmi.dk>
#   Panu Lahtinen <pnuu+git@iki.fi>
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

"""Module defining various utilities.
"""

import logging
import os
import re

import xarray.ufuncs as xu

try:
    import configparser
except ImportError:
    from six.moves import configparser

_is_logging_on = False


class OrderedConfigParser(object):

    """Intercepts read and stores ordered section names.
    Cannot use inheritance and super as ConfigParser use old style classes.
    """

    def __init__(self, *args, **kwargs):
        self.config_parser = configparser.ConfigParser(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.config_parser, name)

    def read(self, filename):
        """Reads config file
        """

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
        """Get sections from config file
        """

        try:
            return self.section_keys
        except:  # noqa: E722
            return self.config_parser.sections()


def ensure_dir(filename):
    """Checks if the dir of f exists, otherwise create it.
    """
    directory = os.path.dirname(filename)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)


def debug_on():
    """Turn debugging logging on.
    """
    logging_on(logging.DEBUG)


def logging_on(level=logging.WARNING):
    """Turn logging on.
    """
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
    """Turn logging off.
    """
    logging.getLogger('').handlers = [logging.NullHandler()]


def get_logger(name):
    """Return logger with null handler added if needed."""
    log = logging.getLogger(name)
    if not log.handlers:
        log.addHandler(logging.NullHandler())
    return log


def in_ipynb():
    """Are we in a jupyter notebook?"""
    try:
        return 'ZMQ' in get_ipython().__class__.__name__
    except NameError:
        return False


# Spherical conversions


def lonlat2xyz(lon, lat):
    """Convert lon lat to cartesian."""
    lat = xu.deg2rad(lat)
    lon = xu.deg2rad(lon)
    x = xu.cos(lat) * xu.cos(lon)
    y = xu.cos(lat) * xu.sin(lon)
    z = xu.sin(lat)
    return x, y, z


def xyz2lonlat(x, y, z):
    """Convert cartesian to lon lat."""
    lon = xu.rad2deg(xu.arctan2(y, x))
    lat = xu.rad2deg(xu.arctan2(z, xu.sqrt(x**2 + y**2)))
    return lon, lat


def angle2xyz(azi, zen):
    """Convert azimuth and zenith to cartesian."""
    azi = xu.deg2rad(azi)
    zen = xu.deg2rad(zen)
    x = xu.sin(zen) * xu.sin(azi)
    y = xu.sin(zen) * xu.cos(azi)
    z = xu.cos(zen)
    return x, y, z


def xyz2angle(x, y, z):
    """Convert cartesian to azimuth and zenith."""
    azi = xu.rad2deg(xu.arctan2(x, y))
    zen = 90 - xu.rad2deg(xu.arctan2(z, xu.sqrt(x**2 + y**2)))
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

    return 24.35 / (2. * cos_zen +
                    xu.sqrt(498.5225 * cos_zen**2 + 1))


def sunzen_corr_cos(data, cos_zen, limit=88.):
    """Perform Sun zenith angle correction.

    The correction is based on the provided cosine of the zenith
    angle (*cos_zen*).  The correction is limited
    to *limit* degrees (default: 88.0 degrees).  For larger zenith
    angles, the correction is the same as at the *limit*.  Both *data*
    and *cos_zen* are given as 2-dimensional Numpy arrays or Numpy
    MaskedArrays, and they should have equal shapes.

    """

    # Convert the zenith angle limit to cosine of zenith angle
    limit = xu.cos(xu.deg2rad(limit))

    # Cosine correction
    corr = 1. / cos_zen
    # Use constant value (the limit) for larger zenith
    # angles
    corr = corr.where(cos_zen > limit).fillna(1 / limit)

    return data * corr


def atmospheric_path_length_correction(data, cos_zen, limit=88.):
    """Perform Sun zenith angle correction.

    This function uses the correction method proposed by
    Li and Shibata (2006): https://doi.org/10.1175/JAS3682.1

    The correction is limited to *limit* degrees (default: 88.0 degrees). For
    larger zenith angles, the correction is the same as at the *limit*. Both
    *data* and *cos_zen* are given as 2-dimensional Numpy arrays or Numpy
    MaskedArrays, and they should have equal shapes.

    """

    # Convert the zenith angle limit to cosine of zenith angle
    limit = xu.cos(xu.radians(limit))

    # Cosine correction
    corr = _get_sunz_corr_li_and_shibata(cos_zen)
    # Use constant value (the limit) for larger zenith
    # angles
    corr_lim = _get_sunz_corr_li_and_shibata(limit)
    corr = corr.where(cos_zen > limit).fillna(corr_lim)

    return data * corr
