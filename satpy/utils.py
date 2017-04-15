#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009, 2012.

# SMHI,
# Folkborgsvägen 1,
# Norrköping,
# Sweden

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>
#   Esben S. Nielsen <esn@dmi.dk>

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# satpy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Module defining various utilities.
"""

import logging
import os
import re

import numpy as np

try:
    import configparser
except:
    from six.moves import configparser


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
        except:
            return self.config_parser.sections()


def ensure_dir(filename):
    """Checks if the dir of f exists, otherwise create it.
    """
    directory = os.path.dirname(filename)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)


class NullHandler(logging.Handler):

    """Empty handler.
    """

    def emit(self, record):
        """Record a message.
        """
        pass


def debug_on():
    """Turn debugging logging on.
    """
    logging_on(logging.DEBUG)

_is_logging_on = False


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
    logging.getLogger('').handlers = [NullHandler()]


def get_logger(name):
    """Return logger with null handle
    """

    log = logging.getLogger(name)
    if not log.handlers:
        log.addHandler(NullHandler())
    return log


###


def strftime(utctime, format_string):
    """Like datetime.strftime, except it works with string formatting
    conversion specifier items on windows, making the assumption that all
    conversion specifiers use mapping keys.

    E.g.:
    >>> from datetime import datetime
    >>> t = datetime.utcnow()
    >>> a = "blabla%Y%d%m-%H%M%S-%(value)s"
    >>> strftime(t, a)
    'blabla20120911-211448-%(value)s'
    """
    res = format_string
    for i in re.finditer("%\w", format_string):
        res = res.replace(i.group(), utctime.strftime(i.group()))
    return res

# Spherical conversions


def lonlat2xyz(lon, lat):
    """Convert lon lat to cartesian."""
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x, y, z


def xyz2lonlat(x, y, z):
    """Convert cartesian to lon lat."""
    lon = np.rad2deg(np.arctan2(y, x))
    lat = np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2)))
    return lon, lat


def angle2xyz(azi, zen):
    """Convert azimuth and zenith to cartesian."""
    azi = np.deg2rad(azi)
    zen = np.deg2rad(zen)
    x = np.sin(zen) * np.sin(azi)
    y = np.sin(zen) * np.cos(azi)
    z = np.cos(zen)
    return x, y, z


def xyz2angle(x, y, z):
    """Convert cartesian to azimuth and zenith."""
    azi = np.rad2deg(np.arctan2(x, y))
    zen = 90 - np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2)))
    return azi, zen
