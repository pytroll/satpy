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

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# mpop is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with mpop.  If not, see <http://www.gnu.org/licenses/>.

"""Module defining various utilities.
"""

import os
import re
import ConfigParser
import logging

from mpop import CONFIG_PATH


class OrderedConfigParser(object):

    """Intercepts read and stores ordered section names.
    Cannot use inheritance and super as ConfigParser use old style classes.
    """

    def __init__(self, *args, **kwargs):
        self.config_parser = ConfigParser.ConfigParser(*args, **kwargs)

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
        except IOError, e:
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
    import os
    directory = os.path.dirname(filename)
    if len(directory) and not os.path.isdir(directory):
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

# Read default log level from mpop's config file
_config = ConfigParser.ConfigParser()
_config.read(os.path.join(CONFIG_PATH, 'mpop.cfg'))
try:
    default_loglevel = _config.get('general', 'loglevel')
except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
    default_loglevel = 'WARNING'
default_loglevel = getattr(logging, default_loglevel.upper())
del _config

# logging_on(default_loglevel)


def logging_on(level=default_loglevel):
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

import re


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
