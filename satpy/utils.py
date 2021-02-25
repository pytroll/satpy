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
import warnings
import contextlib
from typing import Mapping

import numpy as np
import yaml
from yaml import BaseLoader

try:
    from yaml import UnsafeLoader
except ImportError:
    from yaml import Loader as UnsafeLoader

_is_logging_on = False
TRACE_LEVEL = 5


def ensure_dir(filename):
    """Check if the dir of f exists, otherwise create it."""
    directory = os.path.dirname(filename)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)


def debug_on(deprecation_warnings=True):
    """Turn debugging logging on.

    Sets up a StreamHandler to to `sys.stderr` at debug level for all
    loggers, such that all debug messages (and log messages with higher
    severity) are logged to the standard error stream.

    By default, since Satpy 0.26, this also enables the global visibility
    of deprecation warnings.  This can be suppressed by passing a false
    value.

    Args:
        deprecation_warnings (Optional[bool]): Switch on deprecation warnings.
            Defaults to True.

    Returns:
        None
    """
    logging_on(logging.DEBUG)
    if deprecation_warnings:
        deprecation_warnings_on()


def debug_off():
    """Turn debugging logging off.

    This disables both debugging logging and the global visibility of
    deprecation warnings.
    """
    logging_off()
    deprecation_warnings_off()


@contextlib.contextmanager
def debug(deprecation_warnings=True):
    """Context manager to temporarily set debugging on.

    Example::

        >>> with satpy.utils.debug():
        ...     code_here()

    Args:
        deprecation_warnings (Optional[bool]): Switch on deprecation warnings.
            Defaults to True.
    """
    debug_on(deprecation_warnings=deprecation_warnings)
    yield
    debug_off()


def trace_on():
    """Turn trace logging on."""
    logging_on(TRACE_LEVEL)


class _WarningManager:
    """Class to handle switching warnings on and off."""

    filt = None


_warning_manager = _WarningManager()


def deprecation_warnings_on():
    """Switch on deprecation warnings."""
    warnings.filterwarnings("default", category=DeprecationWarning)
    _warning_manager.filt = warnings.filters[0]


def deprecation_warnings_off():
    """Switch off deprecation warnings."""
    if _warning_manager.filt in warnings.filters:
        warnings.filters.remove(_warning_manager.filt)


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


def recursive_dict_update(d, u):
    """Recursive dictionary update.

    Copied from:

        http://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth

    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            r = recursive_dict_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


def _check_yaml_configs(configs, key):
    """Get a diagnostic for the yaml *configs*.

    *key* is the section to look for to get a name for the config at hand.
    """
    diagnostic = {}
    for i in configs:
        for fname in i:
            with open(fname, 'r', encoding='utf-8') as stream:
                try:
                    res = yaml.load(stream, Loader=UnsafeLoader)
                    msg = 'ok'
                except yaml.YAMLError as err:
                    stream.seek(0)
                    res = yaml.load(stream, Loader=BaseLoader)
                    if err.context == 'while constructing a Python object':
                        msg = err.problem
                    else:
                        msg = 'error'
                finally:
                    try:
                        diagnostic[res[key]['name']] = msg
                    except (KeyError, TypeError):
                        # this object doesn't have a 'name'
                        pass
    return diagnostic


def _check_import(module_names):
    """Import the specified modules and provide status."""
    diagnostics = {}
    for module_name in module_names:
        try:
            __import__(module_name)
            res = 'ok'
        except ImportError as err:
            res = str(err)
        diagnostics[module_name] = res
    return diagnostics


def check_satpy(readers=None, writers=None, extras=None):
    """Check the satpy readers and writers for correct installation.

    Args:
        readers (list or None): Limit readers checked to those specified
        writers (list or None): Limit writers checked to those specified
        extras (list or None): Limit extras checked to those specified

    Returns: bool
        True if all specified features were successfully loaded.

    """
    from satpy.readers import configs_for_reader
    from satpy.writers import configs_for_writer

    print('Readers')
    print('=======')
    for reader, res in sorted(_check_yaml_configs(configs_for_reader(reader=readers), 'reader').items()):
        print(reader + ': ', res)
    print()

    print('Writers')
    print('=======')
    for writer, res in sorted(_check_yaml_configs(configs_for_writer(writer=writers), 'writer').items()):
        print(writer + ': ', res)
    print()

    print('Extras')
    print('======')
    module_names = extras if extras is not None else ('cartopy', 'geoviews')
    for module_name, res in sorted(_check_import(module_names).items()):
        print(module_name + ': ', res)
    print()
