#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2018 Satpy developers
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
"""Utilities for EUMETSAT satellite data."""

from datetime import datetime, timedelta
import numpy as np

# 6 bytes, 8 bytes, 10 bytes
time_cds_short = [('Days', '>u2'), ('Milliseconds', '>u4')]
time_cds = time_cds_short + [('Microseconds', '>u2')]
time_cds_expanded = time_cds + [('Nanoseconds', '>u2')]
issue_revision = [('Issue', np.uint16), ('Revision', np.uint16)]


def timecds2datetime(tcds):
    """Convert time_cds-variables to datetime-object.

    Works both with a dictionary and a numpy record_array.
    """
    days = int(tcds['Days'])
    milliseconds = int(tcds['Milliseconds'])
    try:
        microseconds = int(tcds['Microseconds'])
    except (KeyError, ValueError):
        microseconds = 0
    try:
        microseconds += int(tcds['Nanoseconds']) / 1000.
    except (KeyError, ValueError):
        pass

    reference = datetime(1958, 1, 1)
    delta = timedelta(days=days, milliseconds=milliseconds,
                      microseconds=microseconds)

    return reference + delta


def recarray2dict(arr):
    """Convert numpy record array to a dictionary."""
    res = {}
    tcds_types = [time_cds_short, time_cds, time_cds_expanded]

    for dtuple in arr.dtype.descr:
        key = dtuple[0]
        ntype = dtuple[1]
        data = arr[key]
        if ntype in tcds_types:
            if data.size > 1:
                res[key] = np.array([timecds2datetime(item)
                                     for item in data.ravel()]).reshape(data.shape)
            else:
                res[key] = timecds2datetime(data)
        elif isinstance(ntype, list):
            res[key] = recarray2dict(data)
        else:
            if data.size == 1:
                data = data[0]
                if ntype[:2] == '|S':
                    # Python2 and Python3 handle strings differently
                    try:
                        data = data.decode()
                    except ValueError:
                        pass
                    data = data.split(':')[0].strip()
                res[key] = data
            else:
                res[key] = data.squeeze()

    return res


def get_geos_area_naming(instrument_name, ssp_lon, resolution):
    """Get a dictionary containing formatted AreaDefinition naming (area_id, description, proj_id)."""
    area_naming_dict = {}

    platform_name = get_platform_name(instrument_name)
    service_mode = get_service_mode(instrument_name, ssp_lon)
    resolution_strings = get_resolution_and_unit_strings(resolution)

    area_naming_dict['area_id'] = '{}_{}_{}_{}{}'.format(platform_name,
                                                         instrument_name,
                                                         service_mode['name'],
                                                         resolution_strings['value'],
                                                         resolution_strings['unit'])

    area_naming_dict['description'] = '{} {} {} area definition ' \
                                      'with {} {} resolution'.format(platform_name.upper(),
                                                                     instrument_name.upper(),
                                                                     service_mode['desc'],
                                                                     resolution_strings['value'],
                                                                     resolution_strings['unit']
                                                                     )

    # same as area_id but without resolution. Parameter on the way to be deprecated
    area_naming_dict['proj_id'] = '{}_{}_{}'.format(platform_name,
                                                    instrument_name,
                                                    service_mode['name'])

    return area_naming_dict


def get_resolution_and_unit_strings(resolution):
    """Get the resolution value and unit as strings. Expects a resolution in m."""
    if resolution >= 1000:
        return {'value': '{:.0f}'.format(resolution*1e-3),
                'unit': 'km'}
    else:
        return {'value': '{:.0f}'.format(resolution),
                'unit': 'm'}


def get_platform_name(instrument_name):
    """Get the platform name for a given instrument."""
    platform_names = {'seviri': 'msg',
                      'fci': 'mtg',
                      }
    return platform_names[instrument_name]


def get_service_mode(instrument_name, ssp_lon):
    """Get information about service mode for a given instrument and subsatellite longitude."""
    service_modes = {'seviri': {'0.0':  {'name': 'fes', 'desc': 'Full Earth Scanning service'},
                                '9.5':  {'name': 'rss', 'desc': 'Rapid Scanning Service'},
                                '41.5': {'name': 'iodc', 'desc': 'Indian Ocean Data Coverage service'}
                                },
                     'fci':    {'0.0':  {'name': 'fdss', 'desc': 'Full Disk Scanning Service'},
                                '9.5':  {'name': 'rss', 'desc': 'Rapid scanning service'},
                                },
                     }

    return service_modes[instrument_name]['{:.1f}'.format(ssp_lon)]
