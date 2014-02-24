# -*- coding: utf-8 -*-
# Copyright (c) 2014
#
# Author(s):
# 
#   Panu Lahtinen <pnuu+git@iki.fi>
#
# This file is part of mpop.
#
# mpop is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# mpop is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# mpop.  If not, see <http://www.gnu.org/licenses/>.

'''Helper functions for eg. performing Sun zenith angle correction.
'''

import copy
import numpy as np
from mpop.logger import LOG

def sunzen_corr(chan, time_slot, lonlats=None, limit=80., mode='cos'):

    '''Perform Sun zenith angle correction to the given channel
    (*chan*), and return the corrected data as a new channel.  The
    parameter *limit* can be used to set the maximum zenith angle for
    which the correction is calculated.  For larger angles, the
    correction is the same as at the *limit* (default: 80.0 degrees).
    Coordinate values can be given as a 2-tuple or a two-element list
    *lonlats* of numpy arrays; if None, the coordinates will be read
    from the channel data.  Parameter *mode* is a placeholder for
    other possible zenith angle corrections.  The name of the new
    channel will be *original_chan.name+'_SZC'*, eg. "VIS006_SZC".
    This name is also stored to the chan.info dictionary of the
    originating channel.
    '''
    
    try:
        from pyorbital import astronomy
    except ImportError:
        LOG.warning("Could not load pyorbital.astronomy")
        return None

    if lonlats is None or len(lonlats) != 2:
        # Read coordinates
        LOG.debug("No valid coordinates given, reading from the channel data")
        lons, lats = chan.area.get_lonlats()
    else:
        lons, lats = lonlats
    
    # Calculate cosine of Sun zenith angle
    cos_zen = astronomy.cos_zen(time_slot, lons, lats)

    # Convert the zenith angle limit to cosine of zenith angle
    limit = np.cos(np.radians(limit))

    # Copy the channel
    new_ch = copy.deepcopy(chan)
    # Set the channel name
    new_ch.name += '_SZC'

    if mode == 'cos':
        # Cosine correction
        lim_y, lim_x = np.where(cos_zen > limit)
        new_ch.data[lim_y, lim_x] /= cos_zen[lim_y, lim_x]
        # Use constant value (the limit) for larger zenith
        # angles
        lim_y, lim_x = np.where(cos_zen <= limit)
        new_ch.data[lim_y, lim_x] /= limit
    else:
        # placeholder for other corrections
        pass

    # Add information about the corrected version to original
    # channel
    chan.info["sun_zen_corrected"] = new_ch.name

    return new_ch

