#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2013, 2014.

# Author(s):

#   Lars Ørum Rasmussen <ras@dmi.dk>

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# mpop is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# mpop.  If not, see <http://www.gnu.org/licenses/>.
#
"""This modules describes the sarx instrument from the TerraSAR-X satellite.
"""
import numpy as np

from mpop.compositer import Compositer
import logging

LOG = logging.getLogger(__name__)


class SarxCompositer(Compositer):

    """This class sets up the SAR-X instrument channel list.
    """

    instrument_name = "sarx"

    def average(self, downscaling_factor=2, average_window=None):
        """
        Makes a mean convolution of an image.

        :Parameters:
         `downscaling_factor` : int
             image downscaling factor, default is a factor 2.
         `average_window` : int
             window size for calculating mean values, default is
             the same as downscaling_factor.

        :Returns:
         `image` : GeoImage
             mean convoluted image.
        """
        from mpop.imageo.geo_image import GeoImage
        from pyresample import geometry
        import scipy.ndimage as ndi

        self.check_channels(9.65)

        if average_window == None:
            average_window = downscaling_factor

        LOG.info("Downsampling a factor %d and averaging " % downscaling_factor +
                 "in a window of %dx%d" % (average_window, average_window))

        ch = self[9.65]

        # If average window and downscale factor is the same
        # the following could be used:
        #
        #    data = data.reshape([shight, hight/shight,
        #                         swidth, width/swidth]).mean(3).mean(1)

        # avg kernel
        kernel = (np.ones((average_window, average_window), dtype=np.float)
                  / (average_window * average_window))
        # do convolution
        data = ndi.filters.correlate(ch.data.astype(np.float), kernel,
                                     mode='nearest')
        # downscale
        data = data[1::downscaling_factor, 1::downscaling_factor]

        # New area, and correct for integer truncation.
        p_size_x, p_size_y = (ch.area.pixel_size_x * downscaling_factor,
                              ch.area.pixel_size_y * downscaling_factor)
        area_extent = (ch.area.area_extent[0],
                       ch.area.area_extent[1],
                       ch.area.area_extent[0] + data.shape[1] * p_size_x,
                       ch.area.area_extent[1] + data.shape[0] * p_size_y)

        area = geometry.AreaDefinition(self._data_holder.satname +
                                       self._data_holder.instrument_name +
                                       str(area_extent) +
                                       str(data.shape),
                                       "On-the-fly area",
                                       ch.area.proj_id, ch.area.proj_dict,
                                       data.shape[1], data.shape[0],
                                       area_extent)

        return GeoImage(data, area, self.time_slot,
                        fill_value=(0,), mode='L')

    average.prerequisites = set([9.65, ])
