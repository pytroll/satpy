#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Adam Dybbroe <adam.dybbroe@smhi.se>

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

"""This modules describes the MERSI (as on the chinese FY-3A) instrument.
"""

import numpy as np
from pp.instruments.visir import VisirScene
import imageo.geo_image as geo_image
from pp.logger import LOG

# Agregated 1km channels data only:
# EV_250_Aggr.1KM_RefSB (4 bands)
# EV_250_Aggr.1KM_Emissive (1 band)
# EV_1KM_RefSB (15 bands)
MERSI = [["1", (0.420, 0.470, 0.520), 1000],
         ["2", (0.500, 0.550, 0.600), 1000],
         ["3", (0.600, 0.650, 0.700), 1000],
         ["4", (0.815, 0.865, 0.915), 1000],
         ["5", (8.75, 11.25, 13.75), 1000],
         ["6", (0.392, 0.412, 0.432), 1000],
         ["7", (0.423, 0.443, 0.463), 1000],
         ["8", (0.470, 0.490, 0.510), 1000],
         ["9", (0.500, 0.520, 0.540), 1000],
         ["10", (0.545, 0.565, 0.585), 1000],
         ["11", (0.630, 0.650, 0.670), 1000],
         ["12", (0.665, 0.685, 0.705), 1000],
         ["13", (0.745, 0.765, 0.785), 1000],
         ["14", (0.845, 0.865, 0.885), 1000],
         ["15", (0.885, 0.905, 0.925), 1000],
         ["16", (0.920, 0.940, 0.960), 1000],
         ["17", (0.960, 0.980, 1.000), 1000],
         ["18", (0.983, 1.030, 1.050), 1000],
         ["19", (1.590, 1.640, 1.690), 1000],
         ["20", (2.080, 2.130, 2.180), 1000]]


class MersiScene(VisirScene):
    """This class sets up the MERSI instrument channel list.
    """

    channel_list = MERSI
    instrument_name = "mersi"

