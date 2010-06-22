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

"""This module is the loader for Feng-Yun 3A (FY-3A) scenes.
"""

from pp.instruments.mersi import MersiScene

class Fy3aMersiScene(MersiScene):
    """This class implements FY-3A scenes as captured by the MERSI
    instrument. It's constructor accepts the same arguments as
    :class:`satin.satellite.SatelliteScene`.
    """

    satname = "fengyun"
    number = "3a"
