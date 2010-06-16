#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>

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

"""This modules describes the modis instrument.
"""

from pp.instruments.visir import VisirScene

MODIS = [["1", (0.62, 0.645, 0.67), 1000],
         ["2", (0.841, 0.8585, 0.876), 1000],
         ["3", (0.459, 0.469, 0.479), 1000],
         ["4", (0.545, 0.555, 0.565), 1000],
         ["5", (1.23, 1.24, 1.25), 1000],
         ["6", (1.628, 1.64, 1.652), 1000],
         ["7", (2.105, 2.13, 2.155), 1000],
         ["8", (0.405, 0.4125, 0.42), 1000],
         ["9", (0.438, 0.443, 0.448), 1000],
         ["10", (0.483, 0.488, 0.493), 1000],
         ["11", (0.526, 0.531, 0.536), 1000],
         ["12", (0.546, 0.551, 0.556), 1000],
         ["13", (0.662, 0.667, 0.672), 1000],
         ["14", (0.673, 0.678, 0.683), 1000],
         ["15", (0.743, 0.748, 0.753), 1000],
         ["16", (0.862, 0.8695, 0.877), 1000],
         ["17", (0.89, 0.905, 0.92), 1000],
         ["18", (0.931, 0.936, 0.941), 1000],
         ["19", (0.915, 0.94, 0.965), 1000],
         ["20", (3.66, 3.75, 3.84), 1000],
         ["21", (3.929, 3.959, 3.989), 1000],
         ["22", (3.929, 3.959, 3.989), 1000],
         ["23", (4.02, 4.05, 4.08), 1000],
         ["24", (4.433, 4.4655, 4.498), 1000],
         ["25", (4.482, 4.5155, 4.549), 1000],
         ["26", (1.36, 1.375, 1.39), 1000],
         ["27", (6.535, 6.715, 6.895), 1000],
         ["28", (7.175, 7.325, 7.475), 1000],
         ["29", (8.4, 8.55, 8.7), 1000],
         ["30", (9.58, 9.73, 9.88), 1000],
         ["31", (10.78, 11.03, 11.28), 1000],
         ["32", (11.77, 12.02, 12.27), 1000],
         ["33", (13.185, 13.335, 13.485), 1000],
         ["34", (13.485, 13.635, 13.785), 1000],
         ["35", (13.785, 13.935, 14.085), 1000],
         ["36", (14.085, 14.235, 14.385), 1000]]

class ModisScene(VisirScene):
    """This class sets up the Modis instrument channel list.
    """

    channel_list = MODIS
    instrument_name = "modis"
