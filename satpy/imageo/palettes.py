#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009, 2013, 2015, 2016.

# SMHI,
# Folkborgsvägen 1,
# Norrköping,
# Sweden

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

# This file is part of the satpy.

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

"""Palette holder module.
"""


def tv_legend():
    """Palette for TV.
    """
    legend = []
    legend.append((0,   0,   0))  # Unprocessed: Black
    legend.append((0, 120,   0))  # Land
    legend.append((0,   0, 215))  # Sea: Blue
    legend.append((0, 120,   0))  # Land (Snow on land)
    legend.append((0,   0, 215))  # Sea: Blue (Snow/Ice on sea)

    for i in range(5, 256):
        # All other pixel values are grey according to IR temp.
        legend.append((i, i, i))

    return convert_palette(legend)


def vv_legend():
    """Palette for Swedish road authorities (Vägverket).
    """
    legend = []
    legend.append((0,   0,   0))  # Unprocessed: Black
    legend.append((0, 120,   0))  # Land
    legend.append((0,   0, 215))  # Sea: Blue
    # Cloud type values 5 to 8:
    legend.append((255, 150,   0))  # Very low cumuliform
    legend.append((255, 100,   0))  # Very low
    legend.append((255, 220,   0))  # Low cumuliform
    legend.append((255, 180,   0))  # Low

    for i in range(7, 256):
        # All other pixel values are grey according to IR temp.
        legend.append((i, i, i))

    return convert_palette(legend)


def cms_modified():
    """Palette for regular cloud classification.
    """
    return nwcsaf_cloudtype()


def nwcsaf_cloudtype():
    """Palette for regular cloud classification.
    """
    legend = []
    legend.append((100, 100, 100))  # Unprocessed: Grey
    legend.append((0, 120,   0))
    legend.append((0,   0,   0))  # Sea: Black
    legend.append((250, 190, 250))  # Snow
    legend.append((220, 160, 220))  # Sea-ice

    legend.append((255, 150,   0))  # Very low cumuliform
    legend.append((255, 100,   0))  # Very low
    legend.append((255, 220,   0))  # Low cumuliform
    legend.append((255, 180,   0))  # Low
    legend.append((255, 255, 140))  # Medium cumuliform
    legend.append((240, 240,   0))  # Medium
    legend.append((250, 240, 200))  # High cumiliform
    legend.append((215, 215, 150))  # High
    legend.append((255, 255, 255))  # Very high cumuliform
    legend.append((230, 230, 230))  # Very high

    legend.append((0,  80, 215))  # Semi-transparent thin
    legend.append((0, 180, 230))  # Semi-transparent medium
    legend.append((0, 240, 240))  # Semi-transparent thick
    legend.append((90, 200, 160))  # Semi-transparent above
    legend.append((200,   0, 200))  # Broken
    legend.append((95,  60,  30))  # Undefined: Brown

    return convert_palette(legend)


def ctth_height():
    """CTTH height palette.
    """
    legend = []
    legend.append((0,     0,   0))
    legend.append((255,   0, 216))  # 0 meters
    legend.append((126,   0,  43))
    legend.append((153,  20,  47))
    legend.append((178,  51,   0))
    legend.append((255,  76,   0))
    legend.append((255, 102,   0))
    legend.append((255, 164,   0))
    legend.append((255, 216,   0))
    legend.append((216, 255,   0))
    legend.append((178, 255,   0))
    legend.append((153, 255,   0))
    legend.append((0,   255,   0))
    legend.append((0,   140,  48))
    legend.append((0,   178, 255))
    legend.append((0,   216, 255))
    legend.append((0,   255, 255))
    legend.append((238, 214, 210))
    legend.append((239, 239, 223))
    legend.append((255, 255, 255))  # 10,000 meters
    for i in range(79):
        legend.append((255, 255, 255))
    legend.append((224, 224, 224))

    return convert_palette(legend)


def ctth_height_pps():
    """CTTH height palette for NWCSAF/PPS.
    Identical to the one found in the hdf5 files.
    """
    legend = []
    legend.append((255, 0, 216))  # 0 meters
    legend.append((255, 0, 216))  # 0 meters
    legend.append((255, 0, 216))  # 0 meters
    legend.append((126, 0, 43))
    legend.append((126, 0, 43))
    legend.append((153, 20, 47))
    legend.append((153, 20, 47))
    legend.append((153, 20, 47))
    legend.append((178, 51, 0))
    legend.append((178, 51, 0))
    legend.append((255, 76, 0))
    legend.append((255, 76, 0))
    legend.append((255, 76, 0))
    legend.append((255, 102, 0))
    legend.append((255, 102, 0))
    legend.append((255, 164, 0))
    legend.append((255, 164, 0))
    legend.append((255, 164, 0))
    legend.append((255, 216, 0))
    legend.append((255, 216, 0))
    legend.append((216, 255, 0))
    legend.append((216, 255, 0))
    legend.append((178, 255, 0))
    legend.append((178, 255, 0))
    legend.append((178, 255, 0))
    legend.append((153, 255, 0))
    legend.append((153, 255, 0))
    legend.append((0, 255, 0))
    legend.append((0, 255, 0))
    legend.append((0, 255, 0))
    legend.append((0, 140, 48))
    legend.append((0, 140, 48))
    legend.append((0, 178, 255))
    legend.append((0, 178, 255))
    legend.append((0, 178, 255))
    legend.append((0, 216, 255))
    legend.append((0, 216, 255))
    legend.append((0, 255, 255))
    legend.append((0, 255, 255))
    legend.append((0, 255, 255))
    legend.append((238, 214, 210))
    legend.append((238, 214, 210))
    legend.append((239, 239, 223))
    legend.append((239, 239, 223))
    for idx in range(47, 150):
        legend.append((255, 255, 255))  # 10,000 meters

    for idx in range(150, 256):
        legend.append((0, 0, 0))

    return convert_palette(legend)


def sstlut_osisaf_metno():

    legend = []
    legend.append((0, 0, 0))  # /*  Black (out of image)  */
    legend.append((82, 82, 82))  # /*  Dark grey (land)  */
    legend.append((187, 187, 187))  # /* Light grey (cloud contaminated)  */
    legend.append((255, 255, 255))  # /* White (sea ice and snow)  */

    legend.append((255, 0, 255))  # /* Starting at 4 = pink */
    legend.append((195, 0, 129))  # /* Dark pink */
    legend.append((129, 0, 47))  # /* Dark red */
    legend.append((195, 0, 0))  # /* Medium dark red */

    r = 255  # /* Red */
    g = 0
    b = 0
    for i in range(8, 11, 1):
        legend.append((r, g, b))
        r = r - 19
        g = g + 43

    r = 200  # /* Brown */
    g = 128
    b = 0
    for i in range(11, 16, 1):
        legend.append((r, g, b))
        r = r + 11
        g = g + 26
        b = b + 13

    r = 65535 / 256  # /* Yellow */
    g = 65535 / 256
    b = 16185 / 256
    legend.append((r, g, b))
    legend.append((52000 / 256, 65535 / 256, 13500 / 256))
    legend.append((35000 / 256, 65535 / 256, 7000 / 256))

    r = 0  # /* Green */
    g = 65535 / 256
    b = 0
    for i in range(19, 22, 1):
        legend.append((r, g, b))
        g = g - 12422 / 256

    r = 0  # /* Dark Green */
    g = 28269 / 256
    b = 0
    for i in range(22, 26, 1):
        legend.append((r, g, b))
        g = g - 7067 / 256
        b = b + 16383 / 256

    r = 0
    g = 0
    b = 65535 / 256
    legend.append((r, g, b))  # Blue
    legend.append((25700 / 256, 25700 / 256, 65535 / 256))  # Dark purple
    legend.append((38550 / 256, 38550 / 256, 65535 / 256))  # Light purple

    for i in range(29, 256):
        legend.append((0, 0, 0))

    return convert_palette(legend)


def convert_palette(palette):
    """Convert palette from [0,255] range to [0,1].
    """
    new_palette = []
    for i in palette:
        new_palette.append((i[0] / 255.0,
                            i[1] / 255.0,
                            i[2] / 255.0))
    return new_palette
