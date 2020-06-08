#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2012-2019 Satpy developers
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
"""Convert human coordinates (lon and lat) to an area definition.

Here is a usage example.

python coord2area_def.py france stere 42.0 51.5 -5.5 8.0 1.5
The arguments are "name proj min_lat max_lat min_lon max_lon resolution(km)".
The command above yelds the following result.

### +proj=stere +lat_0=46.75 +lon_0=1.25 +ellps=WGS84

france:
  description: france
  projection:
    proj: stere
    ellps: WGS84
    lat_0: 46.75
    lon_0: 1.25
  shape:
    height: 703
    width: 746
  area_extent:
    lower_left_xy: [-559750.381098, -505020.675776]
    upper_right_xy: [559750.381098, 549517.351948]


The first commented line is just a sum-up. The value of "description" can be changed to any descriptive text.

Such a custom yaml configuration can be profitably saved in a local areas.yaml configuration file that won't be
overridden by future updates of SatPy package. For that purpose the local processing script may have suitable
lines as reported below.

# set PPP_CONFIG_DIR for custom composites
import os
os.environ['PPP_CONFIG_DIR'] = '/my_local_path/for_satpy_configuration'

As a further functionality this script may give a quick display of the defined area,
provided the path for the GSHHG library is supplied via the "-s" option
and the modules PyCoast, Pillow and AggDraw have been installed.

python coord2area_def.py france stere 42.0 51.5 -5.5 8.0 1.5 -s /path/for/gshhs/library

The command above would first print the seen area definition and then launch a casual representation
of the area relying on the information about borders involved.

"""

import argparse
import sys

from pyproj import Proj

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("name",
                        help="The name of the area.")
    parser.add_argument("proj",
                        help="The projection to use. Use proj.4 names, like 'stere', 'merc'...")
    parser.add_argument("min_lat",
                        help="The the latitude of the bottom of the area",
                        type=float)
    parser.add_argument("max_lat",
                        help="The the latitude of the top of the area",
                        type=float)
    parser.add_argument("min_lon",
                        help="The the longitude of the left of the area",
                        type=float)
    parser.add_argument("max_lon",
                        help="The the longitude of the right of the area",
                        type=float)
    parser.add_argument("resolution",
                        help="The resolution of the area (in km)",
                        type=float)
    parser.add_argument("-s", "--shapes",
                        help="Show a preview of the area using the coastlines in this directory")

    args = parser.parse_args()
    name = args.name
    proj = args.proj

    left = args.min_lon
    right = args.max_lon
    up = args.min_lat
    down = args.max_lat

    res = args.resolution * 1000

    lat_0 = (up + down) / 2
    lon_0 = (right + left) / 2

    p = Proj(proj=proj, lat_0=lat_0, lon_0=lon_0, ellps="WGS84")

    left_ex1, up_ex1 = p(left, up)
    right_ex1, up_ex2 = p(right, up)
    left_ex2, down_ex1 = p(left, down)
    right_ex2, down_ex2 = p(right, down)
    left_ex3, dummy = p(left, lat_0)
    right_ex3, dummy = p(right, lat_0)

    area_extent = (min(left_ex1, left_ex2, left_ex3),
                   min(up_ex1, up_ex2),
                   max(right_ex1, right_ex2, right_ex3),
                   max(down_ex1, down_ex2))

    xsize = int(round((area_extent[2] - area_extent[0]) / res))
    ysize = int(round((area_extent[3] - area_extent[1]) / res))

    proj4_string = "+" + \
        " +".join(("proj=" + proj + ",lat_0=" + str(lat_0) +
                   ",lon_0=" + str(lon_0) + ",ellps=WGS84").split(","))

    print('### ' + proj4_string)
    print()
    print(name + ":")
    print("  description: " + name)
    print("  projection:")
    print("    proj: " + proj)
    print("    ellps: WGS84")
    print("    lat_0: " + str(lat_0))
    print("    lon_0: " + str(lon_0))
    print("  shape:")
    print("    height: " + str(ysize))
    print("    width: " + str(xsize))
    print("  area_extent:")
    print("    lower_left_xy: [%f, %f]" % (area_extent[0], area_extent[1]))
    print("    upper_right_xy: [%f, %f]" % (area_extent[2], area_extent[3]))

    if args.shapes is None:
        sys.exit(0)
    from PIL import Image
    from pycoast import ContourWriterAGG
    img = Image.new('RGB', (xsize, ysize))

    area_def = (proj4_string, area_extent)
    cw = ContourWriterAGG(args.shapes)

    cw.add_coastlines(img, (proj4_string, area_extent),
                      resolution='l', width=0.5)

    cw.add_grid(img, area_def, (10.0, 10.0), (2.0, 2.0), write_text=False, outline='white', outline_opacity=175,
                width=1.0, minor_outline='white', minor_outline_opacity=175, minor_width=0.2, minor_is_tick=False)
    img.show()
