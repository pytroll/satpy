# Copyright (c) 2012, 2015
#

# Author(s):
#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""
Convert human coordinates (lon and lat) to an area definition.

Here is a usage example:
python coord2area_def.py france stere 42.0 51.5 -5.5 8.0 1.5
(the arguments are "name proj min_lat max_lat min_lon max_lon resolution(km)")


and the result is:
REGION: france {
	NAME:	france
	PCS_ID:	stere_1.25_46.75
	PCS_DEF:	proj=stere,lat_0=46.75,lon_0=1.25,ellps=WGS84
	XSIZE:  746
	YSIZE:  703
	AREA_EXTENT:	(-559750.38109755167, -505020.6757764442,
559750.38109755167, 549517.35194826045)
};

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

    print proj4_string

    print "REGION:", name, "{"
    print "\tNAME:\t", name
    print "\tPCS_ID:\t", proj + "_" + str(lon_0) + "_" + str(lat_0)
    print ("\tPCS_DEF:\tproj=" + proj +
           ",lat_0=" + str(lat_0) +
           ",lon_0=" + str(lon_0) +
           ",ellps=WGS84")
    print "\tXSIZE:\t", xsize
    print "\tYSIZE:\t", ysize
    print "\tAREA_EXTENT:\t", area_extent
    print "};"

    if args.shapes is None:
        sys.exit(0)
    from PIL import Image
    from pycoast import ContourWriterAGG
    img = Image.new('RGB', (xsize, ysize))
    #proj4_string = '+proj=geos +lon_0=0.0 +a=6378169.00 +b=6356583.80 +h=35785831.0'
    #area_extent = (-5570248.4773392612, -5567248.074173444, 5567248.074173444, 5570248.4773392612)
    area_def = (proj4_string, area_extent)
    cw = ContourWriterAGG(args.shapes)
    #cw = ContourWriterAGG('/usr/share/gshhg-gmt-shp/')
    cw.add_coastlines(img, (proj4_string, area_extent),
                      resolution='l', width=0.5)

    cw.add_grid(img, area_def, (10.0, 10.0), (2.0, 2.0), write_text=False, outline='white', outline_opacity=175, width=1.0,
                minor_outline='white', minor_outline_opacity=175, minor_width=0.2, minor_is_tick=False)
    img.show()
