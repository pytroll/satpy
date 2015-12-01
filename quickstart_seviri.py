#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2010, 2011, 2012, 2013, 2014, 2015

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   David Hoese <david.hoese@ssec.wisc.edu>
#   Esben S. Nielsen <esn@dmi.dk>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
# Fixme: fix the utf8 mess
reload(sys)
sys.setdefaultencoding('utf8')
from datetime import datetime
from mpop.scene import Scene
from mpop.resample import get_area_def
import os
if __name__ == '__main__':

    # /home/a001673/data/satellite/Meteosat-10/seviri/lvl1.5/2015/04/20/HRIT
    os.chdir("/home/a001673/data/satellite/Meteosat-10/seviri/lvl1.5/2015/04/20/HRIT")
    scn = Scene(platform_name="Meteosat-10", sensor="seviri",
                start_time=datetime(2015, 4, 20, 10, 0))

    areadef = get_area_def("eurol")
    # TODO: load by area
    scn.load([0.6, 0.8, 10.8])
    print scn
    print scn[0.6].data
    cool_channel = (scn[0.6] - scn[0.8]) * scn[10.8]

    # FIXME: this removes the previously loaded channels (since they are dependencies): it shouldn't
    scn.load(["overview"])

    # FIXME: shouldn't there be a way to get the image directly ?
    from mpop.writers import get_enhanced_image
    img = get_enhanced_image(scn["overview"])
    #img.show()

    scn.load(["natural"])
    img = get_enhanced_image(scn["natural"])
    #img.show()

    newscn = scn.resample(areadef, radius_of_influence=20000, cache_dir=True)
    img = get_enhanced_image(newscn["natural"])
    img.show()

    # TODO: how to add custom composites

    from mpop.composites import RGBCompositor
    compositor = RGBCompositor("myoverview", "bla", "")
    composite = compositor([newscn[0.6],
                            newscn[0.8],
                            newscn[10.8]])
    from mpop.writers import to_image
    img = to_image(composite)
    img.invert([False, False, True])
    img.stretch("linear")
    img.show()



