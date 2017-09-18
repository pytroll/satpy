#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author(s):

#   Pascale Roquet <pascale.roquet@meteo.fr>

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

"""Exemple script for reading MAIA cloud products files.

MAIA files can be created with AAPP with scripts MAIA4_RUN and MAIA4_RUN_AVHRR
scripts for BOTH VIIRS an AVHRR imagers

- Install satpy and pyresample
- pycoast can also be installed if you want to generate plots with coast
- Here is an example of a minimal script that reads MAIA files

MAIA files are named with a specific syntax for example:
    avhCT_M02_GL_20151012_S005503_E005802_DES_D_La-11_Lo0126_00000.h5
    viiCT_npp_DB_20121010_S132824_E132947_ASC_D_La050_Lo-012_00001.h5

References :     https://nwpsaf.eu/site/software/aapp/
      [NWPSAF-MF-UD-003] DATA Formats
      [NWPSAF-MF-UD-009] MAIA version 4 Scientific User Manual

This example uses the MAIA product which can be found in the
MAIA4_test.tgz tar test case file available with the AAPP software.


"""
from satpy import Scene
from satpy.utils import debug_on
import numpy as np
from satpy.enhancements import palettize
from satpy.composites import BWCompositor
from satpy.writers import to_image
import sys


cpool = ['#4e7791',  # not processed 0
         '#008c30',  # Cloud free land 1
         '#000000',  # Cloud free sea 2
         '#ffbbff',  # Snow over ice 3
         '#dda0dd',  # Sea ice 4
         '#ffa500',  # Very low cumuliforme 5      # non utilisé par maia
         '#ff6600',  # Very low no cumuliforme 6
         '#ffd800',  # Low cumuliforme 7           # non utilisé par maia
         '#ffa500',  # Low cumuliforme 8
         '#ffff00',  # Mid-level cumuliforme 9         # non utilisé par maia
         '#d8ff00',  # Mid-level no cumuliforme 10
         '#cccc00',  # High opaque cumuliforme 11    # on utilisé par maia
         '#d8b575',  # High opaque no cumuliforme 12
         '#ffffff',  # Very High opaque cumuliforme 13 # non utilisé par maia
         '#ffe0aa',  # Very High opaque no cumuliforme 14
         '#0000ff',  # Semi transp. thin 15
         "#00b2ff",  # Semi transp. meanly thick 16
         '#00ffe5',  # Semi transp. thick 17
         '#00ffb2',  # semi transp abobe others 18
         '#d800ff',  # fractionnal
         '#660f00',  # Not classified 20
         ]


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return [int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: " + sys.argv[0] + " MAIA_file ")
        sys.exit()

    fnmaia = sys.argv[1]
    maia_scene = Scene(reader='maia', filenames=[fnmaia])
    print(maia_scene.available_dataset_ids())
    maia_scene.load(["CloudType", "ct", "cma", "cma_conf",
                     'opaq_cloud', "CloudTopPres",
                     "CloudTopTemp", "Alt_surface"])

    # CloudType is a bit field containing the actual "ct" with values
    # from 0 to 20 which can be interpreted according to the cpool colormap

    # "ct" can be display in black and white:
    maia_scene.show("ct")

    # but it is better to palettize the image:
    # step 1: creation of the palette
    mycolors = []
    for i in range(21):
        mycolors.append(hex_to_rgb(cpool[i]))
    arr = np.array(mycolors)
    np.save("/tmp/binary_maia_ct_colormap.npy", arr)

    # step2: creation of the composite
    compositor = BWCompositor("test", standard_name="maia_ct")
    composite = compositor((maia_scene["ct"],))
    kwargs = {"palettes": [
        {"filename": "/tmp/binary_maia_ct_colormap.npy",
         "min_value": 0, "max_value": 20}]}
    img = to_image(composite)
    palettize(img, ** kwargs)
    img.show()
    img.pil_save("ct.png")
