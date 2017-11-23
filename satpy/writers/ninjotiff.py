#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017.

# Author(s):

#   David Hoese <david.hoese@ssec.wisc.edu>
#   Martin Raspaud <martin.raspaud@smhi.se>


# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""GeoTIFF writer objects for creating GeoTIFF files from `Dataset` objects.

"""

import logging

import numpy as np

import pyninjotiff.ninjotiff as nt
from satpy.utils import ensure_dir
from satpy.writers import ImageWriter

LOG = logging.getLogger(__name__)


class NinjoTIFFWriter(ImageWriter):
    GDAL_OPTIONS = ("tfw",
                    "rpb",
                    "rpctxt",
                    "interleave",
                    "tiled",
                    "blockxsize",
                    "blockysize",
                    "nbits",
                    "compress",
                    "num_threads",
                    "predictor",
                    "discard_lsb",
                    "sparse_ok",
                    "jpeg_quality",
                    "jpegtablesmode",
                    "zlevel",
                    "photometric",
                    "alpha",
                    "profile",
                    "bigtiff",
                    "pixeltype",
                    "copy_src_overviews", )

    def __init__(self, floating_point=False, tags=None, **kwargs):
        ImageWriter.__init__(self,
                             default_config_filename="writers/ninjotiff.yaml",
                             **kwargs)

        # self.floating_point = bool(self.config_options.get(
        #     "floating_point", None) if floating_point is None else
        #     floating_point)
        self.tags = self.info.get("tags",
                                  None) if tags is None else tags
        if self.tags is None:
            self.tags = {}
        elif not isinstance(self.tags, dict):
            # if it's coming from a config file
            self.tags = dict(tuple(x.split("=")) for x in self.tags.split(","))

    def save_image(self, img, filename=None, **kwargs):  # floating_point=False,
        """Save the image to the given *filename* in ninjotiff_ format.

        .. _ninjotiff: http://www.ssec.wisc.edu/~davidh/polar2grid/misc/NinJo_Satellite_Import_Formats.html
        """

        filename = filename or self.get_filename(**img.info)
        nt.save(img, filename, **kwargs)
