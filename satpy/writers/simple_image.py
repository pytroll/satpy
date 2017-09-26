#!/usr/bin/python
# Copyright (c) 2015.
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
"""

import logging

from satpy.writers import ImageWriter

LOG = logging.getLogger(__name__)


class PillowWriter(ImageWriter):
    def __init__(self, **kwargs):
        ImageWriter.__init__(
            self,
            default_config_filename="writers/simple_image.yaml",
            **kwargs)

    def save_image(self, img, filename=None, **kwargs):
        filename = filename or self.get_filename(**img.info)

        LOG.debug("Saving to image: %s", filename)
        img.save(filename)
