#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   David Hoese <david.hoese@ssec.wisc.edu>

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

"""Base classes for composite objects.
"""

from mpop.projectable import InfoObject

class CompositeBase(InfoObject):

    def __init__(self, uid, format, prerequisites, **kwargs):
        # Required info
        kwargs["uid"] = uid
        kwargs["format"] = format
        kwargs["prerequisites"] = prerequisites.split(",")
        InfoObject.__init__(self, **kwargs)

    @property
    def prerequisites(self):
        # Semi-backward compatible
        return self.info["prerequisites"]

    def __call__(self, scene):
        raise NotImplementedError()


