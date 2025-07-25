# Copyright (c) 2015-2025 Satpy developers
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

"""Compositors performing arithmetic operations."""

from __future__ import annotations

import logging

from satpy.dataset import combine_metadata

from .core import CompositeBase

LOG = logging.getLogger(__name__)


class DifferenceCompositor(CompositeBase):
    """Make the difference of two data arrays."""

    def __call__(self, projectables, nonprojectables=None, **attrs):
        """Generate the composite."""
        if len(projectables) != 2:
            raise ValueError("Expected 2 datasets, got %d" % (len(projectables),))
        projectables = self.match_data_arrays(projectables)
        info = combine_metadata(*projectables)
        info["name"] = self.attrs["name"]
        info.update(self.attrs)  # attrs from YAML/__init__
        info.update(attrs)  # overwriting of DataID properties

        proj = projectables[0] - projectables[1]
        proj.attrs = info
        return proj


class RatioCompositor(CompositeBase):
    """Make the ratio of two data arrays."""

    def __call__(self, projectables, nonprojectables=None, **info):
        """Generate the composite."""
        if len(projectables) != 2:
            raise ValueError("Expected 2 datasets, got %d" % (len(projectables),))
        projectables = self.match_data_arrays(projectables)
        info = combine_metadata(*projectables)
        info.update(self.attrs)

        proj = projectables[0] / projectables[1]
        proj.attrs = info
        return proj


class SumCompositor(CompositeBase):
    """Make the sum of two data arrays."""

    def __call__(self, projectables, nonprojectables=None, **info):
        """Generate the composite."""
        if len(projectables) != 2:
            raise ValueError("Expected 2 datasets, got %d" % (len(projectables),))
        projectables = self.match_data_arrays(projectables)
        info = combine_metadata(*projectables)
        info["name"] = self.attrs["name"]

        proj = projectables[0] + projectables[1]
        proj.attrs = info
        return proj
