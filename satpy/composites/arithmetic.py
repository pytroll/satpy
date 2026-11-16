
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
        projectables = self.match_data_arrays(projectables,
                                              drop_coordinates=False)
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
        projectables = self.match_data_arrays(projectables,
                                              drop_coordinates=False)
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
        projectables = self.match_data_arrays(projectables,
                                              drop_coordinates=False)
        info = combine_metadata(*projectables)
        info["name"] = self.attrs["name"]

        proj = projectables[0] + projectables[1]
        proj.attrs = info
        return proj
