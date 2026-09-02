
"""SEVIRI specific compositors."""

from __future__ import annotations

import logging

from .core import GenericCompositor, IncompatibleAreas

LOG = logging.getLogger(__name__)


class RealisticColors(GenericCompositor):
    """Create a realistic colours composite for SEVIRI."""

    def __call__(self, projectables, *args, **kwargs):
        """Generate the composite."""
        projectables = self.match_data_arrays(projectables)
        vis06 = projectables[0]
        vis08 = projectables[1]
        hrv = projectables[2]

        try:
            ch3 = 3.0 * hrv - vis06 - vis08
            ch3.attrs = hrv.attrs
        except ValueError:
            raise IncompatibleAreas("Areas do not match")

        ndvi = (vis08 - vis06) / (vis08 + vis06)
        ndvi = ndvi.where(ndvi >= 0.0, 0.0)

        ch1 = ndvi * vis06 + (1.0 - ndvi) * vis08
        ch1.attrs = vis06.attrs
        ch2 = ndvi * vis08 + (1.0 - ndvi) * vis06
        ch2.attrs = vis08.attrs

        res = super(RealisticColors, self).__call__((ch1, ch2, ch3),
                                                    *args, **kwargs)
        return res
