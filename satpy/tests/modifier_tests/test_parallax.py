# Copyright (c) 2021 Satpy developers
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

"""Tests related to parallax correction."""

import numpy as np
from pyresample.geometry import SwathDefinition


def test_parallax_correction():
    """Test parallax correction."""
    from ...modifiers.geometry import parallax_correction
    from ..utils import make_fake_scene

    sc = make_fake_scene(
            {"CTH": np.array([[np.nan, np.nan, 5000, 6000, np.nan],
                              [np.nan, 6000, 7000, 7000, 7000],
                              [np.nan, 7000, 8000, 9000, np.nan],
                              [np.nan, 7000, 7000, 7000, np.nan],
                              [np.nan, 4000, 3000, np.nan, np.nan]]),
             "IR108": np.array([[290, 290, 240, 230, 290],
                                [290, 230, 220, 220, 220],
                                [290, 220, 210, 200, 290],
                                [290, 220, 220, 220, 290],
                                [290, 250, 260, 290, 290]])},
            daskify=False,
            area=True)

    new_sc = parallax_correction(sc)
    assert new_sc.keys() == sc.keys()
    assert isinstance(new_sc["CTH"].attrs["area"], SwathDefinition)
