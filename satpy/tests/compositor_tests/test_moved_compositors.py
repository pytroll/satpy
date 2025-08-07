#!/usr/bin/env python
# Copyright (c) 2016-2025 Satpy developers
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

"""Tests that compositors which were moved from package init to new modules raise warnings."""

import pytest


@pytest.mark.parametrize(
    "name",
    [
        "add_alpha_bands",
        "add_bands",
        "BackgroundCompositor",
        "CategoricalDataCompositor",
        "check_times",
        "CloudCompositor",
        "ColorizeCompositor",
        "ColormapCompositor",
        "CompositeBase",
        "DayNightCompositor",
        "DifferenceCompositor",
        "enhance2dataset",
        "Filler",
        "FillingCompositor",
        "GenericCompositor",
        "HighCloudCompositor",
        "IncompatibleAreas",
        "IncompatibleTimes",
        "LongitudeMaskingCompositor",
        "LowCloudCompositor",
        "LuminanceSharpeningCompositor",
        "MaskingCompositor",
        "MultiFiller",
        "NaturalEnh",
        "PaletteCompositor",
        "RatioCompositor",
        "RatioSharpenedRGB",
        "RealisticColors",
        "RGBCompositor",
        "SandwichCompositor",
        "SelfSharpenedRGB",
        "SimpleFireMaskCompositor",
        "SingleBandCompositor",
        "StaticImageCompositor",
        "SumCompositor",
        "zero_missing_data",
    ]
)
def test_import_from_package_init_warns(name):
    """Test that compositor classes and helper functions raise warnings when imported from package."""
    from satpy import composites
    with pytest.warns(UserWarning, match="has been moved to"):
        getattr(composites, name)
