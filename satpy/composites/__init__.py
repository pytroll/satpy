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

"""Composites."""

from __future__ import annotations

from typing import Any

from satpy.utils import _import_and_warn_new_location

IMPORT_PATHS = {
    "DifferenceCompositor": "satpy.composites.arithmetic",
    "RatioCompositor": "satpy.composites.arithmetic",
    "SumCompositor": "satpy.composites.arithmetic",
    "StaticImageCompositor": "satpy.composites.aux_data",
    "IncompatibleAreas": "satpy.composites.core",
    "IncompatibleTimes": "satpy.composites.core",
    "CompositeBase": "satpy.composites.core",
    "GenericCompositor": "satpy.composites.core",
    "RGBCompositor": "satpy.composites.core",
    "SingleBandCompositor": "satpy.composites.core",
    "add_bands": "satpy.composites.core",
    "check_times": "satpy.composites.core",
    "enhance2dataset": "satpy.composites.core",
    "BackgroundCompositor": "satpy.composites.fill",
    "DayNightCompositor": "satpy.composites.fill",
    "Filler": "satpy.composites.fill",
    "FillingCompositor": "satpy.composites.fill",
    "MultiFiller": "satpy.composites.fill",
    "add_alpha_bands": "satpy.composites.fill",
    "zero_missing_data": "satpy.composites.fill",
    "CategoricalDataCompositor": "satpy.composites.lookup",
    "ColorizeCompositor": "satpy.composites.lookup",
    "ColormapCompositor": "satpy.composites.lookup",
    "PaletteCompositor": "satpy.composites.lookup",
    "CloudCompositor": "satpy.composites.mask",
    "HighCloudCompositor": "satpy.composites.mask",
    "LongitudeMaskingCompositor": "satpy.composites.mask",
    "LowCloudCompositor": "satpy.composites.mask",
    "MaskingCompositor": "satpy.composites.mask",
    "SimpleFireMaskCompositor": "satpy.composites.mask",
    "RealisticColors": "satpy.composites.seviri",
    "LuminanceSharpeningCompositor": "satpy.composites.resolution",
    "RatioSharpenedRGB": "satpy.composites.resolution",
    "SandwichCompositor": "satpy.composites.resolution",
    "SelfSharpenedRGB": "satpy.composites.resolution",
    "NaturalEnh": "satpy.composites.spectral",
}


def __getattr__(name: str) -> Any:
    new_module = IMPORT_PATHS.get(name)

    if new_module is None:
        raise AttributeError(f"module {__name__} has no attribute '{name}'")

    return _import_and_warn_new_location(new_module, name)
