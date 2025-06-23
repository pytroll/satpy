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
    "BackgroundCompositor": "satpy.composites.filling",
    "DayNightCompositor": "satpy.composites.filling",
    "Filler": "satpy.composites.filling",
    "FillingCompositor": "satpy.composites.filling",
    "MultiFiller": "satpy.composites.filling",
    "add_alpha_bands": "satpy.composites.filling",
    "zero_missing_data": "satpy.composites.filling",
    "CategoricalDataCompositor": "satpy.composites.lut",
    "ColorizeCompositor": "satpy.composites.lut",
    "ColormapCompositor": "satpy.composites.lut",
    "PaletteCompositor": "satpy.composites.lut",
    "CloudCompositor": "satpy.composites.masks",
    "HighCloudCompositor": "satpy.composites.masks",
    "LongitudeMaskingCompositor": "satpy.composites.masks",
    "LowCloudCompositor": "satpy.composites.masks",
    "MaskingCompositor": "satpy.composites.masks",
    "SimpleFireMaskCompositor": "satpy.composites.masks",
    "RealisticColors": "satpy.composites.seviri",
    "LuminanceSharpeningCompositor": "satpy.composites.sharpening",
    "RatioSharpenedRGB": "satpy.composites.sharpening",
    "SandwichCompositor": "satpy.composites.sharpening",
    "SelfSharpenedRGB": "satpy.composites.sharpening",
    "NaturalEnh": "satpy.composites.spectral",
}


def __getattr__(name: str) -> Any:
    new_module = IMPORT_PATHS.get(name)

    if new_module is None:
        raise AttributeError(f"module {__name__} has no attribute '{name}'")

    return _import_and_warn_new_location(new_module, name)
