
"""Tests that compositors which were moved from package init to new modules raise warnings."""

import pytest


@pytest.mark.parametrize(
    "name",
    [
        "add_alpha_bands",
        "add_bands",
        "BackgroundCompositor",
        "CategoricalDataCompositor",
        "CloudCompositor",
        # 8< v1.0
        "ColorizeCompositor",
        "ColormapCompositor",
        # >8 v1.0
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
        # 8< v1.0
        "PaletteCompositor",
        # >8 v1.0
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
