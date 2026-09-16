"""Unit testing for the HSAF NC enhancement."""

import numpy as np
import xarray as xr

from satpy.enhancements.enhancer import Enhancer, get_enhanced_image


class TestHSAFEnhancement:
    """Test that the HSAF rain rate enhancement is correctly defined and usable."""

    def setup_method(self):
        """Create a small synthetic rain-rate dataset."""
        self.data = xr.DataArray(
            np.array([[0.0, 0.1, 0.2], [5.0, 10.0, 20.0]]),
            dims=("y", "x"),
            name="rr",
            attrs={"standard_name": "instantaneous_rain_rate", "units": "mm/h"}
        )

    def test_hsaf_rr_enhancement_colormap_applied(self):
        """Test application of the enhancement."""
        enh = Enhancer()
        enh.add_sensor_enhancements(["hsaf"])

        img = get_enhanced_image(self.data, enhance=enh)
        enhanced = img.data

        # Result must be an xarray.DataArray
        assert isinstance(enhanced, xr.DataArray)

        # Shape must be (4, y, x) because RGBA colormap applied
        assert enhanced.ndim == 3, f"Expected 3 dimensions, got {enhanced.ndim}"
        assert enhanced.shape[0] == 4, f"Expected 4 bands (RGBA), got {enhanced.shape[0]}"

        # Check transparency handling (lowest values → alpha=0)
        alpha_channel = enhanced[3, :, :]  # bands-first: 4th band = alpha
        min_alpha = alpha_channel.min().compute().item()
        max_alpha = alpha_channel.max().compute().item()
        assert min_alpha == 0, "Min. alpha should be mapped to 0"
        assert max_alpha == 1, "Max. alpha should be mapped to 1"
