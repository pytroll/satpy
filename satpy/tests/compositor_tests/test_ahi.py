"""Tests for AHI compositors."""

import unittest


class TestAHIComposites(unittest.TestCase):
    """Test AHI-specific composites."""

    def test_load_composite_yaml(self):
        """Test loading the yaml for this sensor."""
        from satpy.composites.config_loader import load_compositor_configs_for_sensors
        load_compositor_configs_for_sensors(["ahi"])
