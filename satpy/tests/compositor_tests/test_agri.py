"""Tests for AGRI compositors."""

import unittest


class TestAGRIComposites(unittest.TestCase):
    """Test AGRI-specific composites."""

    def test_load_composite_yaml(self):
        """Test loading the yaml for this sensor."""
        from satpy.composites.config_loader import load_compositor_configs_for_sensors
        load_compositor_configs_for_sensors(["agri"])

    def test_simulated_red(self):
        """Test creating a fake 'red' band."""
        import dask.array as da
        import numpy as np
        import xarray as xr
        from pyresample.geometry import AreaDefinition

        from satpy.composites.agri import SimulatedRed
        rows = 5
        cols = 10
        area = AreaDefinition(
            "test", "test", "test",
            {"proj": "eqc", "lon_0": 0.0,
             "lat_0": 0.0},
            cols, rows,
            (-20037508.34, -10018754.17, 20037508.34, 10018754.17))

        comp = SimulatedRed("red", prerequisites=("C01", "C02", "C03"),
                            standard_name="toa_bidirectional_reflectance")
        c01 = xr.DataArray(da.zeros((rows, cols), chunks=25) + 0.25,
                           dims=("y", "x"),
                           attrs={"name": "C01", "area": area})
        c02 = xr.DataArray(da.zeros((rows, cols), chunks=25) + 0.30,
                           dims=("y", "x"),
                           attrs={"name": "C02", "area": area})
        res = comp((c01, c02))
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs["name"] == "red"
        assert res.attrs["standard_name"] == "toa_bidirectional_reflectance"
        data = res.compute()
        np.testing.assert_allclose(data, 0.24252874)
