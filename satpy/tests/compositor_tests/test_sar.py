"""Tests for SAR compositors."""

import unittest


class TestSARComposites(unittest.TestCase):
    """Test SAR-specific composites."""

    def test_sar_ice(self):
        """Test creating a the sar_ice composite."""
        import dask.array as da
        import numpy as np
        import xarray as xr

        from satpy.composites.sar import SARIce

        rows = 2
        cols = 2
        comp = SARIce("sar_ice", prerequisites=("hh", "hv"),
                      standard_name="sar-ice")
        hh = xr.DataArray(da.zeros((rows, cols), chunks=25) + 2000,
                          dims=("y", "x"),
                          attrs={"name": "hh"})
        hv = xr.DataArray(da.zeros((rows, cols), chunks=25) + 1000,
                          dims=("y", "x"),
                          attrs={"name": "hv"})

        res = comp((hh, hv))
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs["name"] == "sar_ice"
        assert res.attrs["standard_name"] == "sar-ice"
        data = res.compute()
        np.testing.assert_allclose(data.sel(bands="R"), 31.58280822)
        np.testing.assert_allclose(data.sel(bands="G"), 159869.56789876)
        np.testing.assert_allclose(data.sel(bands="B"), 44.68138191)

    def test_sar_ice_log(self):
        """Test creating a the sar_ice_log composite."""
        import dask.array as da
        import numpy as np
        import xarray as xr

        from satpy.composites.sar import SARIceLog

        rows = 2
        cols = 2
        comp = SARIceLog("sar_ice_log", prerequisites=("hh", "hv"),
                         standard_name="sar-ice-log")
        hh = xr.DataArray(da.zeros((rows, cols), chunks=25) - 10,
                          dims=("y", "x"),
                          attrs={"name": "hh"})
        hv = xr.DataArray(da.zeros((rows, cols), chunks=25) - 20,
                          dims=("y", "x"),
                          attrs={"name": "hv"})

        res = comp((hh, hv))
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs["name"] == "sar_ice_log"
        assert res.attrs["standard_name"] == "sar-ice-log"
        data = res.compute()
        np.testing.assert_allclose(data.sel(bands="R"), -20)
        np.testing.assert_allclose(data.sel(bands="G"), -4.6)
        np.testing.assert_allclose(data.sel(bands="B"), -10)
