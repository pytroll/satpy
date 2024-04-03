# Copyright (c) 2018 Satpy developers
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
"""Tests for spectral correction compositors."""

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from satpy.composites.spectral import HybridGreen, NDVIHybridGreen, SpectralBlender
from satpy.tests.utils import CustomScheduler


class TestSpectralComposites:
    """Test composites for spectral channel corrections."""

    def setup_method(self):
        """Initialize channels."""
        rows = 5
        cols = 10
        self.c01 = xr.DataArray(da.zeros((rows, cols), chunks=25) + 0.20, dims=("y", "x"), attrs={"name": "C02"})
        self.c02 = xr.DataArray(da.zeros((rows, cols), chunks=25) + 0.30, dims=("y", "x"), attrs={"name": "C03"})
        self.c03 = xr.DataArray(da.zeros((rows, cols), chunks=25) + 0.40, dims=("y", "x"), attrs={"name": "C04"})

    def test_bad_lengths(self):
        """Test that error is raised if the amount of channels to blend does not match the number of weights."""
        comp = SpectralBlender("blended_channel", fractions=(0.3, 0.7), prerequisites=(0.51, 0.85),
                               standard_name="toa_bidirectional_reflectance")
        with pytest.raises(ValueError, match="fractions and projectables must have the same length."):
            comp((self.c01, self.c02, self.c03))

    def test_spectral_blender(self):
        """Test the base class for spectral blending of channels."""
        comp = SpectralBlender("blended_channel", fractions=(0.3, 0.4, 0.3), prerequisites=(0.51, 0.65, 0.85),
                               standard_name="toa_bidirectional_reflectance")
        res = comp((self.c01, self.c02, self.c03))
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs["name"] == "blended_channel"
        assert res.attrs["standard_name"] == "toa_bidirectional_reflectance"
        data = res.compute()
        np.testing.assert_allclose(data, 0.3)

    def test_hybrid_green(self):
        """Test hybrid green correction of the 'green' band."""
        comp = HybridGreen("hybrid_green", fraction=0.15, prerequisites=(0.51, 0.85),
                           standard_name="toa_bidirectional_reflectance")
        res = comp((self.c01, self.c03))
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs["name"] == "hybrid_green"
        assert res.attrs["standard_name"] == "toa_bidirectional_reflectance"
        data = res.compute()
        np.testing.assert_allclose(data, 0.23)


class TestNdviHybridGreenCompositor:
    """Test NDVI-weighted hybrid green correction of green band."""

    def setup_method(self):
        """Initialize channels."""
        coord_val = [1.0, 2.0]
        self.c01 = xr.DataArray(
            da.from_array(np.array([[0.25, 0.30], [0.20, 0.30]], dtype=np.float32), chunks=25),
            dims=("y", "x"), coords=[coord_val, coord_val], attrs={"name": "C02"})
        self.c02 = xr.DataArray(
            da.from_array(np.array([[0.25, 0.30], [0.25, 0.35]], dtype=np.float32), chunks=25),
            dims=("y", "x"), coords=[coord_val, coord_val], attrs={"name": "C03"})
        self.c03 = xr.DataArray(
            da.from_array(np.array([[0.35, 0.35], [0.28, 0.65]], dtype=np.float32), chunks=25),
            dims=("y", "x"), coords=[coord_val, coord_val], attrs={"name": "C04"})

    def test_ndvi_hybrid_green(self):
        """Test General functionality with linear scaling from ndvi to blend fraction."""
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = NDVIHybridGreen("ndvi_hybrid_green", limits=(0.15, 0.05), prerequisites=(0.51, 0.65, 0.85),
                                   standard_name="toa_bidirectional_reflectance")

            # Test General functionality with linear strength (=1.0)
            res = comp((self.c01, self.c02, self.c03))
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs["name"] == "ndvi_hybrid_green"
        assert res.attrs["standard_name"] == "toa_bidirectional_reflectance"
        data = res.values
        np.testing.assert_array_almost_equal(data, np.array([[0.2633, 0.3071], [0.2115, 0.3420]]), decimal=4)

    def test_ndvi_hybrid_green_dtype(self):
        """Test that the datatype is not altered by the compositor."""
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = NDVIHybridGreen("ndvi_hybrid_green", limits=(0.15, 0.05), prerequisites=(0.51, 0.65, 0.85),
                                   standard_name="toa_bidirectional_reflectance")
            res = comp((self.c01, self.c02, self.c03))
        assert res.data.dtype == np.float32

    def test_nonlinear_scaling(self):
        """Test non-linear scaling using `strength` term."""
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            comp = NDVIHybridGreen("ndvi_hybrid_green", limits=(0.15, 0.05), strength=2.0,
                                   prerequisites=(0.51, 0.65, 0.85),
                                   standard_name="toa_bidirectional_reflectance")
            res = comp((self.c01, self.c02, self.c03))
        res_np = res.data.compute()
        assert res.dtype == res_np.dtype
        assert res.dtype == np.float32
        np.testing.assert_array_almost_equal(res.data, np.array([[0.2646, 0.3075], [0.2120, 0.3471]]), decimal=4)

    def test_invalid_strength(self):
        """Test using invalid `strength` term for non-linear scaling."""
        with pytest.raises(ValueError, match="Expected strength greater than 0.0, got 0.0."):
            _ = NDVIHybridGreen("ndvi_hybrid_green", strength=0.0, prerequisites=(0.51, 0.65, 0.85),
                                standard_name="toa_bidirectional_reflectance")

    def test_with_slightly_mismatching_coord_input(self):
        """Test the case where an input (typically the red band) has a slightly different coordinate.

        If match_data_arrays is called correctly, the coords will be aligned and the array will have the expected shape.

        """
        comp = NDVIHybridGreen("ndvi_hybrid_green", limits=(0.15, 0.05), prerequisites=(0.51, 0.65, 0.85),
                               standard_name="toa_bidirectional_reflectance")

        c02_bad_shape = self.c02.copy()
        c02_bad_shape.coords["y"] = [1.1, 2.]
        res = comp((self.c01, c02_bad_shape, self.c03))
        assert res.shape == (2, 2)
