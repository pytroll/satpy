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

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from satpy.composites.spectral import GreenCorrector, HybridGreen, NDVIHybridGreen, SpectralBlender


class TestSpectralComposites:
    """Test composites for spectral channel corrections."""

    def setup_method(self):
        """Initialize channels."""
        rows = 5
        cols = 10
        self.c01 = xr.DataArray(da.zeros((rows, cols), chunks=25) + 0.20, dims=('y', 'x'), attrs={'name': 'C02'})
        self.c02 = xr.DataArray(da.zeros((rows, cols), chunks=25) + 0.30, dims=('y', 'x'), attrs={'name': 'C03'})
        self.c03 = xr.DataArray(da.zeros((rows, cols), chunks=25) + 0.40, dims=('y', 'x'), attrs={'name': 'C04'})

    def test_bad_lengths(self):
        """Test that error is raised if the amount of channels to blend does not match the number of weights."""
        comp = SpectralBlender('blended_channel', fractions=(0.3, 0.7), prerequisites=(0.51, 0.85),
                               standard_name='toa_bidirectional_reflectance')
        with pytest.raises(ValueError):
            comp((self.c01, self.c02, self.c03))

    def test_spectral_blender(self):
        """Test the base class for spectral blending of channels."""
        comp = SpectralBlender('blended_channel', fractions=(0.3, 0.4, 0.3), prerequisites=(0.51, 0.65, 0.85),
                               standard_name='toa_bidirectional_reflectance')
        res = comp((self.c01, self.c02, self.c03))
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs['name'] == 'blended_channel'
        assert res.attrs['standard_name'] == 'toa_bidirectional_reflectance'
        data = res.compute()
        np.testing.assert_allclose(data, 0.3)

    def test_hybrid_green(self):
        """Test hybrid green correction of the 'green' band."""
        comp = HybridGreen('hybrid_green', fraction=0.15, prerequisites=(0.51, 0.85),
                           standard_name='toa_bidirectional_reflectance')
        res = comp((self.c01, self.c03))
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs['name'] == 'hybrid_green'
        assert res.attrs['standard_name'] == 'toa_bidirectional_reflectance'
        data = res.compute()
        np.testing.assert_allclose(data, 0.23)

    def test_ndvi_hybrid_green(self):
        """Test NDVI-scaled hybrid green correction of 'green' band."""
        self.c01 = xr.DataArray(da.from_array([[0.25, 0.30], [0.20, 0.30]], chunks=25),
                                dims=('y', 'x'), attrs={'name': 'C02'})
        self.c02 = xr.DataArray(da.from_array([[0.25, 0.30], [0.25, 0.35]], chunks=25),
                                dims=('y', 'x'), attrs={'name': 'C03'})
        self.c03 = xr.DataArray(da.from_array([[0.35, 0.35], [0.28, 0.65]], chunks=25),
                                dims=('y', 'x'), attrs={'name': 'C04'})

        comp = NDVIHybridGreen('ndvi_hybrid_green', limits=(0.15, 0.05), prerequisites=(0.51, 0.65, 0.85),
                               standard_name='toa_bidirectional_reflectance')

        res = comp((self.c01, self.c02, self.c03))
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs['name'] == 'ndvi_hybrid_green'
        assert res.attrs['standard_name'] == 'toa_bidirectional_reflectance'
        data = res.values
        np.testing.assert_array_almost_equal(data, np.array([[0.2633, 0.3071], [0.2115, 0.3420]]), decimal=4)

    def test_green_corrector(self):
        """Test the deprecated class for green corrections."""
        comp = GreenCorrector('blended_channel', fractions=(0.85, 0.15), prerequisites=(0.51, 0.85),
                              standard_name='toa_bidirectional_reflectance')
        res = comp((self.c01, self.c03))
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs['name'] == 'blended_channel'
        assert res.attrs['standard_name'] == 'toa_bidirectional_reflectance'
        data = res.compute()
        np.testing.assert_allclose(data, 0.23)
