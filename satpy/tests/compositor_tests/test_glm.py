#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Satpy developers
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
"""Tests for GLM compositors."""


class TestGLMComposites:
    """Test GLM-specific composites."""

    def test_load_composite_yaml(self):
        """Test loading the yaml for this sensor."""
        from satpy.composites.config_loader import load_compositor_configs_for_sensors
        load_compositor_configs_for_sensors(['glm'])

    def test_highlight_compositor(self):
        """Test creating a highlight composite."""
        import dask.array as da
        import numpy as np
        import xarray as xr
        from pyresample.geometry import AreaDefinition

        from satpy.composites.glm import HighlightCompositor
        rows = 5
        cols = 10
        area = AreaDefinition(
            'test', 'test', 'test',
            {'proj': 'eqc', 'lon_0': 0.0,
             'lat_0': 0.0},
            cols, rows,
            (-20037508.34, -10018754.17, 20037508.34, 10018754.17))

        comp = HighlightCompositor(
            'c14_highlight',
            prerequisites=('flash_extent_density', 'C14'),
            min_hightlight=0.0,
            max_hightlight=1.0,
        )
        flash_extent_density = xr.DataArray(
            da.zeros((rows, cols), chunks=25) + 0.5,
            dims=('y', 'x'),
            attrs={'name': 'flash_extent_density', 'area': area})
        c14_data = np.repeat(np.arange(cols, dtype=np.float64)[None, :], rows, axis=0)
        c14 = xr.DataArray(da.from_array(c14_data, chunks=25) + 303.15,
                           dims=('y', 'x'),
                           attrs={
                               'name': 'C14',
                               'area': area,
                               'standard_name': 'toa_brightness_temperature',
                           })
        res = comp((flash_extent_density, c14))
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs['name'] == 'c14_highlight'
        data = res.compute()
        np.testing.assert_almost_equal(data.values.min(), -0.04)
        np.testing.assert_almost_equal(data.values.max(), 1.04)
