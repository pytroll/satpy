#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018, 2022 Satpy developers
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
"""Tests for VIIRS compositors."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition


class TestVIIRSComposites:
    """Test various VIIRS-specific composites."""

    @pytest.fixture
    def area(self):
        """Return fake area for use with DNB tests."""
        rows = 5
        cols = 10
        area = AreaDefinition(
            'test', 'test', 'test',
            {'proj': 'eqc', 'lon_0': 0.0,
             'lat_0': 0.0},
            cols, rows,
            (-20037508.34, -10018754.17, 20037508.34, 10018754.17))
        return area

    @pytest.fixture
    def c01(self, area):
        """Return fake channel 1 data for DNB tests."""
        dnb = np.zeros(area.shape) + 0.25
        dnb[3, :] += 0.25
        dnb[4:, :] += 0.5
        dnb = da.from_array(dnb, chunks=25)
        c01 = xr.DataArray(dnb,
                           dims=('y', 'x'),
                           attrs={'name': 'DNB', 'area': area})
        return c01

    @pytest.fixture
    def c02(self, area):
        """Return fake sza dataset for DNB tests."""
        # data changes by row, sza changes by col for testing
        sza = np.zeros(area.shape) + 70.0
        sza[:, 3] += 20.0
        sza[:, 4:] += 45.0
        sza = da.from_array(sza, chunks=25)
        c02 = xr.DataArray(sza,
                           dims=('y', 'x'),
                           attrs={'name': 'solar_zenith_angle', 'area': area})
        return c02

    @pytest.fixture
    def c03(self, area):
        """Return fake lunal zenith angle dataset for DNB tests."""
        lza = np.zeros(area.shape) + 70.0
        lza[:, 3] += 20.0
        lza[:, 4:] += 45.0
        lza = da.from_array(lza, chunks=25)
        c03 = xr.DataArray(lza,
                           dims=('y', 'x'),
                           attrs={'name': 'lunar_zenith_angle', 'area': area})
        return c03

    def test_load_composite_yaml(self):
        """Test loading the yaml for this sensor."""
        from satpy.composites.config_loader import load_compositor_configs_for_sensors
        load_compositor_configs_for_sensors(['viirs'])

    def test_histogram_dnb(self, c01, c02):
        """Test the 'histogram_dnb' compositor."""
        from satpy.composites.viirs import HistogramDNB

        comp = HistogramDNB('histogram_dnb', prerequisites=('dnb',),
                            standard_name='toa_outgoing_radiance_per_'
                                          'unit_wavelength')
        res = comp((c01, c02))
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs['name'] == 'histogram_dnb'
        assert res.attrs['standard_name'] == 'equalized_radiance'
        data = res.compute()
        unique_values = np.unique(data)
        np.testing.assert_allclose(unique_values, [0.5994, 0.7992, 0.999], rtol=1e-3)

    def test_adaptive_dnb(self, c01, c02):
        """Test the 'adaptive_dnb' compositor."""
        from satpy.composites.viirs import AdaptiveDNB

        comp = AdaptiveDNB('adaptive_dnb', prerequisites=('dnb',),
                           standard_name='toa_outgoing_radiance_per_'
                                         'unit_wavelength')
        res = comp((c01, c02))
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs['name'] == 'adaptive_dnb'
        assert res.attrs['standard_name'] == 'equalized_radiance'
        data = res.compute()
        np.testing.assert_allclose(data.data, 0.999, rtol=1e-4)

    def test_hncc_dnb(self, area, c01, c02, c03):
        """Test the 'hncc_dnb' compositor."""
        from satpy.composites.viirs import NCCZinke

        comp = NCCZinke('hncc_dnb', prerequisites=('dnb',),
                        standard_name='toa_outgoing_radiance_per_'
                                      'unit_wavelength')
        mif = xr.DataArray(da.zeros((5,), chunks=5) + 0.1,
                           dims=('y',),
                           attrs={'name': 'moon_illumination_fraction', 'area': area})
        res = comp((c01, c02, c03, mif))
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs['name'] == 'hncc_dnb'
        assert res.attrs['standard_name'] == 'ncc_radiance'
        data = res.compute()
        unique = np.unique(data)
        np.testing.assert_allclose(
            unique, [3.48479712e-04, 6.96955799e-04, 1.04543189e-03, 4.75394738e-03,
                     9.50784532e-03, 1.42617433e-02, 1.50001560e+03, 3.00001560e+03,
                     4.50001560e+03])

    @pytest.mark.parametrize("dnb_units", ["W m-2 sr-1", "W cm-2 sr-1"])
    @pytest.mark.parametrize("saturation_correction", [False, True])
    def test_erf_dnb(self, dnb_units, saturation_correction, area, c02, c03):
        """Test the 'dynamic_dnb' or ERF DNB compositor."""
        from satpy.composites.viirs import ERFDNB

        comp = ERFDNB('dynamic_dnb', prerequisites=('dnb',),
                      saturation_correction=saturation_correction,
                      standard_name='toa_outgoing_radiance_per_'
                                    'unit_wavelength')
        # c01 is different from in the other tests, so don't use the fixture
        # here
        dnb = np.zeros(area.shape) + 0.25
        cols = area.shape[1]
        dnb[2, :cols // 2] = np.nan
        dnb[3, :] += 0.25
        dnb[4:, :] += 0.5
        if dnb_units == "W cm-2 sr-1":
            dnb /= 10000.0
        dnb = da.from_array(dnb, chunks=25)
        c01 = xr.DataArray(dnb,
                           dims=('y', 'x'),
                           attrs={'name': 'DNB', 'area': area, 'units': dnb_units})
        mif = xr.DataArray(da.zeros((5,), chunks=5) + 0.1,
                           dims=('y',),
                           attrs={'name': 'moon_illumination_fraction', 'area': area})
        res = comp((c01, c02, c03, mif))
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs['name'] == 'dynamic_dnb'
        assert res.attrs['standard_name'] == 'equalized_radiance'
        data = res.compute()
        unique = np.unique(data)
        assert np.isnan(unique).any()
        nonnan_unique = unique[~np.isnan(unique)]
        if saturation_correction:
            exp_unique = [0.000000e+00, 3.978305e-04, 6.500003e-04,
                          8.286927e-04, 5.628335e-01, 7.959671e-01,
                          9.748567e-01]
        else:
            exp_unique = [0.00000000e+00, 1.00446703e-01, 1.64116082e-01,
                          2.09233451e-01, 1.43916324e+02, 2.03528498e+02,
                          2.49270516e+02]
            np.testing.assert_allclose(nonnan_unique, exp_unique)

    def test_snow_age(self, area):
        """Test the 'snow_age' compositor."""
        from satpy.composites.viirs import SnowAge

        projectables = tuple(
           xr.DataArray(
                da.from_array(np.full(area.shape, 5.*i), chunks=5),
                dims=("y", "x"),
                attrs={"name": f"M0{i:d}",
                       "calibration": "reflectance",
                       "units": "%"})
           for i in range(7, 12))
        comp = SnowAge(
                "snow_age",
                prerequisites=("M07", "M08", "M09", "M10", "M11",),
                standard_name="snow_age")
        res = comp(projectables)
        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs["name"] == "snow_age"
        assert "units" not in res.attrs
