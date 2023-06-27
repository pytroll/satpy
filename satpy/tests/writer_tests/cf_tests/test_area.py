#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2023 Satpy developers
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
"""Tests for the CF Area."""
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyresample import AreaDefinition, SwathDefinition


class TestCFArea:
    """Test case for CF Area."""

    def test_assert_xy_unique(self):
        """Test that the x and y coordinates are unique."""
        from satpy.writers.cf.area import assert_xy_unique

        dummy = [[1, 2], [3, 4]]
        datas = {'a': xr.DataArray(data=dummy, dims=('y', 'x'), coords={'y': [1, 2], 'x': [3, 4]}),
                 'b': xr.DataArray(data=dummy, dims=('y', 'x'), coords={'y': [1, 2], 'x': [3, 4]}),
                 'n': xr.DataArray(data=dummy, dims=('v', 'w'), coords={'v': [1, 2], 'w': [3, 4]})}
        assert_xy_unique(datas)

        datas['c'] = xr.DataArray(data=dummy, dims=('y', 'x'), coords={'y': [1, 3], 'x': [3, 4]})
        with pytest.raises(ValueError):
            assert_xy_unique(datas)

    def test_link_coords(self):
        """Check that coordinates link has been established correctly."""
        from satpy.writers.cf.area import link_coords

        data = [[1, 2], [3, 4]]
        lon = np.zeros((2, 2))
        lon2 = np.zeros((1, 2, 2))
        lat = np.ones((2, 2))
        datasets = {
            'var1': xr.DataArray(data=data, dims=('y', 'x'), attrs={'coordinates': 'lon lat'}),
            'var2': xr.DataArray(data=data, dims=('y', 'x')),
            'var3': xr.DataArray(data=data, dims=('y', 'x'), attrs={'coordinates': 'lon2 lat'}),
            'var4': xr.DataArray(data=data, dims=('y', 'x'), attrs={'coordinates': 'not_exist lon lat'}),
            'lon': xr.DataArray(data=lon, dims=('y', 'x')),
            'lon2': xr.DataArray(data=lon2, dims=('time', 'y', 'x')),
            'lat': xr.DataArray(data=lat, dims=('y', 'x'))
        }

        link_coords(datasets)

        # Check that link has been established correctly and 'coordinate' atrribute has been dropped
        assert 'lon' in datasets['var1'].coords
        assert 'lat' in datasets['var1'].coords
        np.testing.assert_array_equal(datasets['var1']['lon'].data, lon)
        np.testing.assert_array_equal(datasets['var1']['lat'].data, lat)
        assert 'coordinates' not in datasets['var1'].attrs

        # There should be no link if there was no 'coordinate' attribute
        assert 'lon' not in datasets['var2'].coords
        assert 'lat' not in datasets['var2'].coords

        # The non-existent dimension or coordinate should be dropped
        assert 'time' not in datasets['var3'].coords
        assert 'not_exist' not in datasets['var4'].coords

    def test_make_alt_coords_unique(self):
        """Test that created coordinate variables are unique."""
        from satpy.writers.cf.area import make_alt_coords_unique

        data = [[1, 2], [3, 4]]
        y = [1, 2]
        x = [1, 2]
        time1 = [1, 2]
        time2 = [3, 4]
        datasets = {'var1': xr.DataArray(data=data,
                                         dims=('y', 'x'),
                                         coords={'y': y, 'x': x, 'acq_time': ('y', time1)}),
                    'var2': xr.DataArray(data=data,
                                         dims=('y', 'x'),
                                         coords={'y': y, 'x': x, 'acq_time': ('y', time2)})}

        # Test that dataset names are prepended to alternative coordinates
        res = make_alt_coords_unique(datasets)
        np.testing.assert_array_equal(res['var1']['var1_acq_time'], time1)
        np.testing.assert_array_equal(res['var2']['var2_acq_time'], time2)
        assert 'acq_time' not in res['var1'].coords
        assert 'acq_time' not in res['var2'].coords

        # Make sure nothing else is modified
        np.testing.assert_array_equal(res['var1']['x'], x)
        np.testing.assert_array_equal(res['var1']['y'], y)
        np.testing.assert_array_equal(res['var2']['x'], x)
        np.testing.assert_array_equal(res['var2']['y'], y)

        # Coords not unique -> Dataset names must be prepended, even if pretty=True
        with pytest.warns(UserWarning, match='Cannot pretty-format "acq_time"'):
            res = make_alt_coords_unique(datasets, pretty=True)
        np.testing.assert_array_equal(res['var1']['var1_acq_time'], time1)
        np.testing.assert_array_equal(res['var2']['var2_acq_time'], time2)
        assert 'acq_time' not in res['var1'].coords
        assert 'acq_time' not in res['var2'].coords

        # Coords unique and pretty=True -> Don't modify coordinate names
        datasets['var2']['acq_time'] = ('y', time1)
        res = make_alt_coords_unique(datasets, pretty=True)
        np.testing.assert_array_equal(res['var1']['acq_time'], time1)
        np.testing.assert_array_equal(res['var2']['acq_time'], time1)
        assert 'var1_acq_time' not in res['var1'].coords
        assert 'var2_acq_time' not in res['var2'].coords

    def test_area2cf(self):
        """Test the conversion of an area to CF standards."""
        from satpy.writers.cf.area import area2cf

        ds_base = xr.DataArray(data=[[1, 2], [3, 4]], dims=('y', 'x'), coords={'y': [1, 2], 'x': [3, 4]},
                               attrs={'name': 'var1'})

        # a) Area Definition and strict=False
        geos = AreaDefinition(
            area_id='geos',
            description='geos',
            proj_id='geos',
            projection={'proj': 'geos', 'h': 35785831., 'a': 6378169., 'b': 6356583.8},
            width=2, height=2,
            area_extent=[-1, -1, 1, 1])
        ds = ds_base.copy(deep=True)
        ds.attrs['area'] = geos

        res = area2cf(ds, include_lonlats=False)
        assert len(res) == 2
        assert res[0].size == 1  # grid mapping variable
        assert res[0].name == res[1].attrs['grid_mapping']

        # b) Area Definition and include_lonlats=False
        ds = ds_base.copy(deep=True)
        ds.attrs['area'] = geos
        res = area2cf(ds, include_lonlats=True)
        # same as above
        assert len(res) == 2
        assert res[0].size == 1  # grid mapping variable
        assert res[0].name == res[1].attrs['grid_mapping']
        # but now also have the lon/lats
        assert 'longitude' in res[1].coords
        assert 'latitude' in res[1].coords

        # c) Swath Definition
        swath = SwathDefinition(lons=[[1, 1], [2, 2]], lats=[[1, 2], [1, 2]])
        ds = ds_base.copy(deep=True)
        ds.attrs['area'] = swath

        res = area2cf(ds, include_lonlats=False)
        assert len(res) == 1
        assert 'longitude' in res[0].coords
        assert 'latitude' in res[0].coords
        assert 'grid_mapping' not in res[0].attrs

    def test__add_grid_mapping(self):
        """Test the conversion from pyresample area object to CF grid mapping."""
        from satpy.writers.cf.area import _add_grid_mapping

        def _gm_matches(gmapping, expected):
            """Assert that all keys in ``expected`` match the values in ``gmapping``."""
            for attr_key, attr_val in expected.attrs.items():
                test_val = gmapping.attrs[attr_key]
                if attr_val is None or isinstance(attr_val, str):
                    assert test_val == attr_val
                else:
                    np.testing.assert_almost_equal(test_val, attr_val, decimal=3)

        ds_base = xr.DataArray(data=[[1, 2], [3, 4]], dims=('y', 'x'), coords={'y': [1, 2], 'x': [3, 4]},
                               attrs={'name': 'var1'})

        # a) Projection has a corresponding CF representation (e.g. geos)
        a = 6378169.
        b = 6356583.8
        h = 35785831.
        geos = AreaDefinition(
            area_id='geos',
            description='geos',
            proj_id='geos',
            projection={'proj': 'geos', 'h': h, 'a': a, 'b': b,
                        'lat_0': 0, 'lon_0': 0},
            width=2, height=2,
            area_extent=[-1, -1, 1, 1])
        geos_expected = xr.DataArray(data=0,
                                     attrs={'perspective_point_height': h,
                                            'latitude_of_projection_origin': 0,
                                            'longitude_of_projection_origin': 0,
                                            'grid_mapping_name': 'geostationary',
                                            'semi_major_axis': a,
                                            'semi_minor_axis': b,
                                            # 'sweep_angle_axis': None,
                                            })

        ds = ds_base.copy()
        ds.attrs['area'] = geos
        new_ds, grid_mapping = _add_grid_mapping(ds)
        if 'sweep_angle_axis' in grid_mapping.attrs:
            # older versions of pyproj might not include this
            assert grid_mapping.attrs['sweep_angle_axis'] == 'y'

        assert new_ds.attrs['grid_mapping'] == 'geos'
        _gm_matches(grid_mapping, geos_expected)
        # should not have been modified
        assert 'grid_mapping' not in ds.attrs

        # b) Projection does not have a corresponding CF representation (COSMO)
        cosmo7 = AreaDefinition(
            area_id='cosmo7',
            description='cosmo7',
            proj_id='cosmo7',
            projection={'proj': 'ob_tran', 'ellps': 'WGS84', 'lat_0': 46, 'lon_0': 4.535,
                        'o_proj': 'stere', 'o_lat_p': 90, 'o_lon_p': -5.465},
            width=597, height=510,
            area_extent=[-1812933, -1003565, 814056, 1243448]
        )

        ds = ds_base.copy()
        ds.attrs['area'] = cosmo7

        new_ds, grid_mapping = _add_grid_mapping(ds)
        assert 'crs_wkt' in grid_mapping.attrs
        wkt = grid_mapping.attrs['crs_wkt']
        assert 'ELLIPSOID["WGS 84"' in wkt
        assert 'PARAMETER["lat_0",46' in wkt
        assert 'PARAMETER["lon_0",4.535' in wkt
        assert 'PARAMETER["o_lat_p",90' in wkt
        assert 'PARAMETER["o_lon_p",-5.465' in wkt
        assert new_ds.attrs['grid_mapping'] == 'cosmo7'

        # c) Projection Transverse Mercator
        lat_0 = 36.5
        lon_0 = 15.0

        tmerc = AreaDefinition(
            area_id='tmerc',
            description='tmerc',
            proj_id='tmerc',
            projection={'proj': 'tmerc', 'ellps': 'WGS84', 'lat_0': 36.5, 'lon_0': 15.0},
            width=2, height=2,
            area_extent=[-1, -1, 1, 1])

        tmerc_expected = xr.DataArray(data=0,
                                      attrs={'latitude_of_projection_origin': lat_0,
                                             'longitude_of_central_meridian': lon_0,
                                             'grid_mapping_name': 'transverse_mercator',
                                             'reference_ellipsoid_name': 'WGS 84',
                                             'false_easting': 0.,
                                             'false_northing': 0.,
                                             })

        ds = ds_base.copy()
        ds.attrs['area'] = tmerc
        new_ds, grid_mapping = _add_grid_mapping(ds)
        assert new_ds.attrs['grid_mapping'] == 'tmerc'
        _gm_matches(grid_mapping, tmerc_expected)

        # d) Projection that has a representation but no explicit a/b
        h = 35785831.
        geos = AreaDefinition(
            area_id='geos',
            description='geos',
            proj_id='geos',
            projection={'proj': 'geos', 'h': h, 'datum': 'WGS84', 'ellps': 'GRS80',
                        'lat_0': 0, 'lon_0': 0},
            width=2, height=2,
            area_extent=[-1, -1, 1, 1])
        geos_expected = xr.DataArray(data=0,
                                     attrs={'perspective_point_height': h,
                                            'latitude_of_projection_origin': 0,
                                            'longitude_of_projection_origin': 0,
                                            'grid_mapping_name': 'geostationary',
                                            # 'semi_major_axis': 6378137.0,
                                            # 'semi_minor_axis': 6356752.314,
                                            # 'sweep_angle_axis': None,
                                            })

        ds = ds_base.copy()
        ds.attrs['area'] = geos
        new_ds, grid_mapping = _add_grid_mapping(ds)

        assert new_ds.attrs['grid_mapping'] == 'geos'
        _gm_matches(grid_mapping, geos_expected)

        # e) oblique Mercator
        area = AreaDefinition(
            area_id='omerc_otf',
            description='On-the-fly omerc area',
            proj_id='omerc',
            projection={'alpha': '9.02638777018478', 'ellps': 'WGS84', 'gamma': '0', 'k': '1',
                        'lat_0': '-0.256794486098476', 'lonc': '13.7888658224205',
                        'proj': 'omerc', 'units': 'm'},
            width=2837,
            height=5940,
            area_extent=[-1460463.0893, 3455291.3877, 1538407.1158, 9615788.8787]
        )

        omerc_dict = {'azimuth_of_central_line': 9.02638777018478,
                      'false_easting': 0.,
                      'false_northing': 0.,
                      # 'gamma': 0,  # this is not CF compliant
                      'grid_mapping_name': "oblique_mercator",
                      'latitude_of_projection_origin': -0.256794486098476,
                      'longitude_of_projection_origin': 13.7888658224205,
                      # 'prime_meridian_name': "Greenwich",
                      'reference_ellipsoid_name': "WGS 84"}
        omerc_expected = xr.DataArray(data=0, attrs=omerc_dict)

        ds = ds_base.copy()
        ds.attrs['area'] = area
        new_ds, grid_mapping = _add_grid_mapping(ds)

        assert new_ds.attrs['grid_mapping'] == 'omerc_otf'
        _gm_matches(grid_mapping, omerc_expected)

        # f) Projection that has a representation but no explicit a/b
        h = 35785831.
        geos = AreaDefinition(
            area_id='geos',
            description='geos',
            proj_id='geos',
            projection={'proj': 'geos', 'h': h, 'datum': 'WGS84', 'ellps': 'GRS80',
                        'lat_0': 0, 'lon_0': 0},
            width=2, height=2,
            area_extent=[-1, -1, 1, 1])
        geos_expected = xr.DataArray(data=0,
                                     attrs={'perspective_point_height': h,
                                            'latitude_of_projection_origin': 0,
                                            'longitude_of_projection_origin': 0,
                                            'grid_mapping_name': 'geostationary',
                                            'reference_ellipsoid_name': 'WGS 84',
                                            })

        ds = ds_base.copy()
        ds.attrs['area'] = geos
        new_ds, grid_mapping = _add_grid_mapping(ds)

        assert new_ds.attrs['grid_mapping'] == 'geos'
        _gm_matches(grid_mapping, geos_expected)

    def test_add_lonlat_coords(self):
        """Test the conversion from areas to lon/lat."""
        from satpy.writers.cf.area import add_lonlat_coords

        area = AreaDefinition(
            'seviri',
            'Native SEVIRI grid',
            'geos',
            "+a=6378169.0 +h=35785831.0 +b=6356583.8 +lon_0=0 +proj=geos",
            2, 2,
            [-5570248.686685662, -5567248.28340708, 5567248.28340708, 5570248.686685662]
        )
        lons_ref, lats_ref = area.get_lonlats()
        dataarray = xr.DataArray(data=[[1, 2], [3, 4]], dims=('y', 'x'), attrs={'area': area})

        res = add_lonlat_coords(dataarray)

        # original should be unmodified
        assert 'longitude' not in dataarray.coords
        assert set(res.coords) == {'longitude', 'latitude'}
        lat = res['latitude']
        lon = res['longitude']
        np.testing.assert_array_equal(lat.data, lats_ref)
        np.testing.assert_array_equal(lon.data, lons_ref)
        assert {'name': 'latitude', 'standard_name': 'latitude', 'units': 'degrees_north'}.items() <= lat.attrs.items()
        assert {'name': 'longitude', 'standard_name': 'longitude', 'units': 'degrees_east'}.items() <= lon.attrs.items()

        area = AreaDefinition(
            'seviri',
            'Native SEVIRI grid',
            'geos',
            "+a=6378169.0 +h=35785831.0 +b=6356583.8 +lon_0=0 +proj=geos",
            10, 10,
            [-5570248.686685662, -5567248.28340708, 5567248.28340708, 5570248.686685662]
        )
        lons_ref, lats_ref = area.get_lonlats()
        dataarray = xr.DataArray(data=da.from_array(np.arange(3 * 10 * 10).reshape(3, 10, 10), chunks=(1, 5, 5)),
                                 dims=('bands', 'y', 'x'), attrs={'area': area})
        res = add_lonlat_coords(dataarray)

        # original should be unmodified
        assert 'longitude' not in dataarray.coords
        assert set(res.coords) == {'longitude', 'latitude'}
        lat = res['latitude']
        lon = res['longitude']
        np.testing.assert_array_equal(lat.data, lats_ref)
        np.testing.assert_array_equal(lon.data, lons_ref)
        assert {'name': 'latitude', 'standard_name': 'latitude', 'units': 'degrees_north'}.items() <= lat.attrs.items()
        assert {'name': 'longitude', 'standard_name': 'longitude', 'units': 'degrees_east'}.items() <= lon.attrs.items()
