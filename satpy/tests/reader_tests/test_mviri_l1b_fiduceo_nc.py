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
"""Unit tests for the FIDUCEO MVIRI FCDR Reader."""

from __future__ import annotations

import os
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition
from pyresample.utils import proj4_radius_parameters

from satpy.readers.mviri_l1b_fiduceo_nc import (
    ALTITUDE,
    EQUATOR_RADIUS,
    POLE_RADIUS,
    DatasetWrapper,
    FiduceoMviriEasyFcdrFileHandler,
    FiduceoMviriFullFcdrFileHandler,
)
from satpy.tests.utils import make_dataid

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - request

attrs_exp: dict = {
    'platform': 'MET7',
    'raw_metadata': {'foo': 'bar'},
    'sensor': 'MVIRI',
    'orbital_parameters': {
        'projection_longitude': 57.0,
        'projection_latitude': 0.0,
        'projection_altitude': 35785860.0,
        'satellite_actual_longitude': 57.1,
        'satellite_actual_latitude': 0.1,
    }

}
attrs_refl_exp = attrs_exp.copy()
attrs_refl_exp.update(
    {'sun_earth_distance_correction_applied': True,
     'sun_earth_distance_correction_factor': 1.}
)
acq_time_vis_exp = [np.datetime64('1970-01-01 00:30'),
                    np.datetime64('1970-01-01 00:30'),
                    np.datetime64('1970-01-01 02:30'),
                    np.datetime64('1970-01-01 02:30')]
vis_counts_exp = xr.DataArray(
    np.array(
        [[0., 17., 34., 51.],
         [68., 85., 102., 119.],
         [136., 153., np.nan, 187.],
         [204., 221., 238., 255]],
        dtype=np.float32
    ),
    dims=('y', 'x'),
    coords={
        'acq_time': ('y', acq_time_vis_exp),
    },
    attrs=attrs_exp
)
vis_rad_exp = xr.DataArray(
    np.array(
        [[np.nan, 18.56, 38.28, 58.],
         [77.72, 97.44, 117.16, 136.88],
         [156.6, 176.32, np.nan, 215.76],
         [235.48, 255.2, 274.92, 294.64]],
        dtype=np.float32
    ),
    dims=('y', 'x'),
    coords={
        'acq_time': ('y', acq_time_vis_exp),
    },
    attrs=attrs_exp
)
vis_refl_exp = xr.DataArray(
    np.array(
        [[np.nan, 23.440929, np.nan, np.nan],
         [40.658744, 66.602233, 147.970867, np.nan],
         [75.688217, 92.240733, np.nan, np.nan],
         [np.nan, np.nan, np.nan, np.nan]],
        dtype=np.float32
    ),
    # (0, 0) and (2, 2) are NaN because radiance is NaN
    # (0, 2) is NaN because SZA >= 90 degrees
    # Last row/col is NaN due to SZA interpolation
    dims=('y', 'x'),
    coords={
        'acq_time': ('y', acq_time_vis_exp),
    },
    attrs=attrs_refl_exp
)
u_vis_refl_exp = xr.DataArray(
    np.array(
        [[0.1, 0.2, 0.3, 0.4],
         [0.5, 0.6, 0.7, 0.8],
         [0.9, 1.0, 1.1, 1.2],
         [1.3, 1.4, 1.5, 1.6]],
        dtype=np.float32
    ),
    dims=('y', 'x'),
    coords={
        'acq_time': ('y', acq_time_vis_exp),
    },
    attrs=attrs_exp
)
acq_time_ir_wv_exp = [np.datetime64('1970-01-01 00:30'),
                      np.datetime64('1970-01-01 02:30')]
wv_counts_exp = xr.DataArray(
    np.array(
        [[0, 85],
         [170, 255]],
        dtype=np.uint8
    ),
    dims=('y', 'x'),
    coords={
        'acq_time': ('y', acq_time_ir_wv_exp),
    },
    attrs=attrs_exp
)
wv_rad_exp = xr.DataArray(
    np.array(
        [[np.nan, 3.75],
         [8, 12.25]],
        dtype=np.float32
    ),
    dims=('y', 'x'),
    coords={
        'acq_time': ('y', acq_time_ir_wv_exp),
    },
    attrs=attrs_exp
)
wv_bt_exp = xr.DataArray(
    np.array(
        [[np.nan, 230.461366],
         [252.507448, 266.863289]],
        dtype=np.float32
    ),
    dims=('y', 'x'),
    coords={
        'acq_time': ('y', acq_time_ir_wv_exp),
    },
    attrs=attrs_exp
)
ir_counts_exp = xr.DataArray(
    np.array(
        [[0, 85],
         [170, 255]],
        dtype=np.uint8
    ),
    dims=('y', 'x'),
    coords={
        'acq_time': ('y', acq_time_ir_wv_exp),
    },
    attrs=attrs_exp
)
ir_rad_exp = xr.DataArray(
    np.array(
        [[np.nan, 80],
         [165, 250]],
        dtype=np.float32
    ),
    dims=('y', 'x'),
    coords={
        'acq_time': ('y', acq_time_ir_wv_exp),
    },
    attrs=attrs_exp
)
ir_bt_exp = xr.DataArray(
    np.array(
        [[np.nan, 178.00013189],
         [204.32955838, 223.28709913]],
        dtype=np.float32
    ),
    dims=('y', 'x'),
    coords={
        'acq_time': ('y', acq_time_ir_wv_exp),
    },
    attrs=attrs_exp
)
quality_pixel_bitmask_exp = xr.DataArray(
    np.array(
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 0]],
        dtype=np.uint8
    ),
    dims=('y', 'x'),
    coords={
        'acq_time': ('y', acq_time_vis_exp),
    },
    attrs=attrs_exp
)
sza_vis_exp = xr.DataArray(
    np.array(
        [[45., 67.5, 90., np.nan],
         [22.5, 45., 67.5, np.nan],
         [0., 22.5, 45., np.nan],
         [np.nan, np.nan, np.nan, np.nan]],
        dtype=np.float32
    ),
    dims=('y', 'x'),
    attrs=attrs_exp
)
sza_ir_wv_exp = xr.DataArray(
    np.array(
        [[45, 90],
         [0, 45]],
        dtype=np.float32
    ),
    dims=('y', 'x'),
    attrs=attrs_exp
)
area_vis_exp = AreaDefinition(
    area_id='geos_mviri_4x4',
    proj_id='geos_mviri_4x4',
    description='MVIRI Geostationary Projection',
    projection={
        'proj': 'geos',
        'lon_0': 57.0,
        'h': ALTITUDE,
        'a': EQUATOR_RADIUS,
        'b': POLE_RADIUS
    },
    width=4,
    height=4,
    area_extent=[5621229.74392, 5621229.74392, -5621229.74392, -5621229.74392]
)
area_ir_wv_exp = area_vis_exp.copy(
    area_id='geos_mviri_2x2',
    proj_id='geos_mviri_2x2',
    width=2,
    height=2
)


@pytest.fixture(name='fake_dataset')
def fixture_fake_dataset():
    """Create fake dataset."""
    count_ir = da.linspace(0, 255, 4, dtype=np.uint8).reshape(2, 2)
    count_wv = da.linspace(0, 255, 4, dtype=np.uint8).reshape(2, 2)
    count_vis = da.linspace(0, 255, 16, dtype=np.uint8).reshape(4, 4)
    sza = da.from_array(
        np.array(
            [[45, 90],
             [0, 45]],
            dtype=np.float32
        )
    )
    mask = da.from_array(
        np.array(
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 1, 0],  # 1 = "invalid"
             [0, 0, 0, 0]],
            dtype=np.uint8
        )
    )
    time = np.arange(4).astype('datetime64[h]').reshape(2, 2)
    ds = xr.Dataset(
        data_vars={
            'count_vis': (('y', 'x'), count_vis),
            'count_wv': (('y_ir_wv', 'x_ir_wv'), count_wv),
            'count_ir': (('y_ir_wv', 'x_ir_wv'), count_ir),
            'toa_bidirectional_reflectance_vis': vis_refl_exp / 100,
            'u_independent_toa_bidirectional_reflectance': u_vis_refl_exp / 100,
            'quality_pixel_bitmask': (('y', 'x'), mask),
            'solar_zenith_angle': (('y_tie', 'x_tie'), sza),
            'time_ir_wv': (('y_ir_wv', 'x_ir_wv'), time),
            'a_ir': -5.0,
            'b_ir': 1.0,
            'bt_a_ir': 10.0,
            'bt_b_ir': -1000.0,
            'a_wv': -0.5,
            'b_wv': 0.05,
            'bt_a_wv': 10.0,
            'bt_b_wv': -2000.0,
            'years_since_launch': 20.0,
            'a0_vis': 1.0,
            'a1_vis': 0.01,
            'a2_vis': -0.0001,
            'mean_count_space_vis': 1.0,
            'distance_sun_earth': 1.0,
            'solar_irradiance_vis': 650.0,
            'sub_satellite_longitude_start': 57.1,
            'sub_satellite_longitude_end': np.nan,
            'sub_satellite_latitude_start': np.nan,
            'sub_satellite_latitude_end': 0.1,
        },
        coords={
            'y': [1, 2, 3, 4],
            'x': [1, 2, 3, 4],
            'y_ir_wv': [1, 2],
            'x_ir_wv': [1, 2],
            'y_tie': [1, 2],
            'x_tie': [1, 2]

        },
        attrs={'foo': 'bar'}
    )
    ds['count_ir'].attrs['ancillary_variables'] = 'a_ir b_ir'
    ds['count_wv'].attrs['ancillary_variables'] = 'a_wv b_wv'
    return ds


@pytest.fixture(
    name='file_handler',
    params=[FiduceoMviriEasyFcdrFileHandler,
            FiduceoMviriFullFcdrFileHandler]
)
def fixture_file_handler(fake_dataset, request):
    """Create mocked file handler."""
    marker = request.node.get_closest_marker("file_handler_data")
    mask_bad_quality = True
    if marker:
        mask_bad_quality = marker.kwargs['mask_bad_quality']
    fh_class = request.param
    with mock.patch('satpy.readers.mviri_l1b_fiduceo_nc.xr.open_dataset') as open_dataset:
        open_dataset.return_value = fake_dataset
        return fh_class(
            filename='filename',
            filename_info={'platform': 'MET7',
                           'sensor': 'MVIRI',
                           'projection_longitude': '57.0'},
            filetype_info={'foo': 'bar'},
            mask_bad_quality=mask_bad_quality
        )


@pytest.fixture(name='reader')
def fixture_reader():
    """Return MVIRI FIDUCEO FCDR reader."""
    from satpy._config import config_search_paths
    from satpy.readers import load_reader

    reader_configs = config_search_paths(
        os.path.join("readers", "mviri_l1b_fiduceo_nc.yaml"))
    reader = load_reader(reader_configs)
    return reader


class TestFiduceoMviriFileHandlers:
    """Unit tests for FIDUCEO MVIRI file handlers."""

    def test_init(self, file_handler):
        """Test file handler initialization."""
        assert file_handler.projection_longitude == 57.0
        assert file_handler.mask_bad_quality is True

    @pytest.mark.parametrize(
        ('name', 'calibration', 'resolution', 'expected'),
        [
            ('VIS', 'counts', 2250, vis_counts_exp),
            ('VIS', 'radiance', 2250, vis_rad_exp),
            ('VIS', 'reflectance', 2250, vis_refl_exp),
            ('WV', 'counts', 4500, wv_counts_exp),
            ('WV', 'radiance', 4500, wv_rad_exp),
            ('WV', 'brightness_temperature', 4500, wv_bt_exp),
            ('IR', 'counts', 4500, ir_counts_exp),
            ('IR', 'radiance', 4500, ir_rad_exp),
            ('IR', 'brightness_temperature', 4500, ir_bt_exp),
            ('quality_pixel_bitmask', None, 2250, quality_pixel_bitmask_exp),
            ('solar_zenith_angle', None, 2250, sza_vis_exp),
            ('solar_zenith_angle', None, 4500, sza_ir_wv_exp),
            ('u_independent_toa_bidirectional_reflectance', None, 4500, u_vis_refl_exp)
        ]
    )
    def test_get_dataset(self, file_handler, name, calibration, resolution,
                         expected):
        """Test getting datasets."""
        id_keys = {'name': name, 'resolution': resolution}
        if calibration:
            id_keys['calibration'] = calibration
        dataset_id = make_dataid(**id_keys)
        dataset_info = {'platform': 'MET7'}

        is_easy = isinstance(file_handler, FiduceoMviriEasyFcdrFileHandler)
        is_vis = name == 'VIS'
        is_refl = calibration == 'reflectance'
        if is_easy and is_vis and not is_refl:
            # VIS counts/radiance not available in easy FCDR
            with pytest.raises(ValueError):
                file_handler.get_dataset(dataset_id, dataset_info)
        else:
            ds = file_handler.get_dataset(dataset_id, dataset_info)
            xr.testing.assert_allclose(ds, expected)
            assert ds.dtype == expected.dtype
            assert ds.attrs == expected.attrs

    def test_get_dataset_corrupt(self, file_handler):
        """Test getting datasets with known corruptions."""
        # Time may have different names and satellite position might be missing
        file_handler.nc.nc = file_handler.nc.nc.rename(
            {'time_ir_wv': 'time'}
        )
        file_handler.nc.nc = file_handler.nc.nc.drop_vars(
            ['sub_satellite_longitude_start']
        )

        dataset_id = make_dataid(
            name='VIS',
            calibration='reflectance',
            resolution=2250
        )
        ds = file_handler.get_dataset(dataset_id, {'platform': 'MET7'})
        assert 'actual_satellite_longitude' not in ds.attrs['orbital_parameters']
        assert 'actual_satellite_latitude' not in ds.attrs['orbital_parameters']
        xr.testing.assert_allclose(ds, vis_refl_exp)

    @mock.patch(
        'satpy.readers.mviri_l1b_fiduceo_nc.Interpolator.interp_acq_time'
    )
    def test_time_cache(self, interp_acq_time, file_handler):
        """Test caching of acquisition times."""
        dataset_id = make_dataid(
            name='VIS',
            resolution=2250,
            calibration='reflectance'
        )
        info = {}
        interp_acq_time.return_value = xr.DataArray([1, 2, 3, 4], dims='y')

        # Cache init
        file_handler.get_dataset(dataset_id, info)
        interp_acq_time.assert_called()

        # Cache hit
        interp_acq_time.reset_mock()
        file_handler.get_dataset(dataset_id, info)
        interp_acq_time.assert_not_called()

        # Cache miss
        interp_acq_time.return_value = xr.DataArray([1, 2], dims='y')
        another_id = make_dataid(
            name='IR',
            resolution=4500,
            calibration='brightness_temperature'
        )
        interp_acq_time.reset_mock()
        file_handler.get_dataset(another_id, info)
        interp_acq_time.assert_called()

    @mock.patch(
        'satpy.readers.mviri_l1b_fiduceo_nc.Interpolator.interp_tiepoints'
    )
    def test_angle_cache(self, interp_tiepoints, file_handler):
        """Test caching of angle datasets."""
        dataset_id = make_dataid(name='solar_zenith_angle',
                                 resolution=2250)
        info = {}

        # Cache init
        file_handler.get_dataset(dataset_id, info)
        interp_tiepoints.assert_called()

        # Cache hit
        interp_tiepoints.reset_mock()
        file_handler.get_dataset(dataset_id, info)
        interp_tiepoints.assert_not_called()

        # Cache miss
        another_id = make_dataid(name='solar_zenith_angle',
                                 resolution=4500)
        interp_tiepoints.reset_mock()
        file_handler.get_dataset(another_id, info)
        interp_tiepoints.assert_called()

    @pytest.mark.parametrize(
        ('name', 'resolution', 'area_exp'),
        [
            ('VIS', 2250, area_vis_exp),
            ('WV', 4500, area_ir_wv_exp),
            ('IR', 4500, area_ir_wv_exp),
            ('quality_pixel_bitmask', 2250, area_vis_exp),
            ('solar_zenith_angle', 2250, area_vis_exp),
            ('solar_zenith_angle', 4500, area_ir_wv_exp)
        ]
    )
    def test_get_area_definition(self, file_handler, name, resolution,
                                 area_exp):
        """Test getting area definitions."""
        dataset_id = make_dataid(name=name, resolution=resolution)
        area = file_handler.get_area_def(dataset_id)
        a, b = proj4_radius_parameters(area.proj_dict)
        a_exp, b_exp = proj4_radius_parameters(area_exp.proj_dict)
        assert a == a_exp
        assert b == b_exp
        assert area.width == area_exp.width
        assert area.height == area_exp.height
        for key in ['h', 'lon_0', 'proj', 'units']:
            assert area.proj_dict[key] == area_exp.proj_dict[key]
        np.testing.assert_allclose(area.area_extent, area_exp.area_extent)

    def test_calib_exceptions(self, file_handler):
        """Test calibration exceptions."""
        with pytest.raises(KeyError):
            file_handler.get_dataset(
                make_dataid(name='solar_zenith_angle', calibration='counts'),
                {}
            )
        with pytest.raises(KeyError):
            file_handler.get_dataset(
                make_dataid(
                    name='VIS',
                    resolution=2250,
                    calibration='brightness_temperature'),
                {}
            )
        with pytest.raises(KeyError):
            file_handler.get_dataset(
                make_dataid(
                    name='IR',
                    resolution=4500,
                    calibration='reflectance'),
                {}
            )
        if isinstance(file_handler, FiduceoMviriEasyFcdrFileHandler):
            with pytest.raises(KeyError):
                file_handler.get_dataset(
                    {'name': 'VIS', 'calibration': 'counts'},
                    {}
                )  # not available in easy FCDR

    @pytest.mark.file_handler_data(mask_bad_quality=False)
    def test_bad_quality_warning(self, file_handler):
        """Test warning about bad VIS quality."""
        file_handler.nc.nc['quality_pixel_bitmask'] = 2
        vis = make_dataid(name='VIS', resolution=2250,
                          calibration='reflectance')
        with pytest.warns(UserWarning):
            file_handler.get_dataset(vis, {})

    def test_file_pattern(self, reader):
        """Test file pattern matching."""
        filenames = [
            "FIDUCEO_FCDR_L15_MVIRI_MET7-57.0_201701201000_201701201030_FULL_v2.6_fv3.1.nc",
            "FIDUCEO_FCDR_L15_MVIRI_MET7-57.0_201701201000_201701201030_EASY_v2.6_fv3.1.nc",
            "FIDUCEO_FCDR_L15_MVIRI_MET7-00.0_201701201000_201701201030_EASY_v2.6_fv3.1.nc",
            "abcde",
        ]

        files = reader.select_files_from_pathnames(filenames)
        # only 3 out of 4 above should match
        assert len(files) == 3


class TestDatasetWrapper:
    """Unit tests for DatasetWrapper class."""

    def test_reassign_coords(self):
        """Test reassigning of coordinates.

        For some reason xarray does not always assign (y, x) coordinates to
        the high resolution datasets, although they have dimensions (y, x) and
        coordinates y and x exist. A dataset with these properties seems
        impossible to create (neither dropping, resetting or deleting
        coordinates seems to work). Instead use mock as a workaround.
        """
        nc = mock.MagicMock(
            coords={
                'y': [.1, .2],
                'x': [.3, .4]
            },
            dims=('y', 'x')
        )
        nc.__getitem__.return_value = xr.DataArray(
            [[1, 2],
             [3, 4]],
            dims=('y', 'x')
        )
        foo_exp = xr.DataArray(
            [[1, 2],
             [3, 4]],
            dims=('y', 'x'),
            coords={
                'y': [.1, .2],
                'x': [.3, .4]
            }
        )
        ds = DatasetWrapper(nc)
        foo = ds['foo']
        xr.testing.assert_equal(foo, foo_exp)
