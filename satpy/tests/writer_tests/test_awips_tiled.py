#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2018 Satpy developers
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
"""Tests for the AWIPS Tiled writer."""

import logging
import os
import shutil
from datetime import datetime, timedelta
from glob import glob

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from satpy.resample import update_resampled_coords

START_TIME = datetime(2018, 1, 1, 12, 0, 0)
END_TIME = START_TIME + timedelta(minutes=20)

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path
# - caplog


def _check_production_location(ds):
    if 'production_site' in ds.attrs:
        prod_loc_name = 'production_site'
    elif 'production_location' in ds.attrs:
        prod_loc_name = 'producton_location'
    else:
        return

    if prod_loc_name in ds.attrs:
        assert len(ds.attrs[prod_loc_name]) == 31


def check_required_properties(unmasked_ds, masked_ds):
    """Check various aspects of coordinates and attributes for correctness."""
    _check_scaled_x_coordinate_variable(unmasked_ds, masked_ds)
    _check_scaled_y_coordinate_variable(unmasked_ds, masked_ds)
    _check_required_common_attributes(unmasked_ds)


def _check_required_common_attributes(ds):
    """Check common properties of the created AWIPS tiles for validity."""
    for attr_name in ('tile_row_offset', 'tile_column_offset',
                      'product_tile_height', 'product_tile_width',
                      'number_product_tiles',
                      'product_rows', 'product_columns'):
        assert attr_name in ds.attrs
    _check_production_location(ds)

    for data_arr in ds.data_vars.values():
        if data_arr.ndim == 0:
            # grid mapping variable
            assert 'grid_mapping_name' in data_arr.attrs
            continue
        assert data_arr.encoding.get('zlib', False)
        assert 'grid_mapping' in data_arr.attrs
        assert data_arr.attrs['grid_mapping'] in ds
        assert 'units' in data_arr.attrs
        if data_arr.name != "DQF":
            assert data_arr.dtype == np.int16
            assert data_arr.attrs["_Unsigned"] == "true"


def _check_scaled_x_coordinate_variable(ds, masked_ds):
    assert 'x' in ds.coords
    x_coord = ds.coords['x']
    np.testing.assert_equal(np.diff(x_coord), 1)
    x_attrs = x_coord.attrs
    assert x_attrs.get('standard_name') == 'projection_x_coordinate'
    assert x_attrs.get('units') == 'meters'
    assert 'scale_factor' in x_attrs
    assert x_attrs['scale_factor'] > 0
    assert 'add_offset' in x_attrs

    unscaled_x = masked_ds.coords['x'].values
    assert (np.diff(unscaled_x) > 0).all()


def _check_scaled_y_coordinate_variable(ds, masked_ds):
    assert 'y' in ds.coords
    y_coord = ds.coords['y']
    np.testing.assert_equal(np.diff(y_coord), 1)
    y_attrs = y_coord.attrs
    assert y_attrs.get('standard_name') == 'projection_y_coordinate'
    assert y_attrs.get('units') == 'meters'
    assert 'scale_factor' in y_attrs
    assert y_attrs['scale_factor'] < 0
    assert 'add_offset' in y_attrs

    unscaled_y = masked_ds.coords['y'].values
    assert (np.diff(unscaled_y) < 0).all()


def _get_test_area(shape=(200, 100), crs=None, extents=None):
    from pyresample.geometry import AreaDefinition
    if crs is None:
        crs = CRS('+proj=lcc +datum=WGS84 +ellps=WGS84 +lon_0=-95. +lat_0=25 +lat_1=25 +units=m +no_defs')
    if extents is None:
        extents = (-1000., -1500., 1000., 1500.)
    area_def = AreaDefinition(
        'test',
        'test',
        'test',
        crs,
        shape[1],
        shape[0],
        extents,
    )
    return area_def


def _get_test_data(shape=(200, 100), chunks=50):
    data = np.linspace(0., 1., shape[0] * shape[1], dtype=np.float32).reshape(shape)
    return da.from_array(data, chunks=chunks)


def _get_test_lcc_data(dask_arr, area_def, extra_attrs=None):
    attrs = dict(
        name='test_ds',
        platform_name='PLAT',
        sensor='SENSOR',
        units='1',
        standard_name='toa_bidirectional_reflectance',
        area=area_def,
        start_time=START_TIME,
        end_time=END_TIME
    )
    if extra_attrs:
        attrs.update(extra_attrs)
    ds = xr.DataArray(
        dask_arr,
        dims=('y', 'x') if dask_arr.ndim == 2 else ('bands', 'y', 'x'),
        attrs=attrs,
    )
    return update_resampled_coords(ds, ds, area_def)


class TestAWIPSTiledWriter:
    """Test basic functionality of AWIPS Tiled writer."""

    def test_init(self, tmp_path):
        """Test basic init method of writer."""
        from satpy.writers.awips_tiled import AWIPSTiledWriter
        AWIPSTiledWriter(base_dir=str(tmp_path))

    @pytest.mark.parametrize('use_save_dataset',
                             [(False,), (True,)])
    @pytest.mark.parametrize(
        ('extra_attrs', 'expected_filename'),
        [
            ({}, 'TESTS_AII_PLAT_SENSOR_test_ds_TEST_T001_20180101_1200.nc'),
            ({'sensor': 'viirs', 'name': 'I01'}, 'TESTS_AII_PLAT_viirs_I01_TEST_T001_20180101_1200.nc'),
        ]
    )
    def test_basic_numbered_1_tile(self, extra_attrs, expected_filename, use_save_dataset, caplog, tmp_path):
        """Test creating a single numbered tile."""
        from satpy.writers.awips_tiled import AWIPSTiledWriter
        data = _get_test_data()
        area_def = _get_test_area()
        input_data_arr = _get_test_lcc_data(data, area_def, extra_attrs)
        with caplog.at_level(logging.DEBUG):
            w = AWIPSTiledWriter(base_dir=str(tmp_path), compress=True)
            if use_save_dataset:
                w.save_dataset(input_data_arr, sector_id='TEST', source_name='TESTS')
            else:
                w.save_datasets([input_data_arr], sector_id='TEST', source_name='TESTS')

        assert "no routine matching" not in caplog.text
        assert "Can't format string" not in caplog.text
        all_files = glob(os.path.join(str(tmp_path), 'TESTS_AII*.nc'))
        assert len(all_files) == 1
        assert os.path.basename(all_files[0]) == expected_filename
        for fn in all_files:
            unmasked_ds = xr.open_dataset(fn, mask_and_scale=False)
            output_ds = xr.open_dataset(fn, mask_and_scale=True)
            check_required_properties(unmasked_ds, output_ds)
            scale_factor = output_ds['data'].encoding['scale_factor']
            np.testing.assert_allclose(input_data_arr.values, output_ds['data'].data,
                                       atol=scale_factor / 2)

    def test_units_length_warning(self, tmp_path):
        """Test long 'units' warnings are raised."""
        from satpy.writers.awips_tiled import AWIPSTiledWriter
        data = _get_test_data()
        area_def = _get_test_area()
        input_data_arr = _get_test_lcc_data(data, area_def)
        input_data_arr.attrs["units"] = "this is a really long units string"
        w = AWIPSTiledWriter(base_dir=str(tmp_path), compress=True)
        with pytest.warns(UserWarning, match=r'.*this is a really long units string.*too long.*'):
            w.save_dataset(input_data_arr, sector_id='TEST', source_name='TESTS')

    @pytest.mark.parametrize(
        ("tile_count", "tile_size"),
        [
            ((3, 3), None),
            (None, (67, 34)),
            (None, None),
        ]
    )
    def test_basic_numbered_tiles(self, tile_count, tile_size, tmp_path):
        """Test creating a multiple numbered tiles."""
        from satpy.tests.utils import CustomScheduler
        from satpy.writers.awips_tiled import AWIPSTiledWriter
        data = _get_test_data()
        area_def = _get_test_area()
        input_data_arr = _get_test_lcc_data(data, area_def)
        w = AWIPSTiledWriter(base_dir=str(tmp_path), compress=True)
        save_kwargs = dict(
            sector_id='TEST',
            source_name="TESTS",
            tile_count=tile_count,
            tile_size=tile_size,
            extra_global_attrs={'my_global': 'TEST'}
        )
        should_error = tile_count is None and tile_size is None
        if should_error:
            with dask.config.set(scheduler=CustomScheduler(0)),\
                 pytest.raises(ValueError, match=r'Either.*tile_count.*'):
                w.save_datasets([input_data_arr], **save_kwargs)
        else:
            with dask.config.set(scheduler=CustomScheduler(1 * 2)):  # precompute=*2
                w.save_datasets([input_data_arr], **save_kwargs)

        all_files = glob(os.path.join(str(tmp_path), 'TESTS_AII*.nc'))
        expected_num_files = 0 if should_error else 9
        assert len(all_files) == expected_num_files
        for fn in all_files:
            unmasked_ds = xr.open_dataset(fn, mask_and_scale=False)
            masked_ds = xr.open_dataset(fn, mask_and_scale=True)
            check_required_properties(unmasked_ds, masked_ds)
            assert unmasked_ds.attrs['my_global'] == 'TEST'
            assert unmasked_ds.attrs['sector_id'] == 'TEST'
            assert 'physical_element' in unmasked_ds.attrs
            stime = input_data_arr.attrs['start_time']
            assert unmasked_ds.attrs['start_date_time'] == stime.strftime('%Y-%m-%dT%H:%M:%S')

    def test_basic_lettered_tiles(self, tmp_path):
        """Test creating a lettered grid."""
        from satpy.writers.awips_tiled import AWIPSTiledWriter
        w = AWIPSTiledWriter(base_dir=str(tmp_path), compress=True)
        data = _get_test_data(shape=(2000, 1000), chunks=500)
        area_def = _get_test_area(shape=(2000, 1000),
                                  extents=(-1000000., -1500000., 1000000., 1500000.))
        ds = _get_test_lcc_data(data, area_def)
        # tile_count should be ignored since we specified lettered_grid
        w.save_datasets([ds], sector_id='LCC', source_name="TESTS", tile_count=(3, 3), lettered_grid=True)
        all_files = glob(os.path.join(str(tmp_path), 'TESTS_AII*.nc'))
        assert len(all_files) == 16
        for fn in all_files:
            unmasked_ds = xr.open_dataset(fn, mask_and_scale=False)
            masked_ds = xr.open_dataset(fn, mask_and_scale=True)
            check_required_properties(unmasked_ds, masked_ds)
            assert masked_ds.attrs['start_date_time'] == START_TIME.strftime('%Y-%m-%dT%H:%M:%S')

    def test_basic_lettered_tiles_diff_projection(self, tmp_path):
        """Test creating a lettered grid from data with differing projection.."""
        from satpy.writers.awips_tiled import AWIPSTiledWriter
        w = AWIPSTiledWriter(base_dir=str(tmp_path), compress=True)
        crs = CRS("+proj=lcc +datum=WGS84 +ellps=WGS84 +lon_0=-95. +lat_0=45 +lat_1=45 +units=m +no_defs")
        data = _get_test_data(shape=(2000, 1000), chunks=500)
        area_def = _get_test_area(shape=(2000, 1000), crs=crs,
                                  extents=(-1000000., -1500000., 1000000., 1500000.))
        ds = _get_test_lcc_data(data, area_def)
        # tile_count should be ignored since we specified lettered_grid
        w.save_datasets([ds], sector_id='LCC', source_name="TESTS", tile_count=(3, 3), lettered_grid=True)
        all_files = sorted(glob(os.path.join(str(tmp_path), 'TESTS_AII*.nc')))
        assert len(all_files) == 24
        assert "TC02" in all_files[0]  # the first tile should be TC02
        for fn in all_files:
            unmasked_ds = xr.open_dataset(fn, mask_and_scale=False)
            masked_ds = xr.open_dataset(fn, mask_and_scale=True)
            check_required_properties(unmasked_ds, masked_ds)
            assert masked_ds.attrs['start_date_time'] == START_TIME.strftime('%Y-%m-%dT%H:%M:%S')

    def test_lettered_tiles_update_existing(self, tmp_path):
        """Test updating lettered tiles with additional data."""
        from satpy.writers.awips_tiled import AWIPSTiledWriter
        first_base_dir = os.path.join(str(tmp_path), 'first')
        w = AWIPSTiledWriter(base_dir=first_base_dir, compress=True)
        shape = (2000, 1000)
        data = np.linspace(0., 1., shape[0] * shape[1], dtype=np.float32).reshape(shape)
        # pixels to be filled in later
        data[:, -200:] = np.nan
        data = da.from_array(data, chunks=500)
        area_def = _get_test_area(shape=(2000, 1000),
                                  extents=(-1000000., -1500000., 1000000., 1500000.))
        ds = _get_test_lcc_data(data, area_def)
        # tile_count should be ignored since we specified lettered_grid
        w.save_datasets([ds], sector_id='LCC', source_name="TESTS", tile_count=(3, 3), lettered_grid=True)
        all_files = sorted(glob(os.path.join(first_base_dir, 'TESTS_AII*.nc')))
        assert len(all_files) == 16
        first_files = []
        second_base_dir = os.path.join(str(tmp_path), 'second')
        os.makedirs(second_base_dir)
        for fn in all_files:
            new_fn = fn.replace(first_base_dir, second_base_dir)
            shutil.copy(fn, new_fn)
            first_files.append(new_fn)

        # Second writing/updating
        # Area is about 100 pixels to the right
        area_def2 = _get_test_area(shape=(2000, 1000),
                                   extents=(-800000., -1500000., 1200000., 1500000.))
        data2 = np.linspace(0., 1., 2000000, dtype=np.float32).reshape((2000, 1000))
        # a gap at the beginning where old values remain
        data2[:, :200] = np.nan
        # a gap at the end where old values remain
        data2[:, -400:-300] = np.nan
        data2 = da.from_array(data2, chunks=500)
        ds2 = _get_test_lcc_data(data2, area_def2)
        w = AWIPSTiledWriter(base_dir=second_base_dir, compress=True)
        # HACK: The _copy_to_existing function hangs when opening the output
        #   file multiple times...sometimes. If we limit dask to one worker
        #   it seems to work fine.
        with dask.config.set(num_workers=1):
            w.save_datasets([ds2], sector_id='LCC', source_name="TESTS", tile_count=(3, 3), lettered_grid=True)
        all_files = glob(os.path.join(second_base_dir, 'TESTS_AII*.nc'))
        # 16 original tiles + 4 new tiles
        assert len(all_files) == 20

        # these tiles should be the right-most edge of the first image
        first_right_edge_files = [x for x in first_files if 'P02' in x or 'P04' in x or 'V02' in x or 'V04' in x]
        for new_file in first_right_edge_files:
            orig_file = new_file.replace(second_base_dir, first_base_dir)
            orig_nc = xr.open_dataset(orig_file)
            orig_data = orig_nc['data'].values
            if not np.isnan(orig_data).any():
                # we only care about the tiles that had NaNs originally
                continue

            new_nc = xr.open_dataset(new_file)
            new_data = new_nc['data'].values
            # there should be at least some areas of the file
            # that old data was present and hasn't been replaced
            np.testing.assert_allclose(orig_data[:, :20], new_data[:, :20])
            # it isn't exactly 200 because the tiles aren't aligned with the
            # data (the left-most tile doesn't have data until some columns
            # in), but it should be at least that many columns
            assert np.isnan(orig_data[:, 200:]).all()
            assert not np.isnan(new_data[:, 200:]).all()

    def test_lettered_tiles_sector_ref(self, tmp_path):
        """Test creating a lettered grid using the sector as reference."""
        from satpy.writers.awips_tiled import AWIPSTiledWriter
        w = AWIPSTiledWriter(base_dir=str(tmp_path), compress=True)
        data = _get_test_data(shape=(2000, 1000), chunks=500)
        area_def = _get_test_area(shape=(2000, 1000),
                                  extents=(-1000000., -1500000., 1000000., 1500000.))
        ds = _get_test_lcc_data(data, area_def)
        w.save_datasets([ds], sector_id='LCC', source_name="TESTS",
                        lettered_grid=True, use_sector_reference=True,
                        use_end_time=True)
        all_files = glob(os.path.join(str(tmp_path), 'TESTS_AII*.nc'))
        assert len(all_files) == 16
        for fn in all_files:
            unmasked_ds = xr.open_dataset(fn, mask_and_scale=False)
            masked_ds = xr.open_dataset(fn, mask_and_scale=True)
            check_required_properties(unmasked_ds, masked_ds)
            expected_start = (START_TIME + timedelta(minutes=20)).strftime('%Y-%m-%dT%H:%M:%S')
            assert masked_ds.attrs['start_date_time'] == expected_start

    def test_lettered_tiles_no_fit(self, tmp_path):
        """Test creating a lettered grid with no data overlapping the grid."""
        from satpy.writers.awips_tiled import AWIPSTiledWriter
        w = AWIPSTiledWriter(base_dir=str(tmp_path), compress=True)
        data = _get_test_data(shape=(2000, 1000), chunks=500)
        area_def = _get_test_area(shape=(2000, 1000),
                                  extents=(4000000., 5000000., 5000000., 6000000.))
        ds = _get_test_lcc_data(data, area_def)
        w.save_datasets([ds], sector_id='LCC', source_name="TESTS", tile_count=(3, 3), lettered_grid=True)
        # No files created
        all_files = glob(os.path.join(str(tmp_path), 'TESTS_AII*.nc'))
        assert not all_files

    def test_lettered_tiles_no_valid_data(self, tmp_path):
        """Test creating a lettered grid with no valid data."""
        from satpy.writers.awips_tiled import AWIPSTiledWriter
        w = AWIPSTiledWriter(base_dir=str(tmp_path), compress=True)
        data = da.full((2000, 1000), np.nan, chunks=500, dtype=np.float32)
        area_def = _get_test_area(shape=(2000, 1000),
                                  extents=(-1000000., -1500000., 1000000., 1500000.))
        ds = _get_test_lcc_data(data, area_def)
        w.save_datasets([ds], sector_id='LCC', source_name="TESTS", tile_count=(3, 3), lettered_grid=True)
        # No files created - all NaNs should result in no tiles being created
        all_files = glob(os.path.join(str(tmp_path), 'TESTS_AII*.nc'))
        assert not all_files

    def test_lettered_tiles_bad_filename(self, tmp_path):
        """Test creating a lettered grid with a bad filename."""
        from satpy.writers.awips_tiled import AWIPSTiledWriter
        w = AWIPSTiledWriter(base_dir=str(tmp_path), compress=True, filename="{Bad Key}.nc")
        data = _get_test_data(shape=(2000, 1000), chunks=500)
        area_def = _get_test_area(shape=(2000, 1000),
                                  extents=(-1000000., -1500000., 1000000., 1500000.))
        ds = _get_test_lcc_data(data, area_def)
        with pytest.raises(KeyError):
            w.save_datasets([ds],
                            sector_id='LCC',
                            source_name='TESTS',
                            tile_count=(3, 3),
                            lettered_grid=True)

    def test_basic_numbered_tiles_rgb(self, tmp_path):
        """Test creating a multiple numbered tiles with RGB."""
        from satpy.writers.awips_tiled import AWIPSTiledWriter
        w = AWIPSTiledWriter(base_dir=str(tmp_path), compress=True)
        data = da.from_array(np.linspace(0., 1., 60000, dtype=np.float32).reshape((3, 200, 100)), chunks=50)
        area_def = _get_test_area()
        ds = _get_test_lcc_data(data, area_def)
        ds = ds.rename(dict((old, new) for old, new in zip(ds.dims, ['bands', 'y', 'x'])))
        ds.coords['bands'] = ['R', 'G', 'B']

        w.save_datasets([ds], sector_id='TEST', source_name="TESTS", tile_count=(3, 3))
        chan_files = glob(os.path.join(str(tmp_path), 'TESTS_AII*test_ds_R*.nc'))
        all_files = chan_files[:]
        assert len(chan_files) == 9
        chan_files = glob(os.path.join(str(tmp_path), 'TESTS_AII*test_ds_G*.nc'))
        all_files.extend(chan_files)
        assert len(chan_files) == 9
        chan_files = glob(os.path.join(str(tmp_path), 'TESTS_AII*test_ds_B*.nc'))
        assert len(chan_files) == 9
        all_files.extend(chan_files)
        for fn in all_files:
            unmasked_ds = xr.open_dataset(fn, mask_and_scale=False)
            masked_ds = xr.open_dataset(fn, mask_and_scale=True)
            check_required_properties(unmasked_ds, masked_ds)

    @pytest.mark.parametrize(
        "sector",
        ['C',
         'F']
    )
    @pytest.mark.parametrize(
        "extra_kwargs",
        [
            {},
            {'environment_prefix': 'AA'},
            {'environment_prefix': 'BB', 'filename': '{environment_prefix}_{name}_GLM_T{tile_number:04d}.nc'},
        ]
    )
    def test_multivar_numbered_tiles_glm(self, sector, extra_kwargs, tmp_path):
        """Test creating a tiles with multiple variables."""
        from satpy.writers.awips_tiled import AWIPSTiledWriter
        os.environ['ORGANIZATION'] = '1' * 50
        w = AWIPSTiledWriter(base_dir=tmp_path, compress=True)
        data = _get_test_data()
        area_def = _get_test_area()
        ds1 = _get_test_lcc_data(data, area_def)
        ds1.attrs.update(
            dict(
                name='total_energy',
                platform_name='GOES-17',
                sensor='SENSOR',
                units='1',
                scan_mode='M3',
                scene_abbr=sector,
                platform_shortname="G17"
            )
        )
        ds2 = ds1.copy()
        ds2.attrs.update({
            'name': 'flash_extent_density',
        })
        ds3 = ds1.copy()
        ds3.attrs.update({
            'name': 'average_flash_area',
        })
        dqf = ds1.copy()
        dqf = (dqf * 255).astype(np.uint8)
        dqf.attrs = ds1.attrs.copy()
        dqf.attrs.update({
            'name': 'DQF',
            '_FillValue': 1,
        })

        w.save_datasets([ds1, ds2, ds3, dqf], sector_id='TEST', source_name="TESTS",
                        tile_count=(3, 3), template='glm_l2_rad{}'.format(sector.lower()),
                        **extra_kwargs)
        fn_glob = self._get_glm_glob_filename(extra_kwargs)
        all_files = glob(os.path.join(str(tmp_path), fn_glob))
        assert len(all_files) == 9
        for fn in all_files:
            unmasked_ds = xr.open_dataset(fn, mask_and_scale=False)
            masked_ds = xr.open_dataset(fn, mask_and_scale=True)
            check_required_properties(unmasked_ds, masked_ds)
            if sector == 'C':
                assert masked_ds.attrs['time_coverage_end'] == END_TIME.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            else:  # 'F'
                assert masked_ds.attrs['time_coverage_end'] == END_TIME.strftime('%Y-%m-%dT%H:%M:%SZ')

    @staticmethod
    def _get_glm_glob_filename(extra_kwargs):
        if 'filename' in extra_kwargs:
            return 'BB*_GLM*.nc'
        elif 'environment_prefix' in extra_kwargs:
            return 'AA*_GLM*.nc'
        return 'DR*_GLM*.nc'
