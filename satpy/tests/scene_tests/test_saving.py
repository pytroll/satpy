# Copyright (c) 2010-2023 Satpy developers
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
"""Unit tests for saving-related functionality in scene.py."""
import os
from datetime import datetime
from unittest import mock

import pytest
import xarray as xr
from dask import array as da

from satpy import Scene
from satpy.tests.utils import make_cid, spy_decorator

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path


class TestSceneSaving:
    """Test the Scene's saving method."""

    def test_save_datasets_default(self, tmp_path):
        """Save a dataset using 'save_datasets'."""
        ds1 = xr.DataArray(
            da.zeros((100, 200), chunks=50),
            dims=('y', 'x'),
            attrs={'name': 'test',
                   'start_time': datetime(2018, 1, 1, 0, 0, 0)}
        )
        scn = Scene()
        scn['test'] = ds1
        scn.save_datasets(base_dir=tmp_path)
        assert os.path.isfile(os.path.join(tmp_path, 'test_20180101_000000.tif'))

    def test_save_datasets_by_ext(self, tmp_path):
        """Save a dataset using 'save_datasets' with 'filename'."""
        ds1 = xr.DataArray(
            da.zeros((100, 200), chunks=50),
            dims=('y', 'x'),
            attrs={'name': 'test',
                   'start_time': datetime(2018, 1, 1, 0, 0, 0)}
        )
        scn = Scene()
        scn['test'] = ds1

        from satpy.writers.simple_image import PillowWriter
        save_image_mock = spy_decorator(PillowWriter.save_image)
        with mock.patch.object(PillowWriter, 'save_image', save_image_mock):
            scn.save_datasets(base_dir=tmp_path, filename='{name}.png')
        save_image_mock.mock.assert_called_once()
        assert os.path.isfile(os.path.join(tmp_path, 'test.png'))

    def test_save_datasets_bad_writer(self, tmp_path):
        """Save a dataset using 'save_datasets' and a bad writer."""
        ds1 = xr.DataArray(
            da.zeros((100, 200), chunks=50),
            dims=('y', 'x'),
            attrs={'name': 'test',
                   'start_time': datetime.utcnow()}
        )
        scn = Scene()
        scn['test'] = ds1
        pytest.raises(ValueError,
                      scn.save_datasets,
                      writer='_bad_writer_',
                      base_dir=tmp_path)

    def test_save_datasets_missing_wishlist(self, tmp_path):
        """Calling 'save_datasets' with no valid datasets."""
        scn = Scene()
        scn._wishlist.add(make_cid(name='true_color'))
        pytest.raises(RuntimeError,
                      scn.save_datasets,
                      writer='geotiff',
                      base_dir=tmp_path)
        pytest.raises(KeyError,
                      scn.save_datasets,
                      datasets=['no_exist'])

    def test_save_dataset_default(self, tmp_path):
        """Save a dataset using 'save_dataset'."""
        ds1 = xr.DataArray(
            da.zeros((100, 200), chunks=50),
            dims=('y', 'x'),
            attrs={'name': 'test',
                   'start_time': datetime(2018, 1, 1, 0, 0, 0)}
        )
        scn = Scene()
        scn['test'] = ds1
        scn.save_dataset('test', base_dir=tmp_path)
        assert os.path.isfile(os.path.join(tmp_path, 'test_20180101_000000.tif'))
