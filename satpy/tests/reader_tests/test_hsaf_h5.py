"""Tests for the H-SAF H5 reader."""
import os
from datetime import datetime

import h5py
import numpy as np
import pytest

from satpy import Scene
from satpy.resample import get_area_def

# real shape is 916, 1902
SHAPE_SC = (916, 1902)
SHAPE_SC_COLORMAP = (256, 3)
AREA_X_OFFSET = 1211
AREA_Y_OFFSET = 62


@pytest.fixture(scope="session")
def sc_h5_file(tmp_path_factory):
    """Create a fake HSAF SC HDF5 file."""
    filename = tmp_path_factory.mktemp("data") / "h10_20221115_day_merged.H5"
    h5f = h5py.File(filename, mode="w")
    h5f.create_dataset('SC', SHAPE_SC, dtype=np.uint8)
    h5f.create_dataset('colormap', SHAPE_SC_COLORMAP, dtype=np.uint8)
    return str(filename)


def _get_scene_with_loaded_sc_datasets(filename):
    """Return a scene with SC and SC_pal loaded."""
    loaded_scene = Scene(filenames=[filename], reader="hsaf_h5")
    loaded_scene.load(['SC', 'SC_pal'])
    return loaded_scene


def test_hsaf_sc_dataset(sc_h5_file):
    """Test the H-SAF SC dataset."""
    loaded_scene = _get_scene_with_loaded_sc_datasets(sc_h5_file)
    assert loaded_scene['SC'].shape == SHAPE_SC


def test_hsaf_sc_colormap_dataset(sc_h5_file):
    """Test the H-SAF SC_pal dataset."""
    loaded_scene = _get_scene_with_loaded_sc_datasets(sc_h5_file)
    assert loaded_scene['SC_pal'].shape == SHAPE_SC_COLORMAP


def test_hsaf_sc_datetime(sc_h5_file):
    """Test the H-SAF reference time."""
    loaded_scene = _get_scene_with_loaded_sc_datasets(sc_h5_file)
    fname = os.path.basename(sc_h5_file)
    dtstr = fname.split('_')[1]
    obs_time = datetime.strptime(dtstr, "%Y%m%d")
    assert loaded_scene['SC'].attrs['data_time'] == obs_time


def test_hsaf_sc_areadef(sc_h5_file):
    """Test the H-SAF SC area definition."""
    loaded_scene = _get_scene_with_loaded_sc_datasets(sc_h5_file)
    fd_def = get_area_def('msg_seviri_fes_3km')
    hsaf_def = fd_def[AREA_Y_OFFSET:AREA_Y_OFFSET+SHAPE_SC[0], AREA_X_OFFSET:AREA_X_OFFSET+SHAPE_SC[1]]
    assert loaded_scene['SC'].area == hsaf_def
