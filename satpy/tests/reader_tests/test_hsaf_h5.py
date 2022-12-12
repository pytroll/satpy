"""Tests for the H-SAF H5 reader."""
import os
from datetime import datetime

import h5py
import numpy as np
import pytest

from satpy import Scene
from satpy.resample import get_area_def

# real shape is 916, 1902
shape_sc = (916, 1902)
shape_sc_colormap = (256, 3)
AREA_X_OFFSET = 1211
AREA_Y_OFFSET = 62


@pytest.fixture(scope="session")
def sc_h5_file(tmp_path_factory):
    """Create a fake HSAF SC HDF5 file."""
    filename = tmp_path_factory.mktemp("data") / "h10_20221115_day_merged.H5"
    h5f = h5py.File(filename, mode="w")
    h5f.create_dataset('SC', shape_sc, dtype=np.uint8)
    h5f.create_dataset('colormap', shape_sc_colormap, dtype=np.uint8)
    return str(filename)


def test_hsaf_sc_dataset(sc_h5_file):
    """Test the H-SAF SC dataset."""
    scn = Scene(filenames=[sc_h5_file], reader="hsaf_h5")
    scn.load(['SC'])
    assert scn['SC'].shape == shape_sc


def test_hsaf_sc_colormap_dataset(sc_h5_file):
    """Test the H-SAF SC_pal dataset."""
    scn = Scene(filenames=[sc_h5_file], reader="hsaf_h5")
    scn.load(['SC_pal'])
    assert scn['SC_pal'].shape == shape_sc_colormap


def test_hsaf_sc_datetime(sc_h5_file):
    """Test the H-SAF reference time."""
    scn = Scene(filenames=[sc_h5_file], reader="hsaf_h5")
    scn.load(['SC'])
    fname = os.path.basename(sc_h5_file)
    dtstr = fname.split('_')[1]
    obs_time = datetime.strptime(dtstr, "%Y%m%d")
    assert scn['SC'].attrs['data_time'] == obs_time


def test_hsaf_sc_areadef(sc_h5_file):
    """Test the H-SAF SC area definition."""
    scn = Scene(filenames=[sc_h5_file], reader="hsaf_h5")
    scn.load(['SC'])
    fd_def = get_area_def('msg_seviri_fes_3km')
    hsaf_def = fd_def[AREA_Y_OFFSET:AREA_Y_OFFSET+916, AREA_X_OFFSET:AREA_X_OFFSET+1902]
    assert scn['SC'].area == hsaf_def
