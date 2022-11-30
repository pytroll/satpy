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


@pytest.fixture(scope="session")
def hsaf_filename():
    """Create a fake HSAF SC HDF5 file."""
    filename = "h10_20221115_day_merged.H5"
    with h5py.File(filename, mode="w") as h5f:
        h5f.create_dataset('SC', shape_sc, dtype=np.uint8)
        h5f.create_dataset('colormap', shape_sc_colormap, dtype=np.uint8)
    return filename


def test_hsaf_sc_dataset(hsaf_filename):
    """Test the H-SAF SC dataset."""
    scn = Scene(filenames=[str(hsaf_filename)], reader="hsaf_h5")
    scn.load(['SC'])
    assert scn['SC'].shape == shape_sc


def test_hsaf_sc_colormap_dataset(hsaf_filename):
    """Test the H-SAF SC_pal dataset."""
    scn = Scene(filenames=[str(hsaf_filename)], reader="hsaf_h5")
    scn.load(['SC_pal'])
    assert scn['SC_pal'].shape == shape_sc_colormap


def test_hsaf_sc_datetime(hsaf_filename):
    """Test the H-SAF reference time."""
    scn = Scene(filenames=[str(hsaf_filename)], reader="hsaf_h5")
    scn.load(['SC'])
    fname = os.path.basename(hsaf_filename)
    dtstr = fname.split('_')[1]
    obs_time = datetime.strptime(dtstr, "%Y%m%d")
    assert scn['SC'].attrs['data_time'] == obs_time


def test_hsaf_sc_areadef(hsaf_filename):
    """Test the H-SAF SC area definition."""
    scn = Scene(filenames=[str(hsaf_filename)], reader="hsaf_h5")
    scn.load(['SC'])
    fd_def = get_area_def('msg_seviri_fes_3km')
    hsaf_def = fd_def[62:62+916, 1211:1211+1902]
    os.remove(hsaf_filename)
    assert scn['SC'].area == hsaf_def
