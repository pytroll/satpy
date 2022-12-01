"""Tests for the H-SAF H5 reader."""
import os
import tempfile
from datetime import datetime

import h5py
import numpy as np

from satpy import Scene
from satpy.resample import get_area_def

# real shape is 916, 1902
shape_sc = (916, 1902)
shape_sc_colormap = (256, 3)


def hsaf_filename():
    """Create a fake HSAF SC HDF5 file."""
    dirname = tempfile.mkdtemp()
    filename = dirname + "/h10_20221115_day_merged.H5"
    with h5py.File(filename, mode="w") as h5f:
        h5f.create_dataset('SC', shape_sc, dtype=np.uint8)
        h5f.create_dataset('colormap', shape_sc_colormap, dtype=np.uint8)
    return filename


def test_hsaf_sc_dataset():
    """Test the H-SAF SC dataset."""
    filename = hsaf_filename()

    try:
        scn = Scene(filenames=[str(filename)], reader="hsaf_h5")
        scn.load(['SC'])
    finally:
        os.remove(filename)
    assert scn['SC'].shape == shape_sc


def test_hsaf_sc_colormap_dataset():
    """Test the H-SAF SC_pal dataset."""
    filename = hsaf_filename()

    try:
        scn = Scene(filenames=[str(filename)], reader="hsaf_h5")
        scn.load(['SC_pal'])
    finally:
        os.remove(filename)
    assert scn['SC_pal'].shape == shape_sc_colormap


def test_hsaf_sc_datetime():
    """Test the H-SAF reference time."""
    filename = hsaf_filename()

    try:
        scn = Scene(filenames=[str(filename)], reader="hsaf_h5")
        scn.load(['SC'])
        fname = os.path.basename(filename)
        dtstr = fname.split('_')[1]
        obs_time = datetime.strptime(dtstr, "%Y%m%d")
    finally:
        os.remove(filename)
    assert scn['SC'].attrs['data_time'] == obs_time


def test_hsaf_sc_areadef():
    """Test the H-SAF SC area definition."""
    filename = hsaf_filename()

    try:
        scn = Scene(filenames=[str(filename)], reader="hsaf_h5")
        scn.load(['SC'])
        fd_def = get_area_def('msg_seviri_fes_3km')
        hsaf_def = fd_def[62:62+916, 1211:1211+1902]
    finally:
        os.remove(filename)
    assert scn['SC'].area == hsaf_def
