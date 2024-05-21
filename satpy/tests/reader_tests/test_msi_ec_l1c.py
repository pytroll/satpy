"""Tests for the EarthCARE MSI L1c reader."""
import os
from datetime import datetime

import h5py
import numpy as np
import pytest

from satpy import Scene

SHAPE_SC = (300, 6000)

@pytest.fixture(scope="session")
def sc_h5_file(tmp_path_factory):
    """Create a fake HSAF SC HDF5 file."""
    filename = tmp_path_factory.mktemp("data") / "ECA_EXAA_MSI_RGR_1C_20250410T213955Z_20210720T084332Z_40874D.h5"
    h5f = h5py.File(filename, mode="w")
    h5f.create_dataset("SC", SHAPE_SC, dtype=np.uint8)
    return str(filename)


def _get_scene_with_loaded_sc_datasets(filename):
    """Return a scene with SC and SC_pal loaded."""
    loaded_scene = Scene(filenames=[filename], reader="hsaf_h5")
    loaded_scene.load(["SC", "SC_pal"])
    return loaded_scene


def test_hsaf_sc_dataset(sc_h5_file):
    """Test the H-SAF SC dataset."""
    loaded_scene = _get_scene_with_loaded_sc_datasets(sc_h5_file)
    assert loaded_scene["SC"].shape == SHAPE_SC


def test_hsaf_sc_datetime(sc_h5_file):
    """Test the H-SAF reference time."""
    loaded_scene = _get_scene_with_loaded_sc_datasets(sc_h5_file)
    fname = os.path.basename(sc_h5_file)
    dtstr = fname.split("_")[1]
    obs_time = datetime.strptime(dtstr, "%Y%m%d")
    assert loaded_scene["SC"].attrs["data_time"] == obs_time
