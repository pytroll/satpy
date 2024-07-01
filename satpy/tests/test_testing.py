"""Tests for the testing helper module."""

import numpy as np
import xarray as xr

from satpy import Scene
from satpy.resample import get_area_def
from satpy.testing import fake_satpy_reading


def test_fake_reading(tmp_path):
    """Test that the fake reading context manager populates a scene."""
    input_files = [tmp_path / "my_input_file"]
    area = get_area_def("euro4")
    random = np.random.default_rng()
    somedata = xr.DataArray(random.uniform(size=area.shape), dims=["y", "x"])
    somedata.attrs["area"] = area

    channel = "VIS006"

    scene_dict = {channel: somedata}

    with fake_satpy_reading(scene_dict):
        scene = Scene(input_files, reader="dummy_reader")
        scene.load([channel])
    assert scene[channel] is somedata
