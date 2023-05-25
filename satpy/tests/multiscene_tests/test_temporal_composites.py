# Copyright (c) 2023 Satpy developers
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

"""Tests for loading temporal composites."""

import datetime

import numpy as np
import pytest
import xarray as xr

composite_definition = """sensor_name: visir

composites:
  temporal:
    compositor: !!python/name:satpy.composites.TemporalRGB
    prerequisites:
      - name: ir
        time: 0
      - name: ir
        time: -10 min
      - name: ir
        time: -20 min
    standard_name: temporal
"""


@pytest.fixture
def fake_config(tmp_path):
    """Make a configuration path with a temporal composite definition."""
    confdir = tmp_path / "etc"
    conffile = tmp_path / "etc" / "composites" / "visir.yaml"
    conffile.parent.mkdir(parents=True)
    with conffile.open(mode="wt", encoding="ascii") as fp:
        fp.write(composite_definition)
    return confdir


@pytest.fixture
def fake_dataset():
    """Create minimal fake Satpy CF NC dataset."""
    ds = xr.Dataset()
    nx = ny = 4
    ds["ir"] = xr.DataArray(
            np.zeros((nx, ny)),
            dims=("y", "x"),
            attrs={"sensor": "visir"})
    return ds


@pytest.fixture
def fake_files(tmp_path, fake_dataset):
    """Make fake files for the Satpy CF reader."""
    start_time = datetime.datetime(2050, 5, 3, 12, 0, 0)
    delta = datetime.timedelta(minutes=10)
    n_timesteps = 5
    ofs = []
    for i in range(n_timesteps):
        begin = start_time + i*delta
        end = start_time + (i+1)*delta
        of = tmp_path / f"Meteosat99-imager-{begin:%Y%m%d%H%M%S}-{end:%Y%m%d%H%M%S}.nc"
        fd = fake_dataset.copy()
        fd["ir"][...] = i
        fd["ir"].attrs["start_time"] = f"{start_time:%Y-%m-%dT%H:%M:%S}"
        fd.to_netcdf(of)
        ofs.append(of)
    return ofs


def test_load_temporal_composite(fake_files, fake_config):
    """Test loading a temporal composite."""
    from satpy import config
    from satpy.multiscene import MultiScene, timeseries
    with config.set(config_path=[fake_config]):
        ms = MultiScene.from_files(fake_files, reader="satpy_cf_nc")
        ms.load(["temporal"])
        sc = ms.blend(blend_function=timeseries)
        assert sc["temporal"].shape == (3, 4, 4)
        np.testing.assert_array_equal(sc["temporal"][0, :, :], np.full((4, 4), 4))
        np.testing.assert_array_equal(sc["temporal"][1, :, :], np.full((4, 4), 3))
        np.testing.assert_array_equal(sc["temporal"][2, :, :], np.full((4, 4), 2))
