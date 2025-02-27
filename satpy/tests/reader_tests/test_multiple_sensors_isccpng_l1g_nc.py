#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023- Satpy developers
#
# satpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# satpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.

"""The multiple_sensors_isccpng_l1g_nc reader tests package.

This version tests the readers for ISCCP L1G data.

"""


import datetime as dt

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def nc_filename(tmp_path):
    """Create nc test data file and return its filename."""
    now = dt.datetime.now(dt.timezone.utc)
    filename = f"ISCCP-NG_L1g_demo_v5_res_0_05deg__temp_11_00um__{now:%Y%m%dT%H%M}.nc"
    filename_str = str(tmp_path / filename)

    jan_1970 = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    delta_t = now - jan_1970
    stime = delta_t.seconds
    etime = delta_t.seconds + 600
    # Create test data
    nscn = 3600
    npix = 7200
    lats = np.linspace(-90, 90, nscn)
    lons = np.linspace(-180, 180, npix)
    array = 27000 * np.ones((1, 3, nscn, npix))
    ds = xr.Dataset({"temp_11_00um": (("time", "layer", "latitude", "longitude"), array),
                     },
                    coords={"start_time": ("time", [stime]),
                            "end_time": ("time", [etime]),
                            "latitude": lats[:],
                            "longitude": lons[:]},
                    attrs={"scale_factor": 0.01, "units": "K"})

    ds["temp_11_00um"].attrs["_FillValue"] = -32767
    ds["temp_11_00um"].attrs["scale_factor"] = 0.01
    ds["temp_11_00um"].attrs["units"] = "K"
    ds["longitude"].attrs["standard_name"] = "longitude"
    ds["latitude"].attrs["standard_name"] = "latitude"
    ds["temp_11_00um"].attrs["standard_name"] = "temp_11_00um"
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(filename_str, encoding=encoding)
    return filename_str


class TestISCCPNGL1gReader:
    """Test the IsccpngL1gFileHandler reader."""

    def test_read_isccpng_l1g(self, nc_filename):
        """Test reading reflectances and BT."""
        from satpy.scene import Scene

        # Read data
        scn_ = Scene(
            reader="multiple_sensors_isccpng_l1g_nc",
            filenames=[nc_filename])
        scn_.load(["temp_11_00um", "lon", "lat"])
        assert (scn_["lat"].shape == (3600, 7200))
        assert (scn_["lon"].shape == (3600, 7200))
        assert (scn_["temp_11_00um"].shape == (3600, 7200))
        assert (scn_["temp_11_00um"].values[0, 0] == 270)
