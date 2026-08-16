# Copyright (c) 2026 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
"""Regression tests for required NetCDF variable collection."""

import numpy as np

from satpy.readers.core.netcdf import NetCDF4FileHandler


def _create_required_variables_file(tmp_path):
    from netCDF4 import Dataset

    filename = tmp_path / "required_variables.nc"
    with Dataset(filename, "w") as nc:
        nc.createDimension("x", 3)
        root_var = nc.createVariable("root_var", np.int16, ("x",))
        root_var[:] = [1, 2, 3]
        root_after_attr = nc.createVariable("root_after_attr", np.int16, ("x",))
        root_after_attr[:] = [4, 5, 6]
        nc.test_attr = "test"
    return filename


def test_required_netcdf_variables_support_root_variables(tmp_path):
    """Collect required variables located at the NetCDF root."""
    filename = _create_required_variables_file(tmp_path)
    filetype_info = {"required_netcdf_variables": ["root_var"]}

    file_handler = NetCDF4FileHandler(filename, {}, filetype_info)

    assert "root_var" in file_handler.file_content
    assert file_handler.file_content["root_var/shape"] == (3,)


def test_required_root_variable_after_attribute_is_collected(tmp_path):
    """Collect a root variable that follows a required attribute."""
    filename = _create_required_variables_file(tmp_path)
    filetype_info = {
        "required_netcdf_variables": ["attr/test_attr", "root_after_attr"]
    }

    file_handler = NetCDF4FileHandler(filename, {}, filetype_info)

    assert file_handler.file_content["attr/test_attr"] == "test"
    assert "root_after_attr" in file_handler.file_content
