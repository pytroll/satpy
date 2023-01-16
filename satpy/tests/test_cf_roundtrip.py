#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2019 Satpy developers
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
"""Test roundripping the cf writer and reader."""

import os

import numpy as np

from satpy import Scene
from satpy.tests.reader_tests.test_viirs_compact import fake_dnb, fake_dnb_file  # noqa

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path


def test_cf_roundtrip(fake_dnb_file, tmp_path):  # noqa
    """Test the cf writing reading cycle."""
    dnb_filename = os.fspath(fake_dnb_file)
    write_scn = Scene(filenames=[dnb_filename], reader="viirs_compact")
    write_scn.load(["DNB"])

    satpy_cf_file = os.fspath(tmp_path / "npp-viirs-20191025061125-20191025061247.nc")
    write_scn.save_datasets(writer="cf", filename=satpy_cf_file)
    read_scn = Scene(filenames=[satpy_cf_file], reader="satpy_cf_nc")
    read_scn.load(["DNB"])

    write_array = write_scn["DNB"]
    read_array = read_scn["DNB"]

    np.testing.assert_allclose(write_array.values, read_array.values)
