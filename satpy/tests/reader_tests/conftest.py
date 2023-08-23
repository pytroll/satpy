#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Satpy developers
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
"""Setup and configuration for all reader tests."""

import pathlib
import shutil

import numpy as np
import pytest
import requests

from ._modis_fixtures import (
    modis_l1b_imapp_1000m_file,
    modis_l1b_imapp_geo_file,
    modis_l1b_nasa_1km_mod03_files,
    modis_l1b_nasa_mod02hkm_file,
    modis_l1b_nasa_mod02qkm_file,
    modis_l1b_nasa_mod03_file,
    modis_l1b_nasa_mod021km_file,
    modis_l2_imapp_mask_byte1_file,
    modis_l2_imapp_mask_byte1_geo_files,
    modis_l2_imapp_snowmask_file,
    modis_l2_imapp_snowmask_geo_files,
    modis_l2_nasa_mod06_file,
    modis_l2_nasa_mod35_file,
    modis_l2_nasa_mod35_mod03_files,
)

_url_sample_file = (
        "https://go.dwd-nextcloud.de/index.php/s/z87KfL72b9dM5xm/download/"
        "IASI_SND_02_M01_20190605002352Z_20190605020856Z_N_O_20190605011702Z.nat")


@pytest.fixture(scope="session")
def iasisndl2_file(tmp_path_factory, request):
    """Obtain sample file."""
    fn = pathlib.Path("/media/nas/x21308/IASI/IASI_SND_02_M01_20190605002352Z_20190605020856Z_N_O_20190605011702Z.nat")
    if not fn.exists():
        fn = tmp_path_factory.mktemp("data") / "IASI_SND_02_M01_20190605002352Z_20190605020856Z_N_O_20190605011702Z.nat"
        data = requests.get(_url_sample_file, stream=True)
        with fn.open(mode="wb") as fp:
            shutil.copyfileobj(data.raw, fp)
    if request.param == "string":
        return fn
    if request.param == "file":
        return open(fn, mode="rb")
    if request.param == "mmap":
        return np.memmap(fn, mode="r", offset=0)
