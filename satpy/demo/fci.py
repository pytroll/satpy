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
"""Demo FCI data download."""

import pathlib
import tarfile
import tempfile

from satpy import config

from . import utils

_fci_uncompressed_nominal = (
    "https://sftp.eumetsat.int/public/folder/UsCVknVOOkSyCdgpMimJNQ/"
    "User-Materials/Test-Data/MTG/MTG_FCI_L1C_Enhanced-NonN_TD-272_May2020/"
    "FCI_1C_UNCOMPRESSED_NOMINAL.tar.gz")


def download_fci_test_data(base_dir=None):
    """Download FCI test data.

    Download the nominal FCI test data from July 2020.
    """
    subdir = get_fci_test_data_dir(base_dir=base_dir)
    with tempfile.TemporaryDirectory() as td:
        nm = pathlib.Path(td) / "fci-test-data.tar.gz"
        utils.download_url(_fci_uncompressed_nominal, nm)
        return _unpack_tarfile_to(nm, subdir)


def get_fci_test_data_dir(base_dir=None):
    """Get directory for FCI test data."""
    base_dir = base_dir or config.get("demo_data_dir", ".")
    return pathlib.Path(base_dir) / "fci" / "test_data"


def _unpack_tarfile_to(filename, subdir):
    """Unpack content of tarfile in filename to subdir."""
    with tarfile.open(filename, mode="r:gz") as tf:
        contents = tf.getnames()
        tf.extractall(path=subdir)
    return contents
