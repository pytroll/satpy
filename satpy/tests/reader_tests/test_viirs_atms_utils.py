#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Satpy Developers


# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Test common VIIRS/ATMS SDR reader functions."""


import logging

from satpy.readers.viirs_atms_sdr_utils import _get_file_units
from satpy.tests.utils import make_dataid


def test_get_file_units(caplog):
    """Test get the file-units from the dataset info."""
    did = make_dataid(name='some_variable', modifiers=())
    ds_info = {'file_units': None}
    with caplog.at_level(logging.DEBUG):
        file_units = _get_file_units(did, ds_info)

    assert file_units is None
    log_output = "Unknown units for file key 'DataID(name='some_variable', modifiers=())'"
    assert log_output in caplog.text
