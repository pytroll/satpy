#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Satpy developers
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

"""FCI base reader tests package."""

from satpy.readers.fci_base import calculate_area_extent
from satpy.tests.utils import make_dataid


def test_calculate_area_extent():
    """Test function for calculate_area_extent."""
    dataset_id = make_dataid(name="dummy", resolution=2000.0)

    area_dict = {
        "nlines": 5568,
        "ncols": 5568,
        "line_step": dataset_id["resolution"],
        "column_step": dataset_id["resolution"],
    }

    area_extent = calculate_area_extent(area_dict)

    expected = (-5568000.0, 5568000.0, 5568000.0, -5568000.0)

    assert area_extent == expected
