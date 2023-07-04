# Copyright (c) 2018-2020 Satpy developers
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
"""Tests for reading a format from an XML file."""

import pytest


@pytest.mark.parametrize("form", ["eps_avhrrl1b_6.5.xml", "eps_iasil2_9.0.xml"])
def test_parse_format(form):
    """Test parsing the XML format."""
    from satpy._config import get_config_path
    from satpy.readers.xmlformat import parse_format
    filename = get_config_path(form)
    parse_format(filename)
