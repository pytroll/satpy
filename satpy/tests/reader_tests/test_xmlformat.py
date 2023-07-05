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

from xml.etree.ElementTree import Element

import numpy as np
import pytest


@pytest.mark.parametrize("form", ["eps_avhrrl1b_6.5.xml", "eps_iasil2_9.0.xml"])
def test_parse_format(form):
    """Test parsing the XML format."""
    from satpy._config import get_config_path
    from satpy.readers.xmlformat import parse_format
    filename = get_config_path(form)
    (_, _, _) = parse_format(filename)


def test_process_array(monkeypatch):
    """Test processing an array tag."""
    from satpy.readers import xmlformat
    from satpy.readers.xmlformat import process_array
    elt = Element(
        "array",
        {"name": "SCENE_RADIANCES",
         "length": "5",
         "labels": "channel1,channel2,channel3,channel4,channel5"})
    elt2 = Element(
        "array",
        {"length": "$NE", "label": "FOV"})
    elt3 = Element(
        "field",
        {"type": "integer2",
         "scaling-factor": "10^2,10^2,10^4,10^2,10^2"})
    elt.append(elt2)
    elt2.append(elt3)
    monkeypatch.setattr(xmlformat, "VARIABLES", {"NE": 10})
    dims = {}
    (name, tp, shp, scl) = process_array(elt, False, dims)
    assert name == "SCENE_RADIANCES"
    assert tp == ">i2"
    assert shp == (5, 10)
    np.testing.assert_allclose(
        scl,
        np.array([0.01, 0.01, 0.0001, 0.01, 0.01]))
    assert dims == {"FOV": "NE"}
