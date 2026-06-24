# Copyright (c) 2026 Satpy developers
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
"""Unit tests for instrument helpers."""

import pytest

import satpy
import satpy._instruments as inst_utils


@pytest.mark.parametrize(
    ("attrs", "to_internal", "expected"),
    [
        ({"instruments": {"AVHRR/3"}}, False, {"AVHRR/3"}),
        ({"instruments": {"AVHRR/3"}}, True, {"avhrr-3"}),
        ({}, False, set()),
    ]
)
def test_get_instruments_from_attrs(attrs, to_internal, expected):
    """Test getting instruments from dataset attributes."""
    assert inst_utils.get_instruments_from_attrs(attrs, to_internal) == expected


# 8< v1.0
@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        ({"sensor": "myinstr"}, {"myinstr"}),
        ({"sensor": {"myinstr"}}, {"myinstr"}),
        ({"instruments": "myinstr"}, {"myinstr"}),
    ]
)
def test_get_instruments_from_attrs_with_warning(attrs, expected):
    """Test deprecation warnings when getting instruments."""
    with pytest.warns(DeprecationWarning, match="v1.0"):
        assert inst_utils.get_instruments_from_attrs(attrs) == expected
# >8 v1.0


@pytest.mark.parametrize(
    ("instrument", "expected"),
    [
        ("AVHRR/3", "avhrr-3"),
        ("IMAGER (GOES 8-11)", "imager_goes_8-11"),
        ("MERSI-1", "mersi-1"),
        ("MSU-GS/A", "msu-gs-a"),
    ]
)
def test_wmo_to_internal(instrument, expected):
    """Test conversion to internal instrument name."""
    assert inst_utils.wmo_to_internal(instrument) == expected


def test_join_instruments():
    """Test joining a set of instruments."""
    instruments = {"mersi-1", "abi"}
    expected = "abi-mersi-1"
    assert inst_utils.join_instrument_names(instruments) == expected


def test_set_instruments_attr():
    """Test setting instruments attribute."""
    attrs = {"instruments": {"myinstrument"}}
    new_instruments = {"i1", "i2"}
    with satpy.config.set(instruments_key="instruments"):
        inst_utils.set_instruments_attr(attrs, new_instruments)
        assert attrs["instruments"] == new_instruments


def test_get_one_instrument_from_attrs():
    """Test getting a single instrument from dataset attributes."""
    attrs = {"instruments": {"i1"}}
    with satpy.config.set(instruments_key="instruments"):
        assert inst_utils.get_one_instrument_from_attrs(attrs) == "i1"


def test_get_one_instrument_from_attrs_with_warning(caplog):
    """Test warnings when getting a single instrument."""
    attrs = {"instruments": {"i1", "i2"}}
    with satpy.config.set(instruments_key="instruments"):
        inst_utils.get_one_instrument_from_attrs(attrs)
        assert "More than one" in caplog.text
        with pytest.raises(KeyError):
            inst_utils.get_one_instrument_from_attrs({})



@pytest.mark.parametrize(
    ("instrument", "expected"),
    [
        ("abi", "ABI"),
        ("ABI", "ABI"),
    ]
)
def test_internal_to_wmo(instrument, expected):
    """Test conversion to WMO instrument name."""
    assert inst_utils.internal_to_wmo(instrument) == expected
