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
import satpy._instruments as instru


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        ({"instruments": {"myinstr"}}, {"myinstr"}),
        ({}, set()),
    ]
)
def test_get_instruments_from_attrs(attrs, expected):
    """Test getting instruments from dataset attributes."""
    assert instru.get_instruments_from_attrs(attrs) == expected

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
    with pytest.warns(DeprecationWarning, match="v1.1"):
        assert instru.get_instruments_from_attrs(attrs) == expected

def test_normalize_instrument_name():
    """Test instrument name normalization."""
    instr = "My Instrument-123/1 (My Platform)"
    expected = "my_instrument123-1_my_platform"
    assert instru.normalize_instrument_name(instr) == expected

def test_serialize_instruments():
    """Test instrument set serialization."""
    instruments = {"My Instrument-123/1 (My Platform)", "ABI"}
    expected = "abi-myinstrument1231myplatform"
    assert instru.serialize_instruments(instruments) == expected

def test_set_instruments_attr():
    """Test setting instruments attribute."""
    attrs = {"instruments": {"myinstrument"}}
    new_instruments = {"i1", "i2"}
    with satpy.config.set(instruments_key="instruments"):
        instru.set_instruments_attr(attrs, new_instruments)
        assert attrs["instruments"] == new_instruments

def test_get_one_instrument_from_attrs():
    """Test getting a single instrument from dataset attributes."""
    attrs = {"instruments": {"i1"}}
    with satpy.config.set(instruments_key="instruments"):
        assert instru.get_one_instrument_from_attrs(attrs) == "i1"

def test_get_one_instrument_from_attrs_with_warning(caplog):
    """Test warnings when getting a single instrument."""
    attrs = {"instruments": {"i1", "i2"}}
    with satpy.config.set(instruments_key="instruments"):
        instru.get_one_instrument_from_attrs(attrs)
        assert "More than one" in caplog.text
        with pytest.raises(KeyError):
            instru.get_one_instrument_from_attrs({})
