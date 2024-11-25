#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 Satpy developers
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

"""Tests for CF decoding."""

import datetime as dt

import pytest

import satpy.cf.decoding


class TestDecodeAttrs:
    """Test decoding of CF-encoded attributes."""

    @pytest.fixture
    def attrs(self):
        """Get CF-encoded attributes."""
        return {
            "my_integer": 0,
            "my_float": 0.0,
            "my_list": [1, 2, 3],
            "my_timestamp1": "2000-01-01",
            "my_timestamp2": "2000-01-01 12:15:33",
            "my_timestamp3": "2000-01-01 12:15:33.123456",
            "my_dict": '{"a": {"b": [1, 2, 3]}, "c": {"d": "2000-01-01 12:15:33.123456"}}'
        }

    @pytest.fixture
    def expected(self):
        """Get expected decoded results."""
        return {
            "my_integer": 0,
            "my_float": 0.0,
            "my_list": [1, 2, 3],
            "my_timestamp1": dt.datetime(2000, 1, 1),
            "my_timestamp2": dt.datetime(2000, 1, 1, 12, 15, 33),
            "my_timestamp3": dt.datetime(2000, 1, 1, 12, 15, 33, 123456),
            "my_dict": {"a": {"b": [1, 2, 3]},
                        "c": {"d": dt.datetime(2000, 1, 1, 12, 15, 33, 123456)}}
        }

    def test_decoding(self, attrs, expected):
        """Test decoding of CF-encoded attributes."""
        res = satpy.cf.decoding.decode_attrs(attrs)
        assert res == expected

    def test_decoding_doesnt_modify_original(self, attrs):
        """Test that decoding doesn't modify the original attributes."""
        satpy.cf.decoding.decode_attrs(attrs)
        assert isinstance(attrs["my_dict"], str)
