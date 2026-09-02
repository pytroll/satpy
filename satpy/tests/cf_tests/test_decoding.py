
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
