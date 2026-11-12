"""Tests for CF-compatible attributes encoding."""
import json


class TestCFAttributeEncoding:
    """Test case for CF attribute encodings."""

    def test__encode_nc_attrs(self):
        """Test attributes encoding."""
        from satpy.cf.attrs import encode_attrs_to_cf
        from satpy.tests.cf_tests._test_data import get_test_attrs
        from satpy.tests.utils import assert_dict_array_equality

        attrs, expected, _ = get_test_attrs()

        # Test encoding
        encoded = encode_attrs_to_cf(attrs)
        assert_dict_array_equality(expected, encoded)

        # Test decoding of json-encoded attributes
        raw_md_roundtrip = {"recarray": [[0, 0], [0, 0], [0, 0]],
                            "flag": "true",
                            "dict": {"a": 1, "b": [1, 2, 3]}}
        assert json.loads(encoded["raw_metadata"]) == raw_md_roundtrip
        assert json.loads(encoded["array_3d"]) == [[[1, 2], [3, 4]], [[1, 2], [3, 4]]]
        assert json.loads(encoded["nested_dict"]) == {"l1": {"l2": {"l3": [1, 2, 3]}}}
        assert json.loads(encoded["nested_list"]) == ["1", ["2", [3]]]
