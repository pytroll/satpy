#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2023 Satpy developers
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
"""Functions and fixture to test CF code."""
import datetime
from collections import OrderedDict

import numpy as np


def get_test_attrs():
    """Create some dataset attributes for testing purpose.

    Returns:
        Attributes, encoded attributes, encoded and flattened attributes

    """
    attrs = {
        "name": "IR_108",
        "start_time": datetime.datetime(2018, 1, 1, 0),
        "end_time": datetime.datetime(2018, 1, 1, 0, 15),
        "int": 1,
        "float": 1.0,
        "none": None,  # should be dropped
        "numpy_int": np.uint8(1),
        "numpy_float": np.float32(1),
        "numpy_bool": True,
        "numpy_void": np.void(0),
        "numpy_bytes": np.bytes_("test"),
        "numpy_string": np.str_("test"),
        "list": [1, 2, np.float64(3)],
        "nested_list": ["1", ["2", [3]]],
        "bool": True,
        "array": np.array([1, 2, 3], dtype="uint8"),
        "array_bool": np.array([True, False, True]),
        "array_2d": np.array([[1, 2], [3, 4]]),
        "array_3d": np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]),
        "dict": {"a": 1, "b": 2},
        "nested_dict": {"l1": {"l2": {"l3": np.array([1, 2, 3], dtype="uint8")}}},
        "raw_metadata": OrderedDict([
            ("recarray", np.zeros(3, dtype=[("x", "i4"), ("y", "u1")])),
            ("flag", np.bool_(True)),
            ("dict", OrderedDict([("a", 1), ("b", np.array([1, 2, 3], dtype="uint8"))]))
            ])
     }
    encoded = {
        "name": "IR_108",
        "start_time": "2018-01-01 00:00:00",
        "end_time": "2018-01-01 00:15:00",
        "int": 1,
        "float": 1.0,
        "numpy_int": np.uint8(1),
        "numpy_float": np.float32(1),
        "numpy_bool": "true",
        "numpy_void": "[]",
        "numpy_bytes": "test",
        "numpy_string": "test",
        "list": [1, 2, np.float64(3)],
        "nested_list": '["1", ["2", [3]]]',
        "bool": "true",
        "array": np.array([1, 2, 3], dtype="uint8"),
        "array_bool": ["true", "false", "true"],
        "array_2d": "[[1, 2], [3, 4]]",
        "array_3d": "[[[1, 2], [3, 4]], [[1, 2], [3, 4]]]",
        "dict": '{"a": 1, "b": 2}',
        "nested_dict": '{"l1": {"l2": {"l3": [1, 2, 3]}}}',
        "raw_metadata": '{"recarray": [[0, 0], [0, 0], [0, 0]], '
                        '"flag": "true", "dict": {"a": 1, "b": [1, 2, 3]}}'
        }
    encoded_flat = {
        "name": "IR_108",
        "start_time": "2018-01-01 00:00:00",
        "end_time": "2018-01-01 00:15:00",
        "int": 1,
        "float": 1.0,
        "numpy_int": np.uint8(1),
        "numpy_float": np.float32(1),
        "numpy_bool": "true",
        "numpy_void": "[]",
        "numpy_bytes": "test",
        "numpy_string": "test",
        "list": [1, 2, np.float64(3)],
        "nested_list": '["1", ["2", [3]]]',
        "bool": "true",
        "array": np.array([1, 2, 3], dtype="uint8"),
        "array_bool": ["true", "false", "true"],
        "array_2d": "[[1, 2], [3, 4]]",
        "array_3d": "[[[1, 2], [3, 4]], [[1, 2], [3, 4]]]",
        "dict_a": 1,
        "dict_b": 2,
        "nested_dict_l1_l2_l3": np.array([1, 2, 3], dtype="uint8"),
        "raw_metadata_recarray": "[[0, 0], [0, 0], [0, 0]]",
        "raw_metadata_flag": "true",
        "raw_metadata_dict_a": 1,
        "raw_metadata_dict_b": np.array([1, 2, 3], dtype="uint8")
    }
    return attrs, encoded, encoded_flat
