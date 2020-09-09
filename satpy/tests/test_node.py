#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Satpy developers
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
"""Unit tests for the dependency tree class and dependencies."""

import unittest
from satpy.node import CompositorNode


class FakeCompositor:
    """A fake compositor."""

    def __init__(self, id):
        """Set up the fake compositor."""
        self.id = id


class TestCompositorNode(unittest.TestCase):
    """Test case for the compositor node object."""

    def setUp(self):
        """Set up the test case."""
        self.name = 'hej'
        self.fake = FakeCompositor(self.name)
        self.c_node = CompositorNode(self.fake)

    def test_compositor_node_init(self):
        """Test compositor node initialization."""
        assert self.c_node.name == self.name
        assert self.fake in self.c_node.data

    def test_add_required_nodes(self):
        """Test adding required nodes."""
        self.c_node.add_required_nodes([1, 2, 3])
        assert self.c_node.required_nodes == [1, 2, 3]

    def test_add_required_nodes_twice(self):
        """Test adding required nodes twice."""
        self.c_node.add_required_nodes([1, 2])
        self.c_node.add_required_nodes([3])
        assert self.c_node.required_nodes == [1, 2, 3]

    def test_add_optional_nodes(self):
        """Test adding optional nodes."""
        self.c_node.add_optional_nodes([1, 2, 3])
        assert self.c_node.optional_nodes == [1, 2, 3]

    def test_add_optional_nodes_twice(self):
        """Test adding optional nodes twice."""
        self.c_node.add_optional_nodes([1, 2])
        self.c_node.add_optional_nodes([3])
        assert self.c_node.optional_nodes == [1, 2, 3]
