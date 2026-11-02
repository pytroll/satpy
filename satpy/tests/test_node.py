"""Unit tests for the dependency tree class and dependencies."""

import unittest
from unittest.mock import MagicMock

from satpy.node import CompositorNode


class FakeCompositor:
    """A fake compositor."""

    def __init__(self, id):  # noqa: A002
        """Set up the fake compositor."""
        self.id = id


class TestCompositorNodeCopy(unittest.TestCase):
    """Test case for copying a node."""

    def setUp(self):
        """Set up the test case."""
        self.node = CompositorNode(MagicMock())
        self.node.add_required_nodes([MagicMock(), MagicMock()])
        self.node.add_optional_nodes([MagicMock()])

        self.node_copy = self.node.copy()

    def test_node_data_is_copied(self):
        """Test that the data of the node is copied."""
        assert self.node_copy.data is not self.node.data

    def test_node_data_required_nodes_are_copies(self):
        """Test that the required nodes of the node data are copied."""
        for req1, req2 in zip(self.node.required_nodes, self.node_copy.required_nodes):
            assert req1 is not req2

    def test_node_data_optional_nodes_are_copies(self):
        """Test that the optional nodes of the node data are copied."""
        for req1, req2 in zip(self.node.optional_nodes, self.node_copy.optional_nodes):
            assert req1 is not req2


class TestCompositorNode(unittest.TestCase):
    """Test case for the compositor node object."""

    def setUp(self):
        """Set up the test case."""
        self.name = "hej"
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
