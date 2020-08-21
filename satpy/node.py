#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2019 Satpy developers
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
"""Nodes to build trees."""

from satpy.utils import get_logger

LOG = get_logger(__name__)
# Empty leaf used for marking composites with no prerequisites
EMPTY_LEAF_NAME = "__EMPTY_LEAF_SENTINEL__"


class MissingDependencies(RuntimeError):
    """Exception when dependencies are missing."""

    def __init__(self, missing_dependencies, *args, **kwargs):
        """Set up the exception."""
        super().__init__(*args, **kwargs)
        self.missing_dependencies = missing_dependencies


class Node:
    """A node object."""

    def __init__(self, name, data=None):
        """Init the node object."""
        self.name = name
        self.data = data
        self.children = []
        self.parents = []

    @property
    def is_leaf(self):
        """Check if the node is a leaf."""
        return not self.children

    def flatten(self, d=None):
        """Flatten tree structure to a one level dictionary.

        Args:
            d (dict, optional): output dictionary to update

        Returns:
            dict: Node.name -> Node. The returned dictionary includes the
                  current Node and all its children.

        """
        if d is None:
            d = {}
        if self.name is not None:
            d[self.name] = self
        for child in self.children:
            child.flatten(d=d)
        return d

    def copy(self, node_cache=None):
        """Make a copy of the node."""
        if node_cache and self.name in node_cache:
            return node_cache[self.name]

        if self.name is EMPTY_LEAF_NAME:
            return self

        s = Node(self.name, self.data)
        for c in self.children:
            c = c.copy(node_cache=node_cache)
            s.add_child(c)
        return s

    def add_child(self, obj):
        """Add a child to the node."""
        self.children.append(obj)
        obj.parents.append(self)

    def __str__(self):
        """Display the node."""
        return self.display()

    def __repr__(self):
        """Generate a representation of the node."""
        return "<Node ({})>".format(repr(self.name))

    def __eq__(self, other):
        """Check equality."""
        return self.name == other.name

    def __hash__(self):
        """Generate the hash of the node."""
        return hash(self.name)

    def display(self, previous=0, include_data=False):
        """Display the node."""
        no_data = " (No Data)" if self.data is None else ""
        return (
            (" +" * previous) + str(self.name) + no_data + '\n' +
            ''.join([child.display(previous + 1) for child in self.children]))

    def leaves(self, unique=True):
        """Get the leaves of the tree starting at this root."""
        if self.name is EMPTY_LEAF_NAME:
            return []
        elif not self.children:
            return [self]

        res = list()
        for child in self.children:
            for sub_child in child.leaves(unique=unique):
                if not unique or sub_child not in res:
                    res.append(sub_child)
        return res

    def trunk(self, unique=True):
        """Get the trunk of the tree starting at this root."""
        # uniqueness is not correct in `trunk` yet
        unique = False
        res = []
        if self.children and self.name is not EMPTY_LEAF_NAME:
            if self.name is not None:
                res.append(self)
            for child in self.children:
                for sub_child in child.trunk(unique=unique):
                    if not unique or sub_child not in res:
                        res.append(sub_child)
        return res


class CompositorNode(Node):
    """Implementation of a compositor-specific node."""

    def __init__(self, compositor):
        """Set up the node."""
        super().__init__(compositor.id, data=(compositor, [], []))

    def add_required_nodes(self, children):
        """Add nodes to the required field."""
        self.data[1].extend(children)

    @property
    def required_nodes(self):
        """Get the required nodes."""
        return self.data[1]

    def add_optional_nodes(self, children):
        """Add nodes to the optional field."""
        self.data[2].extend(children)

    @property
    def optional_nodes(self):
        """Get the optional nodes."""
        return self.data[2]
