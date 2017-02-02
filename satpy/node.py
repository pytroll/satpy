#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016
#
# Author(s):
#
#   Martin Raspaud <martin.raspaud@smhi.se>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Nodes to build trees."""


class Node(object):
    """A node object."""

    def __init__(self, name, data=None):
        """Init the node object."""
        self.name = name
        self.data = data
        self.children = []
        self.parents = []

    def flatten(self, d=None):
        if d is None:
            d = {}
        if self.name is not None:
            d[self.name] = self
        for child in self.children:
            child.flatten(d=d)
        return d

    def add_child(self, obj):
        """Add a child to the node."""
        self.children.append(obj)
        obj.parents.append(self)

    def __str__(self):
        """Display the node."""
        return self.display()

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def display(self, previous=0, include_data=False):
        """Display the node."""
        no_data = " (No Data)" if self.data is None else ""
        return (
            (" +" * previous) + str(self.name) + no_data + '\n' +
            ''.join([child.display(previous + 1) for child in self.children]))

    def leaves(self, unique=True):
        """Get the leaves of the tree starting at this root."""
        if not self.children:
            return [self]
        else:
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
        if self.children:
            if self.name is not None:
                res.append(self)
            for child in self.children:
                for sub_child in child.trunk(unique=unique):
                    if not unique or sub_child not in res:
                        res.append(sub_child)
        return res


class DependencyTree(Node):
    def __init__(self):  #, readers, compositors, modifiers):
        # self.readers = readers
        # self.compositors = compositors
        # self.modifiers = modifiers
        # we act as the root node of the tree
        super(DependencyTree, self).__init__(None)
        # keep a flat dictionary of nodes contained in the tree for better
        # __contains__
        self._flattened = {}

    def add_child(self, obj):
        ret = super(DependencyTree, self).add_child(obj)
        # only works if children are added bottom to top (leaves first)
        obj.flatten(d=self._flattened)
        return ret

    def flatten(self, d=None):
        return super(DependencyTree, self).flatten(d=self._flattened)

    def __contains__(self, item):
        return item in self._flattened

    def update(self, other):
        self.flatten()
        for child in other.children:
            self.children.append(child)
