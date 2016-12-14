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

    def __init__(self, data):
        """Init the node object."""
        self.data = data
        self.children = []
        self.parents = []

    def add_child(self, obj):
        """Add a child to the node."""
        self.children.append(obj)
        obj.parents.append(self)

    def __str__(self):
        """Display the node."""
        return self.display()

    def display(self, previous=0):
        """Display the node."""
        return (
            (" +" * previous) + str(self.data) + '\n' +
            ''.join([child.display(previous + 1) for child in self.children]))

    def leaves(self):
        """Get the leaves of the tree starting at this root."""
        if not self.children:
            return [self]
        else:
            res = list()
            for child in self.children:
                res.extend(child.leaves())
            return res

    def trunk(self):
        """Get the trunk of the tree starting at this root."""
        res = []
        if self.children:
            if self.data is not None:
                res.append(self)
            for child in self.children:
                res.extend(child.trunk())
        return res
