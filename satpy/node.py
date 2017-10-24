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

import logging

from satpy import DatasetDict, DatasetID

LOG = logging.getLogger(__name__)


class Node(object):
    """A node object."""

    def __init__(self, name, data=None):
        """Init the node object."""
        self.name = name
        self.data = data
        self.children = []
        self.parents = []

    @property
    def is_leaf(self):
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

    def copy(self):
        s = Node(self.name, self.data)
        for c in self.children:
            c = c.copy()
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
        return "<Node ({})>".format(repr(self.name))

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
    """Structure to discover and store `Dataset` dependencies

    Used primarily by the `Scene` object to organize dependency finding.
    Dependencies are stored used a series of `Node` objects which this
    class is a subclass of.

    """

    def __init__(self, readers, compositors, modifiers):
        """Collect Dataset generating information.

        Collect the objects that generate and have information about Datasets
        including objects that may depend on certain Datasets being generated.
        This includes readers, compositors, and modifiers.

        Args:
            readers (dict): Reader name -> Reader Object
            compositors (dict): Sensor name -> Composite ID -> Composite Object
            modifiers (dict): Sensor name -> Modifier name -> (Modifier Class, modifier options)

        """
        self.readers = readers
        self.compositors = compositors
        self.modifiers = modifiers
        # we act as the root node of the tree
        super(DependencyTree, self).__init__(None)

        # keep a flat dictionary of nodes contained in the tree for better
        # __contains__
        self._all_nodes = DatasetDict()

    def leaves(self, nodes=None, unique=True):
        """Get the leaves of the tree starting at this root.

        Args:
            nodes (iterable): limit leaves for these node names
            unique: only include individual leaf nodes once

        Returns:
            list of leaf nodes

        """
        if nodes is None:
            return super(DependencyTree, self).leaves(unique=unique)

        res = list()
        for child_id in nodes:
            for sub_child in self._all_nodes[child_id].leaves(unique=unique):
                if not unique or sub_child not in res:
                    res.append(sub_child)
        return res

    def trunk(self, nodes=None, unique=True):
        """Get the trunk nodes of the tree starting at this root.

        Args:
            nodes (iterable): limit trunk nodes to the names specified or the
                              children of them that are also trunk nodes.
            unique: only include individual trunk nodes once

        Returns:
            list of trunk nodes

        """
        if nodes is None:
            return super(DependencyTree, self).trunk(unique=unique)

        res = list()
        for child_id in nodes:
            for sub_child in self._all_nodes[child_id].trunk(unique=unique):
                if not unique or sub_child not in res:
                    res.append(sub_child)
        return res

    def add_child(self, parent, child):
        Node.add_child(parent, child)
        self._all_nodes[child.name] = child

    def add_leaf(self, ds_id, parent=None):
        if parent is None:
            parent = self
        self.add_child(parent, Node(ds_id))

    def copy(self):
        """Copy the this node tree

        Note all references to readers are removed. This is meant to avoid
        tree copies accessing readers that would return incompatible (Area)
        data. Theoretically it should be possible for tree copies to request
        compositor or modifier information as long as they don't depend on
        any datasets not already existing in the dependency tree.
        """
        new_tree = DependencyTree({}, self.compositors, self.modifiers)
        for c in self.children:
            c = c.copy()
            new_tree.add_child(new_tree, c)
        new_tree._all_nodes = new_tree.flatten(d=self._all_nodes)
        return new_tree

    def __contains__(self, item):
        return item in self._all_nodes

    def __getitem__(self, item):
        return self._all_nodes[item]

    def get_compositor(self, key):
        for sensor_name in self.compositors.keys():
            try:
                return self.compositors[sensor_name][key]
            except KeyError:
                continue

        if isinstance(key, DatasetID) and key.modifiers:
            # we must be generating a modifier composite
            return self.get_modifier(key)

        raise KeyError("Could not find compositor '{}'".format(key))

    def get_modifier(self, comp_id):
        # create a DatasetID for the compositor we are generating
        modifier = comp_id.modifiers[-1]
        # source_id = DatasetID(*comp_id[:-1] + (comp_id.modifiers[:-1]))
        for sensor_name in self.modifiers.keys():
            modifiers = self.modifiers[sensor_name]
            compositors = self.compositors[sensor_name]
            if modifier not in modifiers:
                continue

            mloader, moptions = modifiers[modifier]
            moptions = moptions.copy()
            moptions.update(comp_id.to_dict())
            # moptions['prerequisites'] = (
            #     [source_id] + moptions['prerequisites'])
            moptions['sensor'] = sensor_name
            compositors[comp_id] = mloader(**moptions)
            return compositors[comp_id]

        return KeyError("Could not find modifier '{}'".format(modifier))

    def _find_reader_dataset(self,
                             dataset_key,
                             calibration=None,
                             polarization=None,
                             resolution=None):
        for reader_name, reader_instance in self.readers.items():
            try:
                dfilter = {'calibration': calibration,
                           'polarization': polarization,
                           'resolution': resolution}
                ds_id = reader_instance.get_dataset_key(dataset_key, dfilter)
            except KeyError as err:
                # LOG.debug("Can't find dataset %s in reader %s",
                #           str(dataset_key), reader_name)
                pass
            else:
                # LOG.debug("Found {} in reader {}".format(str(ds_id), reader_name))
                return Node(ds_id, {'reader_name': reader_name})

    def _get_compositor_prereqs(self, prereq_names, skip=False):
        """Determine prerequisite Nodes for a composite.

        Args:
            prereq_names (sequence): Strings (names), floats (wavelengths), or
                                     DatasetIDs to analyze.
            skip (bool, optional): If True, prerequisites are considered
                                   optional if they can't be found and a
                                   debug message is logged. If False (default),
                                   the missing prerequisites are not logged
                                   and are expected to be handled by the
                                   caller.

        """
        prereq_ids = []
        unknowns = set()
        for prereq in prereq_names:
            n, u = self._find_dependencies(prereq)
            if u:
                unknowns.update(u)
                if skip:
                    u_str = ", ".join([str(x) for x in u])
                    LOG.debug('Skipping optional %s: Unknown dataset %s',
                              str(prereq), u_str)
            else:
                prereq_ids.append(n)
        return prereq_ids, unknowns

    def _find_compositor(self, dataset_key):
        """Find the compositor object for the given dataset_key."""
        # NOTE: This function can not find a modifier that performs one or more modifications
        # if it has modifiers see if we can find the unmodified version first
        src_node = None
        if isinstance(dataset_key, DatasetID) and dataset_key.modifiers:
            new_prereq = DatasetID(
                *dataset_key[:-1] + (dataset_key.modifiers[:-1],))
            src_node, u = self._find_dependencies(new_prereq)
            if u:
                return None, u

        try:
            compositor = self.get_compositor(dataset_key)
        except KeyError:
            raise KeyError("Can't find anything called {}".format(
                str(dataset_key)))

        dataset_key = compositor.id
        # 2.1 get the prerequisites
        prereqs, unknowns = self._get_compositor_prereqs(
            compositor.info['prerequisites'])
        if unknowns:
            return None, unknowns

        optional_prereqs, _ = self._get_compositor_prereqs(
            compositor.info['optional_prerequisites'],
            skip=True)

        # Is this the right place for that?
        if src_node is not None:
            prereqs.insert(0, src_node)
        root = Node(dataset_key, data=(compositor, prereqs, optional_prereqs))
        # LOG.debug("Found composite {}".format(str(dataset_key)))
        for prereq in prereqs + optional_prereqs:
            if prereq is not None:
                self.add_child(root, prereq)

        return root, set()

    def _find_dependencies(self,
                           dataset_key,
                           calibration=None,
                           polarization=None,
                           resolution=None):
        """Find the dependencies for *dataset_key*.

        Args:
            dataset_key (str, float, DatasetID): Dataset identifier to locate
                                                 and find any additional
                                                 dependencies for.
            calibration (list): List of calibration string levels to load from
                                a reader in order of preference. This is a
                                convenience so individual DatasetIDs don't
                                have to be created by the caller.
            polarization (list): List of polarization strings to load from a
                                 reader in order of preference. This is a
                                 convenience so individual DatasetIDs don't
                                 have to be created by the caller.
            resolution (list): List of resolution levels to load from a
                               reader in order of preference. This is a
                               convenience so individual DatasetIDs don't
                               have to be created by the caller.

        """
        # 0 check if the dataset is already loaded
        try:
            if dataset_key in self:
                return self[dataset_key], set()
        except KeyError:
            # there could be more than one matching dataset, which is fine
            pass

        # 1 try to get dataset from reader
        node = self._find_reader_dataset(dataset_key,
                                         calibration=calibration,
                                         polarization=polarization,
                                         resolution=resolution)
        if node is not None:
            return node, set()

        # 2 try to find a composite that matches
        try:
            node, unknowns = self._find_compositor(dataset_key)
        except KeyError:
            node = None
            unknowns = set([dataset_key])

        return node, unknowns

    def find_dependencies(self,
                          dataset_keys,
                          calibration=None,
                          polarization=None,
                          resolution=None):
        """Create the dependency tree.

        Args:
            dataset_keys (iterable): Strings or DatasetIDs to find dependencies for
            calibration (iterable or None):

        Returns:
            (Node, set): Root node of the dependency tree and a set of unknown datasets

        """
        unknown_datasets = set()
        for key in dataset_keys.copy():
            if key in self:
                n = self[key]
                unknowns = None
            else:
                n, unknowns = self._find_dependencies(key,
                                                      calibration=calibration,
                                                      polarization=polarization,
                                                      resolution=resolution)

            dataset_keys.discard(key)  # remove old non-DatasetID
            if n is not None:
                dataset_keys.add(n.name)  # add equivalent DatasetID
            if unknowns:
                unknown_datasets.update(unknowns)
                continue

            self.add_child(self, n)

        return unknown_datasets
