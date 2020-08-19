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

from satpy.dataset import DataID, DataQuery, ModifierTuple
from satpy.readers import TooManyResults, get_key
from satpy.utils import get_logger
from satpy.dataset import create_filtered_query

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


class DependencyTree:
    """Structure to discover and store `Dataset` dependencies.

    Used primarily by the `Scene` object to organize dependency finding.
    Dependencies are stored used a series of `Node` objects which this
    class is a subclass of.

    """

    # simplify future logic by only having one "sentinel" empty node
    # making it a class attribute ensures it is the same across instances
    empty_node = Node(EMPTY_LEAF_NAME)

    def __init__(self, readers, compositors, modifiers, available_only=False):
        """Collect Dataset generating information.

        Collect the objects that generate and have information about Datasets
        including objects that may depend on certain Datasets being generated.
        This includes readers, compositors, and modifiers.

        Args:
            readers (dict): Reader name -> Reader Object
            compositors (dict): Sensor name -> Composite ID -> Composite Object
            modifiers (dict): Sensor name -> Modifier name -> (Modifier Class, modifier options)
            available_only (bool): Whether only reader's available/loadable
                datasets should be used when searching for dependencies (True)
                or use all known/configured datasets regardless of whether the
                necessary files were provided to the reader (False).
                Note that when ``False`` loadable variations of a dataset will
                have priority over other known variations.
                Default is ``False``.

        """
        self.readers = readers
        self.compositors = compositors
        self.modifiers = modifiers
        self._available_only = available_only
        self._root = Node(None)

        # keep a flat dictionary of nodes contained in the tree for better
        # __contains__
        self._all_nodes = _DataIDContainer()

    def leaves(self, nodes=None, unique=True):
        """Get the leaves of the tree starting at the root.

        Args:
            nodes (iterable): limit leaves for these node names
            unique: only include individual leaf nodes once

        Returns:
            list of leaf nodes

        """
        if nodes is None:
            return self._root.leaves(unique=unique)

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
            return self._root.trunk(unique=unique)

        res = list()
        for child_id in nodes:
            for sub_child in self._all_nodes[child_id].trunk(unique=unique):
                if not unique or sub_child not in res:
                    res.append(sub_child)
        return res

    def add_child(self, parent, child):
        """Add a child to the tree."""
        Node.add_child(parent, child)
        # Sanity check: Node objects should be unique. They can be added
        #               multiple times if more than one Node depends on them
        #               but they should all map to the same Node object.
        if self.contains(child.name):
            assert self._all_nodes[child.name] is child
        if child is self.empty_node:
            # No need to store "empty" nodes
            return
        self._all_nodes[child.name] = child

    def add_leaf(self, ds_id, parent=None):
        """Add a leaf to the tree."""
        if parent is None:
            parent = self._root
        try:
            node = self[ds_id]
        except KeyError:
            node = Node(ds_id)
        self.add_child(parent, node)
        return node

    def copy(self):
        """Copy this node tree.

        Note all references to readers are removed. This is meant to avoid
        tree copies accessing readers that would return incompatible (Area)
        data. Theoretically it should be possible for tree copies to request
        compositor or modifier information as long as they don't depend on
        any datasets not already existing in the dependency tree.
        """
        new_tree = DependencyTree({}, self.compositors, self.modifiers)
        for c in self._root.children:
            c = c.copy(node_cache=new_tree._all_nodes)
            new_tree.add_child(new_tree._root, c)
        return new_tree

    def __contains__(self, item):
        """Check if a item is in the tree."""
        return item in self._all_nodes

    def __getitem__(self, item):
        """Get an item of the tree."""
        return self._all_nodes[item]

    def contains(self, item):
        """Check contains when we know the *exact* DataID or DataQuery."""
        return super(_DataIDContainer, self._all_nodes).__contains__(item)

    def getitem(self, item):
        """Get Node when we know the *exact* DataID or DataQuery."""
        return super(_DataIDContainer, self._all_nodes).__getitem__(item)

    def __str__(self):
        """Render the dependency tree as a string."""
        return self._root.display()

    def get_compositor(self, key):
        """Get a compositor."""
        for sensor_name in self.compositors.keys():
            try:
                return self.compositors[sensor_name][key]
            except KeyError:
                continue
        if isinstance(key, (DataQuery, DataID)) and key.get('modifiers'):
            # we must be generating a modifier composite
            return self.get_modifier(key)

        raise KeyError("Could not find compositor '{}'".format(key))

    def get_modifier(self, comp_id):
        """Get a modifer."""
        # create a DataID for the compositor we are generating
        modifier = comp_id['modifiers'][-1]
        for sensor_name in self.modifiers.keys():
            modifiers = self.modifiers[sensor_name]
            compositors = self.compositors[sensor_name]
            if modifier not in modifiers:
                continue

            mloader, moptions = modifiers[modifier]
            moptions = moptions.copy()
            moptions.update(comp_id.to_dict())
            moptions['sensor'] = sensor_name
            compositors[comp_id] = mloader(_satpy_id=comp_id, **moptions)
            return compositors[comp_id]

        raise KeyError("Could not find modifier '{}'".format(modifier))

    def _get_compositor_prereqs(self, parent, prereqs, query=None):
        """Determine prerequisite Nodes for a composite.

        Args:
            parent (Node): Compositor node to add these prerequisites under
            prereqs (sequence): Strings (names), floats (wavelengths), or
                                DataQuerys to analyze.

        """
        prereq_ids = []
        unknown_datasets = set()
        if not prereqs:
            # this composite has no required prerequisites
            prereq_ids.append(self.empty_node)
            self.add_child(parent, self.empty_node)
            return prereq_ids

        for prereq in prereqs:
            try:
                node = self._create_subtree_for_key(prereq, query=query)
            except MissingDependencies as unknown:
                unknown_datasets.update(unknown.missing_dependencies)

            else:
                prereq_ids.append(node)
                self.add_child(parent, node)
        if unknown_datasets:
            raise MissingDependencies(unknown_datasets)
        return prereq_ids

    def _get_compositor_optional_prereqs(self, parent, prereqs, query=None):
        """Determine optional prerequisite Nodes for a composite.

        Args:
            parent (Node): Compositor node to add these prerequisites under
            prereqs (sequence): Strings (names), floats (wavelengths), or
                                DataQuerys to analyze.

        """
        prereq_ids = []

        for prereq in prereqs:
            try:
                node = self._create_subtree_for_key(prereq, query=query)
            except MissingDependencies as unknown:
                u_str = ", ".join([str(x) for x in unknown.missing_dependencies])
                LOG.debug('Skipping optional %s: Unknown dataset %s',
                          str(prereq), u_str)
            else:
                prereq_ids.append(node)
                self.add_child(parent, node)
        return prereq_ids

    def _promote_query_to_modified_dataid(self, query, dep_key):
        """Promote a query to an id based on the dataset it will modify (dep).

        Typical use case is requesting a modified dataset (orig_key). This
        modified dataset most likely depends on a less-modified
        dataset (dep_key). The less-modified dataset must come from a reader
        (at least for now) or will eventually depend on a reader dataset.
        The original request key may be limited like
        (wavelength=0.67, modifiers=('a', 'b')) while the reader-based key
        should have all of its properties specified. This method updates the
        original request key so it is fully specified and should reduce the
        chance of Node's not being unique.

        """
        orig_dict = query._asdict()
        dep_dict = dep_key._asdict()
        for k, dep_val in dep_dict.items():
            # don't change the modifiers, just cast them to the right class
            if isinstance(dep_val, ModifierTuple):
                orig_dict[k] = dep_val.__class__(orig_dict[k])
            else:
                orig_dict[k] = dep_val
        return dep_key.from_dict(orig_dict)

    def _find_compositor(self, dataset_key):
        """Find the compositor object for the given dataset_key."""
        # NOTE: This function can not find a modifier that performs
        # one or more modifications if it has modifiers see if we can find
        # the unmodified version first

        src_node = self._create_implicit_dependency_node(dataset_key)
        if src_node is not None:
            dataset_key = self._promote_query_to_modified_dataid(dataset_key, src_node.name)

        try:
            compositor = self.get_compositor(dataset_key)
        except KeyError:
            raise KeyError("Can't find anything called {}".format(str(dataset_key)))

        cid = compositor.id
        root = Node(cid, data=(compositor, [], []))
        if src_node is not None:
            self.add_child(root, src_node)
            root.data[1].append(src_node)

        query = cid.create_dep_filter(dataset_key)

        # 2.1 get the prerequisites
        LOG.trace("Looking for composite prerequisites for: {}".format(dataset_key))
        prereqs = self._get_compositor_prereqs(root, compositor.attrs['prerequisites'], query=query)
        root.data[1].extend(prereqs)

        # Get the optionals
        LOG.trace("Looking for optional prerequisites for: {}".format(dataset_key))
        optional_prereqs = self._get_compositor_optional_prereqs(
                root, compositor.attrs['optional_prerequisites'], query=query)
        root.data[2].extend(optional_prereqs)

        return root

    def _create_implicit_dependency_node(self, dataset_key):
        if self._is_modified_key(dataset_key):
            new_prereq = dataset_key._create_less_modified_query()
            src_node = self._create_subtree_for_key(new_prereq)
            return src_node
        else:
            return None

    @staticmethod
    def _is_modified_key(dataset_key):
        modifiers_ = isinstance(dataset_key, DataQuery) and dataset_key.get('modifiers')
        return modifiers_

    def _create_subtree_for_key(self, dataset_key, query=None):
        """Find the dependencies for *dataset_key*.

        Args:
            dataset_key (str, float, DataID, DataQuery): Dataset identifier to locate
                                                         and find any additional
                                                         dependencies for.
            query (DataQuery): Additional filter parameters. See
                               `satpy.readers.get_key` for more details.

        """
        dsq = create_filtered_query(dataset_key, query)
        # 0 check if the *exact* dataset is already loaded
        try:
            node = self._get_subtree_for_existing_key(dsq)
        except MissingDependencies:
            # exact dataset isn't loaded, let's load it below
            pass
        else:
            return node

        # 1 try to get *best* dataset from reader
        try:
            node = self._create_subtree_from_reader(dsq)
        except TooManyResults:
            LOG.warning("Too many possible datasets to load for {}".format(dsq))
            raise MissingDependencies({dataset_key})
        except MissingDependencies:
            pass
        else:
            return node

        # 2 try to find a composite by name (any version of it is good enough)
        try:
            node = self._get_subtree_for_existing_name(dsq)
        except MissingDependencies:
            pass
        else:
            return node

        # 3 try to find a composite that matches
        try:
            node = self._create_subtree_from_compositors(dsq)
        except MissingDependencies:
            raise
        else:
            return node

    def _create_subtree_from_compositors(self, dsq):
        try:
            node = self._find_compositor(dsq)
            LOG.trace("Found composite:\n\tRequested: {}\n\tFound: {}".format(dsq, node and node.name))
        except KeyError:
            LOG.trace("Composite not found: {}".format(dsq))
            raise MissingDependencies({dsq})
        return node

    def _get_subtree_for_existing_name(self, dsq):
        try:
            # assume that there is no such thing as a "better" composite
            # version so if we find any DataIDs already loaded then
            # we want to use them
            node = self[dsq]
            LOG.trace("Composite already loaded:\n\tRequested: {}\n\tFound: {}".format(dsq, node.name))
            return node
        except KeyError:
            # composite hasn't been loaded yet, let's load it below
            LOG.trace("Composite hasn't been loaded yet, will load: {}".format(dsq))
            raise MissingDependencies({dsq})

    def _create_subtree_from_reader(self, dsq):
        try:
            node = self._find_reader_dataset(dsq)
        except MissingDependencies:
            LOG.trace("Could not find dataset in reader: {}".format(dsq))
            raise
        else:
            LOG.trace("Found reader provided dataset:\n\tRequested: {}\n\tFound: {}".format(dsq, node.name))
            return node

    def _find_reader_dataset(self, dataset_key):
        """Attempt to find a `DataID` in the available readers.

        Args:
            dataset_key (str, float, DataID, DataQuery):
                Dataset name, wavelength, `DataID` or `DataQuery`
                to use in searching for the dataset from the
                available readers.

        """
        too_many = False
        for reader_name, reader_instance in self.readers.items():
            try:
                ds_id = reader_instance.get_dataset_key(dataset_key, available_only=self._available_only)
            except TooManyResults:
                LOG.trace("Too many datasets matching key {} in reader {}".format(dataset_key, reader_name))
                too_many = True
                continue
            except KeyError:
                LOG.trace("Can't find dataset %s in reader %s", str(dataset_key), reader_name)
                continue
            LOG.trace("Found {} in reader {} when asking for {}".format(str(ds_id), reader_name, repr(dataset_key)))
            try:
                # now that we know we have the exact DataID see if we have already created a Node for it
                return self.getitem(ds_id)
            except KeyError:
                # we haven't created a node yet, create it now
                return Node(ds_id, {'reader_name': reader_name})
        if too_many:
            raise TooManyResults("Too many keys matching: {}".format(dataset_key))
        raise MissingDependencies({dataset_key})

    def _get_subtree_for_existing_key(self, dsq):
        try:
            node = self.getitem(dsq)
            LOG.trace("Found exact dataset already loaded: {}".format(node.name))
            return node
        except KeyError:
            LOG.trace("Exact dataset {} isn't loaded, will try reader...".format(dsq))
            raise MissingDependencies({dsq})

    def populate_with_keys(self, dataset_keys: set, query=None):
        """Populate the dependency tree.

        Args:
            dataset_keys (set): Strings, DataIDs, DataQuerys to find dependencies for
            query (DataQuery): Additional filter parameters. See
                              `satpy.readers.get_key` for more details.

        Returns:
            (Node, set): Root node of the dependency tree and a set of unknown datasets

        """
        unknown_datasets = set()
        known_nodes = list()
        for key in dataset_keys.copy():
            try:
                node = self._create_subtree_for_key(key, query)
            except MissingDependencies as unknown:
                unknown_datasets.update(unknown.missing_dependencies)
            else:
                known_nodes.append(node)
                self.add_child(self._root, node)

        for key in dataset_keys.copy():
            dataset_keys.discard(key)
        for node in known_nodes:
            dataset_keys.add(node.name)
        if unknown_datasets:
            raise MissingDependencies(unknown_datasets)


class _DataIDContainer(dict):
    """Special dictionary object that can handle dict operations based on dataset name, wavelength, or DataID.

    Note: Internal dictionary keys are `DataID` objects.

    """

    def keys(self):
        """Give currently contained keys."""
        # sort keys so things are a little more deterministic (.keys() is not)
        return sorted(super(_DataIDContainer, self).keys())

    def get_key(self, match_key):
        """Get multiple fully-specified keys that match the provided query.

        Args:
            match_key (DataID): DataID or DataQuery of query parameters to use for
                                searching. Can also be a string representing the
                                dataset name or a number representing the dataset
                                wavelength.

        """
        return get_key(match_key, self.keys())

    def __getitem__(self, item):
        """Get item from container."""
        try:
            # short circuit - try to get the object without more work
            return super(_DataIDContainer, self).__getitem__(item)
        except KeyError:
            key = self.get_key(item)
            return super(_DataIDContainer, self).__getitem__(key)

    def __contains__(self, item):
        """Check if item exists in container."""
        try:
            key = self.get_key(item)
        except KeyError:
            return False
        return super(_DataIDContainer, self).__contains__(key)
