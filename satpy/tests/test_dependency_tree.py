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

import os
import unittest

from satpy.dependency_tree import DependencyTree
from satpy.tests.utils import make_cid, make_dataid


class TestDependencyTree(unittest.TestCase):
    """Test the dependency tree.

    This is what we are working with::

        None (No Data)
         +DataID(name='comp19')
         + +DataID(name='ds5', resolution=250, modifiers=('res_change',))
         + + +DataID(name='ds5', resolution=250, modifiers=())
         + + +__EMPTY_LEAF_SENTINEL__ (No Data)
         + +DataID(name='comp13')
         + + +DataID(name='ds5', resolution=250, modifiers=('res_change',))
         + + + +DataID(name='ds5', resolution=250, modifiers=())
         + + + +__EMPTY_LEAF_SENTINEL__ (No Data)
         + +DataID(name='ds2', resolution=250, calibration=<calibration.reflectance>, modifiers=())

    """

    def setUp(self):
        """Set up the test tree."""
        self.dependency_tree = DependencyTree(None, None, None)

        composite_1 = make_cid(name="comp19")
        dependency_1 = make_dataid(name="ds5", resolution=250, modifiers=("res_change",))
        dependency_1_1 = make_dataid(name="ds5", resolution=250, modifiers=tuple())
        node_composite_1 = self.dependency_tree.add_leaf(composite_1)
        node_dependency_1 = self.dependency_tree.add_leaf(dependency_1, node_composite_1)
        self.dependency_tree.add_leaf(dependency_1_1, node_dependency_1)
        # ToDo: do we really want then empty node to be at the same level as the unmodified data?
        node_dependency_1.add_child(self.dependency_tree.empty_node)

        dependency_2 = make_cid(name="comp13")
        dependency_2_1 = dependency_1
        node_dependency_2 = self.dependency_tree.add_leaf(dependency_2, node_composite_1)
        self.dependency_tree.add_leaf(dependency_2_1, node_dependency_2)
        # We don't need to add the unmodified dependency a second time.

        dependency_3 = make_dataid(name='ds2', resolution=250, calibration="reflectance", modifiers=tuple())
        self.dependency_tree.add_leaf(dependency_3, node_composite_1)

    def test_copy_preserves_unique_empty_node(self):
        """Test that dependency tree copy preserves the uniqueness of the empty node."""
        new_dependency_tree = self.dependency_tree.copy()
        assert self.dependency_tree.empty_node is new_dependency_tree.empty_node

        self.assertIs(self.dependency_tree._root.children[0].children[0].children[1],
                      self.dependency_tree.empty_node)
        self.assertIs(new_dependency_tree._root.children[0].children[0].children[1],
                      self.dependency_tree.empty_node)

    def test_new_dependency_tree_preserves_unique_empty_node(self):
        """Test that dependency tree instantiation preserves the uniqueness of the empty node."""
        new_dependency_tree = DependencyTree(None, None, None)
        assert self.dependency_tree.empty_node is new_dependency_tree.empty_node


class TestMissingDependencies(unittest.TestCase):
    """Test the MissingDependencies exception."""

    def test_new_missing_dependencies(self):
        """Test new MissingDependencies."""
        from satpy.node import MissingDependencies
        error = MissingDependencies('bla')
        assert error.missing_dependencies == 'bla'

    def test_new_missing_dependencies_with_message(self):
        """Test new MissingDependencies with a message."""
        from satpy.node import MissingDependencies
        error = MissingDependencies('bla', "This is a message")
        assert 'This is a message' in str(error)


class TestMultipleResolutionSameChannelDependency(unittest.TestCase):
    """Test that MODIS situations where the same channel is available at multiple resolution works."""

    def test_modis_overview_1000m(self):
        """Test a modis overview dependency calculation with resolution fixed to 1000m."""
        from satpy.config import PACKAGE_CONFIG_PATH
        from satpy.readers.yaml_reader import FileYAMLReader

        from satpy import DataQuery
        from satpy.composites import SunZenithCorrector, GenericCompositor
        from satpy.dataset import DatasetDict

        config_file = os.path.join(PACKAGE_CONFIG_PATH, 'readers', 'modis_l1b.yaml')
        self.reader_instance = FileYAMLReader.from_config_files(config_file)

        overview = {'_satpy_id': make_dataid(name='overview'),
                    'name': 'overview',
                    'optional_prerequisites': [],
                    'prerequisites': [DataQuery(name='1', modifiers=('sunz_corrected',)),
                                      DataQuery(name='2', modifiers=('sunz_corrected',)),
                                      DataQuery(name='31')],
                    'standard_name': 'overview'}
        compositors = {'modis': DatasetDict()}
        compositors['modis']['overview'] = GenericCompositor(**overview)

        modifiers = {'modis': {'sunz_corrected': (SunZenithCorrector,
                                                  {'optional_prerequisites': ['solar_zenith_angle'],
                                                   'name': 'sunz_corrected',
                                                   'prerequisites': []})}}
        dep_tree = DependencyTree({'modis_l1b': self.reader_instance}, compositors, modifiers)
        dep_tree.populate_with_keys({'overview'}, DataQuery(resolution=1000))
        for key in dep_tree._all_nodes.keys():
            assert key.get('resolution', 1000) == 1000
