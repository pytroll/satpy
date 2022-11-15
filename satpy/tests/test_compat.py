#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2022 Satpy developers
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
"""Test backports and compatibility fixes."""

import gc

from satpy._compat import CachedPropertyBackport


class ClassWithCachedProperty:  # noqa
    def __init__(self, x):  # noqa
        self.x = x

    @CachedPropertyBackport
    def property(self):  # noqa
        return 2 * self.x


def test_cached_property_backport():
    """Test cached property backport."""
    c = ClassWithCachedProperty(1)
    assert c.property == 2


def test_cached_property_backport_releases_memory():
    """Test that cached property backport releases memory."""
    c1 = ClassWithCachedProperty(2)
    del c1
    instances = [
        obj for obj in gc.get_objects()
        if isinstance(obj, ClassWithCachedProperty)
    ]
    assert len(instances) == 0
