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
"""Base modifier classes and utilities."""

from satpy.composites import CompositeBase


class ModifierBase(CompositeBase):
    """Base class for all modifiers.

    A modifier in Satpy is a class that takes one input DataArray to be
    changed along with zero or more other input DataArrays used to perform
    these changes. The result of a modifier typically has a lot of the same
    metadata (name, units, etc) as the original DataArray, but the data is
    different. A modified DataArray can be differentiated from the original
    DataArray by the `modifiers` property of its `DataID`.

    See the :class:`~satpy.composites.CompositeBase` class for information
    on the similar concept of "compositors".

    """

    def __call__(self, datasets, optional_datasets=None, **info):
        """Generate a modified copy of the first provided dataset."""
        raise NotImplementedError()
