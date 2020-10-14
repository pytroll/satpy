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
"""Classes used as DataID element types."""

import numbers
from collections import namedtuple

try:
    zlklass = namedtuple("ZLevel", "value units", defaults=('hPa',))
except NameError:  # python 3.6
    zlklass = namedtuple("ZLevel", "value units")
    zlklass.__new__.__defaults__ = ('hPa',)


class ZLevel(zlklass):
    """Container for vertical level information in a DataID."""

    @classmethod
    def convert(cls, zlevel):
        """Convert `zlevel` to this type if possible."""
        if isinstance(zlevel, (tuple, list)):
            return cls(*zlevel)
        elif isinstance(zlevel, numbers.Number):
            return cls(zlevel)
        return zlevel

    def __eq__(self, other):
        """Return if two levels are the same.

        Args:
            other (ZLevel, tuple, list, or scalar): Another ZLevel object, a
                scalar level value, or a tuple/list with either a scalar or
                (value, units_str).

        Return:
            True if other is a scalar and equals this value or if other is
            a tuple equal to self, False otherwise.

        """
        if other is None:
            return False
        is_scalar = isinstance(other, numbers.Number)
        is_single_seq = isinstance(other, (tuple, list)) and len(other) == 1
        if is_scalar or is_single_seq:
            return self == self.convert(other)
        return super().__eq__(other)

    def __ne__(self, other):
        """Return the opposite of `__eq__`."""
        return not self == other

    def __lt__(self, other):
        """Compare to another level."""
        if other is None:
            return False
        # compare using units first
        return self[::-1].__lt__(other[::-1])

    def __gt__(self, other):
        """Compare to another level."""
        if other is None:
            return True
        # compare using units first
        return self[::-1].__gt__(other[::-1])

    def __hash__(self):
        """Hash this tuple."""
        return tuple.__hash__(self)

    def __str__(self):
        """Format for print out."""
        return "{0.value} {0.units}".format(self)
