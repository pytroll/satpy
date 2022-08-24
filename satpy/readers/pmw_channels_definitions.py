#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2022 Satpy Developers

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Passive Microwave instrument and channel specific features."""

import numbers
from contextlib import suppress
from typing import NamedTuple

import numpy as np


class FrequencyQuadrupleSideBandBase(NamedTuple):
    """Base class for a frequency quadruple side band.

    Frequency Quadruple Side Band is supposed to describe the special type of
    bands commonly used in temperature sounding from Passive Microwave
    Sensors. When the absorption band being observed is symmetrical it is
    advantageous (giving better NeDT) to sense in a band both right and left of
    the central absorption frequency. But to avoid (CO2) absorption lines
    symmetrically positioned on each side of the main absorption band it is
    common to split the side bands in two 'side-side' bands.

    This is needed because of this bug: https://bugs.python.org/issue41629

    """

    central: float
    side: float
    sideside: float
    bandwidth: float
    unit: str = "GHz"


class FrequencyQuadrupleSideBand(FrequencyQuadrupleSideBandBase):
    """The frequency quadruple side band class.

    The elements of the quadruple-side-band type frequency band are the
    central frquency, the relative (main) side band frequency (relative to the
    center - left and right), the sub-side band frequency (relative to the
    offset side-band(s)) and their bandwidths. Optionally a unit (defaults to
    GHz) may be specified. No clever unit conversion is done here, it's just
    used for checking that two ranges are comparable.

    Frequency Quadruple Side Band is supposed to describe the special type of
    bands commonly used in temperature sounding from Passive Microwave
    Sensors. When the absorption band being observed is symmetrical it is
    advantageous (giving better NeDT) to sense in a band both right and left of
    the central absorption frequency. But to avoid (CO2) absorption lines
    symmetrically positioned on each side of the main absorption band it is
    common to split the side bands in two 'side-side' bands.

    """

    def __eq__(self, other):
        """Return if two channel frequencies are equal.

        Args:
            other (tuple or scalar): (central frq, side band frq, side-side band frq,
            and band width frq) or scalar frq

        Return:
            True if other is a scalar and min <= other <= max, or if other is a
            tuple equal to self, or if other is a number contained by self.
            False otherwise.

        """
        if other is None:
            return False
        if isinstance(other, numbers.Number):
            return other in self
        if isinstance(other, (tuple, list)) and len(other) == 4:
            return other in self
        return super().__eq__(other)

    def __lt__(self, other):
        """Compare to another frequency."""
        if other is None:
            return False
        return super().__lt__(other)

    def __gt__(self, other):
        """Compare to another frequency."""
        if other is None:
            return True
        return super().__gt__(other)

    def __hash__(self):
        """Hash this tuple."""
        return tuple.__hash__(self)

    def __str__(self):
        """Format for print out."""
        return f"central={self.central} {self.unit} ±{self.side} ±{self.sideside} width={self.bandwidth} {self.unit}"

    def __contains__(self, other):
        """Check if this quadruple-side-band 'contains' *other*."""
        if other is None:
            return False

        # The four centrals:
        central_left_left = self.central - self.side - self.sideside
        central_left_right = self.central - self.side + self.sideside
        central_right_left = self.central + self.side - self.sideside
        central_right_right = self.central + self.side + self.sideside

        if isinstance(other, numbers.Number):
            if _is_inside_interval(other, central_left_left, self.bandwidth):
                return True
            if _is_inside_interval(other, central_left_right, self.bandwidth):
                return True
            if _is_inside_interval(other, central_right_left, self.bandwidth):
                return True
            if _is_inside_interval(other, central_right_right, self.bandwidth):
                return True

            return False

        if isinstance(other, (tuple, list)) and len(other) == 5:
            raise NotImplementedError("Can't check if one frequency quadruple side band is contained in another.")

        with suppress(AttributeError):
            if self.unit != other.unit:
                raise NotImplementedError("Can't compare frequency ranges with different units.")

        return False

    def distance(self, value):
        """Get the distance to the quadruple side band.

        Determining the distance in frequency space between two quadruple side
        bands can be quite ambiguous, as such bands are in effect a set of
        4 narrow bands. To keep it as simple as possible we have until further
        decided to define the distance between such two bands by calculating
        the distances between the outermost left-sides of the two bands and the
        distances of the outermost right sides and then minimising those two.

        If the frequency entered is a single value the distance will be the
        minimum of the distances to the two outermost sides of the quadruple
        side band.

        """
        left_left = self.central - self.side - self.sideside
        right_right = self.central + self.side + self.sideside

        if self == value:
            try:
                left_side_dist = abs(value.central - value.side - value.sideside - left_left)
                right_side_dist = abs(value.central + value.side + value.sideside - right_right)
                return min(left_side_dist, right_side_dist)
            except AttributeError:
                if isinstance(value, (tuple, list)):
                    msg = 'Distance to a quadruple side band frequency not supported for this type'
                    raise NotImplementedError(msg)

                left_side_dist = abs(value - left_left)
                right_side_dist = abs(value - right_right)
                return min(left_side_dist, right_side_dist)
        else:
            return np.inf

    @classmethod
    def convert(cls, frq):
        """Convert `frq` to this type if possible."""
        if isinstance(frq, dict):
            return cls(**frq)
        return frq


class FrequencyDoubleSideBandBase(NamedTuple):
    """Base class for a frequency double side band.

    Frequency Double Side Band is supposed to describe the special type of bands
    commonly used in humidty sounding from Passive Microwave Sensors. When the
    absorption band being observed is symmetrical it is advantageous (giving
    better NeDT) to sense in a band both right and left of the central
    absorption frequency.

    This is needed because of this bug: https://bugs.python.org/issue41629

    """

    central: float
    side: float
    bandwidth: float
    unit: str = "GHz"


class FrequencyDoubleSideBand(FrequencyDoubleSideBandBase):
    """The frequency double side band class.

    The elements of the double-side-band type frequency band are the central
    frquency, the relative side band frequency (relative to the center - left
    and right) and their bandwidths, and optionally a unit (defaults to
    GHz). No clever unit conversion is done here, it's just used for checking
    that two ranges are comparable.

    Frequency Double Side Band is supposed to describe the special type of bands
    commonly used in humidty sounding from Passive Microwave Sensors. When the
    absorption band being observed is symmetrical it is advantageous (giving
    better NeDT) to sense in a band both right and left of the central
    absorption frequency.

    """

    def __eq__(self, other):
        """Return if two channel frequencies are equal.

        Args:
            other (tuple or scalar): (central frq, side band frq and band width frq) or scalar frq

        Return:
            True if other is a scalar and min <= other <= max, or if other is a
            tuple equal to self, or if other is a number contained by self.
            False otherwise.

        """
        if other is None:
            return False
        if isinstance(other, numbers.Number):
            return other in self
        if isinstance(other, (tuple, list)) and len(other) == 3:
            return other in self
        return super().__eq__(other)

    def __lt__(self, other):
        """Compare to another frequency."""
        if other is None:
            return False
        return super().__lt__(other)

    def __gt__(self, other):
        """Compare to another frequency."""
        if other is None:
            return True
        return super().__gt__(other)

    def __hash__(self):
        """Hash this tuple."""
        return tuple.__hash__(self)

    def __str__(self):
        """Format for print out."""
        return f"central={self.central} {self.unit} ±{self.side} width={self.bandwidth} {self.unit}"

    def __contains__(self, other):
        """Check if this double-side-band 'contains' *other*."""
        if other is None:
            return False
        if isinstance(other, numbers.Number):
            if (self.central + self.side - self.bandwidth/2. <= other
                    <= self.central + self.side + self.bandwidth/2.):
                return True
            if (self.central - self.side - self.bandwidth/2. <= other
                    <= self.central - self.side + self.bandwidth/2.):
                return True
            return False

        if isinstance(other, (tuple, list)) and len(other) == 3:
            return ((self.central - self.side - self.bandwidth/2. <=
                     other[0] - other[1] - other[2]/2. and
                     self.central - self.side + self.bandwidth/2. >=
                     other[0] - other[1] + other[2]/2.) or
                    (self.central + self.side - self.bandwidth/2. <=
                     other[0] + other[1] - other[2]/2. and
                     self.central + self.side + self.bandwidth/2. >=
                     other[0] + other[1] + other[2]/2.))

        with suppress(AttributeError):
            if self.unit != other.unit:
                raise NotImplementedError("Can't compare frequency ranges with different units.")
            return ((self.central - self.side - self.bandwidth/2. <=
                     other.central - other.side - other.bandwidth/2. and
                     self.central - self.side + self.bandwidth/2. >=
                     other.central - other.side + other.bandwidth/2.) or
                    (self.central + self.side - self.bandwidth/2. <=
                     other.central + other.side - other.bandwidth/2. and
                     self.central + self.side + self.bandwidth/2. >=
                     other.central + other.side + other.bandwidth/2.))

        return False

    def distance(self, value):
        """Get the distance from value."""
        if self == value:
            try:
                left_side_dist = abs(value.central - value.side - (self.central - self.side))
                right_side_dist = abs(value.central + value.side - (self.central + self.side))
                return min(left_side_dist, right_side_dist)
            except AttributeError:
                if isinstance(value, (tuple, list)):
                    return abs((value[0] - value[1]) - (self.central - self.side))

                left_side_dist = abs(value - (self.central - self.side))
                right_side_dist = abs(value - (self.central + self.side))
                return min(left_side_dist, right_side_dist)
        else:
            return np.inf

    @classmethod
    def convert(cls, frq):
        """Convert `frq` to this type if possible."""
        if isinstance(frq, dict):
            return cls(**frq)
        return frq


class FrequencyRangeBase(NamedTuple):
    """Base class for frequency ranges.

    This is needed because of this bug: https://bugs.python.org/issue41629
    """

    central: float
    bandwidth: float
    unit: str = "GHz"


class FrequencyRange(FrequencyRangeBase):
    """The Frequency range class.

    The elements of the range are central and bandwidth values, and optionally
    a unit (defaults to GHz). No clever unit conversion is done here, it's just
    used for checking that two ranges are comparable.

    This type is used for passive microwave sensors.

    """

    def __eq__(self, other):
        """Check wether two channel frequencies are equal.

        Args:
            other (tuple or scalar): (central frq, band width frq) or scalar frq

        Return:
            True if other is a scalar and min <= other <= max, or if other is a
            tuple equal to self, or if other is a number contained by self.
            False otherwise.

        """
        if other is None:
            return False
        if isinstance(other, numbers.Number):
            return other in self
        if isinstance(other, (tuple, list)) and len(other) == 2:
            return self[:2] == other
        return super().__eq__(other)

    def __lt__(self, other):
        """Compare to another frequency."""
        if other is None:
            return False
        return super().__lt__(other)

    def __gt__(self, other):
        """Compare to another frequency."""
        if other is None:
            return True
        return super().__gt__(other)

    def __hash__(self):
        """Hash this tuple."""
        return tuple.__hash__(self)

    def __str__(self):
        """Format for print out."""
        return f"central={self.central} {self.unit} width={self.bandwidth} {self.unit}"

    def __contains__(self, other):
        """Check if this range contains *other*."""
        if other is None:
            return False
        if isinstance(other, numbers.Number):
            return self.central - self.bandwidth/2. <= other <= self.central + self.bandwidth/2.

        with suppress(AttributeError):
            if self.unit != other.unit:
                raise NotImplementedError("Can't compare frequency ranges with different units.")
            return (self.central - self.bandwidth/2. <= other.central - other.bandwidth/2. and
                    self.central + self.bandwidth/2. >= other.central + other.bandwidth/2.)
        return False

    def distance(self, value):
        """Get the distance from value."""
        if self == value:
            try:
                return abs(value.central - self.central)
            except AttributeError:
                if isinstance(value, (tuple, list)):
                    return abs(value[0] - self.central)
                return abs(value - self.central)
        else:
            return np.inf

    @classmethod
    def convert(cls, frq):
        """Convert `frq` to this type if possible."""
        if isinstance(frq, dict):
            return cls(**frq)
        return frq


def _is_inside_interval(value, central, width):
    return (central - width/2 <= value <= central + width/2)
