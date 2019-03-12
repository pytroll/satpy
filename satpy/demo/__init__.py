#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 SatPy developers
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

from datetime import datetime, timedelta

DEMO_CASES = {
    'abi_l1b': {
        'glob': 'OR_ABI-L1b-RadM1-M3C??_G16_s20182541[3456]?????_e*_c*.nc',
        'datetime': (
            datetime(2018, 9, 11, 13, 0, 0),
            datetime(2018, 9, 11, 16, 0, 0),  # or should it be exclusive
        ),
    },
}

