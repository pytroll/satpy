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
"""Classes and functions related to data identification and querying."""

from .anc_vars import dataset_walker, replace_anc  # noqa
from .data_dict import DatasetDict, get_key  # noqa
from .dataid import DataID, DataQuery, ModifierTuple, WavelengthRange, create_filtered_query  # noqa
from .metadata import combine_metadata  # noqa
