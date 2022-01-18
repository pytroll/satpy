#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009-2019 Satpy developers
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
"""Satpy Package initializer."""

import os

try:
    from satpy.version import version as __version__  # noqa
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "No module named satpy.version. This could mean "
        "you didn't install 'satpy' properly. Try reinstalling ('pip "
        "install').")

CHUNK_SIZE = int(os.getenv('PYTROLL_CHUNK_SIZE', 4096))

from satpy._config import config  # noqa
from satpy.dataset import DataID, DataQuery  # noqa
from satpy.dataset.data_dict import DatasetDict  # noqa
from satpy.multiscene import MultiScene  # noqa
from satpy.readers import available_readers  # noqa
from satpy.readers import find_files_and_readers  # noqa
from satpy.scene import Scene  # noqa
from satpy.utils import get_logger  # noqa
from satpy.writers import available_writers  # noqa

log = get_logger('satpy')
