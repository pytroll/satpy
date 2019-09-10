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
"""Satpy Package initializer.
"""

import os
from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

CHUNK_SIZE = int(os.getenv('PYTROLL_CHUNK_SIZE', 4096))

# Order of "highest" calibration from highest to lowest
DEFAULT_CALIBRATION_ORDER = [
    'brightness_temperature',
    'reflectance',
    'radiance',
    'counts',
    'gamma',
    'sigma_nought',
    'beta_nought',
]
CALIBRATION_ORDER = os.getenv('PYTROLL_CALIBRATION_ORDER', None)
if CALIBRATION_ORDER is None:
    CALIBRATION_ORDER = DEFAULT_CALIBRATION_ORDER
else:
    CALIBRATION_ORDER = [x.strip() for x in CALIBRATION_ORDER.split(' ')]
# convert to a dictionary of priority for faster access (0 higher priority)
CALIBRATION_ORDER = {cal: idx for idx, cal in enumerate(CALIBRATION_ORDER)}

from satpy.utils import get_logger  # noqa
from satpy.dataset import DatasetID, DATASET_KEYS  # noqa
from satpy.readers import (DatasetDict, find_files_and_readers,  # noqa
                           available_readers)  # noqa
from satpy.writers import available_writers  # noqa
from satpy.scene import Scene  # noqa
from satpy.multiscene import MultiScene  # noqa

log = get_logger('satpy')
