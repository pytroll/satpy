#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009-2016.

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""
Reading and calibrating GAC and LAC avhrr data.
Todo:
- Fine grained calibration

"""

import logging
from datetime import datetime, timedelta

import numpy as np
from pyresample.geometry import SwathDefinition

from pygac.gac_calibration import calibrate_solar, calibrate_thermal
from pygac.gac_klm import GACKLMReader
from pygac.gac_pod import GACPODReader
from satpy.dataset import Dataset
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


spacecrafts = {7: "NOAA 15", 3: "NOAA 16", 13: "NOAA 18", 15: "NOAA 19"}

AVHRR3_CHANNEL_NAMES = {"1": 0, "2": 1, "3A": 2, "3B": 3, "4": 4, "5": 5}
AVHRR_CHANNEL_NAMES = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}


class GACLACFile(BaseFileHandler):
    """Reader for GAC and LAC data.
    """

    def __init__(self, filename, filename_info, filetype_info):
        super(GACLACFile, self).__init__(
            filename, filename_info, filetype_info)

        self.reader = None
        self.channels = None
        self.chn_dict = None
        self._start_time = filename_info['start_time']
        self._end_time = datetime.combine(filename_info['start_time'].date(),
                                          filename_info['end_time'].time())
        if self._end_time < self._start_time:
            self._end_time += timedelta(days=1)

    def get_dataset(self, key, info):

        if self.reader is None:

            with open(self.filename) as fdes:
                data = fdes.read(3)
            if data in ["CMS", "NSS", "UKM", "DSS"]:
                reader = GACKLMReader
                self.chn_dict = AVHRR3_CHANNEL_NAMES
            else:
                reader = GACPODReader
                self.chn_dict = AVHRR_CHANNEL_NAMES

            self.reader = reader()
            self.reader.read(self.filename)

        if key.name in ['latitude', 'longitude']:
            if self.reader.lons is None or self.reader.lats is None:
                self.reader.get_lonlat(clock_drift_adjust=False)
            if key.name == 'latitude':
                return Dataset(self.reader.lats, id=key, **info)
            else:
                return Dataset(self.reader.lons, id=key, **info)

        if self.channels is None:
            self.channels = self.reader.get_calibrated_channels()

        data = self.channels[:, :, self.chn_dict[key.name]]
        return Dataset(data, id=key, **info)

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time
