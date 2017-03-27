#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Adam.Dybbroe

# Author(s):

#   Adam Dybbroe <adam.dybbroe@smhi.se>
#   Ulrich Hamann <ulrich.hamann@meteoswiss.ch>
#   Sauli Joro <sauli.joro@icloud.com>

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

"""A reader for the EUMETSAT MSG native format
"""

import logging
from datetime import datetime
import numpy as np

from satpy.dataset import Dataset, DatasetID

from pyresample import geometry
from satpy.readers.hrit_base import (HRITFileHandler, ancillary_text,
                                     annotation_header, base_hdr_map,
                                     image_data_function, make_time_cds_short,
                                     time_cds_short)


class CalibrationError(Exception):
    pass

logger = logging.getLogger('native_msg')


class NativeMSGFileHandler(HRITFileHandler):

    """Native MSG format reader
    """

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super(NativeMSGFileHandler, self).__init__(filename, filename_info,
                                                   filetype_info,
                                                   (msg_hdr_map,
                                                    msg_variable_length_headers,
                                                    msg_text_headers))

        # Don't know yet how to get the pro and epi into the object_
        self.prologue = None
        self.epilogue = None

    @property
    def start_time(self):
        pass
        # pacqtime = self.epilogue['ImageProductionStats'][
        #     'ActualScanningSummary']
        # return pacqtime['ForwardScanStart']

    @property
    def end_time(self):
        pass
        # pacqtime = self.epilogue['ImageProductionStats'][
        #     'ActualScanningSummary']
        # return pacqtime['ForwardScanEnd']

    def get_dataset(self, key, info, out=None,
                    xslice=slice(None), yslice=slice(None)):
        res = super(NativeMSGFileHandler, self).get_dataset(key, info, out,
                                                            xslice, yslice)
        if res is not None:
            out = res

        self.calibrate(out, key.calibration)
        out.info['units'] = info['units']
        out.info['standard_name'] = info['standard_name']
        out.info['platform_name'] = self.platform_name
        out.info['sensor'] = 'seviri'

if __name__ == "__main__":

    TESTFILE = ""
