#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017.
#
# Author(s):
#
# Lorenzo Clementi   <lorenzo.clementi@meteoswiss.ch>
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
import logging
from collections import defaultdict
from datetime import datetime, timedelta

from PIL import Image
import numpy as np

from satpy.dataset import Dataset, DatasetID
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.yaml_reader import FileYAMLReader

logger = logging.getLogger(__name__)


class GenericImageFileHandler(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(GenericImageFileHandler, self).__init__(
            filename, filename_info, filetype_info)
        self.finfo = filename_info
        self.finfo['end_time'] =  datetime(1900,1,27,7,45)
        self.finfo['start_time'] = datetime(1900,1,27,7,45)
        self.selected = None
        self.read(filename)

    def read(self, filename):
        self.file_content = {}
        img = Image.open(filename)
        self.file_content['image'] = img

    @property
    def start_time(self):
        return self.finfo['start_time']

    @property
    def end_time(self):
        return self.finfo['end_time']

    def get_dataset(self, key, info, out=None):
        """Get a dataset from the file."""

        logger.debug("Reading %s.", key.name)
        values = self.file_content[key.name]
        selected = np.array(values)
        ds = Dataset(np.transpose(selected), copy=False, **info)
        return ds

