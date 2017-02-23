#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CALIOP v3 HDF4 reader
"""

import glob
import hashlib
import logging
import math
import multiprocessing
import os.path
from ConfigParser import ConfigParser
from datetime import datetime
from fnmatch import fnmatch

import numpy as np
from pyhdf.error import HDF4Error
from pyhdf.SD import SD, SDC

from pyresample import geometry
from satpy.config import CONFIG_PATH
from satpy.dataset import Dataset, DatasetID
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


class HDF4BandReader(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(HDF4BandReader, self).__init__(self, filename,
                                    filename_info, filetype_info)

        self._start_time = filename_info['start_time']
        self._end_time = filename_info['end_time']

        self.filename = filename
        self.get_filehandle()

    def get_filehandle(self):
        if os.path.exists(self.filename):
            self.filehandle = SD(self.filename, SDC.READ)
            logger.debug("Loading dataset {}".format(self.filename))
        else:
            raise IOError("Path {} does not exist.".format(self.filename))

    def get_dataset(self, key, info):
        """Read data from file and return the corresponding projectables.
        """
        return None

    def get_lonlats(self):
        longitudes = None
        latitudes = None
        return longitudes, latitudes

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time
