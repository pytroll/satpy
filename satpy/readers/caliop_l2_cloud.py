# Copyright (c) 2016
# Author(s):

#   Mikhail Itkin <itkin.m@gmail.com>
#   Martin Raspaud <martin.raspaud@smhi.se>

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


import logging
import os.path
import re

from datetime import datetime
from pyhdf.SD import SD, SDC

from satpy.dataset import Dataset
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


class HDF4BandReader(BaseFileHandler):
    """CALIOP v3 HDF4 reader."""

    def __init__(self, filename, filename_info, filetype_info):
        super(HDF4BandReader, self).__init__(filename,
                                             filename_info,
                                             filetype_info)
        self.lons = None
        self.lats = None
        self._start_time = None
        self._end_time = None

        self.get_filehandle()

        self._start_time = filename_info['start_time']

        logger.debug('Retrieving end time from metadata array')
        self.get_end_time()

    def get_end_time(self):
        """Get observation end time from file metadata."""
        mda_dict = self.filehandle.attributes()
        core_mda = mda_dict['coremetadata']
        end_time_str = self.parse_metadata_string(core_mda)
        self._end_time = datetime.strptime(end_time_str, "%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def parse_metadata_string(metadata_string):
        """Grab end time with regular expression."""
        regex = r"STOP_DATE.+?VALUE\s*=\s*\"(.+?)\""
        match = re.search(regex, metadata_string, re.DOTALL)
        end_time_str = match.group(1)
        return end_time_str

    def get_filehandle(self):
        """Get HDF4 filehandle."""
        if os.path.exists(self.filename):
            self.filehandle = SD(self.filename, SDC.READ)
            logger.debug("Loading dataset {}".format(self.filename))
        else:
            raise IOError("Path {} does not exist.".format(self.filename))

    def get_dataset(self, key, info):
        """Read data from file and return the corresponding projectables."""
        if key.name in ['longitude', 'latitude']:
            logger.debug('Reading coordinate arrays.')

            if self.lons is None or self.lats is None:
                self.lons, self.lats = self.get_lonlats()

            if key.name == 'latitude':
                proj = Dataset(self.lats, id=key, **info)
            else:
                proj = Dataset(self.lons, id=key, **info)

        else:
            data = self.get_sds_variable(key.name)
            proj = Dataset(data, id=key, **info)

        return proj

    def get_sds_variable(self, name):
        """Read variable from the HDF4 file."""
        sds_obj = self.filehandle.select(name)
        data = sds_obj.get()
        return data

    def get_lonlats(self):
        """Get longitude and latitude arrays from the file."""
        longitudes = self.get_sds_variable('Longitude')
        latitudes = self.get_sds_variable('Latitude')
        return longitudes, latitudes

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time
