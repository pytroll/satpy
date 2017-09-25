# Author(s):

#   Lorenzo Clementi <lorenzo.clementi@meteoswiss.ch>

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

"""
Reader for generic image (e.g. gif, png, jpg).

It returns a dataset without coordinates and calibration.
"""


from PIL import Image
import numpy as np
import os
import logging
from satpy.dataset import Dataset, DatasetID
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.yaml_reader import FileYAMLReader

logger = logging.getLogger(__name__)


class GenericImageFileHandler(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(GenericImageFileHandler, self).__init__(
            filename, filename_info, filetype_info)
        self.finfo = filename_info
        self.finfo['end_time'] =  self.finfo['start_time']
        self.finfo['filename'] = filename
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
        out = np.rot90(np.fliplr(np.transpose(selected)))
        info['filename'] = self.finfo['filename']
        ds = Dataset(out, copy=False, **info)
        return ds

