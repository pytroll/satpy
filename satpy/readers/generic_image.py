from PIL import Image
import numpy as np
import os

from satpy.dataset import Dataset, DatasetID
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.yaml_reader import FileYAMLReader
from pyresample import utils
from pyresample.geometry import AreaDefinition
from mpop.projector import get_area_def

logger = logging.getLogger(__name__)


class GenericImageFileHandler(BaseFileHandler):

    def __init__(self, filename, filename_info, filetype_info):
        super(GenericImageFileHandler, self).__init__(
            filename, filename_info, filetype_info)
        self.finfo = filename_info
        self.finfo['end_time'] =  self.finfo['start_time']
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
        ds = Dataset(out, copy=False, **info)
        return ds

