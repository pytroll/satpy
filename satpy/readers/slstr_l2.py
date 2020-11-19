# -*- coding: utf-8 -*-
# Copyright (c) 2017 Satpy developers
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
"""Reader for Sentinel-3 SLSTR SST data."""

from datetime import datetime
from satpy.readers.file_handlers import BaseFileHandler
from satpy import CHUNK_SIZE
import xarray as xr


class SLSTRL2FileHandler(BaseFileHandler):
    """File handler for Sentinel-3 SSL L2 netCDF files."""

    def __init__(self, filename, filename_info, filetype_info, engine=None):
        """Initialize the file handler for Sentinel-3 SSL L2 netCDF data."""
        super(SLSTRL2FileHandler, self).__init__(filename, filename_info, filetype_info)

        if filename.endswith('tar'):
            import tarfile
            import os
            import tempfile
            with tempfile.TemporaryDirectory() as tempdir:
                with tarfile.open(name=filename, mode='r') as tf:
                    sst_filename = next((name for name in tf.getnames()
                                        if name.endswith('nc') and 'GHRSST-SSTskin' in name))
                    tf.extract(sst_filename, tempdir)
                    fullpath = os.path.join(tempdir, sst_filename)
                    self.nc = xr.open_dataset(fullpath,
                                              decode_cf=True,
                                              mask_and_scale=True,
                                              engine=engine,
                                              chunks={'ni': CHUNK_SIZE,
                                                      'nj': CHUNK_SIZE})
        else:
            self.nc = xr.open_dataset(filename,
                                      decode_cf=True,
                                      mask_and_scale=True,
                                      engine=engine,
                                      chunks={'ni': CHUNK_SIZE,
                                              'nj': CHUNK_SIZE})

        self.nc = self.nc.rename({'ni': 'x', 'nj': 'y'})
        self.filename_info['start_time'] = datetime.strptime(
            self.nc.start_time, '%Y%m%dT%H%M%SZ')
        self.filename_info['end_time'] = datetime.strptime(
            self.nc.stop_time, '%Y%m%dT%H%M%SZ')

    def get_dataset(self, key, info):
        """Get any available dataset."""
        stdname = info.get('standard_name')
        return self.nc[stdname].squeeze()

    @property
    def start_time(self):
        """Get start time."""
        return self.filename_info['start_time']

    @property
    def end_time(self):
        """Get end time."""
        return self.filename_info['end_time']
