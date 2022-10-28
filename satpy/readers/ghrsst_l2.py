# -*- coding: utf-8 -*-
# Copyright (c) 2017 - 2022 Satpy developers
#
# This file is part of Satpy.
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
"""Reader for the GHRSST level-2 formatted data."""

import os
import tarfile
from contextlib import suppress
from datetime import datetime
from functools import cached_property

import xarray as xr

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler


class GHRSSTL2FileHandler(BaseFileHandler):
    """File handler for GHRSST L2 netCDF files."""

    def __init__(self, filename, filename_info, filetype_info, engine=None):
        """Initialize the file handler for GHRSST L2 netCDF data."""
        super().__init__(filename, filename_info, filetype_info)
        self._engine = engine
        self._tarfile = None

        self.filename_info['start_time'] = datetime.strptime(
            self.nc.start_time, '%Y%m%dT%H%M%SZ')
        self.filename_info['end_time'] = datetime.strptime(
            self.nc.stop_time, '%Y%m%dT%H%M%SZ')

    @cached_property
    def nc(self):
        """Get the xarray Dataset for the filename."""
        if os.fspath(self.filename).endswith('tar'):
            file_obj = self._open_tarfile()
        else:
            file_obj = self.filename

        nc = xr.open_dataset(file_obj,
                             decode_cf=True,
                             mask_and_scale=True,
                             engine=self._engine,
                             chunks={'ni': CHUNK_SIZE,
                                     'nj': CHUNK_SIZE})

        return nc.rename({'ni': 'x', 'nj': 'y'})

    def _open_tarfile(self):
        self._tarfile = tarfile.open(name=self.filename, mode='r')
        sst_filename = next((name for name in self._tarfile.getnames()
                             if self._is_sst_file(name)))
        file_obj = self._tarfile.extractfile(sst_filename)
        return file_obj

    @staticmethod
    def _is_sst_file(name):
        """Check if file in the tar archive is a valid SST file."""
        return name.endswith('nc') and 'GHRSST-SSTskin' in name

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

    @property
    def sensor(self):
        """Get the sensor name."""
        return self.nc.attrs['sensor'].lower()

    def __del__(self):
        """Close the tarfile object."""
        with suppress(AttributeError):
            self._tarfile.close()
