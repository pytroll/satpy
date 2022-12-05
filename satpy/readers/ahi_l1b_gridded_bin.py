#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Satpy developers
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
"""Advanced Himawari Imager (AHI) gridded format data reader.

This data comes in a flat binary format on a fixed grid, and needs to have
calibration coefficients applied to it in order to retrieve reflectance or BT.
LUTs can be downloaded at: ftp://hmwr829gr.cr.chiba-u.ac.jp/gridded/FD/support/

This data is gridded from the original Himawari geometry. To our knowledge,
only full disk grids are available, not for the Meso or Japan rapid scans.

References:
 - AHI gridded data website:
        http://www.cr.chiba-u.jp/databases/GEO/H8_9/FD/index_jp.html


"""

import logging
import os

import dask.array as da
import numpy as np
import xarray as xr
from appdirs import AppDirs
from pyresample import geometry

from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.utils import unzip_file

# Hardcoded address of the reflectance and BT look-up tables
AHI_REMOTE_LUTS = 'http://www.cr.chiba-u.jp/databases/GEO/H8_9/FD/count2tbb_v102.tgz'

# Full disk image sizes for each spatial resolution
AHI_FULLDISK_SIZES = {0.005: {'x_size': 24000,
                              'y_size': 24000},
                      0.01: {'x_size': 12000,
                             'y_size': 12000},
                      0.02: {'x_size': 6000,
                             'y_size': 6000}}

# Geographic extent of the full disk area in degrees
AHI_FULLDISK_EXTENT = [85., -60., 205., 60.]

# Resolutions of each channel type
AHI_CHANNEL_RES = {'vis': 0.01,
                   'ext': 0.005,
                   'sir': 0.02,
                   'tir': 0.02}

# List of LUT filenames
AHI_LUT_NAMES = ['ext.01', 'vis.01', 'vis.02', 'vis.03',
                 'sir.01', 'sir.02', 'tir.01', 'tir.02',
                 'tir.03', 'tir.04', 'tir.05', 'tir.06',
                 'tir.07', 'tir.08', 'tir.09', 'tir.10']

logger = logging.getLogger('ahi_grid')


class AHIGriddedFileHandler(BaseFileHandler):
    """AHI gridded format reader.

    This data is flat binary, big endian unsigned short.
    It covers the region 85E -> 205E, 60N -> 60S at variable resolution:
    - 0.005 degrees for Band 3
    - 0.01 degrees for Bands 1, 2 and 4
    - 0.02 degrees for all other bands.
    These are approximately equivalent to 0.5, 1 and 2km.

    Files can either be zipped with bz2 compression (like the HSD format
    data), or can be uncompressed flat binary.
    """

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super(AHIGriddedFileHandler, self).__init__(filename, filename_info,
                                                    filetype_info)
        self._unzipped = unzip_file(self.filename)
        # Assume file is not zipped
        if self._unzipped:
            # But if it is, set the filename to point to unzipped temp file
            self.filename = self._unzipped
        # Get the band name, needed for finding area and dimensions
        self.product_name = filetype_info['file_type']
        self.areaname = filename_info['area']
        self.sensor = 'ahi'
        self.res = AHI_CHANNEL_RES[self.product_name[:3]]
        if self.areaname == 'fld':
            self.nlines = AHI_FULLDISK_SIZES[self.res]['y_size']
            self.ncols = AHI_FULLDISK_SIZES[self.res]['x_size']
        else:
            raise NotImplementedError("Only full disk data is supported.")

        # Set up directory path for the LUTs
        app_dirs = AppDirs('ahi_gridded_luts', 'satpy', '1.0.2')
        self.lut_dir = os.path.expanduser(app_dirs.user_data_dir) + '/'
        self.area = None

    def __del__(self):
        """Delete the object."""
        if self._unzipped and os.path.exists(self.filename):
            os.remove(self.filename)

    def _load_lut(self):
        """Determine if LUT is available and, if not, download it."""
        # First, check that the LUT is available. If not, download it.
        lut_file = self.lut_dir + self.product_name
        if not os.path.exists(lut_file):
            self._get_luts()
        try:
            # Load file, it has 2 columns: DN + Refl/BT. We only need latter.
            lut = np.loadtxt(lut_file)[:, 1]
        except FileNotFoundError:
            raise FileNotFoundError("No LUT file found:", lut_file)
        return lut

    def _calibrate(self, data):
        """Load calibration from LUT and apply."""
        lut = self._load_lut()

        # LUT may truncate NaN values, so manually set those in data
        lut_len = len(lut)
        data = np.where(data < lut_len - 1, data, np.nan)
        return lut[data.astype(np.uint16)]

    @staticmethod
    def _download_luts(file_name):
        """Download LUTs from remote server."""
        import shutil
        import urllib

        # Set up an connection and download
        with urllib.request.urlopen(AHI_REMOTE_LUTS) as response:  # nosec
            with open(file_name, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

    @staticmethod
    def _untar_luts(tarred_file, outdir):
        """Uncompress downloaded LUTs, which are a tarball."""
        import tarfile
        tar = tarfile.open(tarred_file)
        tar.extractall(outdir)
        tar.close()
        os.remove(tarred_file)

    def _get_luts(self):
        """Download the LUTs needed for count->Refl/BT conversion."""
        import pathlib
        import shutil

        from satpy import config

        # Check that the LUT directory exists
        pathlib.Path(self.lut_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Download AHI LUTs files and store in directory %s",
                    self.lut_dir)
        tempdir = config["tmp_dir"]
        fname = os.path.join(tempdir, 'tmp.tgz')
        # Download the LUTs
        self._download_luts(fname)

        # The file is tarred, untar and remove the downloaded file
        self._untar_luts(fname, tempdir)

        lut_dl_dir = os.path.join(tempdir, 'count2tbb_v102/')

        # Loop over the LUTs and copy to the correct location
        for lutfile in AHI_LUT_NAMES:
            shutil.move(os.path.join(lut_dl_dir, lutfile), os.path.join(self.lut_dir, lutfile))
        shutil.rmtree(lut_dl_dir)

    def get_dataset(self, key, info):
        """Get the dataset."""
        return self.read_band(key, info)

    def get_area_def(self, dsid):
        """Get the area definition.

        This is fixed, but not defined in the file. So we must
        generate it ourselves with some assumptions.
        """
        if self.areaname == 'fld':
            area_extent = AHI_FULLDISK_EXTENT
        else:
            raise NotImplementedError("Reader only supports full disk data.")

        proj_param = 'EPSG:4326'

        area = geometry.AreaDefinition('gridded_himawari',
                                       'A gridded Himawari area',
                                       'longlat',
                                       proj_param,
                                       self.ncols,
                                       self.nlines,
                                       area_extent)
        self.area = area

        return area

    def _read_data(self, fp_):
        """Read raw binary data from file."""
        return da.from_array(np.memmap(self.filename,
                                       offset=fp_.tell(),
                                       dtype='>u2',
                                       shape=(self.nlines, self.ncols),
                                       mode='r'),
                             chunks=CHUNK_SIZE)

    def read_band(self, key, info):
        """Read the data."""
        with open(self.filename, "rb") as fp_:
            res = self._read_data(fp_)

        # Calibrate
        res = self.calibrate(res, key['calibration'])

        # Update metadata
        new_info = dict(
            units=info['units'],
            standard_name=info['standard_name'],
            wavelength=info['wavelength'],
            resolution=info['resolution'],
            id=key,
            name=key['name'],
            sensor=self.sensor,
        )
        res = xr.DataArray(res, attrs=new_info, dims=['y', 'x'])
        return res

    def calibrate(self, data, calib):
        """Calibrate the data."""
        if calib == 'counts':
            return data
        if calib == 'reflectance' or calib == 'brightness_temperature':
            return self._calibrate(data)
        raise NotImplementedError("ERROR: Unsupported calibration.",
                                  "Only counts, reflectance and ",
                                  "brightness_temperature calibration",
                                  "are supported.")
