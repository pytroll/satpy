#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Satpy developers
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
"""Benchmark satpy."""

import s3fs


class HimawariHSD:
    """Benchmark Himawari HSD reading."""

    timeout = 360
    data_files = ['HS_H08_20210409_0800_B01_FLDK_R10_S0410.DAT.bz2',
                  'HS_H08_20210409_0800_B02_FLDK_R10_S0410.DAT.bz2',
                  'HS_H08_20210409_0800_B03_FLDK_R05_S0410.DAT.bz2',
                  'HS_H08_20210409_0800_B04_FLDK_R10_S0410.DAT.bz2',
                  ]

    def setup_cache(self):
        """Fetch the data files."""
        self.fs = s3fs.S3FileSystem(anon=True)

        for filename in self.data_files:
            to_get = 'noaa-himawari8/AHI-L1b-FLDK/2021/04/09/0800/' + filename
            self.fs.get_file(to_get, filename)

    def time_load_one_channel(self):
        """Time the loading of one channel."""
        from satpy import Scene
        scn = Scene(filenames=self.data_files, reader='ahi_hsd')
        scn.load(['B01'], pad_data=False)
        scn['B01'].compute()

    def peakmem_load_one_channel(self):
        """Check peak memory usage of loading one channel."""
        from satpy import Scene
        scn = Scene(filenames=self.data_files, reader='ahi_hsd')
        scn.load(['B01'], pad_data=False)
        scn['B01'].compute()

    def time_load_true_color(self):
        """Time the loading of the generation of true_color_nocorr."""
        composite = "true_color_nocorr"
        from satpy import Scene
        scn = Scene(filenames=self.data_files, reader='ahi_hsd')
        scn.load([composite], pad_data=False)
        lscn = scn.resample(resampler='native')
        lscn[composite].compute()

    def peakmem_load_true_color(self):
        """Check peak memory usage of the generation of true_color_nocorr."""
        composite = "true_color_nocorr"
        from satpy import Scene
        scn = Scene(filenames=self.data_files, reader='ahi_hsd')
        scn.load([composite], pad_data=False)
        lscn = scn.resample(resampler='native')
        lscn[composite].compute()

    def time_save_true_color_nocorr_to_geotiff(self):
        """Time the generation and saving of true_color_nocorr."""
        composite = "true_color_nocorr"
        from satpy import Scene
        scn = Scene(filenames=self.data_files, reader='ahi_hsd')
        scn.load([composite], pad_data=False)
        lscn = scn.resample(resampler='native')
        lscn.save_dataset(composite, filename='test.tif', tiled=True)

    def peakmem_save_true_color_to_geotiff(self):
        """Check peak memory usage of the generation and saving of true_color_nocorr."""
        composite = "true_color_nocorr"
        from satpy import Scene
        scn = Scene(filenames=self.data_files, reader='ahi_hsd')
        scn.load([composite], pad_data=False)
        lscn = scn.resample(resampler='native')
        lscn.save_dataset(composite, filename='test.tif', tiled=True)
