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

    timeout = 120
    data_files = [  # 'HS_H08_20210409_0800_B01_FLDK_R10_S0110.DAT.bz2',
                  # 'HS_H08_20210409_0800_B01_FLDK_R10_S0210.DAT.bz2',
                  'HS_H08_20210409_0800_B01_FLDK_R10_S0310.DAT.bz2',
                  'HS_H08_20210409_0800_B01_FLDK_R10_S0410.DAT.bz2',
                  # 'HS_H08_20210409_0800_B01_FLDK_R10_S0510.DAT.bz2',
                  # 'HS_H08_20210409_0800_B01_FLDK_R10_S0610.DAT.bz2',
                  # 'HS_H08_20210409_0800_B01_FLDK_R10_S0710.DAT.bz2',
                  # 'HS_H08_20210409_0800_B01_FLDK_R10_S0810.DAT.bz2',
                  # 'HS_H08_20210409_0800_B01_FLDK_R10_S0910.DAT.bz2',
                  # 'HS_H08_20210409_0800_B01_FLDK_R10_S1010.DAT.bz2'
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
        scn.load(['B01'])
        scn['B01'].compute()

    def peakmem_load_one_channel(self):
        """Check peak memory usage of loading one channel."""
        from satpy import Scene
        scn = Scene(filenames=self.data_files, reader='ahi_hsd')
        scn.load(['B01'])
        scn['B01'].compute()
