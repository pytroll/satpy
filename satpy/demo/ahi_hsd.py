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
"""Demo data download helper functions for AHI HSD data."""
import os

from satpy import config


def download_typhoon_surigae_ahi(base_dir=None,
                                 channels=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
                                 segments=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)):
    """Download Himawari 8 data.

    This scene shows the Typhoon Surigae.
    """
    import s3fs
    base_dir = base_dir or config.get('demo_data_dir', '.')
    channel_resolution = {1: 10,
                          2: 10,
                          3: 5,
                          4: 10}
    data_files = []
    for channel in channels:
        resolution = channel_resolution.get(channel, 20)
        for segment in segments:
            data_files.append(f"HS_H08_20210417_0500_B{channel:02d}_FLDK_R{resolution:02d}_S{segment:02d}10.DAT.bz2")

    subdir = os.path.join(base_dir, 'ahi_hsd', '20210417_0500_typhoon_surigae')
    os.makedirs(subdir, exist_ok=True)
    fs = s3fs.S3FileSystem(anon=True)

    result = []
    for filename in data_files:
        destination_filename = os.path.join(subdir, filename)
        result.append(destination_filename)
        if os.path.exists(destination_filename):
            continue
        to_get = 'noaa-himawari8/AHI-L1b-FLDK/2021/04/17/0500/' + filename
        fs.get_file(to_get, destination_filename)

    return result
