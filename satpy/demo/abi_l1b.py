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
"""Demo data download helper functions for ABI L1b data."""
import os

from satpy import config


def get_us_midlatitude_cyclone_abi(base_dir=None, method=None, force=False):
    """Get GOES-16 ABI (CONUS sector) data from 2019-03-14 00:00Z.

    Args:
        base_dir (str): Base directory for downloaded files.
        method (str): Force download method for the data if not already cached.
            Allowed options are: 'gcsfs'. Default of ``None`` will
            choose the best method based on environment settings.
        force (bool): Force re-download of data regardless of its existence on
            the local system. Warning: May delete non-demo files stored in
            download directory.

    Total size: ~110MB

    """
    base_dir = base_dir or config.get('demo_data_dir', '.')
    if method is None:
        method = 'gcsfs'
    if method not in ['gcsfs']:
        raise NotImplementedError("Demo data download method '{}' not "
                                  "implemented yet.".format(method))

    from ._google_cloud_platform import get_bucket_files
    patterns = ['gs://gcp-public-data-goes-16/ABI-L1b-RadC/2019/073/00/*s20190730002*.nc']
    subdir = os.path.join(base_dir, 'abi_l1b', '20190314_us_midlatitude_cyclone')
    os.makedirs(subdir, exist_ok=True)
    filenames = get_bucket_files(patterns, subdir, force=force)
    assert len(filenames) == 16, "Not all files could be downloaded"
    return filenames


def get_hurricane_florence_abi(base_dir=None, method=None, force=False,
                               channels=None, num_frames=10):
    """Get GOES-16 ABI (Meso sector) data from 2018-09-11 13:00Z to 17:00Z.

    Args:
        base_dir (str): Base directory for downloaded files.
        method (str): Force download method for the data if not already cached.
            Allowed options are: 'gcsfs'. Default of ``None`` will
            choose the best method based on environment settings.
        force (bool): Force re-download of data regardless of its existence on
            the local system. Warning: May delete non-demo files stored in
            download directory.
        channels (list): Channels to include in download. Defaults to all
            16 channels.
        num_frames (int or slice): Number of frames to download. Maximum
            240 frames. Default 10 frames.

    Size per frame (all channels): ~15MB

    Total size (default 10 frames, all channels): ~124MB

    Total size (240 frames, all channels): ~3.5GB

    """
    base_dir = base_dir or config.get('demo_data_dir', '.')
    if channels is None:
        channels = range(1, 17)
    if method is None:
        method = 'gcsfs'
    if method not in ['gcsfs']:
        raise NotImplementedError("Demo data download method '{}' not "
                                  "implemented yet.".format(method))
    if isinstance(num_frames, (int, float)):
        frame_slice = slice(0, num_frames)
    else:
        frame_slice = num_frames

    from ._google_cloud_platform import get_bucket_files

    patterns = []
    for channel in channels:
        # patterns += ['gs://gcp-public-data-goes-16/ABI-L1b-RadM/2018/254/1[3456]/'
        #              '*C{:02d}*s20182541[3456]*.nc'.format(channel)]
        patterns += [(
            'gs://gcp-public-data-goes-16/ABI-L1b-RadM/2018/254/13/*RadM1*C{:02d}*s201825413*.nc'.format(channel),
            'gs://gcp-public-data-goes-16/ABI-L1b-RadM/2018/254/14/*RadM1*C{:02d}*s201825414*.nc'.format(channel),
            'gs://gcp-public-data-goes-16/ABI-L1b-RadM/2018/254/15/*RadM1*C{:02d}*s201825415*.nc'.format(channel),
            'gs://gcp-public-data-goes-16/ABI-L1b-RadM/2018/254/16/*RadM1*C{:02d}*s201825416*.nc'.format(channel),
        )]
    subdir = os.path.join(base_dir, 'abi_l1b', '20180911_hurricane_florence_abi_l1b')
    os.makedirs(subdir, exist_ok=True)
    filenames = get_bucket_files(patterns, subdir, force=force, pattern_slice=frame_slice)

    actual_slice = frame_slice.indices(240)  # 240 max frames
    num_frames = int((actual_slice[1] - actual_slice[0]) / actual_slice[2])
    assert len(filenames) == len(channels) * num_frames, "Not all files could be downloaded"
    return filenames
