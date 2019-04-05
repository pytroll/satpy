#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 SatPy developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import logging

try:
    from urllib.request import urlopen
    from urllib.error import URLError
except ImportError:
    # python 2
    from urllib2 import urlopen, URLError

try:
    import gcsfs
except ImportError:
    gcsfs = None

LOG = logging.getLogger(__name__)


def is_google_cloud_instance():
    try:
        return urlopen('http://metadata.google.internal').headers.get('Metadata-Flavor') == 'Google'
    except URLError:
        return False


def get_bucket_files(glob_pattern, base_dir, force=False):
    """Helper function to download files from Google Cloud Storage.

    Args:
        glob_pattern (str or list): Glob pattern string or series of patterns
            used to search for on Google Cloud Storage. The pattern should
            include the "gs://" protocol prefix.
        base_dir (str): Root directory to place downloaded files on the local
            system.
        force (bool): Force re-download of data regardless of its existence on
            the local system. Warning: May delete non-demo files stored in
            download directory.

    """
    if gcsfs is None:
        raise RuntimeError("Missing 'gcsfs' dependency for GCS download.")
    if not os.path.isdir(base_dir):
        # it is the caller's responsibility to make this
        raise OSError("Directory does not exist: {}".format(base_dir))

    if isinstance(glob_pattern, str):
        glob_pattern = [glob_pattern]

    fs = gcsfs.GCSFileSystem(token='anon')
    filenames = []
    for gp in glob_pattern:
        for fn in fs.glob(gp):
            ondisk_fn = os.path.basename(fn)
            ondisk_pathname = os.path.join(base_dir, ondisk_fn)
            filenames.append(ondisk_pathname)
            LOG.info("Downloading: {}".format(ondisk_pathname))

            if force and os.path.isfile(ondisk_pathname):
                os.remove(ondisk_pathname)
            elif os.path.isfile(ondisk_pathname):
                continue
            fs.get('gs://' + fn, ondisk_pathname)

    if not filenames:
        raise OSError("No files could be found or downloaded.")
    return filenames
