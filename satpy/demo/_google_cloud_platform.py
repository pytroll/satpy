#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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


def get_bucket_files(glob_pattern, base_dir, force=False, pattern_slice=slice(None)):
    """Helper function to download files from Google Cloud Storage.

    Args:
        glob_pattern (str or list): Glob pattern string or series of patterns
            used to search for on Google Cloud Storage. The pattern should
            include the "gs://" protocol prefix. If a list of lists, then the
            results of each sublist pattern are concatenated and the result is
            treated as one pattern result. This is important for things like
            ``pattern_slice`` and complicated glob patterns not supported by
            GCP.
        base_dir (str): Root directory to place downloaded files on the local
            system.
        force (bool): Force re-download of data regardless of its existence on
            the local system. Warning: May delete non-demo files stored in
            download directory.
        pattern_slice (slice): Slice object to limit the number of files
            returned by each glob pattern.

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
        # handle multiple glob patterns being treated as one pattern
        # for complicated patterns that GCP can't handle
        if isinstance(gp, str):
            glob_results = list(fs.glob(gp))
        else:
            # flat list of results
            glob_results = [fn for pat in gp for fn in fs.glob(pat)]

        for fn in glob_results[pattern_slice]:
            ondisk_fn = os.path.basename(fn)
            ondisk_pathname = os.path.join(base_dir, ondisk_fn)
            filenames.append(ondisk_pathname)

            if force and os.path.isfile(ondisk_pathname):
                os.remove(ondisk_pathname)
            elif os.path.isfile(ondisk_pathname):
                LOG.info("Found existing: {}".format(ondisk_pathname))
                continue
            LOG.info("Downloading: {}".format(ondisk_pathname))
            fs.get('gs://' + fn, ondisk_pathname)

    if not filenames:
        raise OSError("No files could be found or downloaded.")
    return filenames
