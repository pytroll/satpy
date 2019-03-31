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
"""Demo data download helper functions.

Each ``get_*`` function below downloads files to a local directory and returns
a list of paths to those files. Some (not all) functions have multiple options
for how the data is downloaded (via the ``method`` keyword argument)
including:

- gcsfs: Download data from a public google cloud storage bucket using the
    ``gcsfs`` package.
- unidata_thredds: Access data using OpenDAP or similar method from Unidata's
    public THREDDS server
    (https://thredds.unidata.ucar.edu/thredds/catalog.html).
- uwaos_thredds: Access data using OpenDAP or similar method from the
    University of Wisconsin - Madison's AOS department's THREDDS server.
- http: A last resort download method when nothing else is available of a
    tarball or zip file from one or more servers available to the SatPy
    project.
- uw_arcdata: A network mount available on many servers at the Space Science
    and Engineering Center (SSEC) at the University of Wisconsin - Madison.
    This is method is mainly meant when tutorials are taught at the SSEC
    using a Jupyter Hub server.

"""

import os
import logging

try:
    import gcsfs
except ImportError:
    gcsfs = None

LOG = logging.getLogger(__name__)


def get_us_midlatitude_cyclone_abi(base_dir='.', method=None, force=False):
    """Get GOES-16 ABI data from March 14th 00:00Z.

    Args:
        base_dir (str): Base directory for downloaded files.
        method (str): Force download method for the data if not already cached.
            Allowed options are: 'gcsfs'. Default of ``None`` will
            choose the best method based on environment settings.
        force (bool): Force re-download of data regardless of its existense on
            the local system. Warning: May delete non-demo files stored in
            download directory.

    """
    if method is None:
        method = 'gcsfs'
    if method not in ['gcsfs']:
        raise NotImplementedError("Demo data download method '{}' not "
                                  "implemented yet.".format(method))
    fs = gcsfs.GCSFileSystem(token='anon')
    filenames = []
    for fn in fs.glob('gs://gcp-public-data-goes-16/ABI-L1b-RadC/2019/073/00/*0002*.nc'):
        ondisk_fn = os.path.basename(fn)
        ondisk_pathname = os.path.join(base_dir, ondisk_fn)
        filenames.append(ondisk_pathname)
        LOG.info("Downloading: {}".format(ondisk_pathname))

        if force and os.path.isfile(ondisk_pathname):
            os.remove(ondisk_pathname)
        elif os.path.isfile(ondisk_pathname):
            continue
        fs.get('gs://' + fn, ondisk_pathname)
