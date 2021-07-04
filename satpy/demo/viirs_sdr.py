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
"""Demo data download for VIIRS SDR HDF5 files."""

import os
import logging
from glob import glob

from ._zip import download_and_unzip

VIIRS_20170323_ZIP_URL = "https://bin.ssec.wisc.edu/pub/davidh/viirs_sdr_20170323_204321_204612.zip"

logger = logging.getLogger(__name__)


def get_viirs_sdr_20170323_204321(base_dir=None):
    """Get VIIRS SDR files for 2017-03-23 20:43:21 to 20:46:12.

    These files are downloaded as a zip file and then extracted. The zip
    file is downloaded to your current directory and deleted after extraction.
    The zip file contains data files for the I01, M03, M04, and M05 bands and
    the corresponding GITCO and GMTCO geolocation files.

    """
    base_dir = base_dir or "."
    zip_fn = os.path.basename(VIIRS_20170323_ZIP_URL)
    zip_path = os.path.join(base_dir, zip_fn)
    # assume directory in zip is the same as zip filename without the extension
    extract_dir = zip_path.replace(".zip", "")
    if os.path.isdir(extract_dir):
        logger.info(f"Extracted zip directory {extract_dir} already exists, won't re-download.")
    else:
        download_and_unzip(VIIRS_20170323_ZIP_URL, base_dir, delete_zip=True)
    if not os.path.isdir(extract_dir):
        raise RuntimeError("Unable to download or extract zip file. "
                           f"Extracted directory doesn't exist {extract_dir}")
    return sorted(glob(os.path.join(extract_dir, "*.h5")))
