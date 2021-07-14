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
"""Utilities for downloading and extracting demo data zip files."""

import os
import math
import logging
import requests
from zipfile import ZipFile

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iter_content, **kwargs):
        """Fake tqdm progress bar."""
        return iter_content

logger = logging.getLogger(__name__)


def _download_data_zip(url, output_filename):
    if os.path.isfile(output_filename):
        logger.info("Data zip file already exists, won't re-download: {}".format(output_filename))
        return True

    print("Downloading {}".format(url))
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0
    with open(output_filename, 'wb') as f:
        for data in tqdm(
                r.iter_content(block_size),
                total=math.ceil(total_size//block_size),
                unit='KB',
                unit_scale=True):
            wrote += len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR: something went wrong downloading {}".format(url))
        return False
    return True


def _unzip(filename, output_dir, delete_zip=False):
    print("Extracting {}".format(filename))
    try:
        with ZipFile(filename, 'r') as zip_obj:
            zip_obj.extractall(output_dir)
    except OSError:
        print("FAIL: Could not extract {}".format(filename))
        return False
    finally:
        if delete_zip:
            os.remove(filename)
    return True


def download_and_unzip(url, output_dir, delete_zip=False):
    """Download zip file at 'url', extract it to 'output_dir'.

    Returns:
        ``True`` if the file was successfully downloaded and extracted,
        ``False`` otherwise.

    """
    filename = os.path.basename(url)
    if _download_data_zip(url, filename):
        return _unzip(filename, output_dir, delete_zip=delete_zip)
    return False
