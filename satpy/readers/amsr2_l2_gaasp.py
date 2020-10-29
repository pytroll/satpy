#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Satpy developers
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
"""GCOM-W1 AMSR2 Level 2 files from the GAASP software.

GAASP output files are in the NetCDF4 format. Software is provided by NOAA
and is also distributed by the CSPP group. More information on the products
supported by this reader can be found here:
https://www.star.nesdis.noaa.gov/jpss/gcom.php for more information.

"""

import logging
from satpy.readers.netcdf_utils import NetCDF4FileHandler

logger = logging.getLogger(__name__)


class GAASPFileHandler(NetCDF4FileHandler):
    """Generic file handler for GAASP output files."""

    def available_datasets(self, configured_datasets=None):
        """Dynamically discover what variables can be loaded from this file.

        See :meth:`satpy.readers.file_handlers.BaseHandler.available_datasets`
        for more information.

        """
        for is_avail, ds_info in (configured_datasets or []):
            if is_avail is not None:
                # some other file handler said it has this dataset
                # we don't know any more information than the previous
                # file handler so let's yield early
                yield is_avail, ds_info
                continue
            yield self.file_type_matches(ds_info['file_type']), ds_info
