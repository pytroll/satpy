#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Satpy developers
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
"""Landsat OLI/TIRS Level 1 reader.

Details of the data format can be found here:
  https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1822_Landsat8-9-OLI-TIRS-C2-L1-DFCB-v6.pdf
  https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product

"""

import logging


logger = logging.getLogger(__name__)

PLATFORMS = {"08": "Landsat-8",
             "09": "Landsat-9"}
