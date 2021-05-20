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
"""ENVISAT MERIS reader.

Sentinel 3 like format: https://earth.esa.int/eogateway/documents/20142/37627/MERIS-Sentinel-3-Like-L1-andL2-PFS.pdf

Default:
    scn = Scene(filenames=my_files, reader='meris_l2_sen3')

References:
    - :func:`xarray.open_dataset`
"""

import logging
from contextlib import suppress
from functools import reduce

import dask.array as da
import numpy as np
import xarray as xr

from satpy import CHUNK_SIZE
from satpy.readers import open_file_or_filename
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.olci_nc import NCOLCIBase, NCOLCI2, NCOLCILowResData, NCOLCIMeteo, BitFlags
from satpy.utils import angle2xyz, xyz2angle

from satpy._compat import cached_property

logger = logging.getLogger(__name__)

PLATFORM_NAMES = {'ENV': 'Environmental Satellite'}


class NCMERISCal(NCOLCIBase):
    """Dummy class for calibration."""

    def __init__(self, filename, filename_info, filetype_info, engine=None):
        """Init the meris reader base."""
        super(NCMERISCal, self).__init__(filename, filename_info, filetype_info)
        self.sensor = 'meris'


class NCMERISGeo(NCOLCIBase):
    """Dummy class for navigation."""

    def __init__(self, filename, filename_info, filetype_info, engine=None):
        """Init the meris reader base."""
        super(NCMERISGeo, self).__init__(filename, filename_info, filetype_info)
        self.sensor = 'meris'


class NCMERISChannelBase(NCOLCIBase):
    """Base class for channel reading."""

    def __init__(self, filename, filename_info, filetype_info, engine=None):
        """Init the file handler."""
        super(NCMERISChannelBase, self).__init__(filename, filename_info, filetype_info)
        self.channel = filename_info.get('dataset_name')
        self.sensor = 'meris'


class NCMERIS2(NCOLCI2):
    """File handler for MERIS l2."""

    def __init__(self, filename, filename_info, filetype_info, engine=None):
        """Init the file handler."""
        super(NCMERIS2, self).__init__(filename, filename_info, filetype_info)
        self.sensor = 'meris'
        self.reflectance_prefix = 'M'
        self.reflectance_suffix = '_rho_w'

    def getbitmask(self, wqsf, items=None):
        """Get the bitmask. Experimental default mask."""
        items = items or ['SEA_ICE', 'MEGLINT', 'HIGHGLINT',
                          'HAZE_OVER_WATER', 'WHITECAPS', 'AC_FAIL', 'WHITE_SCATT',
                          'LOWRW', 'HIGHRW', 'OUT_OF_RANGE_AAC', 'OUT_OF_SCOPE_AAC',
                          'OUT_OF_RANGE_OC_NN', 'OUT_OF_SCOPE_OC_NN',
                          'OUT_OF_RANGE_CHL_OC4ME_INPUT', 'OUT_OF_RANGE_CHL_OC4ME']
        bflags = BitFlags(
            wqsf,
            flag_list=['SEA_ICE', 'MEGLINT', 'HIGHGLINT', 'CASE2_S', 'CASE2_ANOM',
                       'HAZE_OVER_WATER', 'WHITECAPS', 'AC_FAIL', 'BPAC_ON', 'WHITE_SCATT',
                       'LOWRW', 'HIGHRW', 'OUT_OF_RANGE_AAC', 'OUT_OF_SCOPE_AAC',
                       'OUT_OF_RANGE_OC_NN', 'OUT_OF_SCOPE_OC_NN',
                       'OUT_OF_RANGE_CHL_OC4ME_INPUT', 'OUT_OF_RANGE_CHL_OC4ME'],
        )
        return reduce(np.logical_or, [bflags[item] for item in items])


class NCMERISAngles(NCOLCILowResData):
    """File handler for the MERIS angles."""

    def __init__(self, filename, filename_info, filetype_info, engine=None):
        """Init the file handler."""
        super(NCMERISAngles, self).__init__(filename, filename_info, filetype_info)
        self.sensor = 'meris'


class NCMERISMeteo(NCOLCIMeteo):
    """File handler for the MERIS meteo data."""

    def __init__(self, filename, filename_info, filetype_info, engine=None):
        """Init the file handler."""
        super(NCMERISMeteo, self).__init__(filename, filename_info, filetype_info)
        self.sensor = 'meris'