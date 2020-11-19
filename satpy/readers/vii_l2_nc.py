#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Satpy developers
#
# satpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# satpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.

"""EUMETSAT EPS-SG Visible/Infrared Imager (VII) Level 2 products reader."""

import logging

from satpy.readers.vii_base_nc import ViiNCBaseFileHandler

logger = logging.getLogger(__name__)


class ViiL2NCFileHandler(ViiNCBaseFileHandler):
    """Reader class for VII L2 products in netCDF format."""

    def _perform_orthorectification(self, variable, orthorect_data_name):
        """Perform the orthorectification.

        Args:
            variable: xarray DataArray containing the dataset to correct for orthorectification.
            orthorect_data_name: name of the orthorectification correction data in the product.

        Returns:
            DataArray: array containing the corrected values and all the original metadata.

        """
        try:
            orthorect_data = self[orthorect_data_name]
            variable += orthorect_data
        except KeyError:
            logger.warning('Required dataset %s for orthorectification not available, skipping', orthorect_data_name)
        return variable
