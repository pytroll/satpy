#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 Satpy developers
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
"""Modis level 3 hdf-eos format reader.

Introduction
------------

The ``modis_l3`` reader reads Modis L3 products in hdf-eos format.
Since there are a multitude of different level 3 datasets not all of theses are implemented (yet).


Currently the reader supports:
    - mcd43c1: BRDF/Albedo Model Parameters dataset
    - mcd43c3: BRDF/Albedo Albedo dataset

To get a list of the available datasets for a given file refer to the "Load data" section in :doc:`../reading`.

"""
import logging

from pyresample import geometry

from satpy.readers.hdfeos_base import HDFEOSGeoReader
from satpy.utils import get_legacy_chunk_size

logger = logging.getLogger(__name__)
CHUNK_SIZE = get_legacy_chunk_size()


class ModisL3GriddedHDFFileHandler(HDFEOSGeoReader):
    """File handler for MODIS HDF-EOS Level 3 CMG gridded files."""

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Init the file handler."""
        super().__init__(filename, filename_info, filetype_info, **kwargs)

        # Initialise number of rows and columns
        self.nrows = self.metadata["GridStructure"]["GRID_1"]["YDim"]
        self.ncols = self.metadata["GridStructure"]["GRID_1"]["XDim"]

        # Get the grid name and other projection info
        gridname = self.metadata["GridStructure"]["GRID_1"]["GridName"]
        if "CMG" not in gridname:
            raise ValueError("Only CMG grids are supported")

        # Get the grid resolution
        pos = gridname.rfind("_") + 1
        pos2 = gridname.rfind("Deg")
        self.resolution = float(gridname[pos:pos2])

        upperleft = self.metadata["GridStructure"]["GRID_1"]["UpperLeftPointMtrs"]
        lowerright = self.metadata["GridStructure"]["GRID_1"]["LowerRightMtrs"]

        self.area_extent = (upperleft[0], lowerright[1], lowerright[0], upperleft[1])


    def available_datasets(self, configured_datasets=None):
        """Automatically determine datasets provided by this file."""
        logger.debug("Available_datasets begin...")

        ds_dict = self.sd.datasets()

        yield from super().available_datasets(configured_datasets)
        common = {"file_type": "mcd43_cmg_hdf", "resolution": self.resolution}
        for key in ds_dict.keys():
            if "/" in key:  # not a dataset
                continue
            yield True, {"name": key} | common

    def get_dataset(self, dataset_id, dataset_info):
        """Get DataArray for specified dataset."""
        dataset_name = dataset_id["name"]
        dataset = self.load_dataset(dataset_name, dataset_info.pop("category", False))
        self._add_satpy_metadata(dataset_id, dataset)

        return dataset


    def get_area_def(self, dsid):
        """Get the area definition.

        This is fixed, but not defined in the file. So we must
        generate it ourselves with some assumptions.
        """
        proj_param = "EPSG:4326"

        area = geometry.AreaDefinition("gridded_modis",
                                       "A gridded L3 MODIS area",
                                       "longlat",
                                       proj_param,
                                       self.ncols,
                                       self.nrows,
                                       self.area_extent)
        self.area = area

        return self.area
