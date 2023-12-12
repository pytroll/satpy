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

The ``modis_l3`` reader reads MODIS L3 products in HDF-EOS format.

There are multiple level 3 products, including some on sinusoidal grids and some on the climate modeling grid (CMG).
This reader supports the CMG products at present, and the sinusoidal products will be added if there is demand.

The reader has been tested with:
    - MCD43c*: BRDF/Albedo data, such as parameters, albedo and nbar
    - MOD09CMG: Surface Reflectance on climate monitoring grid.

To get a list of the available datasets for a given file refer to the "Load data" section in :doc:`../reading`.

"""
import logging
from typing import Iterable

from pyresample import geometry

from satpy.readers.hdfeos_base import HDFEOSGeoReader

logger = logging.getLogger(__name__)


class ModisL3GriddedHDFFileHandler(HDFEOSGeoReader):
    """File handler for MODIS HDF-EOS Level 3 CMG gridded files."""
    def available_datasets(self, configured_datasets=None):
        """Automatically determine datasets provided by this file."""
        # Initialise set of variable names to carry through code
        handled_var_names = set()

        ds_dict = self.sd.datasets()

        for is_avail, ds_info in (configured_datasets or []):
            file_key = ds_info.get("file_key", ds_info["name"])
            # we must add all variables here even if another file handler has
            # claimed the variable. It could be another instance of this file
            # type, and we don't want to add that variable dynamically if the
            # other file handler defined it by the YAML definition.
            handled_var_names.add(file_key)
            if is_avail is not None:
                # some other file handler said it has this dataset
                # we don't know any more information than the previous
                # file handler so let's yield early
                yield is_avail, ds_info
                continue
            if self.file_type_matches(ds_info["file_type"]) is None:
                # this is not the file type for this dataset
                yield None, ds_info
                continue
            yield file_key in ds_dict.keys(), ds_info

        yield from self._dynamic_variables_from_file(handled_var_names)

    def _dynamic_variables_from_file(self, handled_var_names: set) -> Iterable[tuple[bool, dict]]:
        res = self._get_res()
        for var_name in self.sd.datasets().keys():
            if var_name in handled_var_names:
                # skip variables that YAML had configured
                continue
            common = {"file_type": "modis_l3_cmg_hdf",
                      "resolution": res,
                      "name": var_name}
            yield True, common


    def _get_res(self):
        """Compute the resolution from the file metadata."""
        gridname = self.metadata["GridStructure"]["GRID_1"]["GridName"]
        if "CMG" not in gridname:
            raise ValueError("Only CMG grids are supported")

        # Get the grid resolution from the grid name
        pos = gridname.rfind("_") + 1
        pos2 = gridname.rfind("Deg")

        # Initialise number of rows and columns
        # Some products don't have resolution listed.
        if pos < 0 or pos2 < 0:
            return 360. / self.metadata["GridStructure"]["GRID_1"]["XDim"]
        else:
            return float(gridname[pos:pos2])

    def get_dataset(self, dataset_id, dataset_info):
        """Get DataArray for specified dataset."""
        dataset_name = dataset_id["name"]
        dataset = self.load_dataset(dataset_name, dataset_info.pop("category", False))
        self._add_satpy_metadata(dataset_id, dataset)

        return dataset

    def _get_area_extent(self):
        """Get the grid properties."""
        # Now compute the data extent
        upperleft = self.metadata["GridStructure"]["GRID_1"]["UpperLeftPointMtrs"]
        lowerright = self.metadata["GridStructure"]["GRID_1"]["LowerRightMtrs"]

        # For some reason, a few of the CMG products multiply their
        # decimal degree extents by one million. This fixes it.
        if lowerright[0] > 1e6 or upperleft[0] > 1e6:
            upperleft = tuple(val / 1e6 for val in upperleft)
            lowerright = tuple(val / 1e6 for val in lowerright)

        return upperleft[0], lowerright[1], lowerright[0], upperleft[1]

    def get_area_def(self, dsid):
        """Get the area definition.

        This is fixed, but not defined in the file. So we must
        generate it ourselves with some assumptions.
        """
        proj_param = "EPSG:4326"

        # Get the size of the dataset
        nrows = self.metadata["GridStructure"]["GRID_1"]["YDim"]
        ncols = self.metadata["GridStructure"]["GRID_1"]["XDim"]

        # Construct the area definition
        area = geometry.AreaDefinition("gridded_modis",
                                       "A gridded L3 MODIS area",
                                       "longlat",
                                       proj_param,
                                       ncols,
                                       nrows,
                                       self._get_area_extent())

        return area
