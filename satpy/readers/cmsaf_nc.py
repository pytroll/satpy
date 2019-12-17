"""Module containing CMSAF FileHandler
"""

import pyresample.geometry
from .netcdf_utils import NetCDF4FileHandler
from . import _geos_area


class NcCMSAF(NetCDF4FileHandler):
    def get_dataset(self, dataset_id, info):
        return self[dataset_id.name]

    def get_area_def(self, dataset_id):
        return pyresample.geometry.AreaDefinition(
                "some_area_name",
                "on-the-fly area",
                "geos",
                self["/attr/CMSAF_proj4_params"],
                self["/dimension/x"],
                self["/dimension/y"],
                self["/attr/CMSAF_area_extent"])
