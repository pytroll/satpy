# Copyright (c) 2025-2026 Satpy developers
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
"""VectorScene object to hold vector satellite data.

This module contains the VectorScene class, a sibling of the Scene
(RasterScene) class to contain vector data.  Here, by vector data, we mean
anything that is better described as a OGC Simple Feature Geometry by
ISO 19125 (https://www.ogc.org/standards/sfa/) than by a 2-dimensional raster.
Examples of simple feature geometries are LineString, MultiLineString, Polygon,
and MultiPolygon.  Users may be familiar with those due to their implementation
in the shapely library and in geopandas.  Example of satellite data that may be
suitable encoded in this form are geometry objects in the Rapidly Developing
Thunderstorms (RDT) or similar products, flash geometries, atmospheric motion
vectors, or lightning point data.
"""

import warnings
from collections.abc import Iterable

import pyproj

from .dataset import DataID
from .scene import BaseScene


class VectorScene(BaseScene):
    """Experimental class to encode vector data.

    WORK IN PROGRESS.

    The datasets in a VectorScene are not xarray dataarrays but geopandas
    geodataframes.  Some things work differently.

    Area covered by VectorScene datasets is not in the area attribute, but
    in the geometry attribute, like for any geopandas geodataframe.

    VectorScene cannot be resampled, but it can be reprojected with the
    reproject method.

    Auxiliary datasets are not supported.

    This is a WORK IN PROGRESS.

    The VectorScene is EXPERIMENTAL AND SUBJECT TO CHANGE.

    -----------------------------------------------------
    -------------    DO  NOT  USE !!!!!!  ---------------
    -----------------------------------------------------
    """

    def __init__(self, *args, **kwargs):
        """Initialise the VectorScene class.

        This initialisation is currently only a warning that this class
        is highly experimental.
        """
        warnings.warn(
                "VectorScene is an unfinished work in progress. "
                "Use at your own risk!",
                UserWarning)
        super().__init__(*args, **kwargs)

    def reproject(
            self,
            target: pyproj.CRS,
            datasets: Iterable | None = None):
        """Reproject contents to target CRS.

        Args:
            target: target CRS to reproject to
            datasets (optional): Reproject only a subset.  Default is None (all datasets).
        """
        new_scn = self.copy(datasets=datasets)
        all_datasets = new_scn._datasets.values()
        for ds in all_datasets:
            # from_dataarray also works for dataframes
            ds_id = DataID.from_dataarray(ds)
            new_scn._datasets[ds_id] = ds.to_crs(target)
        return new_scn
