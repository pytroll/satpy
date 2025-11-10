# Copyright (c) 2025 Satpy developers
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
anything that is better described as a OGS Simple Feature Geometry by
ISO 19125 (https://www.ogc.org/standards/sfa/) than by a 2-dimensional raster.
Examples of simple feature geometries are LineString, MultiLineString, Polygon,
and MultiPolygon.  Users may be familiar with those due to their implementation
in the shapely library and in geopandas.  Example of satellite data that may be
suitable encoded in this form are geometry objects in the Rapidly Developing
Thunderstorms (RDT) or similar products, flash geometries, atmospheric motion
vectors, or lightning point data.
"""

class VectorScene:
    """Experimental class to encode vector data."""

    def __init__(self):
        """Not initialising anything yet."""
        raise NotImplementedError("VectorScene not implemented yet.")
