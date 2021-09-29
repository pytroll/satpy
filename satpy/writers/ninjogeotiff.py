# Copyright (c) 2021- Satpy developers
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
"""Writer for GeoTIFF images with tags for the NinJo visualization tool."""

from .geotiff import GeoTIFFWriter


class NinJoGeoTIFFWriter(GeoTIFFWriter):
    """Writer for GeoTIFFs with NinJo tags."""

    def save_datasets(self, datasets, ninjo_tags):
        """Save datasets along with NinJo tags.

        Save datasets along with NinJo tags.  Interface as for GeoTIFF, except
        it takes an additional keyword argument ninjo_tags.  Those tags will be
        prepended with ninjo_ and added as GDALMetaData.
        """
        dynamic_tags = {"TransparentPixel": "0"}
        super().save_datasets(
                datasets,
                tags={"ninjo_" + k: v for (k, v) in (ninjo_tags | dynamic_tags).items()})
