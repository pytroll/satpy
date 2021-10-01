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
"""Writer for GeoTIFF images with tags for the NinJo visualization tool.

Since NinJo version 7 (released spring 2022), NinJo is able to read standard
GeoTIFF images, with required metadata encoded as a set of XML tags in the
GDALMetadata TIFF tag.  Each of the XML tags must be prepended with
``'NINJO_'``.
"""

from .geotiff import GeoTIFFWriter


class NinJoGeoTIFFWriter(GeoTIFFWriter):
    """Writer for GeoTIFFs with NinJo tags."""

    def save_dataset(self, dataset, ninjo_tags):
        """Save dataset along with NinJo tags.

        Save dataset along with NinJo tags.  Interface as for GeoTIFF, except
        it takes an additional keyword argument ninjo_tags.  Those tags will be
        prepended with ninjo_ and added as GDALMetaData.

        Args:
            dataset (xr.DataArray): Data array to save.
            ninjo_tags (Mapping[str, str|numeric): tags to add
        """
        tags = calc_tags_from_dataset(dataset, ninjo_tags)
        super().save_dataset(
                dataset,
                tags={"ninjo_" + k: v for (k, v) in tags.items()})


class NinJoTagGenerator:
    """Class to calculate NinJo tags from content."""

    # tags that never change
    fixed_tags = {
        "Magic": "NINJO",
        "HeaderVersion": 2,
        "XMinimum": 1,
        "YMinimum": 1,
        "PhysicValue": "unknown"}

    # tags that should be passed directly by the user
    passed_tags = {"ChannelID", "DataSource", "DataType", "PhysicUnit",
                   "SatelliteNameID"}

    # tags that are calculated dynamically
    dynamic_tags = {
        "AxisIntercept": "axis_intercept",
        "CentralMeridian": "central_meridian",
        "ColorDepth": "color_depth",
        "CreationDateID": "creation_date_id",
        "DateID": "date_id",
        "EarthRadiusLarge": "earth_radius_large",
        "EarthRadiusSmall": "earth_radius_small",
        "FileName": "filename",
        "Gradient": "gradient",
        "IsAtmosphereCorrected": "atmosphere_corrected",
        "IsBlackLineCorrection": "black_line_corrected",
        "IsCalibrated": "is_calibrated",
        "IsNormalized": "is_normalized",
        "MaxGrayValue": "max_gray_value",
        "MeridianEast": "meridian_east",
        "MeridianWest": "meridian_west",
        "MinGrayValue": "min_gray_value",
        "Projection": "projection",
        "ReferenceLatitude1": "ref_lat_1",
        "ReferenceLatitude2": "ref_lat_2",
        "TransparentPixel": "transparent_pixel",
        "XMaximum": "xmaximum",
        "YMaximum": "ymaximum"
        }

    def __init__(self, dataset, args):
        """Initialise tag generator."""
        self.dataset = dataset
        self.args = args
        self.tag_names = (self.fixed_tags.keys() |
                          self.passed_tags |
                          self.dynamic_tags.keys())

    def get_all_tags(self):
        """Get a dictionary with all tags for NinJo."""
        return {tag: self.get_tag(tag) for tag in self.tag_names}

    def get_tag(self, tag):
        """Get value for NinJo tag."""
        if tag in self.fixed_tags:
            return self.fixed_tags[tag]
        elif tag in self.passed_tags:
            return self.args[tag]
        elif tag in self.dynamic_tags:
            return getattr(self, f"get_{self.dynamic_tags[tag]:s}")()
        else:
            raise ValueError(f"Unknown tag: {tag!s}")

    def get_axis_intercept(self):
        """Calculate the axis intercept."""
        return -88.0  # FIXME: derive from content

    def get_central_meridian(self):
        """Calculate central meridian."""
        return 0.0  # FIXME: derive from area

    def get_color_depth(self):
        """Return the color depth."""
        return 24  # FIXME: derive from image type

    def get_creation_date_id(self):
        """Calculate the creation date ID."""
        return 1632820093  # FIXME: derive from metadata

    def get_date_id(self):
        """Calculate the date ID."""
        return 1623581777  # FIXME: derive from metadata

    def get_earth_radius_large(self):
        """Return the Earth semi-major axis."""
        return 6378137.0  # FIXME: derive from area

    def get_earth_radius_small(self):
        """Return the Earth semi-minor axis."""
        return 6356752.5  # FIXME: derive from area

    def get_filename(self):
        """Return the filename."""
        return "papapath.tif"  # FIXME: derive

    def get_gradient(self):
        """Return the gradient."""
        return 0.5  # FIXME: derive from content

    def get_atmosphere_corrected(self):
        """Return whether atmosphere is corrected."""
        return 0  # FIXME: derive from metadata

    def get_black_line_corrected(self):
        """Return whether black line correction applied."""
        return 0  # FIXME: derive from metadata

    def get_is_calibrated(self):
        """Return whether calibration has been applied."""
        return 1  # FIXME: derive from metadata

    def get_is_normalized(self):
        """Return whether data have been normalized."""
        return 0  # FIXME: derive from metadata

    def get_max_gray_value(self):
        """Calculate maximum gray value."""
        return 255  # FIXME: calculate

    def get_meridian_east(self):
        """Calculate meridian east."""
        return 45.0  # FIXME: derive from area

    def get_meridian_west(self):
        """Calculate meridian west."""
        return -135.0  # FIXME: derive from area

    def get_min_gray_value(self):
        """Calculate minimum gray value."""
        return 0  # FIXME: derive from content

    def get_projection(self):
        """Get projection."""
        return "NPOL"  # FIXME: derive from area

    def get_ref_lat_1(self):
        """Get reference latitude one."""
        return 60.0  # FIXME: derive from area

    def get_ref_lat_2(self):
        """Get reference latitude two."""
        return 0  # FIXME: derive from area

    def get_transparent_pixel(self):
        """Get transparent pixel value."""
        return 0  # FIXME: derive from arguments

    def get_xmaximum(self):
        """Get xmaximum."""
        return self.dataset.sizes["x"]

    def get_ymaximum(self):
        """Get ymaximum."""
        return self.dataset.sizes["y"]


def calc_tags_from_dataset(dataset, args):
    """Calculate NinJo tags from dataset.

    For a dataset (xarray.DataArray), calculate content-dependent tags.
    """
    ntg = NinJoTagGenerator(dataset, args)
    return ntg.get_all_tags()
