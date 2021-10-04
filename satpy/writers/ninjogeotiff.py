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

The reference documentation for valid NinJo tags and their meaning is contained
in NinJoPedia at
https://ninjopedia.com/tiki-index.php?page=adm_SatelliteServer_SatelliteImportFormats_en.
Since this page is not in the public web, a (possibly outdated) mirror is
located at https://www.ssec.wisc.edu/~davidh/polar2grid/misc/NinJo_Satellite_Import_Formats.html.
"""

from .geotiff import GeoTIFFWriter


class NinJoGeoTIFFWriter(GeoTIFFWriter):
    """Writer for GeoTIFFs with NinJo tags."""

    def save_dataset(self, dataset, **kwargs):
        """Save dataset along with NinJo tags.

        Save dataset along with NinJo tags.  Interface as for GeoTIFF,
        except NinJo expects some additional tags.  Those tags will be
        prepended with ninjo_ and added as GDALMetaData.

        Args:
            dataset (xr.DataArray): Data array to save.
            ninjo_tags (Mapping[str, str|numeric): tags to add
        """
        tags = calc_tags_from_dataset(dataset, args=kwargs)
        super().save_dataset(
                dataset,
                tags={"ninjo_" + k: v for (k, v) in tags.items()},
                **kwargs)


class NinJoTagGenerator:
    """Class to collect NinJo tags.

    This class contains functionality to collect NinJo tags.  Tags are gathered
    from three sources:

    - Fixed tags, contained in the attribute ``fixed_tags``.  The value of
      those tags is hardcoded and never changes.
    - Tags passed by the user, contained in the attribute ``passed_tags``.
      Those tags must be passed by the user as arguments to the writer, which
      will pass them on when instantiating the class.
    - Tags calculated from data and metadata.  Those tags are defined in the
      attribute ``dynamic_tags``.

    Some tags are mandatory (defined in ``mandatory_tags``).  All tags that are
    not mandatory are optional.  By default, no optional tags are generated.
    Optional tags are only generated if passed on to the writer.
    """

    # tags that never change
    fixed_tags = {
        "Magic": "NINJO",
        "HeaderVersion": 2,
        "XMinimum": 1,
        "YMinimum": 1}

    # tags that should be passed directly by the user
    passed_tags = {"ChannelID", "DataType", "PhysicUnit",
                   "SatelliteNameID", "PhysicValue"}

    # tags that are calculated dynamically from (meta)data
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

    # mandatory tags according to documentation
    mandatory_tags = {"SatelliteNameID", "DateID", "CreationDateID",
                      "ChannelID", "HeaderVersion", "DataType",
                      "SatelliteNumber", "ColorDepth", "XMinimum", "XMaximum",
                      "YMinimum", "YMaximum", "Projection", "PhysicValue",
                      "PhysicUnit", "MinGrayValue", "MaxGrayValue", "Gradient",
                      "AxisIntercept", "TransparentPixel"}

    optional_tags = {"DataSource", "MeridianWest", "MeridianEast",
                     "EarthRadiusLarge", "EarthRadiusSmall", "GeodeticDate",
                     "ReferenceLatitude1", "ReferenceLatitude2",
                     "CentralMeridian", "ColorTable", "Description",
                     "OverflightDirection", "GeoLatitude", "GeoLongitude",
                     "Altitude", "AOSAzimuth", "LOSAzimuth", "MaxElevation",
                     "OverFlightTime", "IsBlackLinesCorrection",
                     "IsAtmosphereCorrected", "IsCalibrated", "IsNormalized",
                     "OriginalHeader", "IsValueTableAvailable",
                     "ValueTableFloatField"}

    def __init__(self, dataset, args):
        """Initialise tag generator."""
        self.dataset = dataset
        self.args = args
        self.tag_names = (self.fixed_tags.keys() |
                          self.passed_tags |
                          self.dynamic_tags.keys() |
                          (self.args.keys() & self.optional_tags))

    def get_all_tags(self):
        """Get a dictionary with all tags for NinJo."""
        return {tag: self.get_tag(tag) for tag in self.tag_names}

    def get_tag(self, tag):
        """Get value for NinJo tag."""
        if tag in self.fixed_tags:
            return self.fixed_tags[tag]
        if tag in self.passed_tags:
            return self.args[tag]
        if tag in self.dynamic_tags:
            return getattr(self, f"get_{self.dynamic_tags[tag]:s}")()
        if tag in self.optional_tags and tag in self.args:
            return self.args[tag]
        if tag in self.optional_tags:
            raise ValueError(
                f"Optional tag {tag!s} must be supplied by user if user wants to "
                "request the value, but wasn't.")
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
