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

import datetime

from .geotiff import GeoTIFFWriter
from . import get_enhanced_image


class NinJoGeoTIFFWriter(GeoTIFFWriter):
    """Writer for GeoTIFFs with NinJo tags.

    This writer is experimental.  API may be subject to change.
    """

    def save_dataset(
            self, dataset, fill_value=None,
            tags=None, **kwargs):
        """Save dataset along with NinJo tags.

        Save dataset along with NinJo tags.  Interface as for GeoTIFF,
        except NinJo expects some additional tags.  Those tags will be
        prepended with ninjo_ and added as GDALMetaData.

        This requires trollimage 1.16 or newer.

        Args:
            dataset (xr.DataArray): Data array to save.
            FIXME DOC
        """
        # some tag calculations, such as image depth, need the image to be
        # present already
        image = get_enhanced_image(dataset)
        ntg = NinJoTagGenerator(image, fill_value=fill_value, args=kwargs)
        ninjo_tags = {f"ninjo_{k:s}": v for (k, v) in ntg.get_all_tags().items()}
        # FIXME: Not really, I've already calculated the image!  What can
        # possibly go wrong?
        return super().save_dataset(
                dataset,
                fill_value=fill_value,
                tags=(tags or {}) | ninjo_tags, scale_label="Gradient",
                offset_label="AxisIntercept", **kwargs)


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

    # tags that can be calculated dynamically from (meta)data
    dynamic_tags = {
        "CentralMeridian": "central_meridian",
        "ColorDepth": "color_depth",
        "CreationDateID": "creation_date_id",
        "DateID": "date_id",
        "EarthRadiusLarge": "earth_radius_large",
        "EarthRadiusSmall": "earth_radius_small",
        "FileName": "filename",
        "MaxGrayValue": "max_gray_value",
        "MinGrayValue": "min_gray_value",
        "Projection": "projection",
        "ReferenceLatitude1": "ref_lat_1",
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

    # tags that are added later in other ways
    postponed_tags = {"AxisIntercept", "Gradient"}

    def __init__(self, image, fill_value, args):
        """Initialise tag generator.

        Args:
            img (trollimage.XRImage): Corresponding image.
            args: Other information.
        """
        self.image = image
        self.dataset = image.data
        self.fill_value = fill_value
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
        if tag in self.postponed_tags:
            raise ValueError(f"Tag {tag!s} is added later by the GeoTIFF writer.")
        if tag in self.optional_tags:
            raise ValueError(
                f"Optional tag {tag!s} must be supplied by user if user wants to "
                "request the value, but wasn't.")
        raise ValueError(f"Unknown tag: {tag!s}")

    def get_central_meridian(self):
        """Calculate central meridian."""
        pams = self.dataset.attrs["area"].crs.coordinate_operation.params
        lon_0 = {p.name: p.value for p in pams}["Longitude of natural origin"]
        return lon_0

    def get_color_depth(self):
        """Return the color depth."""
        # FIXME: Can I easily and reliably get this from the dataset rather
        # than from the image?
        if self.image.mode in "LP":
            return 8
        if self.image.mode in ("LA", "PA"):
            return 16
        if self.image.mode == "RGB":
            return 24
        if self.image.mode == "RGBA":
            return 32
        raise ValueError(
                f"Unsupported image mode: {self.image.mode:s}")

    def get_creation_date_id(self):
        """Calculate the creation date ID.

        That's seconds since UNIX Epoch for the time the image is created.
        """
        return int(datetime.datetime.now().timestamp())

    def get_date_id(self):
        """Calculate the date ID.

        That's seconds since UNIX Epoch for the time corresponding to the
        satellite image.
        """
        return int(self.dataset.attrs["start_time"].timestamp())

    def get_earth_radius_large(self):
        """Return the Earth semi-major axis in metre."""
        return self.dataset.attrs["area"].crs.ellipsoid.semi_major_metre

    def get_earth_radius_small(self):
        """Return the Earth semi-minor axis in metre."""
        return self.dataset.attrs["area"].crs.ellipsoid.semi_minor_metre

    def get_filename(self):
        """Return the filename."""
        return "papapath.tif"  # FIXME: derive

    def get_max_gray_value(self):
        """Calculate maximum gray value."""
        # FIXME: this is wrong - the data have been stretched to [0, 1] by
        # get_enhanced_image, and will only be scaled to [0, 255] later by the
        # GeoTIFFWriter finalize method; how to tell what is correct here?
        return self.dataset.max()

    def get_meridian_east(self):
        """Get the easternmost longitude of the area.

        Currently not implemented.  In pyninjotiff it was implemented but the
        answer was incorrect.
        """
        raise NotImplementedError("This is difficult and probably not needed.")

    def get_meridian_west(self):
        """Get the westernmost longitude of the area.

        Currently not implemented.  In pyninjotiff it was implemented but the
        answer was incorrect.
        """
        raise NotImplementedError("This is difficult and probably not needed.")

    def get_min_gray_value(self):
        """Calculate minimum gray value."""
        return self.dataset.min()

    def get_projection(self):
        """Get NinJo projection string.

        From the documentation, valid values are:

        - NPOL/SPOL: polar-sterographic North/South
        - PLAT: „Plate Carrée“, equirectangular projection
        - MERC: Mercator projection
        """
        name = self.dataset.attrs["area"].crs.coordinate_operation.method_name
        if "Equidistant Cylindrical" in name:
            return "PLAT"
        if "Mercator" in name:
            return "MERC"
        if "Stereographic" in name:
            if self.get_ref_lat_1() >= 0:
                return "NPOL"
            else:
                return "SPOL"
        raise ValueError(
                f"Area {self.dataset.attrs['area'].name} has a CRS coordinate "
                f"operation names {name:s}.  I don't know what that corresponds "
                "to in NinJo, which understands only equidistanc cylindrical, "
                "mercator, or stereographic projections.")

    def get_ref_lat_1(self):
        """Get reference latitude one."""
        pams = {p.name: p.value for p in self.dataset.attrs["area"].crs.coordinate_operation.params}
        for label in ["Latitude of standard parallel",
                      "Latitude of natural origin",
                      "Latitude of 1st standard parallel"]:
            if label in pams:
                return pams[label]
        raise ValueError(
                "Could not find reference latitude for area "
                f"{self.dataset.attrs['area'].name}")

    def get_ref_lat_2(self):
        """Get reference latitude two."""
        raise NotImplementedError("Second reference latitude not implemented.")

    def get_transparent_pixel(self):
        """Get transparent pixel value."""
        return self.fill_value

    def get_xmaximum(self):
        """Get xmaximum."""
        return self.dataset.sizes["x"]

    def get_ymaximum(self):
        """Get ymaximum."""
        return self.dataset.sizes["y"]
