#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Satpy developers
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

The next version of NinJo (release expected spring 2022) will be able
to read standard GeoTIFF images, with required metadata encoded as a set
of XML tags in the GDALMetadata TIFF tag.  Each of the XML tags must be
prepended with ``'NINJO_'``.  For NinJo delivery, these GeoTIFF files
supersede the old NinJoTIFF format.  The :class:`NinJoGeoTIFFWriter`
therefore supersedes the old Satpy NinJoTIFF writer and the pyninjotiff
package.

The reference documentation for valid NinJo tags and their meaning is
contained in `NinJoPedia`_.  Since this page is not in the public web,
there is a (possibly outdated) `mirror`_.

.. _NinJoPedia: https://ninjopedia.com/tiki-index.php?page=adm_SatelliteServer_SatelliteImportFormats_en
.. _mirror: https://www.ssec.wisc.edu/~davidh/polar2grid/misc/NinJo_Satellite_Import_Formats.html

There are some user-facing differences between the old NinJoTIFF writer
and the new NinJoGeoTIFF writer.  Most notably, keyword arguments that
correspond to tags directly passed by the user are now identical,
including case, to how they will be written to the GDALMetaData and
interpreted by NinJo.  That means some keyword arguments have changed,
such as summarised in this table:

.. list-table:: Migrating to NinJoGeoTIFF, keyword arguments for the writer
   :header-rows: 1

   * - ninjotiff (old)
     - ninjogeotiff (new)
     - Notes
   * - ``chan_id``
     - ``ChannelID``
     - mandatory
   * - ``data_cat``
     - ``DataType``
     - mandatory
   * - ``physic_unit``
     - ``PhysicUnit``
     - mandatory
   * - ``physic_val``
     - ``PhysicValue``
     - mandatory
   * - ``sat_id``
     - ``SatelliteNameID``
     - mandatory
   * - ``data_source``
     - ``DataSource``
     - optional

Moreover, two keyword arguments are no longer supported because
their functionality has become redundant.  This applies to
``ch_min_measurement_unit`` and ``ch_max_measurement_unit``.
Instead, pass those values in source units to the
:func:`~satpy.enhancements.stretch` enhancement with the ``min_stretch``
and ``max_stretch`` arguments.
"""

import copy
import datetime
import logging

import numpy as np

from .geotiff import GeoTIFFWriter

logger = logging.getLogger(__name__)


class NinJoGeoTIFFWriter(GeoTIFFWriter):
    """Writer for GeoTIFFs with NinJo tags.

    This writer is experimental.  API may be subject to change.

    For information, see module docstring and documentation for
    :meth:`~NinJoGeoTIFFWriter.save_image`.
    """

    def save_image(
            self, image, filename=None, fill_value=None,
            compute=True, keep_palette=False, cmap=None, overviews=None,
            overviews_minsize=256, overviews_resampling=None,
            tags=None, config_files=None,
            *, ChannelID, DataType, PhysicUnit, PhysicValue,
            SatelliteNameID, **kwargs):
        """Save image along with NinJo tags.

        Save image along with NinJo tags.  Interface as for GeoTIFF,
        except NinJo expects some additional tags.  Those tags will be
        prepended with ``ninjo_`` and added as GDALMetaData.

        Writing such images requires trollimage 1.16 or newer.

        Importing such images with NinJo requires NinJo 7 or newer.

        Args:
            image (:class:`~trollimage.xrimage.XRImage`):
                Image to save.
            filename (str): Where to save the file.
            fill_value (int): Which pixel value is fill value?
            compute (bool): To compute or not to compute, that is the question.
            keep_palette (bool):
                As for parent GeoTIFF :meth:`~satpy.writers.geotiff.GeoTIFFWriter.save_image`.
            cmap (:class:`trollimage.colormap.Colormap`):
                As for parent :meth:`~satpy.writers.geotiff.GeoTIFFWriter.save_image`.
            overviews (list):
                As for :meth:`~satpy.writers.geotiff.GeoTIFFWriter.save_image`.
            overviews_minsize (int):
                As for :meth:`~satpy.writers.geotiff.GeoTIFFWriter.save_image`.
            overviews_resampling (str):
                As for :meth:`~satpy.writers.geotiff.GeoTIFFWriter.save_image`.
            tags (dict): Extra (not NinJo) tags to add to GDAL MetaData
            config_files (Any): Not directly used by this writer, supported
                for compatibility with other writers.

        Remaining keyword arguments are either passed as GDAL options,
        if contained in ``self.GDAL_OPTIONS``, or they are passed
        to :class:`NinJoTagGenerator`, which will include them as
        NinJo tags in GDALMetadata.  Supported tags are defined in
        ``NinJoTagGenerator.optional_tags``.  The meaning of those (and
        other) tags are defined in the NinJo documentation (see module
        documentation for a link to NinJoPedia).  The following tags
        are mandatory and must be provided as keyword arguments:

            ChannelID (int)
                NinJo Channel ID
            DataType (int)
                NinJo Data Type
            SatelliteNameID (int)
                NinJo Satellite ID
            PhysicUnit (str)
                NinJo label for unit (example: "C").  If PhysicValue is set to
                "Temperature", PhysicUnit is set to "C", but data attributes
                incidate the data have unit "K", then the writer will adapt the
                header ``ninjo_AxisIntercept`` such that data are interpreted
                in units of "C".
            PhysicValue (str)
                NinJo label for quantity (example: "temperature")

        """
        dataset = image.data

        # filename not passed on to writer by Scene.save_dataset, but I need
        # it!
        filename = filename or self.get_filename(**dataset.attrs)

        gdal_opts = {}
        ntg_opts = {}
        for (k, v) in kwargs.items():
            if k in self.GDAL_OPTIONS:
                gdal_opts[k] = v
            else:
                ntg_opts[k] = v

        ntg = NinJoTagGenerator(
            image,
            fill_value=fill_value,
            filename=filename,
            ChannelID=ChannelID,
            DataType=DataType,
            PhysicUnit=PhysicUnit,
            PhysicValue=PhysicValue,
            SatelliteNameID=SatelliteNameID,
            **ntg_opts)
        ninjo_tags = {f"ninjo_{k:s}": v for (k, v) in ntg.get_all_tags().items()}
        image = self._fix_units(image, PhysicValue, PhysicUnit)

        return super().save_image(
            image,
            filename=filename,
            fill_value=fill_value,
            compute=compute,
            keep_palette=keep_palette,
            cmap=cmap,
            overviews=overviews,
            overviews_minsize=overviews_minsize,
            overviews_resampling=overviews_resampling,
            tags={**(tags or {}), **ninjo_tags},
            scale_offset_tags=None if image.mode.startswith("RGB") else ("ninjo_Gradient", "ninjo_AxisIntercept"),
            **gdal_opts)

    def _fix_units(self, image, quantity, unit):
        """Adapt units between °C and K.

        This will return a new XRImage, to make sure the old data and
        enhancement history aren't touched.
        """
        data_units = image.data.attrs.get("units")
        if (quantity.lower() == "temperature" and
                unit == "C" and
                data_units == "K"):
            logger.debug("Adding offset for K → °C conversion")
            new_attrs = copy.deepcopy(image.data.attrs)
            im2 = type(image)(image.data.copy())
            im2.data.attrs = new_attrs
            # this scale/offset has to be applied before anything else
            im2.data.attrs["enhancement_history"].insert(0, {"scale": 1, "offset": 273.15})
            return im2
        if unit != data_units and unit.lower() != "n/a":
            logger.warning(
                    f"Writing {unit!s} to ninjogeotiff headers, but "
                    f"data attributes have unit {data_units!s}. "
                    "No conversion applied.")

        return image


class NinJoTagGenerator:
    """Class to collect NinJo tags.

    This class is used by :class:`NinJoGeoTIFFWriter` to collect NinJo tags.
    Most end-users will not need to create instances of this class directly.

    Tags are gathered from three sources:

    - Fixed tags, contained in the attribute ``fixed_tags``.  The value of
      those tags is hardcoded and never changes.
    - Tags passed by the user, contained in the attribute ``passed_tags``.
      Those tags must be passed by the user as arguments to the writer, which
      will pass them on when instantiating this class.
    - Tags calculated from data and metadata.  Those tags are defined in the
      attribute ``dynamic_tags``.  They are either calculated from image data,
      from image metadata, or from arguments passed by the user to the writer.

    Some tags are mandatory (defined in ``mandatory_tags``).  All tags that are
    not mandatory are optional.  By default, optional tags are generated if and
    only if the required information is available.
    """

    # tags that never change
    fixed_tags = {
        "Magic": "NINJO",
        "HeaderVersion": 2,
        "XMinimum": 1,
        "YMinimum": 1}

    # tags that must be passed directly by the user
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

    # optional tags are added on best effort or if passed by user
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

    def __init__(self, image, fill_value, filename, **kwargs):
        """Initialise tag generator.

        Args:
            image (:class:`trollimage.xrimage.XRImage`): XRImage for which
                NinJo tags should be calculated.
            fill_value (int): Fill value corresponding to image.
            filename (str): Filename to be written.
            **kwargs: Any additional tags to be included as-is.
        """
        self.image = image
        self.dataset = image.data
        self.fill_value = fill_value
        self.filename = filename
        self.args = kwargs
        self.tag_names = (self.fixed_tags.keys() |
                          self.passed_tags |
                          self.dynamic_tags.keys() |
                          (self.args.keys() & self.optional_tags))
        if self.args.keys() - self.tag_names:
            raise ValueError("The following tags were not recognised: " +
                             " ".join(self.args.keys() - self.tag_names))

    def get_all_tags(self):
        """Get a dictionary with all tags for NinJo."""
        tags = {}
        for tag in self.tag_names:
            try:
                tags[tag] = self.get_tag(tag)
            except (AttributeError, KeyError) as e:
                if tag in self.mandatory_tags:
                    raise
                logger.debug(
                    f"Unable to obtain value for optional NinJo tag {tag:s}. "
                    f"This is probably expected.  The reason is: {e.args[0]}")
        return tags

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
        if self.image.mode in ("L", "P"):
            return 8
        if self.image.mode in ("LA", "PA"):
            return 16
        if self.image.mode == "RGB":
            return 24
        if self.image.mode == "RGBA":
            return 32
        raise ValueError(
                f"Unsupported image mode: {self.image.mode:s}")

    # Set unix epoch here explicitly, because datetime.timestamp() is
    # apparently not supported on Windows.
    _epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)

    def get_creation_date_id(self):
        """Calculate the creation date ID.

        That's seconds since UNIX Epoch for the time the image is created.
        """
        delta = datetime.datetime.now(tz=datetime.timezone.utc) - self._epoch
        return int(delta.total_seconds())

    def get_date_id(self):
        """Calculate the date ID.

        That's seconds since UNIX Epoch for the time corresponding to the
        satellite image start of measurement time.
        """
        tm = self.dataset.attrs["start_time"]
        delta = tm.replace(tzinfo=datetime.timezone.utc) - self._epoch
        return int(delta.total_seconds())

    def get_earth_radius_large(self):
        """Return the Earth semi-major axis in metre."""
        return self.dataset.attrs["area"].crs.ellipsoid.semi_major_metre

    def get_earth_radius_small(self):
        """Return the Earth semi-minor axis in metre."""
        return self.dataset.attrs["area"].crs.ellipsoid.semi_minor_metre

    def get_filename(self):
        """Return the filename."""
        return self.filename

    def get_min_gray_value(self):
        """Calculate minimum gray value."""
        return self.image._scale_to_dtype(
            self.dataset.min(),
            np.uint8,
            self.fill_value).astype(np.uint8)

    def get_max_gray_value(self):
        """Calculate maximum gray value."""
        return self.image._scale_to_dtype(
            self.dataset.max(),
            np.uint8,
            self.fill_value).astype(np.uint8)

    def get_projection(self):
        """Get NinJo projection string.

        From the documentation, valid values are:

        - NPOL/SPOL: polar-sterographic North/South
        - PLAT: „Plate Carrée“, equirectangular projection
        - MERC: Mercator projection

        Derived from AreaDefinition.
        """
        if self.dataset.attrs["area"].crs.coordinate_system.name == "ellipsoidal":
            # For lat/lon coordinates, we say it's PLAT
            return "PLAT"
        name = self.dataset.attrs["area"].crs.coordinate_operation.method_name
        if "Equidistant Cylindrical" in name:
            return "PLAT"
        if "Mercator" in name:
            return "MERC"
        if "Stereographic" in name:
            if self.get_ref_lat_1() >= 0:
                return "NPOL"
            return "SPOL"
        raise ValueError(
                "Unknown mapping from area "
                f"{self.dataset.attrs['area'].description!r} with CRS coordinate "
                f"operation name {name:s} to NinJo projection.  NinJo understands only "
                "equidistant cylindrical, mercator, or stereographic projections.")

    def get_ref_lat_1(self):
        """Get reference latitude one.

        Derived from area definition.
        """
        pams = {p.name: p.value for p in self.dataset.attrs["area"].crs.coordinate_operation.params}
        for label in ["Latitude of standard parallel",
                      "Latitude of natural origin",
                      "Latitude of 1st standard parallel"]:
            if label in pams:
                return pams[label]
        raise ValueError(
                "Could not find reference latitude for area "
                f"{self.dataset.attrs['area'].description}")

    def get_transparent_pixel(self):
        """Get the transparent pixel value, also known as the fill value.

        When the no fill value is defined (value `None`), such as for RGBA or
        LA images, returns -1, in accordance with the file format
        specification.
        """
        if self.fill_value is None:
            return -1
        return self.fill_value

    def get_xmaximum(self):
        """Get the maximum value of x, i.e. the meridional extent of the image in pixels."""
        return self.dataset.sizes["x"]

    def get_ymaximum(self):
        """Get the maximum value of y, i.e. the zonal extent of the image in pixels."""
        return self.dataset.sizes["y"]

    def get_meridian_east(self):
        """Get the easternmost longitude of the area.

        Currently not implemented.  In pyninjotiff it was implemented but the
        answer was incorrect.
        """
        raise NotImplementedError()

    def get_meridian_west(self):
        """Get the westernmost longitude of the area.

        Currently not implemented.  In pyninjotiff it was implemented but the
        answer was incorrect.
        """
        raise NotImplementedError()

    def get_ref_lat_2(self):
        """Get reference latitude two.

        This is not implemented and never was correctly implemented in
        pyninjotiff either.  It doesn't appear to be used by NinJo.
        """
        raise NotImplementedError("Second reference latitude not implemented.")
