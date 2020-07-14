#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2019 Satpy developers
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
"""Interface to MTG-FCI-FDHSI L1C NetCDF files.

This module defines the :class:`FCIFDHSIFileHandler` file handler, to
be used for reading Meteosat Third Generation (MTG) Flexible Combined
Imager (FCI) Full Disk High Spectral Imagery (FDHSI) data.  FCI will fly
on the MTG Imager (MTG-I) series of satellites, scheduled to be launched
in 2021 by the earliest.  For more information about FCI, see `EUMETSAT`_.

Geolocation is based on information from the data files.  It uses:

    * From the shape of the data variable ``data/<channel>/measured/effective_radiance``,
      start and end line columns of current swath.
    * From the data variable ``data/<channel>/measured/x``, the x-coordinates
      for the grid, in radians
    * From the data variable ``data/<channel>/measured/y``, the y-coordinates
      for the grid, in radians
    * From the attribute ``semi_major_axis`` on the data variable
      ``data/mtg_geos_projection``, the Earth equatorial radius
    * From the attribute ``semi_minor_axis`` on the same, the Earth polar
      radius
    * From the attribute ``perspective_point_height`` on the same data
      variable, the geostationary altitude in the normalised geostationary
      projection (see `PUG`_ §5.2)
    * From the attribute ``longitude_of_projection_origin`` on the same
      data variable, the longitude of the projection origin
    * From the attribute ``inverse_flattening`` on the same data variable, the
      (inverse) flattening of the ellipsoid
    * From the attribute ``sweep_angle_axis`` on the same, the sweep angle
      axis, see https://proj.org/operations/projections/geos.html

From the pixel centre angles in radians and the geostationary altitude, the
extremities of the lower left and upper right corners are calculated in units
of arc length in m.  This extent along with the number of columns and rows, the
sweep angle axis, and a dictionary with equatorial radius, polar radius,
geostationary altitude, and longitude of projection origin, are passed on to
``pyresample.geometry.AreaDefinition``, which then uses proj4 for the actual
geolocation calculations.

The brightness temperature calculation is based on the formulas indicated in
`PUG`_ and `RADTOBR`_.

The reading routine supports channel data in counts, radiances, and (depending
on channel) brightness temperatures or reflectances.  For each channel, it also
supports the pixel quality, obtained by prepending the channel name such as
``"vis_04_pixel_quality"``.

.. warning::
    The API for the direct reading of pixel quality is temporary and likely to
    change.  Currently, for each channel, the pixel quality is available by
    ``<chan>_pixel_quality``.  In the future, they will likely all be called
    ``pixel_quality`` and disambiguated by a to-be-decided property in the
    `DatasetID`.

.. _RADTOBR: https://www.eumetsat.int/website/wcm/idc/idcplg?IdcService=GET_FILE&dDocName=PDF_EFFECT_RAD_TO_BRIGHTNESS&RevisionSelectionMethod=LatestReleased&Rendition=Web
.. _PUG: http://www.eumetsat.int/website/wcm/idc/idcplg?IdcService=GET_FILE&dDocName=PDF_DMT_719113&RevisionSelectionMethod=LatestReleased&Rendition=Web
.. _EUMETSAT: https://www.eumetsat.int/website/home/Satellites/FutureSatellites/MeteosatThirdGeneration/MTGDesign/index.html#fci  # noqa: E501
"""

from __future__ import (division, absolute_import, print_function,
                        unicode_literals)

import logging
import numpy as np
import xarray as xr

from pyresample import geometry
from netCDF4 import default_fillvals

from .netcdf_utils import NetCDF4FileHandler

logger = logging.getLogger(__name__)


class FCIFDHSIFileHandler(NetCDF4FileHandler):
    """Class implementing the MTG FCI FDHSI File .

    This class implements the Meteosat Third Generation (MTG) Flexible
    Combined Imager (FCI) Full Disk High Spectral Imagery (FDHSI) reader.
    It is designed to be used through the :class:`~satpy.Scene`
    class using the :mod:`~satpy.Scene.load` method with the reader
    ``"fci_l1c_fdhsi"``.

    """

    # Platform names according to the MTG FCI L1 Product User Guide,
    # EUM/MTG/USR/13/719113 from 2019-06-27, pages 32 and 124, are MTI1, MTI2,
    # MTI3, and MTI4, but we want to use names such as described in WMO OSCAR
    # MTG-I1, MTG-I2, MTG-I3, and MTG-I4.
    #
    # After launch: translate to METEOSAT-xx instead?  Not sure how the
    # numbering will be considering MTG-S1 and MTG-S2 will be launched
    # in-between.
    _platform_name_translate = {
            "MTI1": "MTG-I1",
            "MTI2": "MTG-I2",
            "MTI3": "MTG-I3",
            "MTI4": "MTG-I4"}

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize file handler."""
        super(FCIFDHSIFileHandler, self).__init__(filename, filename_info,
                                                  filetype_info,
                                                  cache_var_size=10000,
                                                  cache_handle=True)
        logger.debug('Reading: {}'.format(self.filename))
        logger.debug('Start: {}'.format(self.start_time))
        logger.debug('End: {}'.format(self.end_time))

        self._cache = {}

    @property
    def start_time(self):
        """Get start time."""
        return self.filename_info['start_time']

    @property
    def end_time(self):
        """Get end time."""
        return self.filename_info['end_time']

    def get_dataset(self, key, info=None):
        """Load a dataset."""
        logger.debug('Reading {} from {}'.format(key.name, self.filename))
        if "pixel_quality" in key.name:
            return self._get_dataset_quality(key, info=info)
        elif any(lb in key.name for lb in {"vis_", "ir_", "nir_", "wv_"}):
            return self._get_dataset_measurand(key, info=info)
        else:
            raise ValueError("Unknown dataset key, not a channel or quality: "
                             f"{key.name:s}")

    def _get_dataset_measurand(self, key, info=None):
        """Load dataset corresponding to channel measurement.

        Load a dataset when the key refers to a measurand, whether uncalibrated
        (counts) or calibrated in terms of brightness temperature, radiance, or
        reflectance.
        """
        # Get the dataset
        # Get metadata for given dataset
        measured = self.get_channel_measured_group_path(key.name)
        data = self[measured + "/effective_radiance"]

        attrs = data.attrs.copy()
        info = info.copy()

        fv = attrs.pop(
            "FillValue",
            default_fillvals.get(data.dtype.str[1:], np.nan))
        vr = attrs.get("valid_range", [-np.inf, np.inf])
        if key.calibration == "counts":
            attrs["_FillValue"] = fv
            nfv = fv
        else:
            nfv = np.nan
        data = data.where(data >= vr[0], nfv)
        data = data.where(data <= vr[1], nfv)

        res = self.calibrate(data, key)

        # pre-calibration units no longer apply
        info.pop("units")
        attrs.pop("units")

        # For each channel, the effective_radiance contains in the
        # "ancillary_variables" attribute the value "pixel_quality".  In
        # FileYAMLReader._load_ancillary_variables, satpy will try to load
        # "pixel_quality" but is lacking the context from what group to load
        # it.  Until we can have multiple pixel_quality variables defined (for
        # example, with https://github.com/pytroll/satpy/pull/1088), rewrite
        # the ancillary variable to include the channel.  See also
        # https://github.com/pytroll/satpy/issues/1171.
        if "pixel_quality" in attrs["ancillary_variables"]:
            attrs["ancillary_variables"] = attrs["ancillary_variables"].replace(
                    "pixel_quality", key.name + "_pixel_quality")
        else:
            raise ValueError(
                "Unexpected value for attribute ancillary_variables, "
                "which the FCI file handler intends to rewrite (see "
                "https://github.com/pytroll/satpy/issues/1171 for why). "
                f"Expected 'pixel_quality', got {attrs['ancillary_variables']:s}")

        res.attrs.update(key.to_dict())
        res.attrs.update(info)
        res.attrs.update(attrs)

        res.attrs["platform_name"] = self._platform_name_translate.get(
                self["/attr/platform"], self["/attr/platform"])

        # remove unpacking parameters for calibrated data
        if key.calibration in ['brightness_temperature', 'reflectance']:
            res.attrs.pop("add_offset")
            res.attrs.pop("warm_add_offset")
            res.attrs.pop("scale_factor")
            res.attrs.pop("warm_scale_factor")

        # remove attributes from original file which don't apply anymore
        res.attrs.pop('long_name')

        return res

    def _get_dataset_quality(self, key, info=None):
        """Load quality for channel.

        Load a quality field for an FCI channel.  This is a bit involved in
        case of FCI because each channel group (data/<channel>/measured) has
        its own data variable 'pixel_quality', referred to in ancillary
        variables (see also Satpy issue 1171), so some special treatment in
        necessary.
        """
        # FIXME: replace by .removesuffix after we drop support for Python < 3.9
        if key.name.endswith("_pixel_quality"):
            chan_lab = key.name[:-len("_pixel_quality")]
        else:
            raise ValueError("Quality label must end with pixel_quality, got "
                             f"{key.name:s}")
        grp_path = self.get_channel_measured_group_path(chan_lab)
        dv_path = grp_path + "/pixel_quality"
        data = self[dv_path]
        return data

    def get_channel_measured_group_path(self, channel):
        """Get the channel's measured group path."""
        measured_group_path = 'data/{}/measured'.format(channel)

        return measured_group_path

    def calc_area_extent(self, key):
        """Calculate area extent for a dataset."""
        # if a user requests a pixel quality before the channel data, the
        # yaml-reader will ask the area extent of the pixel quality field,
        # which will ultimately end up here
        if key.name.endswith("_pixel_quality"):
            lab = key.name[:-len("_pixel_quality")]
        else:
            lab = key.name
        # Get metadata for given dataset
        measured = self.get_channel_measured_group_path(lab)
        # Get start/end line and column of loaded swath.
        nlines, ncols = self[measured + "/effective_radiance/shape"]

        logger.debug('Channel {} resolution: {}'.format(lab, ncols))
        logger.debug('Row/Cols: {} / {}'.format(nlines, ncols))

        # Calculate full globe line extent
        h = float(self["data/mtg_geos_projection/attr/perspective_point_height"])

        ext = {}
        for c in "xy":
            c_radian = self["data/{:s}/measured/{:s}".format(lab, c)]
            c_radian_num = c_radian[:] * c_radian.scale_factor + c_radian.add_offset

            # FCI defines pixels by centroids (Example Products for Pytroll
            # Workshop, §B.4.2)
            #
            # pyresample defines corners as lower left corner of lower left pixel,
            # upper right corner of upper right pixel (Martin Raspaud, personal
            # communication).

            # the .item() call is needed with the h5netcdf backend, see
            # https://github.com/pytroll/satpy/issues/972#issuecomment-558191583
            # but we need to compute it first if this is dask
            min_c_radian = c_radian_num[0] - c_radian.scale_factor/2
            max_c_radian = c_radian_num[-1] + c_radian.scale_factor/2
            min_c = min_c_radian * h  # arc length in m
            max_c = max_c_radian * h
            try:
                min_c = min_c.compute()
                max_c = max_c.compute()
            except AttributeError:  # not a dask.array
                pass
            ext[c] = (min_c.item(), max_c.item())

        area_extent = (ext["x"][1], ext["y"][1], ext["x"][0], ext["y"][0])
        return area_extent, nlines, ncols

    def get_area_def(self, key, info=None):
        """Calculate on-fly area definition for 0 degree geos-projection for a dataset."""
        # assumption: channels with same resolution should have same area
        # cache results to improve performance
        if key.resolution in self._cache.keys():
            return self._cache[key.resolution]

        a = float(self["data/mtg_geos_projection/attr/semi_major_axis"])
        b = float(self["data/mtg_geos_projection/attr/semi_minor_axis"])
        h = float(self["data/mtg_geos_projection/attr/perspective_point_height"])
        if_ = float(self["data/mtg_geos_projection/attr/inverse_flattening"])
        lon_0 = float(self["data/mtg_geos_projection/attr/longitude_of_projection_origin"])
        sweep = str(self["data/mtg_geos_projection"].sweep_angle_axis)
        # Channel dependent swath resolution
        area_extent, nlines, ncols = self.calc_area_extent(key)
        logger.debug('Calculated area extent: {}'
                     .format(''.join(str(area_extent))))

        proj_dict = {'a': a,
                     'b': b,
                     'lon_0': lon_0,
                     'h': h,
                     "fi": float(if_),
                     'proj': 'geos',
                     'units': 'm',
                     "sweep": sweep}

        area = geometry.AreaDefinition(
            'some_area_name',
            "On-the-fly area",
            'geosfci',
            proj_dict,
            ncols,
            nlines,
            area_extent)

        self._cache[key.resolution] = area
        return area

    def calibrate(self, data, key):
        """Calibrate data."""
        if key.calibration == "counts":
            # from package description, this just means not applying add_offset
            # and scale_factor
            data.attrs["units"] = "1"
        elif key.calibration in ['brightness_temperature', 'reflectance', 'radiance']:
            data = self.calibrate_counts_to_physical_quantity(data, key)
        else:
            logger.error(
                "Received unknown calibration key.  Expected "
                "'brightness_temperature', 'reflectance' or 'radiance', got "
                + key.calibration + ".")

        return data

    def calibrate_counts_to_physical_quantity(self, data, key):
        """Calibrate counts to radiances, brightness temperatures, or reflectances."""
        # counts to radiance scaling

        data = self.calibrate_counts_to_rad(data, key)

        if key.calibration == 'brightness_temperature':
            data = self.calibrate_rad_to_bt(data, key)
        elif key.calibration == 'reflectance':
            data = self.calibrate_rad_to_refl(data, key)

        return data

    def calibrate_counts_to_rad(self, data, key):
        """Calibrate counts to radiances."""
        radiance_units = data.attrs["units"]
        if key.name == 'ir_38':
            data = xr.where(((2 ** 12 - 1 < data) & (data <= 2 ** 13 - 1)),
                            (data * data.attrs.get("warm_scale_factor", 1) +
                             data.attrs.get("warm_add_offset", 0)),
                            (data * data.attrs.get("scale_factor", 1) +
                             data.attrs.get("add_offset", 0))
                            )
        else:
            data = (data * data.attrs.get("scale_factor", 1) +
                    data.attrs.get("add_offset", 0))

        data.attrs["units"] = radiance_units

        return data

    def calibrate_rad_to_bt(self, radiance, key):
        """IR channel calibration."""
        measured = self.get_channel_measured_group_path(key.name)

        # using the method from RADTOBR and PUG
        vc = self[measured + "/radiance_to_bt_conversion_coefficient_wavenumber"]

        a = self[measured + "/radiance_to_bt_conversion_coefficient_a"]
        b = self[measured + "/radiance_to_bt_conversion_coefficient_b"]

        c1 = self[measured + "/radiance_to_bt_conversion_constant_c1"]
        c2 = self[measured + "/radiance_to_bt_conversion_constant_c2"]

        for v in (vc, a, b, c1, c2):
            if v == v.attrs.get("FillValue",
                                default_fillvals.get(v.dtype.str[1:])):
                logger.error(
                    "{:s} set to fill value, cannot produce "
                    "brightness temperatures for {:s}.".format(
                        v.attrs.get("long_name",
                                    "at least one necessary coefficient"),
                        measured))
                return radiance*np.nan

        nom = c2 * vc
        denom = a * np.log(1 + (c1 * vc**3) / radiance)

        res = nom / denom - b / a
        res.attrs["units"] = "K"
        return res

    def calibrate_rad_to_refl(self, radiance, key):
        """VIS channel calibration."""
        measured = self.get_channel_measured_group_path(key.name)

        cesi = self[measured + "/channel_effective_solar_irradiance"]

        if cesi == cesi.attrs.get(
                "FillValue", default_fillvals.get(cesi.dtype.str[1:])):
            logger.error(
                "channel effective solar irradiance set to fill value, "
                "cannot produce reflectance for {:s}.".format(measured))
            return radiance*np.nan

        sun_earth_distance = np.mean(self["state/celestial/earth_sun_distance"]) / 149597870.7  # [AU]

        res = 100 * radiance * np.pi * sun_earth_distance**2 / cesi
        res.attrs["units"] = "%"
        return res
