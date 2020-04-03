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
      projection (see PUG §5.2)
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

.. _PUG: http://www.eumetsat.int/website/wcm/idc/idcplg?IdcService=GET_FILE&dDocName=PDF_DMT_719113&RevisionSelectionMethod=LatestReleased&Rendition=Web
.. _EUMETSAT: https://www.eumetsat.int/website/home/Satellites/FutureSatellites/MeteosatThirdGeneration/MTGDesign/index.html#fci  # noqa: E501
"""

from __future__ import (division, absolute_import, print_function,
                        unicode_literals)

import logging
import numpy as np
import dask.array as da
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
        # Get the dataset
        # Get metadata for given dataset
        measured, root = self.get_channel_dataset(key.name)
        radlab = measured + "/effective_radiance"
        data = self[radlab]

        attrs = data.attrs.copy()
        info = info.copy()
        fv = attrs.pop(
                "FillValue",
                default_fillvals.get(data.dtype.str[1:], np.nan))
        vr = attrs.pop("valid_range", [-np.inf, np.inf])
        if key.calibration == "counts":
            attrs["_FillValue"] = fv
            nfv = fv
        else:
            nfv = np.nan
        data = data.where(data >= vr[0], nfv)
        data = data.where(data <= vr[1], nfv)
        if key.calibration == "counts":
            # from package description, this just means not applying add_offset
            # and scale_factor
            attrs.pop("scale_factor")
            attrs.pop("add_offset")
            data.attrs["units"] = "1"
            res = data
        else:
            data = (data * attrs.pop("scale_factor", 1) +
                    attrs.pop("add_offset", 0))

            if key.calibration in ("brightness_temperature", "reflectance"):
                res = self.calibrate(data, key, measured, root)
            else:
                res = data
                data.attrs["units"] = attrs["units"]
        # pre-calibration units no longer apply
        info.pop("units")
        attrs.pop("units")

        res.attrs.update(key.to_dict())
        res.attrs.update(info)
        res.attrs.update(attrs)
        return res

    def get_channel_dataset(self, channel):
        """Get channel dataset."""
        root_group = 'data/{}'.format(channel)
        group = 'data/{}/measured'.format(channel)

        return group, root_group

    def calc_area_extent(self, key):
        """Calculate area extent for a dataset."""
        # Get metadata for given dataset
        measured, root = self.get_channel_dataset(key.name)
        # Get start/end line and column of loaded swath.
        nlines, ncols = self[measured + "/effective_radiance/shape"]

        logger.debug('Channel {} resolution: {}'.format(key.name, ncols))
        logger.debug('Row/Cols: {} / {}'.format(nlines, ncols))

        # Calculate full globe line extent
        h = float(self["data/mtg_geos_projection/attr/perspective_point_height"])

        ext = {}
        for c in "xy":
            c_radian = self["data/{:s}/measured/{:s}".format(key.name, c)]
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
        return (area_extent, nlines, ncols)

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
        (area_extent, nlines, ncols) = self.calc_area_extent(key)
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

    def calibrate(self, data, key, measured, root):
        """Calibrate data."""
        if key.calibration == 'brightness_temperature':
            data = self._ir_calibrate(data, measured, root)
        elif key.calibration == 'reflectance':
            data = self._vis_calibrate(data, measured)
        else:
            raise ValueError(
                    "Received unknown calibration key.  Expected "
                    "'brightness_temperature' or 'reflectance', got "
                    + key.calibration)

        return data

    def _ir_calibrate(self, radiance, measured, root):
        """IR channel calibration."""
        coef = self[measured + "/radiance_unit_conversion_coefficient"]
        wl_c = self[root + "/central_wavelength_actual"]

        a = self[measured + "/radiance_to_bt_conversion_coefficient_a"]
        b = self[measured + "/radiance_to_bt_conversion_coefficient_b"]

        c1 = self[measured + "/radiance_to_bt_conversion_constant_c1"]
        c2 = self[measured + "/radiance_to_bt_conversion_constant_c2"]

        for v in (coef, wl_c, a, b, c1, c2):
            if v == v.attrs.get("FillValue",
                                default_fillvals.get(v.dtype.str[1:])):
                logger.error(
                    "{:s} set to fill value, cannot produce "
                    "brightness temperatures for {:s}.".format(
                        v.attrs.get("long_name",
                                    "at least one necessary coefficient"),
                        root))
                return xr.DataArray(
                        da.full(shape=radiance.shape,
                                chunks=radiance.chunks,
                                fill_value=np.nan),
                        dims=radiance.dims,
                        coords=radiance.coords,
                        attrs=radiance.attrs)

        Lv = radiance * coef
        vc = 1e6/wl_c  # from wl in um to wn in m^-1
        nom = c2 * vc
        denom = a * np.log(1 + (c1 * vc**3) / Lv)

        res = nom / denom - b / a
        res.attrs["units"] = "K"
        return res

    def _vis_calibrate(self, radiance, measured):
        """VIS channel calibration."""
        # radiance to reflectance taken as in mipp/xrit/MSG.py
        # again FCI User Guide is not clear on how to do this

        cesilab = measured + "/channel_effective_solar_irradiance"
        cesi = self[cesilab]
        if cesi == cesi.attrs.get(
                "FillValue", default_fillvals.get(cesi.dtype.str[1:])):
            logger.error(
                "channel effective solar irradiance set to fill value, "
                "cannot produce reflectance for {:s}.".format(measured))
            return xr.DataArray(
                    da.full(shape=radiance.shape,
                            chunks=radiance.chunks,
                            fill_value=np.nan),
                    dims=radiance.dims,
                    coords=radiance.coords,
                    attrs=radiance.attrs)

        sirr = float(cesi)
        res = radiance / sirr * 100
        res.attrs["units"] = "%"
        return res
