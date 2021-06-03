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
"""Interface to MTG-FCI L1c NetCDF files.

This module defines the :class:`FCIL1cNCFileHandler` file handler, to
be used for reading Meteosat Third Generation (MTG) Flexible Combined
Imager (FCI) Level-1c data.  FCI will fly
on the MTG Imager (MTG-I) series of satellites, scheduled to be launched
in 2022 by the earliest.  For more information about FCI, see `EUMETSAT`_.

For simulated test data to be used with this reader, see `test data release`_.
For the Product User Guide (PUG) of the FCI L1c data, see `PUG`_.

.. note::
    This reader currently supports Full Disk High Spectral Resolution Imagery
    (FDHSI) files. Support for High Spatial Resolution Fast Imagery (HRFI) files
    will be implemented when corresponding test datasets will be available.

Geolocation is based on information from the data files.  It uses:

    * From the shape of the data variable ``data/<channel>/measured/effective_radiance``,
      start and end line columns of current swath.
    * From the data variable ``data/<channel>/measured/x``, the x-coordinates
      for the grid, in radians (azimuth angle positive towards West).
    * From the data variable ``data/<channel>/measured/y``, the y-coordinates
      for the grid, in radians (elevation angle positive towards North).
    * From the attribute ``semi_major_axis`` on the data variable
      ``data/mtg_geos_projection``, the Earth equatorial radius
    * From the attribute ``inverse_flattening`` on the same data variable, the
      (inverse) flattening of the ellipsoid
    * From the attribute ``perspective_point_height`` on the same data
      variable, the geostationary altitude in the normalised geostationary
      projection
    * From the attribute ``longitude_of_projection_origin`` on the same
      data variable, the longitude of the projection origin
    * From the attribute ``sweep_angle_axis`` on the same, the sweep angle
      axis, see https://proj.org/operations/projections/geos.html

From the pixel centre angles in radians and the geostationary altitude, the
extremities of the lower left and upper right corners are calculated in units
of arc length in m.  This extent along with the number of columns and rows, the
sweep angle axis, and a dictionary with equatorial radius, polar radius,
geostationary altitude, and longitude of projection origin, are passed on to
``pyresample.geometry.AreaDefinition``, which then uses proj4 for the actual
geolocation calculations.


The reading routine supports channel data in counts, radiances, and (depending
on channel) brightness temperatures or reflectances. The brightness temperature and reflectance calculation is based on the formulas indicated in
`PUG`_.

For each channel, it also supports a number of auxiliary datasets, such as the pixel quality,
the index map and the related geometric and acquisition parameters: time,
subsatellite latitude, subsatellite longitude, platform altitude, subsolar latitude, subsolar longitude,
earth-sun distance, sun-satellite distance,  swath number, and swath direction.

All auxiliary data can be obtained by prepending the channel name such as
``"vis_04_pixel_quality"``.

.. warning::
    The API for the direct reading of pixel quality is temporary and likely to
    change.  Currently, for each channel, the pixel quality is available by
    ``<chan>_pixel_quality``.  In the future, they will likely all be called
    ``pixel_quality`` and disambiguated by a to-be-decided property in the
    `DataID`.

.. _PUG: https://www-cdn.eumetsat.int/files/2020-07/pdf_mtg_fci_l1_pug.pdf
.. _EUMETSAT: https://www.eumetsat.int/mtg-flexible-combined-imager  # noqa: E501
.. _test data release: https://www.eumetsat.int/simulated-mtg-fci-l1c-enhanced-non-nominal-datasets
"""

from __future__ import (division, absolute_import, print_function,
                        unicode_literals)

import logging
import numpy as np
import xarray as xr

from pyresample import geometry
from netCDF4 import default_fillvals
from satpy.readers._geos_area import get_geos_area_naming
from satpy.readers.eum_base import get_service_mode

from .netcdf_utils import NetCDF4FileHandler

logger = logging.getLogger(__name__)

# dict containing all available auxiliary data parameters to be read using the index map. Keys are the
# parameter name and values are the paths to the variable inside the netcdf
AUX_DATA = {
    'subsatellite_latitude': 'state/platform/subsatellite_latitude',
    'subsatellite_longitude': 'state/platform/subsatellite_longitude',
    'platform_altitude': 'state/platform/platform_altitude',
    'subsolar_latitude': 'state/celestial/subsolar_latitude',
    'subsolar_longitude': 'state/celestial/subsolar_longitude',
    'earth_sun_distance': 'state/celestial/earth_sun_distance',
    'sun_satellite_distance': 'state/celestial/sun_satellite_distance',
    'time': 'time',
    'swath_number': 'data/swath_number',
    'swath_direction': 'data/swath_direction',
}


def _get_aux_data_name_from_dsname(dsname):
    aux_data_name = [key for key in AUX_DATA.keys() if key in dsname]
    if len(aux_data_name) > 0:
        return aux_data_name[0]
    else:
        return None


def _get_channel_name_from_dsname(dsname):
    # FIXME: replace by .removesuffix after we drop support for Python < 3.9
    if dsname.endswith("_pixel_quality"):
        channel_name = dsname[:-len("_pixel_quality")]
    elif dsname.endswith("_index_map"):
        channel_name = dsname[:-len("_index_map")]
    elif _get_aux_data_name_from_dsname(dsname) is not None:
        channel_name = dsname[:-len(_get_aux_data_name_from_dsname(dsname)) - 1]
    else:
        channel_name = dsname

    return channel_name


class FCIL1cNCFileHandler(NetCDF4FileHandler):
    """Class implementing the MTG FCI L1c Filehandler.

    This class implements the Meteosat Third Generation (MTG) Flexible
    Combined Imager (FCI) Level-1c NetCDF reader.
    It is designed to be used through the :class:`~satpy.Scene`
    class using the :mod:`~satpy.Scene.load` method with the reader
    ``"fci_l1c_nc"``.

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
        super().__init__(filename, filename_info,
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
        logger.debug('Reading {} from {}'.format(key['name'], self.filename))
        if "pixel_quality" in key['name']:
            return self._get_dataset_quality(key['name'])
        elif "index_map" in key['name']:
            return self._get_dataset_index_map(key['name'])
        elif _get_aux_data_name_from_dsname(key['name']) is not None:
            return self._get_dataset_aux_data(key['name'])
        elif any(lb in key['name'] for lb in {"vis_", "ir_", "nir_", "wv_"}):
            return self._get_dataset_measurand(key, info=info)
        else:
            raise ValueError("Unknown dataset key, not a channel, quality or auxiliary data: "
                             f"{key['name']:s}")

    def _get_dataset_measurand(self, key, info=None):
        """Load dataset corresponding to channel measurement.

        Load a dataset when the key refers to a measurand, whether uncalibrated
        (counts) or calibrated in terms of brightness temperature, radiance, or
        reflectance.
        """
        # Get the dataset
        # Get metadata for given dataset
        measured = self.get_channel_measured_group_path(key['name'])
        data = self[measured + "/effective_radiance"]

        attrs = data.attrs.copy()
        info = info.copy()

        fv = attrs.pop(
            "FillValue",
            default_fillvals.get(data.dtype.str[1:], np.nan))
        vr = attrs.get("valid_range", [-np.inf, np.inf])
        if key['calibration'] == "counts":
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
        # it: in the FCI format, each channel group (data/<channel>/measured) has
        # its own data variable 'pixel_quality'.
        # Until we can have multiple pixel_quality variables defined (for
        # example, with https://github.com/pytroll/satpy/pull/1088), rewrite
        # the ancillary variable to include the channel. See also
        # https://github.com/pytroll/satpy/issues/1171.
        if "pixel_quality" in attrs["ancillary_variables"]:
            attrs["ancillary_variables"] = attrs["ancillary_variables"].replace(
                "pixel_quality", key['name'] + "_pixel_quality")
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
        if key['calibration'] in ['brightness_temperature', 'reflectance']:
            res.attrs.pop("add_offset")
            res.attrs.pop("warm_add_offset")
            res.attrs.pop("scale_factor")
            res.attrs.pop("warm_scale_factor")

        # remove attributes from original file which don't apply anymore
        res.attrs.pop('long_name')

        return res

    def _get_dataset_quality(self, dsname):
        """Load a quality field for an FCI channel."""
        grp_path = self.get_channel_measured_group_path(_get_channel_name_from_dsname(dsname))
        dv_path = grp_path + "/pixel_quality"
        data = self[dv_path]
        return data

    def _get_dataset_index_map(self, dsname):
        """Load the index map for an FCI channel."""
        grp_path = self.get_channel_measured_group_path(_get_channel_name_from_dsname(dsname))
        dv_path = grp_path + "/index_map"
        data = self[dv_path]

        data = data.where(data != data.attrs.get('_FillValue', 65535))
        return data

    def _get_aux_data_lut_vector(self, aux_data_name):
        """Load the lut vector of an auxiliary variable."""
        lut = self[AUX_DATA[aux_data_name]]

        fv = default_fillvals.get(lut.dtype.str[1:], np.nan)
        lut = lut.where(lut != fv)

        return lut

    @staticmethod
    def _getitem(block, lut):
        return lut[block.astype('uint16')]

    def _get_dataset_aux_data(self, dsname):
        """Get the auxiliary data arrays using the index map."""
        # get index map
        index_map = self._get_dataset_index_map(_get_channel_name_from_dsname(dsname))
        # index map indexing starts from 1
        index_map -= 1

        # get lut values from 1-d vector
        lut = self._get_aux_data_lut_vector(_get_aux_data_name_from_dsname(dsname))

        # assign lut values based on index map indices
        aux = index_map.data.map_blocks(self._getitem, lut.data, dtype=lut.data.dtype)
        aux = xr.DataArray(aux, dims=index_map.dims, attrs=index_map.attrs, coords=index_map.coords)

        # filter out out-of-disk values
        aux = aux.where(index_map >= 0)

        return aux

    @staticmethod
    def get_channel_measured_group_path(channel):
        """Get the channel's measured group path."""
        measured_group_path = 'data/{}/measured'.format(channel)

        return measured_group_path

    def calc_area_extent(self, key):
        """Calculate area extent for a dataset."""
        # if a user requests a pixel quality or index map before the channel data, the
        # yaml-reader will ask the area extent of the pixel quality/index map field,
        # which will ultimately end up here
        channel_name = _get_channel_name_from_dsname(key['name'])
        # Get metadata for given dataset
        measured = self.get_channel_measured_group_path(channel_name)
        # Get start/end line and column of loaded swath.
        nlines, ncols = self[measured + "/effective_radiance/shape"]

        logger.debug('Channel {} resolution: {}'.format(channel_name, ncols))
        logger.debug('Row/Cols: {} / {}'.format(nlines, ncols))

        # Calculate full globe line extent
        h = float(self["data/mtg_geos_projection/attr/perspective_point_height"])

        extents = {}
        for coord in "xy":
            coord_radian = self["data/{:s}/measured/{:s}".format(channel_name, coord)]
            coord_radian_num = coord_radian[:] * coord_radian.scale_factor + coord_radian.add_offset

            # FCI defines pixels by centroids (see PUG), while pyresample
            # defines corners as lower left corner of lower left pixel, upper right corner of upper right pixel
            # (see https://pyresample.readthedocs.io/en/latest/geo_def.html).
            # Therefore, half a pixel (i.e. half scale factor) needs to be added in each direction.

            # The grid origin is in the South-West corner.
            # Note that the azimuth angle (x) is defined as positive towards West (see PUG - Level 1c Reference Grid)
            # The elevation angle (y) is defined as positive towards North as per usual convention. Therefore:
            # The values of x go from positive (West) to negative (East) and the scale factor of x is negative.
            # The values of y go from negative (South) to positive (North) and the scale factor of y is positive.

            # South-West corner (x positive, y negative)
            first_coord_radian = coord_radian_num[0] - coord_radian.scale_factor / 2
            # North-East corner (x negative, y positive)
            last_coord_radian = coord_radian_num[-1] + coord_radian.scale_factor / 2

            # convert to arc length in m
            first_coord = first_coord_radian * h  # arc length in m
            last_coord = last_coord_radian * h

            # the .item() call is needed with the h5netcdf backend, see
            # https://github.com/pytroll/satpy/issues/972#issuecomment-558191583
            # but we need to compute it first if this is dask
            try:
                first_coord = first_coord.compute()
                last_coord = last_coord.compute()
            except AttributeError:  # not a dask.array
                pass

            extents[coord] = (first_coord.item(), last_coord.item())

        # For the final extents, take into account that the image is upside down (lower line is North), and that
        # East is defined as positive azimuth in Proj, so we need to multiply by -1 the azimuth extents.

        # lower left x: west-ward extent: first coord of x, multiplied by -1 to account for azimuth orientation
        # lower left y: north-ward extent: last coord of y
        # upper right x: east-ward extent: last coord of x, multiplied by -1 to account for azimuth orientation
        # upper right y: south-ward extent: first coord of y
        area_extent = (-extents["x"][0], extents["y"][1], -extents["x"][1], extents["y"][0])

        return area_extent, nlines, ncols

    def get_area_def(self, key):
        """Calculate on-fly area definition for a dataset in geos-projection."""
        # assumption: channels with same resolution should have same area
        # cache results to improve performance
        if key['resolution'] in self._cache:
            return self._cache[key['resolution']]

        a = float(self["data/mtg_geos_projection/attr/semi_major_axis"])
        h = float(self["data/mtg_geos_projection/attr/perspective_point_height"])
        rf = float(self["data/mtg_geos_projection/attr/inverse_flattening"])
        lon_0 = float(self["data/mtg_geos_projection/attr/longitude_of_projection_origin"])
        sweep = str(self["data/mtg_geos_projection"].sweep_angle_axis)

        area_extent, nlines, ncols = self.calc_area_extent(key)
        logger.debug('Calculated area extent: {}'
                     .format(''.join(str(area_extent))))

        # use a (semi-major axis) and rf (reverse flattening) to define ellipsoid as recommended by EUM (see PUG)
        proj_dict = {'a': a,
                     'lon_0': lon_0,
                     'h': h,
                     "rf": rf,
                     'proj': 'geos',
                     'units': 'm',
                     "sweep": sweep}

        area_naming_input_dict = {'platform_name': 'mtg',
                                  'instrument_name': 'fci',
                                  'resolution': int(key['resolution'])
                                  }
        area_naming = get_geos_area_naming({**area_naming_input_dict,
                                            **get_service_mode('fci', lon_0)})

        area = geometry.AreaDefinition(
            area_naming['area_id'],
            area_naming['description'],
            "",
            proj_dict,
            ncols,
            nlines,
            area_extent)

        self._cache[key['resolution']] = area
        return area

    def calibrate(self, data, key):
        """Calibrate data."""
        if key['calibration'] == "counts":
            # from package description, this just means not applying add_offset
            # and scale_factor
            data.attrs["units"] = "1"
        elif key['calibration'] in ['brightness_temperature', 'reflectance', 'radiance']:
            data = self.calibrate_counts_to_physical_quantity(data, key)
        else:
            logger.error(
                "Received unknown calibration key.  Expected "
                "'brightness_temperature', 'reflectance' or 'radiance', got "
                + key['calibration'] + ".")

        return data

    def calibrate_counts_to_physical_quantity(self, data, key):
        """Calibrate counts to radiances, brightness temperatures, or reflectances."""
        # counts to radiance scaling

        data = self.calibrate_counts_to_rad(data, key)

        if key['calibration'] == 'brightness_temperature':
            data = self.calibrate_rad_to_bt(data, key)
        elif key['calibration'] == 'reflectance':
            data = self.calibrate_rad_to_refl(data, key)

        return data

    def calibrate_counts_to_rad(self, data, key):
        """Calibrate counts to radiances."""
        radiance_units = data.attrs["units"]
        if key['name'] == 'ir_38':
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
        # using the method from PUG section Converting from Effective Radiance to Brightness Temperature for IR Channels

        measured = self.get_channel_measured_group_path(key['name'])

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
                return radiance * np.nan

        nom = c2 * vc
        denom = a * np.log(1 + (c1 * vc ** 3) / radiance)

        res = nom / denom - b / a
        res.attrs["units"] = "K"
        return res

    def calibrate_rad_to_refl(self, radiance, key):
        """VIS channel calibration."""
        measured = self.get_channel_measured_group_path(key['name'])

        cesi = self[measured + "/channel_effective_solar_irradiance"]

        if cesi == cesi.attrs.get(
                "FillValue", default_fillvals.get(cesi.dtype.str[1:])):
            logger.error(
                "channel effective solar irradiance set to fill value, "
                "cannot produce reflectance for {:s}.".format(measured))
            return radiance * np.nan

        sun_earth_distance = np.mean(self["state/celestial/earth_sun_distance"]) / 149597870.7  # [AU]

        res = 100 * radiance * np.pi * sun_earth_distance ** 2 / cesi
        res.attrs["units"] = "%"
        return res
