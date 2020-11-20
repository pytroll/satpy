#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Satpy developers
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
"""FIDUCEO MVIRI FCDR Reader.

Introduction
------------
The FIDUCEO MVIRI FCDR is a Fundamental Climate Data Record (FCDR) of
re-calibrated Level 1.5 Infrared, Water Vapour, and Visible radiances from
the Meteosat Visible Infra-Red Imager (MVIRI) instrument onboard the
Meteosat First Generation satellites. There are two variants of the dataset:
The *full FCDR* and a simplified version called *easy FCDR*. Some datasets are
only available in one of the two variants, see the corresponding YAML
definition in ``satpy/etc/readers/``.

Dataset Names
-------------
The FIDUCEO MVIRI readers use names ``VIS``, ``WV`` and ``IR`` for the visible,
water vapor and infrared channels, respectively. These are different from
the original netCDF variable names for the following reasons:

- VIS channel is named differently in full FCDR (``counts_vis``) and easy FCDR
  (``toa_bidirectional_reflectance_vis``)
- netCDF variable names contain the calibration level (e.g. ``counts_...``),
  which might be confusing for satpy users if a different calibration level
  is chosen.

Remaining datasets (such as quality flags and uncertainties) have the same
name in the reader as in the netCDF file.


Example
-------
This is how to read FIDUCEO MVIRI FCDR data in satpy:

.. code-block:: python

    from satpy import Scene

    scn = Scene(filenames=['FIDUCEO_FCDR_L15_MVIRI_MET7-57.0...'],
                reader='mviri_l1b_fiduceo_nc',
                reader_kwargs={'mask_bad_quality': True)
    scn.load(['VIS', 'WV', 'IR'])

In the above example pixels considered bad quality are masked, see
:class:`FiduceoMviriBase` for a keyword argument description. Global
netCDF attributes are available in the ``raw_metadata`` attribute of
each loaded dataset.


Image Orientation
-----------------
The images are stored in MVIRI scanning direction, that means South is up and
East is right. This can be changed as follows:

.. code-block:: python

    scn.load(['VIS'], upper_right_corner='NE')


Geolocation
-----------
In addition to the image data, FIDUCEO also provides so called *static FCDRs*
containing latitude and longitude coordinates. In order to simplify their
usage, the FIDUCEO MVIRI readers do not make use of these static files, but
instead provide an area definition that can be used to compute longitude and
latitude coordinates on demand.

.. code-block:: python

    area = scn['VIS'].attrs['area']
    lons, lats = area.get_lonlats()

Those were compared to the static FCDR and they agree very well, however there
are small differences. The mean difference is < 1E3 degrees for all channels
and projection longitudes.


Huge VIS Reflectances
---------------------
You might encounter huge VIS reflectances (10^8 percent and greater) in
situations where both radiance and solar zenith angle are small. The reader
certainly needs some improvement in this regard. Maybe the corresponding
uncertainties can be used to filter these cases before calculating reflectances.


References
----------
    - `[Handbook]`_ MFG User Handbook
    - `[PUG]`_ FIDUCEO MVIRI FCDR Product User Guide

.. _[Handbook]: http://www.eumetsat.int/\
website/wcm/idc/idcplg?IdcService=GET_FILE&dDocName=PDF_TD06_MARF&\
RevisionSelectionMethod=LatestReleased&Rendition=Web
.. _[PUG]: http://doi.org/10.15770/EUM_SEC_CLM_0009
"""

import abc
import functools

import dask.array as da
import numpy as np
import xarray as xr

from satpy import CHUNK_SIZE
from satpy.dataset.dataid import DataQuery
from satpy.readers._geos_area import (ang2fac, get_area_definition,
                                      get_area_extent)
from satpy.readers.file_handlers import BaseFileHandler

EQUATOR_RADIUS = 6378140.0
POLE_RADIUS = 6356755.0
ALTITUDE = 42164000.0 - EQUATOR_RADIUS
"""[Handbook] section 5.2.1."""

MVIRI_FIELD_OF_VIEW = 18.0
"""[Handbook] section 5.3.2.1."""

CHANNELS = ['VIS', 'WV', 'IR']
ANGLES = [
    'solar_zenith_angle_vis',
    'solar_azimuth_angle_vis',
    'satellite_zenith_angle_vis',
    'satellite_azimuth_angle_vis',
    'solar_zenith_angle_ir_wv',
    'solar_azimuth_angle_ir_wv',
    'satellite_zenith_angle_ir_wv',
    'satellite_azimuth_angle_ir_wv'
]
OTHER_REFLECTANCES = [
    'u_independent_toa_bidirectional_reflectance',
    'u_structured_toa_bidirectional_reflectance'
]
HIGH_RESOL = 2250


class InterpCache:
    """Interpolation cache."""

    def __init__(self, func, keys, hash_funcs):
        """Create the cache.

        Args:
            func:
                Interpolation function to be cached.
            keys:
                Function arguments serving as cache key.
            hash_funcs (dict):
                For each key provides a method that extracts a hashable
                value from the given keyword argument. For example, use
                image shape to maintain separate caches for low
                resolution (WV/IR) and high resolution (VIS) channels.
        """
        self.func = func
        self.keys = keys
        self.hash_funcs = hash_funcs
        self.cache = {}

    def __call__(self, *args, **kwargs):
        """Call the interpolation function."""
        key = tuple(
            [self.hash_funcs[key](kwargs[key]) for key in self.keys]
        )
        if key not in self.cache:
            self.cache[key] = self.func(*args, **kwargs)
        return self.cache[key]

    def __get__(self, obj, objtype):
        """To support instance methods."""
        return functools.partial(self.__call__, obj)


def interp_cache(keys, hash_funcs):
    """Interpolation cache."""
    def wrapper(func):
        return InterpCache(func, keys, hash_funcs)
    return wrapper


class FiduceoMviriBase(BaseFileHandler):
    """Baseclass for FIDUCEO MVIRI file handlers."""
    nc_keys = {
        'WV': 'count_wv',
        'IR': 'count_ir',
        'solar_zenith_angle_vis': 'solar_zenith_angle',
        'solar_azimuth_angle_vis': 'solar_azimuth_angle',
        'satellite_zenith_angle_vis': 'satellite_zenith_angle',
        'satellite_azimuth_angle_vis': 'satellite_azimuth_angle',
        'solar_zenith_angle_ir_wv': 'solar_zenith_angle',
        'solar_azimuth_angle_ir_wv': 'solar_azimuth_angle',
        'satellite_zenith_angle_ir_wv': 'satellite_zenith_angle',
        'satellite_azimuth_angle_ir_wv': 'satellite_azimuth_angle',
    }
    nc_keys_coefs = {
        'WV': {
            'radiance': {
                'a': 'a_wv',
                'b': 'b_wv'
            },
            'brightness_temperature': {
                'a': 'bt_a_wv',
                'b': 'bt_b_wv'
            }
        },
        'IR': {
            'radiance': {
                'a': 'a_ir',
                'b': 'b_ir'
            },
            'brightness_temperature': {
                'a': 'bt_a_ir',
                'b': 'bt_b_ir'
            }
        },
    }

    def __init__(self, filename, filename_info, filetype_info,
                 mask_bad_quality=False):
        """Initialize the file handler.

        Args:
             mask_bad_quality: Mask VIS pixels with bad quality, that means
                 everything else than "ok" or "use with caution". If you need
                 more control, use the ``quality_pixel_bitmask`` and
                 ``data_quality_bitmask`` datasets.
        """
        super(FiduceoMviriBase, self).__init__(
            filename, filename_info, filetype_info)
        self.mask_bad_quality = mask_bad_quality
        self.nc = xr.open_dataset(
            filename,
            chunks={'x': CHUNK_SIZE,
                    'y': CHUNK_SIZE,
                    'x_ir_wv': CHUNK_SIZE,
                    'y_ir_wv': CHUNK_SIZE}
        )

        # Projection longitude is not provided in the file, read it from the
        # filename.
        self.projection_longitude = float(filename_info['projection_longitude'])

    def get_dataset(self, dataset_id, dataset_info):
        """Get the dataset."""
        name = dataset_id['name']
        ds = self._read_dataset(name)
        if dataset_id['name'] in CHANNELS:
            ds = self.calibrate(ds, channel=name,
                                calibration=dataset_id['calibration'])
            if self.mask_bad_quality and name == 'VIS':
                ds = self._mask_vis(ds)
            ds['acq_time'] = ('y', self._get_acq_time(ds))
        elif dataset_id['name'] in OTHER_REFLECTANCES:
            ds = ds * 100  # conversion to percent
        elif dataset_id['name'] in ANGLES:
            ds = self._interp_angles(ds, dataset_id)
        self._update_attrs(ds, dataset_info)
        return ds

    def _read_dataset(self, name):
        """Read a dataset from the file."""
        nc_key = self.nc_keys.get(name, name)
        ds = self.nc[nc_key]
        if 'y_ir_wv' in ds.dims:
            ds = ds.rename({'y_ir_wv': 'y', 'x_ir_wv': 'x'})
        if 'y_tie' in ds.dims:
            ds = ds.rename({'y_tie': 'y', 'x_tie': 'x'})
        return ds

    def _update_attrs(self, ds, info):
        """Update dataset attributes."""
        ds.attrs.update(info)
        ds.attrs.update({'platform': self.filename_info['platform'],
                         'sensor': self.filename_info['sensor']})
        ds.attrs['raw_metadata'] = self.nc.attrs
        ds.attrs['orbital_parameters'] = self._get_orbital_parameters()

    def get_area_def(self, dataset_id):
        """Get area definition of the given dataset."""
        ds = self._read_dataset(dataset_id['name'])
        im_size = ds.coords['y'].size

        # Determine line/column offsets and scaling factors. For offsets
        # see variables "asamp" and "aline" of subroutine "refgeo" in
        # [Handbook] and in
        # https://github.com/FIDUCEO/FCDR_MVIRI/blob/master/lib/nrCrunch/cruncher.f
        loff = coff = im_size / 2 + 0.5
        lfac = cfac = ang2fac(np.deg2rad(MVIRI_FIELD_OF_VIEW) / im_size)

        area_name = 'geos_mviri_{}'.format(
            'vis' if self._is_high_resol(dataset_id) else 'ir_wv'
        )
        pdict = {
            'ssp_lon': self.projection_longitude,
            'a': EQUATOR_RADIUS,
            'b': POLE_RADIUS,
            'h': ALTITUDE,
            'units': 'm',
            'loff': loff - im_size,
            'coff': coff,
            'lfac': -lfac,
            'cfac': -cfac,
            'nlines': im_size,
            'ncols': im_size,
            'scandir': 'S2N',  # Reference: [PUG] section 2.
            'p_id': area_name,
            'a_name': area_name,
            'a_desc': 'MVIRI Geostationary Projection'
        }
        extent = get_area_extent(pdict)
        area_def = get_area_definition(pdict, extent)
        return area_def

    def calibrate(self, ds, channel, calibration):
        """Calibrate the given dataset."""
        if channel == 'VIS':
            return self._calibrate_vis(ds, calibration)
        elif channel in ['WV', 'IR']:
            return self._calibrate_ir_wv(ds, channel, calibration)
        else:
            raise KeyError('Don\'t know how to calibrate channel {}'.format(
                channel))

    @abc.abstractmethod
    def _calibrate_vis(self, ds, calibration):
        """Calibrate VIS channel. To be implemented by subclasses."""
        raise NotImplementedError

    def _update_refl_attrs(self, refl):
        """Update attributes of reflectance datasets."""
        refl.attrs['sun_earth_distance_correction_applied'] = True
        refl.attrs['sun_earth_distance_correction_factor'] = self.nc[
            'distance_sun_earth'].item()
        return refl

    def _calibrate_ir_wv(self, ds, channel, calibration):
        """Calibrate IR and WV channel."""
        if calibration == 'counts':
            return ds
        elif calibration in ('radiance', 'brightness_temperature'):
            rad = self._ir_wv_counts_to_radiance(ds, channel)
            if calibration == 'radiance':
                return rad
            bt = self._ir_wv_radiance_to_brightness_temperature(rad, channel)
            return bt
        else:
            raise KeyError('Invalid calibration: {}'.format(calibration.name))

    def _get_coefs_ir_wv(self, channel, calibration):
        """Get calibration coefficients for IR/WV channels.

        Returns:
            Offset (a), Slope (b)
        """
        nc_key_a = self.nc_keys_coefs[channel][calibration]['a']
        nc_key_b = self.nc_keys_coefs[channel][calibration]['b']
        a = np.float32(self.nc[nc_key_a])
        b = np.float32(self.nc[nc_key_b])
        return a, b

    def _ir_wv_counts_to_radiance(self, counts, channel):
        """Convert IR/WV counts to radiance.

        Reference: [PUG], equations (4.1) and (4.2).
        """
        a, b = self._get_coefs_ir_wv(channel, 'radiance')
        rad = a + b * counts
        return rad.where(rad > 0, np.float32(np.nan))

    def _ir_wv_radiance_to_brightness_temperature(self, rad, channel):
        """Convert IR/WV radiance to brightness temperature.

        Reference: [PUG], equations (5.1) and (5.2).
        """
        a, b = self._get_coefs_ir_wv(channel, 'brightness_temperature')
        bt = b / (np.log(rad) - a)
        return bt.where(bt > 0, np.float32(np.nan))

    def _mask_vis(self, ds):
        """Mask VIS pixels with bad quality.

        Pixels are considered bad quality if the "quality_pixel_bitmask" is
        everything else than 0 (no flag set) or 2 ("use_with_caution" and no
        other flag set). According to [PUG] that bitmask is only applicable to
        the VIS channel.
        """
        mask = self.nc['quality_pixel_bitmask']
        return ds.where(np.logical_or(mask == 0, mask == 2),
                        np.float32(np.nan))

    def _get_acq_time(self, ds):
        """Get scanline acquisition time for the given dataset.

        Note that the acquisition time does not increase monotonically
        with the scanline number due to the scan pattern and rectification.
        """
        # Variable is sometimes named "time" and sometimes "time_ir_wv".
        try:
            time2d = self.nc['time_ir_wv']
        except KeyError:
            time2d = self.nc['time']
        return self._get_acq_time_cached(time2d, target_y=ds.coords['y'])

    @interp_cache(
        keys=('target_y',),
        hash_funcs={'target_y': lambda y: y.size}
    )
    def _get_acq_time_cached(self, time2d, target_y):
        """Get scanline acquisition time for the given image coordinates.

        The files provide timestamps per pixel for the low resolution
        channels (IR/WV) only.

        1) Average values in each line to obtain one timestamp per line.
        2) For the VIS channel duplicate values in y-direction (as
           advised by [PUG]).

        Note that the timestamps do not increase monotonically with the
        line number in some cases.

        Returns:
            Mean scanline acquisition timestamps
        """
        # Compute mean timestamp per scanline
        time = time2d.mean(dim='x_ir_wv').rename({'y_ir_wv': 'y'})

        # If required, repeat timestamps in y-direction to obtain higher
        # resolution
        y = time.coords['y'].values
        if y.size < target_y.size:
            reps = target_y.size // y.size
            y_rep = np.repeat(y, reps)
            time_hires = time.reindex(y=y_rep)
            time_hires = time_hires.assign_coords(y=target_y)
            return time_hires
        return time

    def _get_orbital_parameters(self):
        """Get the orbital parameters."""
        orbital_parameters = {
            'projection_longitude': self.projection_longitude,
            'projection_latitude': 0.0,
            'projection_altitude': ALTITUDE
        }
        ssp_lon, ssp_lat = self._get_ssp_lonlat()
        if not np.isnan(ssp_lon) and not np.isnan(ssp_lat):
            orbital_parameters.update({
                'satellite_actual_longitude': ssp_lon,
                'satellite_actual_latitude': ssp_lat,
                # altitude not available
            })
        return orbital_parameters

    def _get_ssp_lonlat(self):
        """Get longitude and latitude at the subsatellite point.

        Easy FCDR files provide satellite position at the beginning and
        end of the scan. This method computes the mean of those two values.
        In the full FCDR the information seems to be missing.

        Returns:
            Subsatellite longitude and latitude
        """
        ssp_lon = self._get_ssp('longitude')
        ssp_lat = self._get_ssp('latitude')
        return ssp_lon, ssp_lat

    def _get_ssp(self, coord):
        key_start = 'sub_satellite_{}_start'.format(coord)
        key_end = 'sub_satellite_{}_end'.format(coord)
        try:
            sub_lonlat = np.nanmean(
                [self.nc[key_start].values,
                 self.nc[key_end].values]
            )
        except KeyError:
            # Variables seem to be missing in full FCDR
            sub_lonlat = np.nan
        return sub_lonlat

    def _interp_angles(self, angles, dataset_id):
        """Get angle dataset.

        Files provide angles (solar/satellite zenith & azimuth) at a coarser
        resolution. Interpolate them to the desired resolution.
        """
        if self._is_high_resol(dataset_id):
            target_x = self.nc.coords['x']
            target_y = self.nc.coords['y']
        else:
            target_x = self.nc.coords['x_ir_wv']
            target_y = self.nc.coords['y_ir_wv']
        return self._interp_angles_cached(
            angles=angles,
            nc_key=self.nc_keys[dataset_id['name']],
            target_x=target_x,
            target_y=target_y
        )

    @interp_cache(
        keys=('nc_key', 'target_x', 'target_y'),
        hash_funcs={
            'nc_key': lambda nc_key: nc_key,
            'target_x': lambda x: x.size,
            'target_y': lambda y: y.size
        }
    )
    def _interp_angles_cached(self, angles, nc_key, target_x, target_y):
        """Interpolate angles to the given resolution."""
        return self._interp_tiepoints(angles,
                                      target_x,
                                      target_y)

    def _interp_tiepoints(self, ds, target_x, target_y):
        """Interpolate dataset between tiepoints.

        Uses linear interpolation.

        FUTURE: [PUG] recommends cubic spline interpolation.

        Args:
            ds:
                Dataset to be interpolated
            target_x:
                Target x coordinates
            target_y:
                Target y coordinates
        """
        # No tiepoint coordinates specified in the files. Use dimensions
        # to calculate tiepoint sampling and assign tiepoint coordinates
        # accordingly.
        sampling = target_x.size // ds.coords['x'].size
        ds = ds.assign_coords(x=target_x.values[::sampling],
                              y=target_y.values[::sampling])

        return ds.interp(x=target_x.values, y=target_y.values)

    def _is_high_resol(self, dataset_id):
        return dataset_id['resolution'] == HIGH_RESOL


class FiduceoMviriEasyFcdrFileHandler(FiduceoMviriBase):
    """File handler for FIDUCEO MVIRI Easy FCDR."""

    nc_keys = FiduceoMviriBase.nc_keys.copy()
    nc_keys['VIS'] = 'toa_bidirectional_reflectance_vis'

    def _calibrate_vis(self, ds, calibration):
        """Calibrate VIS channel.

        Easy FCDR provides reflectance only, no counts or radiance.
        """
        if calibration == 'reflectance':
            refl = 100 * ds  # conversion to percent
            refl = self._update_refl_attrs(refl)
            return refl
        elif calibration in ('counts', 'radiance'):
            raise ValueError('Cannot calibrate to {}. Easy FCDR provides '
                             'reflectance only.'.format(calibration.name))
        else:
            raise KeyError('Invalid calibration: {}'.format(calibration.name))


class FiduceoMviriFullFcdrFileHandler(FiduceoMviriBase):
    """File handler for FIDUCEO MVIRI Full FCDR."""

    nc_keys = FiduceoMviriBase.nc_keys.copy()
    nc_keys['VIS'] = 'count_vis'

    def _calibrate_vis(self, ds, calibration):
        """Calibrate VIS channel.

        All calibration levels are available here.
        """
        if calibration == 'counts':
            return ds
        elif calibration in ('radiance', 'reflectance'):
            rad = self._vis_counts_to_radiance(ds)
            if calibration == 'radiance':
                return rad
            refl = self._vis_radiance_to_reflectance(rad)
            refl = self._update_refl_attrs(refl)
            return refl
        else:
            raise KeyError('Invalid calibration: {}'.format(calibration.name))

    def _vis_counts_to_radiance(self, counts):
        """Convert VIS counts to radiance.

        Reference: [PUG], equations (7) and (8).
        """
        years_since_launch = self.nc['years_since_launch']
        a_cf = (self.nc['a0_vis'] +
                self.nc['a1_vis'] * years_since_launch +
                self.nc['a2_vis'] * years_since_launch ** 2)
        mean_count_space_vis = np.float32(self.nc['mean_count_space_vis'])
        a_cf = np.float32(a_cf)
        rad = (counts - mean_count_space_vis) * a_cf
        return rad.where(rad > 0, np.float32(np.nan))

    def _vis_radiance_to_reflectance(self, rad):
        """Convert VIS radiance to reflectance factor.

        Note: Produces huge reflectances in situations where both radiance and
        solar zenith angle are small. Maybe the corresponding uncertainties
        can be used to filter these cases before calculating reflectances.

        Reference: [PUG], equation (6).
        """
        sza = self.get_dataset(
            DataQuery(name='solar_zenith_angle_vis',
                      resolution=HIGH_RESOL),
            dataset_info={}
        )
        sza = sza.where(da.fabs(sza) < 90,
                        np.float32(np.nan))  # direct illumination only
        cos_sza = np.cos(np.deg2rad(sza))
        distance_sun_earth2 = np.float32(self.nc['distance_sun_earth'] ** 2)
        solar_irradiance_vis = np.float32(self.nc['solar_irradiance_vis'])
        refl = (
           (np.pi * distance_sun_earth2) /
           (solar_irradiance_vis * cos_sza) *
           rad
        )
        refl = refl * 100  # conversion to percent
        return refl
