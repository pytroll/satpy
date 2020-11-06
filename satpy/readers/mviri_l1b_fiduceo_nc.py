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


References:
    - `[Handbook]`_ MFG User Handbook
    - `[PUG]`_ FIDUCEO MVIRI FCDR Product User Guide

.. _[Handbook]: http://www.eumetsat.int/\
website/wcm/idc/idcplg?IdcService=GET_FILE&dDocName=PDF_TD06_MARF&\
RevisionSelectionMethod=LatestReleased&Rendition=Web
.. _[PUG]: http://doi.org/10.15770/EUM_SEC_CLM_0009
"""

import abc
import functools
import numpy as np
import xarray as xr


from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers._geos_area import (ang2fac, get_area_definition,
                                      get_area_extent)


EQUATOR_RADIUS = 6378140.0
POLE_RADIUS = 6356755.0
ALTITUDE = 42164000.0 - EQUATOR_RADIUS
"""[Handbook] section 5.2.1."""

MVIRI_FIELD_OF_VIEW = 18.0
"""[Handbook] section 5.3.2.1."""

CHANNELS = ['VIS', 'WV', 'IR']
OTHER_REFLECTANCES = [
    'u_independent_toa_bidirectional_reflectance',
    'u_structured_toa_bidirectional_reflectance'
]


class shape_cache:
    """Cache function call based on image shape.

    Shape can be used to maintain separate caches for low resolution
    (WV/IR) and high resolution (VIS) channels.
    """

    def __init__(self, func):
        """Create the cache.

        Args:
            func:
                Function to me cached.
        """
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        """Call the decorated function.

        Dataset is expected to be the last argument. If an instance method is
        decorated, it is preceded by a filehandler instance.
        """
        ds = args[-1]
        shape = ds.coords['y'].shape
        if shape not in self.cache:
            self.cache[shape] = self.func(*args)
        return self.cache[shape]

    def __get__(self, obj, objtype):
        """To support instance methods."""
        return functools.partial(self.__call__, obj)


class FiduceoMviriBase(BaseFileHandler):
    """Baseclass for FIDUCEO MVIRI file handlers."""

    def __init__(self, filename, filename_info, filetype_info,
                 mask_bad_quality=False):
        """Initialize the file handler.

        Args:
             mask_bad_quality: Mask pixels with bad quality, that means
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
        self.projection_longitude = filename_info['projection_longitude']

    @abc.abstractproperty
    def nc_keys(self):
        """Map satpy dataset names to netCDF variables."""
        raise NotImplementedError

    def get_dataset(self, dataset_id, info):
        """Get the dataset."""
        name = dataset_id['name']
        ds = self._read_dataset(name)
        if dataset_id['name'] in CHANNELS:
            ds = self.calibrate(ds, channel=name,
                                calibration=dataset_id['calibration'])
            if self.mask_bad_quality:
                ds = self._mask(ds)
            ds['acq_time'] = ('y', self._get_acq_time(ds))
        elif dataset_id['name'] in OTHER_REFLECTANCES:
            ds = ds * 100  # conversion to percent
        self._update_attrs(ds, info)
        return ds

    def _get_nc_key(self, name):
        """Get netCDF key corresponding to the given dataset name."""
        return

    def _read_dataset(self, name):
        """Read a dataset from the file."""
        nc_key = self.nc_keys.get(name, name)
        ds = self.nc[nc_key]
        if 'y_ir_wv' in ds.dims:
            ds = ds.rename({'y_ir_wv': 'y', 'x_ir_wv': 'x'})
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
            'p_id': 'geos_mviri',
            'a_name': 'geos_mviri',
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
            'distance_sun_earth'].values
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

    def _ir_wv_counts_to_radiance(self, counts, channel):
        """Convert IR/WV counts to radiance.

        Reference: [PUG], equations (4.1) and (4.2).
        """
        if channel == 'WV':
            a, b = self.nc['a_wv'], self.nc['b_wv']
        else:
            a, b = self.nc['a_ir'], self.nc['b_ir']
        rad = a + b * counts
        return rad.where(rad > 0)

    def _ir_wv_radiance_to_brightness_temperature(self, rad, channel):
        """Convert IR/WV radiance to brightness temperature."""
        if channel == 'WV':
            a, b = self.nc['bt_a_wv'], self.nc['bt_b_wv']
        else:
            a, b = self.nc['bt_a_ir'], self.nc['bt_b_ir']
        bt = b / (np.log(rad) - a)
        return bt.where(bt > 0)

    def _mask(self, ds):
        """Mask pixels with bad quality.

        Pixels are considered bad quality if the "quality_pixel_bitmask" is
        everything else than 0 (no flag set) or 2 ("use_with_caution" and no
        other flag set).
        """
        mask = self._get_mask(ds)
        return ds.where(np.logical_or(mask == 0, mask == 2))

    @shape_cache
    def _get_mask(self, ds):
        """Get quality bitmask for the given dataset."""
        mask = self.nc['quality_pixel_bitmask']

        # Mask has high (VIS) resolution. Downsample to low (IR/WV) resolution
        # if required.
        if mask.coords['y'].size > ds.coords['y'].size:
            mask = mask.isel(y=slice(None, None, 2),
                             x=slice(None, None, 2))
            mask = mask.assign_coords(y=ds.coords['y'].values,
                                      x=ds.coords['x'].values)
        return mask

    @shape_cache
    def _get_acq_time(self, ds):
        """Get scanline acquisition time.

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
        # Variable is sometimes named "time" and sometimes "time_ir_wv".
        try:
            time_lores = self.nc['time_ir_wv']
        except KeyError:
            time_lores = self.nc['time']

        # Compute mean timestamp per scanline
        time_lores = time_lores.mean(dim='x_ir_wv').rename({'y_ir_wv': 'y'})

        # If required, repeat timestamps in y-direction to obtain higher
        # resolution
        y_lores = time_lores.coords['y'].values
        y_target = ds.coords['y'].values
        if y_lores.size < y_target.size:
            reps = y_target.size // y_lores.size
            y_lores_rep = np.repeat(y_lores, reps)
            time_hires = time_lores.reindex(y=y_lores_rep)
            time_hires = time_hires.assign_coords(y=y_target)
            return time_hires

        return time_lores

    def _get_orbital_parameters(self):
        """Get the orbital parameters."""
        ssp_lon, ssp_lat = self._get_ssp_lonlat()
        orbital_parameters = {
            'projection_longitude': self.projection_longitude,
            'projection_latitude': 0.0,
            'projection_altitude': ALTITUDE
        }
        if not np.isnan(ssp_lon) and not np.isnan(ssp_lat):
            orbital_parameters.update({
                'satellite_actual_longitude': ssp_lon,
                'satellite_actual_latitude': ssp_lat,
                # altitude not available
            })

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

    @shape_cache
    def _get_solar_zenith_angle(self, ds):
        """Get solar zenith angle for the given dataset.

        Files provide solar zenith angle, but at a coarser resolution.
        Interpolate to the resolution of the given dataset.
        """
        return self._interp_tiepoints(self.nc['solar_zenith_angle'],
                                      ds.coords['x'],
                                      ds.coords['y'])

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
        if 'x_tie' not in ds.dims:
            raise ValueError('Dataset has no tiepoints to be interpolated.')

        # No tiepoint coordinates specified in the files. Use dimensions
        # to calculate tiepoint sampling and assign tiepoint coordinates
        # accordingly.
        sampling = target_x.size // ds.coords['x_tie'].size
        ds = ds.assign_coords(x_tie=target_x.values[::sampling],
                              y_tie=target_y.values[::sampling])

        ds_interp = ds.interp(x_tie=target_x, y_tie=target_y)
        return ds_interp.rename({'x_tie': 'x', 'y_tie': 'y'})


class FiduceoMviriEasyFcdrFileHandler(FiduceoMviriBase):
    """File handler for FIDUCEO MVIRI Easy FCDR."""

    nc_keys = {
        'VIS': 'toa_bidirectional_reflectance_vis',
        'WV': 'count_wv',
        'IR': 'count_ir'
    }

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

    nc_keys = {
        'VIS': 'count_vis',
        'WV': 'count_wv',
        'IR': 'count_ir'
    }

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

        rad = (counts - self.nc['mean_count_space_vis']) * a_cf
        return rad.where(rad > 0)

    def _vis_radiance_to_reflectance(self, rad):
        """Convert VIS radiance to reflectance factor.

        Reference: [PUG], equation (6).
        """
        sza = self._get_solar_zenith_angle(rad)
        cos_sza = np.cos(np.deg2rad(sza))
        refl = (
           (np.pi * self.nc['distance_sun_earth'] ** 2) /
           (self.nc['solar_irradiance_vis'] * cos_sza) *
           rad
        )
        refl = refl * 100  # conversion to percent
        return refl.where(refl > 0)
