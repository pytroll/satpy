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
                reader='mviri_l1b_fiduceo_nc')
    scn.load(['VIS', 'WV', 'IR'])

Global netCDF attributes are available in the ``raw_metadata`` attribute of
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


VIS Channel Quality Flags
-------------------------
Quality flags are available for the VIS channel only. A simple approach for
masking bad quality pixels is to set the ``mask_bad_quality`` keyword argument
to ``True``:

.. code-block:: python

    scn = Scene(filenames=['FIDUCEO_FCDR_L15_MVIRI_MET7-57.0...'],
                reader='mviri_l1b_fiduceo_nc',
                reader_kwargs={'mask_bad_quality': True})

See :class:`FiduceoMviriBase` for an argument description. In some situations
however the entire image can be flagged (look out for warnings). In that case
check out the ``quality_pixel_bitmask`` and ``data_quality_bitmask`` datasets
to find out why.


Angles
------
The FIDUCEO MVIRI FCDR provides satellite and solar angles on a coarse tiepoint
grid. By default these datasets will be interpolated to the higher VIS
resolution. This can be changed as follows:

.. code-block:: python

    scn.load(['solar_zenith_angle'], resolution=4500)

If you need the angles in both resolutions, use data queries:

.. code-block:: python

    from satpy import DataQuery

    query_vis = DataQuery(
        name='solar_zenith_angle',
        resolution=2250
    )
    query_ir = DataQuery(
        name='solar_zenith_angle',
        resolution=4500
    )
    scn.load([query_vis, query_ir])

    # Use the query objects to access the datasets as follows
    sza_vis = scn[query_vis]


References
----------
    - `[Handbook]`_ MFG User Handbook
    - `[PUG]`_ FIDUCEO MVIRI FCDR Product User Guide

.. _[Handbook]: https://www.eumetsat.int/media/7323
.. _[PUG]: http://doi.org/10.15770/EUM_SEC_CLM_0009
"""

import abc
import functools
import warnings

import dask.array as da
import numpy as np
import xarray as xr

from satpy import CHUNK_SIZE
from satpy.readers._geos_area import get_area_definition, get_area_extent, sampling_to_lfac_cfac
from satpy.readers.file_handlers import BaseFileHandler

EQUATOR_RADIUS = 6378140.0
POLE_RADIUS = 6356755.0
ALTITUDE = 42164000.0 - EQUATOR_RADIUS
"""[Handbook] section 5.2.1."""

MVIRI_FIELD_OF_VIEW = 18.0
"""[Handbook] section 5.3.2.1."""

CHANNELS = ['VIS', 'WV', 'IR']
ANGLES = [
    'solar_zenith_angle',
    'solar_azimuth_angle',
    'satellite_zenith_angle',
    'satellite_azimuth_angle'
]
OTHER_REFLECTANCES = [
    'u_independent_toa_bidirectional_reflectance',
    'u_structured_toa_bidirectional_reflectance'
]
HIGH_RESOL = 2250


class IRWVCalibrator:
    """Calibrate IR & WV channels."""

    def __init__(self, coefs):
        """Initialize the calibrator.

        Args:
            coefs: Calibration coefficients.
        """
        self.coefs = coefs

    def calibrate(self, counts, calibration):
        """Calibrate IR/WV counts to the given calibration."""
        if calibration == 'counts':
            return counts
        elif calibration in ('radiance', 'brightness_temperature'):
            return self._calibrate_rad_bt(counts, calibration)
        else:
            raise KeyError(
                'Invalid IR/WV calibration: {}'.format(calibration.name)
            )

    def _calibrate_rad_bt(self, counts, calibration):
        """Calibrate counts to radiance or brightness temperature."""
        rad = self._counts_to_radiance(counts)
        if calibration == 'radiance':
            return rad
        bt = self._radiance_to_brightness_temperature(rad)
        return bt

    def _counts_to_radiance(self, counts):
        """Convert IR/WV counts to radiance.

        Reference: [PUG], equations (4.1) and (4.2).
        """
        rad = self.coefs['a'] + self.coefs['b'] * counts
        return rad.where(rad > 0, np.float32(np.nan))

    def _radiance_to_brightness_temperature(self, rad):
        """Convert IR/WV radiance to brightness temperature.

        Reference: [PUG], equations (5.1) and (5.2).
        """
        bt = self.coefs['bt_b'] / (np.log(rad) - self.coefs['bt_a'])
        return bt.where(bt > 0, np.float32(np.nan))


class VISCalibrator:
    """Calibrate VIS channel."""

    def __init__(self, coefs, solar_zenith_angle=None):
        """Initialize the calibrator.

        Args:
            coefs:
                Calibration coefficients.
            solar_zenith_angle (optional):
                Solar zenith angle. Only required for calibration to
                reflectance.
        """
        self.coefs = coefs
        self.solar_zenith_angle = solar_zenith_angle

    def calibrate(self, counts, calibration):
        """Calibrate VIS counts."""
        if calibration == 'counts':
            return counts
        elif calibration in ('radiance', 'reflectance'):
            return self._calibrate_rad_refl(counts, calibration)
        else:
            raise KeyError(
                'Invalid VIS calibration: {}'.format(calibration.name)
            )

    def _calibrate_rad_refl(self, counts, calibration):
        """Calibrate counts to radiance or reflectance."""
        rad = self._counts_to_radiance(counts)
        if calibration == 'radiance':
            return rad
        refl = self._radiance_to_reflectance(rad)
        refl = self.update_refl_attrs(refl)
        return refl

    def _counts_to_radiance(self, counts):
        """Convert VIS counts to radiance.

        Reference: [PUG], equations (7) and (8).
        """
        years_since_launch = self.coefs['years_since_launch']
        a_cf = (self.coefs['a0'] +
                self.coefs['a1'] * years_since_launch +
                self.coefs['a2'] * years_since_launch ** 2)
        mean_count_space_vis = self.coefs['mean_count_space']
        rad = (counts - mean_count_space_vis) * a_cf
        return rad.where(rad > 0, np.float32(np.nan))

    def _radiance_to_reflectance(self, rad):
        """Convert VIS radiance to reflectance factor.

        Note: Produces huge reflectances in situations where both radiance and
        solar zenith angle are small. Maybe the corresponding uncertainties
        can be used to filter these cases before calculating reflectances.

        Reference: [PUG], equation (6).
        """
        sza = self.solar_zenith_angle.where(
            da.fabs(self.solar_zenith_angle) < 90,
            np.float32(np.nan)
        )  # direct illumination only
        cos_sza = np.cos(np.deg2rad(sza))
        refl = (
           (np.pi * self.coefs['distance_sun_earth'] ** 2) /
           (self.coefs['solar_irradiance'] * cos_sza) *
           rad
        )
        return self.refl_factor_to_percent(refl)

    def update_refl_attrs(self, refl):
        """Update attributes of reflectance datasets."""
        refl.attrs['sun_earth_distance_correction_applied'] = True
        refl.attrs['sun_earth_distance_correction_factor'] = self.coefs[
            'distance_sun_earth'].item()
        return refl

    @staticmethod
    def refl_factor_to_percent(refl):
        """Convert reflectance factor to percent."""
        return refl * 100


class Navigator:
    """Navigate MVIRI images."""

    def get_area_def(self, im_size, projection_longitude):
        """Create MVIRI area definition."""
        proj_params = self._get_proj_params(im_size, projection_longitude)
        extent = get_area_extent(proj_params)
        return get_area_definition(proj_params, extent)

    def _get_proj_params(self, im_size, projection_longitude):
        """Get projection parameters for the given settings."""
        area_name = 'geos_mviri_{0}x{0}'.format(im_size)
        lfac, cfac, loff, coff = self._get_factors_offsets(im_size)
        return {
            'ssp_lon': projection_longitude,
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

    def _get_factors_offsets(self, im_size):
        """Determine line/column offsets and scaling factors."""
        # For offsets see variables "asamp" and "aline" of subroutine
        # "refgeo" in [Handbook] and in
        # https://github.com/FIDUCEO/FCDR_MVIRI/blob/master/lib/nrCrunch/cruncher.f
        loff = coff = im_size / 2 + 0.5
        lfac = cfac = sampling_to_lfac_cfac(
            np.deg2rad(MVIRI_FIELD_OF_VIEW) / im_size
        )
        return lfac, cfac, loff, coff


class Interpolator:
    """Interpolate datasets to another resolution."""

    @staticmethod
    def interp_tiepoints(ds, target_x, target_y):
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

    @staticmethod
    def interp_acq_time(time2d, target_y):
        """Interpolate scanline acquisition time to the given coordinates.

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
        time = time2d.mean(dim='x')

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


class VisQualityControl:
    """Simple quality control for VIS channel."""

    def __init__(self, mask):
        """Initialize the quality control."""
        self._mask = mask

    def check(self):
        """Check VIS channel quality and issue a warning if it's bad."""
        use_with_caution = da.bitwise_and(self._mask, 2)
        if use_with_caution.all():
            warnings.warn(
                'All pixels of the VIS channel are flagged as "use with '
                'caution". Use datasets "quality_pixel_bitmask" and '
                '"data_quality_bitmask" to find out why.'
            )

    def mask(self, ds):
        """Mask VIS pixels with bad quality.

        Pixels are considered bad quality if the "quality_pixel_bitmask" is
        everything else than 0 (no flag set).
        """
        return ds.where(self._mask == 0, np.float32(np.nan))


def is_high_resol(resolution):
    """Identify high resolution channel."""
    return resolution == HIGH_RESOL


class DatasetWrapper:
    """Helper class for accessing the dataset."""

    def __init__(self, nc):
        """Wrap the given dataset."""
        self.nc = nc

    @property
    def attrs(self):
        """Exposes dataset attributes."""
        return self.nc.attrs

    def __getitem__(self, item):
        """Get a variable from the dataset."""
        ds = self.nc[item]
        if self._should_dims_be_renamed(ds):
            ds = self._rename_dims(ds)
        elif self._coordinates_not_assigned(ds):
            ds = self._reassign_coords(ds)
        self._cleanup_attrs(ds)
        return ds

    def _should_dims_be_renamed(self, ds):
        """Determine whether dataset dimensions need to be renamed."""
        return 'y_ir_wv' in ds.dims or 'y_tie' in ds.dims

    def _rename_dims(self, ds):
        """Rename dataset dimensions to match satpy's expectations."""
        new_names = {
            'y_ir_wv': 'y',
            'x_ir_wv': 'x',
            'y_tie': 'y',
            'x_tie': 'x'
        }
        for old_name, new_name in new_names.items():
            if old_name in ds.dims:
                ds = ds.rename({old_name: new_name})
        return ds

    def _coordinates_not_assigned(self, ds):
        return 'y' in ds.dims and 'y' not in ds.coords

    def _reassign_coords(self, ds):
        """Re-assign coordinates.

        For some reason xarray doesn't assign coordinates to all high
        resolution data variables.
        """
        return ds.assign_coords({'y': self.nc.coords['y'],
                                 'x': self.nc.coords['x']})

    def _cleanup_attrs(self, ds):
        """Cleanup dataset attributes."""
        # Remove ancillary_variables attribute to avoid downstream
        # satpy warnings.
        ds.attrs.pop('ancillary_variables', None)

    def get_time(self):
        """Get time coordinate.

        Variable is sometimes named "time" and sometimes "time_ir_wv".
        """
        try:
            return self['time_ir_wv']
        except KeyError:
            return self['time']

    def get_xy_coords(self, resolution):
        """Get x and y coordinates for the given resolution."""
        if is_high_resol(resolution):
            return self.nc.coords['x'], self.nc.coords['y']
        return self.nc.coords['x_ir_wv'], self.nc.coords['x_ir_wv']

    def get_image_size(self, resolution):
        """Get image size for the given resolution."""
        if is_high_resol(resolution):
            return self.nc.coords['y'].size
        return self.nc.coords['y_ir_wv'].size


class FiduceoMviriBase(BaseFileHandler):
    """Baseclass for FIDUCEO MVIRI file handlers."""

    nc_keys = {
        'WV': 'count_wv',
        'IR': 'count_ir'
    }

    def __init__(self, filename, filename_info, filetype_info,
                 mask_bad_quality=False):
        """Initialize the file handler.

        Args:
             mask_bad_quality: Mask VIS pixels with bad quality, that means
                 any quality flag except "ok". If you need more control, use
                 the ``quality_pixel_bitmask`` and ``data_quality_bitmask``
                 datasets.
        """
        super(FiduceoMviriBase, self).__init__(
            filename, filename_info, filetype_info)
        self.mask_bad_quality = mask_bad_quality
        nc_raw = xr.open_dataset(
            filename,
            chunks={'x': CHUNK_SIZE,
                    'y': CHUNK_SIZE,
                    'x_ir_wv': CHUNK_SIZE,
                    'y_ir_wv': CHUNK_SIZE}
        )
        self.nc = DatasetWrapper(nc_raw)

        # Projection longitude is not provided in the file, read it from the
        # filename.
        self.projection_longitude = float(filename_info['projection_longitude'])
        self.calib_coefs = self._get_calib_coefs()

        self._get_angles = functools.lru_cache(maxsize=8)(
            self._get_angles_uncached
        )
        self._get_acq_time = functools.lru_cache(maxsize=3)(
            self._get_acq_time_uncached
        )

    def get_dataset(self, dataset_id, dataset_info):
        """Get the dataset."""
        name = dataset_id['name']
        resolution = dataset_id['resolution']
        if name in ANGLES:
            ds = self._get_angles(name, resolution)
        elif name in CHANNELS:
            ds = self._get_channel(name, resolution, dataset_id['calibration'])
        else:
            ds = self._get_other_dataset(name)
        ds = self._cleanup_coords(ds)
        self._update_attrs(ds, dataset_info)
        return ds

    def get_area_def(self, dataset_id):
        """Get area definition of the given dataset."""
        im_size = self.nc.get_image_size(dataset_id['resolution'])
        nav = Navigator()
        return nav.get_area_def(
            im_size=im_size,
            projection_longitude=self.projection_longitude
        )

    def _get_channel(self, name, resolution, calibration):
        """Get and calibrate channel data."""
        ds = self.nc[self.nc_keys[name]]
        ds = self._calibrate(
            ds,
            channel=name,
            calibration=calibration
        )
        if name == 'VIS':
            qc = VisQualityControl(self.nc['quality_pixel_bitmask'])
            if self.mask_bad_quality:
                ds = qc.mask(ds)
            else:
                qc.check()
        ds['acq_time'] = self._get_acq_time(resolution)
        return ds

    def _get_angles_uncached(self, name, resolution):
        """Get angle dataset.

        Files provide angles (solar/satellite zenith & azimuth) at a coarser
        resolution. Interpolate them to the desired resolution.
        """
        angles = self.nc[name]
        target_x, target_y = self.nc.get_xy_coords(resolution)
        return Interpolator.interp_tiepoints(
            angles,
            target_x=target_x,
            target_y=target_y
        )

    def _get_other_dataset(self, name):
        """Get other datasets such as uncertainties."""
        ds = self.nc[name]
        if name in OTHER_REFLECTANCES:
            ds = VISCalibrator.refl_factor_to_percent(ds)
        return ds

    def _update_attrs(self, ds, info):
        """Update dataset attributes."""
        ds.attrs.update(info)
        ds.attrs.update({'platform': self.filename_info['platform'],
                         'sensor': self.filename_info['sensor']})
        ds.attrs['raw_metadata'] = self.nc.attrs
        ds.attrs['orbital_parameters'] = self._get_orbital_parameters()

    def _cleanup_coords(self, ds):
        """Cleanup dataset coordinates.

        Y/x coordinates have been useful for interpolation so far, but they
        only contain row/column numbers. Drop these coordinates so that Satpy
        can assign projection coordinates upstream (based on the area
        definition).
        """
        return ds.drop_vars(['y', 'x'])

    def _calibrate(self, ds, channel, calibration):
        """Calibrate the given dataset."""
        if channel == 'VIS':
            return self._calibrate_vis(ds, channel, calibration)
        calib = IRWVCalibrator(self.calib_coefs[channel])
        return calib.calibrate(ds, calibration)

    @abc.abstractmethod
    def _calibrate_vis(self, ds, channel, calibration):  # pragma: no cover
        """Calibrate VIS channel. To be implemented by subclasses."""
        raise NotImplementedError

    def _get_calib_coefs(self):
        """Get calibration coefficients for all channels.

        Note: Only coefficients present in both file types.
        """
        coefs = {
            'VIS': {
                'distance_sun_earth': self.nc['distance_sun_earth'],
                'solar_irradiance': self.nc['solar_irradiance_vis']
            },
            'IR': {
                'a': self.nc['a_ir'],
                'b': self.nc['b_ir'],
                'bt_a': self.nc['bt_a_ir'],
                'bt_b': self.nc['bt_b_ir']
            },
            'WV': {
                'a': self.nc['a_wv'],
                'b': self.nc['b_wv'],
                'bt_a': self.nc['bt_a_wv'],
                'bt_b': self.nc['bt_b_wv']
            },
        }

        # Convert coefficients to 32bit float to reduce memory footprint
        # of calibrated data.
        for ch in coefs:
            for name in coefs[ch]:
                coefs[ch][name] = np.float32(coefs[ch][name])

        return coefs

    def _get_acq_time_uncached(self, resolution):
        """Get scanline acquisition time for the given resolution.

        Note that the acquisition time does not increase monotonically
        with the scanline number due to the scan pattern and rectification.
        """
        time2d = self.nc.get_time()
        _, target_y = self.nc.get_xy_coords(resolution)
        return Interpolator.interp_acq_time(time2d, target_y=target_y.values)

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


class FiduceoMviriEasyFcdrFileHandler(FiduceoMviriBase):
    """File handler for FIDUCEO MVIRI Easy FCDR."""

    nc_keys = FiduceoMviriBase.nc_keys.copy()
    nc_keys['VIS'] = 'toa_bidirectional_reflectance_vis'

    def _calibrate_vis(self, ds, channel, calibration):
        """Calibrate VIS channel.

        Easy FCDR provides reflectance only, no counts or radiance.
        """
        if calibration == 'reflectance':
            coefs = self.calib_coefs[channel]
            cal = VISCalibrator(coefs)
            refl = cal.refl_factor_to_percent(ds)
            refl = cal.update_refl_attrs(refl)
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

    def _get_calib_coefs(self):
        """Add additional VIS coefficients only present in full FCDR."""
        coefs = super()._get_calib_coefs()
        coefs['VIS'].update({
            'years_since_launch': np.float32(self.nc['years_since_launch']),
            'a0': np.float32(self.nc['a0_vis']),
            'a1': np.float32(self.nc['a1_vis']),
            'a2': np.float32(self.nc['a2_vis']),
            'mean_count_space': np.float32(
                self.nc['mean_count_space_vis']
            )
        })
        return coefs

    def _calibrate_vis(self, ds, channel, calibration):
        """Calibrate VIS channel."""
        sza = None
        if calibration == 'reflectance':
            sza = self._get_angles('solar_zenith_angle', HIGH_RESOL)
        cal = VISCalibrator(self.calib_coefs[channel], sza)
        return cal.calibrate(ds, calibration)
