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
r"""Reader for files produced with the cf netcdf writer in satpy.

Introduction
------------

The ``satpy_cf_nc`` reader reads data written by the satpy cf_writer. Filenames for cf_writer are optional.
There are several readers using the same satpy_cf_nc.py reader.

* Generic reader ``satpy_cf_nc``
* EUMETSAT GAC FDR reader ``avhrr_l1c_eum_gac_fdr_nc``

Generic reader
--------------

The generic ``satpy_cf_nc`` reader reads files of type:

.. code-block:: none

    '{platform_name}-{sensor}-{start_time:%Y%m%d%H%M%S}-{end_time:%Y%m%d%H%M%S}.nc'


Example
-------

Here is an example how to read the data in satpy:

.. code-block:: python

    from satpy import Scene

    filenames = ['data/npp-viirs-mband-20201007075915-20201007080744.nc']
    scn = Scene(reader='satpy_cf_nc', filenames=filenames)
    scn.load(['M05'])
    scn['M05']

Output:

.. code-block:: none

    <xarray.DataArray 'M05' (y: 4592, x: 3200)>
    dask.array<open_dataset-d91cfbf1bf4f14710d27446d91cdc6e4M05, shape=(4592, 3200),
        dtype=float32, chunksize=(4096, 3200), chunktype=numpy.ndarray>
    Coordinates:
        longitude  (y, x) float32 dask.array<chunksize=(4096, 3200), meta=np.ndarray>
        latitude   (y, x) float32 dask.array<chunksize=(4096, 3200), meta=np.ndarray>
    Dimensions without coordinates: y, x
    Attributes:
        start_time:                   2020-10-07 07:59:15
        start_orbit:                  46350
        end_time:                     2020-10-07 08:07:44
        end_orbit:                    46350
        calibration:                  reflectance
        long_name:                    M05
        modifiers:                    ('sunz_corrected',)
        platform_name:                Suomi-NPP
        resolution:                   742
        sensor:                       viirs
        standard_name:                toa_bidirectional_reflectance
        units:                        %
        wavelength:                   0.672 µm (0.662-0.682 µm)
        date_created:                 2020-10-07T08:20:02Z
        instrument:                   VIIRS


Notes:
    Available datasets and attributes will depend on the data saved with the cf_writer.

EUMETSAT AVHRR GAC FDR L1C reader
---------------------------------

The ``avhrr_l1c_eum_gac_fdr_nc`` reader reads files of type:

.. code-block:: none

    ''AVHRR-GAC_FDR_1C_{platform}_{start_time:%Y%m%dT%H%M%SZ}_{end_time:%Y%m%dT%H%M%SZ}_{processing_mode}_{disposition_mode}_{creation_time}_{version_int:04d}.nc'


Example
-------

Here is an example how to read the data in satpy:

.. code-block:: python

    from satpy import Scene

    filenames = ['data/AVHRR-GAC_FDR_1C_N06_19810330T042358Z_19810330T060903Z_R_O_20200101T000000Z_0100.nc']
    scn = Scene(reader='avhrr_l1c_eum_gac_fdr_nc', filenames=filenames)
    scn.load(['brightness_temperature_channel_4'])
    scn['brightness_temperature_channel_4']


Output:

.. code-block:: none

    <xarray.DataArray 'brightness_temperature_channel_4' (y: 11, x: 409)>
    dask.array<open_dataset-55ffbf3623b32077c67897f4283640a5brightness_temperature_channel_4, shape=(11, 409),
        dtype=float32, chunksize=(11, 409), chunktype=numpy.ndarray>
    Coordinates:
      * x          (x) int16 0 1 2 3 4 5 6 7 8 ... 401 402 403 404 405 406 407 408
      * y          (y) int64 0 1 2 3 4 5 6 7 8 9 10
        acq_time   (y) datetime64[ns] dask.array<chunksize=(11,), meta=np.ndarray>
        longitude  (y, x) float64 dask.array<chunksize=(11, 409), meta=np.ndarray>
        latitude   (y, x) float64 dask.array<chunksize=(11, 409), meta=np.ndarray>
    Attributes:
        start_time:                            1981-03-30 04:23:58
        end_time:                              1981-03-30 06:09:03
        calibration:                           brightness_temperature
        modifiers:                             ()
        resolution:                            1050
        standard_name:                         toa_brightness_temperature
        units:                                 K
        wavelength:                            10.8 µm (10.3-11.3 µm)
        Conventions:                           CF-1.8 ACDD-1.3
        comment:                               Developed in cooperation with EUME...
        creator_email:                         ops@eumetsat.int
        creator_name:                          EUMETSAT
        creator_url:                           https://www.eumetsat.int/
        date_created:                          2020-09-14T10:50:51.073707
        disposition_mode:                      O
        gac_filename:                          NSS.GHRR.NA.D81089.S0423.E0609.B09...
        geospatial_lat_max:                    89.95386902434623
        geospatial_lat_min:                    -89.97581969005503
        geospatial_lat_resolution:             1050 meters
        geospatial_lat_units:                  degrees_north
        geospatial_lon_max:                    179.99952992568998
        geospatial_lon_min:                    -180.0
        geospatial_lon_resolution:             1050 meters
        geospatial_lon_units:                  degrees_east
        ground_station:                        GC
        id:                                    DOI:10.5676/EUM/AVHRR_GAC_L1C_FDR/...
        institution:                           EUMETSAT
        instrument:                            Earth Remote Sensing Instruments >...
        keywords:                              ATMOSPHERE > ATMOSPHERIC RADIATION...
        keywords_vocabulary:                   GCMD Science Keywords, Version 9.1
        licence:                               EUMETSAT data policy https://www.e...
        naming_authority:                      int.eumetsat
        orbit_number_end:                      9123
        orbit_number_start:                    9122
        orbital_parameters_tle:                ['1 11416U 79057A   81090.16350942...
        platform:                              Earth Observation Satellites > NOA...
        processing_level:                      1C
        processing_mode:                       R
        product_version:                       1.0.0
        references:                            Devasthale, A., M. Raspaud, C. Sch...
        source:                                AVHRR GAC Level 1 Data
        standard_name_vocabulary:              CF Standard Name Table v73
        summary:                               Fundamental Data Record (FDR) of m...
        sun_earth_distance_correction_factor:  0.9975244779999585
        time_coverage_end:                     19820803T003900Z
        time_coverage_start:                   19800101T000000Z
        title:                                 AVHRR GAC L1C FDR
        version_calib_coeffs:                  PATMOS-x, v2017r1
        version_pygac:                         1.4.0
        version_pygac_fdr:                     0.1.dev107+gceb7b26.d20200910
        version_satpy:                         0.21.1.dev894+g5cf76e6
        history:                               Created by pytroll/satpy on 2020-0...
        name:                                  brightness_temperature_channel_4
        _satpy_id:                             DataID(name='brightness_temperatur...
        ancillary_variables:                   []

"""
import itertools
import json
import logging

import xarray as xr
from pyresample import AreaDefinition

from satpy import CHUNK_SIZE
from satpy.dataset.dataid import WavelengthRange
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


class SatpyCFFileHandler(BaseFileHandler):
    """File handler for Satpy's CF netCDF files."""

    def __init__(self, filename, filename_info, filetype_info, numeric_name_prefix='CHANNEL_'):
        """Initialize file handler."""
        super().__init__(filename, filename_info, filetype_info)
        self.engine = None
        self._numeric_name_prefix = numeric_name_prefix

    @property
    def start_time(self):
        """Get start time."""
        return self.filename_info['start_time']

    @property
    def end_time(self):
        """Get end time."""
        return self.filename_info.get('end_time', self.start_time)

    @property
    def sensor_names(self):
        """Get sensor set."""
        sensors = set()
        for _, ds_info in self.available_datasets():
            try:
                sensors.add(ds_info["sensor"])
            except KeyError:
                continue
        return sensors

    def available_datasets(self, configured_datasets=None):
        """Add information of available datasets."""
        existing = self._existing_datasets(configured_datasets=configured_datasets)
        dynamic = self._dynamic_datasets()
        coordinates = self._coordinate_datasets()
        for dataset_available, dataset_info in itertools.chain(existing, dynamic, coordinates):
            yield dataset_available, dataset_info

    def _existing_datasets(self, configured_datasets=None):
        """Add information of existing datasets."""
        for is_avail, ds_info in (configured_datasets or []):
            yield is_avail, ds_info

    def fix_modifier_attr(self, ds_info):
        """Fix modifiers attribute."""
        # Empty modifiers are read as [], which causes problems later
        if 'modifiers' in ds_info and not ds_info['modifiers']:
            ds_info['modifiers'] = ()
        try:
            try:
                ds_info['modifiers'] = tuple(ds_info['modifiers'].split(' '))
            except AttributeError:
                pass
        except KeyError:
            pass

    def _assign_ds_info(self, var_name, val):
        """Assign ds_info."""
        ds_info = dict(val.attrs)
        ds_info['file_type'] = self.filetype_info['file_type']
        ds_info['name'] = ds_info['nc_store_name'] = var_name
        if 'original_name' in ds_info:
            ds_info['name'] = ds_info['original_name']
        elif self._numeric_name_prefix and var_name.startswith(self._numeric_name_prefix):
            ds_info['name'] = var_name.replace(self._numeric_name_prefix, '')
        try:
            ds_info['wavelength'] = WavelengthRange.from_cf(ds_info['wavelength'])
        except KeyError:
            pass
        return ds_info

    def _dynamic_datasets(self):
        """Add information of dynamic datasets."""
        nc = xr.open_dataset(self.filename, engine=self.engine)
        # get dynamic variables known to this file (that we created)
        for var_name, val in nc.data_vars.items():
            ds_info = self._assign_ds_info(var_name, val)
            self.fix_modifier_attr(ds_info)
            yield True, ds_info

    def _coordinate_datasets(self, configured_datasets=None):
        """Add information of coordinate datasets."""
        nc = xr.open_dataset(self.filename, engine=self.engine)
        for var_name, val in nc.coords.items():
            ds_info = dict(val.attrs)
            ds_info['file_type'] = self.filetype_info['file_type']
            ds_info['name'] = var_name
            self.fix_modifier_attr(ds_info)
            yield True, ds_info

    def _compare_attr(self, _ds_id_dict, key, data):
        if key in ['name', 'modifiers']:
            return True
        elif key == 'wavelength':
            return _ds_id_dict[key] == WavelengthRange.from_cf(data.attrs[key])
        else:
            return data.attrs[key] == _ds_id_dict[key]

    def _dataid_attrs_equal(self, ds_id, data):
        _ds_id_dict = ds_id.to_dict()
        for key in _ds_id_dict:
            try:
                if not self._compare_attr(_ds_id_dict, key, data):
                    return False
            except KeyError:
                pass
        return True

    def get_dataset(self, ds_id, ds_info):
        """Get dataset."""
        logger.debug("Getting data for: %s", ds_id['name'])
        nc = xr.open_dataset(self.filename, engine=self.engine,
                             chunks={'y': CHUNK_SIZE, 'x': CHUNK_SIZE})
        name = ds_info.get('nc_store_name', ds_id['name'])
        data = nc[ds_info.get('file_key', name)]
        if not self._dataid_attrs_equal(ds_id, data):
            return
        if name != ds_id['name']:
            data = data.rename(ds_id['name'])
        data.attrs.update(nc.attrs)  # For now add global attributes to all datasets
        if "orbital_parameters" in data.attrs:
            data.attrs["orbital_parameters"] = _str2dict(data.attrs["orbital_parameters"])

        return data

    def get_area_def(self, dataset_id):
        """Get area definition from CF complient netcdf."""
        try:
            area = AreaDefinition.from_cf(self.filename)
            return area
        except ValueError:
            # No CF compliant projection information was found in the netcdf file or
            # file contains 2D lat/lon arrays. To fall back to generating a SwathDefinition
            # with the yaml_reader NotImplementedError is raised.
            logger.debug("No AreaDefinition to load from nc file. Falling back to SwathDefinition.")
            raise NotImplementedError


def _str2dict(val):
    """Convert string to dictionary."""
    if isinstance(val, str):
        val = json.loads(val)
    return val
