#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2011-2019 Satpy developers
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
"""Interface to VIIRS SDR format.

This reader implements the support of VIIRS SDR files as produced by CSPP and CLASS.
It is comprised of two parts:

 - A subclass of the YAMLFileReader class to allow handling all the files
 - A filehandler class to implement the actual reading

Format documentation:

 - http://npp.gsfc.nasa.gov/science/sciencedocuments/082012/474-00001-03_CDFCBVolIII_RevC.pdf

"""
import logging
from datetime import datetime, timedelta
from glob import glob
import os.path

import numpy as np
import dask.array as da
import xarray as xr

from satpy.readers.hdf5_utils import HDF5FileHandler
from satpy.readers.yaml_reader import FileYAMLReader

NO_DATE = datetime(1958, 1, 1)
EPSILON_TIME = timedelta(days=2)
LOG = logging.getLogger(__name__)


def _get_invalid_info(granule_data):
    """Get a detailed report of the missing data.

    N/A: not applicable
    MISS: required value missing at time of processing
    OBPT: onboard pixel trim (overlapping/bow-tie pixel removed during SDR processing)
    OGPT: on-ground pixel trim (overlapping/bow-tie pixel removed during EDR processing)
    ERR: error occurred during processing / non-convergence
    ELINT: ellipsoid intersect failed / instrument line-of-sight does not intersect the Earth’s surface
    VDNE: value does not exist / processing algorithm did not execute
    SOUB: scaled out-of-bounds / solution not within allowed range
    """
    if issubclass(granule_data.dtype.type, np.integer):
        msg = ("na:" + str((granule_data == 65535).sum()) +
               " miss:" + str((granule_data == 65534).sum()) +
               " obpt:" + str((granule_data == 65533).sum()) +
               " ogpt:" + str((granule_data == 65532).sum()) +
               " err:" + str((granule_data == 65531).sum()) +
               " elint:" + str((granule_data == 65530).sum()) +
               " vdne:" + str((granule_data == 65529).sum()) +
               " soub:" + str((granule_data == 65528).sum()))
    elif issubclass(granule_data.dtype.type, np.floating):
        msg = ("na:" + str((granule_data == -999.9).sum()) +
               " miss:" + str((granule_data == -999.8).sum()) +
               " obpt:" + str((granule_data == -999.7).sum()) +
               " ogpt:" + str((granule_data == -999.6).sum()) +
               " err:" + str((granule_data == -999.5).sum()) +
               " elint:" + str((granule_data == -999.4).sum()) +
               " vdne:" + str((granule_data == -999.3).sum()) +
               " soub:" + str((granule_data == -999.2).sum()))
    return msg


DATASET_KEYS = {'GDNBO': 'VIIRS-DNB-GEO',
                'SVDNB': 'VIIRS-DNB-SDR',
                'GITCO': 'VIIRS-IMG-GEO-TC',
                'GIMGO': 'VIIRS-IMG-GEO',
                'SVI01': 'VIIRS-I1-SDR',
                'SVI02': 'VIIRS-I2-SDR',
                'SVI03': 'VIIRS-I3-SDR',
                'SVI04': 'VIIRS-I4-SDR',
                'SVI05': 'VIIRS-I5-SDR',
                'GMTCO': 'VIIRS-MOD-GEO-TC',
                'GMODO': 'VIIRS-MOD-GEO',
                'SVM01': 'VIIRS-M1-SDR',
                'SVM02': 'VIIRS-M2-SDR',
                'SVM03': 'VIIRS-M3-SDR',
                'SVM04': 'VIIRS-M4-SDR',
                'SVM05': 'VIIRS-M5-SDR',
                'SVM06': 'VIIRS-M6-SDR',
                'SVM07': 'VIIRS-M7-SDR',
                'SVM08': 'VIIRS-M8-SDR',
                'SVM09': 'VIIRS-M9-SDR',
                'SVM10': 'VIIRS-M10-SDR',
                'SVM11': 'VIIRS-M11-SDR',
                'SVM12': 'VIIRS-M12-SDR',
                'SVM13': 'VIIRS-M13-SDR',
                'SVM14': 'VIIRS-M14-SDR',
                'SVM15': 'VIIRS-M15-SDR',
                'SVM16': 'VIIRS-M16-SDR',
                'IVCDB': 'VIIRS-DualGain-Cal-IP'
                }


class VIIRSSDRFileHandler(HDF5FileHandler):
    """VIIRS HDF5 File Reader."""

    def __init__(self, filename, filename_info, filetype_info, use_tc=None, **kwargs):
        """Initialize file handler."""
        self.datasets = filename_info['datasets'].split('-')
        self.use_tc = use_tc
        super(VIIRSSDRFileHandler, self).__init__(filename, filename_info, filetype_info)

    def __getitem__(self, item):
        """Get item."""
        if '*' in item:
            # this is an aggregated field that can't easily be loaded, need to
            # join things together
            idx = 0
            base_item = item
            item = base_item.replace('*', str(idx))
            result = []
            while True:
                try:
                    res = super(VIIRSSDRFileHandler, self).__getitem__(item)
                    result.append(res)
                except KeyError:
                    # no more granule keys
                    LOG.debug("Aggregated granule stopping on '%s'", item)
                    break

                idx += 1
                item = base_item.replace('*', str(idx))
            return result
        else:
            return super(VIIRSSDRFileHandler, self).__getitem__(item)

    def _parse_datetime(self, datestr, timestr):
        try:
            datetime_str = datestr + timestr
        except TypeError:
            datetime_str = str(datestr.astype(str)) + str(timestr.astype(str))
        time_val = datetime.strptime(datetime_str, '%Y%m%d%H%M%S.%fZ')
        if abs(time_val - NO_DATE) < EPSILON_TIME:
            # catch rare case when SDR files have incorrect date
            raise ValueError("Datetime invalid {}".format(time_val))
        return time_val

    @property
    def start_time(self):
        """Get start time."""
        dataset_group = DATASET_KEYS[self.datasets[0]]
        default_start_date = 'Data_Products/{dataset_group}/{dataset_group}_Aggr/attr/AggregateBeginningDate'
        default_start_time = 'Data_Products/{dataset_group}/{dataset_group}_Aggr/attr/AggregateBeginningTime'
        date_var_path = self.filetype_info.get('start_date', default_start_date).format(dataset_group=dataset_group)
        time_var_path = self.filetype_info.get('start_time', default_start_time).format(dataset_group=dataset_group)
        return self._parse_datetime(self[date_var_path], self[time_var_path])

    @property
    def end_time(self):
        """Get end time."""
        dataset_group = DATASET_KEYS[self.datasets[0]]
        default_end_date = 'Data_Products/{dataset_group}/{dataset_group}_Aggr/attr/AggregateEndingDate'
        default_end_time = 'Data_Products/{dataset_group}/{dataset_group}_Aggr/attr/AggregateEndingTime'
        date_var_path = self.filetype_info.get('end_date', default_end_date).format(dataset_group=dataset_group)
        time_var_path = self.filetype_info.get('end_time', default_end_time).format(dataset_group=dataset_group)
        return self._parse_datetime(self[date_var_path], self[time_var_path])

    @property
    def start_orbit_number(self):
        """Get start orbit number."""
        dataset_group = DATASET_KEYS[self.datasets[0]]
        default = 'Data_Products/{dataset_group}/{dataset_group}_Aggr/attr/AggregateBeginningOrbitNumber'
        start_orbit_path = self.filetype_info.get('start_orbit', default).format(dataset_group=dataset_group)
        return int(self[start_orbit_path])

    @property
    def end_orbit_number(self):
        """Get end orbit number."""
        dataset_group = DATASET_KEYS[self.datasets[0]]
        default = 'Data_Products/{dataset_group}/{dataset_group}_Aggr/attr/AggregateEndingOrbitNumber'
        end_orbit_path = self.filetype_info.get('end_orbit', default).format(dataset_group=dataset_group)
        return int(self[end_orbit_path])

    @property
    def platform_name(self):
        """Get platform name."""
        default = '/attr/Platform_Short_Name'
        platform_path = self.filetype_info.get(
            'platform_name', default).format(**self.filetype_info)
        platform_dict = {'NPP': 'Suomi-NPP',
                         'JPSS-1': 'NOAA-20',
                         'J01': 'NOAA-20',
                         'JPSS-2': 'NOAA-21',
                         'J02': 'NOAA-21'}
        return platform_dict.get(self[platform_path], self[platform_path])

    @property
    def sensor_name(self):
        """Get sensor name."""
        dataset_group = DATASET_KEYS[self.datasets[0]]
        default = 'Data_Products/{dataset_group}/attr/Instrument_Short_Name'
        sensor_path = self.filetype_info.get(
            'sensor_name', default).format(dataset_group=dataset_group)
        return self[sensor_path].lower()

    def get_file_units(self, dataset_id, ds_info):
        """Get file units."""
        file_units = ds_info.get("file_units")

        # Guess the file units if we need to (normally we would get this from
        # the file)
        if file_units is None:
            if dataset_id.calibration == 'radiance':
                if "dnb" in dataset_id.name.lower():
                    return 'W m-2 sr-1'
                else:
                    return 'W cm-2 sr-1'
            elif dataset_id.calibration == 'reflectance':
                # CF compliant unit for dimensionless
                file_units = "1"
            elif dataset_id.calibration == 'brightness_temperature':
                file_units = "K"
            else:
                LOG.debug("Unknown units for file key '%s'", dataset_id)

        return file_units

    def scale_swath_data(self, data, scaling_factors):
        """Scale swath data using scaling factors and offsets.

        Multi-granule (a.k.a. aggregated) files will have more than the usual two values.
        """
        num_grans = len(scaling_factors) // 2
        gran_size = data.shape[0] // num_grans
        factors = scaling_factors.where(scaling_factors > -999, np.float32(np.nan))
        factors = factors.data.reshape((-1, 2))
        factors = xr.DataArray(da.repeat(factors, gran_size, axis=0),
                               dims=(data.dims[0], 'factors'))
        data = data * factors[:, 0] + factors[:, 1]
        return data

    def adjust_scaling_factors(self, factors, file_units, output_units):
        """Adjust scaling factors."""
        if file_units == output_units:
            LOG.debug("File units and output units are the same (%s)", file_units)
            return factors
        if factors is None:
            return None
        factors = factors.where(factors != -999., np.float32(np.nan))

        if file_units == "W cm-2 sr-1" and output_units == "W m-2 sr-1":
            LOG.debug("Adjusting scaling factors to convert '%s' to '%s'", file_units, output_units)
            factors = factors * 10000.
            return factors
        elif file_units == "1" and output_units == "%":
            LOG.debug("Adjusting scaling factors to convert '%s' to '%s'", file_units, output_units)
            factors = factors * 100.
            return factors
        else:
            return factors

    def _generate_file_key(self, ds_id, ds_info, factors=False):
        var_path = ds_info.get('file_key', 'All_Data/{dataset_group}_All/{calibration}')
        calibration = {
            'radiance': 'Radiance',
            'reflectance': 'Reflectance',
            'brightness_temperature': 'BrightnessTemperature',
        }.get(ds_id.calibration)
        var_path = var_path.format(calibration=calibration, dataset_group=DATASET_KEYS[ds_info['dataset_group']])
        if ds_id.name in ['dnb_longitude', 'dnb_latitude']:
            if self.use_tc is True:
                return var_path + '_TC'
            elif self.use_tc is None and var_path + '_TC' in self.file_content:
                return var_path + '_TC'
        return var_path

    @staticmethod
    def expand_single_values(var, scans):
        """Expand single valued variable to full scan lengths."""
        if scans.size == 1:
            return var
        else:
            expanded = np.repeat(var, scans)
            expanded.attrs = var.attrs
            expanded.rename({expanded.dims[0]: 'y'})
            return expanded

    def _scan_size(self, dataset_group_name):
        """Get how many rows of data constitute one scanline."""
        if 'I' in dataset_group_name:
            scan_size = 32
        else:
            scan_size = 16
        return scan_size

    def concatenate_dataset(self, dataset_group, var_path):
        """Concatenate dataset."""
        scan_size = self._scan_size(dataset_group)
        number_of_granules_path = 'Data_Products/{dataset_group}/{dataset_group}_Aggr/attr/AggregateNumberGranules'
        nb_granules_path = number_of_granules_path.format(dataset_group=DATASET_KEYS[dataset_group])
        scans = []
        for granule in range(self[nb_granules_path]):
            scans_path = 'Data_Products/{dataset_group}/{dataset_group}_Gran_{granule}/attr/N_Number_Of_Scans'
            scans_path = scans_path.format(dataset_group=DATASET_KEYS[dataset_group], granule=granule)
            scans.append(self[scans_path])
        start_scan = 0
        data_chunks = []
        scans = xr.DataArray(scans)
        variable = self[var_path]
        # check if these are single per-granule value
        if variable.size != scans.size:
            for gscans in scans.values:
                data_chunks.append(self[var_path].isel(y=slice(start_scan, start_scan + gscans * scan_size)))
                start_scan += scan_size * 48
            return xr.concat(data_chunks, 'y')
        else:
            return self.expand_single_values(variable, scans)

    def mask_fill_values(self, data, ds_info):
        """Mask fill values."""
        is_floating = np.issubdtype(data.dtype, np.floating)

        if is_floating:
            # If the data is a float then we mask everything <= -999.0
            fill_max = np.float32(ds_info.pop("fill_max_float", -999.0))
            return data.where(data > fill_max, np.float32(np.nan))
        else:
            # If the data is an integer then we mask everything >= fill_min_int
            fill_min = int(ds_info.pop("fill_min_int", 65528))
            return data.where(data < fill_min, np.float32(np.nan))

    def get_dataset(self, dataset_id, ds_info):
        """Get the dataset corresponding to *dataset_id*.

        The size of the return DataArray will be dependent on the number of
        scans actually sensed, and not necessarily the regular 768 scanlines
        that the file contains for each granule. To that end, the number of
        scans for each granule is read from:
        ``Data_Products/...Gran_x/N_Number_Of_Scans``.
        """
        dataset_group = [ds_group for ds_group in ds_info['dataset_groups'] if ds_group in self.datasets]
        if not dataset_group:
            return
        else:
            dataset_group = dataset_group[0]
            ds_info['dataset_group'] = dataset_group
        var_path = self._generate_file_key(dataset_id, ds_info)
        factor_var_path = ds_info.get("factors_key", var_path + "Factors")

        data = self.concatenate_dataset(dataset_group, var_path)
        data = self.mask_fill_values(data, ds_info)
        factors = self.get(factor_var_path)
        if factors is None:
            LOG.debug("No scaling factors found for %s", dataset_id)

        file_units = self.get_file_units(dataset_id, ds_info)
        output_units = ds_info.get("units", file_units)
        factors = self.adjust_scaling_factors(factors, file_units, output_units)

        if factors is not None:
            data = self.scale_swath_data(data, factors)

        i = getattr(data, 'attrs', {})
        i.update(ds_info)
        i.update({
            "units": ds_info.get("units", file_units),
            "platform_name": self.platform_name,
            "sensor": self.sensor_name,
            "start_orbit": self.start_orbit_number,
            "end_orbit": self.end_orbit_number,
            "rows_per_scan": self._scan_size(dataset_group),
        })
        i.update(dataset_id.to_dict())
        data.attrs.update(i)
        return data

    def get_bounding_box(self):
        """Get the bounding box of this file."""
        from pyproj import Geod
        geod = Geod(ellps='WGS84')
        dataset_group = DATASET_KEYS[self.datasets[0]]
        idx = 0
        lons_ring = None
        lats_ring = None
        while True:
            path = 'Data_Products/{dataset_group}/{dataset_group}_Gran_{idx}/attr/'
            prefix = path.format(dataset_group=dataset_group, idx=idx)
            try:
                lats = self.file_content[prefix + 'G-Ring_Latitude']
                lons = self.file_content[prefix + 'G-Ring_Longitude']
                if lons_ring is None:
                    lons_ring = lons
                    lats_ring = lats
                else:
                    prev_lon = lons_ring[0]
                    prev_lat = lats_ring[0]
                    dists = list(geod.inv(lon, lat, prev_lon, prev_lat)[2] for lon, lat in zip(lons, lats))
                    first_idx = np.argmin(dists)
                    if first_idx == 2 and len(lons) == 8:
                        lons_ring = np.hstack((lons[:3], lons_ring[:-2], lons[4:]))
                        lats_ring = np.hstack((lats[:3], lats_ring[:-2], lats[4:]))
                    else:
                        raise NotImplementedError("Don't know how to handle G-Rings of length %d" % len(lons))

            except KeyError:
                break
            idx += 1

        return lons_ring, lats_ring

    def available_datasets(self, configured_datasets=None):
        """Generate dataset info and their availablity.

        See
        :meth:`satpy.readers.file_handlers.BaseFileHandler.available_datasets`
        for details.

        """
        for is_avail, ds_info in (configured_datasets or []):
            if is_avail is not None:
                yield is_avail, ds_info
                continue
            dataset_group = [ds_group for ds_group in ds_info['dataset_groups'] if ds_group in self.datasets]
            if dataset_group:
                yield True, ds_info
            elif is_avail is None:
                yield is_avail, ds_info


def split_desired_other(fhs, req_geo, rem_geo):
    """Split the provided filehandlers *fhs* into desired filehandlers and others."""
    desired = []
    other = []
    for fh in fhs:
        if req_geo in fh.datasets:
            desired.append(fh)
        elif rem_geo in fh.datasets:
            other.append(fh)
    return desired, other


class VIIRSSDRReader(FileYAMLReader):
    """Custom file reader for finding VIIRS SDR geolocation at runtime."""

    def __init__(self, config_files, use_tc=None, **kwargs):
        """Initialize file reader and adjust geolocation preferences.

        Args:
            config_files (iterable): yaml config files passed to base class
            use_tc (boolean): If `True` use the terrain corrected
                              files. If `False`, switch to non-TC files. If
                              `None` (default), use TC if available, non-TC otherwise.

        """
        super(VIIRSSDRReader, self).__init__(config_files, **kwargs)
        self.use_tc = use_tc

    def filter_filenames_by_info(self, filename_items):
        """Filter out file using metadata from the filenames.

        This sorts out the different lon and lat datasets depending on TC is
        desired or not.
        """
        filename_items = list(filename_items)
        geo_keep = []
        geo_del = []
        for filename, filename_info in filename_items:
            filename_info['datasets'] = datasets = filename_info['datasets'].split('-')
            if ('GITCO' in datasets) or ('GMTCO' in datasets):
                if self.use_tc is False:
                    geo_del.append(filename)
                else:
                    geo_keep.append(filename)
            elif ('GIMGO' in datasets) or ('GMODO' in datasets):
                if self.use_tc is True:
                    geo_del.append(filename)
                else:
                    geo_keep.append(filename)
        if geo_keep:
            fdict = dict(filename_items)
            for to_del in geo_del:
                for dataset in ['GITCO', 'GMTCO', 'GIMGO', 'GMODO']:
                    try:
                        fdict[to_del]['datasets'].remove(dataset)
                    except ValueError:
                        pass
                if not fdict[to_del]['datasets']:
                    del fdict[to_del]
            filename_items = fdict.items()
        for _filename, filename_info in filename_items:
            filename_info['datasets'] = '-'.join(filename_info['datasets'])
        return super(VIIRSSDRReader, self).filter_filenames_by_info(filename_items)

    def _load_from_geo_ref(self, dsid):
        """Load filenames from the N_GEO_Ref attribute of a dataset's file."""
        file_handlers = self._get_file_handlers(dsid)
        if not file_handlers:
            return None

        fns = []
        for fh in file_handlers:
            base_dir = os.path.dirname(fh.filename)
            try:
                # get the filename and remove the creation time
                # which is often wrong
                fn = fh['/attr/N_GEO_Ref'][:46] + '*.h5'
                fns.extend(glob(os.path.join(base_dir, fn)))

                # usually is non-terrain corrected file, add the terrain
                # corrected file too
                if fn[:5] == 'GIMGO':
                    fn = 'GITCO' + fn[5:]
                elif fn[:5] == 'GMODO':
                    fn = 'GMTCO' + fn[5:]
                else:
                    continue
                fns.extend(glob(os.path.join(base_dir, fn)))
            except KeyError:
                LOG.debug("Could not load geo-reference information from {}".format(fh.filename))

        return fns

    def _get_req_rem_geo(self, ds_info):
        """Find out which geolocation files are needed."""
        if ds_info['dataset_groups'][0].startswith('GM'):
            if self.use_tc is False:
                req_geo = 'GMODO'
                rem_geo = 'GMTCO'
            else:
                req_geo = 'GMTCO'
                rem_geo = 'GMODO'
        elif ds_info['dataset_groups'][0].startswith('GI'):
            if self.use_tc is False:
                req_geo = 'GIMGO'
                rem_geo = 'GITCO'
            else:
                req_geo = 'GITCO'
                rem_geo = 'GIMGO'
        else:
            raise ValueError('Unknown dataset group %s' % ds_info['dataset_groups'][0])
        return req_geo, rem_geo

    def get_right_geo_fhs(self, dsid, fhs):
        """Find the right geographical file handlers for given dataset ID *dsid*."""
        ds_info = self.all_ids[dsid]
        req_geo, rem_geo = self._get_req_rem_geo(ds_info)
        desired, other = split_desired_other(fhs, req_geo, rem_geo)
        if desired:
            try:
                ds_info['dataset_groups'].remove(rem_geo)
            except ValueError:
                pass
            return desired
        else:
            return other

    def _get_file_handlers(self, dsid):
        """Get the file handler to load this dataset."""
        ds_info = self.all_ids[dsid]

        fhs = [fh for fh in self.file_handlers['generic_file']
               if set(fh.datasets) & set(ds_info['dataset_groups'])]
        if not fhs:
            LOG.warning("Required file type '%s' not found or loaded for "
                        "'%s'", ds_info['file_type'], dsid.name)
        else:
            if len(set(ds_info['dataset_groups']) & set(['GITCO', 'GIMGO', 'GMTCO', 'GMODO'])) > 1:
                fhs = self.get_right_geo_fhs(dsid, fhs)
            return fhs

    def _get_coordinates_for_dataset_key(self, dsid):
        """Get the coordinate dataset keys for `dsid`.

        Wraps the base class method in order to load geolocation files
        from the geo reference attribute in the datasets file.
        """
        coords = super(VIIRSSDRReader, self)._get_coordinates_for_dataset_key(dsid)
        for c_id in coords:
            c_info = self.all_ids[c_id]  # c_info['dataset_groups'] should be a list of 2 elements
            self._get_file_handlers(c_id)
            if len(c_info['dataset_groups']) == 1:  # filtering already done
                continue
            try:
                req_geo, rem_geo = self._get_req_rem_geo(c_info)
            except ValueError:  # DNB
                continue

            # check the dataset file for the geolocation filename
            geo_filenames = self._load_from_geo_ref(dsid)
            if not geo_filenames:
                c_info['dataset_groups'] = [rem_geo]
            else:
                # concatenate all values
                new_fhs = sum(self.create_filehandlers(geo_filenames).values(), [])
                desired, other = split_desired_other(new_fhs, req_geo, rem_geo)
                if desired:
                    c_info['dataset_groups'].remove(rem_geo)
                else:
                    c_info['dataset_groups'].remove(req_geo)

        return coords
