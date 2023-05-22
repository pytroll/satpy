#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2011-2023 Satpy developers
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
import os.path
from contextlib import suppress
from datetime import datetime, timedelta
from glob import glob

import numpy as np

from satpy.readers.viirs_atms_sdr_base import ATMS_DATASET_KEYS, DATASET_KEYS, VIIRS_DATASET_KEYS, JPSS_SDR_FileHandler
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
    msg = None
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


class VIIRSSDRFileHandler(JPSS_SDR_FileHandler):
    """VIIRS SDR HDF5 File Reader."""

    def __init__(self, filename, filename_info, filetype_info, use_tc=None, **kwargs):
        """Initialize file handler."""
        self.datasets = filename_info['datasets'].split('-')
        self.use_tc = use_tc
        super().__init__(filename, filename_info, filetype_info, **kwargs)

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
                    res = super().__getitem__(item)
                    result.append(res)
                except KeyError:
                    # no more granule keys
                    LOG.debug("Aggregated granule stopping on '%s'", item)
                    break

                idx += 1
                item = base_item.replace('*', str(idx))
            return result
        else:
            return super().__getitem__(item)

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
        dataset_group = dataset_group[0]
        ds_info['dataset_group'] = dataset_group
        var_path = self._generate_file_key(dataset_id, ds_info)

        data = self.concatenate_dataset(dataset_group, var_path)
        data = self.mask_fill_values(data, ds_info)

        data = self.scale_data_to_specified_unit(data, dataset_id, ds_info)
        data = self._update_data_attributes(data, dataset_id, ds_info)

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
                    dists = [geod.inv(lon, lat, prev_lon, prev_lat)[2] for lon, lat in zip(lons, lats)]
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


def split_desired_other(fhs, prime_geo, second_geo):
    """Split the provided filehandlers *fhs* into desired filehandlers and others."""
    desired = []
    other = []
    for fh in fhs:
        if prime_geo in fh.datasets:
            desired.append(fh)
        elif second_geo in fh.datasets:
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
        super().__init__(config_files, **kwargs)
        self.use_tc = use_tc

    def _is_viirs_dataset(self, datasets):
        for dataset in VIIRS_DATASET_KEYS:
            if dataset in datasets:
                return True
        return False

    def filter_filenames_by_info(self, filename_items):
        """Filter out file using metadata from the filenames.

        This sorts out the different lon and lat datasets depending on TC is
        desired or not.
        """
        filename_items = list(filename_items)
        geo_keep = []
        geo_del = []
        viirs_del = []
        for filename, filename_info in filename_items:
            datasets = filename_info['datasets'].split('-')
            if not self._is_viirs_dataset(datasets):
                viirs_del.append(filename)

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
            filename_items = self._remove_geo_datasets_from_files(filename_items, geo_del)

        filename_items = self._remove_non_viirs_datasets_from_files(filename_items, viirs_del)
        return super().filter_filenames_by_info(filename_items)

    def _remove_non_viirs_datasets_from_files(self, filename_items, files_to_edit):
        no_viirs = ATMS_DATASET_KEYS
        return self._remove_datasets_from_files(filename_items, files_to_edit, no_viirs)

    def _remove_geo_datasets_from_files(self, filename_items, files_to_edit):
        datasets_to_consider = ['GITCO', 'GMTCO', 'GIMGO', 'GMODO']
        return self._remove_datasets_from_files(filename_items, files_to_edit, datasets_to_consider)

    def _remove_datasets_from_files(self, filename_items, files_to_edit, considered_datasets):
        fdict = dict(filename_items)
        for to_del in files_to_edit:
            fdict[to_del]['datasets'] = fdict[to_del]['datasets'].split('-')
            for dataset in considered_datasets:
                with suppress(ValueError):
                    fdict[to_del]['datasets'].remove(dataset)
            if not fdict[to_del]['datasets']:
                del fdict[to_del]
            else:
                fdict[to_del]['datasets'] = "-".join(fdict[to_del]['datasets'])
        filename_items = fdict.items()
        return filename_items

    def _load_filenames_from_geo_ref(self, dsid):
        """Load filenames from the N_GEO_Ref attribute of a dataset's file."""
        file_handlers = self._get_file_handlers(dsid)
        if not file_handlers:
            return []

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

    def _get_primary_secondary_geo_groups(self, ds_info):
        """Find out which geolocation files are needed."""
        if ds_info['dataset_groups'][0].startswith('GM'):
            if self.use_tc is False:
                prime_geo = 'GMODO'
                second_geo = 'GMTCO'
            else:
                prime_geo = 'GMTCO'
                second_geo = 'GMODO'
        elif ds_info['dataset_groups'][0].startswith('GI'):
            if self.use_tc is False:
                prime_geo = 'GIMGO'
                second_geo = 'GITCO'
            else:
                prime_geo = 'GITCO'
                second_geo = 'GIMGO'
        else:
            raise ValueError('Unknown dataset group %s' % ds_info['dataset_groups'][0])
        return prime_geo, second_geo

    def get_right_geo_fhs(self, dsid, fhs):
        """Find the right geographical file handlers for given dataset ID *dsid*."""
        ds_info = self.all_ids[dsid]
        prime_geo, second_geo = self._get_primary_secondary_geo_groups(ds_info)
        desired, other = split_desired_other(fhs, prime_geo, second_geo)
        if desired:
            try:
                ds_info['dataset_groups'].remove(second_geo)
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
                        "'%s'", ds_info['file_type'], dsid['name'])
        else:
            if len(set(ds_info['dataset_groups']) & {'GITCO', 'GIMGO', 'GMTCO', 'GMODO'}) > 1:
                fhs = self.get_right_geo_fhs(dsid, fhs)
            return fhs

    def _get_coordinates_for_dataset_key(self, dsid):
        """Get the coordinate dataset keys for `dsid`.

        Wraps the base class method in order to load geolocation files
        from the geo reference attribute in the datasets file.
        """
        coords = super()._get_coordinates_for_dataset_key(dsid)
        for c_id in coords:
            c_info = self.all_ids[c_id]  # c_info['dataset_groups'] should be a list of 2 elements
            self._get_file_handlers(c_id)
            prime_geo, second_geo = self._geo_dataset_groups(c_info)
            if prime_geo is None:
                continue

            # check the dataset file for the geolocation filename
            geo_filenames = self._load_filenames_from_geo_ref(dsid)
            self._create_new_geo_file_handlers(geo_filenames)
            self._remove_not_loaded_geo_dataset_group(c_info['dataset_groups'], prime_geo, second_geo)

        return coords

    def _geo_dataset_groups(self, c_info):
        if len(c_info['dataset_groups']) == 1:  # filtering already done
            return None, None
        try:
            prime_geo, second_geo = self._get_primary_secondary_geo_groups(c_info)
            return prime_geo, second_geo
        except ValueError:  # DNB
            return None, None

    def _create_new_geo_file_handlers(self, geo_filenames):
        existing_filenames = set([fh.filename for fh in self.file_handlers['generic_file']])
        geo_filenames = set(geo_filenames) - existing_filenames
        self.create_filehandlers(geo_filenames)

    def _remove_not_loaded_geo_dataset_group(self, c_dataset_groups, prime_geo, second_geo):
        all_fhs = self.file_handlers['generic_file']
        desired, other = split_desired_other(all_fhs, prime_geo, second_geo)
        group_to_remove = second_geo if desired else prime_geo
        c_dataset_groups.remove(group_to_remove)
