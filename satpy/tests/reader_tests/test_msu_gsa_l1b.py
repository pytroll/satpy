#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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
"""Tests for the 'msu_gsa_l1b' reader."""
import os
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from satpy.tests.reader_tests.test_hdf5_utils import FakeHDF5FileHandler
from satpy.tests.utils import make_dataid

SOLCONST = '273.59'


class FakeHDF5FileHandler2(FakeHDF5FileHandler):
    """Swap-in HDF5 File Handler."""

    def _get_data(self, num_scans, num_cols):
        data = {
            'Data/resolution_1km/Solar_Zenith_Angle':
                xr.DataArray(
                    da.ones((num_scans*4, num_cols*4), chunks=1024,
                            dtype=np.uint16),
                    attrs={
                        'scale': 0.01, 'offset': 0., 'fill_value': -999.
                    },
                    dims=('x', 'y')),
            'Geolocation/resolution_1km/Latitude':
                xr.DataArray(
                    da.ones((num_scans*4, num_cols*4), chunks=1024,
                            dtype=np.uint16),
                    attrs={
                        'scale': 0.01, 'offset': 0., 'fill_value': -999.
                    },
                    dims=('x', 'y')),
            'Geolocation/resolution_1km/Longitude':
                xr.DataArray(
                    da.ones((num_scans*4, num_cols*4), chunks=1024,
                            dtype=np.uint16),
                    attrs={
                        'scale': 0.01, 'offset': 0., 'fill_value': -999.
                    },
                    dims=('x', 'y')),
            'Data/resolution_1km/Radiance_01':
                xr.DataArray(
                    da.ones((num_scans*4, num_cols*4), chunks=1024,
                            dtype=np.uint16),
                    attrs={
                        'scale': 0.01, 'offset': 0., 'fill_value': -999., 'F_solar_constant': SOLCONST
                    },
                    dims=('x', 'y')),
            'Data/resolution_4km/Solar_Zenith_Angle':
                xr.DataArray(
                    da.ones((num_scans, num_cols), chunks=1024,
                            dtype=np.uint16),
                    attrs={
                        'scale': 0.01, 'offset': 0., 'fill_value': -999.
                    },
                    dims=('x', 'y')),
            'Geolocation/resolution_4km/Latitude':
                xr.DataArray(
                    da.ones((num_scans, num_cols), chunks=1024,
                            dtype=np.uint16),
                    attrs={
                        'scale': 0.01, 'offset': 0., 'fill_value': -999.
                    },
                    dims=('x', 'y')),
            'Geolocation/resolution_4km/Longitude':
                xr.DataArray(
                    da.ones((num_scans, num_cols), chunks=1024,
                            dtype=np.uint16),
                    attrs={
                        'scale': 0.01, 'offset': 0., 'fill_value': -999.
                    },
                    dims=('x', 'y')),
            'Data/resolution_4km/Brightness_Temperature_09':
                xr.DataArray(
                    da.ones((num_scans, num_cols), chunks=1024,
                            dtype=np.uint16),
                    attrs={
                        'scale': 0.01, 'offset': 0., 'fill_value': -999.
                    },
                    dims=('x', 'y')),
        }
        return data

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        num_scans = 20
        num_cols = 2048
        global_attrs = {
            '/attr/timestamp_without_timezone': '2022-01-13T12:45:00',
            '/attr/satellite_observation_point_height': '38500.0',
            '/attr/satellite_observation_point_latitude': '71.25',
            '/attr/satellite_observation_point_longitude': '21.44',
        }

        data = self._get_data(num_scans, num_cols)

        test_content = {}
        test_content.update(global_attrs)
        test_content.update(data)
        return test_content


class TestMSUGSABReader:
    """Test MSU GS/A L1B Reader."""

    yaml_file = "msu_gsa_l1b.yaml"

    def setup(self):
        """Wrap HDF5 file handler with our own fake handler."""
        from satpy._config import config_search_paths
        from satpy.readers import load_reader
        from satpy.readers.msu_gsa_l1b import MSUGSAFileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(MSUGSAFileHandler, '__bases__', (FakeHDF5FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

        filenames = ['ArcticaM1_202201131245.h5']
        self.reader = load_reader(self.reader_configs)
        files = self.reader.select_files_from_pathnames(filenames)
        self.reader.create_filehandlers(files)

    def teardown(self):
        """Stop wrapping the HDF5 file handler."""
        self.p.stop()

    def test_irbt(self):
        """Test retrieval in brightness temperature."""
        ds_ids = [make_dataid(name='C09', calibration='brightness_temperature')]
        res = self.reader.load(ds_ids)
        assert 'C09' in res
        assert res['C09'].attrs['calibration'] == 'brightness_temperature'
        assert res['C09'].attrs['platform_name'] == 'Arctica-M-N1'
        assert res['C09'].attrs['sat_latitude'] == 71.25
        assert res['C09'].attrs['sat_longitude'] == 21.44
        assert res['C09'].attrs['sat_altitude'] == 38500.
        assert res['C09'].attrs['resolution'] == 4000

    def test_nocounts(self):
        """Test we can't get IR or VIS data as counts."""
        ds_ids = [make_dataid(name='C01', calibration='counts')]
        with pytest.raises(KeyError):
            self.reader.load(ds_ids)

        ds_ids = [make_dataid(name='C09', calibration='counts')]
        with pytest.raises(KeyError):
            self.reader.load(ds_ids)

    def test_vis_cal(self):
        """Test that we can retrieve VIS data as both radiance and reflectance."""
        ds_ids = [make_dataid(name='C01', calibration='radiance')]
        res = self.reader.load(ds_ids)
        rad = res['C01'].data
        ds_ids = [make_dataid(name='C01', calibration='reflectance')]
        res = self.reader.load(ds_ids)
        refl = res['C01'].data

        # Check the RAD->REFL conversion
        np.testing.assert_allclose(100 * np.pi * rad / float(SOLCONST), refl)
