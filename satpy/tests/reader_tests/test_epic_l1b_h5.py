#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 Satpy developers
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
"""The epic_l1b_h5 reader tests package."""

import os
import tempfile

import h5py
import numpy as np
import pytest

from satpy.readers.epic_l1b_h5 import CALIB_COEFS

b317_data = np.random.uniform(low=0, high=5200, size=(100, 100))
b688_data = np.random.uniform(low=0, high=5200, size=(100, 100))
sza_data = np.random.uniform(low=0, high=100, size=(100, 100))
vaa_data = np.random.uniform(low=-180, high=180, size=(100, 100))
lon_data = np.random.uniform(low=-90, high=90, size=(100, 100))
lat_data = np.random.uniform(low=-180, high=180, size=(100, 100))
mas_data = np.random.choice([0, 1], size=(100, 100))


@pytest.fixture()
def setup_hdf5_file(tmp_path):
    """Create temp hdf5 files."""
    fn = tmp_path / "epic_1b_20150613120251_03.h5"
    make_fake_hdf_epic(fn)
    return fn


def make_fake_hdf_epic(fname):
    """Make a fake HDF5 file for EPIC data testing."""
    fid = h5py.File(fname, 'w')
    g1 = fid.create_group('Band317nm')
    g1.create_dataset('Image', shape=(100, 100), dtype=np.float32, data=b317_data)
    g2 = fid.create_group('Band688nm')
    g2.create_dataset('Image', shape=(100, 100), dtype=np.float32, data=b688_data)
    g3 = g2.create_group('Geolocation')
    g4 = g3.create_group('Earth')
    g4.create_dataset('SunAngleZenith', shape=(100, 100), dtype=np.float32, data=sza_data)
    g4.create_dataset('ViewAngleAzimuth', shape=(100, 100), dtype=np.float32, data=vaa_data)
    g4.create_dataset('Mask', shape=(100, 100), dtype=int, data=mas_data)
    g4.create_dataset('Latitude', shape=(100, 100), dtype=np.float32, data=lat_data)
    g4.create_dataset('Longitude', shape=(100, 100), dtype=np.float32, data=lon_data)

    fid.attrs.create('begin_time', '2015-06-13 12:00:37')
    fid.attrs.create('end_time', '2015-06-13 12:05:01')
    fid.close()


class TestEPICL1bReader:
    """Test the EPIC L1b HDF5 reader."""

    def _setup_h5(self, setup_hdf5_file):
        """Initialise reader for the tests."""
        from satpy.readers import load_reader
        test_reader = load_reader(self.reader_configs)
        loadables = test_reader.select_files_from_pathnames([setup_hdf5_file])
        test_reader.create_filehandlers(loadables)

        return test_reader

    def setup_method(self):
        """Set up the tests."""
        from satpy._config import config_search_paths

        self.yaml_file = "epic_l1b_h5.yaml"

        self.filename_test = os.path.join(
            tempfile.gettempdir(),
            "epic_1b_20150613120251_03.h5",
        )
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))

    def test_times(self, setup_hdf5_file):
        """Test start and end times load properly."""
        from datetime import datetime

        test_reader = self._setup_h5(setup_hdf5_file)
        assert test_reader.start_time == datetime(2015, 6, 13, 12, 0, 37)
        assert test_reader.end_time == datetime(2015, 6, 13, 12, 5, 1)

    def test_counts_calibration(self, setup_hdf5_file):
        """Test that data is correctly calibrated."""
        from satpy.tests.utils import make_dsq

        test_reader = self._setup_h5(setup_hdf5_file)
        # Test counts calibration
        ds = test_reader.load([make_dsq(name='B317', calibration='counts')])
        np.testing.assert_allclose(ds['B317'].data, b317_data)

    def test_refl_calibration(self, setup_hdf5_file):
        """Test that data is correctly calibrated into reflectances."""
        from satpy.tests.utils import make_dsq

        test_reader = self._setup_h5(setup_hdf5_file)

        # Test conversion to reflectance
        ds = test_reader.load([make_dsq(name='B317', calibration='reflectance')])
        np.testing.assert_allclose(ds['B317'].data, b317_data * CALIB_COEFS['B317'] * 100., rtol=1e-5)

    def test_bad_calibration(self, setup_hdf5_file):
        """Test that error is raised if a bad calibration is used."""
        from satpy.tests.utils import make_dsq

        test_reader = self._setup_h5(setup_hdf5_file)

        # Test nonsense calibration
        with pytest.raises(KeyError):
            test_reader.load([make_dsq(name='B317', calibration='potatoes')])

    def test_load_ancillary(self, setup_hdf5_file):
        """Test that ancillary datasets load correctly."""
        from satpy.tests.utils import make_dsq

        test_reader = self._setup_h5(setup_hdf5_file)

        # Load sza
        ds = test_reader.load([make_dsq(name='solar_zenith_angle'),
                               make_dsq(name='satellite_azimuth_angle'),
                               make_dsq(name='latitude'),
                               make_dsq(name='longitude'),
                               make_dsq(name='earth_mask')])

        np.testing.assert_allclose(ds['solar_zenith_angle'].data, sza_data)
        np.testing.assert_allclose(ds['satellite_azimuth_angle'].data, vaa_data)
        np.testing.assert_allclose(ds['latitude'].data, lat_data)
        np.testing.assert_allclose(ds['longitude'].data, lon_data)
        np.testing.assert_allclose(ds['earth_mask'].data, mas_data)
