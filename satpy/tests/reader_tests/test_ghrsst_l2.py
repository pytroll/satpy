#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018, 2022 Satpy developers
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
"""Module for testing the satpy.readers.ghrsst_l2 module."""

import os
import tarfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from satpy.readers.ghrsst_l2 import GHRSSTL2FileHandler


class TestGHRSSTL2Reader:
    """Test Sentinel-3 SST L2 reader."""

    def setup_method(self, tmp_path):
        """Create a fake osisaf ghrsst dataset."""
        self.base_data = np.array(([-32768, 1135, 1125], [1138, 1128, 1080]))
        self.lon_data = np.array(([-13.43, 1.56, 11.25], [-11.38, 1.28, 10.80]))
        self.lat_data = np.array(([43.43, 55.56, 61.25], [41.38, 50.28, 60.80]))
        self.lon = xr.DataArray(
            self.lon_data,
            dims=('nj', 'ni'),
            attrs={'standard_name': 'longitude',
                   'units': 'degrees_east',
                   }
        )
        self.lat = xr.DataArray(
            self.lat_data,
            dims=('nj', 'ni'),
            attrs={'standard_name': 'latitude',
                   'units': 'degrees_north',
                   }
        )
        self.sst = xr.DataArray(
            self.base_data,
            dims=('nj', 'ni'),
            attrs={'scale_factor': 0.01, 'add_offset': 273.15,
                   '_FillValue': -32768, 'units': 'kelvin',
                   }
        )
        self.fake_dataset = xr.Dataset(
            data_vars={
                'sea_surface_temperature': self.sst,
                'longitude': self.lon,
                'latitude': self.lat,
            },
            attrs={
                "start_time": "20220321T112640Z",
                "stop_time": "20220321T145711Z",
                "platform": 'NOAA20',
                "sensor": "VIIRS",
            },
        )

    def _create_tarfile_with_testdata(self, mypath):
        """Create a 'fake' testdata set in a tar file."""
        slstr_fakename = "S3A_SL_2_WST_MAR_O_NR_003.SEN3"
        tarfile_fakename = "S3A_SL_2_WST_MAR_O_NR_003.SEN3.tar"

        slstrdir = mypath / slstr_fakename
        slstrdir.mkdir(parents=True, exist_ok=True)
        tarfile_path = mypath / tarfile_fakename

        ncfilename = slstrdir / 'L2P_GHRSST-SSTskin-202204131200.nc'
        self.fake_dataset.to_netcdf(os.fspath(ncfilename))
        xmlfile_path = slstrdir / 'xfdumanifest.xml'
        xmlfile_path.touch()

        with tarfile.open(name=tarfile_path, mode='w') as tar:
            tar.add(os.fspath(ncfilename), arcname=Path(slstr_fakename) / ncfilename.name)
            tar.add(os.fspath(xmlfile_path), arcname=Path(slstr_fakename) / xmlfile_path.name)

        return tarfile_path

    def test_instantiate_single_netcdf_file(self, tmp_path):
        """Test initialization of file handlers - given a single netCDF file."""
        filename_info = {}
        tmp_filepath = tmp_path / 'fake_dataset.nc'
        self.fake_dataset.to_netcdf(os.fspath(tmp_filepath))

        GHRSSTL2FileHandler(os.fspath(tmp_filepath), filename_info, None)

    def test_instantiate_tarfile(self, tmp_path):
        """Test initialization of file handlers - given a tar file as in the case of the SAFE format."""
        filename_info = {}
        tarfile_path = self._create_tarfile_with_testdata(tmp_path)

        GHRSSTL2FileHandler(os.fspath(tarfile_path), filename_info, None)

    def test_get_dataset(self, tmp_path):
        """Test retrieval of datasets."""
        filename_info = {}
        tmp_filepath = tmp_path / 'fake_dataset.nc'
        self.fake_dataset.to_netcdf(os.fspath(tmp_filepath))

        test = GHRSSTL2FileHandler(os.fspath(tmp_filepath), filename_info, None)

        test.get_dataset('longitude', {'standard_name': 'longitude'})
        test.get_dataset('latitude', {'standard_name': 'latitude'})
        test.get_dataset('sea_surface_temperature', {'standard_name': 'sea_surface_temperature'})

        with pytest.raises(KeyError):
            test.get_dataset('erroneous dataset', {'standard_name': 'erroneous dataset'})

    def test_get_sensor(self, tmp_path):
        """Test retrieval of the sensor name from the netCDF file."""
        dt_valid = datetime(2022, 3, 21, 11, 26, 40)  # 202203211200Z
        filename_info = {'field_type': 'NARSST', 'generating_centre': 'FRA_',
                         'satid': 'NOAA20_', 'valid_time': dt_valid}

        tmp_filepath = tmp_path / 'fake_dataset.nc'
        self.fake_dataset.to_netcdf(os.fspath(tmp_filepath))

        test = GHRSSTL2FileHandler(os.fspath(tmp_filepath), filename_info, None)
        assert test.sensor == 'viirs'

    def test_get_start_and_end_times(self, tmp_path):
        """Test retrieval of the sensor name from the netCDF file."""
        dt_valid = datetime(2022, 3, 21, 11, 26, 40)  # 202203211200Z
        good_start_time = datetime(2022, 3, 21, 11, 26, 40)  # 20220321T112640Z
        good_stop_time = datetime(2022, 3, 21, 14, 57, 11)  # 20220321T145711Z

        filename_info = {'field_type': 'NARSST', 'generating_centre': 'FRA_',
                         'satid': 'NOAA20_', 'valid_time': dt_valid}

        tmp_filepath = tmp_path / 'fake_dataset.nc'
        self.fake_dataset.to_netcdf(os.fspath(tmp_filepath))

        test = GHRSSTL2FileHandler(os.fspath(tmp_filepath), filename_info, None)

        assert test.start_time == good_start_time
        assert test.end_time == good_stop_time
