#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2018 Satpy developers
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
"""Module for testing the satpy.readers.viirs_l1b module."""

import os
from datetime import datetime, timedelta
from unittest import mock

import numpy as np

from satpy.tests.reader_tests.test_netcdf_utils import FakeNetCDF4FileHandler
from satpy.tests.utils import convert_file_content_to_data_array

DEFAULT_FILE_DTYPE = np.uint16
DEFAULT_FILE_SHAPE = (10, 300)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)
DEFAULT_FILE_FACTORS = np.array([2.0, 1.0], dtype=np.float32)
DEFAULT_LAT_DATA = np.linspace(45, 65, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LAT_DATA = np.repeat([DEFAULT_LAT_DATA], DEFAULT_FILE_SHAPE[0], axis=0)
DEFAULT_LON_DATA = np.linspace(5, 45, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LON_DATA = np.repeat([DEFAULT_LON_DATA], DEFAULT_FILE_SHAPE[0], axis=0)


class FakeNetCDF4FileHandlerDay(FakeNetCDF4FileHandler):
    """Swap-in NetCDF4 File Handler."""

    M_REFL_BANDS = [f"M{band_num:02d}" for band_num in range(1, 12)]
    M_BT_BANDS = [f"M{band_num:02d}" for band_num in range(12, 17)]
    M_BANDS = M_REFL_BANDS + M_BT_BANDS
    I_REFL_BANDS = [f"I{band_num:02d}" for band_num in range(1, 4)]
    I_BT_BANDS = [f"I{band_num:02d}" for band_num in range(4, 6)]
    I_BANDS = I_REFL_BANDS + I_BT_BANDS

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        dt = filename_info.get('start_time', datetime(2016, 1, 1, 12, 0, 0))
        file_type = filename[:5].lower()
        num_lines = DEFAULT_FILE_SHAPE[0]
        num_pixels = DEFAULT_FILE_SHAPE[1]
        num_scans = 5
        num_luts = DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1]
        file_content = {
            '/dimension/number_of_scans': num_scans,
            '/dimension/number_of_lines': num_lines,
            '/dimension/number_of_pixels': num_pixels,
            '/dimension/number_of_LUT_values': num_luts,
            '/attr/time_coverage_start': dt.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            '/attr/time_coverage_end': (dt + timedelta(minutes=6)).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            '/attr/orbit_number': 26384,
            '/attr/instrument': 'VIIRS',
            '/attr/platform': 'Suomi-NPP',
        }
        self._fill_contents_with_default_data(file_content, file_type)
        self._set_dataset_specific_metadata(file_content)
        convert_file_content_to_data_array(file_content)
        return file_content

    def _fill_contents_with_default_data(self, file_content, file_type):
        """Fill file contents with default data."""
        if file_type.startswith('vgeo'):
            file_content['/attr/OrbitNumber'] = file_content.pop('/attr/orbit_number')
            file_content['geolocation_data/latitude'] = DEFAULT_LAT_DATA
            file_content['geolocation_data/longitude'] = DEFAULT_LON_DATA
            file_content['geolocation_data/solar_zenith'] = DEFAULT_LON_DATA
            file_content['geolocation_data/solar_azimuth'] = DEFAULT_LON_DATA
            file_content['geolocation_data/sensor_zenith'] = DEFAULT_LON_DATA
            file_content['geolocation_data/sensor_azimuth'] = DEFAULT_LON_DATA
            if file_type.endswith('d'):
                file_content['geolocation_data/lunar_zenith'] = DEFAULT_LON_DATA
                file_content['geolocation_data/lunar_azimuth'] = DEFAULT_LON_DATA
        elif file_type == 'vl1bm':
            for m_band in self.M_BANDS:
                file_content[f'observation_data/{m_band}'] = DEFAULT_FILE_DATA
        elif file_type == 'vl1bi':
            for i_band in self.I_BANDS:
                file_content[f'observation_data/{i_band}'] = DEFAULT_FILE_DATA
        elif file_type == 'vl1bd':
            file_content['observation_data/DNB_observations'] = DEFAULT_FILE_DATA
            file_content['observation_data/DNB_observations/attr/units'] = 'Watts/cm^2/steradian'

    @staticmethod
    def _set_dataset_specific_metadata(file_content):
        """Set dataset-specific metadata."""
        for k in list(file_content.keys()):
            if not k.startswith('observation_data') and not k.startswith('geolocation_data'):
                continue
            file_content[k + '/shape'] = DEFAULT_FILE_SHAPE
            if k[-3:] in ['M12', 'M13', 'M14', 'M15', 'M16', 'I04', 'I05']:
                file_content[k + '_brightness_temperature_lut'] = DEFAULT_FILE_DATA.ravel()
                file_content[k + '_brightness_temperature_lut/attr/units'] = 'Kelvin'
                file_content[k + '_brightness_temperature_lut/attr/valid_min'] = 0
                file_content[k + '_brightness_temperature_lut/attr/valid_max'] = 65534
                file_content[k + '_brightness_temperature_lut/attr/_FillValue'] = 65535
                file_content[k + '/attr/units'] = 'Watts/meter^2/steradian/micrometer'
            elif k[-3:] in ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08',
                            'M09', 'M10', 'M11', 'I01', 'I02', 'I03']:
                file_content[k + '/attr/radiance_units'] = 'Watts/meter^2/steradian/micrometer'
                file_content[k + '/attr/radiance_scale_factor'] = 1.1
                file_content[k + '/attr/radiance_add_offset'] = 0.1
            elif k.endswith('longitude'):
                file_content[k + '/attr/units'] = 'degrees_east'
            elif k.endswith('latitude'):
                file_content[k + '/attr/units'] = 'degrees_north'
            elif k.endswith('zenith') or k.endswith('azimuth'):
                file_content[k + '/attr/units'] = 'degrees'
            file_content[k + '/attr/valid_min'] = 0
            file_content[k + '/attr/valid_max'] = 65534
            file_content[k + '/attr/_FillValue'] = 65535
            file_content[k + '/attr/scale_factor'] = 1.1
            file_content[k + '/attr/add_offset'] = 0.1


class FakeNetCDF4FileHandlerNight(FakeNetCDF4FileHandlerDay):
    """Same as the day file handler, but some day-only bands are missing.

    This matches what happens in real world files where reflectance bands
    are removed in night data to save space.

    """

    M_BANDS = FakeNetCDF4FileHandlerDay.M_BT_BANDS
    I_BANDS = FakeNetCDF4FileHandlerDay.I_BT_BANDS


class TestVIIRSL1BReaderDay:
    """Test VIIRS L1B Reader."""

    yaml_file = "viirs_l1b.yaml"
    fake_cls = FakeNetCDF4FileHandlerDay
    has_reflectance_bands = True

    def setup_method(self):
        """Wrap NetCDF4 file handler with our own fake handler."""
        from satpy._config import config_search_paths
        from satpy.readers.viirs_l1b import VIIRSL1BFileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(VIIRSL1BFileHandler, '__bases__', (self.fake_cls,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def teardown_method(self):
        """Stop wrapping the NetCDF4 file handler."""
        self.p.stop()

    def test_init(self):
        """Test basic init with no extra parameters."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'VL1BM_snpp_d20161130_t012400_c20161130054822.nc',
        ])
        assert len(loadables) == 1
        r.create_filehandlers(loadables)
        # make sure we have some files
        assert r.file_handlers

    def test_available_datasets_m_bands(self):
        """Test available datasets for M band files."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'VL1BM_snpp_d20161130_t012400_c20161130054822.nc',
            'VGEOM_snpp_d20161130_t012400_c20161130054822.nc',
        ])
        r.create_filehandlers(loadables)
        avail_names = r.available_dataset_names
        angles = {"satellite_azimuth_angle", "satellite_zenith_angle", "solar_azimuth_angle", "solar_zenith_angle"}
        geo = {"m_lon", "m_lat"}
        assert set(avail_names) == set(self.fake_cls.M_BANDS) | angles | geo

    def test_load_every_m_band_bt(self):
        """Test loading all M band brightness temperatures."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'VL1BM_snpp_d20161130_t012400_c20161130054822.nc',
            'VGEOM_snpp_d20161130_t012400_c20161130054822.nc',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['M12',
                           'M13',
                           'M14',
                           'M15',
                           'M16'])
        assert len(datasets) == 5
        for v in datasets.values():
            assert v.attrs['calibration'] == 'brightness_temperature'
            assert v.attrs['units'] == 'K'
            assert v.attrs['rows_per_scan'] == 2
            assert v.attrs['area'].lons.attrs['rows_per_scan'] == 2
            assert v.attrs['area'].lats.attrs['rows_per_scan'] == 2
            assert v.attrs['sensor'] == "viirs"

    def test_load_every_m_band_refl(self):
        """Test loading all M band reflectances."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'VL1BM_snpp_d20161130_t012400_c20161130054822.nc',
            'VGEOM_snpp_d20161130_t012400_c20161130054822.nc',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['M01',
                           'M02',
                           'M03',
                           'M04',
                           'M05',
                           'M06',
                           'M07',
                           'M08',
                           'M09',
                           'M10',
                           'M11'])
        assert len(datasets) == (11 if self.has_reflectance_bands else 0)
        for v in datasets.values():
            assert v.attrs['calibration'] == 'reflectance'
            assert v.attrs['units'] == '%'
            assert v.attrs['rows_per_scan'] == 2
            assert v.attrs['area'].lons.attrs['rows_per_scan'] == 2
            assert v.attrs['area'].lats.attrs['rows_per_scan'] == 2
            assert v.attrs['sensor'] == "viirs"

    def test_load_every_m_band_rad(self):
        """Test loading all M bands as radiances."""
        from satpy.readers import load_reader
        from satpy.tests.utils import make_dataid
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'VL1BM_snpp_d20161130_t012400_c20161130054822.nc',
            'VGEOM_snpp_d20161130_t012400_c20161130054822.nc',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load([make_dataid(name='M01', calibration='radiance'),
                           make_dataid(name='M02', calibration='radiance'),
                           make_dataid(name='M03', calibration='radiance'),
                           make_dataid(name='M04', calibration='radiance'),
                           make_dataid(name='M05', calibration='radiance'),
                           make_dataid(name='M06', calibration='radiance'),
                           make_dataid(name='M07', calibration='radiance'),
                           make_dataid(name='M08', calibration='radiance'),
                           make_dataid(name='M09', calibration='radiance'),
                           make_dataid(name='M10', calibration='radiance'),
                           make_dataid(name='M11', calibration='radiance'),
                           make_dataid(name='M12', calibration='radiance'),
                           make_dataid(name='M13', calibration='radiance'),
                           make_dataid(name='M14', calibration='radiance'),
                           make_dataid(name='M15', calibration='radiance'),
                           make_dataid(name='M16', calibration='radiance')])
        assert len(datasets) == (16 if self.has_reflectance_bands else 5)
        for v in datasets.values():
            assert v.attrs['calibration'] == 'radiance'
            assert v.attrs['units'] == 'W m-2 um-1 sr-1'
            assert v.attrs['rows_per_scan'] == 2
            assert v.attrs['area'].lons.attrs['rows_per_scan'] == 2
            assert v.attrs['area'].lats.attrs['rows_per_scan'] == 2
            assert v.attrs['sensor'] == "viirs"

    def test_load_i_band_angles(self):
        """Test loading all M bands as radiances."""
        from satpy.readers import load_reader
        from satpy.tests.utils import make_dataid
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'VL1BI_snpp_d20161130_t012400_c20161130054822.nc',
            'VL1BM_snpp_d20161130_t012400_c20161130054822.nc',
            'VGEOI_snpp_d20161130_t012400_c20161130054822.nc',
            'VGEOM_snpp_d20161130_t012400_c20161130054822.nc',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load([
            make_dataid(name='satellite_zenith_angle'),
            make_dataid(name='satellite_azimuth_angle'),
            make_dataid(name='solar_azimuth_angle'),
            make_dataid(name='solar_zenith_angle'),
        ])
        assert len(datasets) == 4
        for v in datasets.values():
            assert v.attrs['resolution'] == 371
            assert v.attrs['sensor'] == "viirs"

    def test_load_dnb_radiance(self):
        """Test loading the main DNB dataset."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'VL1BD_snpp_d20161130_t012400_c20161130054822.nc',
            'VGEOD_snpp_d20161130_t012400_c20161130054822.nc',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['DNB'])
        assert len(datasets) == 1
        for v in datasets.values():
            assert v.attrs['calibration'] == 'radiance'
            assert v.attrs['units'] == 'W m-2 sr-1'
            assert v.attrs['rows_per_scan'] == 2
            assert v.attrs['area'].lons.attrs['rows_per_scan'] == 2
            assert v.attrs['area'].lats.attrs['rows_per_scan'] == 2
            assert v.attrs['sensor'] == "viirs"

    def test_load_dnb_angles(self):
        """Test loading all DNB angle datasets."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'VL1BD_snpp_d20161130_t012400_c20161130054822.nc',
            'VGEOD_snpp_d20161130_t012400_c20161130054822.nc',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['dnb_solar_zenith_angle',
                           'dnb_solar_azimuth_angle',
                           'dnb_satellite_zenith_angle',
                           'dnb_satellite_azimuth_angle',
                           'dnb_lunar_zenith_angle',
                           'dnb_lunar_azimuth_angle',
                           ])
        assert len(datasets) == 6
        for v in datasets.values():
            assert v.attrs['units'] == 'degrees'
            assert v.attrs['rows_per_scan'] == 2
            assert v.attrs['area'].lons.attrs['rows_per_scan'] == 2
            assert v.attrs['area'].lats.attrs['rows_per_scan'] == 2
            assert v.attrs['sensor'] == "viirs"


class TestVIIRSL1BReaderDayNight(TestVIIRSL1BReaderDay):
    """Test VIIRS L1b with night data.

    Night data files don't have reflectance bands in them.

    """

    fake_cls = FakeNetCDF4FileHandlerNight
    has_reflectance_bands = False
