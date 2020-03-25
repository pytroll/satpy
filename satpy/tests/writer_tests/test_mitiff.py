#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018-2020 Satpy developers
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
"""Tests for the mitiff writer.

Based on the test for geotiff writer

"""
import unittest


class TestMITIFFWriter(unittest.TestCase):
    """Test the MITIFF Writer class."""

    def setUp(self):
        """Create temporary directory to save files to"""
        import tempfile
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary directory created for a test"""
        try:
            import shutil
            shutil.rmtree(self.base_dir, ignore_errors=True)
        except OSError:
            pass

    def _get_test_datasets(self):
        """Helper function to create a datasets list."""
        import xarray as xr
        import dask.array as da
        from datetime import datetime
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj4_str_to_dict('+proj=stere +datum=WGS84 +ellps=WGS84 '
                              '+lon_0=0. +lat_0=90 +lat_ts=60 +units=km'),
            100,
            200,
            (-1000., -1500., 1000., 1500.),
        )

        ds1 = xr.DataArray(
            da.zeros((100, 200), chunks=50),
            dims=('y', 'x'),
            attrs={'name': '1',
                   'start_time': datetime.utcnow(),
                   'platform_name': "TEST_PLATFORM_NAME",
                   'sensor': 'TEST_SENSOR_NAME',
                   'area': area_def,
                   'prerequisites': ['1'],
                   'calibration': 'reflectance',
                   'metadata_requirements': {
                       'order': ['1'],
                       'config': {
                           '1': {'alias': '1-VIS0.63',
                                 'calibration': 'reflectance',
                                 'min-val': '0',
                                 'max-val': '100'},
                       },
                       'translate': {'1': '1',
                                     },
                       'file_pattern': '1_{start_time:%Y%m%d_%H%M%S}.mitiff'
                   }}
        )
        ds2 = xr.DataArray(
            da.zeros((100, 200), chunks=50),
            dims=('y', 'x'),
            attrs={'name': '4',
                   'start_time': datetime.utcnow(),
                   'platform_name': "TEST_PLATFORM_NAME",
                   'sensor': 'TEST_SENSOR_NAME',
                   'area': area_def,
                   'prerequisites': ['4'],
                   'calibration': 'brightness_temperature',
                   'metadata_requirements': {
                       'order': ['4'],
                       'config': {
                           '4': {'alias': '4-IR10.8',
                                 'calibration': 'brightness_temperature',
                                 'min-val': '-150',
                                 'max-val': '50'},
                       },
                       'translate': {'4': '4',
                                     },
                       'file_pattern': '4_{start_time:%Y%m%d_%H%M%S}.mitiff'}
                   }
        )
        return [ds1, ds2]

    def _get_test_datasets_sensor_set(self):
        """Helper function to create a datasets list."""
        import xarray as xr
        import dask.array as da
        from datetime import datetime
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj4_str_to_dict('+proj=stere +datum=WGS84 +ellps=WGS84 '
                              '+lon_0=0. +lat_0=90 +lat_ts=60 +units=km'),
            100,
            200,
            (-1000., -1500., 1000., 1500.),
        )

        ds1 = xr.DataArray(
            da.zeros((100, 200), chunks=50),
            dims=('y', 'x'),
            attrs={'name': '1',
                   'start_time': datetime.utcnow(),
                   'platform_name': "TEST_PLATFORM_NAME",
                   'sensor': set('TEST_SENSOR_NAME'),
                   'area': area_def,
                   'prerequisites': ['1'],
                   'calibration': 'reflectance',
                   'metadata_requirements': {
                       'order': ['1'],
                       'config': {
                           '1': {'alias': '1-VIS0.63',
                                 'calibration': 'reflectance',
                                 'min-val': '0',
                                 'max-val': '100'},
                       },
                       'translate': {'1': '1',
                                     },
                       'file_pattern': '1_{start_time:%Y%m%d_%H%M%S}.mitiff'
                   }}
        )
        ds2 = xr.DataArray(
            da.zeros((100, 200), chunks=50),
            dims=('y', 'x'),
            attrs={'name': '4',
                   'start_time': datetime.utcnow(),
                   'platform_name': "TEST_PLATFORM_NAME",
                   'sensor': set('TEST_SENSOR_NAME'),
                   'area': area_def,
                   'prerequisites': ['4'],
                   'calibration': 'brightness_temperature',
                   'metadata_requirements': {
                       'order': ['4'],
                       'config': {
                           '4': {'alias': '4-IR10.8',
                                 'calibration': 'brightness_temperature',
                                 'min-val': '-150',
                                 'max-val': '50'},
                       },
                       'translate': {'4': '4',
                                     },
                       'file_pattern': '4_{start_time:%Y%m%d_%H%M%S}.mitiff'}
                   }
        )
        return [ds1, ds2]

    def _get_test_dataset(self, bands=3):
        """Helper function to create a single test dataset."""
        import xarray as xr
        import dask.array as da
        from datetime import datetime
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj4_str_to_dict('+proj=stere +datum=WGS84 +ellps=WGS84 '
                              '+lon_0=0. +lat_0=90 +lat_ts=60 +units=km'),
            100,
            200,
            (-1000., -1500., 1000., 1500.),
        )

        ds1 = xr.DataArray(
            da.zeros((bands, 100, 200), chunks=50),
            dims=('bands', 'y', 'x'),
            attrs={'name': 'test',
                   'start_time': datetime.utcnow(),
                   'platform_name': "TEST_PLATFORM_NAME",
                   'sensor': 'TEST_SENSOR_NAME',
                   'area': area_def,
                   'prerequisites': ['1', '2', '3']}
        )
        return ds1

    def _get_test_one_dataset(self):
        """Helper function to create a single test dataset."""
        import xarray as xr
        import dask.array as da
        from datetime import datetime
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj4_str_to_dict('+proj=geos +datum=WGS84 +ellps=WGS84 '
                              '+lon_0=0. h=36000. +units=km'),
            100,
            200,
            (-1000., -1500., 1000., 1500.),
        )

        ds1 = xr.DataArray(
            da.zeros((100, 200), chunks=50),
            dims=('y', 'x'),
            attrs={'name': 'test',
                   'start_time': datetime.utcnow(),
                   'platform_name': "TEST_PLATFORM_NAME",
                   'sensor': 'avhrr',
                   'area': area_def,
                   'prerequisites': [10.8]}
        )
        return ds1

    def _get_test_one_dataset_sensor_set(self):
        """Helper function to create a single test dataset."""
        import xarray as xr
        import dask.array as da
        from datetime import datetime
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj4_str_to_dict('+proj=geos +datum=WGS84 +ellps=WGS84 '
                              '+lon_0=0. h=36000. +units=km'),
            100,
            200,
            (-1000., -1500., 1000., 1500.),
        )

        ds1 = xr.DataArray(
            da.zeros((100, 200), chunks=50),
            dims=('y', 'x'),
            attrs={'name': 'test',
                   'start_time': datetime.utcnow(),
                   'platform_name': "TEST_PLATFORM_NAME",
                   'sensor': set('avhrr'),
                   'area': area_def,
                   'prerequisites': [10.8]}
        )
        return ds1

    def _get_test_dataset_with_bad_values(self, bands=3):
        """Helper function to create a single test dataset."""
        import xarray as xr
        import numpy as np
        from datetime import datetime
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj4_str_to_dict('+proj=stere +datum=WGS84 +ellps=WGS84 '
                              '+lon_0=0. +lat_0=90 +lat_ts=60 +units=km'),
            100,
            200,
            (-1000., -1500., 1000., 1500.),
        )

        data = np.arange(-210, 790, 100).reshape((2, 5)) * 0.95
        data /= 5.605
        data[0, 0] = np.nan  # need a nan value
        data[0, 1] = 0.  # Need a 0 value

        rgb_data = np.stack([data, data, data])
        ds1 = xr.DataArray(rgb_data,
                           dims=('bands', 'y', 'x'),
                           attrs={'name': 'test',
                                  'start_time': datetime.utcnow(),
                                  'platform_name': "TEST_PLATFORM_NAME",
                                  'sensor': 'TEST_SENSOR_NAME',
                                  'area': area_def,
                                  'prerequisites': ['1', '2', '3']})
        return ds1

    def _get_test_dataset_calibration(self, bands=6):
        """Helper function to create a single test dataset."""
        import xarray as xr
        import dask.array as da
        from datetime import datetime
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        from satpy import DatasetID
        from satpy.scene import Scene
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj4_str_to_dict('+proj=stere +datum=WGS84 +ellps=WGS84 '
                              '+lon_0=0. +lat_0=90 +lat_ts=60 +units=km'),
            100,
            200,
            (-1000., -1500., 1000., 1500.),
        )

        d = [
            DatasetID(name='1', calibration='reflectance'),
            DatasetID(name='2', calibration='reflectance'),
            DatasetID(name='3', calibration='brightness_temperature'),
            DatasetID(name='4', calibration='brightness_temperature'),
            DatasetID(name='5', calibration='brightness_temperature'),
            DatasetID(name='6', calibration='reflectance')
        ]
        scene = Scene()
        scene["1"] = xr.DataArray(da.zeros((100, 200), chunks=50),
                                  dims=('y', 'x'),
                                  attrs={'calibration': 'reflectance'})
        scene["2"] = xr.DataArray(da.zeros((100, 200), chunks=50),
                                  dims=('y', 'x'),
                                  attrs={'calibration': 'reflectance'})
        scene["3"] = xr.DataArray(da.zeros((100, 200), chunks=50),
                                  dims=('y', 'x'),
                                  attrs={'calibration': 'brightness_temperature'})
        scene["4"] = xr.DataArray(da.zeros((100, 200), chunks=50),
                                  dims=('y', 'x'),
                                  attrs={'calibration': 'brightness_temperature'})
        scene["5"] = xr.DataArray(da.zeros((100, 200), chunks=50),
                                  dims=('y', 'x'),
                                  attrs={'calibration': 'brightness_temperature'})
        scene["6"] = xr.DataArray(da.zeros((100, 200), chunks=50),
                                  dims=('y', 'x'),
                                  attrs={'calibration': 'reflectance'})

        data = xr.concat(scene, 'bands', coords='minimal')
        bands = []
        calibration = []
        for p in scene:
            calibration.append(p.attrs['calibration'])
            bands.append(p.attrs['name'])
        data['bands'] = list(bands)
        new_attrs = {'name': 'datasets',
                     'start_time': datetime.utcnow(),
                     'platform_name': "TEST_PLATFORM_NAME",
                     'sensor': 'test-sensor',
                     'area': area_def,
                     'prerequisites': d,
                     'metadata_requirements': {
                         'order': ['1', '2', '3', '4', '5', '6'],
                         'config': {
                             '1': {'alias': '1-VIS0.63',
                                   'calibration': 'reflectance',
                                   'min-val': '0',
                                   'max-val': '100'},
                             '2': {'alias': '2-VIS0.86',
                                   'calibration': 'reflectance',
                                   'min-val': '0',
                                   'max-val': '100'},
                             '3': {'alias': '3(3B)-IR3.7',
                                   'calibration': 'brightness_temperature',
                                   'min-val': '-150',
                                   'max-val': '50'},
                             '4': {'alias': '4-IR10.8',
                                   'calibration': 'brightness_temperature',
                                   'min-val': '-150',
                                   'max-val': '50'},
                             '5': {'alias': '5-IR11.5',
                                   'calibration': 'brightness_temperature',
                                   'min-val': '-150',
                                   'max-val': '50'},
                             '6': {'alias': '6(3A)-VIS1.6',
                                   'calibration': 'reflectance',
                                   'min-val': '0',
                                   'max-val': '100'}
                         },
                         'translate': {'1': '1',
                                       '2': '2',
                                       '3': '3',
                                       '4': '4',
                                       '5': '5',
                                       '6': '6'
                                       },
                         'file_pattern': 'test-dataset-{start_time:%Y%m%d%H%M%S}.mitiff'
                     }
                     }
        ds1 = xr.DataArray(data=data.data, attrs=new_attrs,
                           dims=data.dims, coords=data.coords)
        return ds1

    def _get_test_dataset_calibration_one_dataset(self, bands=1):
        """Helper function to create a single test dataset."""
        import xarray as xr
        import dask.array as da
        from datetime import datetime
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        from satpy import DatasetID
        from satpy.scene import Scene
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj4_str_to_dict('+proj=stere +datum=WGS84 +ellps=WGS84 '
                              '+lon_0=0. +lat_0=90 +lat_ts=60 +units=km'),
            100,
            200,
            (-1000., -1500., 1000., 1500.),
        )

        d = [DatasetID(name='4', calibration='brightness_temperature')]
        scene = Scene()
        scene["4"] = xr.DataArray(da.zeros((100, 200), chunks=50),
                                  dims=('y', 'x'),
                                  attrs={'calibration': 'brightness_temperature'})

        data = scene['4']
        calibration = []
        for p in scene:
            calibration.append(p.attrs['calibration'])
        new_attrs = {'name': 'datasets',
                     'start_time': datetime.utcnow(),
                     'platform_name': "TEST_PLATFORM_NAME",
                     'sensor': 'test-sensor',
                     'area': area_def,
                     'prerequisites': d,
                     'metadata_requirements': {
                         'order': ['4'],
                         'config': {
                             '4': {'alias': 'BT',
                                   'calibration': 'brightness_temperature',
                                   'min-val': '-150',
                                   'max-val': '50'},
                         },
                         'translate': {'4': '4',
                                       },
                         'file_pattern': 'test-dataset-{start_time:%Y%m%d%H%M%S}.mitiff'
                     }
                     }
        ds1 = xr.DataArray(data=data.data, attrs=new_attrs,
                           dims=data.dims, coords=data.coords)
        return ds1

    def _get_test_dataset_three_bands_two_prereq(self, bands=3):
        """Helper function to create a single test dataset."""
        import xarray as xr
        import dask.array as da
        from datetime import datetime
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        from satpy import DatasetID
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj4_str_to_dict('+proj=stere +datum=WGS84 +ellps=WGS84 '
                              '+lon_0=0. +lat_0=90 +lat_ts=60 +units=km'),
            100,
            200,
            (-1000., -1500., 1000., 1500.),
        )

        ds1 = xr.DataArray(
            da.zeros((bands, 100, 200), chunks=50),
            coords=[['R', 'G', 'B'], list(range(100)), list(range(200))],
            dims=('bands', 'y', 'x'),
            attrs={'name': 'test',
                   'start_time': datetime.utcnow(),
                   'platform_name': "TEST_PLATFORM_NAME",
                   'sensor': 'TEST_SENSOR_NAME',
                   'area': area_def,
                   'prerequisites': [DatasetID(name='1', calibration='reflectance'),
                                     DatasetID(name='2', calibration='reflectance')]}
        )
        return ds1

    def test_init(self):
        """Test creating the writer with no arguments."""
        from satpy.writers.mitiff import MITIFFWriter
        MITIFFWriter()

    def test_simple_write(self):
        """Test basic writer operation."""
        from satpy.writers.mitiff import MITIFFWriter
        dataset = self._get_test_dataset()
        w = MITIFFWriter(base_dir=self.base_dir)
        w.save_dataset(dataset)

    def test_save_datasets(self):
        """Test basic writer operation save_datasets."""
        import os
        import numpy as np
        from libtiff import TIFF
        from satpy.writers.mitiff import MITIFFWriter
        expected = np.full((100, 200), 0)
        dataset = self._get_test_datasets()
        w = MITIFFWriter(base_dir=self.base_dir)
        w.save_datasets(dataset)
        filename = (dataset[0].attrs['metadata_requirements']['file_pattern']).format(
            start_time=dataset[0].attrs['start_time'])
        tif = TIFF.open(os.path.join(self.base_dir, filename))
        for image in tif.iter_images():
            np.testing.assert_allclose(image, expected, atol=1.e-6, rtol=0)

    def test_save_datasets_sensor_set(self):
        """Test basic writer operation save_datasets."""
        import os
        import numpy as np
        from libtiff import TIFF
        from satpy.writers.mitiff import MITIFFWriter
        expected = np.full((100, 200), 0)
        dataset = self._get_test_datasets_sensor_set()
        w = MITIFFWriter(base_dir=self.base_dir)
        w.save_datasets(dataset)
        filename = (dataset[0].attrs['metadata_requirements']['file_pattern']).format(
            start_time=dataset[0].attrs['start_time'])
        tif = TIFF.open(os.path.join(self.base_dir, filename))
        for image in tif.iter_images():
            np.testing.assert_allclose(image, expected, atol=1.e-6, rtol=0)

    def test_save_one_dataset(self):
        """Test basic writer operation with one dataset ie. no bands."""
        import os
        from libtiff import TIFF
        from satpy.writers.mitiff import MITIFFWriter
        dataset = self._get_test_one_dataset()
        w = MITIFFWriter(base_dir=self.base_dir)
        w.save_dataset(dataset)
        tif = TIFF.open(os.path.join(self.base_dir, os.listdir(self.base_dir)[0]))
        IMAGEDESCRIPTION = 270
        imgdesc = (tif.GetField(IMAGEDESCRIPTION)).decode('utf-8').split('\n')
        for key in imgdesc:
            if 'In this file' in key:
                self.assertEqual(key, ' Channels: 1 In this file: 1')

    def test_save_one_dataset_sesnor_set(self):
        """Test basic writer operation with one dataset ie. no bands."""
        import os
        from libtiff import TIFF
        from satpy.writers.mitiff import MITIFFWriter
        dataset = self._get_test_one_dataset_sensor_set()
        w = MITIFFWriter(base_dir=self.base_dir)
        w.save_dataset(dataset)
        tif = TIFF.open(os.path.join(self.base_dir, os.listdir(self.base_dir)[0]))
        IMAGEDESCRIPTION = 270
        imgdesc = (tif.GetField(IMAGEDESCRIPTION)).decode('utf-8').split('\n')
        for key in imgdesc:
            if 'In this file' in key:
                self.assertEqual(key, ' Channels: 1 In this file: 1')

    def test_save_dataset_with_calibration(self):
        """Test writer operation with calibration."""
        import os
        import numpy as np
        from libtiff import TIFF
        from satpy.writers.mitiff import MITIFFWriter

        expected_ir = np.full((100, 200), 255)
        expected_vis = np.full((100, 200), 0)
        expected = np.stack([expected_vis, expected_vis, expected_ir, expected_ir, expected_ir, expected_vis])
        expected_key_channel = ['Table_calibration: 1-VIS0.63, Reflectance(Albedo), [%], 8, [ 0.00 0.39 0.78 1.18 1.57 '
                                '1.96 2.35 2.75 3.14 3.53 3.92 4.31 4.71 5.10 5.49 5.88 6.27 6.67 7.06 7.45 7.84 8.24 '
                                '8.63 9.02 9.41 9.80 10.20 10.59 10.98 11.37 11.76 12.16 12.55 12.94 13.33 13.73 14.12 '
                                '14.51 14.90 15.29 15.69 16.08 16.47 16.86 17.25 17.65 18.04 18.43 18.82 19.22 19.61 '
                                '20.00 20.39 20.78 21.18 21.57 21.96 22.35 22.75 23.14 23.53 23.92 24.31 24.71 25.10 '
                                '25.49 25.88 26.27 26.67 27.06 27.45 27.84 28.24 28.63 29.02 29.41 29.80 30.20 30.59 '
                                '30.98 31.37 31.76 32.16 32.55 32.94 33.33 33.73 34.12 34.51 34.90 35.29 35.69 36.08 '
                                '36.47 36.86 37.25 37.65 38.04 38.43 38.82 39.22 39.61 40.00 40.39 40.78 41.18 41.57 '
                                '41.96 42.35 42.75 43.14 43.53 43.92 44.31 44.71 45.10 45.49 45.88 46.27 46.67 47.06 '
                                '47.45 47.84 48.24 48.63 49.02 49.41 49.80 50.20 50.59 50.98 51.37 51.76 52.16 52.55 '
                                '52.94 53.33 53.73 54.12 54.51 54.90 55.29 55.69 56.08 56.47 56.86 57.25 57.65 58.04 '
                                '58.43 58.82 59.22 59.61 60.00 60.39 60.78 61.18 61.57 61.96 62.35 62.75 63.14 63.53 '
                                '63.92 64.31 64.71 65.10 65.49 65.88 66.27 66.67 67.06 67.45 67.84 68.24 68.63 69.02 '
                                '69.41 69.80 70.20 70.59 70.98 71.37 71.76 72.16 72.55 72.94 73.33 73.73 74.12 74.51 '
                                '74.90 75.29 75.69 76.08 76.47 76.86 77.25 77.65 78.04 78.43 78.82 79.22 79.61 80.00 '
                                '80.39 80.78 81.18 81.57 81.96 82.35 82.75 83.14 83.53 83.92 84.31 84.71 85.10 85.49 '
                                '85.88 86.27 86.67 87.06 87.45 87.84 88.24 88.63 89.02 89.41 89.80 90.20 90.59 90.98 '
                                '91.37 91.76 92.16 92.55 92.94 93.33 93.73 94.12 94.51 94.90 95.29 95.69 96.08 96.47 '
                                '96.86 97.25 97.65 98.04 98.43 98.82 99.22 99.61 100.00 ]',
                                'Table_calibration: 2-VIS0.86, Reflectance(Albedo), [%], 8, [ 0.00 0.39 0.78 1.18 1.57 '
                                '1.96 2.35 2.75 3.14 3.53 3.92 4.31 4.71 5.10 5.49 5.88 6.27 6.67 7.06 7.45 7.84 8.24 '
                                '8.63 9.02 9.41 9.80 10.20 10.59 10.98 11.37 11.76 12.16 12.55 12.94 13.33 13.73 14.12 '
                                '14.51 14.90 15.29 15.69 16.08 16.47 16.86 17.25 17.65 18.04 18.43 18.82 19.22 19.61 '
                                '20.00 20.39 20.78 21.18 21.57 21.96 22.35 22.75 23.14 23.53 23.92 24.31 24.71 25.10 '
                                '25.49 25.88 26.27 26.67 27.06 27.45 27.84 28.24 28.63 29.02 29.41 29.80 30.20 30.59 '
                                '30.98 31.37 31.76 32.16 32.55 32.94 33.33 33.73 34.12 34.51 34.90 35.29 35.69 36.08 '
                                '36.47 36.86 37.25 37.65 38.04 38.43 38.82 39.22 39.61 40.00 40.39 40.78 41.18 41.57 '
                                '41.96 42.35 42.75 43.14 43.53 43.92 44.31 44.71 45.10 45.49 45.88 46.27 46.67 47.06 '
                                '47.45 47.84 48.24 48.63 49.02 49.41 49.80 50.20 50.59 50.98 51.37 51.76 52.16 52.55 '
                                '52.94 53.33 53.73 54.12 54.51 54.90 55.29 55.69 56.08 56.47 56.86 57.25 57.65 58.04 '
                                '58.43 58.82 59.22 59.61 60.00 60.39 60.78 61.18 61.57 61.96 62.35 62.75 63.14 63.53 '
                                '63.92 64.31 64.71 65.10 65.49 65.88 66.27 66.67 67.06 67.45 67.84 68.24 68.63 69.02 '
                                '69.41 69.80 70.20 70.59 70.98 71.37 71.76 72.16 72.55 72.94 73.33 73.73 74.12 74.51 '
                                '74.90 75.29 75.69 76.08 76.47 76.86 77.25 77.65 78.04 78.43 78.82 79.22 79.61 80.00 '
                                '80.39 80.78 81.18 81.57 81.96 82.35 82.75 83.14 83.53 83.92 84.31 84.71 85.10 85.49 '
                                '85.88 86.27 86.67 87.06 87.45 87.84 88.24 88.63 89.02 89.41 89.80 90.20 90.59 90.98 '
                                '91.37 91.76 92.16 92.55 92.94 93.33 93.73 94.12 94.51 94.90 95.29 95.69 96.08 96.47 '
                                '96.86 97.25 97.65 98.04 98.43 98.82 99.22 99.61 100.00 ]',
                                u'Table_calibration: 3(3B)-IR3.7, BT, °[C], 8, [ 50.00 49.22 48.43 47.65 46.86 46.08 '
                                '45.29 44.51 43.73 42.94 42.16 41.37 40.59 39.80 39.02 38.24 37.45 36.67 35.88 35.10 '
                                '34.31 33.53 32.75 31.96 31.18 30.39 29.61 28.82 28.04 27.25 26.47 25.69 24.90 24.12 '
                                '23.33 22.55 21.76 20.98 20.20 19.41 18.63 17.84 17.06 16.27 15.49 14.71 13.92 13.14 '
                                '12.35 11.57 10.78 10.00 9.22 8.43 7.65 6.86 6.08 5.29 4.51 3.73 2.94 2.16 1.37 0.59 '
                                '-0.20 -0.98 -1.76 -2.55 -3.33 -4.12 -4.90 -5.69 -6.47 -7.25 -8.04 -8.82 -9.61 -10.39 '
                                '-11.18 -11.96 -12.75 -13.53 -14.31 -15.10 -15.88 -16.67 -17.45 -18.24 -19.02 -19.80 '
                                '-20.59 -21.37 -22.16 -22.94 -23.73 -24.51 -25.29 -26.08 -26.86 -27.65 -28.43 -29.22 '
                                '-30.00 -30.78 -31.57 -32.35 -33.14 -33.92 -34.71 -35.49 -36.27 -37.06 -37.84 -38.63 '
                                '-39.41 -40.20 -40.98 -41.76 -42.55 -43.33 -44.12 -44.90 -45.69 -46.47 -47.25 -48.04 '
                                '-48.82 -49.61 -50.39 -51.18 -51.96 -52.75 -53.53 -54.31 -55.10 -55.88 -56.67 -57.45 '
                                '-58.24 -59.02 -59.80 -60.59 -61.37 -62.16 -62.94 -63.73 -64.51 -65.29 -66.08 -66.86 '
                                '-67.65 -68.43 -69.22 -70.00 -70.78 -71.57 -72.35 -73.14 -73.92 -74.71 -75.49 -76.27 '
                                '-77.06 -77.84 -78.63 -79.41 -80.20 -80.98 -81.76 -82.55 -83.33 -84.12 -84.90 -85.69 '
                                '-86.47 -87.25 -88.04 -88.82 -89.61 -90.39 -91.18 -91.96 -92.75 -93.53 -94.31 -95.10 '
                                '-95.88 -96.67 -97.45 -98.24 -99.02 -99.80 -100.59 -101.37 -102.16 -102.94 -103.73 '
                                '-104.51 -105.29 -106.08 -106.86 -107.65 -108.43 -109.22 -110.00 -110.78 -111.57 '
                                '-112.35 -113.14 -113.92 -114.71 -115.49 -116.27 -117.06 -117.84 -118.63 -119.41 '
                                '-120.20 -120.98 -121.76 -122.55 -123.33 -124.12 -124.90 -125.69 -126.47 -127.25 '
                                '-128.04 -128.82 -129.61 -130.39 -131.18 -131.96 -132.75 -133.53 -134.31 -135.10 '
                                '-135.88 -136.67 -137.45 -138.24 -139.02 -139.80 -140.59 -141.37 -142.16 -142.94 '
                                '-143.73 -144.51 -145.29 -146.08 -146.86 -147.65 -148.43 -149.22 -150.00 ]',
                                u'Table_calibration: 4-IR10.8, BT, °[C], 8, [ 50.00 49.22 48.43 47.65 46.86 46.08 '
                                '45.29 '
                                '44.51 43.73 42.94 42.16 41.37 40.59 39.80 39.02 38.24 37.45 36.67 35.88 35.10 34.31 '
                                '33.53 32.75 31.96 31.18 30.39 29.61 28.82 28.04 27.25 26.47 25.69 24.90 24.12 23.33 '
                                '22.55 21.76 20.98 20.20 19.41 18.63 17.84 17.06 16.27 15.49 14.71 13.92 13.14 12.35 '
                                '11.57 10.78 10.00 9.22 8.43 7.65 6.86 6.08 5.29 4.51 3.73 2.94 2.16 1.37 0.59 -0.20 '
                                '-0.98 -1.76 -2.55 -3.33 -4.12 -4.90 -5.69 -6.47 -7.25 -8.04 -8.82 -9.61 -10.39 -11.18 '
                                '-11.96 -12.75 -13.53 -14.31 -15.10 -15.88 -16.67 -17.45 -18.24 -19.02 -19.80 -20.59 '
                                '-21.37 -22.16 -22.94 -23.73 -24.51 -25.29 -26.08 -26.86 -27.65 -28.43 -29.22 -30.00 '
                                '-30.78 -31.57 -32.35 -33.14 -33.92 -34.71 -35.49 -36.27 -37.06 -37.84 -38.63 -39.41 '
                                '-40.20 -40.98 -41.76 -42.55 -43.33 -44.12 -44.90 -45.69 -46.47 -47.25 -48.04 -48.82 '
                                '-49.61 -50.39 -51.18 -51.96 -52.75 -53.53 -54.31 -55.10 -55.88 -56.67 -57.45 -58.24 '
                                '-59.02 -59.80 -60.59 -61.37 -62.16 -62.94 -63.73 -64.51 -65.29 -66.08 -66.86 -67.65 '
                                '-68.43 -69.22 -70.00 -70.78 -71.57 -72.35 -73.14 -73.92 -74.71 -75.49 -76.27 -77.06 '
                                '-77.84 -78.63 -79.41 -80.20 -80.98 -81.76 -82.55 -83.33 -84.12 -84.90 -85.69 -86.47 '
                                '-87.25 -88.04 -88.82 -89.61 -90.39 -91.18 -91.96 -92.75 -93.53 -94.31 -95.10 -95.88 '
                                '-96.67 -97.45 -98.24 -99.02 -99.80 -100.59 -101.37 -102.16 -102.94 -103.73 -104.51 '
                                '-105.29 -106.08 -106.86 -107.65 -108.43 -109.22 -110.00 -110.78 -111.57 -112.35 '
                                '-113.14 -113.92 -114.71 -115.49 -116.27 -117.06 -117.84 -118.63 -119.41 -120.20 '
                                '-120.98 -121.76 -122.55 -123.33 -124.12 -124.90 -125.69 -126.47 -127.25 -128.04 '
                                '-128.82 -129.61 -130.39 -131.18 -131.96 -132.75 -133.53 -134.31 -135.10 -135.88 '
                                '-136.67 -137.45 -138.24 -139.02 -139.80 -140.59 -141.37 -142.16 -142.94 -143.73 '
                                '-144.51 -145.29 -146.08 -146.86 -147.65 -148.43 -149.22 -150.00 ]',
                                u'Table_calibration: 5-IR11.5, BT, °[C], 8, [ 50.00 49.22 48.43 47.65 46.86 46.08 '
                                '45.29 '
                                '44.51 43.73 42.94 42.16 41.37 40.59 39.80 39.02 38.24 37.45 36.67 35.88 35.10 34.31 '
                                '33.53 32.75 31.96 31.18 30.39 29.61 28.82 28.04 27.25 26.47 25.69 24.90 24.12 23.33 '
                                '22.55 21.76 20.98 20.20 19.41 18.63 17.84 17.06 16.27 15.49 14.71 13.92 13.14 12.35 '
                                '11.57 10.78 10.00 9.22 8.43 7.65 6.86 6.08 5.29 4.51 3.73 2.94 2.16 1.37 0.59 -0.20 '
                                '-0.98 -1.76 -2.55 -3.33 -4.12 -4.90 -5.69 -6.47 -7.25 -8.04 -8.82 -9.61 -10.39 -11.18 '
                                '-11.96 -12.75 -13.53 -14.31 -15.10 -15.88 -16.67 -17.45 -18.24 -19.02 -19.80 -20.59 '
                                '-21.37 -22.16 -22.94 -23.73 -24.51 -25.29 -26.08 -26.86 -27.65 -28.43 -29.22 -30.00 '
                                '-30.78 -31.57 -32.35 -33.14 -33.92 -34.71 -35.49 -36.27 -37.06 -37.84 -38.63 -39.41 '
                                '-40.20 -40.98 -41.76 -42.55 -43.33 -44.12 -44.90 -45.69 -46.47 -47.25 -48.04 -48.82 '
                                '-49.61 -50.39 -51.18 -51.96 -52.75 -53.53 -54.31 -55.10 -55.88 -56.67 -57.45 -58.24 '
                                '-59.02 -59.80 -60.59 -61.37 -62.16 -62.94 -63.73 -64.51 -65.29 -66.08 -66.86 -67.65 '
                                '-68.43 -69.22 -70.00 -70.78 -71.57 -72.35 -73.14 -73.92 -74.71 -75.49 -76.27 -77.06 '
                                '-77.84 -78.63 -79.41 -80.20 -80.98 -81.76 -82.55 -83.33 -84.12 -84.90 -85.69 -86.47 '
                                '-87.25 -88.04 -88.82 -89.61 -90.39 -91.18 -91.96 -92.75 -93.53 -94.31 -95.10 -95.88 '
                                '-96.67 -97.45 -98.24 -99.02 -99.80 -100.59 -101.37 -102.16 -102.94 -103.73 -104.51 '
                                '-105.29 -106.08 -106.86 -107.65 -108.43 -109.22 -110.00 -110.78 -111.57 -112.35 '
                                '-113.14 -113.92 -114.71 -115.49 -116.27 -117.06 -117.84 -118.63 -119.41 -120.20 '
                                '-120.98 -121.76 -122.55 -123.33 -124.12 -124.90 -125.69 -126.47 -127.25 -128.04 '
                                '-128.82 -129.61 -130.39 -131.18 -131.96 -132.75 -133.53 -134.31 -135.10 -135.88 '
                                '-136.67 -137.45 -138.24 -139.02 -139.80 -140.59 -141.37 -142.16 -142.94 -143.73 '
                                '-144.51 -145.29 -146.08 -146.86 -147.65 -148.43 -149.22 -150.00 ]',
                                'Table_calibration: 6(3A)-VIS1.6, Reflectance(Albedo), [%], 8, [ 0.00 0.39 0.78 1.18 '
                                '1.57 1.96 2.35 2.75 3.14 3.53 3.92 4.31 4.71 5.10 5.49 5.88 6.27 6.67 7.06 7.45 7.84 '
                                '8.24 8.63 9.02 9.41 9.80 10.20 10.59 10.98 11.37 11.76 12.16 12.55 12.94 13.33 13.73 '
                                '14.12 14.51 14.90 15.29 15.69 16.08 16.47 16.86 17.25 17.65 18.04 18.43 18.82 19.22 '
                                '19.61 20.00 20.39 20.78 21.18 21.57 21.96 22.35 22.75 23.14 23.53 23.92 24.31 24.71 '
                                '25.10 25.49 25.88 26.27 26.67 27.06 27.45 27.84 28.24 28.63 29.02 29.41 29.80 30.20 '
                                '30.59 30.98 31.37 31.76 32.16 32.55 32.94 33.33 33.73 34.12 34.51 34.90 35.29 35.69 '
                                '36.08 36.47 36.86 37.25 37.65 38.04 38.43 38.82 39.22 39.61 40.00 40.39 40.78 41.18 '
                                '41.57 41.96 42.35 42.75 43.14 43.53 43.92 44.31 44.71 45.10 45.49 45.88 46.27 46.67 '
                                '47.06 47.45 47.84 48.24 48.63 49.02 49.41 49.80 50.20 50.59 50.98 51.37 51.76 52.16 '
                                '52.55 52.94 53.33 53.73 54.12 54.51 54.90 55.29 55.69 56.08 56.47 56.86 57.25 57.65 '
                                '58.04 58.43 58.82 59.22 59.61 60.00 60.39 60.78 61.18 61.57 61.96 62.35 62.75 63.14 '
                                '63.53 63.92 64.31 64.71 65.10 65.49 65.88 66.27 66.67 67.06 67.45 67.84 68.24 68.63 '
                                '69.02 69.41 69.80 70.20 70.59 70.98 71.37 71.76 72.16 72.55 72.94 73.33 73.73 74.12 '
                                '74.51 74.90 75.29 75.69 76.08 76.47 76.86 77.25 77.65 78.04 78.43 78.82 79.22 79.61 '
                                '80.00 80.39 80.78 81.18 81.57 81.96 82.35 82.75 83.14 83.53 83.92 84.31 84.71 85.10 '
                                '85.49 85.88 86.27 86.67 87.06 87.45 87.84 88.24 88.63 89.02 89.41 89.80 90.20 90.59 '
                                '90.98 91.37 91.76 92.16 92.55 92.94 93.33 93.73 94.12 94.51 94.90 95.29 95.69 96.08 '
                                '96.47 96.86 97.25 97.65 98.04 98.43 98.82 99.22 99.61 100.00 ]']
        dataset = self._get_test_dataset_calibration()
        w = MITIFFWriter(filename=dataset.attrs['metadata_requirements']['file_pattern'], base_dir=self.base_dir)
        w.save_dataset(dataset)
        filename = (dataset.attrs['metadata_requirements']['file_pattern']).format(
            start_time=dataset.attrs['start_time'])
        tif = TIFF.open(os.path.join(self.base_dir, filename))
        IMAGEDESCRIPTION = 270
        imgdesc = (tif.GetField(IMAGEDESCRIPTION)).decode('utf-8').split('\n')
        found_table_calibration = False
        number_of_calibrations = 0
        for key in imgdesc:
            if 'Table_calibration' in key:
                found_table_calibration = True
                if '1-VIS0.63' in key:
                    self.assertEqual(key, expected_key_channel[0])
                    number_of_calibrations += 1
                elif '2-VIS0.86' in key:
                    self.assertEqual(key, expected_key_channel[1])
                    number_of_calibrations += 1
                elif '3(3B)-IR3.7' in key:
                    self.assertEqual(key, expected_key_channel[2])
                    number_of_calibrations += 1
                elif '4-IR10.8' in key:
                    self.assertEqual(key, expected_key_channel[3])
                    number_of_calibrations += 1
                elif '5-IR11.5' in key:
                    self.assertEqual(key, expected_key_channel[4])
                    number_of_calibrations += 1
                elif '6(3A)-VIS1.6' in key:
                    self.assertEqual(key, expected_key_channel[5])
                    number_of_calibrations += 1
                else:
                    self.fail("Not a valid channel description i the given key.")
        self.assertTrue(found_table_calibration, "Table_calibration is not found in the imagedescription.")
        self.assertEqual(number_of_calibrations, 6)
        for i, image in enumerate(tif.iter_images()):
            np.testing.assert_allclose(image, expected[i], atol=1.e-6, rtol=0)

    def test_save_dataset_with_calibration_one_dataset(self):
        """Test saving if mitiff as dataset with only one channel."""
        import os
        import numpy as np
        from libtiff import TIFF
        from satpy.writers.mitiff import MITIFFWriter

        expected = np.full((100, 200), 255)
        expected_key_channel = [u'Table_calibration: BT, BT, °[C], 8, [ 50.00 49.22 48.43 47.65 46.86 46.08 45.29 '
                                '44.51 43.73 42.94 42.16 41.37 40.59 39.80 39.02 38.24 37.45 36.67 35.88 35.10 34.31 '
                                '33.53 32.75 31.96 31.18 30.39 29.61 28.82 28.04 27.25 26.47 25.69 24.90 24.12 23.33 '
                                '22.55 21.76 20.98 20.20 19.41 18.63 17.84 17.06 16.27 15.49 14.71 13.92 13.14 12.35 '
                                '11.57 10.78 10.00 9.22 8.43 7.65 6.86 6.08 5.29 4.51 3.73 2.94 2.16 1.37 0.59 -0.20 '
                                '-0.98 -1.76 -2.55 -3.33 -4.12 -4.90 -5.69 -6.47 -7.25 -8.04 -8.82 -9.61 -10.39 -11.18 '
                                '-11.96 -12.75 -13.53 -14.31 -15.10 -15.88 -16.67 -17.45 -18.24 -19.02 -19.80 -20.59 '
                                '-21.37 -22.16 -22.94 -23.73 -24.51 -25.29 -26.08 -26.86 -27.65 -28.43 -29.22 -30.00 '
                                '-30.78 -31.57 -32.35 -33.14 -33.92 -34.71 -35.49 -36.27 -37.06 -37.84 -38.63 -39.41 '
                                '-40.20 -40.98 -41.76 -42.55 -43.33 -44.12 -44.90 -45.69 -46.47 -47.25 -48.04 -48.82 '
                                '-49.61 -50.39 -51.18 -51.96 -52.75 -53.53 -54.31 -55.10 -55.88 -56.67 -57.45 -58.24 '
                                '-59.02 -59.80 -60.59 -61.37 -62.16 -62.94 -63.73 -64.51 -65.29 -66.08 -66.86 -67.65 '
                                '-68.43 -69.22 -70.00 -70.78 -71.57 -72.35 -73.14 -73.92 -74.71 -75.49 -76.27 -77.06 '
                                '-77.84 -78.63 -79.41 -80.20 -80.98 -81.76 -82.55 -83.33 -84.12 -84.90 -85.69 -86.47 '
                                '-87.25 -88.04 -88.82 -89.61 -90.39 -91.18 -91.96 -92.75 -93.53 -94.31 -95.10 -95.88 '
                                '-96.67 -97.45 -98.24 -99.02 -99.80 -100.59 -101.37 -102.16 -102.94 -103.73 -104.51 '
                                '-105.29 -106.08 -106.86 -107.65 -108.43 -109.22 -110.00 -110.78 -111.57 -112.35 '
                                '-113.14 -113.92 -114.71 -115.49 -116.27 -117.06 -117.84 -118.63 -119.41 -120.20 '
                                '-120.98 -121.76 -122.55 -123.33 -124.12 -124.90 -125.69 -126.47 -127.25 -128.04 '
                                '-128.82 -129.61 -130.39 -131.18 -131.96 -132.75 -133.53 -134.31 -135.10 -135.88 '
                                '-136.67 -137.45 -138.24 -139.02 -139.80 -140.59 -141.37 -142.16 -142.94 -143.73 '
                                '-144.51 -145.29 -146.08 -146.86 -147.65 -148.43 -149.22 -150.00 ]', ]

        dataset = self._get_test_dataset_calibration_one_dataset()
        w = MITIFFWriter(filename=dataset.attrs['metadata_requirements']['file_pattern'], base_dir=self.base_dir)
        w.save_dataset(dataset)
        filename = (dataset.attrs['metadata_requirements']['file_pattern']).format(
            start_time=dataset.attrs['start_time'])
        tif = TIFF.open(os.path.join(self.base_dir, filename))
        IMAGEDESCRIPTION = 270
        imgdesc = (tif.GetField(IMAGEDESCRIPTION)).decode('utf-8').split('\n')
        found_table_calibration = False
        number_of_calibrations = 0
        for key in imgdesc:
            if 'Table_calibration' in key:
                found_table_calibration = True
                if 'BT' in key:
                    self.assertEqual(key, expected_key_channel[0])
                    number_of_calibrations += 1
        self.assertTrue(found_table_calibration, "Expected table_calibration is not found in the imagedescription.")
        self.assertEqual(number_of_calibrations, 1)
        for image in tif.iter_images():
            np.testing.assert_allclose(image, expected, atol=1.e-6, rtol=0)

    def test_save_dataset_with_bad_value(self):
        """Test writer operation with bad values."""
        import os
        import numpy as np
        from libtiff import TIFF
        from satpy.writers.mitiff import MITIFFWriter

        expected = np.array([[0, 4, 1, 37, 73],
                             [110, 146, 183, 219, 255]])

        dataset = self._get_test_dataset_with_bad_values()
        w = MITIFFWriter(base_dir=self.base_dir)
        w.save_dataset(dataset)
        filename = "{:s}_{:%Y%m%d_%H%M%S}.mitiff".format(dataset.attrs['name'],
                                                         dataset.attrs['start_time'])
        tif = TIFF.open(os.path.join(self.base_dir, filename))
        for image in tif.iter_images():
            np.testing.assert_allclose(image, expected, atol=1.e-6, rtol=0)

    def test_convert_proj4_string(self):
        import xarray as xr
        import dask.array as da
        from satpy.writers.mitiff import MITIFFWriter
        from pyresample.geometry import AreaDefinition
        checks = [{'epsg': '+init=EPSG:32631',
                   'proj4': (' Proj string: +proj=etmerc +lat_0=0 +lon_0=3 +k=0.9996 '
                             '+ellps=WGS84 +datum=WGS84 +units=km +x_0=501020.000000 '
                             '+y_0=1515.000000\n')},
                  {'epsg': '+init=EPSG:32632',
                   'proj4': (' Proj string: +proj=etmerc +lat_0=0 +lon_0=9 +k=0.9996 '
                             '+ellps=WGS84 +datum=WGS84 +units=km +x_0=501020.000000 '
                             '+y_0=1515.000000\n')},
                  {'epsg': '+init=EPSG:32633',
                   'proj4': (' Proj string: +proj=etmerc +lat_0=0 +lon_0=15 +k=0.9996 '
                             '+ellps=WGS84 +datum=WGS84 +units=km +x_0=501020.000000 '
                             '+y_0=1515.000000\n')},
                  {'epsg': '+init=EPSG:32634',
                   'proj4': (' Proj string: +proj=etmerc +lat_0=0 +lon_0=21 +k=0.9996 '
                             '+ellps=WGS84 +datum=WGS84 +units=km +x_0=501020.000000 '
                             '+y_0=1515.000000\n')},
                  {'epsg': '+init=EPSG:32635',
                   'proj4': (' Proj string: +proj=etmerc +lat_0=0 +lon_0=27 +k=0.9996 '
                             '+ellps=WGS84 +datum=WGS84 +units=km +x_0=501020.000000 '
                             '+y_0=1515.000000\n')}]
        for check in checks:
            area_def = AreaDefinition(
                'test',
                'test',
                'test',
                check['epsg'],
                100,
                200,
                (-1000., -1500., 1000., 1500.),
            )

            ds1 = xr.DataArray(
                da.zeros((10, 20), chunks=20),
                dims=('y', 'x'),
                attrs={'area': area_def}
            )

            w = MITIFFWriter(filename='dummy.tif', base_dir=self.base_dir)
            proj4_string = w._add_proj4_string(ds1, ds1)
            self.assertEqual(proj4_string, check['proj4'])

    def test_save_dataset_palette(self):
        """Test writer operation as palette."""
        import os
        import numpy as np
        from libtiff import TIFF
        from satpy.writers.mitiff import MITIFFWriter

        expected = np.full((100, 200), 0)

        exp_c = ([0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        color_map = [[0, 3], [1, 4], [2, 5]]
        pal_desc = ['test', 'test2']
        unit = "Test"

        dataset = self._get_test_one_dataset()
        palette = {'palette': True,
                   'palette_color_map': color_map,
                   'palette_description': pal_desc,
                   'palette_unit': unit,
                   'palette_channel_name': dataset.attrs['name']}

        w = MITIFFWriter(base_dir=self.base_dir)
        w.save_dataset(dataset, **palette)
        filename = "{:s}_{:%Y%m%d_%H%M%S}.mitiff".format(dataset.attrs['name'],
                                                         dataset.attrs['start_time'])
        tif = TIFF.open(os.path.join(self.base_dir, filename))
        # Need to check PHOTOMETRIC is 3, ie palette
        self.assertEqual(tif.GetField('PHOTOMETRIC'), 3)
        colormap = tif.GetField('COLORMAP')
        # Check the colormap of the palette image
        self.assertEqual(colormap, exp_c)
        IMAGEDESCRIPTION = 270
        imgdesc = (tif.GetField(IMAGEDESCRIPTION)).decode('utf-8').split('\n')
        found_color_info = False
        unit_name_found = False
        name_length_found = False
        name_length = 0
        names = []
        unit_name = None
        for key in imgdesc:
            if name_length_found and name_length > len(names):
                names.append(key)
                continue
            elif unit_name_found:
                name_length = int(key)
                name_length_found = True
                unit_name_found = False
            elif found_color_info:
                unit_name = key
                unit_name_found = True
                found_color_info = False
            elif 'COLOR INFO:' in key:
                found_color_info = True
        # Check the name of the palette description
        self.assertEqual(name_length, 2)
        # Check the name and unit name of the palette
        self.assertEqual(unit_name, ' Test')
        # Check the palette description of the palette
        self.assertEqual(names, [' test', ' test2'])
        for image in tif.iter_images():
            np.testing.assert_allclose(image, expected, atol=1.e-6, rtol=0)

    def test_simple_write_two_bands(self):
        """Test basic writer operation with 3 bands from 2 prerequisites"""
        from satpy.writers.mitiff import MITIFFWriter
        dataset = self._get_test_dataset_three_bands_two_prereq()
        w = MITIFFWriter(base_dir=self.base_dir)
        w.save_dataset(dataset)
