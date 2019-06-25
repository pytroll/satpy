#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Satpy developers
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
import sys

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
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
                             '4': {'alias': '4-IR10.8',
                                   'calibration': 'brightness_temperature',
                                   'min-val': '-150',
                                   'max-val': '50'},
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

    def test_save_dataset_with_calibration(self):
        """Test writer operation with calibration."""
        import os
        import numpy as np
        from libtiff import TIFF
        from satpy.writers.mitiff import MITIFFWriter

        expected_ir = np.full((100, 200), 255)
        expected_vis = np.full((100, 200), 0)
        expected = np.stack([expected_vis, expected_vis, expected_ir, expected_ir, expected_ir, expected_vis])
        dataset = self._get_test_dataset_calibration()
        w = MITIFFWriter(filename=dataset.attrs['metadata_requirements']['file_pattern'], base_dir=self.base_dir)
        w.save_dataset(dataset)
        filename = (dataset.attrs['metadata_requirements']['file_pattern']).format(
            start_time=dataset.attrs['start_time'])
        tif = TIFF.open(os.path.join(self.base_dir, filename))
        for i, image in enumerate(tif.iter_images()):
            np.testing.assert_allclose(image, expected[i], atol=1.e-6, rtol=0)

    def test_save_dataset_with_calibration_one_dataset(self):
        """Test saving if mitiff as dataset with only one channel."""
        import os
        import numpy as np
        from libtiff import TIFF
        from satpy.writers.mitiff import MITIFFWriter

        expected = np.full((100, 200), 255)

        dataset = self._get_test_dataset_calibration_one_dataset()
        w = MITIFFWriter(filename=dataset.attrs['metadata_requirements']['file_pattern'], base_dir=self.base_dir)
        w.save_dataset(dataset)
        filename = (dataset.attrs['metadata_requirements']['file_pattern']).format(
            start_time=dataset.attrs['start_time'])
        tif = TIFF.open(os.path.join(self.base_dir, filename))
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


def suite():
    """The test suite for this writer's tests.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestMITIFFWriter))
    return mysuite


if __name__ == '__main__':
    unittest.main()
