#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Trygve Aspenes
#
# Author(s):
#
#   Trygve Aspenes <trygveas@met.no>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Tests for the mitiff writer.
Based on the test for geotiff writer
"""
import sys

import logging

logger = logging.getLogger()
logger.level = logging.DEBUG
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

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
            proj_dict=proj4_str_to_dict('+proj=stere +datum=WGS84 +ellps=WGS84 \
            +lon_0=0. +lat_0=90 +lat_ts=60 +units=km'),
            x_size=100,
            y_size=200,
            area_extent=(-1000., -1500., 1000., 1500.),
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
            proj_dict=proj4_str_to_dict('+proj=stere +datum=WGS84 +ellps=WGS84 \
            +lon_0=0. +lat_0=90 +lat_ts=60 +units=km'),
            x_size=100,
            y_size=200,
            area_extent=(-1000., -1500., 1000., 1500.),
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

        print(scene)
        data = xr.concat(scene, 'bands', coords='minimal')
        bands = []
        calibration = []
        for p in scene:
            calibration.append(p.attrs['calibration'])
            bands.append(p.attrs['name'])
        data['bands'] = list(bands)
        new_attrs = {}
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

    def test_init(self):
        """Test creating the writer with no arguments."""
        from satpy.writers.mitiff import MITIFFWriter
        MITIFFWriter()

    def test_simple_write(self):
        """Test basic writer operation."""
        from satpy.writers.mitiff import MITIFFWriter
        dataset = self._get_test_dataset()
        w = MITIFFWriter(mitiff_dir=self.base_dir)
        w.save_dataset(dataset, writer='mitiff', mitiff_dir=self.base_dir)

    def test_save_datasets(self):
        """Test basic writer operation."""
        from satpy.writers.mitiff import MITIFFWriter
        dataset = self._get_test_dataset()
        w = MITIFFWriter(mitiff_dir=self.base_dir)
        w.save_datasets(dataset, writer='mitiff', mitiff_dir=self.base_dir)

    def test_save_dataset_with_calibration(self):
        """Test basic writer operation."""
        from satpy.writers.mitiff import MITIFFWriter
        dataset = self._get_test_dataset_calibration()
        w = MITIFFWriter(mitiff_dir=self.base_dir)
        w.save_dataset(dataset, writer='mitiff', mitiff_dir=self.base_dir)


def suite():
    """The test suite for this writer's tests.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestMITIFFWriter))
    return mysuite


if __name__ == '__main__':
    unittest.main()
