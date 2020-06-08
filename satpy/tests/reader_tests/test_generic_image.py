#!/usr/bin/python
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
"""Unittests for generic image reader."""

import os
import unittest

import xarray as xr
import dask.array as da
import numpy as np


class TestGenericImage(unittest.TestCase):
    """Test generic image reader."""

    def setUp(self):
        """Create temporary images to test on."""
        import tempfile
        from datetime import datetime

        from pyresample.geometry import AreaDefinition
        from satpy.scene import Scene

        self.date = datetime(2018, 1, 1)

        # Create area definition
        pcs_id = 'ETRS89 / LAEA Europe'
        proj4_dict = {'init': 'epsg:3035'}
        self.x_size = 100
        self.y_size = 100
        area_extent = (2426378.0132, 1528101.2618, 6293974.6215, 5446513.5222)
        self.area_def = AreaDefinition('geotiff_area', pcs_id, pcs_id,
                                       proj4_dict, self.x_size, self.y_size,
                                       area_extent)

        # Create datasets for L, LA, RGB and RGBA mode images
        r__ = da.random.randint(0, 256, size=(self.y_size, self.x_size),
                                chunks=(50, 50)).astype(np.uint8)
        g__ = da.random.randint(0, 256, size=(self.y_size, self.x_size),
                                chunks=(50, 50)).astype(np.uint8)
        b__ = da.random.randint(0, 256, size=(self.y_size, self.x_size),
                                chunks=(50, 50)).astype(np.uint8)
        a__ = 255 * np.ones((self.y_size, self.x_size), dtype=np.uint8)
        a__[:10, :10] = 0
        a__ = da.from_array(a__, chunks=(50, 50))

        ds_l = xr.DataArray(da.stack([r__]), dims=('bands', 'y', 'x'),
                            attrs={'name': 'test_l',
                                   'start_time': self.date})
        ds_l['bands'] = ['L']
        ds_la = xr.DataArray(da.stack([r__, a__]), dims=('bands', 'y', 'x'),
                             attrs={'name': 'test_la',
                                    'start_time': self.date})
        ds_la['bands'] = ['L', 'A']
        ds_rgb = xr.DataArray(da.stack([r__, g__, b__]),
                              dims=('bands', 'y', 'x'),
                              attrs={'name': 'test_rgb',
                                     'start_time': self.date})
        ds_rgb['bands'] = ['R', 'G', 'B']
        ds_rgba = xr.DataArray(da.stack([r__, g__, b__, a__]),
                               dims=('bands', 'y', 'x'),
                               attrs={'name': 'test_rgba',
                                      'start_time': self.date})
        ds_rgba['bands'] = ['R', 'G', 'B', 'A']

        # Temp dir for the saved images
        self.base_dir = tempfile.mkdtemp()

        # Put the datasets to Scene for easy saving
        scn = Scene()
        scn['l'] = ds_l
        scn['l'].attrs['area'] = self.area_def
        scn['la'] = ds_la
        scn['la'].attrs['area'] = self.area_def
        scn['rgb'] = ds_rgb
        scn['rgb'].attrs['area'] = self.area_def
        scn['rgba'] = ds_rgba
        scn['rgba'].attrs['area'] = self.area_def

        # Save the images.  Two images in PNG and two in GeoTIFF
        scn.save_dataset('l', os.path.join(self.base_dir, 'test_l.png'), writer='simple_image')
        scn.save_dataset('la', os.path.join(self.base_dir, '20180101_0000_test_la.png'), writer='simple_image')
        scn.save_dataset('rgb', os.path.join(self.base_dir, '20180101_0000_test_rgb.tif'), writer='geotiff')
        scn.save_dataset('rgba', os.path.join(self.base_dir, 'test_rgba.tif'), writer='geotiff')

        self.scn = scn

    def tearDown(self):
        """Remove the temporary directory created for a test"""
        try:
            import shutil
            shutil.rmtree(self.base_dir, ignore_errors=True)
        except OSError:
            pass

    def test_png_scene(self):
        """Test reading PNG images via satpy.Scene()."""
        from satpy import Scene

        fname = os.path.join(self.base_dir, 'test_l.png')
        scn = Scene(reader='generic_image', filenames=[fname])
        scn.load(['image'])
        self.assertEqual(scn['image'].shape, (1, self.y_size, self.x_size))
        self.assertEqual(scn.attrs['sensor'], set(['images']))
        self.assertEqual(scn.attrs['start_time'], None)
        self.assertEqual(scn.attrs['end_time'], None)
        self.assertNotIn('area', scn['image'].attrs)

        fname = os.path.join(self.base_dir, '20180101_0000_test_la.png')
        scn = Scene(reader='generic_image', filenames=[fname])
        scn.load(['image'])
        data = da.compute(scn['image'].data)
        self.assertEqual(scn['image'].shape, (1, self.y_size, self.x_size))
        self.assertEqual(scn.attrs['sensor'], set(['images']))
        self.assertEqual(scn.attrs['start_time'], self.date)
        self.assertEqual(scn.attrs['end_time'], self.date)
        self.assertNotIn('area', scn['image'].attrs)
        self.assertEqual(np.sum(np.isnan(data)), 100)

    def test_geotiff_scene(self):
        """Test reading PNG images via satpy.Scene()."""
        from satpy import Scene

        fname = os.path.join(self.base_dir, '20180101_0000_test_rgb.tif')
        scn = Scene(reader='generic_image', filenames=[fname])
        scn.load(['image'])
        self.assertEqual(scn['image'].shape, (3, self.y_size, self.x_size))
        self.assertEqual(scn.attrs['sensor'], set(['images']))
        self.assertEqual(scn.attrs['start_time'], self.date)
        self.assertEqual(scn.attrs['end_time'], self.date)
        self.assertEqual(scn['image'].area, self.area_def)

        fname = os.path.join(self.base_dir, 'test_rgba.tif')
        scn = Scene(reader='generic_image', filenames=[fname])
        scn.load(['image'])
        self.assertEqual(scn['image'].shape, (3, self.y_size, self.x_size))
        self.assertEqual(scn.attrs['sensor'], set(['images']))
        self.assertEqual(scn.attrs['start_time'], None)
        self.assertEqual(scn.attrs['end_time'], None)
        self.assertEqual(scn['image'].area, self.area_def)

    def test_GenericImageFileHandler(self):
        """Test direct use of the reader."""
        from satpy.readers.generic_image import GenericImageFileHandler
        from satpy.readers.generic_image import mask_image_data

        fname = os.path.join(self.base_dir, 'test_rgba.tif')
        fname_info = {'start_time': self.date}
        ftype_info = {}
        reader = GenericImageFileHandler(fname, fname_info, ftype_info)

        class Foo(object):
            """Mock class for dataset id"""
            def __init__(self):
                self.name = 'image'

        foo = Foo()
        self.assertTrue(reader.file_content)
        self.assertEqual(reader.finfo['filename'], fname)
        self.assertEqual(reader.finfo['start_time'], self.date)
        self.assertEqual(reader.finfo['end_time'], self.date)
        self.assertEqual(reader.area, self.area_def)
        self.assertEqual(reader.get_area_def(None), self.area_def)
        self.assertEqual(reader.start_time, self.date)
        self.assertEqual(reader.end_time, self.date)

        dataset = reader.get_dataset(foo, None)
        self.assertTrue(isinstance(dataset, xr.DataArray))
        self.assertTrue('crs' in dataset.attrs)
        self.assertTrue('transform' in dataset.attrs)
        self.assertTrue(np.all(np.isnan(dataset.data[:, :10, :10].compute())))

        # Test masking of floats
        data = self.scn['rgba']
        self.assertRaises(ValueError, mask_image_data, data / 255.)
        data = data.astype(np.uint32)
        self.assertTrue(data.bands.size == 4)
        data = mask_image_data(data)
        self.assertTrue(data.bands.size == 3)
