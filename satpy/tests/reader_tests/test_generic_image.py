#!/usr/bin/python
# Copyright (c) 2018.
#

# Author(s):
#   Panu Lahtinen <panu.lahtinen@fmi.fi>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""
"""

import os
import shutil
import unittest

import xarray as xr
import dask.array as da
import numpy as np

from satpy.scene import Scene


class TestGenericImage(unittest.TestCase):
    """Test generic image reader."""

    def setUp(self):
        """Create temporary images to test on."""
        import tempfile
        from datetime import datetime

        from pyresample.geometry import AreaDefinition

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
                                chunks=(50, 50))
        g__ = da.random.randint(0, 256, size=(self.y_size, self.x_size),
                                chunks=(50, 50))
        b__ = da.random.randint(0, 256, size=(self.y_size, self.x_size),
                                chunks=(50, 50))
        a__ = 255 * np.ones((self.y_size, self.x_size))
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
        scn['l'].attrs['area'] = self.area_def
        scn['rgb'] = ds_rgb
        scn['l'].attrs['area'] = self.area_def
        scn['rgba'] = ds_rgba
        scn['l'].attrs['area'] = self.area_def

        # Save the images.  Two images in PNG and two in GeoTIFF
        scn.save_dataset('l', os.path.join(self.base_dir, 'test_l.png'),
                         writer='simple_image')
        scn.save_dataset('la', os.path.join(self.base_dir,
                                            '20180101_0000_test_la.png'),
                         writer='simple_image')
        scn.save_dataset('rgb', os.path.join(self.base_dir,
                                             '20180101_0000_test_rgb.tif'),
                         writer='geotiff')
        scn.save_dataset('rgba', os.path.join(self.base_dir,
                                              'test_rgba.tif'),
                         writer='geotiff')

    def tearDown(self):
        """Remove the temporary directory created for a test"""
        try:
            import shutil
            # shutil.rmtree(self.base_dir, ignore_errors=True)
        except OSError:
            pass

    def test_png(self):
        """Test reading PNG images."""
        fname = os.path.join(self.base_dir, 'test_l.png')
        scn = Scene(reader='generic_image', filenames=[fname])
        scn.load(['image'])
        self.assertEqual(scn['image'].shape, (1, self.y_size, self.x_size))
        self.assertEqual(scn.attrs['sensor'], set(['images']))
        self.assertEqual(scn.attrs['start_time'], None)
        self.assertEqual(scn.attrs['end_time'], None)

        fname = os.path.join(self.base_dir, '20180101_0000_test_la.png')
        scn = Scene(reader='generic_image', filenames=[fname])
        scn.load(['image'])
        data = da.compute(scn['image'].data)
        self.assertEqual(scn['image'].shape, (1, self.y_size, self.x_size))
        self.assertEqual(scn.attrs['sensor'], set(['images']))
        self.assertEqual(scn.attrs['start_time'], self.date)
        self.assertEqual(scn.attrs['end_time'], self.date)
        self.assertEqual(np.sum(np.isnan(data)), 100)

    def test_geotiff(self):
        """Test reading PNG images."""
        fname = os.path.join(self.base_dir, '20180101_0000_test_rgb.tif')
        scn = Scene(reader='generic_image', filenames=[fname])
        scn.load(['image'])
        self.assertEqual(scn['image'].shape, (3, self.y_size, self.x_size))
        self.assertEqual(scn.attrs['sensor'], set(['images']))
        self.assertEqual(scn.attrs['start_time'], self.date)
        self.assertEqual(scn.attrs['end_time'], self.date)
        # self.assertEqual(scn['image'].area, self.area_def)

        fname = os.path.join(self.base_dir, 'test_rgba.tif')
        scn = Scene(reader='generic_image', filenames=[fname])
        scn.load(['image'])
        self.assertEqual(scn['image'].shape, (3, self.y_size, self.x_size))
        self.assertEqual(scn.attrs['sensor'], set(['images']))
        self.assertEqual(scn.attrs['start_time'], None)
        self.assertEqual(scn.attrs['end_time'], None)
        # self.assertEqual(scn['image'].area, self.area_def)

def suite():
    """The test suite for test_writers."""
    loader = unittest.TestLoader()
    my_suite = unittest.TestSuite()
    my_suite.addTest(loader.loadTestsFromTestCase(TestGenericImage))

    return my_suite

if __name__ == '__main__':
    unittest.main()
