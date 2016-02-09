#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2014

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# satpy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Module for testing the pp.geo_image module.
"""
import datetime
import unittest
import sys

import numpy as np
from mock import patch, MagicMock

# Mock some modules, so we don't need them for tests.

sys.modules['osgeo'] = MagicMock()
sys.modules['pyresample'] = MagicMock()

import satpy.imageo.geo_image as geo_image

class TestGeoImage(unittest.TestCase):
    """Class for testing pp.geo_image.
    """
    def setUp(self):
        """Setup the test.
        """
        
        self.time_slot = datetime.datetime(2009, 10, 8, 14, 30)
        self.data = np.zeros((512, 512), dtype=np.uint8)
        self.img = geo_image.GeoImage(self.data,
                                      area="euro",
                                      start_time=self.time_slot)


    @patch.object(geo_image.GeoImage, 'geotiff_save')
    def test_save(self, mock_save):
        """Save a geo image.
        """
        
        self.img.save("test.tif", compression=0)
        mock_save.assert_called_once_with("test.tif", 0, None, None, 256)
        mock_save.reset_mock()
        self.img.save("test.tif", compression=9)
        mock_save.assert_called_once_with("test.tif", 9, None, None, 256)
        mock_save.reset_mock()
        self.img.save("test.tif", compression=9, floating_point=True)
        mock_save.assert_called_once_with("test.tif", 9, None, None, 256,
                                          floating_point=True)

        mock_save.reset_mock()
        self.img.save("test.tif", compression=9, tags={"NBITS": 20})
        mock_save.assert_called_once_with("test.tif", 9, {"NBITS": 20},
                                          None, 256)

        with patch.object(geo_image.Image, 'save') as mock_isave:
            self.img.save("test.png")
            mock_isave.assert_called_once_with(self.img, 'test.png', 6,
                                               fformat='png')
            mock_isave.side_effect = geo_image.UnknownImageFormat("Boom!")
            self.assertRaises(geo_image.UnknownImageFormat,
                              self.img.save, "test.dummy")


    @patch('osgeo.osr.SpatialReference')
    @patch('satpy.projector.get_area_def')
    @patch('osgeo.gdal.GDT_Float64')
    @patch('osgeo.gdal.GDT_Byte')
    @patch('osgeo.gdal.GDT_UInt16')
    @patch('osgeo.gdal.GDT_UInt32')
    @patch('osgeo.gdal.GetDriverByName')
    @patch.object(geo_image.GeoImage, '_gdal_write_channels')
    def test_save_geotiff(self, mock_write_channels, gtbn, gui32, gui16, gby, gf, gad, spaceref):
        """Save to geotiff format.
        """
        gadr = gad.return_value
        gadr.area_extent = [1, 2, 3, 4]
        gadr.pixel_size_x = 10
        gadr.pixel_size_y = 11
        gadr.proj4_string = "+proj=geos +ellps=WGS84"
        gadr.proj_dict = {"proj": "geos", "ellps": "WGS84"}
        gadr.proj_id = "geos0"

        # test with 0 compression

        raster = gtbn.return_value
        
        self.img.geotiff_save("test.tif", 0, None, {"BLA": "09"}, 256)
        gtbn.assert_called_once_with("GTiff")

        raster.Create.assert_called_once_with("test.tif",
                                              self.data.shape[0],
                                              self.data.shape[1],
                                              2,
                                              gby,
                                              ["BLA=09",
                                               'TILED=YES',
                                               'BLOCKXSIZE=256',
                                               'BLOCKYSIZE=256',
                                               'ALPHA=YES'])
        dst_ds = raster.Create.return_value

        #mock_write_channels.assert_called_once_with(dst_ds, self.data,
        #                                            255, None)

        self.assertEquals(mock_write_channels.call_count, 1)
        self.assertEquals(mock_write_channels.call_args[0][0], dst_ds)
        self.assertEquals(mock_write_channels.call_args[0][2], 255)
        self.assertTrue(mock_write_channels.call_args[0][3] is None)
        self.assertTrue(np.all(mock_write_channels.call_args[0][1]
                               == self.data))
        
        
        dst_ds.SetGeoTransform.assert_called_once_with([1, 10, 0, 4, 0, -11])
        srs = spaceref.return_value.ExportToWkt.return_value
        dst_ds.SetProjection.assert_called_once_with(srs)

        time_tag = {"TIFFTAG_DATETIME":
                    self.img.time_slot.strftime("%Y:%m:%d %H:%M:%S")}
        dst_ds.SetMetadata.assert_called_once_with(time_tag, '')

    @patch('osgeo.osr.SpatialReference')
    @patch('satpy.projector.get_area_def')
    @patch('osgeo.gdal.GDT_Float64')
    @patch('osgeo.gdal.GDT_Byte')
    @patch('osgeo.gdal.GDT_UInt16')
    @patch('osgeo.gdal.GDT_UInt32')
    @patch('osgeo.gdal.GetDriverByName')
    @patch.object(geo_image.GeoImage, '_gdal_write_channels')
    def test_save_geotiff_compress(self, mock_write_channels, gtbn, gui32, gui16, gby, gf, gad, spaceref):
        """Save to geotiff format with compression.
        """
        gadr = gad.return_value
        gadr.area_extent = [1, 2, 3, 4]
        gadr.pixel_size_x = 10
        gadr.pixel_size_y = 11
        gadr.proj4_string = "+proj=geos +ellps=WGS84"
        gadr.proj_dict = {"proj": "geos", "ellps": "WGS84"}
        gadr.proj_id = "geos0"

        raster = gtbn.return_value

        self.img.geotiff_save("test.tif", 9, None, None, 256)
        gtbn.assert_called_once_with("GTiff")

        raster.Create.assert_called_once_with("test.tif",
                                              self.data.shape[0],
                                              self.data.shape[1],
                                              2,
                                              gby,
                                              ['COMPRESS=DEFLATE',
                                               'ZLEVEL=9',
                                               'TILED=YES',
                                               'BLOCKXSIZE=256',
                                               'BLOCKYSIZE=256',
                                               'ALPHA=YES'])
        dst_ds = raster.Create.return_value

        #mock_write_channels.assert_called_once_with(dst_ds, self.data,
        #                                            255, None)

        self.assertEquals(mock_write_channels.call_count, 1)
        self.assertEquals(mock_write_channels.call_args[0][0], dst_ds)
        self.assertEquals(mock_write_channels.call_args[0][2], 255)
        self.assertTrue(mock_write_channels.call_args[0][3] is None)
        self.assertTrue(np.all(mock_write_channels.call_args[0][1] == self.data))
        
        dst_ds.SetGeoTransform.assert_called_once_with([1, 10, 0, 4, 0, -11])
        srs = spaceref.return_value.ExportToWkt.return_value
        dst_ds.SetProjection.assert_called_once_with(srs)

        time_tag = {"TIFFTAG_DATETIME":
                    self.img.time_slot.strftime("%Y:%m:%d %H:%M:%S")}
        dst_ds.SetMetadata.assert_called_once_with(time_tag, '')
        
    @patch('osgeo.osr.SpatialReference')
    @patch('satpy.projector.get_area_def')
    @patch('osgeo.gdal.GDT_Float64')
    @patch('osgeo.gdal.GDT_Byte')
    @patch('osgeo.gdal.GDT_UInt16')
    @patch('osgeo.gdal.GDT_UInt32')
    @patch('osgeo.gdal.GetDriverByName')
    @patch.object(geo_image.GeoImage, '_gdal_write_channels')
    def test_save_geotiff_floats(self, mock_write_channels, gtbn, gui32, gui16, gby, gf, gad, spaceref):
        """Save to geotiff format with floats.
        """
        gadr = gad.return_value
        gadr.area_extent = [1, 2, 3, 4]
        gadr.pixel_size_x = 10
        gadr.pixel_size_y = 11
        gadr.proj4_string = "+proj=geos +ellps=WGS84"
        gadr.proj_dict = {"proj": "geos", "ellps": "WGS84"}
        gadr.proj_id = "geos0"
        # test with floats

        raster = gtbn.return_value
        
        self.img.geotiff_save("test.tif", 0, None, None, 256,
                              floating_point=True)
        gtbn.assert_called_once_with("GTiff")

        raster.Create.assert_called_once_with("test.tif",
                                              self.data.shape[0],
                                              self.data.shape[1],
                                              1,
                                              gf,
                                              ['TILED=YES',
                                               'BLOCKXSIZE=256',
                                               'BLOCKYSIZE=256'])

        dst_ds = raster.Create.return_value

        #mock_write_channels.assert_called_once_with(dst_ds, self.data,
        #                                            255, None)

        self.assertEquals(mock_write_channels.call_count, 1)
        self.assertEquals(mock_write_channels.call_args[0][0], dst_ds)
        self.assertEquals(mock_write_channels.call_args[0][2], 0)
        self.assertEquals(mock_write_channels.call_args[0][3], [0])
        self.assertTrue(np.all(mock_write_channels.call_args[0][1]
                               == self.data))
        
        
        dst_ds.SetGeoTransform.assert_called_once_with([1, 10, 0, 4, 0, -11])
        srs = spaceref.return_value.ExportToWkt.return_value
        dst_ds.SetProjection.assert_called_once_with(srs)

        time_tag = {"TIFFTAG_DATETIME":
                    self.img.time_slot.strftime("%Y:%m:%d %H:%M:%S")}
        dst_ds.SetMetadata.assert_called_once_with(time_tag, '')

        self.fill_value = None
        self.img.mode = "RGB"
        self.assertRaises(ValueError, self.img.geotiff_save, "test.tif", 0,
                          None, None, 256,
                          floating_point=True)
        
    @patch('osgeo.osr.SpatialReference')
    @patch('satpy.projector.get_area_def')
    @patch('osgeo.gdal.GDT_Float64')
    @patch('osgeo.gdal.GDT_Byte')
    @patch('osgeo.gdal.GDT_UInt16')
    @patch('osgeo.gdal.GDT_UInt32')
    @patch('osgeo.gdal.GetDriverByName')
    @patch.object(geo_image.GeoImage, '_gdal_write_channels')
    def test_save_geotiff_32(self, mock_write_channels, gtbn, gui32, gui16, gby, gf, gad, spaceref):
        """Save to geotiff 32-bits format.
        """
        gadr = gad.return_value
        gadr.area_extent = [1, 2, 3, 4]
        gadr.pixel_size_x = 10
        gadr.pixel_size_y = 11
        gadr.proj4_string = "+proj=geos +ellps=WGS84"
        gadr.proj_dict = {"proj": "geos", "ellps": "WGS84"}
        gadr.proj_id = "geos0"

        raster = gtbn.return_value
        

        self.img.geotiff_save("test.tif", 9, {"NBITS": 20}, None, 256)
        gtbn.assert_called_once_with("GTiff")

        raster.Create.assert_called_once_with("test.tif",
                                              self.data.shape[0],
                                              self.data.shape[1],
                                              2,
                                              gui32,
                                              ['COMPRESS=DEFLATE',
                                               'ZLEVEL=9',
                                               'TILED=YES',
                                               'BLOCKXSIZE=256',
                                               'BLOCKYSIZE=256',
                                               'ALPHA=YES'])
        dst_ds = raster.Create.return_value

        #mock_write_channels.assert_called_once_with(dst_ds, self.data,
        #                                            255, None)

        self.assertEquals(mock_write_channels.call_count, 1)
        self.assertEquals(mock_write_channels.call_args[0][0], dst_ds)
        self.assertEquals(mock_write_channels.call_args[0][2], 2**32 - 1)
        self.assertTrue(mock_write_channels.call_args[0][3] is None)
        self.assertTrue(np.all(mock_write_channels.call_args[0][1] == self.data))
        
        dst_ds.SetGeoTransform.assert_called_once_with([1, 10, 0, 4, 0, -11])
        srs = spaceref.return_value.ExportToWkt.return_value
        dst_ds.SetProjection.assert_called_once_with(srs)

        time_tag = {"TIFFTAG_DATETIME":
                    self.img.time_slot.strftime("%Y:%m:%d %H:%M:%S"),
                    "NBITS": 20}
        dst_ds.SetMetadata.assert_called_once_with(time_tag, '')


    @patch('osgeo.osr.SpatialReference')
    @patch('satpy.projector.get_area_def')
    @patch('osgeo.gdal.GDT_Float64')
    @patch('osgeo.gdal.GDT_Byte')
    @patch('osgeo.gdal.GDT_UInt16')
    @patch('osgeo.gdal.GDT_UInt32')
    @patch('osgeo.gdal.GetDriverByName')
    @patch.object(geo_image.GeoImage, '_gdal_write_channels')
    def test_save_geotiff_16(self, mock_write_channels, gtbn, gui32, gui16, gby, gf, gad, spaceref):
        """Save to geotiff 16-bits format.
        """
        gadr = gad.return_value
        gadr.area_extent = [1, 2, 3, 4]
        gadr.pixel_size_x = 10
        gadr.pixel_size_y = 11
        gadr.proj4_string = "+proj=geos +ellps=WGS84"
        gadr.proj_dict = {"proj": "geos", "ellps": "WGS84"}
        gadr.proj_id = "geos0"

        raster = gtbn.return_value
        

        self.img.geotiff_save("test.tif", 9, {"NBITS": 15}, None, 256)
        gtbn.assert_called_once_with("GTiff")

        raster.Create.assert_called_once_with("test.tif",
                                              self.data.shape[0],
                                              self.data.shape[1],
                                              2,
                                              gui16,
                                              ['COMPRESS=DEFLATE',
                                               'ZLEVEL=9',
                                               'TILED=YES',
                                               'BLOCKXSIZE=256',
                                               'BLOCKYSIZE=256',
                                               'ALPHA=YES'])
        dst_ds = raster.Create.return_value

        #mock_write_channels.assert_called_once_with(dst_ds, self.data,
        #                                            255, None)

        self.assertEquals(mock_write_channels.call_count, 1)
        self.assertEquals(mock_write_channels.call_args[0][0], dst_ds)
        self.assertEquals(mock_write_channels.call_args[0][2], 2**16 - 1)
        self.assertTrue(mock_write_channels.call_args[0][3] is None)
        self.assertTrue(np.all(mock_write_channels.call_args[0][1] == self.data))
        
        dst_ds.SetGeoTransform.assert_called_once_with([1, 10, 0, 4, 0, -11])
        srs = spaceref.return_value.ExportToWkt.return_value
        dst_ds.SetProjection.assert_called_once_with(srs)

        time_tag = {"TIFFTAG_DATETIME":
                    self.img.time_slot.strftime("%Y:%m:%d %H:%M:%S"),
                    "NBITS": 15}
        dst_ds.SetMetadata.assert_called_once_with(time_tag, '')


    @patch('osgeo.osr.SpatialReference')
    @patch('satpy.projector.get_area_def')
    @patch('osgeo.gdal.GDT_Float64')
    @patch('osgeo.gdal.GDT_Byte')
    @patch('osgeo.gdal.GDT_UInt16')
    @patch('osgeo.gdal.GDT_UInt32')
    @patch('osgeo.gdal.GetDriverByName')
    @patch.object(geo_image.GeoImage, '_gdal_write_channels')
    def test_save_geotiff_geotransform(self, mock_write_channels, gtbn, gui32, gui16, gby, gf, gad, spaceref):
        """Save to geotiff format with custom geotransform
        """
        gadr = gad.return_value
        gadr.area_extent = [1, 2, 3, 4]
        gadr.pixel_size_x = 10
        gadr.pixel_size_y = 11
        gadr.proj4_string = "+proj=geos +ellps=WGS84"
        gadr.proj_dict = {"proj": "geos", "ellps": "WGS84"}
        gadr.proj_id = "geos0"

        # test with 0 compression

        raster = gtbn.return_value
        
        self.img.geotiff_save("test.tif", 0, None, None, 256,
                              geotransform="best geotransform of the world",
                              spatialref=spaceref())
        gtbn.assert_called_once_with("GTiff")

        raster.Create.assert_called_once_with("test.tif",
                                              self.data.shape[0],
                                              self.data.shape[1],
                                              2,
                                              gby,
                                              ['TILED=YES',
                                               'BLOCKXSIZE=256',
                                               'BLOCKYSIZE=256',
                                               'ALPHA=YES'])
        dst_ds = raster.Create.return_value

        #mock_write_channels.assert_called_once_with(dst_ds, self.data,
        #                                            255, None)

        self.assertEquals(mock_write_channels.call_count, 1)
        self.assertEquals(mock_write_channels.call_args[0][0], dst_ds)
        self.assertEquals(mock_write_channels.call_args[0][2], 255)
        self.assertTrue(mock_write_channels.call_args[0][3] is None)
        self.assertTrue(np.all(mock_write_channels.call_args[0][1]
                               == self.data))
        
        
        dst_ds.SetGeoTransform.assert_called_once_with("best geotransform of"
                                                       " the world")
        srs = spaceref.return_value.ExportToWkt.return_value
        dst_ds.SetProjection.assert_called_once_with(srs)

        time_tag = {"TIFFTAG_DATETIME":
                    self.img.time_slot.strftime("%Y:%m:%d %H:%M:%S")}
        dst_ds.SetMetadata.assert_called_once_with(time_tag, '')


def suite():
    """The test suite for test_geo_image.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestGeoImage))
    
    return mysuite
