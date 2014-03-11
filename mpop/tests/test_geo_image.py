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
import mpop.imageo.geo_image as geo_image




class TestGeoImage(unittest.TestCase):
    """Class for testing pp.geo_image.
    """
    def setUp(self):
        """Setup the test.
        """
        time_slot = datetime.datetime(2009, 10, 8, 14, 30)
        self.data = np.zeros((512, 512), dtype=np.uint8)
        self.img = geo_image.GeoImage(self.data,
                                      area="euro",
                                      time_slot=time_slot)
        

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

    @patch('osgeo.osr.SpatialReference')
    @patch('mpop.projector.get_area_def')
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
        
        self.img.geotiff_save("test.tif", 0, None, None, 256)
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
        
        
        dst_ds.SetGeoTransform.assert_called_once_with([1, 10, 0, 4, 0, -11])
        srs = spaceref.return_value.ExportToWkt.return_value
        dst_ds.SetProjection.assert_called_once_with(srs)

        time_tag = {"TIFFTAG_DATETIME":
                    self.img.time_slot.strftime("%Y:%m:%d %H:%M:%S")}
        dst_ds.SetMetadata.assert_called_once_with(time_tag, '')

        # with compression set to 9

        gtbn.reset_mock()
        mock_write_channels.reset_mock()
        raster.Create.reset_mock()
        
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
        
        


def suite():
    """The test suite for test_geo_image.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestGeoImage))
    
    return mysuite
