"""Module for testing the pp.geo_image module.
"""
import datetime
import unittest
from mock import patch, MagicMock

import numpy as np

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
        

#     def test_add_overlay(self):
#         """Add overlay to the image.
#         """
#         self.img.add_overlay((1,1,1))

#         model = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                           [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                           [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                           [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                           [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                           [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        
#         self.assert_(np.all(self.img.channels[0][300:310, 300:310] == model))
    @patch.object(geo_image.GeoImage, 'geotiff_save')
    def test_save(self, mock_save):
        """Save a geo image.
        """

        
        #assert geo_image.GeoImage.geotiff_save == mock_save
        #self.assertEquals(self.img.geotiff_save, mock_save)
        self.img.save("test.tif", compression=0)
        mock_save.assert_called_once_with("test.tif", 0, None, None, 256)
        mock_save.reset_mock()
        self.img.save("test.tif", compression=9)
        mock_save.assert_called_once_with("test.tif", 9, None, None, 256)
        mock_save.reset_mock()
        self.img.save("test.tif", compression=9, floating_point=True)
        mock_save.assert_called_once_with("test.tif", 9, None, None, 256,
                                          floating_point=True)
        
    @patch('osgeo.gdal.GDT_Float64')
    @patch('osgeo.gdal.GDT_Byte')
    @patch('osgeo.gdal.GDT_UInt16')
    @patch('osgeo.gdal.GDT_UInt32')
    @patch('osgeo.gdal.GetDriverByName')
    @patch.object(geo_image.GeoImage, '_gdal_write_channels')
    def test_save_geotiff(self, mock_write_channels, gtbn, gui32, gui16, gby, gf):
        """Save to geotiff format.
        """
        #gtbn.Create = MagicMock()
        self.img.geotiff_save("test.tif", 0, None, None, 256)
        gtbn.assert_called_once_with("GTiff")
        ds_dst, data, opacity, fill_value = mock_write_channels.call_args[0]
        raster_call_args = gtbn.mock_calls[1][1]
        raster_call_kw = gtbn.mock_calls[1][2]
        #print raster_call_args
        #print raster_call_kw
        self.assertEquals(raster_call_args[0], "test.tif")
        self.assertEquals(tuple(raster_call_args[1:3]), self.data.shape)
        self.assertEquals(raster_call_args[3], 2)

        self.assertEquals(raster_call_args[5],
                          ['TILED=YES',
                           'BLOCKXSIZE=256',
                           'BLOCKYSIZE=256',
                           'ALPHA=YES'])
        #self.assertEquals(raster_call_args[4].name, "GDT_Byte")
        self.assertTrue(isinstance(ds_dst, MagicMock))
        self.assertTrue(np.all(data == self.data))
        self.assertEquals(opacity, 255)
        self.assertEquals(fill_value, None)

        self.img.geotiff_save("test.tif", 9, None, None, 256)
        gtbn.assert_called_with("GTiff")
        self.assertEquals(gtbn.call_count, 2)
        ds_dst, data, opacity, fill_value = mock_write_channels.call_args[0]
        self.assertTrue(isinstance(ds_dst, MagicMock))
        self.assertTrue(np.all(data == self.data))
        self.assertEquals(opacity, 255)
        self.assertEquals(fill_value, None)

        


def suite():
    """The test suite for test_geo_image.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestGeoImage))
    
    return mysuite
