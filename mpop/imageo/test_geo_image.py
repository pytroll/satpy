"""Module for testing the pp.geo_image module.
"""
import datetime
import unittest

import numpy as np

import mpop.imageo.geo_image as geo_image


class TestGeoImage(unittest.TestCase):
    """Class for testing pp.geo_image.
    """
    def setUp(self):
        """Setup the test.
        """
        time_slot = datetime.datetime(2009, 10, 8, 14, 30)
        self.img = geo_image.GeoImage(np.zeros((512, 512), dtype = np.uint8),
                                      area_id = "euro", time_slot = time_slot)
        

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
        
#     def test_save_geotiff(self):
#         """Save to geotiff format.
#         """
#         self.img.save("test.tif", compression = 0)
#         self.assertEqual(str(os.popen("tiffdump test.tif").read()),
#                          "test.tif:\n"
#                          "Magic: 0x4949 <little-endian> Version: 0x2a\n"
#                          "Directory 0: offset 524688 (0x80190) next 0 (0)\n"
#                          "ImageWidth (256) SHORT (3) 1<512>\n"
#                          "ImageLength (257) SHORT (3) 1<512>\n"
#                          "BitsPerSample (258) SHORT (3) 2<8 8>\n"
#                          "Compression (259) SHORT (3) 1<1>\n"
#                          "Photometric (262) SHORT (3) 1<1>\n"
#                          "SamplesPerPixel (277) SHORT (3) 1<2>\n"
#                          "PlanarConfig (284) SHORT (3) 1<1>\n"
#                          "DateTime (306) ASCII (2) 20<2009:10:08 14:30:00\\0>\n"
#                          "TileWidth (322) SHORT (3) 1<256>\n"
#                          "TileLength (323) SHORT (3) 1<256>\n"
#                          "TileOffsets (324) LONG (4) 4<400 131472 262544 393616"
#                          ">\n"
#                          "TileByteCounts (325) LONG (4) 4<131072 131072 131072 "
#                          "131072>\n"
#                          "ExtraSamples (338) SHORT (3) 1<0>\n"
#                          "SampleFormat (339) SHORT (3) 2<1 1>\n"
#                          "33550 (0x830e) DOUBLE (12) 3<8000 8000 0>\n"
#                          "33922 (0x8482) DOUBLE (12) 6<0 0 0 -2.71718e+06 -1.47"
#                          "505e+06 0>\n"
#                          "34735 (0x87af) SHORT (3) 64<1 1 0 15 1024 0 1 1 1025 "
#                          "0 1 1 1026 34737 8 0 2048 0 1 4326 2049 34737 7 8 ..."
#                          ">\n"
#                          "34736 (0x87b0) DOUBLE (12) 5<60 14 1 0 0>\n"
#                          "34737 (0x87b1) ASCII (2) 16<unnamed|WGS 84|\\0>\n"
#                          "42112 (0xa480) ASCII (2) 154<<GDALMetadata>\\n  <Item"
#                          " n ...>\n")

        
#         os.remove("test.tif")

#         self.img.save("test.tif", compression = 6)
#         self.assertEqual(os.popen("tiffdump test.tif").read(),
#                          "test.tif:\n"
#                          "Magic: 0x4949 <little-endian> Version: 0x2a\n"
#                          "Directory 0: offset 1012 (0x3f4) next 0 (0)\n"
#                          "ImageWidth (256) SHORT (3) 1<512>\n"
#                          "ImageLength (257) SHORT (3) 1<512>\n"
#                          "BitsPerSample (258) SHORT (3) 2<8 8>\n"
#                          "Compression (259) SHORT (3) 1<8>\n"
#                          "Photometric (262) SHORT (3) 1<1>\n"
#                          "SamplesPerPixel (277) SHORT (3) 1<2>\n"
#                          "PlanarConfig (284) SHORT (3) 1<1>\n"
#                          "DateTime (306) ASCII (2) 20<2009:10:08 14:30:00\\0>\n"
#                          "Predictor (317) SHORT (3) 1<1>\n"
#                          "TileWidth (322) SHORT (3) 1<256>\n"
#                          "TileLength (323) SHORT (3) 1<256>\n"
#                          "TileOffsets (324) LONG (4) 4<412 562 712 862>\n"
#                          "TileByteCounts (325) LONG (4) 4<150 150 150 150>\n"
#                          "ExtraSamples (338) SHORT (3) 1<0>\n"
#                          "SampleFormat (339) SHORT (3) 2<1 1>\n"
#                          "33550 (0x830e) DOUBLE (12) 3<8000 8000 0>\n"
#                          "33922 (0x8482) DOUBLE (12) 6<0 0 0 -2.71718e+06 -1.47"
#                          "505e+06 0>\n"
#                          "34735 (0x87af) SHORT (3) 64<1 1 0 15 1024 0 1 1 1025 "
#                          "0 1 1 1026 34737 8 0 2048 0 1 4326 2049 34737 7 8 ..."
#                          ">\n"
#                          "34736 (0x87b0) DOUBLE (12) 5<60 14 1 0 0>\n"
#                          "34737 (0x87b1) ASCII (2) 16<unnamed|WGS 84|\\0>\n"
#                          "42112 (0xa480) ASCII (2) 154<<GDALMetadata>\\n  <Item"
#                          " n ...>\n")

#         os.remove("test.tif")


if __name__ == "__main__":
    unittest.main()
