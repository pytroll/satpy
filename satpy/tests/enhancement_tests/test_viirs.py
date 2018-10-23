"""Unit testing for the VIIRS enhancement function
"""

import unittest
import numpy as np
import xarray as xr
import dask.array as da
from .test_enhancements import TestEnhancementStretch

class TestVIIRSEnhancement(unittest.TestCase):
    """Class for testing the VIIRS enhancement function in satpy.enhancements.viirs"""
    
    def setUp(self):
        """Setup the test"""
        data = np.arange(15, 301, 15).reshape(2, 10)
        self.da = xr.DataArray(data, dims=('y', 'x'), attrs={'test' : 'test'})
        self.pal = {'colors': 
                [[14, [0.0, 0.0, 0.0]], 
                [15, [0.0, 0.0, 0.39215686274509803]], 
                [16, [0.7686274509803922, 0.6352941176470588, 0.4470588235294118]], 
                [17, [0.7686274509803922, 0.6352941176470588, 0.4470588235294118]], 
                [18, [0.0, 0.0, 1.0]], 
                [20, [1.0, 1.0, 1.0]], 
                [27, [0.0, 1.0, 1.0]], 
                [30, [0.7843137254901961, 0.7843137254901961, 0.7843137254901961]], 
                [31, [0.39215686274509803, 0.39215686274509803, 0.39215686274509803]], 
                [88, [0.7058823529411765, 0.0, 0.9019607843137255]], 
                [100, [0.19607843137254902, 1.0, 0.39215686274509803]], 
                [120, [0.19607843137254902, 1.0, 0.39215686274509803]], 
                [121, [0.0, 1.0, 0.0]], 
                [130, [0.0, 1.0, 0.0]], 
                [131, [0.7843137254901961, 1.0, 0.0]], 
                [140, [0.7843137254901961, 1.0, 0.0]], 
                [141, [1.0, 1.0, 0.5882352941176471]], 
                [150, [1.0, 1.0, 0.5882352941176471]], 
                [151, [1.0, 1.0, 0.0]], 
                [160, [1.0, 1.0, 0.0]], 
                [161, [1.0, 0.7843137254901961, 0.0]], 
                [170, [1.0, 0.7843137254901961, 0.0]], 
                [171, [1.0, 0.5882352941176471, 0.19607843137254902]], 
                [180, [1.0, 0.5882352941176471, 0.19607843137254902]], 
                [181, [1.0, 0.39215686274509803, 0.0]], 
                [190, [1.0, 0.39215686274509803, 0.0]], 
                [191, [1.0, 0.0, 0.0]], 
                [200, [1.0, 0.0, 0.0]], 
                [201, [0.0, 0.0, 0.0]]], 
                'min_value': 0,
                'max_value': 201}
    
    def test_viirs(self):
        from satpy.enhancements.viirs import water_detection
        expected = [[[1, 7, 8, 8, 8, 9, 10, 11, 14, 8], [20, 23, 26, 10, 12, 15, 18, 21, 24, 27]]] 
        TestEnhancementStretch._test_enhancement(self, water_detection, self.da, expected, palettes=self.pal)
        
    def tearDown(self):
        """Clean up"""
        pass

def suite():
    """The test suite for test_viirs.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestVIIRSEnhancement))
    
    return mysuite

