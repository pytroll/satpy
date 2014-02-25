#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module for testing the mpop.channel module.
"""

import unittest
from datetime import datetime
from mpop.satin.viirs_sdr import _get_swathsegment

class TestViirsSDRReader(unittest.TestCase):
    """Class for testing the VIIRS SDR reader class.
    """
    
    def test_get_swath_segment(self):
        """
        Test choosing swath segments based on datatime interval
        """
        
        filenames = [
            "SVM15_npp_d20130312_t1034305_e1035546_b07108_c20130312110058559507_cspp_dev.h5", 
            "SVM15_npp_d20130312_t1035559_e1037201_b07108_c20130312110449303310_cspp_dev.h5",
            "SVM15_npp_d20130312_t1037213_e1038455_b07108_c20130312110755391459_cspp_dev.h5",
            "SVM15_npp_d20130312_t1038467_e1040109_b07108_c20130312111106961103_cspp_dev.h5",
            "SVM15_npp_d20130312_t1040121_e1041363_b07108_c20130312111425464510_cspp_dev.h5",
            "SVM15_npp_d20130312_t1041375_e1043017_b07108_c20130312111720550253_cspp_dev.h5",
            "SVM15_npp_d20130312_t1043029_e1044271_b07108_c20130312112246726129_cspp_dev.h5",
            "SVM15_npp_d20130312_t1044283_e1045525_b07108_c20130312113037160389_cspp_dev.h5",
            "SVM15_npp_d20130312_t1045537_e1047179_b07108_c20130312114330237590_cspp_dev.h5",
            "SVM15_npp_d20130312_t1047191_e1048433_b07108_c20130312120148075096_cspp_dev.h5",
            "SVM15_npp_d20130312_t1048445_e1050070_b07108_c20130312120745231147_cspp_dev.h5",
            ]
       

        #
        # Test search for multiple granules
        result = [
            "SVM15_npp_d20130312_t1038467_e1040109_b07108_c20130312111106961103_cspp_dev.h5",
            "SVM15_npp_d20130312_t1040121_e1041363_b07108_c20130312111425464510_cspp_dev.h5",
            "SVM15_npp_d20130312_t1041375_e1043017_b07108_c20130312111720550253_cspp_dev.h5",
            "SVM15_npp_d20130312_t1043029_e1044271_b07108_c20130312112246726129_cspp_dev.h5",
            "SVM15_npp_d20130312_t1044283_e1045525_b07108_c20130312113037160389_cspp_dev.h5",
            ]

        tstart = datetime(2013, 3, 12, 10, 39)
        tend = datetime(2013, 3, 12, 10, 45)

        

        sublist = _get_swathsegment(filenames, tstart, tend)

        self.assert_(sublist == result)


        #
        # Test search for single granule
        tslot = datetime(2013, 3, 12, 10, 45)

        result_file = [
            "SVM15_npp_d20130312_t1044283_e1045525_b07108_c20130312113037160389_cspp_dev.h5",
            ]
        
        single_file = _get_swathsegment(filenames, tslot)

        self.assert_(result_file == single_file)

def suite():
    """The test suite for test_viirs_sdr.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestViirsSDRReader))
    
    return mysuite
