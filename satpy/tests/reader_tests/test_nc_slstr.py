#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module for testing the satpy.readers.nc_slstr module.
"""

try:
    import unittest.mock as mock
except ImportError:
    import mock
from unittest import TestCase


def mock_dataset():
    """Create a mock xarray"""
    return mock.MagicMock()


class TestSLSTRReader(TestCase):

    @mock.patch('xarray.open_dataset')
    def test_instantiate(self, mocked_Dataset):

        mocked_Dataset.return_value = mock_dataset()

        from satpy.readers.nc_slstr import NCSLSTR1B, NCSLSTRGeo, NCSLSTRAngles, NCSLSTRFlag

        class Dummy():
            def __init__(self):
                self.name = 'foo'

        class Base():
            def copy(self):
                return None

            def __getitem__(self, a):
                return 'S3A'

            def update(self, a):
                return

        test = NCSLSTR1B('somedir/S1_radiance_an.nc', Base(), 'c')
        assert(test.view == 'n')
        assert(test.stripe == 'a')
        test.get_dataset(Dummy(), Base())
        mocked_Dataset.assert_called()
        mocked_Dataset.reset_mock()

        test = NCSLSTR1B('somedir/S1_radiance_co.nc', Base(), 'c')
        assert(test.view == 'o')
        assert(test.stripe == 'c')
        test.get_dataset(Dummy(), Base())
        mocked_Dataset.assert_called()
        mocked_Dataset.reset_mock()

        test = NCSLSTRGeo('somedir/S1_radiance_an.nc', Base(), 'c')
        test.get_dataset(Dummy(), Base())
        mocked_Dataset.assert_called()
        mocked_Dataset.reset_mock()
        
        
        test = NCSLSTRAngles('somedir/S1_radiance_an.nc', Base(), 'c')
        mocked_Dataset.assert_called()
        mocked_Dataset.reset_mock()

        test = NCSLSTRFlag('somedir/S1_radiance_an.nc', Base(), 'c')
        assert(test.view == 'n')
        assert(test.stripe == 'a')
        mocked_Dataset.assert_called()
        mocked_Dataset.reset_mock()
        
