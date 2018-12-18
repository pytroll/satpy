"""The enhancements tests package.
"""

import sys

from satpy.tests.enhancement_tests import test_enhancements, test_viirs

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


def suite():
    """Test suite for all enhancement tests"""
    mysuite = unittest.TestSuite()
    mysuite.addTests(test_enhancements.suite())
    mysuite.addTests(test_viirs.suite())

    return mysuite
