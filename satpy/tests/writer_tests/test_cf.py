#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 David Hoese
#
# Author(s):
#
#   David Hoese <david.hoese@ssec.wisc.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Tests for the CF writer.
"""
import os
import sys

import numpy as np

try:
    from unittest import mock
except ImportError:
    import mock

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class TestCFWriter(unittest.TestCase):
    def test_init(self):
        from satpy.writers.cf_writer import CFWriter
        import satpy.config
        w = CFWriter(config_files=[os.path.join(satpy.config.CONFIG_PATH, 
                                                'writers', 'cf.yaml')])
        

    def test_save_array(self):
        from satpy import Scene
        import xarray as xr
        import tempfile
        scn = Scene()
        scn['test-array'] = xr.DataArray([1,2,3])
        handle, filename = tempfile.mkstemp()
        scn.save_datasets(filename=filename, writer='cf')
        import h5netcdf as nc4
        f = nc4.File(filename)
        self.assertTrue(all(f['test-array'][:] == [1,2,3]))
        os.remove(filename)

    def test_encoding_kwarg(self):
        from satpy import Scene
        import xarray as xr
        import tempfile
        scn = Scene()
        scn['test-array'] = xr.DataArray([1,2,3])
        handle, filename = tempfile.mkstemp()
        encoding = {'test-array': {'dtype': 'int8',
                                   'scale_factor': 0.1,
                                   'add_offset': 0.0,
                                   '_FillValue': 3 }}
        scn.save_datasets(filename=filename, encoding=encoding, writer='cf')
        import h5netcdf as nc4
        f = nc4.File(filename)
        self.assertTrue(all(f['test-array'][:] == [10,20,30]))
        self.assertTrue(f['test-array'].attrs['scale_factor'] == 0.1)
        self.assertTrue(f['test-array'].attrs['_FillValue'] == 3)
        #check that dtype behave as int8
        self.assertTrue(np.iinfo(f['test-array'][:].dtype).max == 127)
        os.remove(filename)

    def test_header_attrs(self):
        from satpy import Scene
        import xarray as xr
        import tempfile
        scn = Scene()
        scn['test-array'] = xr.DataArray([1,2,3])
        handle, filename = tempfile.mkstemp()
        header_attrs= {'sensor': 'SEVIRI',
                       'orbit': None}
        scn.save_datasets(filename=filename,
                          header_attrs=header_attrs,
                          writer='cf')
        import h5netcdf as nc4
        f = nc4.File(filename)
        self.assertTrue(f.attrs['sensor'] == 'SEVIRI')
        self.assertTrue('sensor' in f.attrs.keys())
        self.assertTrue('orbit' not in f.attrs.keys())
        os.remove(filename)




def suite():
    """The test suite for this writer's tests.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestCFWriter))
    return mysuite

if __name__ == "__main__":
    unittest.main()
