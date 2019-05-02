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
from datetime import datetime
from satpy import DatasetID

import numpy as np

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class TestCFWriter(unittest.TestCase):
    def test_init(self):
        from satpy.writers.cf_writer import CFWriter
        import satpy.config
        CFWriter(config_files=[os.path.join(satpy.config.CONFIG_PATH,
                                            'writers', 'cf.yaml')])

    def test_save_array(self):
        from satpy import Scene
        import xarray as xr
        import tempfile
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        scn['test-array'] = xr.DataArray([1, 2, 3],
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time,
                                                    prerequisites=[DatasetID('hej')]))
        try:
            handle, filename = tempfile.mkstemp()
            os.close(handle)
            scn.save_datasets(filename=filename, writer='cf')
            import h5netcdf as nc4
            with nc4.File(filename) as f:
                self.assertTrue(all(f['test-array'][:] == [1, 2, 3]))
                expected_prereq = ("DatasetID(name='hej', wavelength=None, "
                                   "resolution=None, polarization=None, "
                                   "calibration=None, level=None, modifiers=())")
                self.assertEqual(f['test-array'].attrs['prerequisites'][0],
                                 np.string_(expected_prereq))
        finally:
            os.remove(filename)

    def test_single_time_value(self):
        from satpy import Scene
        import xarray as xr
        import tempfile
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        test_array = np.array([[1, 2], [3, 4]])
        scn['test-array'] = xr.DataArray(test_array,
                                         dims=['x', 'y'],
                                         coords={'time': np.datetime64('2018-05-30T10:05:00')},
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time))
        try:
            handle, filename = tempfile.mkstemp()
            os.close(handle)
            scn.save_datasets(filename=filename, writer='cf')
            import h5netcdf as nc4
            with nc4.File(filename) as f:
                self.assertTrue(all(f['time_bnds'][:] == np.array([-300.,  600.])))
        finally:
            os.remove(filename)

    def test_bounds(self):
        from satpy import Scene
        import xarray as xr
        import tempfile
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        test_array = np.array([[1, 2], [3, 4]]).reshape(2, 2, 1)
        scn['test-array'] = xr.DataArray(test_array,
                                         dims=['x', 'y', 'time'],
                                         coords={'time': [np.datetime64('2018-05-30T10:05:00')]},
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time))
        try:
            handle, filename = tempfile.mkstemp()
            os.close(handle)
            scn.save_datasets(filename=filename, writer='cf')
            import h5netcdf as nc4
            with nc4.File(filename) as f:
                self.assertTrue(all(f['time_bnds'][:] == np.array([-300.,  600.])))
        finally:
            os.remove(filename)

    def test_bounds_minimum(self):
        from satpy import Scene
        import xarray as xr
        import tempfile
        scn = Scene()
        start_timeA = datetime(2018, 5, 30, 10, 0)  # expected to be used
        end_timeA = datetime(2018, 5, 30, 10, 20)
        start_timeB = datetime(2018, 5, 30, 10, 3)
        end_timeB = datetime(2018, 5, 30, 10, 15)  # expected to be used
        test_arrayA = np.array([[1, 2], [3, 4]]).reshape(2, 2, 1)
        test_arrayB = np.array([[1, 2], [3, 5]]).reshape(2, 2, 1)
        scn['test-arrayA'] = xr.DataArray(test_arrayA,
                                          dims=['x', 'y', 'time'],
                                          coords={'time': [np.datetime64('2018-05-30T10:05:00')]},
                                          attrs=dict(start_time=start_timeA,
                                                     end_time=end_timeA))
        scn['test-arrayB'] = xr.DataArray(test_arrayB,
                                          dims=['x', 'y', 'time'],
                                          coords={'time': [np.datetime64('2018-05-30T10:05:00')]},
                                          attrs=dict(start_time=start_timeB,
                                                     end_time=end_timeB))
        try:
            handle, filename = tempfile.mkstemp()
            os.close(handle)
            scn.save_datasets(filename=filename, writer='cf')
            import h5netcdf as nc4
            with nc4.File(filename) as f:
                self.assertTrue(all(f['time_bnds'][:] == np.array([-300.,  600.])))
        finally:
            os.remove(filename)

    def test_bounds_missing_time_info(self):
        from satpy import Scene
        import xarray as xr
        import tempfile
        scn = Scene()
        start_timeA = datetime(2018, 5, 30, 10, 0)
        end_timeA = datetime(2018, 5, 30, 10, 15)
        test_arrayA = np.array([[1, 2], [3, 4]]).reshape(2, 2, 1)
        test_arrayB = np.array([[1, 2], [3, 5]]).reshape(2, 2, 1)
        scn['test-arrayA'] = xr.DataArray(test_arrayA,
                                          dims=['x', 'y', 'time'],
                                          coords={'time': [np.datetime64('2018-05-30T10:05:00')]},
                                          attrs=dict(start_time=start_timeA,
                                                     end_time=end_timeA))
        scn['test-arrayB'] = xr.DataArray(test_arrayB,
                                          dims=['x', 'y', 'time'],
                                          coords={'time': [np.datetime64('2018-05-30T10:05:00')]})
        try:
            handle, filename = tempfile.mkstemp()
            os.close(handle)
            scn.save_datasets(filename=filename, writer='cf')
            import h5netcdf as nc4
            with nc4.File(filename) as f:
                self.assertTrue(all(f['time_bnds'][:] == np.array([-300.,  600.])))
        finally:
            os.remove(filename)

    def test_encoding_kwarg(self):
        from satpy import Scene
        import xarray as xr
        import tempfile
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        scn['test-array'] = xr.DataArray([1, 2, 3],
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time))
        try:
            handle, filename = tempfile.mkstemp()
            os.close(handle)
            encoding = {'test-array': {'dtype': 'int8',
                                       'scale_factor': 0.1,
                                       'add_offset': 0.0,
                                       '_FillValue': 3}}
            scn.save_datasets(filename=filename, encoding=encoding, writer='cf')
            import h5netcdf as nc4
            with nc4.File(filename) as f:
                self.assertTrue(all(f['test-array'][:] == [10, 20, 30]))
                self.assertTrue(f['test-array'].attrs['scale_factor'] == 0.1)
                self.assertTrue(f['test-array'].attrs['_FillValue'] == 3)
                # check that dtype behave as int8
                self.assertTrue(np.iinfo(f['test-array'][:].dtype).max == 127)
        finally:
            os.remove(filename)

    def test_header_attrs(self):
        from satpy import Scene
        import xarray as xr
        import tempfile
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        scn['test-array'] = xr.DataArray([1, 2, 3],
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time))
        try:
            handle, filename = tempfile.mkstemp()
            os.close(handle)
            header_attrs = {'sensor': 'SEVIRI',
                            'orbit': None}
            scn.save_datasets(filename=filename,
                              header_attrs=header_attrs,
                              writer='cf')
            import h5netcdf as nc4
            with nc4.File(filename) as f:
                self.assertTrue(f.attrs['sensor'] == 'SEVIRI')
                self.assertTrue('sensor' in f.attrs.keys())
                self.assertTrue('orbit' not in f.attrs.keys())
        finally:
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
