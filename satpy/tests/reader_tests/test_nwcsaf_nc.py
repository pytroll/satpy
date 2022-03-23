#!/usr/bin/env python
# Copyright (c) 2018-2022 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Unittests for NWC SAF reader."""

import unittest
from unittest import mock

import numpy as np
import pytest
import xarray as xr

PROJ_KM = {'gdal_projection': '+proj=geos +a=6378.137000 +b=6356.752300 +lon_0=0.000000 +h=35785.863000',
           'gdal_xgeo_up_left': -5569500.0,
           'gdal_ygeo_up_left': 5437500.0,
           'gdal_xgeo_low_right': 5566500.0,
           'gdal_ygeo_low_right': 2653500.0}
PROJ = {'gdal_projection': '+proj=geos +a=6378137.000 +b=6356752.300 +lon_0=0.000000 +h=35785863.000',
        'gdal_xgeo_up_left': -5569500.0,
        'gdal_ygeo_up_left': 5437500.0,
        'gdal_xgeo_low_right': 5566500.0,
        'gdal_ygeo_low_right': 2653500.0}


class TestNcNWCSAF(unittest.TestCase):
    """Test the NcNWCSAF reader."""

    @mock.patch('satpy.readers.nwcsaf_nc.unzip_file')
    @mock.patch('satpy.readers.nwcsaf_nc.xr.open_dataset')
    def setUp(self, xr_open_dataset, unzip):
        """Set up the test case."""
        from satpy.readers.nwcsaf_nc import NcNWCSAF
        xr_open_dataset.return_value = xr.Dataset({"nx": xr.DataArray(), "ny": xr.DataArray()},
                                                  attrs={"source": "bla",
                                                         "satellite_identifier": "blu"})
        self.fake_dataset = xr_open_dataset.return_value
        unzip.return_value = ''
        self.filehandler_class = NcNWCSAF
        self.fh = self.filehandler_class('filename', {}, {})

    def test_sensor_name(self):
        """Test that the correct sensor name is being set."""
        self.fh.set_platform_and_sensor(platform_name='Metop-B')
        self.assertEqual(self.fh.sensor, set(['avhrr-3']))
        self.assertEqual(self.fh.sensor_names, set(['avhrr-3']))

        self.fh.set_platform_and_sensor(platform_name='NOAA-20')
        self.assertEqual(self.fh.sensor, set(['viirs']))
        self.assertEqual(self.fh.sensor_names, set(['viirs']))

        self.fh.set_platform_and_sensor(platform_name='Himawari-8')
        self.assertEqual(self.fh.sensor, set(['ahi']))
        self.assertEqual(self.fh.sensor_names, set(['ahi']))

        self.fh.set_platform_and_sensor(sat_id='GOES16')
        self.assertEqual(self.fh.sensor, set(['abi']))
        self.assertEqual(self.fh.sensor_names, set(['abi']))

        self.fh.set_platform_and_sensor(platform_name='GOES-17')
        self.assertEqual(self.fh.sensor, set(['abi']))
        self.assertEqual(self.fh.sensor_names, set(['abi']))

        self.fh.set_platform_and_sensor(sat_id='MSG4')
        self.assertEqual(self.fh.sensor, set(['seviri']))

        self.fh.set_platform_and_sensor(platform_name='Meteosat-11')
        self.assertEqual(self.fh.sensor, set(['seviri']))
        self.assertEqual(self.fh.sensor_names, set(['seviri']))

    def test_get_area_def(self):
        """Test that get_area_def() returns proper area."""
        dsid = {'name': 'foo'}
        self.fh.nc[dsid['name']] = xr.DataArray(np.zeros((5, 10)))

        # a, b and h in kilometers
        self.fh.nc.attrs = PROJ_KM
        _check_area_def(self.fh.get_area_def(dsid))

        # a, b and h in meters
        self.fh.nc.attrs = PROJ
        _check_area_def(self.fh.get_area_def(dsid))

    def test_scale_dataset_attr_removal(self):
        """Test the scaling of the dataset and removal of obsolete attributes."""
        import numpy as np
        import xarray as xr

        attrs = {'scale_factor': np.array(10),
                 'add_offset': np.array(20)}
        var = xr.DataArray([1, 2, 3], attrs=attrs)
        var = self.fh.scale_dataset(var, 'dummy')
        np.testing.assert_allclose(var, [30, 40, 50])
        self.assertNotIn('scale_factor', var.attrs)
        self.assertNotIn('add_offset', var.attrs)

    def test_scale_dataset_floating(self):
        """Test the scaling of the dataset with floating point values."""
        import numpy as np
        import xarray as xr
        attrs = {'scale_factor': np.array(1.5),
                 'add_offset': np.array(2.5),
                 '_FillValue': 1}
        var = xr.DataArray([1, 2, 3], attrs=attrs)
        var = self.fh.scale_dataset(var, 'dummy')
        np.testing.assert_allclose(var, [np.nan, 5.5, 7])
        self.assertNotIn('scale_factor', var.attrs)
        self.assertNotIn('add_offset', var.attrs)

        attrs = {'scale_factor': np.array(1.5),
                 'add_offset': np.array(2.5),
                 'valid_min': 1.1}
        var = xr.DataArray([1, 2, 3], attrs=attrs)
        var = self.fh.scale_dataset(var, 'dummy')
        np.testing.assert_allclose(var, [np.nan, 5.5, 7])
        self.assertNotIn('scale_factor', var.attrs)
        self.assertNotIn('add_offset', var.attrs)

        attrs = {'scale_factor': np.array(1.5),
                 'add_offset': np.array(2.5),
                 'valid_max': 2.1}
        var = xr.DataArray([1, 2, 3], attrs=attrs)
        var = self.fh.scale_dataset(var, 'dummy')
        np.testing.assert_allclose(var, [4, 5.5, np.nan])
        self.assertNotIn('scale_factor', var.attrs)
        self.assertNotIn('add_offset', var.attrs)

        attrs = {'scale_factor': np.array(1.5),
                 'add_offset': np.array(2.5),
                 'valid_range': (1.1, 2.1)}
        var = xr.DataArray([1, 2, 3], attrs=attrs)
        var = self.fh.scale_dataset(var, 'dummy')
        np.testing.assert_allclose(var, [np.nan, 5.5, np.nan])
        self.assertNotIn('scale_factor', var.attrs)
        self.assertNotIn('add_offset', var.attrs)

        # CTTH NWCSAF/Geo v2016/v2018:
        attrs = {'scale_factor': np.array(1.),
                 'add_offset': np.array(-2000.),
                 'valid_range': (0., 27000.)}
        var = xr.DataArray([1, 2, 3], attrs=attrs)
        var = self.fh.scale_dataset(var, 'dummy')
        np.testing.assert_allclose(var, [-1999., -1998., -1997.])
        self.assertNotIn('scale_factor', var.attrs)
        self.assertNotIn('add_offset', var.attrs)
        self.assertEqual(var.attrs['valid_range'][0], -2000.)
        self.assertEqual(var.attrs['valid_range'][1], 25000.)

    def test_get_dataset_scales_and_offsets(self):
        """Test that get_dataset() returns scaled and offseted data."""
        dsid = {'name': 'cpp_cot'}
        scale = 4
        offset = 8
        the_array = xr.DataArray(np.ones((5, 10)), attrs={"scale_factor": np.array(scale, dtype=float),
                                                          "add_offset": np.array(offset, dtype=float)})
        self.fh.nc[dsid['name']] = the_array

        info = dict(name="cpp_cot",
                    file_type="nc_nwcsaf_cpp")

        res = self.fh.get_dataset(dsid, info)
        np.testing.assert_allclose(res, the_array * scale + offset)

    def test_get_dataset_scales_and_offsets_palette_meanings_using_other_dataset(self):
        """Test that get_dataset() returns scaled palette_meanings while another dataset as scaling source."""
        dsid = {'name': 'cpp_cot'}
        scale = 4
        offset = 8
        array = xr.DataArray(np.ones((5, 3)), attrs={"palette_meanings": "1 2 3 4",
                                                     "fill_value_color": (0, 0, 0)})
        self.fh.nc[dsid['name']] = array

        so_array = xr.DataArray(np.ones((10, 10)),
                                attrs={"scale_factor": np.array(scale, dtype=float),
                                       "add_offset": np.array(offset, dtype=float)},
                                dims=["lines", "colors"])

        info = dict(name="cpp_cot",
                    file_type="nc_nwcsaf_cpp",
                    scale_offset_dataset="scaleoffset")
        self.fh.nc["scaleoffset"] = so_array

        res = self.fh.get_dataset(dsid, info)
        np.testing.assert_allclose(res.attrs["palette_meanings"], np.arange(5) * scale + offset)

    def test_get_dataset_raises_when_dataset_missing(self):
        """Test that get_dataset() raises an error when the requested dataset is missing."""
        dsid = {'name': 'cpp_cot'}
        info = dict(name="cpp_cot",
                    file_type="nc_nwcsaf_cpp")
        with pytest.raises(KeyError):
            self.fh.get_dataset(dsid, info)

    def test_get_dataset_uses_file_key_if_present(self):
        """Test that get_dataset() uses a file_key if present."""
        dsid_cpp = {'name': 'cpp_cot'}
        dsid_cmic = {'name': 'cmic_cot'}
        scale = 4
        offset = 8
        the_array = xr.DataArray(np.ones((5, 10)), attrs={"scale_factor": np.array(scale, dtype=float),
                                                          "add_offset": np.array(offset, dtype=float)})
        file_key = "cmic_cot"
        self.fh.nc[file_key] = the_array

        info_cpp = dict(name="cpp_cot",
                        file_key=file_key,
                        file_type="nc_nwcsaf_cpp")

        res_cpp = self.fh.get_dataset(dsid_cpp, info_cpp)

        info_cmic = dict(name="cmic_cot",
                         file_type="nc_nwcsaf_cpp")

        res_cmic = self.fh.get_dataset(dsid_cmic, info_cmic)
        np.testing.assert_allclose(res_cpp, res_cmic)


class TestNcNWCSAFFileKeyPrefix(unittest.TestCase):
    """Test the NcNWCSAF reader when using a file key prefix."""

    @mock.patch('satpy.readers.nwcsaf_nc.unzip_file')
    @mock.patch('satpy.readers.nwcsaf_nc.xr.open_dataset')
    def setUp(self, xr_open_dataset, unzip):
        """Set up the test case."""
        from satpy.readers.nwcsaf_nc import NcNWCSAF
        xr_open_dataset.return_value = xr.Dataset({"nx": xr.DataArray(), "ny": xr.DataArray()},
                                                  attrs={"source": "bla",
                                                         "satellite_identifier": "blu"})
        self.fake_dataset = xr_open_dataset.return_value
        unzip.return_value = ''
        self.filehandler_class = NcNWCSAF
        self.file_key_prefix = "cmic_"
        self.fh = self.filehandler_class('filename', {}, {"file_key_prefix": self.file_key_prefix})

    def test_get_dataset_uses_file_key_prefix(self):
        """Test that get_dataset() uses a file_key_prefix."""
        dsid_cpp = {'name': 'cpp_cot'}
        dsid_cmic = {'name': 'cmic_cot'}
        scale = 4
        offset = 8
        the_array = xr.DataArray(np.ones((5, 10)), attrs={"scale_factor": np.array(scale, dtype=float),
                                                          "add_offset": np.array(offset, dtype=float)})
        file_key = "cot"
        self.fh.nc[self.file_key_prefix + file_key] = the_array

        info_cpp = dict(name="cpp_cot",
                        file_key=file_key,
                        file_type="nc_nwcsaf_cpp")

        res_cpp = self.fh.get_dataset(dsid_cpp, info_cpp)

        info_cmic = dict(name="cmic_cot",
                         file_type="nc_nwcsaf_cpp")

        res_cmic = self.fh.get_dataset(dsid_cmic, info_cmic)
        np.testing.assert_allclose(res_cpp, res_cmic)

    def test_get_dataset_scales_and_offsets_palette_meanings_using_other_dataset(self):
        """Test that get_dataset() returns scaled palette_meanings using another dataset as scaling source."""
        dsid = {'name': 'cpp_cot_pal'}
        scale = 4
        offset = 8
        array = xr.DataArray(np.ones((5, 3)), attrs={"palette_meanings": "1 2 3 4",
                                                     "fill_value_color": (0, 0, 0)})
        self.fh.nc[dsid['name']] = array

        so_array = xr.DataArray(np.ones((10, 10)),
                                attrs={"scale_factor": np.array(scale, dtype=float),
                                       "add_offset": np.array(offset, dtype=float)},
                                dims=["lines", "colors"])

        info = dict(name="cpp_cot_pal",
                    file_type="nc_nwcsaf_cpp",
                    scale_offset_dataset="scaleoffset")
        self.fh.nc[self.file_key_prefix + "scaleoffset"] = so_array

        res = self.fh.get_dataset(dsid, info)
        np.testing.assert_allclose(res.attrs["palette_meanings"], np.arange(5) * scale + offset)


def _check_area_def(area_definition):
    correct_h = float(PROJ['gdal_projection'].split('+h=')[-1])
    correct_a = float(PROJ['gdal_projection'].split('+a=')[-1].split()[0])
    assert area_definition.proj_dict['h'] == correct_h
    assert area_definition.proj_dict['a'] == correct_a
    assert area_definition.proj_dict['units'] == 'm'
    correct_extent = (PROJ["gdal_xgeo_up_left"],
                      PROJ["gdal_ygeo_low_right"],
                      PROJ["gdal_xgeo_low_right"],
                      PROJ["gdal_ygeo_up_left"])
    assert area_definition.area_extent == correct_extent
