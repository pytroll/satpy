#!/usr/bin/env python
# Copyright (c) 2018-2023 Satpy developers
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

import h5netcdf
import numpy as np
import pytest
import xarray as xr

from satpy.readers.nwcsaf_nc import NcNWCSAF, read_nwcsaf_time

PROJ_KM = {'gdal_projection': '+proj=geos +a=6378.137000 +b=6356.752300 +lon_0=0.000000 +h=35785.863000',
           'gdal_xgeo_up_left': -5569500.0,
           'gdal_ygeo_up_left': 5437500.0,
           'gdal_xgeo_low_right': 5566500.0,
           'gdal_ygeo_low_right': 2653500.0}

NOMINAL_ALTITUDE = 35785863.0

PROJ = {'gdal_projection': f'+proj=geos +a=6378137.000 +b=6356752.300 +lon_0=0.000000 +h={NOMINAL_ALTITUDE:.3f}',
        'gdal_xgeo_up_left': -5569500.0,
        'gdal_ygeo_up_left': 5437500.0,
        'gdal_xgeo_low_right': 5566500.0,
        'gdal_ygeo_low_right': 2653500.0}


dimensions = {"nx": 1530,
              "ny": 928,
              "pal_colors_250": 250,
              "pal_rgb": 3}

NOMINAL_LONGITUDE = 0.0
START_TIME = "2023-01-18T10:39:17Z"
END_TIME = "2023-01-18T10:42:22Z"
START_TIME_PPS = "20230118T103917000Z"
END_TIME_PPS = "20230118T104222000Z"

global_attrs = {"source": "NWC/GEO version v2021.1",
                "satellite_identifier": "MSG4",
                "sub-satellite_longitude": NOMINAL_LONGITUDE,
                "time_coverage_start": START_TIME,
                "time_coverage_end": END_TIME}

global_attrs.update(PROJ)

COT_PALETTE_MEANINGS = ("0 2 5 8 10 13 16 19 23 26 29 33 36 40 43 47 51 55 59 63 68 72 77 81 86 91 96"
                        " 101 107 112 118 123 129 135 142 148 154 161 168 175 182 190 198 205 213 222"
                        " 230 239 248 257 266 276 286 296 307 317 328 340 351 363 375 388 401 414 428"
                        " 442 456 470 485 501 517 533 550 567 584 602 621 640 660 680 700 721 743 765"
                        " 788 811 835 860 885 911 938 965 993 1022 1052 1082 1113 1145 1178 1212 1246"
                        " 1282 1318 1355 1394 1433 1474 1515 1558 1601 1646 1692 1739 1788 1837 1889 "
                        "1941 1995 2050 2107 2165 2224 2286 2348 2413 2479 2547 2617 2688 2762 2837 "
                        "2915 2994 3076 3159 3245 3333 3424 3517 3612 3710 3810 3913 4019 4127 4239 "
                        "4353 4470 4591 4714 4841 4971 5105 5242 5383 5527 5676 5828 5984 6144 6309 "
                        "6478 6651 6829 7011 7199 7391 7588 7791 7999 8212 8431 8656 8886 9123 9366 "
                        "9615 9871 10134 10404 10680 10964 11256 11555 11862 12177 12501 12833 13173 "
                        "13523 13882 14250 14628 15016 15414 15823 16243 16673 17115 17569 18034 18512"
                        " 19002 19505 20022 20552 21096 21654 22227 22816 23419 24039 24675 25327 "
                        "25997 26685 27390 28114 28858 29621 30404 31207 32032 32878 33747 34639 35554"
                        " 36493 37457 38446 39462 40504 41574 42672 43798 44955 46142 47360 48610 "
                        "49893 51210 52562 53949 55373 56834 58334 59873 61453 63075 64739")

COT_SCALE = 0.01
COT_OFFSET = 0.0

CRE_ARRAY = np.random.randint(0, 65535, size=(928, 1530), dtype=np.uint16)
COT_ARRAY = np.random.randint(0, 65535, size=(928, 1530), dtype=np.uint16)
PAL_ARRAY = np.random.randint(0, 255, size=(250, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def nwcsaf_geo_ct_filename(tmp_path_factory):
    """Create a CT file and return the filename."""
    return create_nwcsaf_geo_ct_file(tmp_path_factory.mktemp("data"))


def create_nwcsaf_geo_ct_file(directory, attrs=global_attrs):
    """Create a CT file."""
    filename = directory / "S_NWC_CT_MSG4_MSG-N-VISIR_20230118T103000Z_PLAX.nc"
    with h5netcdf.File(filename, mode="w") as nc_file:
        nc_file.dimensions = dimensions
        nc_file.attrs.update(attrs)
        var_name = "ct"

        var = nc_file.create_variable(var_name, ("ny", "nx"), np.uint16,
                                      chunks=(256, 256))
        var[:] = np.random.randint(0, 255, size=(928, 1530), dtype=np.uint8)

    return filename


@pytest.fixture
def nwcsaf_geo_ct_filehandler(nwcsaf_geo_ct_filename):
    """Create a CT filehandler."""
    return NcNWCSAF(nwcsaf_geo_ct_filename, {}, {})


@pytest.fixture(scope="session")
def nwcsaf_pps_cmic_filename(tmp_path_factory):
    """Create a CMIC file."""
    attrs = global_attrs.copy()
    attrs.update(PROJ_KM)
    attrs["time_coverage_start"] = START_TIME_PPS
    attrs["time_coverage_end"] = END_TIME_PPS

    filename = create_cmic_file(tmp_path_factory.mktemp("data"), filetype="cmic", attrs=attrs)

    return filename


def create_cmic_file(path, filetype, attrs=global_attrs):
    """Create a cmic file."""
    filename = path / f"S_NWC_{filetype.upper()}_npp_00000_20230118T1427508Z_20230118T1429150Z.nc"
    with h5netcdf.File(filename, mode="w") as nc_file:
        nc_file.dimensions = dimensions
        nc_file.attrs.update(attrs)
        create_cot_variable(nc_file, f"{filetype}_cot")
        create_cot_pal_variable(nc_file, f"{filetype}_cot_pal")
        create_cre_variables(nc_file, f"{filetype}_cre")

    return filename


@pytest.fixture
def nwcsaf_pps_cmic_filehandler(nwcsaf_pps_cmic_filename):
    """Create a CMIC filehandler."""
    return NcNWCSAF(nwcsaf_pps_cmic_filename, {}, {"file_key_prefix": "cmic_"})


@pytest.fixture(scope="session")
def nwcsaf_pps_cpp_filename(tmp_path_factory):
    """Create a CPP file."""
    filename = create_cmic_file(tmp_path_factory.mktemp("data"), filetype="cpp")

    return filename


def create_cre_variables(nc_file, var_name):
    """Create a CRE variable."""
    var = nc_file.create_variable(var_name, ("ny", "nx"), np.uint16, chunks=(256, 256))
    var[:] = CRE_ARRAY


def create_cot_pal_variable(nc_file, var_name):
    """Create a palette variable."""
    var = nc_file.create_variable(var_name, ("pal_colors_250", "pal_rgb"), np.uint8)
    var[:] = PAL_ARRAY
    var.attrs["palette_meanings"] = COT_PALETTE_MEANINGS


def create_cot_variable(nc_file, var_name):
    """Create a COT variable."""
    var = nc_file.create_variable(var_name, ("ny", "nx"), np.uint16, chunks=(256, 256))
    var[:] = COT_ARRAY
    var.attrs["scale_factor"] = COT_SCALE
    var.attrs["add_offset"] = COT_OFFSET


@pytest.fixture
def nwcsaf_pps_cpp_filehandler(nwcsaf_pps_cpp_filename):
    """Create a CPP filehandler."""
    return NcNWCSAF(nwcsaf_pps_cpp_filename, {}, {"file_key_prefix": "cpp_"})


@pytest.fixture(scope="session")
def nwcsaf_old_geo_ct_filename(tmp_path_factory):
    """Create a CT file and return the filename."""
    attrs = global_attrs.copy()
    attrs.update(PROJ_KM)
    attrs["time_coverage_start"] = np.array(["2023-01-18T10:39:17Z"], dtype="S20")
    return create_nwcsaf_geo_ct_file(tmp_path_factory.mktemp("data-old"), attrs=attrs)


@pytest.fixture
def nwcsaf_old_geo_ct_filehandler(nwcsaf_old_geo_ct_filename):
    """Create a CT filehandler."""
    return NcNWCSAF(nwcsaf_old_geo_ct_filename, {}, {})


class TestNcNWCSAFGeo:
    """Test the NcNWCSAF reader for Geo products."""

    @pytest.mark.parametrize("platform, instrument", [("Metop-B", "avhrr-3"),
                                                      ("NOAA-20", "viirs"),
                                                      ("Himawari-8", "ahi"),
                                                      ("GOES-17", "abi"),
                                                      ("Meteosat-11", "seviri")])
    def test_sensor_name_platform(self, nwcsaf_geo_ct_filehandler, platform, instrument):
        """Test that the correct sensor name is being set."""
        nwcsaf_geo_ct_filehandler.set_platform_and_sensor(platform_name=platform)
        assert nwcsaf_geo_ct_filehandler.sensor == set([instrument])
        assert nwcsaf_geo_ct_filehandler.sensor_names == set([instrument])

    @pytest.mark.parametrize("platform, instrument", [("GOES16", "abi"),
                                                      ("MSG4", "seviri")])
    def test_sensor_name_sat_id(self, nwcsaf_geo_ct_filehandler, platform, instrument):
        """Test that the correct sensor name is being set."""
        nwcsaf_geo_ct_filehandler.set_platform_and_sensor(sat_id=platform)
        assert nwcsaf_geo_ct_filehandler.sensor == set([instrument])
        assert nwcsaf_geo_ct_filehandler.sensor_names == set([instrument])

    def test_get_area_def(self, nwcsaf_geo_ct_filehandler):
        """Test that get_area_def() returns proper area."""
        dsid = {'name': 'ct'}

        _check_area_def(nwcsaf_geo_ct_filehandler.get_area_def(dsid))

    def test_get_area_def_km(self, nwcsaf_old_geo_ct_filehandler):
        """Test that get_area_def() returns proper area when the projection is in km."""
        dsid = {'name': 'ct'}
        _check_area_def(nwcsaf_old_geo_ct_filehandler.get_area_def(dsid))

    def test_scale_dataset_attr_removal(self, nwcsaf_geo_ct_filehandler):
        """Test the scaling of the dataset and removal of obsolete attributes."""
        import numpy as np
        import xarray as xr

        attrs = {'scale_factor': np.array(10),
                 'add_offset': np.array(20)}
        var = xr.DataArray([1, 2, 3], attrs=attrs)

        var = nwcsaf_geo_ct_filehandler.scale_dataset(var, 'dummy')
        np.testing.assert_allclose(var, [30, 40, 50])
        assert 'scale_factor' not in var.attrs
        assert 'add_offset' not in var.attrs

    @pytest.mark.parametrize("attrs, expected", [({'scale_factor': np.array(1.5),
                                                   'add_offset': np.array(2.5),
                                                   '_FillValue': 1},
                                                  [np.nan, 5.5, 7]),
                                                 ({'scale_factor': np.array(1.5),
                                                   'add_offset': np.array(2.5),
                                                   'valid_min': 1.1},
                                                  [np.nan, 5.5, 7]),
                                                 ({'scale_factor': np.array(1.5),
                                                   'add_offset': np.array(2.5),
                                                   'valid_max': 2.1},
                                                  [4, 5.5, np.nan]),
                                                 ({'scale_factor': np.array(1.5),
                                                   'add_offset': np.array(2.5),
                                                   'valid_range': (1.1, 2.1)},
                                                  [np.nan, 5.5, np.nan])])
    def test_scale_dataset_floating(self, nwcsaf_geo_ct_filehandler, attrs, expected):
        """Test the scaling of the dataset with floating point values."""
        var = xr.DataArray([1, 2, 3], attrs=attrs)
        var = nwcsaf_geo_ct_filehandler.scale_dataset(var, 'dummy')
        np.testing.assert_allclose(var, expected)
        assert 'scale_factor' not in var.attrs
        assert 'add_offset' not in var.attrs

    def test_scale_dataset_floating_nwcsaf_geo_ctth(self, nwcsaf_geo_ct_filehandler):
        """Test the scaling of the dataset with floating point values for CTTH NWCSAF/Geo v2016/v2018."""
        attrs = {'scale_factor': np.array(1.),
                 'add_offset': np.array(-2000.),
                 'valid_range': (0., 27000.)}
        var = xr.DataArray([1, 2, 3], attrs=attrs)
        var = nwcsaf_geo_ct_filehandler.scale_dataset(var, 'dummy')
        np.testing.assert_allclose(var, [-1999., -1998., -1997.])
        assert 'scale_factor' not in var.attrs
        assert 'add_offset' not in var.attrs
        np.testing.assert_equal(var.attrs['valid_range'], (-2000., 25000.))

    def test_orbital_parameters_are_correct(self, nwcsaf_geo_ct_filehandler):
        """Test that orbital parameters are present in the dataset attributes."""
        dsid = {'name': 'ct'}
        var = nwcsaf_geo_ct_filehandler.get_dataset(dsid, {})
        assert "orbital_parameters" in var.attrs
        for param in var.attrs['orbital_parameters']:
            assert isinstance(var.attrs['orbital_parameters'][param], (float, int))

        assert var.attrs["orbital_parameters"]["satellite_nominal_altitude"] == NOMINAL_ALTITUDE
        assert var.attrs["orbital_parameters"]["satellite_nominal_longitude"] == NOMINAL_LONGITUDE
        assert var.attrs["orbital_parameters"]["satellite_nominal_latitude"] == 0

    def test_times_are_in_dataset_attributes(self, nwcsaf_geo_ct_filehandler):
        """Check that start/end times are in the attributes of datasets."""
        dsid = {'name': 'ct'}
        var = nwcsaf_geo_ct_filehandler.get_dataset(dsid, {})
        assert "start_time" in var.attrs
        assert "end_time" in var.attrs

    def test_start_time(self, nwcsaf_geo_ct_filehandler):
        """Test the start time property."""
        assert nwcsaf_geo_ct_filehandler.start_time == read_nwcsaf_time(START_TIME)

    def test_end_time(self, nwcsaf_geo_ct_filehandler):
        """Test the end time property."""
        assert nwcsaf_geo_ct_filehandler.end_time == read_nwcsaf_time(END_TIME)


class TestNcNWCSAFPPS:
    """Test the NcNWCSAF reader for PPS products."""

    def test_start_time(self, nwcsaf_pps_cmic_filehandler):
        """Test the start time property."""
        assert nwcsaf_pps_cmic_filehandler.start_time == read_nwcsaf_time(START_TIME_PPS)

    def test_end_time(self, nwcsaf_pps_cmic_filehandler):
        """Test the start time property."""
        assert nwcsaf_pps_cmic_filehandler.end_time == read_nwcsaf_time(END_TIME_PPS)

    def test_drop_xycoords(self, nwcsaf_pps_cmic_filehandler):
        """Test the drop of x and y coords."""
        y_line = xr.DataArray(list(range(5)), dims=('y'), attrs={"long_name": "scan line number"})
        x_pixel = xr.DataArray(list(range(10)), dims=('x'), attrs={"long_name": "pixel number"})
        lat = xr.DataArray(np.ones((5, 10)),
                           dims=('y', 'x'),
                           coords={'y': y_line, 'x': x_pixel},
                           attrs={'name': 'lat',
                                  'standard_name': 'latitude'})
        lon = xr.DataArray(np.ones((5, 10)),
                           dims=('y', 'x'),
                           coords={'y': y_line, 'x': x_pixel},
                           attrs={'name': 'lon',
                                  'standard_name': 'longitude'})
        data_array_in = xr.DataArray(np.ones((5, 10)),
                                     attrs={"scale_factor": np.array(0, dtype=float),
                                            "add_offset": np.array(1, dtype=float)},
                                     dims=('y', 'x'),
                                     coords={'lon': lon, 'lat': lat, 'y': y_line, 'x': x_pixel})
        data_array_out = nwcsaf_pps_cmic_filehandler.drop_xycoords(data_array_in)
        assert 'y' not in data_array_out.coords

    def test_get_dataset_scales_and_offsets(self, nwcsaf_pps_cpp_filehandler):
        """Test that get_dataset() returns scaled and offseted data."""
        dsid = {'name': 'cpp_cot'}

        info = dict(name="cpp_cot",
                    file_type="nc_nwcsaf_cpp")

        res = nwcsaf_pps_cpp_filehandler.get_dataset(dsid, info)
        np.testing.assert_allclose(res, COT_ARRAY * COT_SCALE + COT_OFFSET)

    def test_get_dataset_scales_and_offsets_palette_meanings_using_other_dataset(self, nwcsaf_pps_cpp_filehandler):
        """Test that get_dataset() returns scaled palette_meanings with another dataset as scaling source."""
        dsid = {'name': 'cpp_cot_pal'}

        info = dict(name="cpp_cot_pal",
                    file_type="nc_nwcsaf_cpp",
                    scale_offset_dataset="cot")

        res = nwcsaf_pps_cpp_filehandler.get_dataset(dsid, info)
        palette_meanings = np.array(COT_PALETTE_MEANINGS.split()).astype(int)
        np.testing.assert_allclose(res.attrs["palette_meanings"], palette_meanings * COT_SCALE + COT_OFFSET)

    def test_get_dataset_raises_when_dataset_missing(self, nwcsaf_pps_cpp_filehandler):
        """Test that get_dataset() raises an error when the requested dataset is missing."""
        dsid = {'name': 'cpp_phase'}
        info = dict(name="cpp_phase",
                    file_type="nc_nwcsaf_cpp")
        with pytest.raises(KeyError):
            nwcsaf_pps_cpp_filehandler.get_dataset(dsid, info)

    def test_get_dataset_uses_file_key_if_present(self, nwcsaf_pps_cmic_filehandler, nwcsaf_pps_cpp_filehandler):
        """Test that get_dataset() uses a file_key if present."""
        dsid_cpp = {'name': 'cpp_cot'}
        dsid_cmic = {'name': 'cmic_cot'}

        file_key = "cmic_cot"

        nwcsaf_pps_cmic_filehandler.file_key_prefix = ""

        info_cpp = dict(name="cpp_cot",
                        file_key=file_key,
                        file_type="nc_nwcsaf_cpp")

        res_cpp = nwcsaf_pps_cmic_filehandler.get_dataset(dsid_cpp, info_cpp)

        info_cmic = dict(name="cmic_cot",
                         file_type="nc_nwcsaf_cpp")

        res_cmic = nwcsaf_pps_cmic_filehandler.get_dataset(dsid_cmic, info_cmic)
        np.testing.assert_allclose(res_cpp, res_cmic)

    def test_get_dataset_can_handle_file_key_list(self, nwcsaf_pps_cmic_filehandler, nwcsaf_pps_cpp_filehandler):
        """Test that get_dataset() can handle a list of file_keys."""
        dsid_cpp = {'name': 'cpp_reff'}
        dsid_cmic = {'name': 'cmic_cre'}

        info_cpp = dict(name="cmic_reff",
                        file_key=['reff', 'cre'],
                        file_type="nc_nwcsaf_cpp")

        res_cpp = nwcsaf_pps_cpp_filehandler.get_dataset(dsid_cpp, info_cpp)

        info_cmic = dict(name="cmic_reff",
                         file_key=['reff', 'cre'],
                         file_type="nc_nwcsaf_cpp")

        res_cmic = nwcsaf_pps_cmic_filehandler.get_dataset(dsid_cmic, info_cmic)
        np.testing.assert_allclose(res_cpp, res_cmic)


class TestNcNWCSAFFileKeyPrefix:
    """Test the NcNWCSAF reader when using a file key prefix."""

    def test_get_dataset_uses_file_key_prefix(self, nwcsaf_pps_cmic_filehandler):
        """Test that get_dataset() uses a file_key_prefix."""
        dsid_cpp = {'name': 'cpp_cot'}
        dsid_cmic = {'name': 'cmic_cot'}

        file_key = "cot"

        info_cpp = dict(name="cpp_cot",
                        file_key=file_key,
                        file_type="nc_nwcsaf_cpp")

        res_cpp = nwcsaf_pps_cmic_filehandler.get_dataset(dsid_cpp, info_cpp)

        info_cmic = dict(name="cmic_cot",
                         file_type="nc_nwcsaf_cpp")

        res_cmic = nwcsaf_pps_cmic_filehandler.get_dataset(dsid_cmic, info_cmic)
        np.testing.assert_allclose(res_cpp, res_cmic)

    def test_get_dataset_scales_and_offsets_palette_meanings_using_other_dataset(self, nwcsaf_pps_cmic_filehandler):
        """Test that get_dataset() returns scaled palette_meanings using another dataset as scaling source."""
        dsid = {'name': 'cpp_cot_pal'}

        info = dict(name="cpp_cot_pal",
                    file_key="cot_pal",
                    file_type="nc_nwcsaf_cpp",
                    scale_offset_dataset="cot")

        res = nwcsaf_pps_cmic_filehandler.get_dataset(dsid, info)
        palette_meanings = np.array(COT_PALETTE_MEANINGS.split()).astype(int)
        np.testing.assert_allclose(res.attrs["palette_meanings"], palette_meanings * COT_SCALE + COT_OFFSET)


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
