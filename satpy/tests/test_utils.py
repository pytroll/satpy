# Copyright (c) 2019-2023 Satpy developers
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
"""Testing of utils."""
from __future__ import annotations

import datetime
import logging
import typing
import unittest
import warnings
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from satpy.utils import (
    angle2xyz,
    get_satpos,
    import_error_helper,
    lonlat2xyz,
    proj_units_to_meters,
    xyz2angle,
    xyz2lonlat,
)

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - caplog


class TestUtils(unittest.TestCase):
    """Testing utils."""

    def test_lonlat2xyz(self):
        """Test the lonlat2xyz function."""
        x__, y__, z__ = lonlat2xyz(0, 0)
        self.assertAlmostEqual(x__, 1)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, 0)

        x__, y__, z__ = lonlat2xyz(90, 0)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, 1)
        self.assertAlmostEqual(z__, 0)

        x__, y__, z__ = lonlat2xyz(0, 90)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, 1)

        x__, y__, z__ = lonlat2xyz(180, 0)
        self.assertAlmostEqual(x__, -1)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, 0)

        x__, y__, z__ = lonlat2xyz(-90, 0)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, -1)
        self.assertAlmostEqual(z__, 0)

        x__, y__, z__ = lonlat2xyz(0, -90)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, -1)

        x__, y__, z__ = lonlat2xyz(0, 45)
        self.assertAlmostEqual(x__, np.sqrt(2) / 2)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, np.sqrt(2) / 2)

        x__, y__, z__ = lonlat2xyz(0, 60)
        self.assertAlmostEqual(x__, np.sqrt(1) / 2)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, np.sqrt(3) / 2)

    def test_angle2xyz(self):
        """Test the lonlat2xyz function."""
        x__, y__, z__ = angle2xyz(0, 0)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, 1)

        x__, y__, z__ = angle2xyz(90, 0)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, 1)

        x__, y__, z__ = angle2xyz(0, 90)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, 1)
        self.assertAlmostEqual(z__, 0)

        x__, y__, z__ = angle2xyz(180, 0)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, 1)

        x__, y__, z__ = angle2xyz(-90, 0)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, 1)

        x__, y__, z__ = angle2xyz(0, -90)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, -1)
        self.assertAlmostEqual(z__, 0)

        x__, y__, z__ = angle2xyz(90, 90)
        self.assertAlmostEqual(x__, 1)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, 0)

        x__, y__, z__ = angle2xyz(-90, 90)
        self.assertAlmostEqual(x__, -1)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, 0)

        x__, y__, z__ = angle2xyz(180, 90)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, -1)
        self.assertAlmostEqual(z__, 0)

        x__, y__, z__ = angle2xyz(0, -90)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, -1)
        self.assertAlmostEqual(z__, 0)

        x__, y__, z__ = angle2xyz(0, 45)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, np.sqrt(2) / 2)
        self.assertAlmostEqual(z__, np.sqrt(2) / 2)

        x__, y__, z__ = angle2xyz(0, 60)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, np.sqrt(3) / 2)
        self.assertAlmostEqual(z__, np.sqrt(1) / 2)

    def test_xyz2lonlat(self):
        """Test xyz2lonlat."""
        lon, lat = xyz2lonlat(1, 0, 0)
        self.assertAlmostEqual(lon, 0)
        self.assertAlmostEqual(lat, 0)

        lon, lat = xyz2lonlat(0, 1, 0)
        self.assertAlmostEqual(lon, 90)
        self.assertAlmostEqual(lat, 0)

        lon, lat = xyz2lonlat(0, 0, 1, asin=True)
        self.assertAlmostEqual(lon, 0)
        self.assertAlmostEqual(lat, 90)

        lon, lat = xyz2lonlat(0, 0, 1)
        self.assertAlmostEqual(lon, 0)
        self.assertAlmostEqual(lat, 90)

        lon, lat = xyz2lonlat(np.sqrt(2) / 2, np.sqrt(2) / 2, 0)
        self.assertAlmostEqual(lon, 45)
        self.assertAlmostEqual(lat, 0)

    def test_xyz2angle(self):
        """Test xyz2angle."""
        azi, zen = xyz2angle(1, 0, 0)
        self.assertAlmostEqual(azi, 90)
        self.assertAlmostEqual(zen, 90)

        azi, zen = xyz2angle(0, 1, 0)
        self.assertAlmostEqual(azi, 0)
        self.assertAlmostEqual(zen, 90)

        azi, zen = xyz2angle(0, 0, 1)
        self.assertAlmostEqual(azi, 0)
        self.assertAlmostEqual(zen, 0)

        azi, zen = xyz2angle(0, 0, 1, acos=True)
        self.assertAlmostEqual(azi, 0)
        self.assertAlmostEqual(zen, 0)

        azi, zen = xyz2angle(np.sqrt(2) / 2, np.sqrt(2) / 2, 0)
        self.assertAlmostEqual(azi, 45)
        self.assertAlmostEqual(zen, 90)

        azi, zen = xyz2angle(-1, 0, 0)
        self.assertAlmostEqual(azi, -90)
        self.assertAlmostEqual(zen, 90)

        azi, zen = xyz2angle(0, -1, 0)
        self.assertAlmostEqual(azi, 180)
        self.assertAlmostEqual(zen, 90)

    def test_proj_units_to_meters(self):
        """Test proj units to meters conversion."""
        prj = '+asd=123123123123'
        res = proj_units_to_meters(prj)
        self.assertEqual(res, prj)
        prj = '+a=6378.137'
        res = proj_units_to_meters(prj)
        self.assertEqual(res, '+a=6378137.000')
        prj = '+a=6378.137 +units=km'
        res = proj_units_to_meters(prj)
        self.assertEqual(res, '+a=6378137.000')
        prj = '+a=6378.137 +b=6378.137'
        res = proj_units_to_meters(prj)
        self.assertEqual(res, '+a=6378137.000 +b=6378137.000')
        prj = '+a=6378.137 +b=6378.137 +h=35785.863'
        res = proj_units_to_meters(prj)
        self.assertEqual(res, '+a=6378137.000 +b=6378137.000 +h=35785863.000')


class TestGetSatPos:
    """Tests for 'get_satpos'."""

    @pytest.mark.parametrize(
        ("included_prefixes", "preference", "expected_result"),
        [
            (("nadir_", "satellite_actual_", "satellite_nominal_", "projection_"), None, (1, 2, 3)),
            (("satellite_actual_", "satellite_nominal_", "projection_"), None, (1.1, 2.1, 3)),
            (("satellite_nominal_", "projection_"), None, (1.2, 2.2, 3.1)),
            (("projection_",), None, (1.3, 2.3, 3.2)),
            (("nadir_", "satellite_actual_", "satellite_nominal_", "projection_"), "nadir", (1, 2, 3)),
            (("nadir_", "satellite_actual_", "satellite_nominal_", "projection_"), "actual", (1.1, 2.1, 3)),
            (("nadir_", "satellite_actual_", "satellite_nominal_", "projection_"), "nominal", (1.2, 2.2, 3.1)),
            (("nadir_", "satellite_actual_", "satellite_nominal_", "projection_"), "projection", (1.3, 2.3, 3.2)),
            (("satellite_nominal_", "projection_"), "actual", (1.2, 2.2, 3.1)),
            (("projection_",), "projection", (1.3, 2.3, 3.2)),
        ]
    )
    def test_get_satpos(self, included_prefixes, preference, expected_result):
        """Test getting the satellite position."""
        all_orb_params = {
            'nadir_longitude': 1,
            'satellite_actual_longitude': 1.1,
            'satellite_nominal_longitude': 1.2,
            'projection_longitude': 1.3,
            'nadir_latitude': 2,
            'satellite_actual_latitude': 2.1,
            'satellite_nominal_latitude': 2.2,
            'projection_latitude': 2.3,
            'satellite_actual_altitude': 3,
            'satellite_nominal_altitude': 3.1,
            'projection_altitude': 3.2
        }
        orb_params = {key: value for key, value in all_orb_params.items() if
                      any(in_prefix in key for in_prefix in included_prefixes)}
        data_arr = xr.DataArray((), attrs={'orbital_parameters': orb_params})

        with warnings.catch_warnings(record=True) as caught_warnings:
            lon, lat, alt = get_satpos(data_arr, preference=preference)
        has_satpos_warnings = any("using projection" in str(msg.message) for msg in caught_warnings)
        expect_warning = included_prefixes == ("projection_",) and preference != "projection"
        if expect_warning:
            assert has_satpos_warnings
        else:
            assert not has_satpos_warnings
        assert (lon, lat, alt) == expected_result

    @pytest.mark.parametrize(
        "attrs",
        (
                {},
                {'orbital_parameters':  {'projection_longitude': 1}},
                {'satellite_altitude': 1}
        )
    )
    def test_get_satpos_fails_with_informative_error(self, attrs):
        """Test that get_satpos raises an informative error message."""
        data_arr = xr.DataArray((), attrs=attrs)
        with pytest.raises(KeyError, match="Unable to determine satellite position.*"):
            get_satpos(data_arr)

    def test_get_satpos_from_satname(self, caplog):
        """Test getting satellite position from satellite name only."""
        import pyorbital.tlefile

        data_arr = xr.DataArray(
                (),
                attrs={
                    "platform_name": "Meteosat-42",
                    "sensor": "irives",
                    "start_time": datetime.datetime(2031, 11, 20, 19, 18, 17)})
        with mock.patch("pyorbital.tlefile.read") as plr:
            plr.return_value = pyorbital.tlefile.Tle(
                    "Meteosat-42",
                    line1="1 40732U 15034A   22011.84285506  .00000004  00000+0  00000+0 0  9995",
                    line2="2 40732   0.2533 325.0106 0000976 118.8734 330.4058  1.00272123 23817")
            with caplog.at_level(logging.WARNING):
                (lon, lat, alt) = get_satpos(data_arr, use_tle=True)
            assert "Orbital parameters missing from metadata" in caplog.text
            np.testing.assert_allclose(
                (lon, lat, alt),
                (119.39533705010592, -1.1491628298731498, 35803.19986408156),
                rtol=1e-4,
            )


def test_make_fake_scene():
    """Test the make_fake_scene utility.

    Although the make_fake_scene utility is for internal testing
    purposes, it has grown sufficiently complex that it needs its own
    testing.
    """
    from satpy.tests.utils import make_fake_scene

    assert make_fake_scene({}).keys() == []
    sc = make_fake_scene({
        "six": np.arange(25).reshape(5, 5)})
    assert len(sc.keys()) == 1
    assert sc.keys().pop()['name'] == "six"
    assert sc["six"].attrs["area"].shape == (5, 5)
    sc = make_fake_scene({
        "seven": np.arange(3*7).reshape(3, 7),
        "eight": np.arange(3*8).reshape(3, 8)},
        daskify=True,
        area=False,
        common_attrs={"repetency": "fourteen hundred per centimetre"})
    assert "area" not in sc["seven"].attrs.keys()
    assert (sc["seven"].attrs["repetency"] == sc["eight"].attrs["repetency"] ==
            "fourteen hundred per centimetre")
    assert isinstance(sc["seven"].data, da.Array)
    sc = make_fake_scene({
        "nine": xr.DataArray(
            np.arange(2*9).reshape(2, 9),
            dims=("y", "x"),
            attrs={"please": "preserve", "answer": 42})},
        common_attrs={"bad words": "semprini bahnhof veerooster winterbanden"})
    assert sc["nine"].attrs.keys() >= {"please", "answer", "bad words", "area"}


class TestCheckSatpy(unittest.TestCase):
    """Test the 'check_satpy' function."""

    def test_basic_check_satpy(self):
        """Test 'check_satpy' basic functionality."""
        from satpy.utils import check_satpy
        check_satpy()

    def test_specific_check_satpy(self):
        """Test 'check_satpy' with specific features provided."""
        from satpy.utils import check_satpy
        with mock.patch('satpy.utils.print') as print_mock:
            check_satpy(readers=['viirs_sdr'], extras=('cartopy', '__fake'))
            checked_fake = False
            for call in print_mock.mock_calls:
                if len(call[1]) > 0 and '__fake' in call[1][0]:
                    self.assertNotIn('ok', call[1][1])
                    checked_fake = True
            self.assertTrue(checked_fake, "Did not find __fake module "
                                          "mentioned in checks")


def test_debug_on(caplog):
    """Test that debug_on is working as expected."""
    from satpy.utils import debug, debug_off, debug_on

    def depwarn():
        logger = logging.getLogger("satpy.silly")
        logger.debug("But now it's just got SILLY.")
        warnings.warn("Stop that! It's SILLY.", DeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    debug_on(False)
    filts_before = warnings.filters.copy()
    # test that logging on, but deprecation warnings still off
    with caplog.at_level(logging.DEBUG):
        depwarn()
    assert warnings.filters == filts_before
    assert "But now it's just got SILLY." in caplog.text
    debug_on(True)
    # test that logging on and deprecation warnings on
    with pytest.warns(DeprecationWarning):
        depwarn()
    assert warnings.filters != filts_before
    debug_off()  # other tests assume debugging is off
    # test that filters were reset
    assert warnings.filters == filts_before
    with debug():
        assert warnings.filters != filts_before
    assert warnings.filters == filts_before


def test_logging_on_and_off(caplog):
    """Test that switching logging on and off works."""
    from satpy.utils import logging_off, logging_on
    logger = logging.getLogger("satpy.silly")
    logging_on()
    with caplog.at_level(logging.WARNING):
        logger.debug("I'd like to leave the army please, sir.")
        logger.warning("Stop that!  It's SILLY.")
    assert "Stop that!  It's SILLY" in caplog.text
    assert "I'd like to leave the army please, sir." not in caplog.text
    logging_off()
    with caplog.at_level(logging.DEBUG):
        logger.warning("You've got a nice army base here, Colonel.")
    assert "You've got a nice army base here, Colonel." not in caplog.text


@pytest.mark.parametrize(
    ("shapes", "chunks", "dims", "exp_unified"),
    [
        (
                ((3, 5, 5), (5, 5)),
                (-1, -1),
                (("bands", "y", "x"), ("y", "x")),
                True,
        ),
        (
                ((3, 5, 5), (5, 5)),
                (-1, 2),
                (("bands", "y", "x"), ("y", "x")),
                True,
        ),
        (
                ((4, 5, 5), (3, 5, 5)),
                (-1, -1),
                (("bands", "y", "x"), ("bands", "y", "x")),
                False,
        ),
    ],
)
def test_unify_chunks(shapes, chunks, dims, exp_unified):
    """Test unify_chunks utility function."""
    from satpy.utils import unify_chunks
    inputs = list(_data_arrays_from_params(shapes, chunks, dims))
    results = unify_chunks(*inputs)
    if exp_unified:
        _verify_unified(results)
    else:
        _verify_unchanged_chunks(results, inputs)


def _data_arrays_from_params(shapes: list[tuple[int, ...]],
                             chunks: list[tuple[int, ...]],
                             dims: list[tuple[int, ...]]
                             ) -> typing.Generator[xr.DataArray, None, None]:
    for shape, chunk, dim in zip(shapes, chunks, dims):
        yield xr.DataArray(da.ones(shape, chunks=chunk), dims=dim)


def _verify_unified(data_arrays: list[xr.DataArray]) -> None:
    dim_chunks: dict[str, int] = {}
    for data_arr in data_arrays:
        for dim, chunk_size in zip(data_arr.dims, data_arr.chunks):
            exp_chunks = dim_chunks.setdefault(dim, chunk_size)
            assert exp_chunks == chunk_size


def _verify_unchanged_chunks(data_arrays: list[xr.DataArray],
                             orig_arrays: list[xr.DataArray]) -> None:
    for data_arr, orig_arr in zip(data_arrays, orig_arrays):
        assert data_arr.chunks == orig_arr.chunks


def test_chunk_pixel_size():
    """Check the chunk pixel size computations."""
    from unittest.mock import patch

    from satpy.utils import get_chunk_pixel_size
    with patch("satpy.utils.CHUNK_SIZE", None):
        assert get_chunk_pixel_size() is None
    with patch("satpy.utils.CHUNK_SIZE", 10):
        assert get_chunk_pixel_size() == 100
    with patch("satpy.utils.CHUNK_SIZE", (10, 20)):
        assert get_chunk_pixel_size() == 200


def test_chunk_size_limit():
    """Check the chunk size limit computations."""
    from unittest.mock import patch

    from satpy.utils import get_chunk_size_limit
    with patch("satpy.utils.CHUNK_SIZE", None):
        assert get_chunk_size_limit(np.uint8) is None
    with patch("satpy.utils.CHUNK_SIZE", 10):
        assert get_chunk_size_limit(np.float64) == 800
    with patch("satpy.utils.CHUNK_SIZE", (10, 20)):
        assert get_chunk_size_limit(np.int32) == 800


def test_convert_remote_files_to_fsspec_local_files():
    """Test convertion of remote files to fsspec objects.

    Case without scheme/protocol, which should default to plain filenames.
    """
    from satpy.utils import convert_remote_files_to_fsspec

    filenames = ["/tmp/file1.nc", "file:///tmp/file2.nc"]
    res = convert_remote_files_to_fsspec(filenames)
    assert res == filenames


def test_convert_remote_files_to_fsspec_local_pathlib_files():
    """Test convertion of remote files to fsspec objects.

    Case using pathlib objects as filenames.
    """
    import pathlib

    from satpy.utils import convert_remote_files_to_fsspec

    filenames = [pathlib.Path("/tmp/file1.nc"), pathlib.Path("c:\tmp\file2.nc")]
    res = convert_remote_files_to_fsspec(filenames)
    assert res == filenames


def test_convert_remote_files_to_fsspec_mixed_sources():
    """Test convertion of remote files to fsspec objects.

    Case with mixed local and remote files.
    """
    from satpy.readers import FSFile
    from satpy.utils import convert_remote_files_to_fsspec

    filenames = ["/tmp/file1.nc", "s3://data-bucket/file2.nc", "file:///tmp/file3.nc"]
    res = convert_remote_files_to_fsspec(filenames)
    # Two local files, one remote
    assert filenames[0] in res
    assert filenames[2] in res
    assert sum([isinstance(f, FSFile) for f in res]) == 1


def test_convert_remote_files_to_fsspec_filename_dict():
    """Test convertion of remote files to fsspec objects.

    Case where filenames is a dictionary mapping readers and filenames.
    """
    from satpy.readers import FSFile
    from satpy.utils import convert_remote_files_to_fsspec

    filenames = {
        "reader1": ["/tmp/file1.nc", "/tmp/file2.nc"],
        "reader2": ["s3://tmp/file3.nc", "file:///tmp/file4.nc", "/tmp/file5.nc"]
    }
    res = convert_remote_files_to_fsspec(filenames)

    assert res["reader1"] == filenames["reader1"]
    assert filenames["reader2"][1] in res["reader2"]
    assert filenames["reader2"][2] in res["reader2"]
    assert sum([isinstance(f, FSFile) for f in res["reader2"]]) == 1


def test_convert_remote_files_to_fsspec_fsfile():
    """Test convertion of remote files to fsspec objects.

    Case where the some of the files are already FSFile objects.
    """
    from satpy.readers import FSFile
    from satpy.utils import convert_remote_files_to_fsspec

    filenames = ["/tmp/file1.nc", "s3://data-bucket/file2.nc", FSFile("ssh:///tmp/file3.nc")]
    res = convert_remote_files_to_fsspec(filenames)

    assert sum([isinstance(f, FSFile) for f in res]) == 2


def test_convert_remote_files_to_fsspec_windows_paths():
    """Test convertion of remote files to fsspec objects.

    Case where windows paths are used.
    """
    from satpy.utils import convert_remote_files_to_fsspec

    filenames = [r"C:\wintendo\file1.nc", "e:\\wintendo\\file2.nc", r"wintendo\file3.nc"]
    res = convert_remote_files_to_fsspec(filenames)

    assert res == filenames


@mock.patch('fsspec.open_files')
def test_convert_remote_files_to_fsspec_storage_options(open_files):
    """Test convertion of remote files to fsspec objects.

    Case with storage options given.
    """
    from satpy.utils import convert_remote_files_to_fsspec

    filenames = ["s3://tmp/file1.nc"]
    storage_options = {'anon': True}

    _ = convert_remote_files_to_fsspec(filenames, storage_options=storage_options)

    open_files.assert_called_once_with(filenames, **storage_options)


def test_import_error_helper():
    """Test the import error helper."""
    module = "some_crazy_name_for_unknow_dependency_module"
    with pytest.raises(ImportError) as err:
        with import_error_helper(module):
            import unknow_dependency_module  # noqa
    assert module in str(err)


def test_find_in_ancillary():
    """Test finding a dataset in ancillary variables."""
    from satpy.utils import find_in_ancillary
    index_finger = xr.DataArray(
            data=np.arange(25).reshape(5, 5),
            dims=("y", "x"),
            attrs={"name": "index-finger"})
    ring_finger = xr.DataArray(
            data=np.arange(25).reshape(5, 5),
            dims=("y", "x"),
            attrs={"name": "ring-finger"})

    hand = xr.DataArray(
            data=np.arange(25).reshape(5, 5),
            dims=("y", "x"),
            attrs={"name": "hand",
                   "ancillary_variables": [index_finger, index_finger, ring_finger]})

    assert find_in_ancillary(hand, "ring-finger") is ring_finger
    with pytest.raises(
            ValueError,
            match=("Expected exactly one dataset named index-finger in "
                   "ancillary variables for dataset 'hand', found 2")):
        find_in_ancillary(hand, "index-finger")
    with pytest.raises(
            ValueError,
            match=("Could not find dataset named thumb in "
                   "ancillary variables for dataset 'hand'")):
        find_in_ancillary(hand, "thumb")
