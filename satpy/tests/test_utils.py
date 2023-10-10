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
from math import sqrt
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from satpy.utils import (
    angle2xyz,
    get_legacy_chunk_size,
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


class TestGeoUtils:
    """Testing geo-related utility functions."""

    @pytest.mark.parametrize(
        ("lonlat", "xyz"),
        [
            ((0, 0), (1, 0, 0)),
            ((90, 0), (0, 1, 0)),
            ((0, 90), (0, 0, 1)),
            ((180, 0), (-1, 0, 0)),
            ((-90, 0), (0, -1, 0)),
            ((0, -90), (0, 0, -1)),
            ((0, 45), (sqrt(2) / 2, 0, sqrt(2) / 2)),
            ((0, 60), (sqrt(1) / 2, 0, sqrt(3) / 2)),
        ],
    )
    def test_lonlat2xyz(self, lonlat, xyz):
        """Test the lonlat2xyz function."""
        x__, y__, z__ = lonlat2xyz(*lonlat)
        assert x__ == pytest.approx(xyz[0])
        assert y__ == pytest.approx(xyz[1])
        assert z__ == pytest.approx(xyz[2])

    @pytest.mark.parametrize(
        ("azizen", "xyz"),
        [
            ((0, 0), (0, 0, 1)),
            ((90, 0), (0, 0, 1)),
            ((0, 90), (0, 1, 0)),
            ((180, 0), (0, 0, 1)),
            ((-90, 0), (0, 0, 1)),
            ((0, -90), (0, -1, 0)),
            ((90, 90), (1, 0, 0)),
            ((-90, 90), (-1, 0, 0)),
            ((180, 90), (0, -1, 0)),
            ((0, -90), (0, -1, 0)),
            ((0, 45), (0, sqrt(2) / 2, sqrt(2) / 2)),
            ((0, 60), (0, sqrt(3) / 2, sqrt(1) / 2)),
        ],
    )
    def test_angle2xyz(self, azizen, xyz):
        """Test the angle2xyz function."""
        x__, y__, z__ = angle2xyz(*azizen)
        assert x__ == pytest.approx(xyz[0])
        assert y__ == pytest.approx(xyz[1])
        assert z__ == pytest.approx(xyz[2])

    @pytest.mark.parametrize(
        ("xyz", "asin", "lonlat"),
        [
            ((1, 0, 0), False, (0, 0)),
            ((0, 1, 0), False, (90, 0)),
            ((0, 0, 1), True, (0, 90)),
            ((0, 0, 1), False, (0, 90)),
            ((sqrt(2) / 2, sqrt(2) / 2, 0), False, (45, 0)),
        ],
    )
    def test_xyz2lonlat(self, xyz, asin, lonlat):
        """Test xyz2lonlat."""
        lon, lat = xyz2lonlat(*xyz, asin=asin)
        assert lon == pytest.approx(lonlat[0])
        assert lat == pytest.approx(lonlat[1])

    @pytest.mark.parametrize(
        ("xyz", "acos", "azizen"),
        [
            ((1, 0, 0), False, (90, 90)),
            ((0, 1, 0), False, (0, 90)),
            ((0, 0, 1), False, (0, 0)),
            ((0, 0, 1), True, (0, 0)),
            ((sqrt(2) / 2, sqrt(2) / 2, 0), False, (45, 90)),
            ((-1, 0, 0), False, (-90, 90)),
            ((0, -1, 0), False, (180, 90)),
        ],
    )
    def test_xyz2angle(self, xyz, acos, azizen):
        """Test xyz2angle."""
        azi, zen = xyz2angle(*xyz, acos=acos)
        assert azi == pytest.approx(azi)
        assert zen == pytest.approx(zen)

    @pytest.mark.parametrize(
        ("prj", "exp_prj"),
        [
            ("+asd=123123123123", "+asd=123123123123"),
            ("+a=6378.137", "+a=6378137.000"),
            ("+a=6378.137 +units=km", "+a=6378137.000"),
            ("+a=6378.137 +b=6378.137", "+a=6378137.000 +b=6378137.000"),
            ("+a=6378.137 +b=6378.137 +h=35785.863", "+a=6378137.000 +b=6378137.000 +h=35785863.000"),
        ],
    )
    def test_proj_units_to_meters(self, prj, exp_prj):
        """Test proj units to meters conversion."""
        assert proj_units_to_meters(prj) == exp_prj


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
                {'orbital_parameters': {'projection_longitude': 1}},
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
                "start_time": datetime.datetime(2031, 11, 20, 19, 18, 17)
            })
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
        "six": np.arange(25).reshape(5, 5)
    })
    assert len(sc.keys()) == 1
    assert sc.keys().pop()['name'] == "six"
    assert sc["six"].attrs["area"].shape == (5, 5)
    sc = make_fake_scene({
        "seven": np.arange(3 * 7).reshape(3, 7),
        "eight": np.arange(3 * 8).reshape(3, 8)
    },
        daskify=True,
        area=False,
        common_attrs={"repetency": "fourteen hundred per centimetre"})
    assert "area" not in sc["seven"].attrs.keys()
    assert (sc["seven"].attrs["repetency"] == sc["eight"].attrs["repetency"] ==
            "fourteen hundred per centimetre")
    assert isinstance(sc["seven"].data, da.Array)
    sc = make_fake_scene({
        "nine": xr.DataArray(
            np.arange(2 * 9).reshape(2, 9),
            dims=("y", "x"),
            attrs={"please": "preserve", "answer": 42})
    },
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
        warnings.warn(
            "Stop that! It's SILLY.",
            DeprecationWarning,
            stacklevel=2
        )

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


def test_chunk_size_limit():
    """Check the chunk size limit computations."""
    from unittest.mock import patch

    from satpy.utils import get_chunk_size_limit
    with patch("satpy.utils._get_pytroll_chunk_size") as ptc:
        ptc.return_value = 10
        assert get_chunk_size_limit(np.int32) == 400
        assert get_chunk_size_limit() == 800


def test_chunk_size_limit_from_dask_config():
    """Check the chunk size limit computations."""
    import dask.config

    from satpy.utils import get_chunk_size_limit
    with dask.config.set({"array.chunk-size": "1KiB"}):
        assert get_chunk_size_limit(np.uint8) == 1024


def test_get_legacy_chunk_size():
    """Test getting the legacy chunk size."""
    import dask.config
    assert get_legacy_chunk_size() == 4096
    with dask.config.set({"array.chunk-size": "32MiB"}):
        assert get_legacy_chunk_size() == 2048


@pytest.mark.parametrize(
    ("chunks", "shape", "previous_chunks", "lr_mult", "chunk_dtype", "exp_result"),
    [
        # 1km swath
        (("auto", -1), (1000, 3200), (40, 40), (4, 4), np.float32, (160, -1)),
        # 5km swath
        (("auto", -1), (1000 // 5, 3200 // 5), (40, 40), (20, 20), np.float32, (160 / 5, -1)),
        # 250m swath
        (("auto", -1), (1000 * 4, 3200 * 4), (40, 40), (1, 1), np.float32, (160 * 4, -1)),
        # 1km area (ABI chunk 226):
        (("auto", "auto"), (21696 // 2, 21696 // 2), (226*4, 226*4), (2, 2), np.float32, (1356, 1356)),
        # 1km area (64-bit)
        (("auto", "auto"), (21696 // 2, 21696 // 2), (226*4, 226*4), (2, 2), np.float64, (904, 904)),
        # 3km area
        (("auto", "auto"), (21696 // 3, 21696 // 3), (226*4, 226*4), (6, 6), np.float32, (452, 452)),
        # 500m area
        (("auto", "auto"), (21696, 21696), (226*4, 226*4), (1, 1), np.float32, (1356 * 2, 1356 * 2)),
        # 500m area (64-bit)
        (("auto", "auto"), (21696, 21696), (226*4, 226*4), (1, 1), np.float64, (904 * 2, 904 * 2)),
        # 250m swath with bands:
        ((1, "auto", -1), (7, 1000 * 4, 3200 * 4), (1, 40, 40), (1, 1, 1), np.float32, (1, 160 * 4, -1)),
        # lots of dimensions:
        ((1, 1, "auto", -1), (1, 7, 1000, 3200), (1, 1, 40, 40), (1, 1, 1, 1), np.float32, (1, 1, 1000, -1)),
    ],
)
def test_resolution_chunking(chunks, shape, previous_chunks, lr_mult, chunk_dtype, exp_result):
    """Test normalize_low_res_chunks helper function."""
    import dask.config

    from satpy.utils import normalize_low_res_chunks

    with dask.config.set({"array.chunk-size": "32MiB"}):
        chunk_results = normalize_low_res_chunks(
            chunks,
            shape,
            previous_chunks,
            lr_mult,
            chunk_dtype,
        )
    assert chunk_results == exp_result
    for chunk_size in chunk_results:
        assert isinstance(chunk_size[0], int) if isinstance(chunk_size, tuple) else isinstance(chunk_size, int)

    # make sure the chunks are understandable by dask
    da.zeros(shape, dtype=chunk_dtype, chunks=chunk_results)


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
        attrs={
            "name": "hand",
            "ancillary_variables": [index_finger, index_finger, ring_finger]
        })

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
