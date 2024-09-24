#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2023 Satpy developers
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
"""Module for testing the satpy.readers.olci_nc module."""
import datetime
import unittest
import unittest.mock as mock


class TestOLCIReader(unittest.TestCase):
    """Test various olci_nc filehandlers."""

    @mock.patch("xarray.open_dataset")
    def test_instantiate(self, mocked_dataset):
        """Test initialization of file handlers."""
        import xarray as xr

        from satpy.readers.olci_nc import NCOLCI1B, NCOLCI2, NCOLCIBase, NCOLCICal, NCOLCIChannelBase, NCOLCIGeo
        from satpy.tests.utils import make_dataid

        cal_data = xr.Dataset(
            {
                "solar_flux": (("bands"), [0, 1, 2]),
                "detector_index": (("bands"), [0, 1, 2]),
            },
            {"bands": [0, 1, 2], },
        )

        ds_id = make_dataid(name="Oa01", calibration="reflectance")
        ds_id2 = make_dataid(name="wsqf", calibration="reflectance")
        filename_info = {"mission_id": "S3A", "dataset_name": "Oa01", "start_time": 0, "end_time": 0}

        test = NCOLCIBase("somedir/somefile.nc", filename_info, "c")
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCOLCICal("somedir/somefile.nc", filename_info, "c")
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCOLCIGeo("somedir/somefile.nc", filename_info, "c")
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCOLCIChannelBase("somedir/somefile.nc", filename_info, "c")
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        cal = mock.Mock()
        cal.nc = cal_data
        test = NCOLCI1B("somedir/somefile.nc", filename_info, "c", cal)
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCOLCI2("somedir/somefile.nc", filename_info, "c")
        test.get_dataset(ds_id, {"nc_key": "the_key"})
        test.get_dataset(ds_id2, {"nc_key": "the_key"})
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

    @mock.patch("xarray.open_dataset")
    def test_open_file_objects(self, mocked_open_dataset):
        """Test initialization of file handlers."""
        from satpy.readers.olci_nc import NCOLCIBase
        filename_info = {"mission_id": "S3A", "dataset_name": "Oa01", "start_time": 0, "end_time": 0}

        open_file = mock.MagicMock()

        file_handler = NCOLCIBase(open_file, filename_info, "c")
        #  deepcode ignore W0104: This is a property that is actually a function call.
        file_handler.nc  # pylint: disable=W0104
        mocked_open_dataset.assert_called()
        open_file.open.assert_called()
        assert (open_file.open.return_value in mocked_open_dataset.call_args[0] or
                open_file.open.return_value == mocked_open_dataset.call_args[1].get("filename_or_obj"))

    @mock.patch("xarray.open_dataset")
    def test_get_l2_mask(self, mocked_dataset):
        """Test reading datasets."""
        import numpy as np
        import xarray as xr

        from satpy.readers.olci_nc import NCOLCI2
        from satpy.tests.utils import make_dataid
        mocked_dataset.return_value = xr.Dataset({"mask": (["rows", "columns"],
                                                           np.array([1 << x for x in range(30)]).reshape(5, 6))},
                                                 coords={"rows": np.arange(5),
                                                         "columns": np.arange(6)})
        ds_id = make_dataid(name="mask")
        filename_info = {"mission_id": "S3A", "dataset_name": "mask", "start_time": 0, "end_time": 0}
        test = NCOLCI2("somedir/somefile.nc", filename_info, "c")
        res = test.get_dataset(ds_id, {"nc_key": "mask"})
        assert res.dtype == np.dtype("bool")
        expected = np.array([[True, False, True, True, True, True],
                             [False, False, True, True, False, False],
                             [False, False, False, False, False, True],
                             [False, True, False, False, False, True],
                             [True, False, False, True, False, False]])
        np.testing.assert_array_equal(res.values, expected)

    @mock.patch("xarray.open_dataset")
    def test_get_l2_mask_with_alternative_items(self, mocked_dataset):
        """Test reading datasets."""
        import numpy as np
        import xarray as xr

        from satpy.readers.olci_nc import NCOLCI2
        from satpy.tests.utils import make_dataid
        mocked_dataset.return_value = xr.Dataset({"mask": (["rows", "columns"],
                                                           np.array([1 << x for x in range(30)]).reshape(5, 6))},
                                                 coords={"rows": np.arange(5),
                                                         "columns": np.arange(6)})
        ds_id = make_dataid(name="mask")
        filename_info = {"mission_id": "S3A", "dataset_name": "mask", "start_time": 0, "end_time": 0}
        test = NCOLCI2("somedir/somefile.nc", filename_info, "c", mask_items=["INVALID"])
        res = test.get_dataset(ds_id, {"nc_key": "mask"})
        assert res.dtype == np.dtype("bool")
        expected = np.array([True] + [False] * 29).reshape(5, 6)
        np.testing.assert_array_equal(res.values, expected)


    @mock.patch("xarray.open_dataset")
    def test_get_l1b_default_mask(self, mocked_dataset):
        """Test reading mask datasets from L1B products."""
        import numpy as np
        import xarray as xr

        from satpy.readers.olci_nc import NCOLCI1B
        from satpy.tests.utils import make_dataid
        mocked_dataset.return_value = xr.Dataset({"quality_flags": (["rows", "columns"],
                                                           np.array([1 << (x % 32) for x in range(35)]).reshape(5, 7))},
                                                 coords={"rows": np.arange(5),
                                                         "columns": np.arange(7)})
        ds_id = make_dataid(name="mask")
        filename_info = {"mission_id": "S3A", "dataset_name": "mask", "start_time": 0, "end_time": 0}
        test = NCOLCI1B("somedir/somefile.nc", filename_info, "c")
        res = test.get_dataset(ds_id, {"nc_key": "quality_flags"})
        assert res.dtype == np.dtype("bool")

        expected = np.array([[False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [True, True, True, True, True, True, True],
                             [True, False, True, True, False, False, False]])

        np.testing.assert_array_equal(res.values, expected)


    @mock.patch("xarray.open_dataset")
    def test_get_l1b_customized_mask(self, mocked_dataset):
        """Test reading mask datasets from L1B products."""
        import numpy as np
        import xarray as xr

        from satpy.readers.olci_nc import NCOLCI1B
        from satpy.tests.utils import make_dataid
        mocked_dataset.return_value = xr.Dataset({"quality_flags": (["rows", "columns"],
                                                           np.array([1 << (x % 32) for x in range(35)]).reshape(5, 7))},
                                                 coords={"rows": np.arange(5),
                                                         "columns": np.arange(7)})
        ds_id = make_dataid(name="mask")
        filename_info = {"mission_id": "S3A", "dataset_name": "mask", "start_time": 0, "end_time": 0}
        test = NCOLCI1B("somedir/somefile.nc", filename_info, "c", mask_items=["bright", "invalid"])
        res = test.get_dataset(ds_id, {"nc_key": "quality_flags"})
        assert res.dtype == np.dtype("bool")

        expected = np.array([[False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, False, False, False],
                             [False, False, False, False, True, False, True],
                             [False, False, False, False, False, False, False]])

        np.testing.assert_array_equal(res.values, expected)


    @mock.patch("xarray.open_dataset")
    def test_olci_angles(self, mocked_dataset):
        """Test reading datasets."""
        import numpy as np
        import xarray as xr

        from satpy.readers.olci_nc import NCOLCIAngles
        from satpy.tests.utils import make_dataid
        attr_dict = {
            "ac_subsampling_factor": 1,
            "al_subsampling_factor": 2,
        }
        mocked_dataset.return_value = xr.Dataset({"SAA": (["tie_rows", "tie_columns"],
                                                          np.array([1 << x for x in range(30)]).reshape(5, 6)),
                                                  "SZA": (["tie_rows", "tie_columns"],
                                                          np.array([1 << x for x in range(30)]).reshape(5, 6)),
                                                  "OAA": (["tie_rows", "tie_columns"],
                                                          np.array([1 << x for x in range(30)]).reshape(5, 6)),
                                                  "OZA": (["tie_rows", "tie_columns"],
                                                          np.array([1 << x for x in range(30)]).reshape(5, 6))},
                                                 coords={"rows": np.arange(5),
                                                         "columns": np.arange(6)},
                                                 attrs=attr_dict)
        filename_info = {"mission_id": "S3A", "dataset_name": "Oa01", "start_time": 0, "end_time": 0}

        ds_id = make_dataid(name="solar_azimuth_angle")
        ds_id2 = make_dataid(name="satellite_zenith_angle")
        test = NCOLCIAngles("somedir/somefile.nc", filename_info, "c")
        test.get_dataset(ds_id, filename_info)
        test.get_dataset(ds_id2, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

    @mock.patch("xarray.open_dataset")
    def test_olci_meteo(self, mocked_dataset):
        """Test reading datasets."""
        import numpy as np
        import xarray as xr

        from satpy.readers.olci_nc import NCOLCIMeteo
        from satpy.tests.utils import make_dataid
        attr_dict = {
            "ac_subsampling_factor": 1,
            "al_subsampling_factor": 2,
        }
        data = {"humidity": (["tie_rows", "tie_columns"],
                             np.array([1 << x for x in range(30)]).reshape(5, 6)),
                "total_ozone": (["tie_rows", "tie_columns"],
                                np.array([1 << x for x in range(30)]).reshape(5, 6)),
                "sea_level_pressure": (["tie_rows", "tie_columns"],
                                       np.array([1 << x for x in range(30)]).reshape(5, 6)),
                "total_columnar_water_vapour": (["tie_rows", "tie_columns"],
                                                np.array([1 << x for x in range(30)]).reshape(5, 6))}
        mocked_dataset.return_value = xr.Dataset(data,
                                                 coords={"rows": np.arange(5),
                                                         "columns": np.arange(6)},
                                                 attrs=attr_dict)
        filename_info = {"mission_id": "S3A", "dataset_name": "humidity", "start_time": 0, "end_time": 0}

        ds_id = make_dataid(name="humidity")
        ds_id2 = make_dataid(name="total_ozone")
        test = NCOLCIMeteo("somedir/somefile.nc", filename_info, "c")
        test.get_dataset(ds_id, filename_info)
        test.get_dataset(ds_id2, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

    @mock.patch("xarray.open_dataset")
    def test_chl_nn(self, mocked_dataset):
        """Test unlogging the chl_nn product."""
        import numpy as np
        import xarray as xr

        from satpy.readers.olci_nc import NCOLCI2
        from satpy.tests.utils import make_dataid
        attr_dict = {
            "ac_subsampling_factor": 64,
            "al_subsampling_factor": 1,
        }
        data = {"CHL_NN": (["rows", "columns"],
                           np.arange(30).reshape(5, 6).astype(float),
                           {"units": "lg(re mg.m-3)"})}
        mocked_dataset.return_value = xr.Dataset(data,
                                                 coords={"rows": np.arange(5),
                                                         "columns": np.arange(6)},
                                                 attrs=attr_dict)
        ds_info = {"name": "chl_nn", "sensor": "olci", "resolution": 300,
                   "standard_name": "algal_pigment_concentration", "units": "lg(re mg.m-3)",
                   "coordinates": ("longitude", "latitude"), "file_type": "esa_l2_chl_nn", "nc_key": "CHL_NN",
                   "modifiers": ()}
        filename_info = {"mission_id": "S3A", "datatype_id": "WFR",
                         "start_time": datetime.datetime(2019, 9, 24, 9, 29, 39),
                         "end_time": datetime.datetime(2019, 9, 24, 9, 32, 39),
                         "creation_time": datetime.datetime(2019, 9, 24, 11, 40, 26), "duration": 179, "cycle": 49,
                         "relative_orbit": 307, "frame": 1800, "centre": "MAR", "mode": "O", "timeliness": "NR",
                         "collection": "002"}
        ds_id = make_dataid(name="chl_nn")
        file_handler = NCOLCI2("somedir/somefile.nc", filename_info, None, unlog=True)
        res = file_handler.get_dataset(ds_id, ds_info)

        assert res.attrs["units"] == "mg.m-3"
        assert res.values[-1, -1] == 1e29


class TestL2BitFlags(unittest.TestCase):
    """Test the bitflag reading."""

    def test_bitflags(self):
        """Test the BitFlags class."""
        from functools import reduce

        import numpy as np

        from satpy.readers.olci_nc import BitFlags
        flag_list = ["INVALID", "WATER", "LAND", "CLOUD", "SNOW_ICE",
                     "INLAND_WATER", "TIDAL", "COSMETIC", "SUSPECT", "HISOLZEN",
                     "SATURATED", "MEGLINT", "HIGHGLINT", "WHITECAPS",
                     "ADJAC", "WV_FAIL", "PAR_FAIL", "AC_FAIL", "OC4ME_FAIL",
                     "OCNN_FAIL", "Extra_1", "KDM_FAIL", "Extra_2",
                     "CLOUD_AMBIGUOUS", "CLOUD_MARGIN", "BPAC_ON",
                     "WHITE_SCATT", "LOWRW", "HIGHRW"]

        bits = np.array([1 << x for x in range(len(flag_list))])

        bflags = BitFlags(bits)

        items = ["INVALID", "SNOW_ICE", "INLAND_WATER", "SUSPECT",
                 "AC_FAIL", "CLOUD", "HISOLZEN", "OCNN_FAIL",
                 "CLOUD_MARGIN", "CLOUD_AMBIGUOUS", "LOWRW", "LAND"]

        mask = reduce(np.logical_or, [bflags[item] for item in items])
        expected = np.array([True, False, True, True, True, True, False,
                             False, True, True, False, False, False, False,
                             False, False, False, True, False, True, False,
                             False, False, True, True, False, False, True,
                             False])
        assert all(mask == expected)

    def test_bitflags_with_flags_from_array(self):
        """Test reading bitflags from DataArray attributes."""
        from functools import reduce

        import numpy as np
        import xarray as xr

        from satpy.readers.olci_nc import BitFlags

        flag_masks = [1, 2, 4, 8, 4194304, 8388608, 16777216, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                      32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 33554432, 67108864, 134217728, 268435456,
                      536870912, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472,
                      274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208,
                      17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312,
                      1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984,
                      36028797018963968]
        flag_meanings = ("INVALID WATER LAND CLOUD TURBID_ATM CLOUD_AMBIGUOUS CLOUD_MARGIN SNOW_ICE INLAND_WATER "
                         "COASTLINE TIDAL COSMETIC SUSPECT HISOLZEN SATURATED MEGLINT HIGHGLINT WHITECAPS ADJAC "
                         "WV_FAIL PAR_FAIL AC_FAIL OC4ME_FAIL OCNN_FAIL KDM_FAIL BPAC_ON WHITE_SCATT LOWRW HIGHRW "
                         "IOP_LSD_FAIL ANNOT_ANGSTROM ANNOT_AERO_B ANNOT_ABSO_D ANNOT_ACLIM ANNOT_ABSOA ANNOT_MIXR1 "
                         "ANNOT_DROUT ANNOT_TAU06 RWNEG_O1 RWNEG_O2 RWNEG_O3 RWNEG_O4 RWNEG_O5 RWNEG_O6 RWNEG_O7 "
                         "RWNEG_O8 RWNEG_O9 RWNEG_O10 RWNEG_O11 RWNEG_O12 RWNEG_O16 RWNEG_O17 RWNEG_O18 RWNEG_O21")

        bits = np.array([1 << x for x in range(int(np.log2(max(flag_masks))) + 1)])
        bits_array = xr.DataArray(bits, attrs=dict(flag_masks=flag_masks, flag_meanings=flag_meanings))
        bflags = BitFlags(bits_array)

        items = ["INVALID", "TURBID_ATM"]
        mask = reduce(np.logical_or, [bflags[item] for item in items])

        assert mask[0].item() is True
        assert any(mask[1:22]) is False
        assert mask[22].item() is True
        assert any(mask[23:]) is False

    def test_bitflags_with_dataarray_without_flags(self):
        """Test the BitFlags class."""
        from functools import reduce

        import numpy as np
        import xarray as xr

        from satpy.readers.olci_nc import BitFlags
        flag_list = ["INVALID", "WATER", "LAND", "CLOUD", "SNOW_ICE",
                     "INLAND_WATER", "TIDAL", "COSMETIC", "SUSPECT", "HISOLZEN",
                     "SATURATED", "MEGLINT", "HIGHGLINT", "WHITECAPS",
                     "ADJAC", "WV_FAIL", "PAR_FAIL", "AC_FAIL", "OC4ME_FAIL",
                     "OCNN_FAIL", "Extra_1", "KDM_FAIL", "Extra_2",
                     "CLOUD_AMBIGUOUS", "CLOUD_MARGIN", "BPAC_ON",
                     "WHITE_SCATT", "LOWRW", "HIGHRW"]

        bits = np.array([1 << x for x in range(len(flag_list))])

        bflags = BitFlags(xr.DataArray(bits))

        items = ["INVALID", "SNOW_ICE", "INLAND_WATER", "SUSPECT",
                 "AC_FAIL", "CLOUD", "HISOLZEN", "OCNN_FAIL",
                 "CLOUD_MARGIN", "CLOUD_AMBIGUOUS", "LOWRW", "LAND"]

        mask = reduce(np.logical_or, [bflags[item] for item in items])
        expected = np.array([True, False, True, True, True, True, False,
                             False, True, True, False, False, False, False,
                             False, False, False, True, False, True, False,
                             False, False, True, True, False, False, True,
                             False])
        assert all(mask == expected)


    def test_bitflags_with_custom_flag_list(self):
        """Test the BitFlags class providing a flag list."""
        from functools import reduce

        import numpy as np

        from satpy.readers.olci_nc import BitFlags
        flag_list = ["INVALID", "WATER", "LAND", "CLOUD", "SNOW_ICE",
                     "INLAND_WATER", "TIDAL", "COSMETIC", "SUSPECT", "HISOLZEN",
                     "SATURATED", "MEGLINT", "HIGHGLINT", "WHITECAPS",
                     "ADJAC", "WV_FAIL", "PAR_FAIL", "AC_FAIL", "OC4ME_FAIL",
                     "OCNN_FAIL", "Extra_1", "KDM_FAIL", "Extra_2",
                     "CLOUD_AMBIGUOUS", "CLOUD_MARGIN", "BPAC_ON",
                     "WHITE_SCATT", "LOWRW", "HIGHRW"]

        bits = np.array([1 << x for x in range(len(flag_list))])

        bflags = BitFlags(bits, flag_list)

        items = ["INVALID", "SNOW_ICE", "INLAND_WATER", "SUSPECT",
                 "AC_FAIL", "CLOUD", "HISOLZEN", "OCNN_FAIL",
                 "CLOUD_MARGIN", "CLOUD_AMBIGUOUS", "LOWRW", "LAND"]

        mask = reduce(np.logical_or, [bflags[item] for item in items])
        expected = np.array([True, False, True, True, True, True, False,
                             False, True, True, False, False, False, False,
                             False, False, False, True, False, True, False,
                             False, False, True, True, False, False, True,
                             False])
        assert all(mask == expected)


class TestL1bBitFlags(unittest.TestCase):
    """Test the bitflag reading."""

    def test_bitflags(self):
        """Test the BitFlags class."""
        from functools import reduce

        import numpy as np

        from satpy.readers.olci_nc import BitFlags


        L1B_QUALITY_FLAGS = ["saturated@Oa21", "saturated@Oa20", "saturated@Oa19", "saturated@Oa18",
                             "saturated@Oa17", "saturated@Oa16", "saturated@Oa15", "saturated@Oa14",
                             "saturated@Oa13", "saturated@Oa12", "saturated@Oa11", "saturated@Oa10",
                             "saturated@Oa09", "saturated@Oa08", "saturated@Oa07", "saturated@Oa06",
                             "saturated@Oa05", "saturated@Oa04", "saturated@Oa03", "saturated@Oa02",
                             "saturated@Oa01", "dubious", "sun-glint_risk", "duplicated",
                             "cosmetic", "invalid", "straylight_risk", "bright",
                             "tidal_region", "fresh_inland_water", "coastline", "land"]

        DEFAULT_L1B_MASK_ITEMS = ["dubious", "sun-glint_risk", "duplicated", "cosmetic", "invalid",
                                  "straylight_risk", "bright", "tidal_region", "coastline", "land"]

        bits = np.array([1 << x for x in range(len(L1B_QUALITY_FLAGS))])

        bflags = BitFlags(bits, flag_list=L1B_QUALITY_FLAGS)

        mask = reduce(np.logical_or, [bflags[item] for item in DEFAULT_L1B_MASK_ITEMS])

        expected = np.array([False, False, False, False,
                             False, False, False, False,
                             False, False, False, False,
                             False, False, False, False,
                             False, False, False, False,
                             False, True, True, True,
                             True, True, True, True,
                             True, False, True, True,
                            ])
        assert all(mask == expected)
