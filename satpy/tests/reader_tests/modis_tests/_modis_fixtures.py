#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Satpy developers
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

"""MODIS L1b and L2 test fixtures."""

from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
import pytest
from pyhdf.SD import SD, SDC

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmpdir_factory

# Level 1 Fixtures

AVAILABLE_1KM_VIS_PRODUCT_NAMES = [str(x) for x in range(8, 13)]
AVAILABLE_1KM_VIS_PRODUCT_NAMES += ["13lo", "13hi", "14lo", "14hi"]
AVAILABLE_1KM_VIS_PRODUCT_NAMES += [str(x) for x in range(15, 20)]
AVAILABLE_1KM_IR_PRODUCT_NAMES = [str(x) for x in range(20, 37)]
AVAILABLE_1KM_PRODUCT_NAMES = AVAILABLE_1KM_VIS_PRODUCT_NAMES + AVAILABLE_1KM_IR_PRODUCT_NAMES
AVAILABLE_HKM_PRODUCT_NAMES = [str(x) for x in range(3, 8)]
AVAILABLE_QKM_PRODUCT_NAMES = ["1", "2"]
SCAN_LEN_5KM = 6  # 3 scans of 5km data
SCAN_WIDTH_5KM = 270
SCALE_FACTOR = 0.5
ADD_OFFSET = -0.5
RES_TO_REPEAT_FACTOR = {
    250: 20,
    500: 10,
    1000: 5,
    5000: 1,
}


def _shape_for_resolution(resolution: int) -> tuple[int, int]:
    assert resolution in RES_TO_REPEAT_FACTOR
    factor = RES_TO_REPEAT_FACTOR[resolution]
    if factor == 1:
        return SCAN_LEN_5KM, SCAN_WIDTH_5KM

    factor_1km = RES_TO_REPEAT_FACTOR[1000]
    shape_1km = (factor_1km * SCAN_LEN_5KM, factor_1km * SCAN_WIDTH_5KM + 4)
    factor //= 5
    return factor * shape_1km[0], factor * shape_1km[1]


def _generate_lonlat_data(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    shape = _shape_for_resolution(resolution)
    lat = np.repeat(np.linspace(35., 45., shape[0])[:, None], shape[1], 1)
    lat *= np.linspace(0.9, 1.1, shape[1])
    lon = np.repeat(np.linspace(-45., -35., shape[1])[None, :], shape[0], 0)
    lon *= np.linspace(0.9, 1.1, shape[0])[:, None]
    return lon.astype(np.float32), lat.astype(np.float32)


def _generate_angle_data(resolution: int) -> np.ndarray:
    shape = _shape_for_resolution(resolution)
    data = np.repeat(abs(np.linspace(-65.2, 65.4, shape[1]))[None, :], shape[0], 0)
    return (data * 100).astype(np.int16)


def _generate_visible_data(resolution: int, num_bands: int, dtype=np.uint16) -> np.ndarray:
    shape = _shape_for_resolution(resolution)
    data = np.ones((num_bands, shape[0], shape[1]), dtype=dtype)

    # add fill value to every band
    data[:, -1, -1] = 65535

    # add band 2 saturation and can't aggregate fill values
    data[1, -1, -2] = 65533
    data[1, -1, -3] = 65528
    return data


def _generate_visible_uncertainty_data(shape: tuple) -> np.ndarray:
    uncertainty = np.zeros(shape, dtype=np.uint8)
    uncertainty[:, -1, -1] = 15  # fill value
    uncertainty[:, -1, -2] = 15  # saturated
    uncertainty[:, -1, -3] = 15  # can't aggregate
    return uncertainty


def _get_lonlat_variable_info(resolution: int) -> dict:
    lon_5km, lat_5km = _generate_lonlat_data(resolution)
    return {
        "Latitude": {"data": lat_5km,
                     "type": SDC.FLOAT32,
                     "fill_value": -999,
                     "attrs": {"dim_labels": ["Cell_Along_Swath_5km:mod35", "Cell_Across_Swath_5km:mod35"]}},
        "Longitude": {"data": lon_5km,
                      "type": SDC.FLOAT32,
                      "fill_value": -999,
                      "attrs": {"dim_labels": ["Cell_Along_Swath_5km:mod35", "Cell_Across_Swath_5km:mod35"]}},
    }


def _get_angles_variable_info(resolution: int) -> dict:
    angle_data = _generate_angle_data(resolution)
    dim_factor = RES_TO_REPEAT_FACTOR[resolution] * 2
    angle_info = {
        "data": angle_data,
        "type": SDC.INT16,
        "fill_value": -32767,
        "attrs": {
            "dim_labels": [
                f"{dim_factor}*nscans:MODIS_SWATH_Type_L1B",
                "1KM_geo_dim:MODIS_SWATH_Type_L1B"],
            "scale_factor": 0.01,
            "add_offset": -0.01,
        },
    }
    angles_info = {}
    for var_name in ("SensorAzimuth", "SensorZenith", "SolarAzimuth", "SolarZenith"):
        angles_info[var_name] = angle_info
    return angles_info


def _get_visible_variable_info(var_name: str, resolution: int, bands: list[str]):
    num_bands = len(bands)
    data = _generate_visible_data(resolution, len(bands))
    uncertainty = _generate_visible_uncertainty_data(data.shape)
    dim_factor = RES_TO_REPEAT_FACTOR[resolution] * 2
    band_dim_name = f"Band_{resolution}_{num_bands}_RefSB:MODIS_SWATH_Type_L1B"
    row_dim_name = f"{dim_factor}*nscans:MODIS_SWATH_Type_L1B"
    col_dim_name = "Max_EV_frames:MODIS_SWATH_Type_L1B"
    return {
        var_name: {
            "data": data,
            "type": SDC.UINT16,
            "fill_value": 0,
            "attrs": {
                # dim_labels are just unique dimension names, may not match exactly with real world files
                "dim_labels": [band_dim_name,
                               row_dim_name,
                               col_dim_name],
                "valid_range": (0, 32767),
                "reflectance_scales": (2.0,) * num_bands,
                "reflectance_offsets": (-0.5,) * num_bands,
                "band_names": ",".join(bands),
            },
        },
        var_name + "_Uncert_Indexes": {
            "data": uncertainty,
            "type": SDC.UINT8,
            "fill_value": 255,
            "attrs": {
                "dim_labels": [band_dim_name,
                               row_dim_name,
                               col_dim_name],
            },
        },
    }


def _get_emissive_variable_info(var_name: str, resolution: int, bands: list[str]):
    num_bands = len(bands)
    data = _generate_visible_data(resolution, len(bands))
    dim_factor = RES_TO_REPEAT_FACTOR[resolution] * 2
    band_dim_name = f"Band_{resolution}_{num_bands}_Emissive:MODIS_SWATH_Type_L1B"
    row_dim_name = f"{dim_factor}*nscans:MODIS_SWATH_Type_L1B"
    col_dim_name = "Max_EV_frames:MODIS_SWATH_Type_L1B"
    return {
        var_name: {
            "data": data,
            "type": SDC.UINT16,
            "fill_value": 0,
            "attrs": {
                "dim_labels": [band_dim_name,
                               row_dim_name,
                               col_dim_name],
                "valid_range": (0, 32767),
                "band_names": ",".join(bands),
            },
        },
        var_name + "_Uncert_Indexes": {
            "data": np.zeros(data.shape, dtype=np.uint8),
            "type": SDC.UINT8,
            "fill_value": 255,
            "attrs": {
                "dim_labels": [band_dim_name,
                               row_dim_name,
                               col_dim_name],
            },
        },
    }


def _get_l1b_geo_variable_info(filename: str,
                               geo_resolution: int,
                               include_angles: bool = True
                               ) -> dict:
    variables_info = {}
    variables_info.update(_get_lonlat_variable_info(geo_resolution))
    if include_angles:
        variables_info.update(_get_angles_variable_info(geo_resolution))
    return variables_info

def _get_l3_land_cover_info() -> dict:

    lc_data = np.zeros((2400, 2400), dtype=np.uint8)

    variables_info = \
    {
        "LC_Type1": {"data": lc_data,
                     "type": SDC.UINT8,
                     "fill_value": 255,
                     "attrs": {
                         "dim_labels": ["YDim:MCD12Q1", "XDim:MCD12Q1"],
                     },
                     },
        "LC_Type2": {"data": lc_data,
                     "type": SDC.UINT8,
                     "fill_value": 255,
                     "attrs": {
                         "dim_labels": ["YDim:MCD12Q1", "XDim:MCD12Q1"],
                     },
                     },
    }

    return variables_info


def generate_nasa_l1b_filename(prefix):
    """Generate a filename that follows NASA MODIS L1b convention."""
    now = dt.datetime.now()
    return f"{prefix}_A{now:%y%j_%H%M%S}_{now:%Y%j%H%M%S}.hdf"


def generate_imapp_filename(suffix):
    """Generate a filename that follows IMAPP MODIS L1b convention."""
    now = dt.datetime.now()
    return f"t1.{now:%y%j.%H%M}.{suffix}.hdf"


def create_hdfeos_test_file(filename: str,
                            variable_infos: dict,
                            struct_meta: Optional[str] = None,
                            core_meta: Optional[str] = None,
                            archive_meta: Optional[str] = None,
                            ) -> None:
    """Create a fake MODIS L1b HDF4 file with headers.

    Args:
        filename: Full path of filename to be created.
        variable_infos: Dictionary mapping HDF4 variable names to dictionary
            of variable information (see ``_add_variable_to_file``).
        struct_meta: Contents of the 'StructMetadata.0' header.
        core_meta: Contents of the 'CoreMetadata.0' header.
        archive_meta:Contents of the 'ArchiveMetadata.0' header.

    """
    h = SD(filename, SDC.WRITE | SDC.CREATE)

    if struct_meta:
        setattr(h, "StructMetadata.0", struct_meta)
    if core_meta:
        setattr(h, "CoreMetadata.0", core_meta)
    if archive_meta:
        setattr(h, "ArchiveMetadata.0", archive_meta)

    for var_name, var_info in variable_infos.items():
        _add_variable_to_file(h, var_name, var_info)

    h.end()


def _add_variable_to_file(h, var_name, var_info):
    v = h.create(var_name, var_info["type"], var_info["data"].shape)
    v[:] = var_info["data"]
    dim_count = 0
    for dimension_name in var_info["attrs"]["dim_labels"]:
        v.dim(dim_count).setname(dimension_name)
        dim_count += 1
    v.setfillvalue(var_info["fill_value"])
    v.scale_factor = var_info["attrs"].get("scale_factor", SCALE_FACTOR)
    v.add_offset = var_info["attrs"].get("add_offset", ADD_OFFSET)
    for attr_key, attr_val in var_info["attrs"].items():
        if attr_key == "dim_labels":
            continue
        setattr(v, attr_key, attr_val)


def _create_core_metadata(file_shortname: str) -> str:
    beginning_date = dt.datetime.now()
    ending_date = beginning_date + dt.timedelta(minutes=5)
    core_metadata_header = "GROUP = INVENTORYMETADATA\nGROUPTYPE = MASTERGROUP\n\n" \
                           'GROUP = RANGEDATETIME\n\nOBJECT = RANGEBEGINNINGDATE\nNUM_VAL = 1\nVALUE = "{}"\n' \
                           "END_OBJECT = RANGEBEGINNINGDATE\n\nOBJECT = RANGEBEGINNINGTIME\n" \
                           'NUM_VAL = 1\nVALUE = "{}"\n' \
                           "END_OBJECT = RANGEBEGINNINGTIME\n\nOBJECT = RANGEENDINGDATE\n" \
                           'NUM_VAL = 1\nVALUE = "{}"\n' \
                           'END_OBJECT = RANGEENDINGDATE\n\nOBJECT = RANGEENDINGTIME\nNUM_VAL = 1\nVALUE = "{}"\n' \
                           "END_OBJECT = RANGEENDINGTIME\nEND_GROUP = RANGEDATETIME"
    core_metadata_header = core_metadata_header.format(
        beginning_date.strftime("%Y-%m-%d"),
        beginning_date.strftime("%H:%M:%S.%f"),
        ending_date.strftime("%Y-%m-%d"),
        ending_date.strftime("%H:%M:%S.%f")
    )
    inst_metadata = "GROUP = ASSOCIATEDPLATFORMINSTRUMENTSENSOR\n\n" \
                    'OBJECT = ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER\nCLASS = "1"\n\n' \
                    'OBJECT = ASSOCIATEDSENSORSHORTNAME\nCLASS = "1"\nNUM_VAL = 1\n' \
                    'VALUE = "MODIS"\nEND_OBJECT = ASSOCIATEDSENSORSHORTNAME\n\n' \
                    'OBJECT = ASSOCIATEDPLATFORMSHORTNAME\nCLASS = "1"\nNUM_VAL = 1\n' \
                    'VALUE = "Terra"\nEND_OBJECT = ASSOCIATEDPLATFORMSHORTNAME\n\n' \
                    'OBJECT = ASSOCIATEDINSTRUMENTSHORTNAME\nCLASS = "1"\nNUM_VAL = 1\n' \
                    'VALUE = "MODIS"\nEND_OBJECT = ASSOCIATEDINSTRUMENTSHORTNAME\n\n' \
                    "END_OBJECT = ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER\n\n" \
                    "END_GROUP              = ASSOCIATEDPLATFORMINSTRUMENTSENSOR\n\n"
    collection_metadata = "GROUP = COLLECTIONDESCRIPTIONCLASS\n\nOBJECT = SHORTNAME\nNUM_VAL = 1\n" \
                          f"VALUE = {file_shortname!r}\nEND_OBJECT = SHORTNAME\n\n" \
                          "OBJECT = VERSIONID\nNUM_VAL = 1\nVALUE = 6\nEND_OBJECT = VERSIONID\n\n" \
                          "END_GROUP = COLLECTIONDESCRIPTIONCLASS\n\n"
    core_metadata_header += "\n\n" + inst_metadata + collection_metadata
    return core_metadata_header


def _create_struct_metadata(geo_resolution: int) -> str:
    geo_dim_factor = RES_TO_REPEAT_FACTOR[geo_resolution] * 2
    struct_metadata_header = "GROUP=SwathStructure\n" \
                             "GROUP=SWATH_1\n" \
                             "GROUP=DimensionMap\n" \
                             "OBJECT=DimensionMap_2\n" \
                             f'GeoDimension="{geo_dim_factor}*nscans"\n' \
                             "END_OBJECT=DimensionMap_2\n" \
                             "END_GROUP=DimensionMap\n" \
                             "END_GROUP=SWATH_1\n" \
                             "END_GROUP=SwathStructure\nEND"
    return struct_metadata_header


def _create_struct_metadata_cmg(ftype: str) -> str:
    # Case of a MOD09 file
    if ftype == "MOD09":
        gridline = 'GridName="MOD09CMG"\n'
        upleft = "UpperLeftPointMtrs=(-180000000.000000,90000000.000000)\n"
        upright = "LowerRightMtrs=(180000000.000000,-90000000.000000)\n"
        XDim=7200
        YDim=3600
    # Case of a MCD12Q1 file
    elif ftype == "MCD12Q1":
        gridline = 'GridName="MCD12Q1"\n'
        upleft = "UpperLeftPointMtrs=(-8895604.157333,-1111950.519667)\n"
        upright = "LowerRightMtrs=(-7783653.637667,-2223901.039333)\n"
        XDim=2400
        YDim=2400
    # Case of a MCD43 file
    else:
        gridline = 'GridName="MCD_CMG_BRDF_0.05Deg"\n'
        upleft = "UpperLeftPointMtrs=(-180.000000,90.000000)\n"
        upright = "LowerRightMtrs=(180.000000,-90.000000)\n"
        XDim=7200
        YDim=3600

    struct_metadata_header = ("GROUP=SwathStructure\n"
                              "END_GROUP=SwathStructure\n"
                              "GROUP=GridStructure\n"
                              "GROUP=GRID_1\n"
                              f"{gridline}\n"
                              f"XDim={XDim}\n"
                              f"YDim={YDim}\n"
                              f"{upleft}\n"
                              f"{upright}\n"
                              "END_GROUP=GRID_1\n"
                              "END_GROUP=GridStructure\nEND")
    return struct_metadata_header


def _create_header_metadata() -> str:
    archive_metadata_header = ("GROUP = ARCHIVEDMETADATA\n"
                               'TEST_URL = "http://modis.gsfc.nasa.gov/?some_val=100"\n'
                               "END_GROUP = ARCHIVEDMETADATA\nEND")
    return archive_metadata_header


@pytest.fixture(scope="session")
def modis_l1b_nasa_mod021km_file(tmpdir_factory) -> list[str]:
    """Create a single MOD021KM file following standard NASA file scheme."""
    filename = generate_nasa_l1b_filename("MOD021km")
    full_path = str(tmpdir_factory.mktemp("modis_l1b").join(filename))
    variable_infos = _get_l1b_geo_variable_info(filename, 5000, include_angles=True)
    variable_infos.update(_get_visible_variable_info("EV_1KM_RefSB", 1000, AVAILABLE_1KM_VIS_PRODUCT_NAMES))
    variable_infos.update(_get_visible_variable_info("EV_500_Aggr1km_RefSB", 1000, AVAILABLE_HKM_PRODUCT_NAMES))
    variable_infos.update(_get_visible_variable_info("EV_250_Aggr1km_RefSB", 1000, AVAILABLE_QKM_PRODUCT_NAMES))
    variable_infos.update(_get_emissive_variable_info("EV_1KM_Emissive", 1000, AVAILABLE_1KM_IR_PRODUCT_NAMES))
    create_hdfeos_test_file(full_path,
                            variable_infos,
                            _create_struct_metadata(5000),
                            _create_core_metadata("MOD021KM"),
                            _create_header_metadata())
    return [full_path]


@pytest.fixture(scope="session")
def modis_l1b_imapp_1000m_file(tmpdir_factory) -> list[str]:
    """Create a single MOD021KM file following IMAPP file scheme."""
    filename = generate_imapp_filename("1000m")
    full_path = str(tmpdir_factory.mktemp("modis_l1b").join(filename))
    variable_infos = _get_l1b_geo_variable_info(filename, 5000, include_angles=True)
    variable_infos.update(_get_visible_variable_info("EV_1KM_RefSB", 1000, AVAILABLE_1KM_VIS_PRODUCT_NAMES))
    variable_infos.update(_get_visible_variable_info("EV_500_Aggr1km_RefSB", 1000, AVAILABLE_HKM_PRODUCT_NAMES))
    variable_infos.update(_get_visible_variable_info("EV_250_Aggr1km_RefSB", 1000, AVAILABLE_QKM_PRODUCT_NAMES))
    variable_infos.update(_get_emissive_variable_info("EV_1KM_Emissive", 1000, AVAILABLE_1KM_IR_PRODUCT_NAMES))
    create_hdfeos_test_file(full_path,
                            variable_infos,
                            _create_struct_metadata(5000),
                            _create_core_metadata("MOD021KM"),
                            _create_header_metadata())
    return [full_path]


@pytest.fixture(scope="session")
def modis_l1b_nasa_mod02hkm_file(tmpdir_factory) -> list[str]:
    """Create a single MOD02HKM file following standard NASA file scheme."""
    filename = generate_nasa_l1b_filename("MOD02Hkm")
    full_path = str(tmpdir_factory.mktemp("modis_l1b").join(filename))
    variable_infos = _get_l1b_geo_variable_info(filename, 1000, include_angles=False)
    variable_infos.update(_get_visible_variable_info("EV_500_RefSB", 250, AVAILABLE_QKM_PRODUCT_NAMES))
    create_hdfeos_test_file(full_path,
                            variable_infos,
                            _create_struct_metadata(1000),
                            _create_core_metadata("MOD02HKM"),
                            _create_header_metadata())
    return [full_path]


@pytest.fixture
def modis_l1b_nasa_mod02qkm_file(tmpdir_factory) -> list[str]:
    """Create a single MOD02QKM file following standard NASA file scheme."""
    filename = generate_nasa_l1b_filename("MOD02Qkm")
    full_path = str(tmpdir_factory.mktemp("modis_l1b").join(filename))
    variable_infos = _get_l1b_geo_variable_info(filename, 1000, include_angles=False)
    variable_infos.update(_get_visible_variable_info("EV_250_RefSB", 250, AVAILABLE_QKM_PRODUCT_NAMES))
    create_hdfeos_test_file(full_path,
                            variable_infos,
                            _create_struct_metadata(1000),
                            _create_core_metadata("MOD02QKM"),
                            _create_header_metadata())
    return [full_path]


@pytest.fixture(scope="session")
def modis_l1b_nasa_mod03_file(tmpdir_factory) -> list[str]:
    """Create a single MOD03 file following standard NASA file scheme."""
    filename = generate_nasa_l1b_filename("MOD03")
    full_path = str(tmpdir_factory.mktemp("modis_l1b").join(filename))
    variable_infos = _get_l1b_geo_variable_info(filename, 1000, include_angles=True)
    create_hdfeos_test_file(full_path,
                            variable_infos,
                            _create_struct_metadata(1000),
                            _create_core_metadata("MOD03"),
                            _create_header_metadata())
    return [full_path]


@pytest.fixture(scope="session")
def modis_l1b_imapp_geo_file(tmpdir_factory) -> list[str]:
    """Create a single geo file following standard IMAPP file scheme."""
    filename = generate_imapp_filename("geo")
    full_path = str(tmpdir_factory.mktemp("modis_l1b").join(filename))
    variable_infos = _get_l1b_geo_variable_info(filename, 1000, include_angles=True)
    create_hdfeos_test_file(full_path,
                            variable_infos,
                            _create_struct_metadata(1000),
                            _create_core_metadata("MOD03"),
                            _create_header_metadata())
    return [full_path]


@pytest.fixture(scope="session")
def modis_l1b_nasa_1km_mod03_files(modis_l1b_nasa_mod021km_file, modis_l1b_nasa_mod03_file) -> list[str]:
    """Create input files including the 1KM and MOD03 files."""
    return modis_l1b_nasa_mod021km_file + modis_l1b_nasa_mod03_file


# Level 2 Fixtures


def _get_basic_variable_info(var_name: str, resolution: int) -> dict:
    shape = _shape_for_resolution(resolution)
    data = np.ones((shape[0], shape[1]), dtype=np.uint16)
    row_dim_name = f"Cell_Along_Swath_{resolution}m:modl2"
    col_dim_name = f"Cell_Across_Swath_{resolution}m:modl2"
    return {
        var_name: {
            "data": data,
            "type": SDC.UINT16,
            "fill_value": 0,
            "attrs": {
                # dim_labels are just unique dimension names, may not match exactly with real world files
                "dim_labels": [row_dim_name,
                               col_dim_name],
                "valid_range": (0, 32767),
                "scale_factor": 2.0,
                "add_offset": -1.0,
            },
        },
    }


def _get_cloud_mask_variable_info(var_name: str, resolution: int) -> dict:
    num_bytes = 6
    shape = _shape_for_resolution(resolution)
    data = np.zeros((num_bytes, shape[0], shape[1]), dtype=np.int8)
    byte_dim_name = "Byte_Segment:mod35"
    row_dim_name = "Cell_Along_Swath_1km:mod35"
    col_dim_name = "Cell_Across_Swath_1km:mod35"
    return {
        var_name: {
            "data": data,
            "type": SDC.INT8,
            "fill_value": 0,
            "attrs": {
                # dim_labels are just unique dimension names, may not match exactly with real world files
                "dim_labels": [byte_dim_name,
                               row_dim_name,
                               col_dim_name],
                "valid_range": (0, -1),
                "scale_factor": 1.,
                "add_offset": 0.,
            },
        },
        "Quality_Assurance": {
            "data": np.ones((shape[0], shape[1], 10), dtype=np.int8),
            "type": SDC.INT8,
            "fill_value": 0,
            "attrs": {
                # dim_labels are just unique dimension names, may not match exactly with real world files
                "dim_labels": [row_dim_name,
                               col_dim_name,
                               "Quality_Dimension:mod35"],
                "valid_range": (0, -1),
                "scale_factor": 2.,
                "add_offset": -0.5,
            },
        },
    }


def _get_l3_refl_variable_info(var_name: str) -> dict:
    shape = (3600, 7200)
    data = np.zeros((shape[0], shape[1]), dtype=np.int16)
    row_dim_name = "XDim"
    col_dim_name = "YDim"
    return {
        var_name: {
            "data": data,
            "type": SDC.INT16,
            "fill_value": -28672,
            "attrs": {
                # dim_labels are just unique dimension names, may not match exactly with real world files
                "dim_labels": [row_dim_name,
                               col_dim_name],
                "valid_range": (-100, 16000),
                "scale_factor": 1e-4,
                "add_offset": 0.,
            },
        },
    }


def _get_mask_byte1_variable_info() -> dict:
    shape = _shape_for_resolution(1000)
    data = np.zeros((shape[0], shape[1]), dtype=np.uint16)
    row_dim_name = "Cell_Along_Swath_1km:mod35"
    col_dim_name = "Cell_Across_Swath_1km:mod35"
    return {
        "MODIS_Cloud_Mask": {
            "data": data,
            "type": SDC.UINT16,
            "fill_value": 9999,
            "attrs": {
                # dim_labels are just unique dimension names, may not match exactly with real world files
                "dim_labels": [row_dim_name,
                               col_dim_name],
                "valid_range": (0, 4),
                "scale_factor": 2,
                "add_offset": -1,
            },

        },
        "MODIS_Simple_LandSea_Mask": {
            "data": data,
            "type": SDC.UINT16,
            "fill_value": 9999,
            "attrs": {
                # dim_labels are just unique dimension names, may not match exactly with real world files
                "dim_labels": [row_dim_name,
                               col_dim_name],
                "valid_range": (0, 4),
                "scale_factor": 2,
                "add_offset": -1,
            },
        },
        "MODIS_Snow_Ice_Flag": {
            "data": data,
            "type": SDC.UINT16,
            "fill_value": 9999,
            "attrs": {
                # dim_labels are just unique dimension names, may not match exactly with real world files
                "dim_labels": [row_dim_name,
                               col_dim_name],
                "valid_range": (0, 2),
                "scale_factor": 2,
                "add_offset": -1,
            },
        },
    }


def generate_nasa_l2_filename(prefix: str) -> str:
    """Generate a file name that follows MODIS 35 L2 convention in a temporary directory."""
    now = dt.datetime.now()
    return f"{prefix}_L2.A{now:%Y%j.%H%M}.061.{now:%Y%j%H%M%S}.hdf"

def generate_nasa_l3_tile_filename(prefix: str) -> str:
    """Generate a file name that follows MODIS sinusoidal grid tile pattern."""
    now = dt.datetime.now()
    return f"{prefix}.A{now:%Y}001.h34v07.061.{now:%Y%j%H%M%S}.hdf"

@pytest.fixture(scope="session")
def modis_l2_nasa_mod35_file(tmpdir_factory) -> list[str]:
    """Create a single MOD35 L2 HDF4 file with headers."""
    filename = generate_nasa_l2_filename("MOD35")
    full_path = str(tmpdir_factory.mktemp("modis_l2").join(filename))
    variable_infos = _get_l1b_geo_variable_info(filename, 5000, include_angles=True)
    variable_infos.update(_get_cloud_mask_variable_info("Cloud_Mask", 1000))
    create_hdfeos_test_file(full_path,
                            variable_infos,
                            _create_struct_metadata(5000),
                            _create_core_metadata("MOD35"),
                            _create_header_metadata())
    return [full_path]

@pytest.fixture(scope="session")
def modis_l3_nasa_mcd12q1_file(tmpdir_factory) -> list[str]:
    """Create a single MOD35 L2 HDF4 file with headers."""
    filename = generate_nasa_l3_tile_filename("MCD12Q1")
    full_path = str(tmpdir_factory.mktemp("modis_l2").join(filename))
    variable_infos = _get_l3_land_cover_info()
    archive_header = \
"""GROUP                  = ARCHIVEDMETADATA
  GROUPTYPE            = MASTERGROUP

  OBJECT                 = NADIRDATARESOLUTION
    NUM_VAL              = 1
    VALUE                = "500m"
  END_OBJECT             = NADIRDATARESOLUTION

END_GROUP              = ARCHIVEDMETADATA

END
"""
    create_hdfeos_test_file(full_path,
                            variable_infos,
                            _create_struct_metadata_cmg("MCD12Q1"),
                            _create_core_metadata("MCD12Q1"),
                            archive_header)
    return [full_path]

def generate_nasa_l3_filename(prefix: str) -> str:
    """Generate a file name that follows MODIS 09 L3 convention in a temporary directory."""
    now = dt.datetime.now()
    return f"{prefix}.A{now:%Y%j}.061.{now:%Y%j%H%M%S}.hdf"


def modis_l3_file(tmpdir_factory, f_prefix, var_name, f_short):
    """Create a MODIS L3 file of the desired type."""
    filename = generate_nasa_l3_filename(f_prefix)
    full_path = str(tmpdir_factory.mktemp("modis_l3").join(filename))
    variable_infos = _get_l3_refl_variable_info(var_name)
    create_hdfeos_test_file(full_path,
                            variable_infos,
                            _create_struct_metadata_cmg(f_short),
                            _create_core_metadata(f_short),
                            _create_header_metadata())
    return [full_path]


@pytest.fixture(scope="session")
def modis_l3_nasa_mod09_file(tmpdir_factory) -> list[str]:
    """Create a single MOD09 L3 HDF4 file with headers."""
    return modis_l3_file(tmpdir_factory,
                         "MOD09CMG",
                         "Coarse_Resolution_Surface_Reflectance_Band_2",
                         "MOD09")


@pytest.fixture(scope="session")
def modis_l3_nasa_mod43_file(tmpdir_factory) -> list[str]:
    """Create a single MVCD43 L3 HDF4 file with headers."""
    return modis_l3_file(tmpdir_factory,
                         "MCD43C1",
                         "BRDF_Albedo_Parameter1_Band2",
                         "MCD43C1")


@pytest.fixture(scope="session")
def modis_l2_nasa_mod35_mod03_files(modis_l2_nasa_mod35_file, modis_l1b_nasa_mod03_file) -> list[str]:
    """Create a MOD35 L2 HDF4 file and MOD03 L1b geolocation file."""
    return modis_l2_nasa_mod35_file + modis_l1b_nasa_mod03_file


@pytest.fixture(scope="session")
def modis_l2_nasa_mod06_file(tmpdir_factory) -> list[str]:
    """Create a single MOD06 L2 HDF4 file with headers."""
    filename = generate_nasa_l2_filename("MOD06")
    full_path = str(tmpdir_factory.mktemp("modis_l2").join(filename))
    variable_infos = _get_l1b_geo_variable_info(filename, 5000, include_angles=True)
    variable_infos.update(_get_basic_variable_info("Surface_Pressure", 5000))
    create_hdfeos_test_file(full_path,
                            variable_infos,
                            _create_struct_metadata(5000),
                            _create_core_metadata("MOD06"),
                            _create_header_metadata())
    return [full_path]


@pytest.fixture(scope="session")
def modis_l2_imapp_snowmask_file(tmpdir_factory) -> list[str]:
    """Create a single IMAPP snowmask L2 HDF4 file with headers."""
    filename = generate_imapp_filename("snowmask")
    full_path = str(tmpdir_factory.mktemp("modis_l2").join(filename))
    variable_infos = _get_l1b_geo_variable_info(filename, 5000, include_angles=False)
    variable_infos.update(_get_basic_variable_info("Snow_Mask", 1000))
    create_hdfeos_test_file(full_path, variable_infos)
    return [full_path]


@pytest.fixture(scope="session")
def modis_l2_imapp_snowmask_geo_files(modis_l2_imapp_snowmask_file, modis_l1b_nasa_mod03_file) -> list[str]:
    """Create the IMAPP snowmask and geo HDF4 files."""
    return modis_l2_imapp_snowmask_file + modis_l1b_nasa_mod03_file


@pytest.fixture(scope="session")
def modis_l2_imapp_mask_byte1_file(tmpdir_factory) -> list[str]:
    """Create a single IMAPP mask_byte1 L2 HDF4 file with headers."""
    filename = generate_imapp_filename("mask_byte1")
    full_path = str(tmpdir_factory.mktemp("modis_l2").join(filename))
    variable_infos = _get_l1b_geo_variable_info(filename, 5000, include_angles=False)
    variable_infos.update(_get_mask_byte1_variable_info())
    create_hdfeos_test_file(full_path, variable_infos)
    return [full_path]


@pytest.fixture(scope="session")
def modis_l2_imapp_mask_byte1_geo_files(modis_l2_imapp_mask_byte1_file, modis_l1b_nasa_mod03_file) -> list[str]:
    """Create the IMAPP mask_byte1 and geo HDF4 files."""
    return modis_l2_imapp_mask_byte1_file + modis_l1b_nasa_mod03_file
