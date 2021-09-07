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
"""Unit tests for MODIS L1b HDF reader."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pytest

from pyhdf.SD import SD, SDC

from satpy import available_readers, Scene

# Mock MODIS HDF4 file
AVAILABLE_1KM_VIS_PRODUCT_NAMES = list(range(8, 13)) + ['13lo', '13hi', '14lo', '14hi'] + list(range(15, 20))
AVAILABLE_1KM_VIS_PRODUCT_NAMES = [str(x) for x in AVAILABLE_1KM_VIS_PRODUCT_NAMES]
AVAILABLE_1KM_IR_PRODUCT_NAMES = [str(x) for x in range(20, 37)]
AVAILABLE_1KM_PRODUCT_NAMES = AVAILABLE_1KM_VIS_PRODUCT_NAMES + AVAILABLE_1KM_IR_PRODUCT_NAMES
AVAILABLE_HKM_PRODUCT_NAMES = [str(x) for x in list(range(3, 8))]
AVAILABLE_QKM_PRODUCT_NAMES = ['1', '2']

SCAN_LEN_5KM = 406
SCAN_WIDTH_5KM = 270
SCALE_FACTOR = 1


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


def _generate_lonlat_data(resolution: int) -> np.ndarray:
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
    data = np.zeros((num_bands, shape[0], shape[1]), dtype=dtype)
    return data


def _get_lonlat_variable_info(resolution: int) -> dict:
    lon_5km, lat_5km = _generate_lonlat_data(resolution)
    return {
        'Latitude': {'data': lat_5km,
                     'type': SDC.FLOAT32,
                     'fill_value': -999,
                     'attrs': {'dim_labels': ['Cell_Along_Swath_5km:mod35', 'Cell_Across_Swath_5km:mod35']}},
        'Longitude': {'data': lon_5km,
                      'type': SDC.FLOAT32,
                      'fill_value': -999,
                      'attrs': {'dim_labels': ['Cell_Along_Swath_5km:mod35', 'Cell_Across_Swath_5km:mod35']}},
    }


def _get_angles_variable_info(resolution: int) -> dict:
    angle_data = _generate_angle_data(resolution)
    dim_factor = RES_TO_REPEAT_FACTOR[resolution] * 2
    angle_info = {
        'data': angle_data,
        'type': SDC.INT16,
        'fill_value': -32767,
        'attrs': {
            'dim_labels': [
                f'{dim_factor}*nscans:MODIS_SWATH_Type_L1B',
                '1KM_geo_dim:MODIS_SWATH_Type_L1B'],
            'scale_factor': 0.01
        },
    }
    angles_info = {}
    for var_name in ('SensorAzimuth', 'SensorZenith', 'SolarAzimuth', 'SolarZenith'):
        angles_info[var_name] = angle_info
    return angles_info


def _get_visible_variable_info(var_name: str, resolution: int, bands: list[str]):
    num_bands = len(bands)
    data = _generate_visible_data(resolution, len(bands))
    dim_factor = RES_TO_REPEAT_FACTOR[resolution] * 2
    band_dim_name = f"Band_{resolution}_{num_bands}_RefSB:MODIS_SWATH_Type_L1B"
    row_dim_name = f'{dim_factor}*nscans:MODIS_SWATH_Type_L1B'
    col_dim_name = 'Max_EV_frames:MODIS_SWATH_Type_L1B'
    return {
        var_name: {
            'data': data,
            'type': SDC.UINT16,
            'fill_value': 0,
            'attrs': {
                # dim_labels are just unique dimension names, may not match exactly with real world files
                'dim_labels': [band_dim_name,
                               row_dim_name,
                               col_dim_name],
                'valid_range': (0, 32767),
                'reflectance_scales': (1,) * num_bands,
                'reflectance_offsets': (0,) * num_bands,
                'band_names': ",".join(bands),
            },
        },
        var_name + '_Uncert_Indexes': {
            'data': np.zeros(data.shape, dtype=np.uint8),
            'type': SDC.UINT8,
            'fill_value': 255,
            'attrs': {
                'dim_labels': [band_dim_name,
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
    row_dim_name = f'{dim_factor}*nscans:MODIS_SWATH_Type_L1B'
    col_dim_name = 'Max_EV_frames:MODIS_SWATH_Type_L1B'
    return {
        var_name: {
            'data': data,
            'type': SDC.UINT16,
            'fill_value': 0,
            'attrs': {
                'dim_labels': [band_dim_name,
                               row_dim_name,
                               col_dim_name],
                'valid_range': (0, 32767),
                'band_names': ",".join(bands),
            },
        },
        var_name + '_Uncert_Indexes': {
            'data': np.zeros(data.shape, dtype=np.uint8),
            'type': SDC.UINT8,
            'fill_value': 255,
            'attrs': {
                'dim_labels': [band_dim_name,
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


def generate_nasa_l1b_filename(prefix):
    """Generate a filename that follows NASA MODIS L1b convention."""
    now = datetime.now()
    return f'{prefix}_A{now:%y%j_%H%M%S}_{now:%Y%j%H%M%S}.hdf'


def generate_imapp_l1b_filename(suffix):
    """Generate a filename that follows IMAPP MODIS L1b convention."""
    now = datetime.now()
    return f'a1.{now:%y%j.%H%M}.{suffix}.hdf'


def create_hdfeos_test_file(filename: str,
                            variable_infos: dict,
                            geo_resolution: Optional[int] = None,
                            file_shortname: Optional[str] = None,
                            include_metadata: bool = True):
    """Create a fake MODIS L1b HDF4 file with headers."""
    h = SD(filename, SDC.WRITE | SDC.CREATE)

    if include_metadata:
        if geo_resolution is None or file_shortname is None:
            raise ValueError("'geo_resolution' and 'file_shortname' are required when including metadata.")
        setattr(h, 'CoreMetadata.0', _create_core_metadata(file_shortname))  # noqa
        setattr(h, 'StructMetadata.0', _create_struct_metadata(geo_resolution))  # noqa
        setattr(h, 'ArchiveMetadata.0', _create_header_metadata())  # noqa

    for var_name, var_info in variable_infos.items():
        _add_variable_to_file(h, var_name, var_info)

    h.end()


def _add_variable_to_file(h, var_name, var_info):
    v = h.create(var_name, var_info['type'], var_info['data'].shape)
    v[:] = var_info['data']
    dim_count = 0
    for dimension_name in var_info['attrs']['dim_labels']:
        v.dim(dim_count).setname(dimension_name)
        dim_count += 1
    v.setfillvalue(var_info['fill_value'])
    v.scale_factor = var_info['attrs'].get('scale_factor', SCALE_FACTOR)
    for attr_key, attr_val in var_info['attrs'].items():
        if attr_key == 'dim_labels':
            continue
        setattr(v, attr_key, attr_val)


def _create_core_metadata(file_shortname: str) -> str:
    beginning_date = datetime.now()
    ending_date = beginning_date + timedelta(minutes=5)
    core_metadata_header = "GROUP = INVENTORYMETADATA\nGROUPTYPE = MASTERGROUP\n\n" \
                           "GROUP = RANGEDATETIME\n\nOBJECT = RANGEBEGINNINGDATE\nNUM_VAL = 1\nVALUE = \"{}\"\n" \
                           "END_OBJECT = RANGEBEGINNINGDATE\n\nOBJECT = RANGEBEGINNINGTIME\n" \
                           "NUM_VAL = 1\nVALUE = \"{}\"\n" \
                           "END_OBJECT = RANGEBEGINNINGTIME\n\nOBJECT = RANGEENDINGDATE\n" \
                           "NUM_VAL = 1\nVALUE = \"{}\"\n" \
                           "END_OBJECT = RANGEENDINGDATE\n\nOBJECT = RANGEENDINGTIME\nNUM_VAL = 1\nVALUE = \"{}\"\n" \
                           "END_OBJECT = RANGEENDINGTIME\nEND_GROUP = RANGEDATETIME"
    core_metadata_header = core_metadata_header.format(
        beginning_date.strftime("%Y-%m-%d"),
        beginning_date.strftime("%H:%M:%S.%f"),
        ending_date.strftime("%Y-%m-%d"),
        ending_date.strftime("%H:%M:%S.%f")
    )
    inst_metadata = "GROUP = ASSOCIATEDPLATFORMINSTRUMENTSENSOR\n\n" \
                    "OBJECT = ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER\nCLASS = \"1\"\n\n" \
                    "OBJECT = ASSOCIATEDSENSORSHORTNAME\nCLASS = \"1\"\nNUM_VAL = 1\n" \
                    "VALUE = \"MODIS\"\nEND_OBJECT = ASSOCIATEDSENSORSHORTNAME\n\n" \
                    "OBJECT = ASSOCIATEDPLATFORMSHORTNAME\nCLASS = \"1\"\nNUM_VAL = 1\n" \
                    "VALUE = \"Terra\"\nEND_OBJECT = ASSOCIATEDPLATFORMSHORTNAME\n\n" \
                    "OBJECT = ASSOCIATEDINSTRUMENTSHORTNAME\nCLASS = \"1\"\nNUM_VAL = 1\n" \
                    "VALUE = \"MODIS\"\nEND_OBJECT = ASSOCIATEDINSTRUMENTSHORTNAME\n\n" \
                    "END_OBJECT = ASSOCIATEDPLATFORMINSTRUMENTSENSORCONTAINER\n\n" \
                    "END_GROUP              = ASSOCIATEDPLATFORMINSTRUMENTSENSOR\n\n"
    collection_metadata = "GROUP = COLLECTIONDESCRIPTIONCLASS\n\nOBJECT = SHORTNAME\nNUM_VAL = 1\n" \
                          f"VALUE = \"{file_shortname}\"\nEND_OBJECT = SHORTNAME\n\n" \
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
                             f"GeoDimension=\"{geo_dim_factor}*nscans\"\n" \
                             "END_OBJECT=DimensionMap_2\n" \
                             "END_GROUP=DimensionMap\n" \
                             "END_GROUP=SWATH_1\n" \
                             "END_GROUP=SwathStructure\nEND"
    return struct_metadata_header


def _create_header_metadata() -> str:
    archive_metadata_header = "GROUP = ARCHIVEDMETADATA\nEND_GROUP = ARCHIVEDMETADATA\nEND"
    return archive_metadata_header


@pytest.fixture
def modis_l1b_nasa_mod021km_file(tmpdir) -> list[str]:
    """Create a single MOD021KM file following standard NASA file scheme."""
    filename = generate_nasa_l1b_filename("MOD021km")
    full_path = os.path.join(str(tmpdir), filename)
    variable_infos = _get_l1b_geo_variable_info(filename, 5000, include_angles=True)
    variable_infos.update(_get_visible_variable_info("EV_1KM_RefSB", 1000, AVAILABLE_1KM_VIS_PRODUCT_NAMES))
    variable_infos.update(_get_visible_variable_info("EV_500_Aggr1km_RefSB", 1000, AVAILABLE_HKM_PRODUCT_NAMES))
    variable_infos.update(_get_visible_variable_info("EV_250_Aggr1km_RefSB", 1000, AVAILABLE_QKM_PRODUCT_NAMES))
    variable_infos.update(_get_emissive_variable_info("EV_1KM_Emissive", 1000, AVAILABLE_1KM_IR_PRODUCT_NAMES))
    create_hdfeos_test_file(full_path, variable_infos, geo_resolution=5000, file_shortname="MOD021KM")
    return [full_path]


@pytest.fixture
def modis_l1b_imapp_1000m_file(tmpdir) -> list[str]:
    """Create a single MOD021KM file following IMAPP file scheme."""
    filename = generate_imapp_l1b_filename("1000m")
    full_path = os.path.join(str(tmpdir), filename)
    variable_infos = _get_l1b_geo_variable_info(filename, 5000, include_angles=True)
    variable_infos.update(_get_visible_variable_info("EV_1KM_RefSB", 1000, AVAILABLE_1KM_VIS_PRODUCT_NAMES))
    variable_infos.update(_get_visible_variable_info("EV_500_Aggr1km_RefSB", 1000, AVAILABLE_HKM_PRODUCT_NAMES))
    variable_infos.update(_get_visible_variable_info("EV_250_Aggr1km_RefSB", 1000, AVAILABLE_QKM_PRODUCT_NAMES))
    variable_infos.update(_get_emissive_variable_info("EV_1KM_Emissive", 1000, AVAILABLE_1KM_IR_PRODUCT_NAMES))
    create_hdfeos_test_file(full_path, variable_infos, geo_resolution=5000, file_shortname="MOD021KM")
    return [full_path]


@pytest.fixture
def modis_l1b_nasa_mod02hkm_file(tmpdir) -> list[str]:
    """Create a single MOD02HKM file following standard NASA file scheme."""
    filename = generate_nasa_l1b_filename("MOD02Hkm")
    full_path = os.path.join(str(tmpdir), filename)
    variable_infos = _get_l1b_geo_variable_info(filename, 1000, include_angles=False)
    variable_infos.update(_get_visible_variable_info("EV_500_RefSB", 250, AVAILABLE_QKM_PRODUCT_NAMES))
    create_hdfeos_test_file(full_path, variable_infos, geo_resolution=1000, file_shortname="MOD02HKM")
    return [full_path]


@pytest.fixture
def modis_l1b_nasa_mod02qkm_file(tmpdir) -> list[str]:
    """Create a single MOD02QKM file following standard NASA file scheme."""
    filename = generate_nasa_l1b_filename("MOD02Qkm")
    full_path = os.path.join(str(tmpdir), filename)
    variable_infos = _get_l1b_geo_variable_info(filename, 1000, include_angles=False)
    variable_infos.update(_get_visible_variable_info("EV_250_RefSB", 250, AVAILABLE_QKM_PRODUCT_NAMES))
    create_hdfeos_test_file(full_path, variable_infos, geo_resolution=1000, file_shortname="MOD02QKM")
    return [full_path]


@pytest.fixture
def modis_l1b_nasa_mod03_file(tmpdir) -> list[str]:
    """Create a single MOD03 file following standard NASA file scheme."""
    filename = generate_nasa_l1b_filename("MOD03")
    full_path = os.path.join(str(tmpdir), filename)
    variable_infos = _get_l1b_geo_variable_info(filename, 1000, include_angles=True)
    create_hdfeos_test_file(full_path, variable_infos, geo_resolution=1000, file_shortname="MOD03")
    return [full_path]


@pytest.fixture
def modis_l1b_nasa_1km_mod03_files(modis_l1b_nasa_mod021km_file, modis_l1b_nasa_mod03_file) -> list[str]:
    """Create input files including the 1KM and MOD03 files."""
    return modis_l1b_nasa_mod021km_file + modis_l1b_nasa_mod03_file


class TestModisL1b:
    """Test MODIS L1b reader."""

    @staticmethod
    def _check_shared_metadata(data_arr):
        assert data_arr.attrs["sensor"] == "modis"
        assert data_arr.attrs["platform_name"] == "EOS-Terra"
        assert "rows_per_scan" in data_arr.attrs
        assert isinstance(data_arr.attrs["rows_per_scan"], int)
        assert data_arr.attrs['reader'] == 'modis_l1b'

    def test_available_reader(self):
        """Test that MODIS L1b reader is available."""
        assert 'modis_l1b' in available_readers()

    @pytest.mark.parametrize(
        ('input_files', 'expected_names', 'expected_data_res', 'expected_geo_res'),
        [
            [pytest.lazy_fixture('modis_l1b_nasa_mod021km_file'),
             AVAILABLE_1KM_PRODUCT_NAMES + AVAILABLE_HKM_PRODUCT_NAMES + AVAILABLE_QKM_PRODUCT_NAMES,
             [1000], [5000, 1000]],
            [pytest.lazy_fixture('modis_l1b_imapp_1000m_file'),
             AVAILABLE_1KM_PRODUCT_NAMES + AVAILABLE_HKM_PRODUCT_NAMES + AVAILABLE_QKM_PRODUCT_NAMES,
             [1000], [5000, 1000]],
            [pytest.lazy_fixture('modis_l1b_nasa_mod02hkm_file'),
             AVAILABLE_HKM_PRODUCT_NAMES + AVAILABLE_QKM_PRODUCT_NAMES, [500], [1000, 500, 250]],
            [pytest.lazy_fixture('modis_l1b_nasa_mod02qkm_file'),
             AVAILABLE_QKM_PRODUCT_NAMES, [250], [1000, 500, 250]],
        ]
    )
    def test_scene_available_datasets(self, input_files, expected_names, expected_data_res, expected_geo_res):
        """Test that datasets are available."""
        scene = Scene(reader='modis_l1b', filenames=input_files)
        available_datasets = scene.available_dataset_names()
        assert len(available_datasets) > 0
        assert 'longitude' in available_datasets
        assert 'latitude' in available_datasets
        for chan_name in expected_names:
            assert chan_name in available_datasets

        available_data_ids = scene.available_dataset_ids()
        available_datas = {x: [] for x in expected_data_res}
        available_geos = {x: [] for x in expected_geo_res}
        # Make sure that every resolution from the reader is what we expect
        for data_id in available_data_ids:
            res = data_id['resolution']
            if data_id['name'] in ['longitude', 'latitude']:
                assert res in expected_geo_res
                available_geos[res].append(data_id)
            else:
                assert res in expected_data_res
                available_datas[res].append(data_id)

        # Make sure that every resolution we expect has at least one dataset
        for exp_res, avail_id in available_datas.items():
            assert avail_id, f"Missing datasets for data resolution {exp_res}"
        for exp_res, avail_id in available_geos.items():
            assert avail_id, f"Missing geo datasets for geo resolution {exp_res}"

    @pytest.mark.parametrize(
        ('input_files', 'has_5km', 'has_500', 'has_250', 'default_res'),
        [
            [pytest.lazy_fixture('modis_l1b_nasa_mod021km_file'),
             True, False, False, 1000],
            [pytest.lazy_fixture('modis_l1b_imapp_1000m_file'),
             True, False, False, 1000],
            [pytest.lazy_fixture('modis_l1b_nasa_mod02hkm_file'),
             False, True, True, 250],
            [pytest.lazy_fixture('modis_l1b_nasa_mod02qkm_file'),
             False, True, True, 250],
            [pytest.lazy_fixture('modis_l1b_nasa_1km_mod03_files'),
             True, True, True, 250],
        ]
    )
    def test_load_longitude_latitude(self, input_files, has_5km, has_500, has_250, default_res):
        """Test that longitude and latitude datasets are loaded correctly."""
        from satpy.tests.utils import make_dataid

        def test_func(dname, x, y):
            if dname == 'longitude':
                # assert less
                np.testing.assert_array_less(x, y)
            else:
                # assert greater
                np.testing.assert_array_less(y, x)

        scene = Scene(reader='modis_l1b', filenames=input_files)
        shape_5km = (SCAN_LEN_5KM, SCAN_WIDTH_5KM)
        shape_1km = (5 * SCAN_LEN_5KM, 5 * SCAN_WIDTH_5KM + 4)
        shape_500m = (shape_1km[0] * 2, shape_1km[1] * 2)
        shape_250m = (shape_1km[0] * 4, shape_1km[1] * 4)
        res_to_shape = {
            250: shape_250m,
            500: shape_500m,
            1000: shape_1km,
            5000: shape_5km,
        }
        default_shape = res_to_shape[default_res]
        for dataset_name in ['longitude', 'latitude']:
            # default resolution should be the maximum resolution from these datasets
            scene.load([dataset_name])
            longitude_def_id = make_dataid(name=dataset_name, resolution=default_res)
            longitude_def = scene[longitude_def_id]
            assert longitude_def.shape == default_shape
            test_func(dataset_name, longitude_def.values, 0)
            self._check_shared_metadata(longitude_def)

            # Specify original 5km scale
            scene.load([dataset_name], resolution=5000)
            longitude_5km_id = make_dataid(name=dataset_name, resolution=5000)
            if has_5km:
                longitude_5km = scene[longitude_5km_id]
                assert longitude_5km.shape == shape_5km
                test_func(dataset_name, longitude_5km.values, 0)
                self._check_shared_metadata(longitude_5km)
            else:
                pytest.raises(KeyError, scene.__getitem__, longitude_5km_id)

            # Specify higher resolution geolocation
            scene.load([dataset_name], resolution=500)
            longitude_500_id = make_dataid(name=dataset_name, resolution=500)
            if has_500:
                longitude_500 = scene[longitude_500_id]
                assert longitude_500.shape == shape_500m
                test_func(dataset_name, longitude_500.values, 0)
                self._check_shared_metadata(longitude_500)
            else:
                pytest.raises(KeyError, scene.__getitem__, longitude_500_id)

            scene.load([dataset_name], resolution=250)
            longitude_250_id = make_dataid(name=dataset_name, resolution=250)
            if has_250:
                longitude_250 = scene[longitude_250_id]
                assert longitude_250.shape == shape_250m
                test_func(dataset_name, longitude_250.values, 0)
                self._check_shared_metadata(longitude_250)
            else:
                pytest.raises(KeyError, scene.__getitem__, longitude_250_id)

    def test_load_sat_zenith_angle(self, modis_l1b_nasa_mod021km_file):
        """Test loading satellite zenith angle band."""
        scene = Scene(reader='modis_l1b', filenames=modis_l1b_nasa_mod021km_file)
        dataset_name = 'satellite_zenith_angle'
        scene.load([dataset_name])
        dataset = scene[dataset_name]
        assert dataset.shape == (5 * SCAN_LEN_5KM, 5 * SCAN_WIDTH_5KM + 4)
        assert dataset.attrs['resolution'] == 1000
        self._check_shared_metadata(dataset)

    def test_load_vis(self, modis_l1b_nasa_mod021km_file):
        """Test loading visible band."""
        scene = Scene(reader='modis_l1b', filenames=modis_l1b_nasa_mod021km_file)
        dataset_name = '1'
        scene.load([dataset_name])
        dataset = scene[dataset_name]
        assert dataset.shape == (5 * SCAN_LEN_5KM, 5 * SCAN_WIDTH_5KM + 4)
        self._check_shared_metadata(dataset)
