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

import numpy as np
import pytest

from pyhdf.SD import SD, SDC

from satpy import available_readers, Scene

# Mock MODIS HDF4 file
AVAILABLE_1KM_PRODUCT_NAMES = list(range(1, 13)) + ['13lo', '13hi', '14lo', '14hi'] + list(range(15, 37))
AVAILABLE_1KM_PRODUCT_NAMES = [str(x) for x in AVAILABLE_1KM_PRODUCT_NAMES]
AVAILABLE_HKM_PRODUCT_NAMES = [str(x) for x in list(range(3, 8))]
AVAILABLE_QKM_PRODUCT_NAMES = ['1', '2']

SCAN_WIDTH = 406
SCAN_LEN = 270
SCALE_FACTOR = 1
TEST_LAT = np.repeat(np.linspace(35., 45., SCAN_WIDTH)[:, None], SCAN_LEN, 1)
TEST_LAT *= np.linspace(0.9, 1.1, SCAN_LEN)
TEST_LON = np.repeat(np.linspace(-45., -35., SCAN_LEN)[None, :], SCAN_WIDTH, 0)
TEST_LON *= np.linspace(0.9, 1.1, SCAN_WIDTH)[:, None]
TEST_SATZ = (np.repeat(abs(np.linspace(-65.2, 65.4, SCAN_LEN))[None, :], SCAN_WIDTH, 0) * 100).astype(np.int16)
TEST_DATA = {
    'Latitude': {'data': TEST_LAT.astype(np.float32),
                 'type': SDC.FLOAT32,
                 'fill_value': -999,
                 'attrs': {'dim_labels': ['Cell_Along_Swath_5km:mod35', 'Cell_Across_Swath_5km:mod35']}},
    'Longitude': {'data': TEST_LON.astype(np.float32),
                  'type': SDC.FLOAT32,
                  'fill_value': -999,
                  'attrs': {'dim_labels': ['Cell_Along_Swath_5km:mod35', 'Cell_Across_Swath_5km:mod35']}},
    'EV_1KM_RefSB': {
        'data': np.zeros((15, 5*SCAN_WIDTH, 5*SCAN_LEN+4), dtype=np.uint16),
        'type': SDC.UINT16,
        'fill_value': 0,
        'attrs': {
            'dim_labels': ['Band_1KM_RefSB:MODIS_SWATH_Type_L1B',
                           '10*nscans:MODIS_SWATH_Type_L1B',
                           'Max_EV_frames:MODIS_SWATH_Type_L1B'],
            'valid_range': (0, 32767),
            'reflectance_scales': (1,) * 15,
            'reflectance_offsets': (0,) * 15,
            'band_names': '8,9,10,11,12,13lo,13hi,14lo,14hi,15,16,17,18,19,26',
        },
    },
    'EV_1KM_RefSB_Uncert_Indexes': {
        'data': np.zeros((15, 5*SCAN_WIDTH, 5*SCAN_LEN+4), dtype=np.uint8),
        'type': SDC.UINT8,
        'fill_value': 255,
        'attrs': {
            'dim_labels': ['Band_1KM_RefSB:MODIS_SWATH_Type_L1B',
                           '10*nscans:MODIS_SWATH_Type_L1B',
                           'Max_EV_frames:MODIS_SWATH_Type_L1B'],
        },
    },
    'EV_500_Aggr1km_RefSB': {
        'data': np.zeros((5, 5*SCAN_WIDTH, 5*SCAN_LEN+4), dtype=np.uint16),
        'type': SDC.UINT16,
        'fill_value': 0,
        'attrs': {
            'dim_labels': ['Band_500M:MODIS_SWATH_Type_L1B',
                           '10*nscans:MODIS_SWATH_Type_L1B',
                           'Max_EV_frames:MODIS_SWATH_Type_L1B'],
            'valid_range': (0, 32767),
            'reflectance_scales': (1,) * 5,
            'reflectance_offsets': (0,) * 5,
            'band_names': '3,4,5,6,7',
        },
    },
    'EV_500_Aggr1km_RefSB_Uncert_Indexes': {
        'data': np.zeros((5, 5*SCAN_WIDTH, 5*SCAN_LEN+4), dtype=np.uint8),
        'type': SDC.UINT8,
        'fill_value': 255,
        'attrs': {
            'dim_labels': ['Band_500M:MODIS_SWATH_Type_L1B',
                           '10*nscans:MODIS_SWATH_Type_L1B',
                           'Max_EV_frames:MODIS_SWATH_Type_L1B'],
        },
    },
    'EV_250_Aggr1km_RefSB': {
        'data': np.zeros((2, 5*SCAN_WIDTH, 5*SCAN_LEN+4), dtype=np.uint16),
        'type': SDC.UINT16,
        'fill_value': 0,
        'attrs': {
            'dim_labels': ['Band_250M:MODIS_SWATH_Type_L1B',
                           '10*nscans:MODIS_SWATH_Type_L1B',
                           'Max_EV_frames:MODIS_SWATH_Type_L1B'],
            'valid_range': (0, 32767),
            'reflectance_scales': (1,) * 2,
            'reflectance_offsets': (0,) * 2,
            'band_names': '1,2',
        },
    },
    'EV_250_Aggr1km_RefSB_Uncert_Indexes': {
        'data': np.zeros((2, 5*SCAN_WIDTH, 5*SCAN_LEN+4), dtype=np.uint8),
        'type': SDC.UINT8,
        'fill_value': 255,
        'attrs': {
            'dim_labels': ['Band_250M:MODIS_SWATH_Type_L1B',
                           '10*nscans:MODIS_SWATH_Type_L1B',
                           'Max_EV_frames:MODIS_SWATH_Type_L1B'],
        },
    },
    'EV_1KM_Emmissive': {
        'data': np.zeros((16, 5*SCAN_WIDTH, 5*SCAN_LEN+4), dtype=np.uint16),
        'type': SDC.UINT16,
        'fill_value': 0,
        'attrs': {
            'dim_labels': ['Band_1KM_Emissive:MODIS_SWATH_Type_L1B',
                           '10*nscans:MODIS_SWATH_Type_L1B',
                           'Max_EV_frames:MODIS_SWATH_Type_L1B'],
            'valid_range': (0, 32767),
            'band_names': '20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36',
        },
    },
    'EV_1KM_Emissive_Uncert_Indexes': {
        'data': np.zeros((16, 5*SCAN_WIDTH, 5*SCAN_LEN+4), dtype=np.uint8),
        'type': SDC.UINT8,
        'fill_value': 255,
        'attrs': {
            'dim_labels': ['Band_1KM_Emissive:MODIS_SWATH_Type_L1B',
                           '10*nscans:MODIS_SWATH_Type_L1B',
                           'Max_EV_frames:MODIS_SWATH_Type_L1B'],
        },
    },
    'SensorZenith': {'data': TEST_SATZ,
                     'type': SDC.INT16,
                     'fill_value': -32767,
                     'attrs': {'dim_labels': ['2*nscans:MODIS_SWATH_Type_L1B', '1KM_geo_dim:MODIS_SWATH_Type_L1B'],
                               'scale_factor': 0.01}},
    'SensorAzimuth': {'data': TEST_SATZ,
                      'type': SDC.INT16,
                      'fill_value': -32767,
                      'attrs': {'dim_labels': ['2*nscans:MODIS_SWATH_Type_L1B', '1KM_geo_dim:MODIS_SWATH_Type_L1B'],
                                'scale_factor': 0.01}},
}


def generate_nasa_l1b_filename(prefix):
    """Generate a filename that follows NASA MODIS L1b convention."""
    now = datetime.now()
    return f'{prefix}_A{now:%y%j_%H%M%S}_{now:%Y%j%H%M%S}.hdf'


def generate_imapp_l1b_filename(suffix):
    """Generate a filename that follows IMAPP MODIS L1b convention."""
    now = datetime.now()
    return f'a1.{now:%y%j.%H%M}.{suffix}.hdf'


def create_test_data(filename, include_metadata=True):
    """Create a fake MODIS L1b HDF4 file with headers."""
    h = SD(filename, SDC.WRITE | SDC.CREATE)

    if include_metadata:
        setattr(h, 'CoreMetadata.0', _create_core_metadata())  # noqa
        setattr(h, 'StructMetadata.0', _create_struct_metadata())  # noqa
        setattr(h, 'ArchiveMetadata.0', _create_header_metadata())  # noqa

    for var_name, var_info in TEST_DATA.items():
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


def _create_core_metadata() -> str:
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
                          "VALUE = \"MOD021KM\"\nEND_OBJECT = SHORTNAME\n\n" \
                          "OBJECT = VERSIONID\nNUM_VAL = 1\nVALUE = 6\nEND_OBJECT = VERSIONID\n\n" \
                          "END_GROUP = COLLECTIONDESCRIPTIONCLASS\n\n"
    core_metadata_header += "\n\n" + inst_metadata + collection_metadata
    return core_metadata_header


def _create_struct_metadata() -> str:
    struct_metadata_header = "GROUP=SwathStructure\n" \
                             "GROUP=SWATH_1\n" \
                             "GROUP=DimensionMap\n" \
                             "OBJECT=DimensionMap_2\n" \
                             "GeoDimension=\"2*nscans\"\n" \
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
    create_test_data(full_path)
    return [full_path]


@pytest.fixture
def modis_l1b_imapp_1000m_file(tmpdir) -> list[str]:
    """Create a single MOD021KM file following IMAPP file scheme."""
    filename = generate_imapp_l1b_filename("1000m")
    full_path = os.path.join(str(tmpdir), filename)
    create_test_data(full_path)
    return [full_path]


@pytest.fixture
def modis_l1b_nasa_mod02hkm_file(tmpdir) -> list[str]:
    """Create a single MOD02HKM file following standard NASA file scheme."""
    filename = generate_nasa_l1b_filename("MOD02Hkm")
    full_path = os.path.join(str(tmpdir), filename)
    create_test_data(full_path)
    return [full_path]


@pytest.fixture
def modis_l1b_nasa_mod02qkm_file(tmpdir) -> list[str]:
    """Create a single MOD02QKM file following standard NASA file scheme."""
    filename = generate_nasa_l1b_filename("MOD02Qkm")
    full_path = os.path.join(str(tmpdir), filename)
    create_test_data(full_path)
    return [full_path]


@pytest.fixture
def modis_l1b_nasa_mod03_file(tmpdir) -> list[str]:
    """Create a single MOD03 file following standard NASA file scheme."""
    filename = generate_nasa_l1b_filename("MOD03")
    full_path = os.path.join(str(tmpdir), filename)
    create_test_data(full_path)
    return [full_path]


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
             AVAILABLE_1KM_PRODUCT_NAMES, [1000], [5000, 1000]],
            [pytest.lazy_fixture('modis_l1b_imapp_1000m_file'),
             AVAILABLE_1KM_PRODUCT_NAMES, [1000], [5000, 1000]],
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
        for data_id in available_data_ids:
            res = data_id['resolution']
            if data_id['name'] in ['longitude', 'latitude']:
                assert res in expected_geo_res
                available_geos[res].append(data_id)
            else:
                assert res in expected_data_res
                available_datas[res].append(data_id)

        for exp_res, avail_id in available_datas.items():
            assert avail_id, f"Missing datasets for data resolution {exp_res}"
        for exp_res, avail_id in available_geos.items():
            assert avail_id, f"Missing geo datasets for geo resolution {exp_res}"

    def test_load_longitude_latitude(self, modis_l1b_nasa_mod021km_file):
        """Test that longitude and latitude datasets are loaded correctly."""
        from satpy.tests.utils import make_dataid

        def test_func(dname, x, y):
            if dname == 'longitude':
                # assert less
                np.testing.assert_array_less(x, y)
            else:
                # assert greater
                np.testing.assert_array_less(y, x)

        scene = Scene(reader='modis_l1b', filenames=modis_l1b_nasa_mod021km_file)
        for dataset_name in ['longitude', 'latitude']:
            # Default resolution should be the interpolated 1km
            scene.load([dataset_name])
            longitude_1km_id = make_dataid(name=dataset_name, resolution=1000)
            longitude_1km = scene[longitude_1km_id]
            assert longitude_1km.shape == (5*SCAN_WIDTH, 5*SCAN_LEN+4)
            test_func(dataset_name, longitude_1km.values, 0)
            self._check_shared_metadata(longitude_1km)

            # Specify original 5km scale
            scene.load([dataset_name], resolution=5000)
            longitude_5km_id = make_dataid(name=dataset_name, resolution=5000)
            longitude_5km = scene[longitude_5km_id]
            assert longitude_5km.shape == TEST_DATA[dataset_name.capitalize()]['data'].shape
            test_func(dataset_name, longitude_5km.values, 0)
            self._check_shared_metadata(longitude_5km)

    def test_load_sat_zenith_angle(self, modis_l1b_nasa_mod021km_file):
        """Test loading satellite zenith angle band."""
        scene = Scene(reader='modis_l1b', filenames=modis_l1b_nasa_mod021km_file)
        dataset_name = 'satellite_zenith_angle'
        scene.load([dataset_name])
        dataset = scene[dataset_name]
        assert dataset.shape == (5*SCAN_WIDTH, 5*SCAN_LEN+4)
        assert dataset.attrs['resolution'] == 1000
        self._check_shared_metadata(dataset)

    def test_load_vis(self, modis_l1b_nasa_mod021km_file):
        """Test loading visible band."""
        scene = Scene(reader='modis_l1b', filenames=modis_l1b_nasa_mod021km_file)
        dataset_name = '1'
        scene.load([dataset_name])
        dataset = scene[dataset_name]
        assert dataset.shape == (5*SCAN_WIDTH, 5*SCAN_LEN+4)
        self._check_shared_metadata(dataset)
