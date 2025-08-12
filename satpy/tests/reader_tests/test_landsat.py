#!/usr/bin/python
# Copyright (c) 2018 Satpy developers
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
"""Unittests for Landsat image readers."""

import os
import shutil
from datetime import datetime, timezone

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition
from pytest_lazy_fixtures.lazy_fixture import lf

from satpy import Scene
from satpy.readers.core.landsat import (
    ETMCHReader,
    ETML2CHReader,
    LandsatL1MDReader,
    LandsatL2MDReader,
    MSSCHReader,
    OLITIRSCHReader,
    OLITIRSL2CHReader,
    TMCHReader,
    TML2CHReader,
)

ETC_DIR = os.path.join(os.path.dirname(__file__), "landsat_metadata")

x_size = 100
y_size = 100

oli_tirs_l1_date = datetime(2024, 5, 12, tzinfo=timezone.utc)
oli_tirs_l2_date = datetime(2024, 6, 16, tzinfo=timezone.utc)
etm_l1_date = datetime(2023, 12, 8, tzinfo=timezone.utc)
etm_l2_date = datetime(2023, 8, 17, tzinfo=timezone.utc)
tm_l1_date = datetime(1989, 8, 18, tzinfo=timezone.utc)
tm_l2_date = datetime(2011, 8, 17, tzinfo=timezone.utc)
mss_l1_landsat1_date = datetime(1972, 7, 29, tzinfo=timezone.utc)
mss_l1_landsat4_date = datetime(1984, 4, 15, tzinfo=timezone.utc)

oli_tirs_l1_datetime = datetime(2024, 5, 2, 18, 0, 24, tzinfo=timezone.utc)
oli_tirs_l2_datetime = datetime(2024, 6, 16, 17, 10, 58, tzinfo=timezone.utc)
etm_l1_datetime = datetime(2023, 12, 8, 11, 44, 51, tzinfo=timezone.utc)
etm_l2_datetime = datetime(2023, 8, 17, 14, 54, 20, tzinfo=timezone.utc)
tm_l1_datetime = datetime(1989, 8, 18, 4, 26, 11, tzinfo=timezone.utc)
tm_l2_datetime = datetime(2011, 8, 17, 7, 10, 40, tzinfo=timezone.utc)
mss_l1_landsat1_datetime = datetime(1972, 7, 29, 16, 49, 31, tzinfo=timezone.utc)
mss_l1_landsat4_datetime = datetime(1984, 4, 15, 16, 38, 15, tzinfo=timezone.utc)

oli_tirs_l1_extent = (619485.0, 2440485.0, 850515.0, 2675715.0)
oli_tirs_l2_extent = (534885.0, 4665585.0, 765015.0, 4899315.0)
etm_l1_extent = (240885.0, -3301215.0, 483015.0, -3091485.0)
etm_l2_extent = (85785.0, 4680885.0, 336915.0, 4904115.0)
tm_l1_extent = (322185.0, 6085185.0, 567615.0, 6311415.0)
tm_l2_extent = (258285.0, 855285.0, 495315.0, 1064415.0)
mss_l1_landsat1_extent = (442110.0, 4667550.0, 673530.0, 4887990.0)
mss_l1_landsat4_extent = (540030.0, 4679970.0, 774450.0, 4888350.0)
antarctic_extent = (-138315.0, 590385.0, 61215.0, 808815.0)

oli_tirs_l1_pan_extent = (619492.5, 2440492.5, 850507.5, 2675707.5)
etm_l1_pan_extent = (240892.5, -3301207.5, 483007.5, -3091492.5)
antarctic_pan_extent = (-138307.5, 590392.5, 61207.5, 808807.5)

oli_tirs_l1_cal_dict = {"B1": (0.012357, -61.78647, 2e-05, -0.1),
                        "B5": (0.0060172, -30.08607, 2e-05, -0.1),
                        "B10": (0.0003342, 0.1, 774.8853, 1321.0789)}
oli_tirs_l2_cal_dict = {"B1": (2.75e-05, -0.2),
                        "B5": (2.75e-05, -0.2),
                        "B10": (0.00341802, 149.0)}
etm_l1_cal_dict = {"B1": (7.7874e-01, -6.97874, 1.1661e-03, -0.010450),
                   "B5": (1.2622e-01, -1.12622, 1.7365e-03, -0.015494),
                   "B6_VCID_2": (3.7205e-02, 3.16280, 666.09, 1282.71)}
etm_l2_cal_dict = {"B1": (2.75e-05, -0.2),
                   "B5": (2.75e-05, -0.2),
                   "B6": (0.00341802, 149.0)}
tm_l1_cal_dict = {"B1": (6.7921e-01, -2.19921, 1.1252e-03, -0.003643),
                  "B5": (1.2508e-01, -0.49508, 1.8160e-03, -0.007188),
                  "B6": (5.5375e-02, 1.18243, 671.62, 1284.30)}
tm_l2_cal_dict = {"B1": (2.75e-05, -0.2),
                  "B5": (2.75e-05, -0.2),
                  "B6": (0.00341802, 149.0)}
mss_l1_landsat1_cal_dict = {"B5": (6.4843e-01, -0.74843, 1.3660e-03, -0.001577),
                            "B6": (6.5236e-01, -0.75236, 1.6580e-03, -0.001912),
                            "B7": (6.0866e-01, -0.60866, 2.3287e-03, -0.002329)}
mss_l1_landsat4_cal_dict = {"B1": (8.7520e-01, 2.92480, 1.5680e-03, 0.005240),
                            "B2": (6.2008e-01, 3.07992, 1.2865e-03, 0.006390),
                            "B3": (5.4921e-01, 4.55079, 1.4070e-03, 0.011659)}

oli_tirs_l1_sp_params = [0.0098329, -49.16426, 2e-05, -0.1]
oli_tirs_l1_th_params = [0.0003342, 0.100000, 480.8883, 1201.1442]
oli_tirs_l2_sp_params = [2.75e-05, -0.2]
oli_tirs_l2_th_params = [0.00341802, 149.0]
etm_l1_sp_params = [9.6929e-01, -6.06929, 2.7591e-03, -0.017277]
etm_l1_th_params = [6.7087e-02, -0.06709, 666.09, 1282.71]
etm_l2_sp_params = [2.75e-05, -0.2]
etm_l2_th_params = [0.00341802, 149.0]
tm_l1_sp_params = [8.7602e-01, -2.38602, 2.7296e-03, -0.007435]
tm_l1_th_params = [5.5375e-02, 1.18243, 671.62, 1284.30]
tm_l2_sp_params = [2.75e-05, -0.2]
tm_l2_th_params = [0.00341802, 149.0]
mss_l1_landsat1_sp_params = [9.5591e-01, -18.55591, 1.7282e-03, -0.033547]
mss_l1_landsat4_sp_params = [4.7638e-01, 3.82362, 1.7954e-03, 0.014411]


@pytest.fixture(scope="module")
def spectral_data():
    """Get the data for the generic spectral channel."""
    return da.random.randint(12000, 16000,
                             size=(y_size, x_size),
                             chunks=(50, 50)).astype(np.uint16)


@pytest.fixture(scope="module")
def thermal_data():
    """Get the data for the generic thermal channel."""
    return da.random.randint(8000, 14000,
                             size=(y_size, x_size),
                             chunks=(50, 50)).astype(np.uint16)


@pytest.fixture(scope="module")
def sza_rad_data():
    """Get the data for the sza or radiance channel."""
    return da.random.randint(1, 10000,
                             size=(y_size, x_size),
                             chunks=(50, 50)).astype(np.uint16)


def create_tif_file(data, name, area, filename, date):
    """Create a tif file."""
    data_array = xr.DataArray(data,
                              dims=("y", "x"),
                              attrs={"name": name,
                                     "start_time": date})
    scn = Scene()
    scn["band_data"] = data_array
    scn["band_data"].attrs["area"] = area
    scn.save_dataset("band_data", writer="geotiff", enhance=False, fill_value=0,
                     filename=os.fspath(filename))


def get_area_fixture(name, zone, extent):
    """Get area def."""
    @pytest.fixture(scope="module")
    def _fixture():
        """Fixture body."""
        pcs_id = "WGS84 / UTM zone {zone}N"
        proj4_dict = {"proj": "utm", "zone": zone, "datum": "WGS84", "units": "m", "no_defs": None, "type": "crs"}
        area_extent = extent
        _fixture.__name__ = f"{name}_area"
        return AreaDefinition("geotiff_area", pcs_id, pcs_id, proj4_dict, x_size, y_size, area_extent)
    _fixture.__name__ = f"{name}_area"
    return _fixture


def make_temp_path_fixture(name):
    """Create a temp directory for a Landsat product."""
    @pytest.fixture(scope="module")
    def _fixture(tmp_path_factory):
        return tmp_path_factory.mktemp(f"{name}_files")
    _fixture.__name__ = f"{name}_files_path"
    return _fixture


def convert_to_fsfile(files):
    """Turn pathes to FSFile objects."""
    from fsspec.implementations.local import LocalFileSystem

    from satpy.readers.core.remote import FSFile

    fs = LocalFileSystem()
    files = (
        FSFile(os.path.abspath(file), fs=fs)
        for file in files
    )
    return files


oli_tirs_l1_area = get_area_fixture("oli_tirs_l1", 40, oli_tirs_l1_extent)
oli_tirs_l2_area = get_area_fixture("oli_tirs_l2", 14, oli_tirs_l2_extent)
etm_l1_area = get_area_fixture("etm_l1", 20, etm_l1_extent)
etm_l2_area = get_area_fixture("etm_l2", 15, etm_l2_extent)
tm_l1_area = get_area_fixture("tm_l1", 46, tm_l1_extent)
tm_l2_area = get_area_fixture("tm_l2", 38, tm_l2_extent)
mss_l1_landsat1_area = get_area_fixture("mss_l1_landsat1", 14, mss_l1_landsat1_extent)
mss_l1_landsat4_area = get_area_fixture("mss_l1_landsat4", 14, mss_l1_landsat4_extent)

oli_tirs_l1_files_path = make_temp_path_fixture("oli_tirs_l1")
oli_tirs_l2_files_path = make_temp_path_fixture("oli_tirs_l2")
etm_l1_files_path = make_temp_path_fixture("etm_l1")
etm_l2_files_path = make_temp_path_fixture("etm_l2")
tm_l1_files_path = make_temp_path_fixture("tm_l1")
tm_l2_files_path = make_temp_path_fixture("tm_l2")
mss_l1_landsat1_files_path = make_temp_path_fixture("mss_l1_landsat1")
mss_l1_landsat4_files_path = make_temp_path_fixture("mss_l1_landsat4")
antarctic_files_path = make_temp_path_fixture("antarctic")


@pytest.fixture(scope="module")
def oli_tirs_l1_b4_file(oli_tirs_l1_files_path, spectral_data, oli_tirs_l1_area):
    """Create the file for the Landsat OLI TIRS L1 B4 channel."""
    filename = oli_tirs_l1_files_path / "LC08_L1TP_026200_20240502_20240513_02_T2_B4.TIF"
    create_tif_file(spectral_data, "B4", oli_tirs_l1_area, filename, oli_tirs_l1_date)
    return os.fspath(filename)


@pytest.fixture(scope="module")
def oli_tirs_l1_b11_file(oli_tirs_l1_files_path, thermal_data, oli_tirs_l1_area):
    """Create the file for the Landsat OLI TIRS L1 B11 channel."""
    filename = oli_tirs_l1_files_path / "LC08_L1TP_026200_20240502_20240513_02_T2_B11.TIF"
    create_tif_file(thermal_data, "B11", oli_tirs_l1_area, filename, oli_tirs_l1_date)
    return os.fspath(filename)


@pytest.fixture(scope="module")
def oli_tirs_l1_sza_file(oli_tirs_l1_files_path, sza_rad_data, oli_tirs_l1_area):
    """Create the file for the Landsat OLI TIRS L1 sza."""
    filename = oli_tirs_l1_files_path / "LC08_L1TP_026200_20240502_20240513_02_T2_SZA.TIF"
    create_tif_file(sza_rad_data, "sza", oli_tirs_l1_area, filename, oli_tirs_l1_date)
    return os.fspath(filename)


@pytest.fixture(scope="module")
def oli_tirs_l1_mda_file(oli_tirs_l1_files_path):
    """Create the Landsat OLI TIRS L1 metadata xml file."""
    filename = oli_tirs_l1_files_path / "LC08_L1TP_026200_20240502_20240513_02_T2_MTL.xml"
    shutil.copyfile(os.path.join(ETC_DIR, "oli_tirs_l1_metadata.xml"), filename)
    return os.fspath(filename)


@pytest.fixture(scope="module")
def oli_tirs_l1_all_files(oli_tirs_l1_b4_file, oli_tirs_l1_b11_file, oli_tirs_l1_mda_file, oli_tirs_l1_sza_file):
    """Return all the files."""
    return oli_tirs_l1_b4_file, oli_tirs_l1_b11_file, oli_tirs_l1_mda_file, oli_tirs_l1_sza_file


@pytest.fixture(scope="module")
def oli_tirs_l2_b4_file(oli_tirs_l2_files_path, spectral_data, oli_tirs_l2_area):
    """Create the file for the Landsat OLI TIRS L2 B4 channel."""
    filename = oli_tirs_l2_files_path / "LC09_L2SP_029030_20240616_20240617_02_T1_SR_B4.TIF"
    create_tif_file(spectral_data, "B4", oli_tirs_l2_area, filename, oli_tirs_l2_date)
    return os.fspath(filename)

@pytest.fixture(scope="module")
def oli_tirs_l2_b10_file(oli_tirs_l2_files_path, thermal_data, oli_tirs_l2_area):
    """Create the file for the Landsat OLI TIRS L2 B10 channel."""
    filename = oli_tirs_l2_files_path / "LC09_L2SP_029030_20240616_20240617_02_T1_ST_B10.TIF"
    create_tif_file(thermal_data, "B10", oli_tirs_l2_area, filename, oli_tirs_l2_date)
    return os.fspath(filename)

@pytest.fixture(scope="module")
def oli_tirs_l2_rad_file(oli_tirs_l2_files_path, sza_rad_data, oli_tirs_l2_area):
    """Create the file for the Landsat OLI TIRS L2 TRAD channel."""
    filename = oli_tirs_l2_files_path / "LC09_L2SP_029030_20240616_20240617_02_T1_ST_TRAD.TIF"
    create_tif_file(sza_rad_data, "TRAD", oli_tirs_l2_area, filename, oli_tirs_l2_date)
    return os.fspath(filename)


@pytest.fixture(scope="module")
def oli_tirs_l2_mda_file(oli_tirs_l2_files_path):
    """Create the Landsat OLI TIRS L2 metadata xml file."""
    filename = oli_tirs_l2_files_path / "LC09_L2SP_029030_20240616_20240617_02_T1_MTL.xml"
    shutil.copyfile(os.path.join(ETC_DIR, "oli_tirs_l2_metadata.xml"), filename)
    return os.fspath(filename)


@pytest.fixture(scope="module")
def oli_tirs_l2_all_files(oli_tirs_l2_b4_file, oli_tirs_l2_b10_file, oli_tirs_l2_mda_file, oli_tirs_l2_rad_file):
    """Return all the files."""
    return oli_tirs_l2_b4_file, oli_tirs_l2_b10_file, oli_tirs_l2_mda_file, oli_tirs_l2_rad_file


@pytest.fixture(scope="module")
def etm_l1_b4_file(etm_l1_files_path, spectral_data, etm_l1_area):
    """Create the file for the Landsat ETM+ L1 B4 channel."""
    filename = etm_l1_files_path / "LE07_L1TP_230080_20231208_20240103_02_T1_B4.TIF"
    create_tif_file(spectral_data, "B4", etm_l1_area, filename, etm_l1_date)
    return os.fspath(filename)

@pytest.fixture(scope="module")
def etm_l1_b6_file(etm_l1_files_path, thermal_data, etm_l1_area):
    """Create the file for the Landsat ETM+ L1 B6 channel."""
    filename = etm_l1_files_path / "LE07_L1TP_230080_20231208_20240103_02_T1_B6_VCID_1.TIF"
    create_tif_file(thermal_data, "B6_VCID_1", etm_l1_area, filename, etm_l1_date)
    return os.fspath(filename)

@pytest.fixture(scope="module")
def etm_l1_sza_file(etm_l1_files_path, sza_rad_data, etm_l1_area):
    """Create the file for the Landsat ETM+ L1 sza."""
    filename = etm_l1_files_path / "LE07_L1TP_230080_20231208_20240103_02_T1_SZA.TIF"
    create_tif_file(sza_rad_data, "sza", etm_l1_area, filename, etm_l1_date)
    return os.fspath(filename)


@pytest.fixture(scope="module")
def etm_l1_mda_file(etm_l1_files_path):
    """Create the Landsat ETM+ L1 metadata xml file."""
    filename = etm_l1_files_path / "LE07_L1TP_230080_20231208_20240103_02_T1_MTL.xml"
    shutil.copyfile(os.path.join(ETC_DIR, "etm_l1_metadata.xml"), filename)
    return os.fspath(filename)


@pytest.fixture(scope="module")
def etm_l1_all_files(etm_l1_b4_file, etm_l1_b6_file, etm_l1_mda_file, etm_l1_sza_file):
    """Return all the files."""
    return etm_l1_b4_file, etm_l1_b6_file, etm_l1_mda_file, etm_l1_sza_file


@pytest.fixture(scope="module")
def etm_l2_b4_file(etm_l2_files_path, spectral_data, etm_l2_area):
    """Create the file for the Landsat ETM+ L2 B4 channel."""
    filename = etm_l2_files_path / "LE07_L2SP_028030_20230817_20230912_02_T1_SR_B4.TIF"
    create_tif_file(spectral_data, "B4", etm_l2_area, filename, etm_l2_date)
    return os.fspath(filename)

@pytest.fixture(scope="module")
def etm_l2_b6_file(etm_l2_files_path, thermal_data, etm_l2_area):
    """Create the file for the Landsat ETM+ L2 B6 channel."""
    filename = etm_l2_files_path / "LE07_L2SP_028030_20230817_20230912_02_T1_ST_B6.TIF"
    create_tif_file(thermal_data, "B6", etm_l2_area, filename, etm_l2_date)
    return os.fspath(filename)

@pytest.fixture(scope="module")
def etm_l2_rad_file(etm_l2_files_path, sza_rad_data, etm_l2_area):
    """Create the file for the Landsat ETM+ L2 TRAD channel."""
    filename = etm_l2_files_path / "LE07_L2SP_028030_20230817_20230912_02_T1_ST_TRAD.TIF"
    create_tif_file(sza_rad_data, "TRAD", etm_l2_area, filename, etm_l2_date)
    return os.fspath(filename)


@pytest.fixture(scope="module")
def etm_l2_mda_file(etm_l2_files_path):
    """Create the Landsat ETM+ L2 metadata xml file."""
    filename = etm_l2_files_path / "LE07_L2SP_028030_20230817_20230912_02_T1_MTL.xml"
    shutil.copyfile(os.path.join(ETC_DIR, "etm_l2_metadata.xml"), filename)
    return os.fspath(filename)


@pytest.fixture(scope="module")
def etm_l2_all_files(etm_l2_b4_file, etm_l2_b6_file, etm_l2_mda_file, etm_l2_rad_file):
    """Return all the files."""
    return etm_l2_b4_file, etm_l2_b6_file, etm_l2_mda_file, etm_l2_rad_file


@pytest.fixture(scope="module")
def tm_l1_b4_file(tm_l1_files_path, spectral_data, tm_l1_area):
    """Create the file for the Landsat TM L1 B4 channel."""
    filename = tm_l1_files_path / "LT04_L1TP_143021_19890818_20200916_02_T1_B4.TIF"
    create_tif_file(spectral_data, "B4", tm_l1_area, filename, tm_l1_date)
    return os.fspath(filename)

@pytest.fixture(scope="module")
def tm_l1_b6_file(tm_l1_files_path, thermal_data, tm_l1_area):
    """Create the file for the Landsat TM L1 B6 channel."""
    filename = tm_l1_files_path / "LT04_L1TP_143021_19890818_20200916_02_T1_B6.TIF"
    create_tif_file(thermal_data, "B6", tm_l1_area, filename, tm_l1_date)
    return os.fspath(filename)

@pytest.fixture(scope="module")
def tm_l1_sza_file(tm_l1_files_path, sza_rad_data, tm_l1_area):
    """Create the file for the Landsat TM L1 sza."""
    filename = tm_l1_files_path / "LT04_L1TP_143021_19890818_20200916_02_T1_SZA.TIF"
    create_tif_file(sza_rad_data, "sza", tm_l1_area, filename, tm_l1_date)
    return os.fspath(filename)


@pytest.fixture(scope="module")
def tm_l1_mda_file(tm_l1_files_path):
    """Create the Landsat TM L1 metadata xml file."""
    filename = tm_l1_files_path / "LT04_L1TP_143021_19890818_20200916_02_T1_MTL.xml"
    shutil.copyfile(os.path.join(ETC_DIR, "tm_l1_metadata.xml"), filename)
    return os.fspath(filename)


@pytest.fixture(scope="module")
def tm_l1_all_files(tm_l1_b4_file, tm_l1_b6_file, tm_l1_mda_file, tm_l1_sza_file):
    """Return all the files."""
    return tm_l1_b4_file, tm_l1_b6_file, tm_l1_mda_file, tm_l1_sza_file


@pytest.fixture(scope="module")
def tm_l2_b4_file(tm_l2_files_path, spectral_data, tm_l2_area):
    """Create the file for the Landsat TM L2 B4 channel."""
    filename = tm_l2_files_path / "LT05_L2SP_165054_20110817_20200820_02_T1_SR_B4.TIF"
    create_tif_file(spectral_data, "B4", tm_l2_area, filename, tm_l2_date)
    return os.fspath(filename)

@pytest.fixture(scope="module")
def tm_l2_b6_file(tm_l2_files_path, thermal_data, tm_l2_area):
    """Create the file for the Landsat TM L2 B6 channel."""
    filename = tm_l2_files_path / "LT05_L2SP_165054_20110817_20200820_02_T1_ST_B6.TIF"
    create_tif_file(thermal_data, "B6", tm_l2_area, filename, tm_l2_date)
    return os.fspath(filename)

@pytest.fixture(scope="module")
def tm_l2_rad_file(tm_l2_files_path, sza_rad_data, tm_l2_area):
    """Create the file for the Landsat TM L2 TRAD channel."""
    filename = tm_l2_files_path / "LT05_L2SP_165054_20110817_20200820_02_T1_ST_TRAD.TIF"
    create_tif_file(sza_rad_data, "TRAD", tm_l2_area, filename, tm_l2_date)
    return os.fspath(filename)


@pytest.fixture(scope="module")
def tm_l2_mda_file(tm_l2_files_path):
    """Create the Landsat TM L2 metadata xml file."""
    filename = tm_l2_files_path / "LT05_L2SP_165054_20110817_20200820_02_T1_MTL.xml"
    shutil.copyfile(os.path.join(ETC_DIR, "tm_l2_metadata.xml"), filename)
    return os.fspath(filename)


@pytest.fixture(scope="module")
def tm_l2_all_files(tm_l2_b4_file, tm_l2_b6_file, tm_l2_mda_file, tm_l2_rad_file):
    """Return all the files."""
    return tm_l2_b4_file, tm_l2_b6_file, tm_l2_mda_file, tm_l2_rad_file


@pytest.fixture(scope="module")
def mss_l1_landsat1_b4_file(mss_l1_landsat1_files_path, spectral_data, mss_l1_landsat1_area):
    """Create the file for the Landsat-1 MSS L1 B4 channel."""
    filename = mss_l1_landsat1_files_path / "LM01_L1TP_032030_19720729_20200909_02_T2_B4.TIF"
    create_tif_file(spectral_data, "B4", mss_l1_landsat1_area, filename, mss_l1_landsat1_date)
    return os.fspath(filename)


@pytest.fixture(scope="module")
def mss_l1_landsat1_mda_file(mss_l1_landsat1_files_path):
    """Create the Landsat-1 MSS L1 metadata xml file."""
    filename = mss_l1_landsat1_files_path / "LM01_L1TP_032030_19720729_20200909_02_T2_MTL.xml"
    shutil.copyfile(os.path.join(ETC_DIR, "mss_l1_landsat1_metadata.xml"), filename)
    return os.fspath(filename)


@pytest.fixture(scope="module")
def mss_l1_landsat1_all_files(mss_l1_landsat1_b4_file, mss_l1_landsat1_mda_file):
    """Return all the files."""
    return mss_l1_landsat1_b4_file, mss_l1_landsat1_mda_file


@pytest.fixture(scope="module")
def mss_l1_landsat4_b4_file(mss_l1_landsat4_files_path, spectral_data, mss_l1_landsat4_area):
    """Create the file for the Landsat-4 MSS L1 B4 channel."""
    filename = mss_l1_landsat4_files_path / "LM04_L1TP_029030_19840415_20200903_02_T2_B4.TIF"
    create_tif_file(spectral_data, "B4", mss_l1_landsat4_area, filename, mss_l1_landsat4_date)
    return os.fspath(filename)


@pytest.fixture(scope="module")
def mss_l1_landsat4_mda_file(mss_l1_landsat4_files_path):
    """Create the Landsat-4 MSS L1 metadata xml file."""
    filename = mss_l1_landsat4_files_path / "LM04_L1TP_029030_19840415_20200903_02_T2_MTL.xml"
    shutil.copyfile(os.path.join(ETC_DIR, "mss_l1_landsat4_metadata.xml"), filename)
    return os.fspath(filename)


@pytest.fixture(scope="module")
def mss_l1_landsat4_all_files(mss_l1_landsat4_b4_file, mss_l1_landsat4_mda_file):
    """Return all the files."""
    return mss_l1_landsat4_b4_file, mss_l1_landsat4_mda_file


@pytest.fixture(scope="module")
def antarctic_mda_file(antarctic_files_path):
    """Create the Antarctic Landsat OLI-TIRS L1 metadata xml file."""
    filename = antarctic_files_path / "LC08_L1GT_132122_20220301_20220308_02_T2_MTL.xml"
    shutil.copyfile(os.path.join(ETC_DIR, "antarctic_metadata.xml"), filename)
    return os.fspath(filename)


def get_filename_info(date, level_correction, spacecraft, data_type):
    """Set up a filename info dict."""
    return dict(observation_date=date,
                platform_type="L",
                process_level_correction=level_correction,
                spacecraft_id=spacecraft,
                data_type=data_type,
                collection_id="02")


class TestLandsat:
    """Test Landsat image readers."""

    ftype_info = {"file_type": "granule_B4"}

    @pytest.mark.parametrize(
        ("reader", "spectral_name", "thermal_name", "all_files", "area"),
        [
            pytest.param(
                "oli_tirs_l1_tif", "B4", "B11", lf("oli_tirs_l1_all_files"), lf("oli_tirs_l1_area"), id="oli_tirs_l1",
            ),
            pytest.param(
                "oli_tirs_l2_tif", "B4", "B10", lf("oli_tirs_l2_all_files"), lf("oli_tirs_l2_area"), id="oli_tirs_l2",
            ),
            pytest.param("etm_l1_tif", "B4", "B6_VCID_1", lf("etm_l1_all_files"), lf("etm_l1_area"), id="etm_l1"),
            pytest.param("etm_l2_tif", "B4", "B6", lf("etm_l2_all_files"), lf("etm_l2_area"), id="etm_l2"),
            pytest.param("tm_l1_tif", "B4", "B6", lf("tm_l1_all_files"), lf("tm_l1_area"), id="tm_l1"),
            pytest.param("tm_l2_tif", "B4", "B6", lf("tm_l2_all_files"), lf("tm_l2_area"), id="tm_l2"),
            pytest.param(
                "mss_l1_tif", "B4", None, lf("mss_l1_landsat1_all_files"), lf("mss_l1_landsat1_area"),
                id="mss_l1_landsat1",
            ),
            pytest.param(
                "mss_l1_tif", "B4", None, lf("mss_l1_landsat4_all_files"), lf("mss_l1_landsat4_area"),
                id="mss_l1_landsat4",
            ),
        ],
    )
    @pytest.mark.parametrize("remote", [True, False])
    def test_basicload(self, reader, area, all_files, spectral_name, thermal_name, remote):
        """Test loading a Landsat Scene."""
        if remote:
            all_files = convert_to_fsfile(all_files)

        scn = Scene(reader=reader, filenames=all_files)
        if thermal_name is not None:
            scn.load([spectral_name, thermal_name])
        else:
            scn.load([spectral_name])

        self._check_basicload_thermal(scn, spectral_name, thermal_name, area, reader)
        self._check_basicload_mss_l1_tif(reader, scn)


    @staticmethod
    def _check_basicload_thermal(scn, spectral_name, thermal_name, area, reader):
        # Check dataset is loaded correctly
        assert scn[spectral_name].shape == (100, 100)
        assert scn[spectral_name].attrs["area"] == area
        assert scn[spectral_name].attrs["saturated"]
        if thermal_name is not None:
            assert scn[thermal_name].shape == (100, 100)
            assert scn[thermal_name].attrs["area"] == area
            if reader in ["oli_tirs_l1_tif", "oli_tirs_l2_tif", "etm_l2_tif"]:
                # OLI TIRS and ETM+ L2 do not have saturation flag on thermal band
                with pytest.raises(KeyError, match="saturated"):
                    assert not scn[thermal_name].attrs["saturated"]
            else:
                assert not scn[thermal_name].attrs["saturated"]


    @staticmethod
    def _check_basicload_mss_l1_tif(reader, scn):
        # Check if MSS sets wavelength correctly
        if reader == "mss_l1_tif":
            if scn["B4"].attrs["platform_name"] == "Landsat-4":
                assert scn["B4"].attrs["_satpy_id"]["wavelength"].min == 0.8
                assert scn["B4"].attrs["_satpy_id"]["wavelength"].central == 0.95
                assert scn["B4"].attrs["_satpy_id"]["wavelength"].max == 1.1
            elif scn["B4"].attrs["platform_name"] == "Landsat-1":
                assert scn["B4"].attrs["_satpy_id"]["wavelength"].min == 0.5
                assert scn["B4"].attrs["_satpy_id"]["wavelength"].central == 0.55
                assert scn["B4"].attrs["_satpy_id"]["wavelength"].max == 0.6


    @pytest.mark.parametrize(
        ("reader", "date_time", "spectral_file", "spectral_name", "mda_file"),
        [
            pytest.param(
                "oli_tirs_l1_tif", oli_tirs_l1_datetime, lf("oli_tirs_l1_b4_file"), "B4", lf("oli_tirs_l1_mda_file"),
                id="oli_tirs_l1",
            ),
            pytest.param(
                "oli_tirs_l2_tif", oli_tirs_l2_datetime, lf("oli_tirs_l2_b4_file"), "B4", lf("oli_tirs_l2_mda_file"),
                id="oli_tirs_l2",
            ),
            pytest.param("etm_l1_tif", etm_l1_datetime, lf("etm_l1_b4_file"), "B4", lf("etm_l1_mda_file"), id="etm_l1"),
            pytest.param("etm_l2_tif", etm_l2_datetime, lf("etm_l2_b4_file"), "B4", lf("etm_l2_mda_file"), id="etm_l2"),
            pytest.param("tm_l1_tif", tm_l1_datetime, lf("tm_l1_b4_file"), "B4", lf("tm_l1_mda_file"), id="tm_l1"),
            pytest.param("tm_l2_tif", tm_l2_datetime, lf("tm_l2_b4_file"), "B4", lf("tm_l2_mda_file"), id="tm_l2"),
            pytest.param(
                "mss_l1_tif", mss_l1_landsat1_datetime, lf("mss_l1_landsat1_b4_file"), "B4",
                lf("mss_l1_landsat1_mda_file"), id="mss_l1_landsat1",
            ),
            pytest.param(
                "mss_l1_tif", mss_l1_landsat4_datetime, lf("mss_l1_landsat4_b4_file"), "B4",
                lf("mss_l1_landsat4_mda_file"), id="mss_l1_landsat4",
            ),
        ],
    )
    def test_ch_startend(self, reader, spectral_file, mda_file, spectral_name, date_time):
        """Test correct retrieval of start/end times."""
        scn = Scene(reader=reader, filenames=[spectral_file, mda_file])
        bnds = scn.available_dataset_names()
        assert bnds == [spectral_name]

        scn.load(["B4"])
        assert scn.start_time == date_time
        assert scn.end_time == date_time

    @pytest.mark.parametrize(
        (
            "spectral_file", "mda_file", "CH_reader_class", "MD_reader_class",
            "filename_info",
        ),
        [
            pytest.param(
                lf("oli_tirs_l1_b4_file"), lf("oli_tirs_l1_mda_file"), OLITIRSCHReader, LandsatL1MDReader,
                get_filename_info(oli_tirs_l1_date, "L1TP", "08", "C"), id="oli_tirs_l1",
            ),
            pytest.param(
                lf("oli_tirs_l2_b4_file"), lf("oli_tirs_l2_mda_file"), OLITIRSL2CHReader, LandsatL2MDReader,
                get_filename_info(oli_tirs_l2_date, "L2SP", "09", "C"), id="oli_tirs_l2",
            ),
            pytest.param(
               lf( "etm_l1_b4_file"), lf("etm_l1_mda_file"), ETMCHReader, LandsatL1MDReader,
                get_filename_info(etm_l1_date, "L1TP", "07", "E"), id="etm_l1",
            ),
            pytest.param(
                lf("etm_l2_b4_file"), lf("etm_l2_mda_file"), ETML2CHReader, LandsatL2MDReader,
                get_filename_info(etm_l2_date, "L2SP", "07", "E"), id="etm_l2",
            ),
            pytest.param(
                lf("tm_l1_b4_file"), lf("tm_l1_mda_file"), TMCHReader, LandsatL1MDReader,
                get_filename_info(tm_l1_date, "L1TP", "04", "T"), id="tm_l1",
            ),
            pytest.param(
                lf("tm_l2_b4_file"), lf("tm_l2_mda_file"), TML2CHReader, LandsatL2MDReader,
                get_filename_info(tm_l2_date, "L2SP", "05", "T"), id="tm_l2",
            ),
            pytest.param(
                lf("mss_l1_landsat1_b4_file"), lf("mss_l1_landsat1_mda_file"), MSSCHReader, LandsatL1MDReader,
                get_filename_info(mss_l1_landsat1_date, "L1TP", "01", "M"), id="mss_l1_landsat1",
            ),
            pytest.param(
                lf("mss_l1_landsat4_b4_file"), lf("mss_l1_landsat4_mda_file"), MSSCHReader, LandsatL1MDReader,
                get_filename_info(mss_l1_landsat4_date, "L1TP", "04", "M"), id="mss_l1_landsat4",
            ),
        ],
    )
    def test_loading_gd(
        self,
        CH_reader_class,
        MD_reader_class,
        mda_file,
        spectral_file,
        filename_info,
    ):
        """Test loading a Landsat Scene with good channel requests."""
        good_mda = MD_reader_class(mda_file, filename_info, {})
        rdr = CH_reader_class(spectral_file, filename_info, self.ftype_info, good_mda)

        # Check case with good file data and load request
        rdr.get_dataset({"name": "B4", "calibration": "counts"}, {"standard_name": "test_data", "units": "test_units"})

    @pytest.mark.parametrize(
        (
            "spectral_file", "mda_file", "CH_reader_class", "MD_reader_class",
            "filename_info",
        ),
        [
            pytest.param(
                lf("oli_tirs_l1_b4_file"), lf("oli_tirs_l1_mda_file"), OLITIRSCHReader, LandsatL1MDReader,
                get_filename_info(oli_tirs_l1_date, "L1TP", "08", "C"), id="oli_tirs_l1",
            ),
            pytest.param(
                lf("oli_tirs_l2_b4_file"), lf("oli_tirs_l2_mda_file"), OLITIRSL2CHReader, LandsatL2MDReader,
                get_filename_info(oli_tirs_l2_date, "L2SP", "09", "C"), id="oli_tirs_l2",
            ),
            pytest.param(
                lf("etm_l1_b4_file"), lf("etm_l1_mda_file"), ETMCHReader, LandsatL1MDReader,
                get_filename_info(etm_l1_date, "L1TP", "07", "E"), id="etm_l1",
            ),
            pytest.param(
                lf("etm_l2_b4_file"), lf("etm_l2_mda_file"), ETML2CHReader, LandsatL2MDReader,
                get_filename_info(etm_l2_date, "L2SP", "07", "E"), id="etm_l2",
            ),
            pytest.param(
                lf("tm_l1_b4_file"), lf("tm_l1_mda_file"), TMCHReader, LandsatL1MDReader,
                get_filename_info(tm_l1_date, "L1TP", "04", "T"), id="tm_l1",
            ),
            pytest.param(
                lf("tm_l2_b4_file"), lf("tm_l2_mda_file"), TML2CHReader, LandsatL2MDReader,
                get_filename_info(tm_l2_date, "L2SP", "05", "T"), id="tm_l2",
            ),
            pytest.param(
                lf("mss_l1_landsat1_b4_file"), lf("mss_l1_landsat1_mda_file"), MSSCHReader, LandsatL1MDReader,
                get_filename_info(mss_l1_landsat1_date, "L1TP", "01", "M"), id="mss_l1_landsat1",
            ),
            pytest.param(
                lf("mss_l1_landsat4_b4_file"), lf("mss_l1_landsat4_mda_file"), MSSCHReader, LandsatL1MDReader,
                get_filename_info(mss_l1_landsat4_date, "L1TP", "04", "M"), id="mss_l1_landsat4",
            ),
        ],
    )
    def test_loading_badfil(
        self,
        CH_reader_class,
        MD_reader_class,
        mda_file,
        spectral_file,
        filename_info,
    ):
        """Test loading a Landsat Scene with bad channel requests."""
        good_mda = MD_reader_class(mda_file, filename_info, {})
        rdr = CH_reader_class(spectral_file, filename_info, self.ftype_info, good_mda)

        ftype = {"standard_name": "test_data", "units": "test_units"}
        # Check case with request to load channel not matching filename
        with pytest.raises(ValueError, match="Requested channel B5 does not match the reader channel B4"):
            rdr.get_dataset({"name": "B5", "calibration": "counts"}, ftype)

    @pytest.mark.parametrize(
        (
            "thermal_file", "mda_file", "CH_reader_class", "MD_reader_class",
            "filename_info",
        ),
        [
            pytest.param(
                lf("oli_tirs_l1_b11_file"), lf("oli_tirs_l1_mda_file"), OLITIRSCHReader, LandsatL1MDReader,
                get_filename_info(oli_tirs_l1_date, "L1TP", "08", "C"), id="oli_tirs_l1",
            ),
            pytest.param(
                lf("oli_tirs_l2_b10_file"), lf("oli_tirs_l2_mda_file"), OLITIRSL2CHReader, LandsatL2MDReader,
                get_filename_info(oli_tirs_l2_date, "L2SP", "09", "C"), id="oli_tirs_l2",
            ),
        ],
    )
    def test_loading_badchan(
        self,
        CH_reader_class,
        MD_reader_class,
        mda_file,
        thermal_file,
        filename_info,
    ):
        """Test loading a Landsat Scene with bad channel requests."""
        good_mda = MD_reader_class(mda_file, filename_info, {})
        ftype = {"standard_name": "test_data", "units": "test_units"}
        bad_finfo = filename_info.copy()
        bad_finfo["data_type"] = "T"

        # Check loading invalid channel for data type
        rdr = CH_reader_class(thermal_file, bad_finfo, self.ftype_info, good_mda)
        with pytest.raises(ValueError, match="Requested channel B4 is not available in this granule"):
            rdr.get_dataset({"name": "B4", "calibration": "counts"}, ftype)

        if filename_info["process_level_correction"] == "L1TP":
            # L1 test
            bad_finfo["data_type"] = "O"
            ftype_b11 = self.ftype_info.copy()
            ftype_b11["file_type"] = "granule_B11"
            rdr = CH_reader_class(thermal_file, bad_finfo, ftype_b11, good_mda)
            with pytest.raises(ValueError, match="Requested channel B11 is not available in this granule"):
                rdr.get_dataset({"name": "B11", "calibration": "counts"}, ftype)
        else:
            # L2 test
            bad_finfo["data_type"] = "O"
            ftype_b10 = self.ftype_info.copy()
            ftype_b10["file_type"] = "granule_B10"
            rdr = OLITIRSL2CHReader(thermal_file, bad_finfo, ftype_b10, good_mda)
            with pytest.raises(ValueError, match="Requested channel B10 is not available in this granule"):
                rdr.get_dataset({"name": "B10", "calibration": "counts"}, ftype)

    @pytest.mark.parametrize(
        (
            "spectral_file", "mda_file", "CH_reader_class", "MD_reader_class",
            "filename_info",
        ),
        [
            pytest.param(
                lf("oli_tirs_l1_b4_file"),  lf("oli_tirs_l1_mda_file"), OLITIRSCHReader, LandsatL1MDReader,
                get_filename_info(oli_tirs_l1_date, "L1TP", "08", "C"), id="oli_tirs_l1",
            ),
            pytest.param(
                lf("oli_tirs_l2_b4_file"),  lf("oli_tirs_l2_mda_file"), OLITIRSL2CHReader, LandsatL2MDReader,
                get_filename_info(oli_tirs_l2_date, "L2SP", "09", "C"), id="oli_tirs_l2",
            ),
            pytest.param(
                lf("etm_l1_b4_file"),  lf("etm_l1_mda_file"), ETMCHReader, LandsatL1MDReader,
                get_filename_info(etm_l1_date, "L1TP", "07", "E"), id="etm_l1",
            ),
            pytest.param(
                lf("etm_l2_b4_file"),  lf("etm_l2_mda_file"), ETML2CHReader, LandsatL2MDReader,
                get_filename_info(etm_l2_date, "L2SP", "07", "E"), id="etm_l2",
            ),
            pytest.param(
                lf("tm_l1_b4_file"),  lf("tm_l1_mda_file"), TMCHReader, LandsatL1MDReader,
                get_filename_info(tm_l1_date, "L1TP", "04", "T"), id="tm_l1",
            ),
            pytest.param(
                lf("tm_l2_b4_file"),  lf("tm_l2_mda_file"), TML2CHReader, LandsatL2MDReader,
                get_filename_info(tm_l2_date, "L2SP", "05", "T"), id="tm_l2",
            ),
            pytest.param(
                lf("mss_l1_landsat1_b4_file"),  lf("mss_l1_landsat1_mda_file"), MSSCHReader, LandsatL1MDReader,
                get_filename_info(mss_l1_landsat1_date, "L1TP", "01", "M"), id="mss_l1_landsat1",
            ),
            pytest.param(
                lf("mss_l1_landsat4_b4_file"),  lf("mss_l1_landsat4_mda_file"), MSSCHReader, LandsatL1MDReader,
                get_filename_info(mss_l1_landsat4_date, "L1TP", "04", "M"), id="mss_l1_landsat4",
            ),
        ],
    )
    def test_badfiles(
        self,
        CH_reader_class,
        MD_reader_class,
        mda_file,
        spectral_file,
        filename_info,
    ):
        """Test loading a Landsat Scene with bad data."""
        bad_fname_info = filename_info.copy()
        bad_fname_info["platform_type"] = "B"

        bad_ftype_info = self.ftype_info.copy()
        bad_ftype_info["file_type"] = "granule-b05"

        ftype = {"standard_name": "test_data", "units": "test_units"}

        # Test that metadata reader initialises with correct filename
        good_mda = MD_reader_class(mda_file, filename_info, ftype)

        # Check metadata reader fails if platform type is wrong
        with pytest.raises(ValueError, match="This reader only supports Landsat data"):
            MD_reader_class(mda_file, bad_fname_info, ftype)

        # Test that metadata reader initialises with correct filename
        CH_reader_class(spectral_file, filename_info, self.ftype_info, good_mda)

        # Check metadata reader fails if platform type is wrong
        with pytest.raises(ValueError, match="This reader only supports Landsat data"):
            CH_reader_class(spectral_file, bad_fname_info, self.ftype_info, good_mda)

        with pytest.raises(ValueError, match="Invalid file type: granule-b05"):
            CH_reader_class(spectral_file, filename_info, bad_ftype_info, good_mda)

    @pytest.mark.parametrize(
        ("reader", "spectral_name", "thermal_name", "all_files"),
        [
            pytest.param("oli_tirs_l1_tif", "B4", "B11", lf("oli_tirs_l1_all_files"), id="oli_tirs_l1"),
            pytest.param("oli_tirs_l2_tif", "B4", "B10", lf("oli_tirs_l2_all_files"), id="oli_tirs_l2"),
            pytest.param("etm_l1_tif", "B4", "B6_VCID_1", lf("etm_l1_all_files"), id="etm_l1"),
            pytest.param("etm_l2_tif", "B4", "B6", lf("etm_l2_all_files"), id="etm_l2"),
            pytest.param("tm_l1_tif", "B4", "B6", lf("tm_l1_all_files"), id="tm_l1"),
            pytest.param("tm_l2_tif", "B4", "B6", lf("tm_l2_all_files"), id="tm_l2"),
            pytest.param("mss_l1_tif", "B4", None, lf("mss_l1_landsat1_all_files"), id="mss_l1_landsat1"),
            pytest.param("mss_l1_tif", "B4", None, lf("mss_l1_landsat4_all_files"), id="mss_l1_landsat4"),
        ],
    )
    def test_calibration_counts(
        self,
        reader,
        all_files,
        spectral_name,
        thermal_name,
        spectral_data,
        thermal_data,
    ):
        """Test counts calibration mode for the reader."""
        scn = Scene(reader=reader, filenames=all_files)
        if thermal_name is not None:
            scn.load([spectral_name, thermal_name], calibration="counts")
        else:
            scn.load([spectral_name], calibration="counts")

        np.testing.assert_allclose(scn[spectral_name].values, spectral_data)
        assert scn[spectral_name].attrs["units"] == "1"
        assert scn[spectral_name].attrs["standard_name"] == "counts"

        if thermal_name is not None:
            np.testing.assert_allclose(scn[thermal_name].values, thermal_data)
            assert scn[thermal_name].attrs["units"] == "1"
            assert scn[thermal_name].attrs["standard_name"] == "counts"

    @pytest.mark.parametrize(
        ("reader", "spectral_name", "thermal_name", "all_files", "cal_spectral_params", "cal_thermal_params"),
        [
            pytest.param(
                "oli_tirs_l1_tif", "B4", "B11", lf("oli_tirs_l1_all_files"),
                oli_tirs_l1_sp_params, oli_tirs_l1_th_params, id="oli_tirs_l1",
            ),
            pytest.param(
                "etm_l1_tif", "B4", "B6_VCID_1", lf("etm_l1_all_files"), etm_l1_sp_params, etm_l1_th_params,
                id="etm_l1",
            ),
            pytest.param("tm_l1_tif", "B4", "B6", lf("tm_l1_all_files"), tm_l1_sp_params, tm_l1_th_params,
                         id="tm_l1",
                         ),
            pytest.param(
                "mss_l1_tif", "B4", None, lf("mss_l1_landsat1_all_files"), mss_l1_landsat1_sp_params, None,
                id="mss_l1_landsat1",
            ),
            pytest.param(
                "mss_l1_tif", "B4", None, lf("mss_l1_landsat4_all_files"), mss_l1_landsat4_sp_params, None,
                id="mss_l1_landsat4",
            ),
        ],
    )
    def test_calibration_radiance(
        self,
        reader,
        all_files,
        spectral_name,
        thermal_name,
        cal_spectral_params,
        cal_thermal_params,
        spectral_data,
        thermal_data,
    ):
        """Test radiance calibration mode for the reader."""
        exp_spectral = (spectral_data * cal_spectral_params[0] + cal_spectral_params[1]).astype(np.float32)
        if thermal_name is not None:
            exp_thermal = (thermal_data * cal_thermal_params[0] + cal_thermal_params[1]).astype(np.float32)

        scn = Scene(reader=reader, filenames=all_files)
        if thermal_name is not None:
            scn.load([spectral_name, thermal_name], calibration="radiance")
        else:
            scn.load([spectral_name], calibration="radiance")

        assert scn[spectral_name].attrs["units"] == "W m-2 um-1 sr-1"
        assert scn[spectral_name].attrs["standard_name"] == "toa_outgoing_radiance_per_unit_wavelength"
        np.testing.assert_allclose(scn[spectral_name].values, exp_spectral, rtol=1e-4)

        if thermal_name is not None:
            assert scn[thermal_name].attrs["units"] == "W m-2 um-1 sr-1"
            assert scn[thermal_name].attrs["standard_name"] == "toa_outgoing_radiance_per_unit_wavelength"
            np.testing.assert_allclose(scn[thermal_name].values, exp_thermal, rtol=1e-4)

    @pytest.mark.parametrize(
        (
            "reader", "spectral_name", "thermal_name", "sza_rad_name", "all_files",
            "cal_spectral_params", "cal_thermal_params",
        ),
        [
            pytest.param(
                "oli_tirs_l1_tif", "B4", "B11", None, lf("oli_tirs_l1_all_files"),
                oli_tirs_l1_sp_params, oli_tirs_l1_th_params, id="oli_tirs_l1",
            ),
            pytest.param(
                "oli_tirs_l2_tif", "B4", "B10", "TRAD", lf("oli_tirs_l2_all_files"),
                oli_tirs_l2_sp_params, oli_tirs_l2_th_params, id="oli_tirs_l2",
            ),
            pytest.param(
                "etm_l1_tif", "B4", "B6_VCID_1", None, lf("etm_l1_all_files"),
                etm_l1_sp_params, etm_l1_th_params, id="etm_l1",
            ),
            pytest.param(
                "etm_l2_tif", "B4", "B6", "TRAD", lf("etm_l2_all_files"), etm_l2_sp_params, etm_l2_th_params,
                id="etm_l2",
            ),
            pytest.param(
                "tm_l1_tif", "B4", "B6", None, lf("tm_l1_all_files"), tm_l1_sp_params, tm_l1_th_params,
                id="tm_l1",
            ),
            pytest.param(
                "tm_l2_tif", "B4", "B6", "TRAD", lf("tm_l2_all_files"), tm_l2_sp_params, tm_l2_th_params,
                id="tm_l2",
            ),
            pytest.param(
                "mss_l1_tif", "B4", None, None, lf("mss_l1_landsat1_all_files"),
                mss_l1_landsat1_sp_params, None, id="mss_l1_landsat1",
            ),
            pytest.param(
                "mss_l1_tif", "B4", None, None, lf("mss_l1_landsat4_all_files"),
                mss_l1_landsat4_sp_params, None, id="mss_l1_landsat4",
            ),
        ],
    )
    def test_calibration_highlevel(
        self,
        reader,
        all_files,
        spectral_name,
        thermal_name,
        sza_rad_name,
        cal_spectral_params,
        cal_thermal_params,
        spectral_data,
        thermal_data,
        sza_rad_data,
    ):
        """Test high level calibration modes for the reader."""
        exp_spectral = self._get_expected_highlevel_spectral(reader, spectral_data, cal_spectral_params)
        exp_thermal, exp_rad = self._get_expected_highlevel_thermal(
            reader, thermal_data, cal_thermal_params, sza_rad_data, thermal_name)

        scn = self._get_scn_highlevel(reader, all_files, spectral_name, thermal_name, sza_rad_name)

        assert scn[spectral_name].attrs["units"] == "%"
        if "_l2_" in reader:
            assert scn[spectral_name].attrs["standard_name"] == "surface_bidirectional_reflectance"
        else:
            assert scn[spectral_name].attrs["standard_name"] == "toa_bidirectional_reflectance"
        np.testing.assert_allclose(np.array(scn[spectral_name].values), np.array(exp_spectral), rtol=1e-4)

        if thermal_name is not None:
            self._check_thermal_highlevel(scn, thermal_name, exp_thermal)
            self._check_thermal_l2_highlevel(reader, scn, sza_rad_name, exp_rad)

    @staticmethod
    def _get_expected_highlevel_spectral(reader, spectral_data, cal_spectral_params):
        if "_l2_" in reader:
            return (spectral_data * cal_spectral_params[0] + cal_spectral_params[1]).astype(np.float32) * 100
        return (spectral_data * cal_spectral_params[2] + cal_spectral_params[3]).astype(np.float32) * 100

    @staticmethod
    def _get_expected_highlevel_thermal(reader, thermal_data, cal_thermal_params, sza_rad_data, thermal_name):
        exp_thermal, exp_rad = None, None
        if "_l2_" in reader and thermal_name is not None:
            exp_thermal = (thermal_data * cal_thermal_params[0] + cal_thermal_params[1]).astype(np.float32)
            exp_rad = (sza_rad_data * 0.001).astype(np.float32)
        elif thermal_name is not None:
            exp_thermal = (thermal_data * cal_thermal_params[0] + cal_thermal_params[1])
            exp_thermal = (cal_thermal_params[3] / np.log((cal_thermal_params[2] / exp_thermal) + 1))
            exp_thermal = exp_thermal.astype(np.float32)
        return exp_thermal, exp_rad

    @staticmethod
    def _get_scn_highlevel(reader, all_files, spectral_name, thermal_name, sza_rad_name):
        scn = Scene(reader=reader, filenames=all_files)
        if thermal_name is not None:
            if "_l2_" in reader:
                scn.load([spectral_name, thermal_name, sza_rad_name])
            else:
                scn.load([spectral_name, thermal_name])
        else:
            scn.load([spectral_name])
        return scn

    @staticmethod
    def _check_thermal_highlevel(scn, thermal_name, exp_thermal):
        assert scn[thermal_name].attrs["units"] == "K"
        assert scn[thermal_name].attrs["standard_name"] == "brightness_temperature"
        np.testing.assert_allclose(scn[thermal_name].values, exp_thermal, rtol=1e-6)

    @staticmethod
    def _check_thermal_l2_highlevel(reader, scn, sza_rad_name, exp_rad):
        if "_l2_" in reader:
            assert scn[sza_rad_name].attrs["units"] == "W m-2 um-1 sr-1"
            assert scn[sza_rad_name].attrs["standard_name"] == "toa_outgoing_radiance_per_unit_wavelength"
            np.testing.assert_allclose(scn[sza_rad_name].values, exp_rad, rtol=1e-6)

    @pytest.mark.parametrize(
        ("reader", "all_files"),
        [
            pytest.param("oli_tirs_l1_tif", lf("oli_tirs_l1_all_files"), id="oli_tirs_l1"),
            pytest.param("etm_l1_tif", lf("etm_l1_all_files"), id="etm_l1",),
            pytest.param("tm_l1_tif", lf("tm_l1_all_files"), id="tm_l1"),
        ],
    )
    def test_angles(self, reader, all_files, sza_rad_data):
        """Test calibration modes for the reader."""
        # Check angles are calculated correctly
        scn = Scene(reader=reader, filenames=all_files)
        scn.load(["solar_zenith_angle"])
        assert scn["solar_zenith_angle"].attrs["units"] == "degrees"
        assert scn["solar_zenith_angle"].attrs["standard_name"] == "solar_zenith_angle"
        np.testing.assert_allclose(scn["solar_zenith_angle"].values * 100,
                                   np.array(sza_rad_data),
                                   atol=0.01,
                                   rtol=1e-3)

    @pytest.mark.parametrize(
        (
            "reader", "MD_reader_class", "mda_file", "cal_dict",
            "filename_info", "platform_name", "earth_sun_distance",
        ),
        [
            pytest.param(
                "oli_tirs_l1_tif", LandsatL1MDReader,  lf("oli_tirs_l1_mda_file"), oli_tirs_l1_cal_dict,
                get_filename_info(oli_tirs_l1_date, "L1TP", "08", "C"),
                "Landsat-8", 1.0079981, id="oli_tirs_l1",
            ),
            pytest.param(
                "oli_tirs_l2_tif", LandsatL2MDReader,  lf("oli_tirs_l2_mda_file"), oli_tirs_l2_cal_dict,
                get_filename_info(oli_tirs_l2_date, "L2SP", "09", "C"),
                "Landsat-9", 1.0158933, id="oli_tirs_l2",
            ),
            pytest.param(
                "etm_l1_tif", LandsatL1MDReader,  lf("etm_l1_mda_file"), etm_l1_cal_dict,
                get_filename_info(etm_l1_date, "L1TP", "07", "E"),
                "Landsat-7", 0.9850987, id="etm_l1",
            ),
            pytest.param(
                "etm_l2_tif", LandsatL2MDReader,  lf("etm_l2_mda_file"), etm_l2_cal_dict,
                get_filename_info(etm_l2_date, "L2SP", "07", "E"),
                "Landsat-7", 1.0124651, id="etm_l2",
            ),
            pytest.param(
                "tm_l1_tif", LandsatL1MDReader,  lf("tm_l1_mda_file"), tm_l1_cal_dict,
                get_filename_info(tm_l1_date, "L1TP", "04", "T"),
                "Landsat-4", 1.0122057, id="tm_l1",
            ),
            pytest.param(
                "tm_l2_tif", LandsatL2MDReader,  lf("tm_l2_mda_file"), tm_l2_cal_dict,
                get_filename_info(tm_l2_date, "L2SP", "05", "T"),
                "Landsat-5", 1.0125021, id="tm_l2",
            ),
            pytest.param(
                "mss_l1_tif", LandsatL1MDReader,  lf("mss_l1_landsat1_mda_file"), mss_l1_landsat1_cal_dict,
                get_filename_info(mss_l1_landsat1_date, "L1TP", "01", "M"),
                "Landsat-1", 1.0152109, id="mss_l1_landsat1",
            ),
            pytest.param(
                "mss_l1_tif", LandsatL1MDReader,  lf("mss_l1_landsat4_mda_file"), mss_l1_landsat4_cal_dict,
                get_filename_info(mss_l1_landsat4_date, "L1TP", "04", "M"),
                "Landsat-4", 1.0035512, id="mss_l1_landsat4",
            ),
        ],
    )
    def test_metadata(
        self,
        reader,
        MD_reader_class,
        mda_file,
        cal_dict,
        platform_name,
        earth_sun_distance,
        filename_info,
    ):
        """Check that metadata values loaded correctly."""
        mda = MD_reader_class(mda_file, filename_info, {})

        assert mda.platform_name == platform_name
        assert mda.earth_sun_distance() == earth_sun_distance

        cal_bands = list(cal_dict.keys())
        assert mda.band_calibration[cal_bands[0]] == cal_dict[cal_bands[0]]
        assert mda.band_calibration[cal_bands[1]] == cal_dict[cal_bands[1]]
        assert mda.band_calibration[cal_bands[2]] == cal_dict[cal_bands[2]]
        assert not mda.band_saturation[cal_bands[0]]
        assert mda.band_saturation["B4"]
        assert not mda.band_saturation[cal_bands[1]]
        if reader in ["oli_tirs_l1_tif", "oli_tirs_l2_tif", "etm_l2_tif"]:
            with pytest.raises(KeyError):
                mda.band_saturation[cal_bands[2]]
        else:
            assert not mda.band_saturation[cal_bands[2]]

    @pytest.mark.parametrize(
        (
            "MD_reader_class", "mda_file", "extent", "pan_extent",
            "filename_info",
        ),
        [
            pytest.param(
                LandsatL1MDReader,  lf("oli_tirs_l1_mda_file"), oli_tirs_l1_extent, oli_tirs_l1_pan_extent,
                get_filename_info(oli_tirs_l1_date, "L1TP", "08", "C"), id="oli_tirs_l1",
            ),
            pytest.param(
                LandsatL2MDReader,  lf("oli_tirs_l2_mda_file"), oli_tirs_l2_extent, None,
                get_filename_info(oli_tirs_l2_date, "L2SP", "09", "C"), id="oli_tirs_l2",
            ),
            pytest.param(
                LandsatL1MDReader,  lf("etm_l1_mda_file"), etm_l1_extent, etm_l1_pan_extent,
                get_filename_info(etm_l1_date, "L1TP", "07", "E"), id="etm_l1",
            ),
            pytest.param(
                LandsatL2MDReader,  lf("etm_l2_mda_file"), etm_l2_extent, None,
                get_filename_info(etm_l2_date, "L2SP", "07", "E"), id="etm_l2",
            ),
            pytest.param(
                LandsatL1MDReader,  lf("tm_l1_mda_file"), tm_l1_extent, None,
                get_filename_info(tm_l1_date, "L1TP", "04", "T"), id="tm_l1",
            ),
            pytest.param(
                LandsatL2MDReader,  lf("tm_l2_mda_file"), tm_l2_extent, None,
                get_filename_info(tm_l2_date, "L2SP", "05", "T"), id="tm_l2",
            ),
            pytest.param(
                LandsatL1MDReader,  lf("mss_l1_landsat1_mda_file"), mss_l1_landsat1_extent, None,
                get_filename_info(mss_l1_landsat1_date, "L1TP", "01", "M"), id="mss_l1_landsat1",
            ),
            pytest.param(
                LandsatL1MDReader,  lf("mss_l1_landsat4_mda_file"), mss_l1_landsat4_extent, None,
                get_filename_info(mss_l1_landsat4_date, "L1TP", "04", "M"), id="mss_l1_landsat4",
            ),
            pytest.param(
                LandsatL1MDReader, lf("antarctic_mda_file"), antarctic_extent, antarctic_pan_extent,
                get_filename_info(oli_tirs_l1_date, "L1TP", "08", "C"), id="antarctic",
            ),
        ],
    )
    def test_area_def(
        self,
        MD_reader_class,
        mda_file,
        extent,
        pan_extent,
        filename_info,
    ):
        """Check we can get the area defs properly."""
        mda = MD_reader_class(mda_file, filename_info, {})
        standard_area = mda.build_area_def("B4")
        assert standard_area.area_extent == extent

        if pan_extent is not None:
            pan_area = mda.build_area_def("B8")
            assert pan_area.area_extent == pan_extent
