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


def get_filename_info(date, level_correction, spacecraft, data_type):
    """Set up a filename info dict."""
    return dict(observation_date=date,
                platform_type="L",
                process_level_correction=level_correction,
                spacecraft_id=spacecraft,
                data_type=data_type,
                collection_id="02")


class BaseLandsatTest:
    """Basic class for landsat tests."""

    files_path: str
    spectral_filename: str
    thermal_filename: str
    mda_filename: str
    mda_etc: str

    date: datetime
    filename_info: dict
    ftype_info = {"file_type": "granule_B4"}

    spectral_name: str
    thermal_name: str | None
    sza_rad_name: str | None

    reader: str
    CH_reader_class: object
    MD_reader_class: object

    zone: int
    date_time: datetime
    calibration_spectral_params: list
    calibration_thermal_params: list | None
    calibration_dict: dict
    extent: tuple
    pan_extent: tuple | None
    platform_name: str
    earth_sun_distance: float

    basic_band_saturated = True
    thermal_band_saturated = False
    bad_spectral_name = "B5"

    @property
    def area(self):
        """Get area def."""
        pcs_id = f"WGS84 / UTM zone {self.zone}N"
        proj4_dict = {"proj": "utm", "zone": self.zone, "datum": "WGS84", "units": "m", "no_defs": None, "type": "crs"}
        return AreaDefinition("geotiff_area", pcs_id, pcs_id, proj4_dict, x_size, y_size, self.extent)

    @pytest.fixture
    def path(self, tmp_path_factory):
        """Create a temporary directory."""
        return tmp_path_factory.mktemp(self.files_path)

    @pytest.fixture
    def spectral_file(self, path, spectral_data):
        """Create the file for the Landsat B4 channel."""
        filename = path / self.spectral_filename
        create_tif_file(spectral_data, self.spectral_name, self.area, filename, self.date)
        return os.fspath(filename)

    @pytest.fixture
    def thermal_file(self, path, thermal_data):
        """Create the file for the Landsat thermal channel."""
        filename = path / self.thermal_filename
        create_tif_file(thermal_data, self.thermal_name, self.area, filename, self.date)
        return os.fspath(filename)

    @pytest.fixture
    def mda_file(self, path):
        """Create the Landsat metadata xml file."""
        filename = path / self.mda_filename
        shutil.copyfile(os.path.join(ETC_DIR, self.mda_etc), filename)
        return os.fspath(filename)

    @pytest.mark.parametrize("remote", [True, False])
    def test_basicload(self, all_files, remote):
        """Test loading a Landsat Scene."""
        if remote:
            all_files = self.convert_to_fsfile(all_files)

        scn = Scene(reader=self.reader, filenames=all_files)
        if self.thermal_name is not None:
            scn.load([self.spectral_name, self.thermal_name])
        else:
            scn.load([self.spectral_name])

        self._check_basicload_basic_band(scn)
        if self.thermal_name is not None:
            self._check_basicload_thermal_band(scn)

    @staticmethod
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

    def _check_basicload_basic_band(self, scn):
        """Check if the basic band loaded correctly."""
        # Check dataset is loaded correctly
        assert scn[self.spectral_name].shape == (100, 100)
        assert scn[self.spectral_name].attrs["area"] == self.area
        assert scn[self.spectral_name].attrs["saturated"] == self.basic_band_saturated

    def _check_basicload_thermal_band(self, scn):
        """Check if thermal band loaded correctly."""
        assert scn[self.thermal_name].shape == (100, 100)
        assert scn[self.thermal_name].attrs["area"] == self.area
        if self.reader in ["oli_tirs_l1_tif", "oli_tirs_l2_tif", "etm_l2_tif"]:
            # OLI TIRS and ETM+ L2 do not have saturation flag on thermal band
            with pytest.raises(KeyError, match="saturated"):
                assert scn[self.thermal_name].attrs["saturated"] == self.thermal_band_saturated
        else:
            assert scn[self.thermal_name].attrs["saturated"] == self.thermal_band_saturated

    def test_ch_startend(self, spectral_file, mda_file):
        """Test correct retrieval of start/end times."""
        scn = Scene(reader=self.reader, filenames=[spectral_file, mda_file])
        bnds = scn.available_dataset_names()
        assert bnds == [self.spectral_name]

        scn.load([self.spectral_name])
        assert scn.start_time == self.date_time
        assert scn.end_time == self.date_time

    def test_loading_gd(self, spectral_file, mda_file):
        """Test loading a Landsat Scene with good channel requests."""
        good_mda = self.MD_reader_class(mda_file, self.filename_info, {})
        rdr = self.CH_reader_class(spectral_file, self.filename_info, self.ftype_info, good_mda)

        # Check case with good file data and load request
        rdr.get_dataset(
            {"name": self.spectral_name, "calibration": "counts"},
            {"standard_name": "test_data", "units": "test_units"},
        )

    def test_loading_badfil(self, spectral_file, mda_file):
        """Test loading a Landsat Scene with bad channel requests."""
        good_mda = self.MD_reader_class(mda_file, self.filename_info, {})
        rdr = self.CH_reader_class(spectral_file, self.filename_info, self.ftype_info, good_mda)

        ftype = {"standard_name": "test_data", "units": "test_units"}
        # Check case with request to load channel not matching filename
        with pytest.raises(
            ValueError,
            match=f"Requested channel {self.bad_spectral_name} does not match the reader channel {self.spectral_name}",
        ):
            rdr.get_dataset({"name": self.bad_spectral_name, "calibration": "counts"}, ftype)

    def test_badfiles(self, spectral_file, mda_file):
        """Test loading a Landsat Scene with bad data."""
        bad_fname_info = self.filename_info.copy()
        bad_fname_info["platform_type"] = "B"

        bad_ftype_info = self.ftype_info.copy()
        bad_ftype_info["file_type"] = "granule-b05"

        ftype = {"standard_name": "test_data", "units": "test_units"}

        # Test that metadata reader initialises with correct filename
        good_mda = self.MD_reader_class(mda_file, self.filename_info, ftype)

        # Check metadata reader fails if platform type is wrong
        with pytest.raises(ValueError, match="This reader only supports Landsat data"):
            self.MD_reader_class(mda_file, bad_fname_info, ftype)

        # Test that metadata reader initialises with correct filename
        self.CH_reader_class(spectral_file, self.filename_info, self.ftype_info, good_mda)

        # Check metadata reader fails if platform type is wrong
        with pytest.raises(ValueError, match="This reader only supports Landsat data"):
            self.CH_reader_class(spectral_file, bad_fname_info, self.ftype_info, good_mda)

        with pytest.raises(ValueError, match="Invalid file type: granule-b05"):
            self.CH_reader_class(spectral_file, self.filename_info, bad_ftype_info, good_mda)

    def test_calibration_counts(self, all_files, spectral_data, thermal_data):
        """Test counts calibration mode for the reader."""
        scn = Scene(reader=self.reader, filenames=all_files)
        if self.thermal_name is not None:
            scn.load([self.spectral_name, self.thermal_name], calibration="counts")
        else:
            scn.load([self.spectral_name], calibration="counts")

        np.testing.assert_allclose(scn[self.spectral_name].values, spectral_data)
        assert scn[self.spectral_name].attrs["units"] == "1"
        assert scn[self.spectral_name].attrs["standard_name"] == "counts"

        if self.thermal_name is not None:
            np.testing.assert_allclose(scn[self.thermal_name].values, thermal_data)
            assert scn[self.thermal_name].attrs["units"] == "1"
            assert scn[self.thermal_name].attrs["standard_name"] == "counts"

    def test_calibration_highlevel(self, all_files, spectral_data, thermal_data, sza_rad_data):
        """Test high level calibration modes for the reader."""
        exp_spectral = self._get_expected_highlevel_spectral(spectral_data)
        exp_thermal, exp_rad = self._get_expected_highlevel_thermal(thermal_data, sza_rad_data)

        scn = self._get_scn_highlevel(all_files)

        assert scn[self.spectral_name].attrs["units"] == "%"
        if "_l2_" in self.reader:
            assert scn[self.spectral_name].attrs["standard_name"] == "surface_bidirectional_reflectance"
        else:
            assert scn[self.spectral_name].attrs["standard_name"] == "toa_bidirectional_reflectance"
        np.testing.assert_allclose(np.array(scn[self.spectral_name].values), np.array(exp_spectral), rtol=1e-4)

        if self.thermal_name is not None:
            self._check_thermal_highlevel(scn, exp_thermal)
            self._check_thermal_l2_highlevel(scn, exp_rad)

    def _get_expected_highlevel_spectral(self, spectral_data):
        if "_l2_" in self.reader:
            return (
                spectral_data * self.calibration_spectral_params[0] + self.calibration_spectral_params[1]
            ).astype(np.float32) * 100
        return (
            spectral_data * self.calibration_spectral_params[2] + self.calibration_spectral_params[3]
        ).astype(np.float32) * 100

    def _get_expected_highlevel_thermal(self, thermal_data, sza_rad_data):
        exp_thermal, exp_rad = None, None
        if "_l2_" in self.reader and self.thermal_name is not None:
            exp_thermal = (
                thermal_data * self.calibration_thermal_params[0] + self.calibration_thermal_params[1]
            ).astype(np.float32)
            exp_rad = (sza_rad_data * 0.001).astype(np.float32)
        elif self.thermal_name is not None:
            exp_thermal = (
                thermal_data * self.calibration_thermal_params[0] + self.calibration_thermal_params[1]
            )
            exp_thermal = (
                self.calibration_thermal_params[3] / np.log((self.calibration_thermal_params[2] / exp_thermal) + 1)
            )
            exp_thermal = exp_thermal.astype(np.float32)
        return exp_thermal, exp_rad

    def _get_scn_highlevel(self, all_files):
        scn = Scene(reader=self.reader, filenames=all_files)
        if self.thermal_name is not None:
            if "_l2_" in self.reader:
                scn.load([self.spectral_name, self.thermal_name, self.sza_rad_name])
            else:
                scn.load([self.spectral_name, self.thermal_name])
        else:
            scn.load([self.spectral_name])
        return scn

    def _check_thermal_highlevel(self, scn, exp_thermal):
        assert scn[self.thermal_name].attrs["units"] == "K"
        assert scn[self.thermal_name].attrs["standard_name"] == "brightness_temperature"
        np.testing.assert_allclose(scn[self.thermal_name].values, exp_thermal, rtol=1e-6)

    def _check_thermal_l2_highlevel(self, scn, exp_rad):
        if "_l2_" in self.reader:
            assert scn[self.sza_rad_name].attrs["units"] == "W m-2 um-1 sr-1"
            assert scn[self.sza_rad_name].attrs["standard_name"] == "toa_outgoing_radiance_per_unit_wavelength"
            np.testing.assert_allclose(scn[self.sza_rad_name].values, exp_rad, rtol=1e-6)

    def test_metadata(self, mda_file):
        """Check that metadata values loaded correctly."""
        mda = self.MD_reader_class(mda_file, self.filename_info, {})

        assert mda.platform_name == self.platform_name
        assert mda.earth_sun_distance() == self.earth_sun_distance

        cal_bands = list(self.calibration_dict.keys())
        assert mda.band_calibration[cal_bands[0]] == self.calibration_dict[cal_bands[0]]
        assert mda.band_calibration[cal_bands[1]] == self.calibration_dict[cal_bands[1]]
        assert mda.band_calibration[cal_bands[2]] == self.calibration_dict[cal_bands[2]]
        assert not mda.band_saturation[cal_bands[0]]
        assert mda.band_saturation[self.spectral_name]
        assert not mda.band_saturation[cal_bands[1]] == self.basic_band_saturated
        if self.reader in ["oli_tirs_l1_tif", "oli_tirs_l2_tif", "etm_l2_tif"]:
            with pytest.raises(KeyError):
                mda.band_saturation[cal_bands[2]]
        else:
            assert not mda.band_saturation[cal_bands[2]]

    def test_area_def(self, mda_file):
        """Check we can get the area defs properly."""
        mda = self.MD_reader_class(mda_file, self.filename_info, {})
        standard_area = mda.build_area_def("B4")
        assert standard_area.area_extent == self.extent

        if self.reader in ["oli_tirs_l1_tif", "etm_l1_tif"]:
            pan_area = mda.build_area_def("B8")
            assert pan_area.area_extent == self.pan_extent


class BaseL1Test:
    """Test functions specific to level-1 products."""

    sza_filename: str

    @pytest.fixture
    def sza_file(self, path, sza_rad_data):
        """Create the file for the Landsat sza."""
        filename = path / self.sza_filename
        create_tif_file(sza_rad_data, "sza", self.area, filename, self.date)
        return os.fspath(filename)

    @pytest.fixture
    def all_files(self, spectral_file, thermal_file, sza_file, mda_file):
        """Return all files."""
        return spectral_file, thermal_file, sza_file, mda_file

    def test_calibration_radiance(self, all_files, spectral_data, thermal_data):
        """Test radiance calibration mode for the reader."""
        exp_spectral = (
            spectral_data * self.calibration_spectral_params[0] + self.calibration_spectral_params[1]
        ).astype(np.float32)
        if self.thermal_name is not None:
            exp_thermal = (
                thermal_data * self.calibration_thermal_params[0] + self.calibration_thermal_params[1]
            ).astype(np.float32)

        scn = Scene(reader=self.reader, filenames=all_files)
        if self.thermal_name is not None:
            scn.load([self.spectral_name, self.thermal_name], calibration="radiance")
        else:
            scn.load([self.spectral_name], calibration="radiance")

        assert scn[self.spectral_name].attrs["units"] == "W m-2 um-1 sr-1"
        assert scn[self.spectral_name].attrs["standard_name"] == "toa_outgoing_radiance_per_unit_wavelength"
        np.testing.assert_allclose(scn[self.spectral_name].values, exp_spectral, rtol=1e-4)

        if self.thermal_name is not None:
            assert scn[self.thermal_name].attrs["units"] == "W m-2 um-1 sr-1"
            assert scn[self.thermal_name].attrs["standard_name"] == "toa_outgoing_radiance_per_unit_wavelength"
            np.testing.assert_allclose(scn[self.thermal_name].values, exp_thermal, rtol=1e-4)

    def test_angles(self, all_files, sza_rad_data):
        """Test calibration modes for the reader."""
        # Check angles are calculated correctly
        if self.reader in ["oli_tirs_l1_tif", "etm_l1_tif", "tm_l1_tif"]:
            scn = Scene(reader=self.reader, filenames=all_files)
            scn.load(["solar_zenith_angle"])
            assert scn["solar_zenith_angle"].attrs["units"] == "degrees"
            assert scn["solar_zenith_angle"].attrs["standard_name"] == "solar_zenith_angle"
            np.testing.assert_allclose(scn["solar_zenith_angle"].values * 100,
                                       np.array(sza_rad_data),
                                       atol=0.01,
                                       rtol=1e-3)


class BaseL2Test:
    """Base test class for level-2 products."""

    rad_filename: str

    @pytest.fixture
    def rad_file(self, path, sza_rad_data):
        """Create the file for the Landsat TRAD channel."""
        filename = path / self.rad_filename
        create_tif_file(sza_rad_data, "TRAD", self.area, filename, self.date)
        return os.fspath(filename)

    @pytest.fixture
    def all_files(self, spectral_file, thermal_file, rad_file, mda_file):
        """Return all files."""
        return spectral_file, thermal_file, rad_file, mda_file


class BaseOLITIRSTest:
    """Base test class for OLI-TIRS products."""

    def test_loading_badchan(self, spectral_file, thermal_file, mda_file):
        """Test loading a Landsat Scene with bad channel requests."""
        good_mda = self.MD_reader_class(mda_file, self.filename_info, {})
        ftype = {"standard_name": "test_data", "units": "test_units"}
        bad_finfo = self.filename_info.copy()
        bad_finfo["data_type"] = "T"

        # Check loading invalid channel for data type
        rdr = self.CH_reader_class(spectral_file, bad_finfo, self.ftype_info, good_mda)
        with pytest.raises(ValueError, match="Requested channel B4 is not available in this granule"):
            rdr.get_dataset({"name": "B4", "calibration": "counts"}, ftype)

        # Thermal band test
        bad_finfo["data_type"] = "O"
        ftype_thermal = self.ftype_info.copy()
        ftype_thermal["file_type"] = f"granule_{self.thermal_name}"
        rdr = self.CH_reader_class(thermal_file, bad_finfo, ftype_thermal, good_mda)
        with pytest.raises(
            ValueError, match=f"Requested channel {self.thermal_name} is not available in this granule"
        ):
            rdr.get_dataset({"name": self.thermal_name, "calibration": "counts"}, ftype)


class BaseETMTest:
    """Base test class for ETM+ products."""


class BaseTMTest:
    """Base test class for TM products."""


class BaseMSSTest:
    """Base test class for MSS products."""

    additional_filename: str
    additional_name: str
    spectral_band_wavelengths: list
    additional_band_wavelengths: list

    thermal_name = None
    calibration_thermal_params = None

    def thermal_file(self):
        """No thermal band in MSS."""
        pass

    def sza_file(self):
        """No sza band in MSS."""
        pass

    @pytest.fixture
    def additional_file(self, path, spectral_data):
        """Create the file for the Landsat thermal channel."""
        filename = path / self.additional_filename
        create_tif_file(spectral_data, self.additional_name, self.area, filename, self.date)
        return os.fspath(filename)

    @pytest.fixture
    def all_files(self, spectral_file, additional_file, mda_file):
        """Return all files."""
        return spectral_file, additional_file, mda_file

    def test_band_names_correction(self, all_files):
        """Test if band names and calibration parameters are set correctly."""
        scn = Scene(reader=self.reader, filenames=all_files)
        scn.load([self.spectral_name, self.additional_name])

        self._assert_wavelengths(self, scn, self.spectral_name, self.spectral_band_wavelengths)
        self._assert_wavelengths(self, scn, self.additional_name, self.additional_band_wavelengths)

    @staticmethod
    def _assert_wavelengths(self, scn, band_name, wavelengths):
        assert scn[band_name].attrs["_satpy_id"]["wavelength"].min == wavelengths[0]
        assert scn[band_name].attrs["_satpy_id"]["wavelength"].central == wavelengths[1]
        assert scn[band_name].attrs["_satpy_id"]["wavelength"].max == wavelengths[2]

    @pytest.mark.skip(reason="Not applicable for MSS")
    def test_angles(self):
        """Override the method as it is not applicable here."""
        pass


class TestOLITIRSL1(BaseOLITIRSTest, BaseL1Test, BaseLandsatTest):
    """Tests for OLI-TIRS Level-1."""

    files_path = "oli_tirs_l1"
    spectral_filename = "LC08_L1TP_026200_20240502_20240513_02_T2_B4.TIF"
    thermal_filename = "LC08_L1TP_026200_20240502_20240513_02_T2_B11.TIF"
    sza_filename = "LC08_L1TP_026200_20240502_20240513_02_T2_SZA.TIF"
    mda_filename = "LC08_L1TP_026200_20240502_20240513_02_T2_MTL.xml"
    mda_etc = "oli_tirs_l1_metadata.xml"

    date = datetime(2024, 5, 12, tzinfo=timezone.utc)
    filename_info = get_filename_info(date, "L1TP", "08", "C")

    spectral_name = "B4"
    thermal_name = "B11"
    sza_rad_name = None

    reader = "oli_tirs_l1_tif"
    CH_reader_class = OLITIRSCHReader
    MD_reader_class = LandsatL1MDReader

    date_time = datetime(2024, 5, 2, 18, 0, 24, tzinfo=timezone.utc)
    calibration_spectral_params = [0.0098329, -49.16426, 2e-05, -0.1]
    calibration_thermal_params = [0.0003342, 0.100000, 480.8883, 1201.1442]
    calibration_dict = {"B1": (0.012357, -61.78647, 2e-05, -0.1),
                        "B5": (0.0060172, -30.08607, 2e-05, -0.1),
                        "B10": (0.0003342, 0.1, 774.8853, 1321.0789)}
    extent = (619485.0, 2440485.0, 850515.0, 2675715.0)
    pan_extent = (619492.5, 2440492.5, 850507.5, 2675707.5)
    zone = 40
    platform_name = "Landsat-8"
    earth_sun_distance = 1.0079981


class TestOLITIRSL2(BaseOLITIRSTest, BaseL2Test, BaseLandsatTest):
    """Tests for OLI-TIRS Level-2."""

    files_path = "oli_tirs_l2"
    spectral_filename = "LC09_L2SP_029030_20240616_20240617_02_T1_SR_B4.TIF"
    thermal_filename = "LC09_L2SP_029030_20240616_20240617_02_T1_ST_B10.TIF"
    rad_filename = "LC09_L2SP_029030_20240616_20240617_02_T1_ST_TRAD.TIF"
    mda_filename = "LC09_L2SP_029030_20240616_20240617_02_T1_MTL.xml"
    mda_etc = "oli_tirs_l2_metadata.xml"

    date = datetime(2024, 6, 16, tzinfo=timezone.utc)
    filename_info = get_filename_info(date, "L2SP", "09", "C")

    spectral_name = "B4"
    thermal_name = "B10"
    sza_rad_name = "TRAD"

    reader = "oli_tirs_l2_tif"
    CH_reader_class = OLITIRSL2CHReader
    MD_reader_class = LandsatL2MDReader

    date_time = datetime(2024, 6, 16, 17, 10, 58, tzinfo=timezone.utc)
    calibration_spectral_params = [2.75e-05, -0.2]
    calibration_thermal_params = [0.00341802, 149.0]
    calibration_dict = {"B1": (2.75e-05, -0.2),
                        "B5": (2.75e-05, -0.2),
                        "B10": (0.00341802, 149.0)}
    extent = (534885.0, 4665585.0, 765015.0, 4899315.0)
    pan_extent = None
    zone = 14
    platform_name = "Landsat-9"
    earth_sun_distance = 1.0158933


class TestETML1(BaseETMTest, BaseL1Test, BaseLandsatTest):
    """Tests for ETM+ Level-1."""

    files_path = "etm_l1"
    spectral_filename = "LE07_L1TP_230080_20231208_20240103_02_T1_B4.TIF"
    thermal_filename = "LE07_L1TP_230080_20231208_20240103_02_T1_B6_VCID_1.TIF"
    sza_filename = "LE07_L1TP_230080_20231208_20240103_02_T1_SZA.TIF"
    mda_filename = "LE07_L1TP_230080_20231208_20240103_02_T1_MTL.xml"
    mda_etc = "etm_l1_metadata.xml"

    date = datetime(2023, 12, 8, tzinfo=timezone.utc)
    filename_info = get_filename_info(date, "L1TP", "07", "E")

    spectral_name = "B4"
    thermal_name = "B6_VCID_1"
    sza_rad_name = None

    reader = "etm_l1_tif"
    CH_reader_class = ETMCHReader
    MD_reader_class = LandsatL1MDReader

    date_time = datetime(2023, 12, 8, 11, 44, 51, tzinfo=timezone.utc)
    calibration_spectral_params = [9.6929e-01, -6.06929, 2.7591e-03, -0.017277]
    calibration_thermal_params = [6.7087e-02, -0.06709, 666.09, 1282.71]
    calibration_dict = {"B1": (7.7874e-01, -6.97874, 1.1661e-03, -0.010450),
                        "B5": (1.2622e-01, -1.12622, 1.7365e-03, -0.015494),
                        "B6_VCID_2": (3.7205e-02, 3.16280, 666.09, 1282.71)}
    extent = (240885.0, -3301215.0, 483015.0, -3091485.0)
    pan_extent = (240892.5, -3301207.5, 483007.5, -3091492.5)
    zone = 20
    platform_name = "Landsat-7"
    earth_sun_distance = 0.9850987


class TestETML2(BaseETMTest, BaseL2Test, BaseLandsatTest):
    """Tests for ETM+ Level-2."""

    files_path = "etm_l2"
    spectral_filename = "LE07_L2SP_028030_20230817_20230912_02_T1_SR_B4.TIF"
    thermal_filename = "LE07_L2SP_028030_20230817_20230912_02_T1_ST_B6.TIF"
    rad_filename = "LE07_L2SP_028030_20230817_20230912_02_T1_ST_TRAD.TIF"
    mda_filename = "LE07_L2SP_028030_20230817_20230912_02_T1_MTL.xml"
    mda_etc = "etm_l2_metadata.xml"

    date = datetime(2023, 8, 17, tzinfo=timezone.utc)
    filename_info = get_filename_info(date, "L2SP", "07", "E")

    spectral_name = "B4"
    thermal_name = "B6"
    sza_rad_name = "TRAD"

    reader = "etm_l2_tif"
    CH_reader_class = ETML2CHReader
    MD_reader_class = LandsatL2MDReader

    date_time = datetime(2023, 8, 17, 14, 54, 20, tzinfo=timezone.utc)
    calibration_spectral_params = [2.75e-05, -0.2]
    calibration_thermal_params = [0.00341802, 149.0]
    calibration_dict = {"B1": (2.75e-05, -0.2),
                        "B5": (2.75e-05, -0.2),
                        "B6": (0.00341802, 149.0)}
    extent = (85785.0, 4680885.0, 336915.0, 4904115.0)
    pan_extent = None
    zone = 15
    platform_name = "Landsat-7"
    earth_sun_distance = 1.0124651


class TestTML1(BaseTMTest, BaseL1Test, BaseLandsatTest):
    """Tests for TM Level-1."""

    files_path = "tm_l1"
    spectral_filename = "LT04_L1TP_143021_19890818_20200916_02_T1_B4.TIF"
    thermal_filename = "LT04_L1TP_143021_19890818_20200916_02_T1_B6.TIF"
    sza_filename = "LT04_L1TP_143021_19890818_20200916_02_T1_SZA.TIF"
    mda_filename = "LT04_L1TP_143021_19890818_20200916_02_T1_MTL.xml"
    mda_etc = "tm_l1_metadata.xml"

    date = datetime(1989, 8, 18, tzinfo=timezone.utc)
    filename_info = get_filename_info(date, "L1TP", "04", "T")

    spectral_name = "B4"
    thermal_name = "B6"
    sza_rad_name = None

    reader = "tm_l1_tif"
    CH_reader_class = TMCHReader
    MD_reader_class = LandsatL1MDReader

    date_time = datetime(1989, 8, 18, 4, 26, 11, tzinfo=timezone.utc)
    calibration_spectral_params = [8.7602e-01, -2.38602, 2.7296e-03, -0.007435]
    calibration_thermal_params = [5.5375e-02, 1.18243, 671.62, 1284.30]
    calibration_dict = {"B1": (6.7921e-01, -2.19921, 1.1252e-03, -0.003643),
                        "B5": (1.2508e-01, -0.49508, 1.8160e-03, -0.007188),
                        "B6": (5.5375e-02, 1.18243, 671.62, 1284.30)}
    extent = (322185.0, 6085185.0, 567615.0, 6311415.0)
    pan_extent = None
    zone = 46
    platform_name = "Landsat-4"
    earth_sun_distance = 1.0122057


class TestTML2(BaseTMTest, BaseL2Test, BaseLandsatTest):
    """Tests for TM Level-2."""

    files_path = "tm_l2"
    spectral_filename = "LT05_L2SP_165054_20110817_20200820_02_T1_SR_B4.TIF"
    thermal_filename = "LT05_L2SP_165054_20110817_20200820_02_T1_ST_B6.TIF"
    rad_filename = "LT05_L2SP_165054_20110817_20200820_02_T1_ST_TRAD.TIF"
    mda_filename = "LT05_L2SP_165054_20110817_20200820_02_T1_MTL.xml"
    mda_etc = "tm_l2_metadata.xml"

    date = datetime(2011, 8, 17, tzinfo=timezone.utc)
    filename_info = get_filename_info(date, "L2SP", "05", "T")

    spectral_name = "B4"
    thermal_name = "B6"
    sza_rad_name = "TRAD"

    reader = "tm_l2_tif"
    CH_reader_class = TML2CHReader
    MD_reader_class = LandsatL2MDReader

    date_time = datetime(2011, 8, 17, 7, 10, 40, tzinfo=timezone.utc)
    calibration_spectral_params = [2.75e-05, -0.2]
    calibration_thermal_params = [0.00341802, 149.0]
    calibration_dict = {"B1": (2.75e-05, -0.2),
                        "B5": (2.75e-05, -0.2),
                        "B6": (0.00341802, 149.0)}
    extent = (258285.0, 855285.0, 495315.0, 1064415.0)
    pan_extent = None
    zone = 38
    platform_name = "Landsat-5"
    earth_sun_distance = 1.0125021


class TestMSSLandsat1(BaseMSSTest, BaseL1Test, BaseLandsatTest):
    """Tests for Landsat-1 MSS."""

    files_path = "mss_l1_landsat1"
    spectral_filename = "LM01_L1TP_032030_19720729_20200909_02_T2_B4.TIF"
    additional_filename = "LM01_L1TP_032030_19720729_20200909_02_T2_B5.TIF"
    mda_filename = "LM01_L1TP_032030_19720729_20200909_02_T2_MTL.xml"
    mda_etc = "mss_l1_landsat1_metadata.xml"

    date = datetime(1972, 7, 29, tzinfo=timezone.utc)
    filename_info = get_filename_info(date, "L1TP", "01", "M")

    spectral_name = "B4"
    thermal_name = None
    sza_rad_name = None

    reader = "mss_l1_tif"
    CH_reader_class = MSSCHReader
    MD_reader_class = LandsatL1MDReader

    date_time = datetime(1972, 7, 29, 16, 49, 31, tzinfo=timezone.utc)
    calibration_spectral_params = [9.5591e-01, -18.55591, 1.7282e-03, -0.033547]
    calibration_thermal_params = None
    calibration_dict = {"B5": (6.4843e-01, -0.74843, 1.3660e-03, -0.001577),
                        "B6": (6.5236e-01, -0.75236, 1.6580e-03, -0.001912),
                        "B7": (6.0866e-01, -0.60866, 2.3287e-03, -0.002329)}
    extent = (442110.0, 4667550.0, 673530.0, 4887990.0)
    pan_extent = None
    zone = 14
    platform_name = "Landsat-1"
    earth_sun_distance = 1.0152109

    additional_name = "B5"
    spectral_band_wavelengths = [0.5, 0.55, 0.6]
    additional_band_wavelengths = [0.6, 0.65, 0.7]


class TestMSSLandsat4(BaseMSSTest, BaseL1Test, BaseLandsatTest):
    """Tests for Landsat-4 MSS."""

    files_path = "mss_l1_landsat4"
    spectral_filename = "LM04_L1TP_029030_19840415_20200903_02_T2_B4.TIF"
    additional_filename = "LM04_L1TP_029030_19840415_20200903_02_T2_B3.TIF"
    mda_filename = "LM04_L1TP_029030_19840415_20200903_02_T2_MTL.xml"
    mda_etc = "mss_l1_landsat4_metadata.xml"

    date = datetime(1984, 4, 15, tzinfo=timezone.utc)
    filename_info = get_filename_info(date, "L1TP", "04", "M")

    spectral_name = "B4"
    thermal_name = None
    sza_rad_name = None

    reader = "mss_l1_tif"
    CH_reader_class = MSSCHReader
    MD_reader_class = LandsatL1MDReader

    date_time = datetime(1984, 4, 15, 16, 38, 15, tzinfo=timezone.utc)
    calibration_spectral_params = [4.7638e-01, 3.82362, 1.7954e-03, 0.014411]
    calibration_thermal_params = None
    calibration_dict = {"B1": (8.7520e-01, 2.92480, 1.5680e-03, 0.005240),
                        "B2": (6.2008e-01, 3.07992, 1.2865e-03, 0.006390),
                        "B3": (5.4921e-01, 4.55079, 1.4070e-03, 0.011659)}
    extent = (540030.0, 4679970.0, 774450.0, 4888350.0)
    pan_extent = None
    zone = 14
    platform_name = "Landsat-4"
    earth_sun_distance = 1.0035512

    additional_name = "B3"
    spectral_band_wavelengths = [0.8, 0.95, 1.1]
    additional_band_wavelengths = [0.7, 0.75, 0.8]


class TestAntarctic:
    """Tests if specific Antarctic CRS params can be handled. Only area test is needed here."""

    files_path = "antarctic"
    mda_filename = "LC08_L1GT_132122_20220301_20220308_02_T2_MTL.xml"
    mda_etc = "antarctic_metadata.xml"

    date = datetime(2024, 5, 12, tzinfo=timezone.utc)

    filename_info = get_filename_info(date, "L1TP", "08", "C")

    MD_reader_class = LandsatL1MDReader

    extent = (-138315.0, 590385.0, 61215.0, 808815.0)
    pan_extent = (-138307.5, 590392.5, 61207.5, 808807.5)
    zone = 40

    @pytest.fixture
    def path(self, tmp_path_factory):
        """Create a temporary directory."""
        return tmp_path_factory.mktemp(self.files_path)

    @pytest.fixture
    def mda_file(self, path):
        """Create the Landsat metadata xml file."""
        filename = path / self.mda_filename
        shutil.copyfile(os.path.join(ETC_DIR, self.mda_etc), filename)
        return os.fspath(filename)

    def test_area_def(self, mda_file):
        """Check we can get the area defs properly."""
        mda = self.MD_reader_class(mda_file, self.filename_info, {})
        standard_area = mda.build_area_def("B4")
        assert standard_area.area_extent == self.extent

        pan_area = mda.build_area_def("B8")
        assert pan_area.area_extent == self.pan_extent
