"""Tests for the EarthCARE MSI L1c reader."""
import os
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from satpy.tests.reader_tests.test_hdf5_utils import FakeHDF5FileHandler

N_BANDS = 7
N_SCANS = 20
N_COLS = 2048
SHAPE_SC = (300, 6000)
SOL_IRRAD = [30.9, 19.59, 14.77, 8.25, 0., 0., 0.]
DIMLIST = np.ones((N_BANDS, N_SCANS, N_COLS))


class FakeHDF5FileHandler2(FakeHDF5FileHandler):
    """Swap-in HDF5 File Handler."""


    def _setup_test_data(self, N_BANDS, N_SCANS, N_COLS):
        # Set some default attributes
        data = {
            "ScienceData/pixel_values":
                xr.DataArray(
                    da.ones((N_BANDS, N_SCANS, N_COLS), chunks=1024, dtype=np.float32),
                    attrs={"units": "Wm-2 sr-1 or K", "DIMENSION_LIST": DIMLIST},
                    dims=("band", "dim_2", "dim_1")),
            "ScienceData/land_flag":
                xr.DataArray(
                    da.ones((N_SCANS, N_COLS), chunks=1024, dtype=np.uint16),
                    attrs={"units": ""},
                    dims=("along_track", "across_track")),
            "ScienceData/solar_azimuth_angle":
                xr.DataArray(
                    da.ones((N_SCANS, N_COLS), chunks=1024, dtype=np.float32),
                    attrs={"units": "degrees"},
                    dims=("along_track", "across_track")),
            "ScienceData/longitude":
                xr.DataArray(
                    da.ones((N_SCANS, N_COLS), chunks=1024, dtype=np.float32),
                    attrs={"units": "degrees"},
                    dims=("along_track", "across_track")),
            "ScienceData/latitude":
                xr.DataArray(
                    da.ones((N_SCANS, N_COLS), chunks=1024, dtype=np.float32),
                    attrs={"units": "degrees"},
                    dims=("along_track", "across_track")),
            "NonStandard/solar_irradiance":
                xr.DataArray(
                    da.array(SOL_IRRAD),
                    attrs={"units": "W m-2"},
                    dims=("band")),
        }

        return data

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        test_content = self._setup_test_data(N_BANDS, N_SCANS, N_COLS)
        return test_content


class ECMSIL1CTester:
    """Test MSI/EarthCARE L1C Reader."""

    def setup_method(self):
        """Wrap HDF5 file handler with our own fake handler."""
        from satpy._config import config_search_paths
        from satpy.readers.msi_ec_l1c_h5 import MSIECL1CFileHandler
        self.reader_configs = config_search_paths(os.path.join("readers", self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(MSIECL1CFileHandler, "__bases__", (FakeHDF5FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def teardown_method(self):
        """Stop wrapping the HDF5 file handler."""
        self.p.stop()


class TestECMSIL1C(ECMSIL1CTester):
    """Test the EarthCARE MSI L1C reader."""

    yaml_file = "msi_l1c_earthcare.yaml"
    filename = "ECA_EXAA_MSI_RGR_1C_20250625T005649Z_20250625T024013Z_42043E.h5"


    @mock.patch("satpy.readers.hdf5_utils.HDF5FileHandler._get_reference")
    @mock.patch("h5py.File")
    def test_get_pixvalues(self, mock_h5py_file, mock_hdf5_utils_get_reference):
        """Test loadingpixel values from file."""
        from satpy.readers import load_reader
        mock_h5py_file.return_value = mock.MagicMock()
        mock_hdf5_utils_get_reference.return_value = DIMLIST
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames([self.filename])
        assert 1 == len(files)
        reader.create_filehandlers(files)
        # Make sure we have some files
        assert reader.file_handlers
        available_datasets = list(reader.available_dataset_ids)
        assert len(available_datasets) == 27

        res = reader.load(["VIS", "NIR", "TIR1", "TIR3", "solar_azimuth_angle", "land_water_mask"])
        assert len(res) == 6
        with pytest.raises(KeyError):
            res["TIR2"]
        with pytest.raises(KeyError):
            res["SWIR1"]

        assert res["VIS"].shape == (20, 2048)
        assert res["VIS"].attrs["calibration"] == "reflectance"
        assert res["VIS"].attrs["units"] == "%"

        assert res["TIR1"].shape == (20, 2048)
        assert res["TIR1"].attrs["calibration"] == "brightness_temperature"
        assert res["TIR1"].attrs["units"] == "K"
        assert res["TIR1"].dtype == np.float32

        assert res["solar_azimuth_angle"].shape == (20, 2048)
        assert res["solar_azimuth_angle"].attrs["units"] == "degrees"
        assert res["solar_azimuth_angle"].dtype == np.float32

        assert res["land_water_mask"].shape == (20, 2048)
        assert res["land_water_mask"].attrs["units"] == 1
        assert res["land_water_mask"].dtype == np.uint16


    @mock.patch("satpy.readers.hdf5_utils.HDF5FileHandler._get_reference")
    @mock.patch("h5py.File")
    def test_calibration(self, mock_h5py_file, mock_hdf5_utils_get_reference):
        """Test loadingpixel values from file."""
        from satpy.readers import load_reader
        from satpy.tests.utils import make_dataid

        mock_h5py_file.return_value = mock.MagicMock()
        mock_hdf5_utils_get_reference.return_value = DIMLIST

        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames([self.filename])
        reader.create_filehandlers(files)

        with pytest.raises(KeyError):
            reader.load([make_dataid(name="VIS", calibration="counts")])
        with pytest.raises(KeyError):
            reader.load([make_dataid(name="TIR1", calibration="counts")])
        with pytest.raises(KeyError):
            reader.load([make_dataid(name="TIR1", calibration="radiance")])

        res = reader.load([make_dataid(name="VIS", calibration="radiance")])
        assert res["VIS"].attrs["calibration"] == "radiance"
        assert res["VIS"].attrs["units"] == "W m-2 sr-1"
        assert np.all(np.array(res["VIS"].data) == 1)

        res = reader.load([make_dataid(name="NIR", calibration="reflectance")])
        assert res["NIR"].attrs["calibration"] == "reflectance"
        assert res["NIR"].attrs["units"] == "%"
        assert np.all(np.array(res["NIR"].data) == 1 * np.pi * 100 / SOL_IRRAD[1])
