"""Tests for the EarthCARE MSI L1c reader."""

import dask.array as da
import h5py
import numpy as np
import pytest
import xarray as xr

from satpy import Scene
from satpy.tests.utils import make_dataid

N_BANDS = 7
N_SCANS = 20
N_COLS = 384
SHAPE_SC = (300, 6000)
SOL_IRRAD = np.array([30.9, 19.59, 14.77, 8.25], dtype=np.float32)
# Create 384 values for SOL_IRRAD
SOL_IRRAD = SOL_IRRAD.reshape((-1, 1)) * np.ones(N_COLS, dtype=np.float32).reshape((1, -1))
DIMLIST = np.ones((N_BANDS, N_SCANS, N_COLS))

def _setup_science_data(N_BANDS, N_SCANS, N_COLS):
    # Set some default attributes
    data = {
        "pixel_values":
        xr.DataArray(
            da.ones((N_BANDS, N_SCANS, N_COLS), chunks=1024, dtype=np.float32),
            attrs={"units": "Wm-2 sr-1 or K", "DIMENSION_LIST": DIMLIST},
            dims=("band", "dim_2", "dim_1")),
        "land_flag":
        xr.DataArray(
            da.ones((N_SCANS, N_COLS), chunks=1024, dtype=np.uint16),
            attrs={"units": ""},
            dims=("along_track", "across_track")),
        "solar_azimuth_angle":
        xr.DataArray(
            da.ones((N_SCANS, N_COLS), chunks=1024, dtype=np.float32),
            attrs={"units": "degrees"},
            dims=("along_track", "across_track")),
        "longitude":
        xr.DataArray(
            da.ones((N_SCANS, N_COLS), chunks=1024, dtype=np.float32),
            attrs={"units": "degrees"},
            dims=("along_track", "across_track")),
        "latitude":
        xr.DataArray(
            da.ones((N_SCANS, N_COLS), chunks=1024, dtype=np.float32),
            attrs={"units": "degrees"},
            dims=("along_track", "across_track")),
        "solar_spectral_irradiance":
        xr.DataArray(
            da.array(SOL_IRRAD),
            attrs={"units": "W m-2"},
            dims=("band", "across_track")),
    }

    return data


@pytest.fixture(scope="session")
def msi_ec_l1c_dummy_file(tmp_path_factory):
    """Create a fake insat MSI 1C file."""
    filename = tmp_path_factory.mktemp("data") / "ECA_EXAA_MSI_RGR_1C_20250625T005649Z_20250625T024013Z_42043E.h5"

    with h5py.File(filename, "w") as fid:
        ScienceData_group = fid.create_group("ScienceData")
        for name, value in _setup_science_data(N_BANDS, N_SCANS, N_COLS).items():
            ScienceData_group[name] = value

    return filename


def test_get_pixvalues(msi_ec_l1c_dummy_file):
        """Test loadingpixel values from file."""
        res = Scene(reader="msi_l1c_earthcare", filenames=[msi_ec_l1c_dummy_file])

        available_datasets = list(res.available_dataset_ids())
        assert len(available_datasets) == 27

        res.load(["VIS", "VNIR", "TIR1", "TIR3", "solar_azimuth_angle", "land_water_mask"])
        #assert len(res) == 6
        with pytest.raises(KeyError):
            res["TIR2"]
        with pytest.raises(KeyError):
            res["SWIR1"]

        assert res["VIS"].shape == (20, N_COLS)
        assert res["VIS"].attrs["calibration"] == "reflectance"
        assert res["VIS"].attrs["units"] == "%"

        assert res["TIR1"].shape == (20, N_COLS)
        assert res["TIR1"].attrs["calibration"] == "brightness_temperature"
        assert res["TIR1"].attrs["units"] == "K"
        assert res["TIR1"].dtype == np.float32

        assert res["solar_azimuth_angle"].shape == (20, N_COLS)
        assert res["solar_azimuth_angle"].attrs["units"] == "degrees"
        assert res["solar_azimuth_angle"].dtype == np.float32

        assert res["land_water_mask"].shape == (20, N_COLS)
        assert res["land_water_mask"].attrs["units"] == 1
        assert res["land_water_mask"].dtype == np.uint16


def test_calibration(msi_ec_l1c_dummy_file):
    """Test loadingpixel values from file."""
    res = Scene(reader="msi_l1c_earthcare", filenames=[msi_ec_l1c_dummy_file])

    with pytest.raises(KeyError):
        res.load([make_dataid(name="VIS", calibration="counts")])
    with pytest.raises(KeyError):
        res.load([make_dataid(name="TIR1", calibration="counts")])
    with pytest.raises(KeyError):
        res.load([make_dataid(name="TIR1", calibration="radiance")])

    res.load([make_dataid(name="VIS", calibration="radiance")])
    assert res["VIS"].attrs["calibration"] == "radiance"
    assert res["VIS"].attrs["units"] == "W m-2 sr-1"
    assert np.all(np.array(res["VIS"].data) == 1)

    res.load([make_dataid(name="VNIR", calibration="reflectance")])
    assert res["VNIR"].attrs["calibration"] == "reflectance"
    assert res["VNIR"].attrs["units"] == "%"
    assert np.all(np.array(res["VNIR"].data) == 1 * np.pi * 100 / SOL_IRRAD[1])
