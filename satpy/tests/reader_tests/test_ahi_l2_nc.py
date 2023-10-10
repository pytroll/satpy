"""Tests for the Himawari L2 netCDF reader."""

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from satpy.readers.ahi_l2_nc import HIML2NCFileHandler
from satpy.tests.utils import make_dataid

rng = np.random.default_rng()
clmk_data = rng.integers(0, 3, (5500, 5500), dtype=np.uint16)
cprob_data = rng.uniform(0, 1, (5500, 5500))
lat_data = rng.uniform(-90, 90, (5500, 5500))
lon_data = rng.uniform(-180, 180, (5500, 5500))

start_time = datetime(2023, 8, 24, 5, 40, 21)
end_time = datetime(2023, 8, 24, 5, 49, 40)

dimensions = {'Columns': 5500, 'Rows': 5500}

exp_ext = (-5499999.9012, -5499999.9012, 5499999.9012, 5499999.9012)

global_attrs = {"time_coverage_start": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "time_coverage_end": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "instrument_name": "AHI",
                "satellite_name": "Himawari-9",
                "cdm_data_type": "Full Disk",
                }

badarea_attrs = global_attrs.copy()
badarea_attrs['cdm_data_type'] = 'bad_area'


def ahil2_filehandler(fname, platform='h09'):
    """Instantiate a Filehandler."""
    fileinfo = {'platform': platform}
    filetype = None
    fh = HIML2NCFileHandler(fname, fileinfo, filetype)
    return fh


@pytest.fixture(scope="session")
def himl2_filename(tmp_path_factory):
    """Create a fake himawari l2 file."""
    fname = f'{tmp_path_factory.mktemp("data")}/AHI-CMSK_v1r1_h09_s202308240540213_e202308240549407_c202308240557548.nc'
    ds = xr.Dataset({'CloudMask': (['Rows', 'Columns'], clmk_data)},
                    coords={'Latitude': (['Rows', 'Columns'], lat_data),
                            'Longitude': (['Rows', 'Columns'], lon_data)},
                    attrs=global_attrs)
    ds.to_netcdf(fname)
    return fname


@pytest.fixture(scope="session")
def himl2_filename_bad(tmp_path_factory):
    """Create a fake himawari l2 file."""
    fname = f'{tmp_path_factory.mktemp("data")}/AHI-CMSK_v1r1_h09_s202308240540213_e202308240549407_c202308240557548.nc'
    ds = xr.Dataset({'CloudMask': (['Rows', 'Columns'], clmk_data)},
                    coords={'Latitude': (['Rows', 'Columns'], lat_data),
                            'Longitude': (['Rows', 'Columns'], lon_data)},
                    attrs=badarea_attrs)
    ds.to_netcdf(fname)

    return fname


def test_startend(himl2_filename):
    """Test start and end times are set correctly."""
    fh = ahil2_filehandler(himl2_filename)
    assert fh.start_time == start_time
    assert fh.end_time == end_time


def test_ahi_l2_area_def(himl2_filename, caplog):
    """Test reader handles area definition correctly."""
    ps = '+proj=geos +lon_0=140.7 +h=35785863 +x_0=0 +y_0=0 +a=6378137 +rf=298.257024882273 +units=m +no_defs +type=crs'

    # Check case where input data is correct size.
    fh = ahil2_filehandler(himl2_filename)
    clmk_id = make_dataid(name="cloudmask")
    area_def = fh.get_area_def(clmk_id)
    assert area_def.width == dimensions['Columns']
    assert area_def.height == dimensions['Rows']
    assert np.allclose(area_def.area_extent, exp_ext)
    assert area_def.proj4_string == ps

    # Check case where input data is incorrect size.
    with pytest.raises(ValueError):
        fh = ahil2_filehandler(himl2_filename)
        fh.nlines = 3000
        fh.get_area_def(clmk_id)


def test_bad_area_name(himl2_filename_bad):
    """Check case where area name is not correct."""
    global_attrs['cdm_data_type'] = 'bad_area'
    with pytest.raises(ValueError):
        ahil2_filehandler(himl2_filename_bad)
    global_attrs['cdm_data_type'] = 'Full Disk'


def test_load_data(himl2_filename):
    """Test that data is loaded successfully."""
    fh = ahil2_filehandler(himl2_filename)
    clmk_id = make_dataid(name="cloudmask")
    clmk = fh.get_dataset(clmk_id, {'file_key': 'CloudMask'})
    assert np.allclose(clmk.data, clmk_data)
