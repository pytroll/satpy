# Copyright (c) 2023 Satpy developers
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
"""Test module for the datatailor netcdf reader."""

import numpy as np
import pytest
import xarray
from pyresample import SwathDefinition


@pytest.fixture
def fake_datatailor_dataset():
    """Generate the contents for a fake data tailor NetCDF file."""
    nx = 5
    ny = 5
    nz = 3
    lats = np.linspace(-89, 89, nx*ny).reshape(nx, ny)
    lons = np.linspace(-189, 189, nx*ny).reshape(nx, ny)
    iwv = np.linspace(0, 1, nx*ny).reshape(nx, ny)
    temp = np.linspace(200, 330, nx*ny*nz).reshape(nx, ny, nz)

    ds = xarray.Dataset(
        {"lat":
            ((yx := ("along_track", "across_track")),
             lats,
             {"long_name": "grid_latitude",
              "standard_name": "latitude",
              "units": "degrees_north"}),
         "lon":
             (yx, lons,
              {"long_name": "grid_longitude",
               "standard_name": "longitude",
               "units": "degrees_east"}),
         "integrated_water_vapour":
            (yx, iwv,
             {'long_name': 'integrated water vapour',
              'standard_name': 'atmosphere_mass_content_of_water_vapour',
              'units': 'kg.m^-2',
              'comment': 'Integrated water vapour (for 120 IFOV)'}),
         "atmospheric_temperature":
            (yx + ("nlt",), temp,
             {'standard_name': 'air_temperature',
              'comment': 'Temperature (for 120 IFOV with up to 101 vertical levels)',
              'units': 'degree(K)'})},
        {"nlt": np.arange(nz)},
        {'Conventions': 'CF-1.7',
         'Metadata_Conventions': 'Unidata Dataset Discovery v1.0',
         'title': 'IASI 2b the Infrared Atmospheric Sounding Interferometer',
         'title_short_name': 'IASI L2',
         'summary': 'NA',
         'source': 'MetOp-B IASI',
         'references': 'https://www.eumetsat.int/website/home/Data/DataDelivery/EUMETSATDataCentre/index.html',
         'comment': 'Search for Infrared Atmospheric Sounding Interferometer in the references URL',
         'keywords': 'EUMETSAT, DATA CENTRE, EPS, IASI NetCDF',
         'history': '2019-11-13T13:54:03Z - Created by EUMETSAT',
         'reference_url': 'https://navigator.eumetsat.int/product/EO:EUM:DAT:METOP:MXI-N-SO2',
         'data_format_type': 'NetCDF-4 classic model',
         'product_native_version': '11.0',
         'product_netcdf_version': '2.0',
         'producer_agency': 'EUMETSAT',
         'processing_centre': 'CGS1',
         'platform_type': 'spacecraft',
         'platform': 'M01',
         'platform_long_name': 'MetOp-B',
         'sensor': 'IASI',
         'sensor_model': 1,
         'processing_level': '02',
         'product_type': 'SND',
         'processor_major_version': 6,
         'processor_minor_version': 7,
         'instrument_calibration_version': 'xxxxx',
         'format_major_version': 11,
         'format_minor_version': 0,
         'granule_name': 'IASI_SND_02_M01_20230602193252Z_20230602211452Z_N_O_20230602203938Z',
         'parent_granule_name': 'IASI_xxx_1C_M01_20230602193252Z_20230602211452Z_N_O_20230602202801Z',
         'contents': 'IASI Level 2 Measurements',
         'native_product_size': 348809171,
         'production_date_time': '20230622145343Z',
         'start_sensing_time': '20230602193252Z',
         'stop_sensing_time': '20230602211452Z',
         'start_orbit_number': 55551,
         'end_orbit_number': 55552,
         'orbit_semi_major_axis': 7204538444,
         'orbit_eccentricity': 0.00125,
         'orbit_inclination': 98.673,
         'rev_orbit_period': 6081.7,
         'equator_crossing_date_time': '20191112113554000Z',
         'subsat_track_start_lat': 80.916,
         'subsat_track_start_lon': -79.04,
         'subsat_track_end_lat': 80.209,
         'subsat_track_end_lon': -114.619,
         'qa_duration_product': 6120000,
         'software_name': 'DataTailor',
         'software_ version': '3.2.0',
         'plugin_name': 'epct_plugin_gis',
         'plugin_version': '3.1.0',
         'process_id': 'EPCT_IASISND02_54de2452',
         'process_uuid': '54de2452',
         'creation_time': '2023-06-22T14:53:48.307508Z',
         'chain name': 'Custom',
         'filter': 'null',
         'aggregation': 'null',
         'projection': 'null',
         'roi': 'null',
         'format': 'netcdf4',
         'quicklook': 'null',
         'resample_method': 'null',
         'resample_resolution': 'null',
         'stretch_method': 'null',
         'sensing_start': 'null',
         'sensing_stop': 'null'})
    return ds


@pytest.fixture
def fake_datatailor_nc_file(fake_datatailor_dataset, tmp_path):
    """Generate a fake data tailor NetCDF file."""
    of = tmp_path / "fakesat-fakesensor-30010401000000-30010401010000.nc"
    fake_datatailor_dataset.to_netcdf(of)
    return of


def test_datatailor_nc(fake_datatailor_nc_file):
    """Test the datatailor NC reader."""
    from satpy import Scene
    sc = Scene(filenames=[fake_datatailor_nc_file], reader=["datatailor_nc"])
    exp = {"lat", "lon", "nlt", "atmospheric_temperature", "integrated_water_vapour"}
    assert set(sc.available_dataset_names()) == exp
    sc.load(exp)
    assert sc["atmospheric_temperature"].dims == ("y", "x", "nlt")
    assert isinstance(sc["atmospheric_temperature"].attrs["area"],
                      SwathDefinition)
