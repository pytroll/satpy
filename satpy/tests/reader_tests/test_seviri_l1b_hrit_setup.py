#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Satpy developers
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
"""Setup for SEVIRI HRIT reader tests."""

from datetime import datetime
from unittest import mock

import numpy as np

from satpy.readers.seviri_l1b_hrit import HRITMSGFileHandler, HRITMSGPrologueFileHandler
from satpy.tests.reader_tests.test_seviri_base import ORBIT_POLYNOMIALS


def new_get_hd(instance, hdr_info):
    """Generate some metadata."""
    instance.mda = {'spectral_channel_id': 1}
    instance.mda.setdefault('number_of_bits_per_pixel', 10)

    instance.mda['projection_parameters'] = {'a': 6378169.00,
                                             'b': 6356583.80,
                                             'h': 35785831.00,
                                             'SSP_longitude': 0.0}
    instance.mda['orbital_parameters'] = {}
    instance.mda['total_header_length'] = 12


def get_new_read_prologue(prologue):
    """Create mocked read_prologue() method."""
    def new_read_prologue(self):
        self.prologue = prologue
    return new_read_prologue


def get_fake_file_handler(start_time, nlines, ncols, projection_longitude=0,
                          orbit_polynomials=ORBIT_POLYNOMIALS):
    """Create a mocked SEVIRI HRIT file handler."""
    prologue = get_fake_prologue(projection_longitude, orbit_polynomials)
    mda = get_fake_mda(nlines=nlines, ncols=ncols, start_time=start_time)
    filename_info = get_fake_filename_info(start_time)
    epilogue = get_fake_epilogue()

    m = mock.mock_open()
    with mock.patch('satpy.readers.seviri_l1b_hrit.np.fromfile') as fromfile, \
            mock.patch('satpy.readers.hrit_base.open', m, create=True) as newopen, \
            mock.patch('satpy.readers.utils.open', m, create=True) as utilopen, \
            mock.patch('satpy.readers.seviri_l1b_hrit.CHANNEL_NAMES'), \
            mock.patch.object(HRITMSGFileHandler, '_get_hd', new=new_get_hd), \
            mock.patch.object(HRITMSGPrologueFileHandler, 'read_prologue',
                              new=get_new_read_prologue(prologue)):

        fromfile.return_value = np.array(
            [(1, 2)],
            dtype=[('total_header_length', int),
                   ('hdr_id', int)]
        )
        newopen.return_value.__enter__.return_value.tell.return_value = 1
        # The size of the return value hereafter was chosen arbitrarily with the expectation
        # that it would return sufficiently many bytes for testing the fake-opening of HRIT
        # files.
        utilopen.return_value.__enter__.return_value.read.return_value = bytes([0]*8192)
        prologue = HRITMSGPrologueFileHandler(
            filename='dummy_prologue_filename',
            filename_info=filename_info,
            filetype_info={}
        )
        epilogue = mock.MagicMock(epilogue=epilogue)

        reader = HRITMSGFileHandler(
            'filename',
            filename_info,
            {'filetype': 'info'},
            prologue,
            epilogue
        )
        reader.mda.update(mda)
        return reader


def get_fake_prologue(projection_longitude, orbit_polynomials):
    """Create a fake HRIT prologue."""
    return {
         "SatelliteStatus": {
             "SatelliteDefinition": {
                 "SatelliteId": 324,
                 "NominalLongitude": -3.5
             },
             'Orbit': {
                 'OrbitPolynomial': orbit_polynomials,
             }
         },
         'GeometricProcessing': {
             'EarthModel': {
                 'TypeOfEarthModel': 2,
                 'EquatorialRadius': 6378.169,
                 'NorthPolarRadius': 6356.5838,
                 'SouthPolarRadius': 6356.5838
             }
         },
         'ImageDescription': {
             'ProjectionDescription': {
                 'LongitudeOfSSP': projection_longitude
             },
             'Level15ImageProduction': {
                 'ImageProcDirection': 1
             }
         },
         'ImageAcquisition': {
            'PlannedAcquisitionTime': {
                'TrueRepeatCycleStart': datetime(2006, 1, 1, 12, 15, 9, 304888),
                'PlannedRepeatCycleEnd': datetime(2006, 1, 1, 12, 30, 0, 0)
            }
         }
    }


def get_fake_epilogue():
    """Create a fake HRIT epilogue."""
    return {
            'ImageProductionStats': {
                'ActualL15CoverageHRV': {
                    'LowerSouthLineActual': 1,
                    'LowerNorthLineActual': 8256,
                    'LowerEastColumnActual': 2877,
                    'LowerWestColumnActual': 8444,
                    'UpperSouthLineActual': 8257,
                    'UpperNorthLineActual': 11136,
                    'UpperEastColumnActual': 1805,
                    'UpperWestColumnActual': 7372
                }
            }
        }


def get_fake_mda(nlines, ncols, start_time):
    """Create fake metadata."""
    nbits = 10
    tline = get_acq_time_cds(start_time, nlines)
    return {
        'number_of_bits_per_pixel': nbits,
        'number_of_lines': nlines,
        'number_of_columns': ncols,
        'data_field_length': nlines * ncols * nbits,
        'cfac': 5,
        'lfac': 5,
        'coff': 10,
        'loff': 10,
        'image_segment_line_quality': {
            'line_mean_acquisition': tline,
            'line_validity': np.full(nlines, 3),
            'line_radiometric_quality': np.full(nlines, 4),
            'line_geometric_quality': np.full(nlines, 4)
        }
    }


def get_fake_filename_info(start_time):
    """Create fake filename information."""
    return {
        'platform_shortname': 'MSG3',
        'start_time': start_time,
        'service': 'MSG'
    }


def get_fake_dataset_info():
    """Create fake dataset info."""
    return {
        'units': 'units',
        'wavelength': 'wavelength',
        'standard_name': 'standard_name'
    }


def get_acq_time_cds(start_time, nlines):
    """Get fake scanline acquisition times."""
    days_since_1958 = (start_time - datetime(1958, 1, 1)).days
    tline = np.zeros(
        nlines,
        dtype=[('days', '>u2'), ('milliseconds', '>u4')]
    )
    tline['days'][1:-1] = days_since_1958 * np.ones(nlines - 2)
    tline['milliseconds'][1:-1] = np.arange(nlines - 2)
    return tline


def get_acq_time_exp(start_time, nlines):
    """Get expected scanline acquisition times."""
    tline_exp = np.zeros(464, dtype='datetime64[ms]')
    tline_exp[0] = np.datetime64('NaT')
    tline_exp[-1] = np.datetime64('NaT')
    tline_exp[1:-1] = np.datetime64(start_time)
    tline_exp[1:-1] += np.arange(nlines - 2).astype('timedelta64[ms]')
    return tline_exp


def get_attrs_exp(projection_longitude=0.0):
    """Get expected dataset attributes."""
    return {
        'units': 'units',
        'wavelength': 'wavelength',
        'standard_name': 'standard_name',
        'platform_name': 'Meteosat-11',
        'sensor': 'seviri',
        'orbital_parameters': {'projection_longitude': projection_longitude,
                               'projection_latitude': 0.,
                               'projection_altitude': 35785831.0,
                               'satellite_nominal_longitude': -3.5,
                               'satellite_nominal_latitude': 0.0,
                               'satellite_actual_longitude': -3.55117540817073,
                               'satellite_actual_latitude': -0.5711243456528018,
                               'satellite_actual_altitude': 35783296.150123544},
        'georef_offset_corrected': True,
        'nominal_start_time': datetime(2006, 1, 1, 12, 15, 9, 304888),
        'nominal_end_time': datetime(2006, 1, 1, 12, 30, 0, 0)
    }
