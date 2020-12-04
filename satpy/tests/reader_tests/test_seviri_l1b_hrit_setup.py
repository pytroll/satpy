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

from satpy.readers.seviri_l1b_hrit import HRITMSGFileHandler


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


def get_fake_file_handler(filename_info, mda, prologue, epilogue):
    """Create a mocked SEVIRI HRIT file handler."""
    m = mock.mock_open()
    with mock.patch('satpy.readers.seviri_l1b_hrit.np.fromfile') as fromfile:
        fromfile.return_value = np.array(
            [(1, 2)],
            dtype=[('total_header_length', int),
                   ('hdr_id', int)]
        )
        with mock.patch('satpy.readers.hrit_base.open', m, create=True) as newopen:
            with mock.patch('satpy.readers.seviri_l1b_hrit.CHANNEL_NAMES'):
                with mock.patch.object(HRITMSGFileHandler, '_get_hd', new=new_get_hd):
                    newopen.return_value.__enter__.return_value.tell.return_value = 1
                    prologue = mock.MagicMock(prologue=prologue)
                    epilogue = mock.MagicMock(epilogue=epilogue)
                    prologue.get_satpos.return_value = None, None, None
                    prologue.get_earth_radii.return_value = None, None

                    reader = HRITMSGFileHandler(
                        'filename',
                        filename_info,
                        {'filetype': 'info'},
                        prologue,
                        epilogue
                    )
                    reader.mda.update(mda)
                    return reader


def get_fake_prologue():
    """Create a fake HRIT prologue."""
    return {
         "SatelliteStatus": {
             "SatelliteDefinition": {
                 "SatelliteId": 324,
                 "NominalLongitude": 47
             }
         },
         'GeometricProcessing': {
             'EarthModel': {
                 'TypeOfEarthModel': 2,
                 'NorthPolarRadius': 10,
                 'SouthPolarRadius': 10,
                 'EquatorialRadius': 10}
         },
         'ImageDescription': {
             'ProjectionDescription': {
                 'LongitudeOfSSP': 0.0
             },
             'Level15ImageProduction': {
                 'ImageProcDirection': 1
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
        'projection_parameters': {
            'a': 6378169.0,
            'b': 6356583.8,
            'h': 35785831.0,
            'SSP_longitude': 44,
            'SSP_latitude': 0.0
        },
        'orbital_parameters': {
            'satellite_nominal_longitude': 47,
            'satellite_nominal_latitude': 0.0,
            'satellite_actual_longitude': 47.5,
            'satellite_actual_latitude': -0.5,
            'satellite_actual_altitude': 35783328
        },
        'image_segment_line_quality': {
            'line_mean_acquisition': tline
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


def get_attrs_exp():
    """Get expected dataset attributes."""
    return {
        'units': 'units',
        'wavelength': 'wavelength',
        'standard_name': 'standard_name',
        'platform_name': 'Meteosat-11',
        'sensor': 'seviri',
        'satellite_longitude': 44,
        'satellite_latitude': 0.0,
        'satellite_altitude': 35785831.0,
        'orbital_parameters': {'projection_longitude': 44,
                               'projection_latitude': 0.,
                               'projection_altitude': 35785831.0,
                               'satellite_nominal_longitude': 47,
                               'satellite_nominal_latitude': 0.0,
                               'satellite_actual_longitude': 47.5,
                               'satellite_actual_latitude': -0.5,
                               'satellite_actual_altitude': 35783328},
        'georef_offset_corrected': True
    }
