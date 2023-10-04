#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
"""Unit tests for GERB L2 HR HDF5 reader."""

import h5py
import numpy as np
import pytest

from satpy import Scene

FNAME = "G4_SEV4_L20_HR_SOL_TH_20190606_130000_V000.hdf"


def make_h5_null_string(length):
    """Make a HDF5 type for a NULL terminated string of fixed length."""
    dt = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
    dt.set_size(7)
    dt.set_strpad(h5py.h5t.STR_NULLTERM)
    return dt


def write_h5_null_string_att(loc_id, name, s):
    """Write a NULL terminated string attribute at loc_id."""
    dt = make_h5_null_string(length=7)
    name = bytes(name.encode('ascii'))
    s = bytes(s.encode('ascii'))
    at = h5py.h5a.create(loc_id, name, dt, h5py.h5s.create(h5py.h5s.SCALAR))
    at.write(np.array(s, dtype=f'|S{len(s)+1}'))


@pytest.fixture(scope="session")
def gerb_l2_hr_h5_dummy_file(tmp_path_factory):
    """Create a dummy HDF5 file for the GERB L2 HR product."""
    filename = tmp_path_factory.mktemp("data") / FNAME

    with h5py.File(filename, 'w') as fid:
        fid.create_group('/Angles')
        fid['/Angles/Relative Azimuth'] = np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Angles/Relative Azimuth'].attrs['Quantisation Factor'] = np.array(0.1, dtype='float64')
        fid['/Angles/Solar Zenith'] = np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Angles/Solar Zenith'].attrs['Quantisation Factor'] = np.array(0.1, dtype='float64')
        write_h5_null_string_att(fid['/Angles/Relative Azimuth'].id, 'Unit', 'Degree')
        fid['/Angles/Viewing Azimuth'] = np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Angles/Viewing Azimuth'].attrs['Quantisation Factor'] = np.array(0.1, dtype='float64')
        write_h5_null_string_att(fid['/Angles/Viewing Azimuth'].id, 'Unit', 'Degree')
        fid['/Angles/Viewing Zenith'] = np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Angles/Viewing Zenith'].attrs['Quantisation Factor'] = np.array(0.1, dtype='float64')
        write_h5_null_string_att(fid['/Angles/Viewing Zenith'].id, 'Unit', 'Degree')
        fid.create_group('/GERB')
        dt = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
        dt.set_size(3)
        dt.set_strpad(h5py.h5t.STR_NULLTERM)
        write_h5_null_string_att(fid['/GERB'].id, 'Instrument Identifier', 'G4')
        fid.create_group('/GGSPS')
        fid['/GGSPS'].attrs['L1.5 NANRG Product Version'] = np.array(-1, dtype='int32')
        fid.create_group('/Geolocation')
        write_h5_null_string_att(fid['/Geolocation'].id, 'Geolocation File Name',
                                 'G4_SEV4_L20_HR_GEO_20180111_181500_V010.hdf')
        fid['/Geolocation'].attrs['Nominal Satellite Longitude (degrees)'] = np.array(0.0, dtype='float64')
        fid.create_group('/Imager')
        fid['/Imager'].attrs['Instrument Identifier'] = np.array(4, dtype='int32')
        write_h5_null_string_att(fid['/Imager'].id, 'Type', 'SEVIRI')
        fid.create_group('/RMIB')
        fid.create_group('/Radiometry')
        fid['/Radiometry'].attrs['SEVIRI Radiance Definition Flag'] = np.array(2, dtype='int32')
        fid['/Radiometry/A Values (per GERB detector cell)'] = np.ones(shape=(256,), dtype=np.dtype('>f8'))
        fid['/Radiometry/C Values (per GERB detector cell)'] = np.ones(shape=(256,), dtype=np.dtype('>f8'))
        fid['/Radiometry/Longwave Correction'] = np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Radiometry/Longwave Correction'].attrs['Offset'] = np.array(1.0, dtype='float64')
        fid['/Radiometry/Longwave Correction'].attrs['Quantisation Factor'] = np.array(0.005, dtype='float64')
        fid['/Radiometry/Shortwave Correction'] = np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Radiometry/Shortwave Correction'].attrs['Offset'] = np.array(1.0, dtype='float64')
        fid['/Radiometry/Shortwave Correction'].attrs['Quantisation Factor'] = np.array(0.005, dtype='float64')
        fid['/Radiometry/Solar Flux'] = np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Radiometry/Solar Flux'].attrs['Quantisation Factor'] = np.array(0.25, dtype='float64')
        write_h5_null_string_att(fid['/Radiometry/Solar Flux'].id, 'Unit', 'Watt per square meter')
        fid['/Radiometry/Solar Radiance'] = np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Radiometry/Solar Radiance'].attrs['Quantisation Factor'] = np.array(0.05, dtype='float64')
        write_h5_null_string_att(fid['/Radiometry/Solar Radiance'].id, 'Unit', 'Watt per square meter per steradian')
        fid['/Radiometry/Thermal Flux'] = np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Radiometry/Thermal Flux'].attrs['Quantisation Factor'] = np.array(0.25, dtype='float64')
        write_h5_null_string_att(fid['/Radiometry/Thermal Flux'].id, 'Unit', 'Watt per square meter')
        fid['/Radiometry/Thermal Radiance'] = np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Radiometry/Thermal Radiance'].attrs['Quantisation Factor'] = np.array(0.05, dtype='float64')
        write_h5_null_string_att(fid['/Radiometry/Thermal Radiance'].id, 'Unit', 'Watt per square meter per steradian')
        fid.create_group('/Scene Identification')
        write_h5_null_string_att(fid['/Scene Identification'].id,
                                 'Solar Angular Dependency Models Set Version', 'CERES_TRMM.1')
        write_h5_null_string_att(fid['/Scene Identification'].id,
                                 'Thermal Angular Dependency Models Set Version', 'RMIB.3')
        fid['/Scene Identification/Cloud Cover'] = np.ones(shape=(1237, 1237), dtype=np.dtype('uint8'))
        fid['/Scene Identification/Cloud Cover'].attrs['Quantisation Factor'] = np.array(0.01, dtype='float64')
        write_h5_null_string_att(fid['/Scene Identification/Cloud Cover'].id, 'Unit', 'Percent')
        fid['/Scene Identification/Cloud Optical Depth (logarithm)'] = \
            np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Scene Identification/Cloud Optical Depth (logarithm)'].attrs['Quantisation Factor'] = \
            np.array(0.00025, dtype='float64')
        fid['/Scene Identification/Cloud Phase'] = np.ones(shape=(1237, 1237), dtype=np.dtype('uint8'))
        fid['/Scene Identification/Cloud Phase'].attrs['Quantisation Factor'] = np.array(0.01, dtype='float64')
        write_h5_null_string_att(fid['/Scene Identification/Cloud Phase'].id, 'Unit',
                                 'Percent (Water=0%,Mixed,Ice=100%)')
        fid.create_group('/Times')
        fid['/Times/Time (per row)'] = np.ones(shape=(1237,), dtype=np.dtype('|S22'))

    return filename


@pytest.mark.parametrize("name", ["Solar Flux", "Thermal Flux", "Solar Radiance", "Thermal Radiance"])
def test_dataset_load(gerb_l2_hr_h5_dummy_file, name):
    """Test loading the solar flux component."""
    scene = Scene(reader='gerb_l2_hr_h5', filenames=[gerb_l2_hr_h5_dummy_file])
    scene.load([name])
    assert scene[name].shape == (1237, 1237)
    assert np.nanmax((scene[name].to_numpy().flatten() - 0.25)) < 1e-6
