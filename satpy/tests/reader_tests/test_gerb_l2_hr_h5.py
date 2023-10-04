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

import numpy as np
import pytest
import h5py
from satpy import Scene

FNAME = "G4_SEV4_L20_HR_SOL_TH_20190606_130000_V000.hdf"


@pytest.fixture(scope="session")
def gerb_l2_hr_h5_dummy_file(tmp_path_factory):
    """Create a dummy HDF5 file for the GERB L2 HR product."""
    filename = tmp_path_factory.mktemp("data") / FNAME

    with h5py.File(filename, 'w') as fid:
        fid.create_group('/Angles')
        fid['/Angles/Relative Azimuth'] = np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Angles/Relative Azimuth'].attrs['Quantisation Factor'] = np.array(0.1, dtype='float64')
        dt = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
        dt.set_size(7)
        dt.set_strpad(h5py.h5t.STR_NULLTERM)
        at = h5py.h5a.create(fid['/Angles/Relative Azimuth'].id, b'Unit', dt, h5py.h5s.create(h5py.h5s.SCALAR))
        at.write(np.array(b'Degree', dtype='|S7'))
        fid['/Angles/Solar Zenith'] = np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Angles/Solar Zenith'].attrs['Quantisation Factor'] = np.array(0.1, dtype='float64')
        dt = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
        dt.set_size(7)
        dt.set_strpad(h5py.h5t.STR_NULLTERM)
        at = h5py.h5a.create(fid['/Angles/Solar Zenith'].id, b'Unit', dt, h5py.h5s.create(h5py.h5s.SCALAR))
        at.write(np.array(b'Degree', dtype='|S7'))
        fid['/Angles/Viewing Azimuth'] = np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Angles/Viewing Azimuth'].attrs['Quantisation Factor'] = np.array(0.1, dtype='float64')
        dt = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
        dt.set_size(7)
        dt.set_strpad(h5py.h5t.STR_NULLTERM)
        at = h5py.h5a.create(fid['/Angles/Viewing Azimuth'].id, b'Unit', dt, h5py.h5s.create(h5py.h5s.SCALAR))
        at.write(np.array(b'Degree', dtype='|S7'))
        fid['/Angles/Viewing Zenith'] = np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Angles/Viewing Zenith'].attrs['Quantisation Factor'] = np.array(0.1, dtype='float64')
        dt = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
        dt.set_size(7)
        dt.set_strpad(h5py.h5t.STR_NULLTERM)
        at = h5py.h5a.create(fid['/Angles/Viewing Zenith'].id, b'Unit', dt, h5py.h5s.create(h5py.h5s.SCALAR))
        at.write(np.array(b'Degree', dtype='|S7'))
        fid.create_group('/GERB')
        dt = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
        dt.set_size(3)
        dt.set_strpad(h5py.h5t.STR_NULLTERM)
        at = h5py.h5a.create(fid['/GERB'].id, b'Instrument Identifier', dt, h5py.h5s.create(h5py.h5s.SCALAR))
        at.write(np.array(b'G4', dtype='|S3'))
        fid.create_group('/GGSPS')
        fid['/GGSPS'].attrs['L1.5 NANRG Product Version'] = np.array(-1, dtype='int32')
        fid.create_group('/Geolocation')
        dt = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
        dt.set_size(44)
        dt.set_strpad(h5py.h5t.STR_NULLTERM)
        at = h5py.h5a.create(fid['/Geolocation'].id, b'Geolocation File Name', dt, h5py.h5s.create(h5py.h5s.SCALAR))
        at.write(np.array(b'G4_SEV4_L20_HR_GEO_20180111_181500_V010.hdf', dtype='|S44'))
        fid['/Geolocation'].attrs['Line of Sight North-South Speed'] = np.array(0.0, dtype='float64')
        fid['/Geolocation'].attrs['Nominal Satellite Longitude (degrees)'] = np.array(0.0, dtype='float64')
        fid.create_group('/Geolocation/Rectified Grid')
        fid['/Geolocation/Rectified Grid'].attrs['Grid Orientation'] = np.array(0.0, dtype='float64')
        fid['/Geolocation/Rectified Grid'].attrs['Lap'] = np.array(0.0, dtype='float64')
        fid['/Geolocation/Rectified Grid'].attrs['Lop'] = np.array(0.0, dtype='float64')
        fid['/Geolocation/Rectified Grid'].attrs['Nr'] = np.array(6.610674630916804, dtype='float64')
        fid['/Geolocation/Rectified Grid'].attrs['Nx'] = np.array(1237, dtype='int32')
        fid['/Geolocation/Rectified Grid'].attrs['Ny'] = np.array(1237, dtype='int32')
        fid['/Geolocation/Rectified Grid'].attrs['Xp'] = np.array(618.3333333333334, dtype='float64')
        fid['/Geolocation/Rectified Grid'].attrs['Yp'] = np.array(617.6666666666666, dtype='float64')
        fid['/Geolocation/Rectified Grid'].attrs['dx'] = np.array(1207.4379446281002, dtype='float64')
        fid['/Geolocation/Rectified Grid'].attrs['dy'] = np.array(1203.3201568249945, dtype='float64')
        fid.create_group('/Geolocation/Rectified Grid/Resolution Flags')
        fid['/Geolocation/Rectified Grid/Resolution Flags'].attrs['East West'] = np.array(0.014411607, dtype='float64')
        fid['/Geolocation/Rectified Grid/Resolution Flags'].attrs['North South'] = \
            np.array(0.014411607, dtype='float64')
        fid.create_group('/Imager')
        fid['/Imager'].attrs['Instrument Identifier'] = np.array(4, dtype='int32')
        dt = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
        dt.set_size(7)
        dt.set_strpad(h5py.h5t.STR_NULLTERM)
        at = h5py.h5a.create(fid['/Imager'].id, b'Type', dt, h5py.h5s.create(h5py.h5s.SCALAR))
        at.write(np.array(b'SEVIRI', dtype='|S7'))
        fid.create_group('/RMIB')
        fid['/RMIB'].attrs['Product Version'] = np.array(10, dtype='int32')
        dt = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
        dt.set_size(16)
        dt.set_strpad(h5py.h5t.STR_NULLTERM)
        at = h5py.h5a.create(fid['/RMIB'].id, b'Software Identifier', dt, h5py.h5s.create(h5py.h5s.SCALAR))
        at.write(np.array(b'20220812_151631', dtype='|S16'))
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
        dt = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
        dt.set_size(22)
        dt.set_strpad(h5py.h5t.STR_NULLTERM)
        at = h5py.h5a.create(fid['/Radiometry/Solar Flux'].id, b'Unit', dt, h5py.h5s.create(h5py.h5s.SCALAR))
        at.write(np.array(b'Watt per square meter', dtype='|S22'))
        fid['/Radiometry/Solar Radiance'] = np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Radiometry/Solar Radiance'].attrs['Quantisation Factor'] = np.array(0.05, dtype='float64')
        dt = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
        dt.set_size(36)
        dt.set_strpad(h5py.h5t.STR_NULLTERM)
        at = h5py.h5a.create(fid['/Radiometry/Solar Radiance'].id, b'Unit', dt, h5py.h5s.create(h5py.h5s.SCALAR))
        at.write(np.array(b'Watt per square meter per steradian', dtype='|S36'))
        fid['/Radiometry/Thermal Flux'] = np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Radiometry/Thermal Flux'].attrs['Quantisation Factor'] = np.array(0.25, dtype='float64')
        dt = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
        dt.set_size(22)
        dt.set_strpad(h5py.h5t.STR_NULLTERM)
        at = h5py.h5a.create(fid['/Radiometry/Thermal Flux'].id, b'Unit', dt, h5py.h5s.create(h5py.h5s.SCALAR))
        at.write(np.array(b'Watt per square meter', dtype='|S22'))
        fid['/Radiometry/Thermal Radiance'] = np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Radiometry/Thermal Radiance'].attrs['Quantisation Factor'] = np.array(0.05, dtype='float64')
        dt = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
        dt.set_size(36)
        dt.set_strpad(h5py.h5t.STR_NULLTERM)
        at = h5py.h5a.create(fid['/Radiometry/Thermal Radiance'].id, b'Unit', dt, h5py.h5s.create(h5py.h5s.SCALAR))
        at.write(np.array(b'Watt per square meter per steradian', dtype='|S36'))
        fid.create_group('/Scene Identification')
        dt = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
        dt.set_size(13)
        dt.set_strpad(h5py.h5t.STR_NULLTERM)
        at = h5py.h5a.create(fid['/Scene Identification'].id, b'Solar Angular Dependency Models Set Version', dt,
                             h5py.h5s.create(h5py.h5s.SCALAR))
        at.write(np.array(b'CERES_TRMM.1', dtype='|S13'))
        dt = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
        dt.set_size(7)
        dt.set_strpad(h5py.h5t.STR_NULLTERM)
        at = h5py.h5a.create(fid['/Scene Identification'].id, b'Thermal Angular Dependency Models Set Version', dt,
                             h5py.h5s.create(h5py.h5s.SCALAR))
        at.write(np.array(b'RMIB.3', dtype='|S7'))
        fid['/Scene Identification/Aerosol Optical Depth IR 1.6'] = np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Scene Identification/Aerosol Optical Depth IR 1.6'].attrs['Quantisation Factor'] = \
            np.array(0.001, dtype='float64')
        fid['/Scene Identification/Aerosol Optical Depth VIS 0.6'] = np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Scene Identification/Aerosol Optical Depth VIS 0.6'].attrs['Quantisation Factor'] = \
            np.array(0.001, dtype='float64')
        fid['/Scene Identification/Aerosol Optical Depth VIS 0.8'] = np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Scene Identification/Aerosol Optical Depth VIS 0.8'].attrs['Quantisation Factor'] = \
            np.array(0.001, dtype='float64')
        fid['/Scene Identification/Cloud Cover'] = np.ones(shape=(1237, 1237), dtype=np.dtype('uint8'))
        fid['/Scene Identification/Cloud Cover'].attrs['Quantisation Factor'] = np.array(0.01, dtype='float64')
        dt = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
        dt.set_size(8)
        dt.set_strpad(h5py.h5t.STR_NULLTERM)
        at = h5py.h5a.create(fid['/Scene Identification/Cloud Cover'].id, b'Unit', dt, h5py.h5s.create(h5py.h5s.SCALAR))
        at.write(np.array(b'Percent', dtype='|S8'))
        fid['/Scene Identification/Cloud Optical Depth (logarithm)'] = \
            np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Scene Identification/Cloud Optical Depth (logarithm)'].attrs['Quantisation Factor'] = \
            np.array(0.00025, dtype='float64')
        fid['/Scene Identification/Cloud Phase'] = np.ones(shape=(1237, 1237), dtype=np.dtype('uint8'))
        fid['/Scene Identification/Cloud Phase'].attrs['Quantisation Factor'] = np.array(0.01, dtype='float64')
        dt = h5py.h5t.TypeID.copy(h5py.h5t.C_S1)
        dt.set_size(34)
        dt.set_strpad(h5py.h5t.STR_NULLTERM)
        at = h5py.h5a.create(fid['/Scene Identification/Cloud Phase'].id, b'Unit', dt, h5py.h5s.create(h5py.h5s.SCALAR))
        at.write(np.array(b'Percent (Water=0%,Mixed,Ice=100%)', dtype='|S34'))
        fid['/Scene Identification/Dust Detection'] = np.ones(shape=(1237, 1237), dtype=np.dtype('uint8'))
        fid['/Scene Identification/Dust Detection'].attrs['Quantisation Factor'] = np.array(0.01, dtype='float64')
        fid['/Scene Identification/Solar Angular Dependency Model'] = np.ones(shape=(1237, 1237), dtype=np.dtype('>i2'))
        fid['/Scene Identification/Surface Type'] = np.ones(shape=(1237, 1237), dtype=np.dtype('uint8'))
        fid['/Scene Identification/Thermal Angular Dependency Model'] = \
            np.ones(shape=(1237, 1237), dtype=np.dtype('uint8'))
        fid.create_group('/Times')
        fid['/Times/Time (per row)'] = np.ones(shape=(1237,), dtype=np.dtype('|S22'))

    return filename


def test_gerb_solar_flux_dataset(gerb_l2_hr_h5_dummy_file):
    """Test the GERB L2 HR HDF5 file.

    Load the solar flux component.
    """
    scene = Scene(reader='gerb_l2_hr_h5', filenames=[gerb_l2_hr_h5_dummy_file])
    scene.load(['Solar Flux'])
    assert scene['Solar Flux'].shape == (1237, 1237)
    assert np.nanmax((scene['Solar Flux'].to_numpy().flatten() - 0.25)) < 1e-6


def test_gerb_thermal_flux_dataset(gerb_l2_hr_h5_dummy_file):
    """Test the GERB L2 HR HDF5 file.

    Load the thermal flux component.
    """
    scene = Scene(reader='gerb_l2_hr_h5', filenames=[gerb_l2_hr_h5_dummy_file])
    scene.load(['Thermal Flux'])
    assert scene['Thermal Flux'].shape == (1237, 1237)
    assert np.nanmax((scene['Thermal Flux'].to_numpy().flatten() - 0.25)) < 1e-6
