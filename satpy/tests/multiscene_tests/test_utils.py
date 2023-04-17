#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018-2023 Satpy developers
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
"""Utilties to assist testing the Multiscene functionality.

Creating fake test data for use in the other Multiscene test modules.
"""


import dask.array as da
import numpy as np
import xarray as xr
from pyresample.geometry import AreaDefinition

from satpy.dataset.dataid import ModifierTuple, WavelengthRange

DEFAULT_SHAPE = (5, 10)

local_id_keys_config = {'name': {
    'required': True,
},
    'wavelength': {
    'type': WavelengthRange,
},
    'resolution': None,
    'calibration': {
    'enum': [
        'reflectance',
        'brightness_temperature',
        'radiance',
        'counts'
    ]
},
    'polarization': None,
    'level': None,
    'modifiers': {
    'required': True,
    'default': ModifierTuple(),
    'type': ModifierTuple,
},
}


def _fake_get_enhanced_image(img, enhance=None, overlay=None, decorate=None):
    from trollimage.xrimage import XRImage
    return XRImage(img)


def _create_test_area(proj_str=None, shape=DEFAULT_SHAPE, extents=None):
    """Create a test area definition."""
    from pyresample.utils import proj4_str_to_dict
    if proj_str is None:
        proj_str = '+proj=lcc +datum=WGS84 +ellps=WGS84 +lon_0=-95. ' \
                   '+lat_0=25 +lat_1=25 +units=m +no_defs'
    proj_dict = proj4_str_to_dict(proj_str)
    extents = extents or (-1000., -1500., 1000., 1500.)

    return AreaDefinition(
        'test',
        'test',
        'test',
        proj_dict,
        shape[1],
        shape[0],
        extents
    )


def _create_test_int8_dataset(name, shape=DEFAULT_SHAPE, area=None, values=None):
    """Create a test DataArray object."""
    return xr.DataArray(
        da.ones(shape, dtype=np.uint8, chunks=shape) * values, dims=('y', 'x'),
        attrs={'_FillValue': 255,
               'valid_range': [1, 15],
               'name': name, 'area': area, '_satpy_id_keys': local_id_keys_config})


def _create_test_dataset(name, shape=DEFAULT_SHAPE, area=None, values=None):
    """Create a test DataArray object."""
    if values:
        return xr.DataArray(
            da.ones(shape, dtype=np.float32, chunks=shape) * values, dims=('y', 'x'),
            attrs={'name': name, 'area': area, '_satpy_id_keys': local_id_keys_config})

    return xr.DataArray(
        da.zeros(shape, dtype=np.float32, chunks=shape), dims=('y', 'x'),
        attrs={'name': name, 'area': area, '_satpy_id_keys': local_id_keys_config})


def _create_test_scenes(num_scenes=2, shape=DEFAULT_SHAPE, area=None):
    """Create some test scenes for various test cases."""
    from satpy import Scene
    ds1 = _create_test_dataset('ds1', shape=shape, area=area)
    ds2 = _create_test_dataset('ds2', shape=shape, area=area)
    scenes = []
    for _ in range(num_scenes):
        scn = Scene()
        scn['ds1'] = ds1.copy()
        scn['ds2'] = ds2.copy()
        scenes.append(scn)
    return scenes
