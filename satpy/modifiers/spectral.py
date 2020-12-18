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
"""Modifier classes dealing with spectral domain changes or corrections."""

import logging

import xarray as xr

from satpy.modifiers import ModifierBase

try:
    from pyspectral.near_infrared_reflectance import Calculator
except ImportError:
    Calculator = None

try:
    from pyorbital.astronomy import sun_zenith_angle
except ImportError:
    sun_zenith_angle = None


logger = logging.getLogger(__name__)


class NIRReflectance(ModifierBase):
    """Get the reflective part of NIR bands."""

    TERMINATOR_LIMIT = 85.0
    MASKING_LIMIT = 88.0

    def __init__(self, sunz_threshold=TERMINATOR_LIMIT,
                 masking_limit=MASKING_LIMIT, **kwargs):
        """Collect custom configuration values.

        Args:
            sunz_threshold: The threshold sun zenith angle used when deriving
                the near infrared reflectance. Above this angle the derivation
                will assume this sun-zenith everywhere. Unless overridden, the
                default threshold of 85.0 degrees will be used.
            masking_limit: Mask the data (set to NaN) above this Sun zenith angle.
                By default the limit is at 88.0 degrees.  If set to `None`, no masking
                is done.

        """
        self.sun_zenith_threshold = sunz_threshold
        self.masking_limit = masking_limit
        super(NIRReflectance, self).__init__(**kwargs)

    def __call__(self, projectables, optional_datasets=None, **info):
        """Get the reflectance part of an NIR channel.

        Not supposed to be used for wavelength outside [3, 4] µm.
        """
        projectables = self.match_data_arrays(projectables)
        return self._get_reflectance_as_dataarray(projectables, optional_datasets)

    def _get_reflectance_as_dataarray(self, projectables, optional_datasets):
        """Get the reflectance as a dataarray."""
        _nir, _tb11 = projectables
        da_nir = _nir.data
        da_tb11 = _tb11.data
        da_tb13_4 = self._get_tb13_4_from_optionals(optional_datasets)
        da_sun_zenith = self._get_sun_zenith_from_provided_data(projectables, optional_datasets)

        logger.info('Getting reflective part of %s', _nir.attrs['name'])
        reflectance = self._get_reflectance_as_dask(da_nir, da_tb11, da_tb13_4, da_sun_zenith, _nir.attrs)

        proj = self._create_modified_dataarray(reflectance, base_dataarray=_nir)
        proj.attrs['units'] = '%'
        return proj

    @staticmethod
    def _get_tb13_4_from_optionals(optional_datasets):
        tb13_4 = None
        for dataset in optional_datasets:
            wavelengths = dataset.attrs.get('wavelength', [100., 0, 0])
            if (dataset.attrs.get('units') == 'K' and
                    wavelengths[0] <= 13.4 <= wavelengths[2]):
                tb13_4 = dataset.data
        return tb13_4

    @staticmethod
    def _get_sun_zenith_from_provided_data(projectables, optional_datasets):
        """Get the sunz from available data or compute it if unavailable."""
        sun_zenith = None

        for dataset in optional_datasets:
            if dataset.attrs.get("standard_name") == "solar_zenith_angle":
                sun_zenith = dataset.data

        if sun_zenith is None:
            if sun_zenith_angle is None:
                raise ImportError("Module pyorbital.astronomy needed to compute sun zenith angles.")
            _nir = projectables[0]
            lons, lats = _nir.attrs["area"].get_lonlats(chunks=_nir.data.chunks)
            sun_zenith = sun_zenith_angle(_nir.attrs['start_time'], lons, lats)
        return sun_zenith

    def _create_modified_dataarray(self, reflectance, base_dataarray):
        proj = xr.DataArray(reflectance, dims=base_dataarray.dims,
                            coords=base_dataarray.coords, attrs=base_dataarray.attrs.copy())
        proj.attrs['sun_zenith_threshold'] = self.sun_zenith_threshold
        proj.attrs['sun_zenith_masking_limit'] = self.masking_limit
        self.apply_modifier_info(base_dataarray, proj)
        return proj

    def _get_reflectance_as_dask(self, da_nir, da_tb11, da_tb13_4, da_sun_zenith, metadata):
        """Calculate 3.x reflectance in % with pyspectral from dask arrays."""
        reflectance_3x_calculator = self._init_reflectance_calculator(metadata)
        return reflectance_3x_calculator.reflectance_from_tbs(da_sun_zenith, da_nir, da_tb11, tb_ir_co2=da_tb13_4) * 100

    def _init_reflectance_calculator(self, metadata):
        """Initialize the 3.x reflectance derivations."""
        if not Calculator:
            logger.info("Couldn't load pyspectral")
            raise ImportError("No module named pyspectral.near_infrared_reflectance")

        reflectance_3x_calculator = Calculator(metadata['platform_name'], metadata['sensor'], metadata['name'],
                                               sunz_threshold=self.sun_zenith_threshold,
                                               masking_limit=self.masking_limit)
        return reflectance_3x_calculator


class NIREmissivePartFromReflectance(NIRReflectance):
    """Get the emissive part of NIR bands."""

    def __init__(self, sunz_threshold=None, **kwargs):
        """Collect custom configuration values.

        Args:
            sunz_threshold: The threshold sun zenith angle used when deriving
                the near infrared reflectance. Above this angle the derivation
                will assume this sun-zenith everywhere. Default None, in which
                case the default threshold defined in Pyspectral will be used.

        """
        self.sunz_threshold = sunz_threshold
        super(NIREmissivePartFromReflectance, self).__init__(sunz_threshold=sunz_threshold, **kwargs)

    def __call__(self, projectables, optional_datasets=None, **info):
        """Get the emissive part an NIR channel after having derived the reflectance.

        Not supposed to be used for wavelength outside [3, 4] µm.

        """
        projectables = self.match_data_arrays(projectables)
        return self._get_emissivity_as_dataarray(projectables, optional_datasets)

    def _get_emissivity_as_dataarray(self, projectables, optional_datasets):
        """Get the emissivity as a dataarray."""
        _nir, _tb11 = projectables
        da_nir = _nir.data
        da_tb11 = _tb11.data
        da_tb13_4 = self._get_tb13_4_from_optionals(optional_datasets)
        da_sun_zenith = self._get_sun_zenith_from_provided_data(projectables, optional_datasets)

        logger.info('Getting emissive part of %s', _nir.attrs['name'])
        emissivity = self._get_emissivity_as_dask(da_nir, da_tb11, da_tb13_4, da_sun_zenith, _nir.attrs)

        proj = self._create_modified_dataarray(emissivity, base_dataarray=_nir)
        proj.attrs['units'] = 'K'
        return proj

    def _get_emissivity_as_dask(self, da_nir, da_tb11, da_tb13_4, da_sun_zenith, metadata):
        """Get the emissivity from pyspectral."""
        reflectance_3x_calculator = self._init_reflectance_calculator(metadata)
        # Use the nir and thermal ir brightness temperatures and derive the reflectance using
        # PySpectral. The reflectance is stored internally in PySpectral and
        # needs to be derived first in order to get the emissive part.
        reflectance_3x_calculator.reflectance_from_tbs(da_sun_zenith, da_nir, da_tb11, tb_ir_co2=da_tb13_4)
        return reflectance_3x_calculator.emissive_part_3x()
