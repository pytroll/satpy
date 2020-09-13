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
"""Modifier base classes and other utilities."""
import time
import logging
from weakref import WeakValueDictionary

import numpy as np
import xarray as xr
from dask import array as da

from satpy.composites import CompositeBase
from satpy.utils import sunzen_corr_cos, atmospheric_path_length_correction, get_satpos

try:
    from pyspectral.near_infrared_reflectance import Calculator
except ImportError:
    Calculator = None

try:
    from pyorbital.astronomy import sun_zenith_angle
except ImportError:
    sun_zenith_angle = None

logger = logging.getLogger(__name__)


class ModifierBase(CompositeBase):
    """Base class for all modifiers.

    A modifier in Satpy is a class that takes one input DataArray to be
    changed along with zero or more other input DataArrays used to perform
    these changes. The result of a modifier typically has a lot of the same
    metadata (name, units, etc) as the original DataArray, but the data is
    different. A modified DataArray can be differentiated from the original
    DataArray by the `modifiers` property of its `DataID`.

    See the :class:`~satpy.composites.CompositeBase` class for information
    on the similar concept of "compositors".

    """

    def __init__(self, name, prerequisites=None, optional_prerequisites=None, **kwargs):
        """Initialise the compositor."""
        # Required info
        kwargs["name"] = name
        kwargs["prerequisites"] = prerequisites or []
        kwargs["optional_prerequisites"] = optional_prerequisites or []
        self.attrs = kwargs

    def __call__(self, datasets, optional_datasets=None, **info):
        """Generate a modified copy of the first provided dataset."""
        raise NotImplementedError()


class SunZenithCorrectorBase(ModifierBase):
    """Base class for sun zenith correction modifiers."""

    coszen = WeakValueDictionary()

    def __init__(self, max_sza=95.0, **kwargs):
        """Collect custom configuration values.

        Args:
            max_sza (float): Maximum solar zenith angle in degrees that is
                considered valid and correctable. Default 95.0.

        """
        self.max_sza = max_sza
        self.max_sza_cos = np.cos(np.deg2rad(max_sza)) if max_sza is not None else None
        super(SunZenithCorrectorBase, self).__init__(**kwargs)

    def __call__(self, projectables, **info):
        """Generate the composite."""
        projectables = self.match_data_arrays(list(projectables) + list(info.get('optional_datasets', [])))
        vis = projectables[0]
        if vis.attrs.get("sunz_corrected"):
            logger.debug("Sun zen correction already applied")
            return vis

        area_name = hash(vis.attrs['area'])
        key = (vis.attrs["start_time"], area_name)
        tic = time.time()
        logger.debug("Applying sun zen correction")
        coszen = self.coszen.get(key)
        if coszen is None and not info.get('optional_datasets'):
            # we were not given SZA, generate SZA then calculate cos(SZA)
            from pyorbital.astronomy import cos_zen
            logger.debug("Computing sun zenith angles.")
            lons, lats = vis.attrs["area"].get_lonlats(chunks=vis.data.chunks)

            coords = {}
            if 'y' in vis.coords and 'x' in vis.coords:
                coords['y'] = vis['y']
                coords['x'] = vis['x']
            coszen = xr.DataArray(cos_zen(vis.attrs["start_time"], lons, lats),
                                  dims=['y', 'x'], coords=coords)
            if self.max_sza is not None:
                coszen = coszen.where(coszen >= self.max_sza_cos)
            self.coszen[key] = coszen
        elif coszen is None:
            # we were given the SZA, calculate the cos(SZA)
            coszen = np.cos(np.deg2rad(projectables[1]))
            self.coszen[key] = coszen

        proj = self._apply_correction(vis, coszen)
        proj.attrs = vis.attrs.copy()
        self.apply_modifier_info(vis, proj)
        logger.debug("Sun-zenith correction applied. Computation time: %5.1f (sec)", time.time() - tic)
        return proj

    def _apply_correction(self, proj, coszen):
        raise NotImplementedError("Correction method shall be defined!")


class SunZenithCorrector(SunZenithCorrectorBase):
    """Standard sun zenith correction using ``1 / cos(sunz)``.

    In addition to adjusting the provided reflectances by the cosine of the
    solar zenith angle, this modifier forces all reflectances beyond a
    solar zenith angle of ``max_sza`` to 0. It also gradually reduces the
    amount of correction done between ``correction_limit`` and ``max_sza``. If
    ``max_sza`` is ``None`` then a constant correction is applied to zenith
    angles beyond ``correction_limit``.

    To set ``max_sza`` to ``None`` in a YAML configuration file use:

    .. code-block:: yaml

      sunz_corrected:
        compositor: !!python/name:satpy.composites.SunZenithCorrector
        max_sza: !!null
        optional_prerequisites:
        - solar_zenith_angle

    """

    def __init__(self, correction_limit=88., **kwargs):
        """Collect custom configuration values.

        Args:
            correction_limit (float): Maximum solar zenith angle to apply the
                correction in degrees. Pixels beyond this limit have a
                constant correction applied. Default 88.
            max_sza (float): Maximum solar zenith angle in degrees that is
                considered valid and correctable. Default 95.0.

        """
        self.correction_limit = correction_limit
        super(SunZenithCorrector, self).__init__(**kwargs)

    def _apply_correction(self, proj, coszen):
        logger.debug("Apply the standard sun-zenith correction [1/cos(sunz)]")
        return sunzen_corr_cos(proj, coszen, limit=self.correction_limit, max_sza=self.max_sza)


class EffectiveSolarPathLengthCorrector(SunZenithCorrectorBase):
    """Special sun zenith correction with the method proposed by Li and Shibata.

    (2006): https://doi.org/10.1175/JAS3682.1

    In addition to adjusting the provided reflectances by the cosine of the
    solar zenith angle, this modifier forces all reflectances beyond a
    solar zenith angle of `max_sza` to 0 to reduce noise in the final data.
    It also gradually reduces the amount of correction done between
    ``correction_limit`` and ``max_sza``. If ``max_sza`` is ``None`` then a
    constant correction is applied to zenith angles beyond
    ``correction_limit``.

    To set ``max_sza`` to ``None`` in a YAML configuration file use:

    .. code-block:: yaml

      effective_solar_pathlength_corrected:
        compositor: !!python/name:satpy.composites.EffectiveSolarPathLengthCorrector
        max_sza: !!null
        optional_prerequisites:
        - solar_zenith_angle

    """

    def __init__(self, correction_limit=88., **kwargs):
        """Collect custom configuration values.

        Args:
            correction_limit (float): Maximum solar zenith angle to apply the
                correction in degrees. Pixels beyond this limit have a
                constant correction applied. Default 88.
            max_sza (float): Maximum solar zenith angle in degrees that is
                considered valid and correctable. Default 95.0.

        """
        self.correction_limit = correction_limit
        super(EffectiveSolarPathLengthCorrector, self).__init__(**kwargs)

    def _apply_correction(self, proj, coszen):
        logger.debug("Apply the effective solar atmospheric path length correction method by Li and Shibata")
        return atmospheric_path_length_correction(proj, coszen, limit=self.correction_limit, max_sza=self.max_sza)


class PSPRayleighReflectance(ModifierBase):
    """Pyspectral-based rayleigh corrector for visible channels."""

    _rayleigh_cache = WeakValueDictionary()

    def get_angles(self, vis):
        """Get the sun and satellite angles from the current dataarray."""
        from pyorbital.astronomy import get_alt_az, sun_zenith_angle
        from pyorbital.orbital import get_observer_look

        lons, lats = vis.attrs['area'].get_lonlats(chunks=vis.data.chunks)
        lons = da.where(lons >= 1e30, np.nan, lons)
        lats = da.where(lats >= 1e30, np.nan, lats)
        sunalt, suna = get_alt_az(vis.attrs['start_time'], lons, lats)
        suna = np.rad2deg(suna)
        sunz = sun_zenith_angle(vis.attrs['start_time'], lons, lats)

        sat_lon, sat_lat, sat_alt = get_satpos(vis)
        sata, satel = get_observer_look(
            sat_lon,
            sat_lat,
            sat_alt / 1000.0,  # km
            vis.attrs['start_time'],
            lons, lats, 0)
        satz = 90 - satel
        return sata, satz, suna, sunz

    def __call__(self, projectables, optional_datasets=None, **info):
        """Get the corrected reflectance when removing Rayleigh scattering.

        Uses pyspectral.
        """
        from pyspectral.rayleigh import Rayleigh
        if not optional_datasets or len(optional_datasets) != 4:
            vis, red = self.match_data_arrays(projectables)
            sata, satz, suna, sunz = self.get_angles(vis)
            red.data = da.rechunk(red.data, vis.data.chunks)
        else:
            vis, red, sata, satz, suna, sunz = self.match_data_arrays(
                projectables + optional_datasets)
            sata, satz, suna, sunz = optional_datasets
            # get the dask array underneath
            sata = sata.data
            satz = satz.data
            suna = suna.data
            sunz = sunz.data

        # First make sure the two azimuth angles are in the range 0-360:
        sata = sata % 360.
        suna = suna % 360.
        ssadiff = da.absolute(suna - sata)
        ssadiff = da.minimum(ssadiff, 360 - ssadiff)
        del sata, suna

        atmosphere = self.attrs.get('atmosphere', 'us-standard')
        aerosol_type = self.attrs.get('aerosol_type', 'marine_clean_aerosol')
        rayleigh_key = (vis.attrs['platform_name'],
                        vis.attrs['sensor'], atmosphere, aerosol_type)
        logger.info("Removing Rayleigh scattering with atmosphere '{}' and aerosol type '{}' for '{}'".format(
            atmosphere, aerosol_type, vis.attrs['name']))
        if rayleigh_key not in self._rayleigh_cache:
            corrector = Rayleigh(vis.attrs['platform_name'], vis.attrs['sensor'],
                                 atmosphere=atmosphere,
                                 aerosol_type=aerosol_type)
            self._rayleigh_cache[rayleigh_key] = corrector
        else:
            corrector = self._rayleigh_cache[rayleigh_key]

        try:
            refl_cor_band = corrector.get_reflectance(sunz, satz, ssadiff,
                                                      vis.attrs['name'],
                                                      red.data)
        except (KeyError, IOError):
            logger.warning("Could not get the reflectance correction using band name: %s", vis.attrs['name'])
            logger.warning("Will try use the wavelength, however, this may be ambiguous!")
            refl_cor_band = corrector.get_reflectance(sunz, satz, ssadiff,
                                                      vis.attrs['wavelength'][1],
                                                      red.data)
        proj = vis - refl_cor_band
        proj.attrs = vis.attrs
        self.apply_modifier_info(vis, proj)
        return proj


class NIRReflectance(ModifierBase):
    """Get the reflective part of NIR bands."""

    def __init__(self, sunz_threshold=None, **kwargs):
        """Collect custom configuration values.

        Args:
            sunz_threshold: The threshold sun zenith angle used when deriving
                the near infrared reflectance. Above this angle the derivation
                will assume this sun-zenith everywhere. Default None, in which
                case the default threshold defined in Pyspectral will be used.

        """
        self.sun_zenith_threshold = sunz_threshold
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

        if self.sun_zenith_threshold is not None:
            reflectance_3x_calculator = Calculator(metadata['platform_name'], metadata['sensor'], metadata['name'],
                                                   sunz_threshold=self.sun_zenith_threshold)
        else:
            reflectance_3x_calculator = Calculator(metadata['platform_name'], metadata['sensor'], metadata['name'])
            self.sun_zenith_threshold = reflectance_3x_calculator.sunz_threshold
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


class PSPAtmosphericalCorrection(ModifierBase):
    """Correct for atmospheric effects."""

    def __call__(self, projectables, optional_datasets=None, **info):
        """Get the atmospherical correction.

        Uses pyspectral.
        """
        from pyspectral.atm_correction_ir import AtmosphericalCorrection

        band = projectables[0]

        if optional_datasets:
            satz = optional_datasets[0]
        else:
            from pyorbital.orbital import get_observer_look
            lons, lats = band.attrs['area'].get_lonlats(chunks=band.data.chunks)
            sat_lon, sat_lat, sat_alt = get_satpos(band)
            try:
                dummy, satel = get_observer_look(sat_lon,
                                                 sat_lat,
                                                 sat_alt / 1000.0,  # km
                                                 band.attrs['start_time'],
                                                 lons, lats, 0)
            except KeyError:
                raise KeyError(
                    'Band info is missing some meta data!')
            satz = 90 - satel
            del satel

        logger.info('Correction for limb cooling')
        corrector = AtmosphericalCorrection(band.attrs['platform_name'],
                                            band.attrs['sensor'])

        atm_corr = corrector.get_correction(satz, band.attrs['name'], band)
        proj = band - atm_corr
        proj.attrs = band.attrs
        self.apply_modifier_info(band, proj)

        return proj


class CO2Corrector(ModifierBase):
    """Correct for CO2."""

    def __call__(self, projectables, optional_datasets=None, **info):
        """CO2 correction of the brightness temperature of the MSG 3.9um channel.

        .. math::

          T4_CO2corr = (BT(IR3.9)^4 + Rcorr)^0.25
          Rcorr = BT(IR10.8)^4 - (BT(IR10.8)-dt_CO2)^4
          dt_CO2 = (BT(IR10.8)-BT(IR13.4))/4.0

        """
        (ir_039, ir_108, ir_134) = projectables
        logger.info('Applying CO2 correction')
        dt_co2 = (ir_108 - ir_134) / 4.0
        rcorr = ir_108**4 - (ir_108 - dt_co2)**4
        t4_co2corr = (ir_039**4 + rcorr).clip(0.0) ** 0.25

        t4_co2corr.attrs = ir_039.attrs.copy()

        self.apply_modifier_info(ir_039, t4_co2corr)

        return t4_co2corr
