#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010-2018 Satpy developers
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
"""Shared utilities for correcting reflectance data using the 'crefl' algorithm.

The CREFL algorithm in this module is based on the `NASA CREFL SPA`_ software,
the `NASA CVIIRS SPA`_, and customizations of these algorithms for ABI/AHI by
Ralph Kuehn and Min Oo at the Space Science and Engineering Center (SSEC).

The CREFL SPA documentation page describes the algorithm by saying:

    The CREFL_SPA processes MODIS Aqua and Terra Level 1B DB data to create the
    MODIS Level 2 Corrected Reflectance product. The algorithm performs a simple
    atmospheric correction with MODIS visible, near-infrared, and short-wave
    infrared bands (bands 1 through 16).

    It corrects for molecular (Rayleigh) scattering and gaseous absorption (water
    vapor and ozone) using climatological values for gas contents. It requires no
    real-time input of ancillary data. The algorithm performs no aerosol
    correction. The Corrected Reflectance products created by CREFL_SPA are very
    similar to the MODIS Land Surface Reflectance product (MOD09) in clear
    atmospheric conditions, since the algorithms used to derive both are based on
    the 6S Radiative Transfer Model. The products show differences in the presence
    of aerosols, however, because the MODIS Land Surface Reflectance product uses
    a more complex atmospheric correction algorithm that includes a correction for
    aerosols.

The additional logic to support ABI (AHI support not included) was originally
written by Ralph Kuehn and Min Oo at SSEC. Additional modifications were
performed by Martin Raspaud, David Hoese, and Will Roberts to make the code
work together and be more dask compatible.

The AHI/ABI implementation is based on the MODIS collection 6 algorithm, where
a spherical-shell atmosphere was assumed rather than a plane-parallel. See
Appendix A in: "The Collection 6 MODIS aerosol products over land and ocean"
Atmos. Meas. Tech., 6, 2989–3034, 2013 www.atmos-meas-tech.net/6/2989/2013/
:doi:`10.5194/amt-6-2989-2013`.


The original CREFL code is similar to what is described in appendix A1 (page
74) of the ATBD for the `MODIS MOD04/MYD04`_ data product.

.. _NASA CREFL SPA: https://directreadout.sci.gsfc.nasa.gov/?id=dspContent&cid=92&type=software
.. _NASA CVIIRS SPA: https://directreadout.sci.gsfc.nasa.gov/?id=dspContent&cid=277&type=software
.. _MODIS MOD04/MYD04: https://modis.gsfc.nasa.gov/data/atbd/atbd_mod02.pdf


"""
from __future__ import annotations

import logging
from typing import Optional, Type, Union

import dask.array as da
import numpy as np
import xarray as xr

from satpy.dataset.dataid import WavelengthRange

LOG = logging.getLogger(__name__)

UO3_MODIS = 0.319
UH2O_MODIS = 2.93
UO3_VIIRS = 0.285
UH2O_VIIRS = 2.93

MAXSOLZ = 86.5
MAXAIRMASS = 18
SCALEHEIGHT = 8000
FILL_INT16 = 32767
TAUSTEP4SPHALB_ABI = .0003
TAUSTEP4SPHALB = .0001

MAXNUMSPHALBVALUES = 4000  # with no aerosol taur <= 0.4 in all bands everywhere
REFLMIN = -0.01
REFLMAX = 1.6


class _Coefficients:
    LUTS: list[np.ndarray] = []
    # resolution -> wavelength -> coefficient index
    # resolution -> band name -> coefficient index
    COEFF_INDEX_MAP: dict[int, dict[Union[tuple, str], int]] = {}

    def __init__(self, wavelength_range, resolution=0):
        self._wv_range = wavelength_range
        self._resolution = resolution

    def __call__(self):
        idx = self._find_coefficient_index(self._wv_range, resolution=self._resolution)
        band_luts = [lut_array[idx] for lut_array in self.LUTS]
        return band_luts

    def _find_coefficient_index(self, wavelength_range, resolution=0):
        """Return index in to coefficient arrays for this band's wavelength.

        This function search through the `COEFF_INDEX_MAP` dictionary and
        finds the first key where the nominal wavelength of `wavelength_range`
        falls between the minimum wavelength and maximum wavelength of the key.
        `wavelength_range` can also be the standard name of the band. For
        example, "M05" for VIIRS or "1" for MODIS.

        Args:
            wavelength_range: 3-element tuple of
                (min wavelength, nominal wavelength, max wavelength) or the
                string name of the band.
            resolution: resolution of the band to be corrected

        Returns:
            index in to coefficient arrays like `aH2O`, `aO3`, etc.
            None is returned if no matching wavelength is found

        """
        index_map = self.COEFF_INDEX_MAP
        # Find the best resolution of coefficients
        for res in sorted(index_map.keys()):
            if resolution <= res:
                index_map = index_map[res]
                break
        else:
            raise ValueError("Unrecognized data resolution: {}", resolution)
        # Find the best wavelength of coefficients
        if isinstance(wavelength_range, str):
            # wavelength range is actually a band name
            return index_map[wavelength_range]
        for lut_wvl_range, v in index_map.items():
            if isinstance(lut_wvl_range, str):
                # we are analyzing wavelengths and ignoring dataset names
                continue
            if wavelength_range[1] in lut_wvl_range:
                return v
        raise ValueError(f"Can't find LUT for {wavelength_range}.")


class _ABICoefficients(_Coefficients):
    RG_FUDGE = .55  # This number is what Ralph says "looks good" for ABI/AHI
    LUTS = [
        # aH2O
        np.array([2.4111e-003, 7.8454e-003 * RG_FUDGE, 7.9258e-3, 9.3392e-003, 2.53e-2]),
        # aO2 (bH2O for other instruments)
        np.array([1.2360e-003, 3.7296e-003, 177.7161e-006, 10.4899e-003, 1.63e-2]),
        # aO3
        np.array([4.2869e-003, 25.6509e-003 * RG_FUDGE, 802.4319e-006, 0.0000e+000, 2e-5]),
        # taur0
        np.array([184.7200e-003, 52.3490e-003, 15.8450e-003, 1.3074e-003, 311.2900e-006]),
    ]
    # resolution -> wavelength -> coefficient index
    # resolution -> band name -> coefficient index
    COEFF_INDEX_MAP = {
        2000: {
            WavelengthRange(0.450, 0.470, 0.490): 0,  # C01
            "C01": 0,
            WavelengthRange(0.590, 0.640, 0.690): 1,  # C02
            "C02": 1,
            WavelengthRange(0.8455, 0.865, 0.8845): 2,  # C03
            "C03": 2,
            # WavelengthRange((1.3705, 1.378, 1.3855)): None,  # C04 - No coefficients yet
            # "C04": None,
            WavelengthRange(1.580, 1.610, 1.640): 3,  # C05
            "C05": 3,
            WavelengthRange(2.225, 2.250, 2.275): 4,  # C06
            "C06": 4
        },
    }


class _VIIRSCoefficients(_Coefficients):
    # Values from crefl 1.7.1
    LUTS = [
        # aH2O
        np.array([0.000406601, 0.0015933, 0, 1.78644e-05, 0.00296457, 0.000617252, 0.000996563, 0.00222253, 0.00094005,
                  0.000563288, 0, 0, 0, 0, 0, 0]),
        # bH2O
        np.array([0.812659, 0.832931, 1., 0.8677850, 0.806816, 0.944958, 0.78812, 0.791204, 0.900564, 0.942907, 0, 0,
                  0, 0, 0, 0]),
        # aO3
        np.array([0.0433461, 0.0, 0.0178299, 0.0853012, 0, 0, 0, 0.0813531, 0, 0, 0.0663, 0.0836, 0.0485, 0.0395,
                  0.0119, 0.00263]),
        # taur0
        np.array([0.04350, 0.01582, 0.16176, 0.09740, 0.00369, 0.00132, 0.00033, 0.05373, 0.01561, 0.00129, 0.1131,
                  0.0994, 0.0446, 0.0416, 0.0286, 0.0155]),
    ]
    # resolution -> wavelength -> coefficient index
    # resolution -> band name -> coefficient index
    COEFF_INDEX_MAP = {
        1000: {
            WavelengthRange(0.662, 0.6720, 0.682): 0,  # M05
            "M05": 0,
            WavelengthRange(0.846, 0.8650, 0.885): 1,  # M07
            "M07": 1,
            WavelengthRange(0.478, 0.4880, 0.498): 2,  # M03
            "M03": 2,
            WavelengthRange(0.545, 0.5550, 0.565): 3,  # M04
            "M04": 3,
            WavelengthRange(1.230, 1.2400, 1.250): 4,  # M08
            "M08": 4,
            WavelengthRange(1.580, 1.6100, 1.640): 5,  # M10
            "M10": 5,
            WavelengthRange(2.225, 2.2500, 2.275): 6,  # M11
            "M11": 6,
        },
        500: {
            WavelengthRange(0.600, 0.6400, 0.680): 7,  # I01
            "I01": 7,
            WavelengthRange(0.845, 0.8650, 0.884): 8,  # I02
            "I02": 8,
            WavelengthRange(1.580, 1.6100, 1.640): 9,  # I03
            "I03": 9,
        },
    }


class _MODISCoefficients(_Coefficients):
    # Values from crefl 1.7.1
    LUTS = [
        # aH2O
        np.array([-5.60723, -5.25251, 0, 0, -6.29824, -7.70944, -3.91877, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        # bH2O
        np.array([0.820175, 0.725159, 0, 0, 0.865732, 0.966947, 0.745342, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        # aO3
        np.array([0.0715289, 0, 0.00743232, 0.089691, 0, 0, 0, 0.001, 0.00383, 0.0225, 0.0663,
                  0.0836, 0.0485, 0.0395, 0.0119, 0.00263]),
        # taur0
        np.array([0.05100, 0.01631, 0.19325, 0.09536, 0.00366, 0.00123, 0.00043, 0.3139, 0.2375, 0.1596, 0.1131,
                  0.0994, 0.0446, 0.0416, 0.0286, 0.0155]),
    ]
    # Map of pixel resolutions -> wavelength -> coefficient index
    # Map of pixel resolutions -> band name -> coefficient index
    COEFF_INDEX_MAP = {
        1000: {
            WavelengthRange(0.620, 0.6450, 0.670): 0,
            "1": 0,
            WavelengthRange(0.841, 0.8585, 0.876): 1,
            "2": 1,
            WavelengthRange(0.459, 0.4690, 0.479): 2,
            "3": 2,
            WavelengthRange(0.545, 0.5550, 0.565): 3,
            "4": 3,
            WavelengthRange(1.230, 1.2400, 1.250): 4,
            "5": 4,
            WavelengthRange(1.628, 1.6400, 1.652): 5,
            "6": 5,
            WavelengthRange(2.105, 2.1300, 2.155): 6,
            "7": 6,
        }
    }
    COEFF_INDEX_MAP[500] = COEFF_INDEX_MAP[1000]
    COEFF_INDEX_MAP[250] = COEFF_INDEX_MAP[1000]


def run_crefl(refl,
              sensor_azimuth,
              sensor_zenith,
              solar_azimuth,
              solar_zenith,
              avg_elevation=None,
              ):
    """Run main crefl algorithm.

    All input parameters are per-pixel values meaning they are the same size
    and shape as the input reflectance data, unless otherwise stated.

    :param refl: tuple of reflectance band arrays
    :param sensor_azimuth: input swath sensor azimuth angle array
    :param sensor_zenith: input swath sensor zenith angle array
    :param solar_azimuth: input swath solar azimuth angle array
    :param solar_zenith: input swath solar zenith angle array
    :param avg_elevation: average elevation (usually pre-calculated and stored in CMGDEM.hdf)

    """
    runner_cls = _runner_class_for_sensor(refl.attrs['sensor'])
    runner = runner_cls(refl)
    corr_refl = runner(sensor_azimuth, sensor_zenith, solar_azimuth, solar_zenith, avg_elevation)
    return corr_refl


class _CREFLRunner:
    def __init__(self, refl_data_arr):
        self._is_percent = refl_data_arr.attrs["units"] == "%"
        if self._is_percent:
            attrs = refl_data_arr.attrs
            refl_data_arr = refl_data_arr / 100.0
            refl_data_arr.attrs = attrs
        self._refl = refl_data_arr

    @property
    def coeffs_cls(self) -> Type[_Coefficients]:
        raise NotImplementedError()

    def __call__(self, sensor_azimuth, sensor_zenith, solar_azimuth, solar_zenith, avg_elevation):
        refl = self._refl
        height = self._height_from_avg_elevation(avg_elevation)
        coeffs_helper = self.coeffs_cls(refl.attrs["wavelength"], refl.attrs["resolution"])
        coeffs = coeffs_helper()
        mus = np.cos(np.deg2rad(solar_zenith))
        mus = mus.where(mus >= 0)
        muv = np.cos(np.deg2rad(sensor_zenith))
        phi = solar_azimuth - sensor_azimuth
        corr_refl = self._run_crefl(mus, muv, phi, solar_zenith, sensor_zenith, height, coeffs)
        if self._is_percent:
            corr_refl = corr_refl * 100.0
        return xr.DataArray(corr_refl, dims=refl.dims, coords=refl.coords, attrs=refl.attrs)

    def _run_crefl(self, mus, muv, phi, solar_zenith, sensor_zenith, height, coeffs):
        raise NotImplementedError()

    def _height_from_avg_elevation(self, avg_elevation: Optional[np.ndarray]) -> da.Array:
        """Get digital elevation map data for our granule with ocean fill value set to 0."""
        if avg_elevation is None:
            LOG.debug("No average elevation information provided in CREFL")
            # height = np.zeros(lon.shape, dtype=np.float64)
            height = 0.
        else:
            LOG.debug("Using average elevation information provided to CREFL")
            lon, lat = self._refl.attrs['area'].get_lonlats(chunks=self._refl.chunks)
            height = da.map_blocks(_space_mask_height, lon, lat, avg_elevation,
                                   chunks=lon.chunks, dtype=avg_elevation.dtype)
        return height


class _ABICREFLRunner(_CREFLRunner):
    @property
    def coeffs_cls(self) -> Type[_Coefficients]:
        return _ABICoefficients

    def _run_crefl(self, mus, muv, phi, solar_zenith, sensor_zenith, height, coeffs):
        LOG.debug("Using ABI CREFL algorithm")
        return da.map_blocks(_run_crefl_abi, self._refl.data, mus.data, muv.data, phi.data,
                             solar_zenith.data, sensor_zenith.data, height, *coeffs,
                             meta=np.ndarray((), dtype=self._refl.dtype),
                             chunks=self._refl.chunks, dtype=self._refl.dtype,
                             )


class _VIIRSMODISCREFLRunner(_CREFLRunner):
    def _run_crefl(self, mus, muv, phi, solar_zenith, sensor_zenith, height, coeffs):
        return da.map_blocks(_run_crefl, self._refl.data, mus.data, muv.data, phi.data,
                             height, self._refl.attrs.get("sensor"), *coeffs,
                             meta=np.ndarray((), dtype=self._refl.dtype),
                             chunks=self._refl.chunks, dtype=self._refl.dtype,
                             )


class _VIIRSCREFLRunner(_VIIRSMODISCREFLRunner):
    @property
    def coeffs_cls(self) -> Type[_Coefficients]:
        return _VIIRSCoefficients

    def _run_crefl(self, mus, muv, phi, solar_zenith, sensor_zenith, height, coeffs):
        LOG.debug("Using VIIRS CREFL algorithm")
        return super()._run_crefl(mus, muv, phi, solar_zenith, sensor_zenith, height, coeffs)


class _MODISCREFLRunner(_VIIRSMODISCREFLRunner):
    @property
    def coeffs_cls(self) -> Type[_Coefficients]:
        return _MODISCoefficients

    def _run_crefl(self, mus, muv, phi, solar_zenith, sensor_zenith, height, coeffs):
        LOG.debug("Using MODIS CREFL algorithm")
        return super()._run_crefl(mus, muv, phi, solar_zenith, sensor_zenith, height, coeffs)


_SENSOR_TO_RUNNER = {
    "abi": _ABICREFLRunner,
    "viirs": _VIIRSCREFLRunner,
    "modis": _MODISCREFLRunner,
}


def _runner_class_for_sensor(sensor_name: str) -> Type[_CREFLRunner]:
    try:
        return _SENSOR_TO_RUNNER[sensor_name]
    except KeyError:
        raise NotImplementedError(f"Don't know how to apply CREFL to data from sensor {sensor_name}.")


def _space_mask_height(lon, lat, avg_elevation):
    lat[(lat <= -90) | (lat >= 90)] = np.nan
    lon[(lon <= -180) | (lon >= 180)] = np.nan
    row = ((90.0 - lat) * avg_elevation.shape[0] / 180.0).astype(np.int32)
    col = ((lon + 180.0) * avg_elevation.shape[1] / 360.0).astype(np.int32)
    space_mask = np.isnan(lon) | np.isnan(lat)
    row[space_mask] = 0
    col[space_mask] = 0

    height = avg_elevation[row, col]
    # negative heights aren't allowed, clip to 0
    height[(height < 0.0) | np.isnan(height) | space_mask] = 0.0
    return height


def _run_crefl(refl, mus, muv, phi, height, sensor_name, *coeffs):
    atm_vars_cls = _VIIRSAtmosphereVariables if sensor_name.lower() == "viirs" else _MODISAtmosphereVariables
    atm_vars = atm_vars_cls(mus, muv, phi, height, *coeffs)
    sphalb, rhoray, TtotraytH2O, tOG = atm_vars()
    return _correct_refl(refl, tOG, rhoray, TtotraytH2O, sphalb)


def _run_crefl_abi(refl, mus, muv, phi, solar_zenith, sensor_zenith, height,
                   *coeffs):
    a_O3 = [268.45, 0.5, 115.42, -3.2922]
    a_H2O = [0.0311, 0.1, 92.471, -1.3814]
    a_O2 = [0.4567, 0.007, 96.4884, -1.6970]
    G_O3 = _G_calc(solar_zenith, a_O3) + _G_calc(sensor_zenith, a_O3)
    G_H2O = _G_calc(solar_zenith, a_H2O) + _G_calc(sensor_zenith, a_H2O)
    G_O2 = _G_calc(solar_zenith, a_O2) + _G_calc(sensor_zenith, a_O2)
    # Note: bh2o values are actually ao2 values for abi
    atm_vars = _ABIAtmosphereVariables(G_O3, G_H2O, G_O2,
                                       mus, muv, phi, height, *coeffs)
    sphalb, rhoray, TtotraytH2O, tOG = atm_vars()
    return _correct_refl(refl, tOG, rhoray, TtotraytH2O, sphalb)


def _G_calc(zenith, a_coeff):
    return (np.cos(np.deg2rad(zenith))+(a_coeff[0]*(zenith**a_coeff[1])*(a_coeff[2]-zenith)**a_coeff[3]))**-1


def _correct_refl(refl, tOG, rhoray, TtotraytH2O, sphalb):
    corr_refl = (refl / tOG - rhoray) / TtotraytH2O
    corr_refl /= (1.0 + corr_refl * sphalb)
    return corr_refl.clip(REFLMIN, REFLMAX)


class _AtmosphereVariables:
    def __init__(self, mus, muv, phi, height, ah2o, bh2o, ao3, tau):
        self._mus = mus
        self._muv = muv
        self._phi = phi
        self._height = height
        self._ah2o = ah2o
        self._bh2o = bh2o
        self._ao3 = ao3
        self._tau = tau
        self._taustep4sphalb = TAUSTEP4SPHALB

    def __call__(self):
        tau_step = np.linspace(
            self._taustep4sphalb,
            MAXNUMSPHALBVALUES * self._taustep4sphalb,
            MAXNUMSPHALBVALUES)
        sphalb0 = _csalbr(tau_step)
        taur = self._tau * np.exp(-self._height / SCALEHEIGHT)
        rhoray, trdown, trup = _chand(self._phi, self._muv, self._mus, taur)
        sphalb = sphalb0[(taur / self._taustep4sphalb + 0.5).astype(np.int32)]
        Ttotrayu = ((2 / 3. + self._muv) + (2 / 3. - self._muv) * trup) / (4 / 3. + taur)
        Ttotrayd = ((2 / 3. + self._mus) + (2 / 3. - self._mus) * trdown) / (4 / 3. + taur)

        tH2O = self._get_th2o()
        TtotraytH2O = Ttotrayu * Ttotrayd * tH2O

        tO2 = self._get_to2()
        tO3 = self._get_to3()
        tOG = tO3 * tO2
        return sphalb, rhoray, TtotraytH2O, tOG

    def _get_to2(self):
        return 1.0

    def _get_to3(self):
        raise NotImplementedError()

    def _get_th2o(self):
        raise NotImplementedError()


class _ABIAtmosphereVariables(_AtmosphereVariables):
    def __init__(self, G_O3, G_H2O, G_O2, *args):
        super().__init__(*args)
        self._G_O3 = G_O3
        self._G_H2O = G_H2O
        self._G_O2 = G_O2
        self._taustep4sphalb = TAUSTEP4SPHALB_ABI

    def _get_to2(self):
        # NOTE: bh2o is actually ao2 for ABI
        return np.exp(-self._G_O2 * self._bh2o)

    def _get_to3(self):
        return np.exp(-self._G_O3 * self._ao3) if self._ao3 != 0 else 1.0

    def _get_th2o(self):
        return np.exp(-self._G_H2O * self._ah2o) if self._ah2o != 0 else 1.0


class _VIIRSAtmosphereVariables(_AtmosphereVariables):
    def __init__(self, *args):
        super().__init__(*args)
        self._airmass = self._compute_airmass()

    def _compute_airmass(self):
        air_mass = 1.0 / self._mus + 1 / self._muv
        air_mass[air_mass > MAXAIRMASS] = -1.0
        return air_mass

    def _get_to3(self):
        if self._ao3 == 0:
            return 1.0
        return np.exp(-self._airmass * UO3_VIIRS * self._ao3)

    def _get_th2o(self):
        if self._bh2o == 0:
            return 1.0
        return np.exp(-(self._ah2o * ((self._airmass * UH2O_VIIRS) ** self._bh2o)))


class _MODISAtmosphereVariables(_VIIRSAtmosphereVariables):
    def _get_to3(self):
        if self._ao3 == 0:
            return 1.0
        return np.exp(-self._airmass * UO3_MODIS * self._ao3)

    def _get_th2o(self):
        if self._bh2o == 0:
            return 1.0
        return np.exp(-np.exp(self._ah2o + self._bh2o * np.log(self._airmass * UH2O_MODIS)))


def _csalbr(tau):
    # Previously 3 functions csalbr fintexp1, fintexp3
    a = [-.57721566, 0.99999193, -0.24991055, 0.05519968, -0.00976004,
         0.00107857]
    # xx = a[0] + a[1] * tau + a[2] * tau**2 + a[3] * tau**3 + a[4] * tau**4 + a[5] * tau**5
    # xx = np.polyval(a[::-1], tau)

    # xx = a[0]
    # xftau = 1.0
    # for i in xrange(5):
    #     xftau = xftau*tau
    #     xx = xx + a[i] * xftau
    fintexp1 = np.polyval(a[::-1], tau) - np.log(tau)
    fintexp3 = (np.exp(-tau) * (1.0 - tau) + tau**2 * fintexp1) / 2.0

    return (3.0 * tau - fintexp3 *
            (4.0 + 2.0 * tau) + 2.0 * np.exp(-tau)) / (4.0 + 3.0 * tau)


def _chand(phi, muv, mus, taur):
    # FROM FUNCTION CHAND
    # phi: azimuthal difference between sun and observation in degree
    #      (phi=0 in backscattering direction)
    # mus: cosine of the sun zenith angle
    # muv: cosine of the observation zenith angle
    # taur: molecular optical depth
    # rhoray: molecular path reflectance
    # constant xdep: depolarization factor (0.0279)
    #          xfd = (1-xdep/(2-xdep)) / (1 + 2*xdep/(2-xdep)) = 2 * (1 - xdep) / (2 + xdep) = 0.958725775
    # */
    xfd = 0.958725775
    xbeta2 = 0.5
    #         float pl[5];
    #         double fs01, fs02, fs0, fs1, fs2;
    as0 = [0.33243832, 0.16285370, -0.30924818, -0.10324388, 0.11493334,
           -6.777104e-02, 1.577425e-03, -1.240906e-02, 3.241678e-02,
           -3.503695e-02]
    as1 = [0.19666292, -5.439061e-02]
    as2 = [0.14545937, -2.910845e-02]
    #         float phios, xcos1, xcos2, xcos3;
    #         float xph1, xph2, xph3, xitm1, xitm2;
    #         float xlntaur, xitot1, xitot2, xitot3;
    #         int i, ib;

    xph1 = 1.0 + (3.0 * mus * mus - 1.0) * (3.0 * muv * muv - 1.0) * xfd / 8.0
    xph2 = -xfd * xbeta2 * 1.5 * mus * muv * np.sqrt(
        1.0 - mus * mus) * np.sqrt(1.0 - muv * muv)
    xph3 = xfd * xbeta2 * 0.375 * (1.0 - mus * mus) * (1.0 - muv * muv)

    # pl[0] = 1.0
    # pl[1] = mus + muv
    # pl[2] = mus * muv
    # pl[3] = mus * mus + muv * muv
    # pl[4] = mus * mus * muv * muv

    fs01 = as0[0] + (mus + muv) * as0[1] + (mus * muv) * as0[2] + (
            mus * mus + muv * muv) * as0[3] + (mus * mus * muv * muv) * as0[4]
    fs02 = as0[5] + (mus + muv) * as0[6] + (mus * muv) * as0[7] + (
            mus * mus + muv * muv) * as0[8] + (mus * mus * muv * muv) * as0[9]
    #         for (i = 0; i < 5; i++) {
    #                 fs01 += (double) (pl[i] * as0[i]);
    #                 fs02 += (double) (pl[i] * as0[5 + i]);
    #         }

    # for refl, (ah2o, bh2o, ao3, tau) in zip(reflectance_bands, coefficients):

    # ib = _find_coefficient_index(center_wl)
    # if ib is None:
    #     raise ValueError("Can't handle band with wavelength '{}'".format(center_wl))

    xlntaur = np.log(taur)

    fs0 = fs01 + fs02 * xlntaur
    fs1 = as1[0] + xlntaur * as1[1]
    fs2 = as2[0] + xlntaur * as2[1]
    del xlntaur, fs01, fs02

    trdown = np.exp(-taur / mus)
    trup = np.exp(-taur / muv)

    xitm1 = (1.0 - trdown * trup) / 4.0 / (mus + muv)
    xitm2 = (1.0 - trdown) * (1.0 - trup)
    xitot1 = xph1 * (xitm1 + xitm2 * fs0)
    xitot2 = xph2 * (xitm1 + xitm2 * fs1)
    xitot3 = xph3 * (xitm1 + xitm2 * fs2)
    del xph1, xph2, xph3, xitm1, xitm2, fs0, fs1, fs2

    phios = np.deg2rad(phi + 180.0)
    xcos1 = 1.0
    xcos2 = np.cos(phios)
    xcos3 = np.cos(2.0 * phios)
    del phios

    rhoray = xitot1 * xcos1 + xitot2 * xcos2 * 2.0 + xitot3 * xcos3 * 2.0
    return rhoray, trdown, trup
