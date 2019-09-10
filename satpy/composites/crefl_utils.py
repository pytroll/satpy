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

Original code written by Ralph Kuehn with modifications by David Hoese and Martin Raspaud.
Ralph's code was originally based on the C crefl code distributed for VIIRS and MODIS.
"""
import logging

import numpy as np
import xarray as xr
import dask.array as da

LOG = logging.getLogger(__name__)

bUseV171 = False

if bUseV171:
    UO3 = 0.319
    UH2O = 2.93
else:
    UO3 = 0.285
    UH2O = 2.93

MAXSOLZ = 86.5
MAXAIRMASS = 18
SCALEHEIGHT = 8000
FILL_INT16 = 32767
TAUSTEP4SPHALB_ABI = .0003
TAUSTEP4SPHALB = .0001

MAXNUMSPHALBVALUES = 4000  # with no aerosol taur <= 0.4 in all bands everywhere
REFLMIN = -0.01
REFLMAX = 1.6


def csalbr(tau):
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


# From crefl.1.7.1
if bUseV171:
    aH2O = np.array([-5.60723, -5.25251, 0, 0, -6.29824, -7.70944, -3.91877, 0,
                     0, 0, 0, 0, 0, 0, 0, 0])
    bH2O = np.array([0.820175, 0.725159, 0, 0, 0.865732, 0.966947, 0.745342, 0,
                     0, 0, 0, 0, 0, 0, 0, 0])
    # const float aO3[Nbands]={ 0.0711,    0.00313, 0.0104,     0.0930,   0,
    # 0, 0, 0.00244, 0.00383, 0.0225, 0.0663, 0.0836, 0.0485, 0.0395, 0.0119,
    # 0.00263};*/
    aO3 = np.array(
        [0.0715289, 0, 0.00743232, 0.089691, 0, 0, 0, 0.001, 0.00383, 0.0225,
         0.0663, 0.0836, 0.0485, 0.0395, 0.0119, 0.00263])
    # const float taur0[Nbands] = { 0.0507,  0.0164,  0.1915,  0.0948,
    # 0.0036,  0.0012,  0.0004,  0.3109, 0.2375, 0.1596, 0.1131, 0.0994,
    # 0.0446, 0.0416, 0.0286, 0.0155};*/
    taur0 = np.array(
        [0.05100, 0.01631, 0.19325, 0.09536, 0.00366, 0.00123, 0.00043, 0.3139,
         0.2375, 0.1596, 0.1131, 0.0994, 0.0446, 0.0416, 0.0286, 0.0155])
else:
    # From polar2grid cviirs.c
    # This number is what Ralph says "looks good"
    rg_fudge = .55
    aH2O = np.array(
        [0.000406601, 0.0015933, 0, 1.78644e-05, 0.00296457, 0.000617252,
         0.000996563, 0.00222253, 0.00094005, 0.000563288, 0, 0, 0, 0, 0, 0,
         2.4111e-003, 7.8454e-003*rg_fudge, 7.9258e-3, 9.3392e-003, 2.53e-2])
    bH2O = np.array([0.812659, 0.832931, 1., 0.8677850, 0.806816, 0.944958,
                     0.78812, 0.791204, 0.900564, 0.942907, 0, 0, 0, 0, 0, 0,
                     # These are actually aO2 values for abi calculations
                     1.2360e-003, 3.7296e-003, 177.7161e-006, 10.4899e-003, 1.63e-2])
    # /*const float aO3[Nbands]={ 0.0711,    0.00313, 0.0104,     0.0930,   0, 0, 0, 0.00244,
    # 0.00383, 0.0225, 0.0663, 0.0836, 0.0485, 0.0395, 0.0119, 0.00263};*/
    aO3 = np.array([0.0433461, 0.0, 0.0178299, 0.0853012, 0, 0, 0, 0.0813531,
                    0, 0, 0.0663, 0.0836, 0.0485, 0.0395, 0.0119, 0.00263,
                    4.2869e-003, 25.6509e-003*rg_fudge, 802.4319e-006, 0.0000e+000, 2e-5])
    # /*const float taur0[Nbands] = { 0.0507,  0.0164,  0.1915,  0.0948,  0.0036,  0.0012,  0.0004,
    # 0.3109, 0.2375, 0.1596, 0.1131, 0.0994, 0.0446, 0.0416, 0.0286, 0.0155};*/
    taur0 = np.array([0.04350, 0.01582, 0.16176, 0.09740, 0.00369, 0.00132,
                      0.00033, 0.05373, 0.01561, 0.00129, 0.1131, 0.0994,
                      0.0446, 0.0416, 0.0286, 0.0155,
                      184.7200e-003, 52.3490e-003, 15.8450e-003, 1.3074e-003, 311.2900e-006])
    # add last 5 from bH2O to aO2
    aO2 = 0

# Map of pixel resolutions -> wavelength -> coefficient index
# Map of pixel resolutions -> band name -> coefficient index
# Index is used in aH2O, bH2O, aO3, and taur0 arrays above
MODIS_COEFF_INDEX_MAP = {
    1000: {
        (0.620, 0.6450, 0.670): 0,
        "1": 0,
        (0.841, 0.8585, 0.876): 1,
        "2": 1,
        (0.459, 0.4690, 0.479): 2,
        "3": 2,
        (0.545, 0.5550, 0.565): 3,
        "4": 3,
        (1.230, 1.2400, 1.250): 4,
        "5": 4,
        (1.628, 1.6400, 1.652): 5,
        "6": 5,
        (2.105, 2.1300, 2.155): 6,
        "7": 6,
    }
}
MODIS_COEFF_INDEX_MAP[500] = MODIS_COEFF_INDEX_MAP[1000]
MODIS_COEFF_INDEX_MAP[250] = MODIS_COEFF_INDEX_MAP[1000]

# resolution -> wavelength -> coefficient index
# resolution -> band name -> coefficient index
VIIRS_COEFF_INDEX_MAP = {
    1000: {
        (0.662, 0.6720, 0.682): 0,  # M05
        "M05": 0,
        (0.846, 0.8650, 0.885): 1,  # M07
        "M07": 1,
        (0.478, 0.4880, 0.498): 2,  # M03
        "M03": 2,
        (0.545, 0.5550, 0.565): 3,  # M04
        "M04": 3,
        (1.230, 1.2400, 1.250): 4,  # M08
        "M08": 4,
        (1.580, 1.6100, 1.640): 5,  # M10
        "M10": 5,
        (2.225, 2.2500, 2.275): 6,  # M11
        "M11": 6,
    },
    500: {
        (0.600, 0.6400, 0.680): 7,  # I01
        "I01": 7,
        (0.845, 0.8650, 0.884): 8,  # I02
        "I02": 8,
        (1.580, 1.6100, 1.640): 9,  # I03
        "I03": 9,
    },
}


# resolution -> wavelength -> coefficient index
# resolution -> band name -> coefficient index
ABI_COEFF_INDEX_MAP = {
    2000: {
        (0.450, 0.470, 0.490): 16,  # C01
        "C01": 16,
        (0.590, 0.640, 0.690): 17,  # C02
        "C02": 17,
        (0.8455, 0.865, 0.8845): 18,  # C03
        "C03": 18,
        # (1.3705, 1.378, 1.3855): None,  # C04
        # "C04": None,
        (1.580, 1.610, 1.640): 19,  # C05
        "C05": 19,
        (2.225, 2.250, 2.275): 20,  # C06
        "C06": 20
    },
}


COEFF_INDEX_MAP = {
    "viirs": VIIRS_COEFF_INDEX_MAP,
    "modis": MODIS_COEFF_INDEX_MAP,
    "abi": ABI_COEFF_INDEX_MAP,
}


def find_coefficient_index(sensor, wavelength_range, resolution=0):
    """Return index in to coefficient arrays for this band's wavelength.

    This function search through the `COEFF_INDEX_MAP` dictionary and
    finds the first key where the nominal wavelength of `wavelength_range`
    falls between the minimum wavelength and maximum wavelength of the key.
    `wavelength_range` can also be the standard name of the band. For
    example, "M05" for VIIRS or "1" for MODIS.

    :param sensor: sensor of band to be corrected
    :param wavelength_range: 3-element tuple of (min wavelength, nominal wavelength, max wavelength)
    :param resolution: resolution of the band to be corrected
    :return: index in to coefficient arrays like `aH2O`, `aO3`, etc.
             None is returned if no matching wavelength is found
    """

    index_map = COEFF_INDEX_MAP[sensor.lower()]
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
    else:
        for k, v in index_map.items():
            if isinstance(k, str):
                # we are analyzing wavelengths and ignoring dataset names
                continue
            if k[0] <= wavelength_range[1] <= k[2]:
                return v


def get_coefficients(sensor, wavelength_range, resolution=0):
    """

    :param sensor: sensor of the band to be corrected
    :param wavelength_range: 3-element tuple of (min wavelength, nominal wavelength, max wavelength)
    :param resolution: resolution of the band to be corrected
    :return: aH2O, bH2O, aO3, taur0 coefficient values
    """
    idx = find_coefficient_index(sensor,
                                 wavelength_range,
                                 resolution=resolution)
    return aH2O[idx], bH2O[idx], aO3[idx], taur0[idx]


def chand(phi, muv, mus, taur):
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
    xph2 = -xfd * xbeta2 * 1.5 * mus * muv * da.sqrt(
        1.0 - mus * mus) * da.sqrt(1.0 - muv * muv)
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

    # ib = find_coefficient_index(center_wl)
    # if ib is None:
    #     raise ValueError("Can't handle band with wavelength '{}'".format(center_wl))

    xlntaur = da.log(taur)

    fs0 = fs01 + fs02 * xlntaur
    fs1 = as1[0] + xlntaur * as1[1]
    fs2 = as2[0] + xlntaur * as2[1]
    del xlntaur, fs01, fs02

    trdown = da.exp(-taur / mus)
    trup = da.exp(-taur / muv)

    xitm1 = (1.0 - trdown * trup) / 4.0 / (mus + muv)
    xitm2 = (1.0 - trdown) * (1.0 - trup)
    xitot1 = xph1 * (xitm1 + xitm2 * fs0)
    xitot2 = xph2 * (xitm1 + xitm2 * fs1)
    xitot3 = xph3 * (xitm1 + xitm2 * fs2)
    del xph1, xph2, xph3, xitm1, xitm2, fs0, fs1, fs2

    phios = da.deg2rad(phi + 180.0)
    xcos1 = 1.0
    xcos2 = da.cos(phios)
    xcos3 = da.cos(2.0 * phios)
    del phios

    rhoray = xitot1 * xcos1 + xitot2 * xcos2 * 2.0 + xitot3 * xcos3 * 2.0
    return rhoray, trdown, trup


def _sphalb_index(index_arr, sphalb0):
    # FIXME: if/when dask can support lazy index arrays then remove this
    return sphalb0[index_arr]


def atm_variables_finder(mus, muv, phi, height, tau, tO3, tH2O, taustep4sphalb, tO2=1.0):
    tau_step = da.linspace(taustep4sphalb, MAXNUMSPHALBVALUES * taustep4sphalb, MAXNUMSPHALBVALUES,
                           chunks=int(MAXNUMSPHALBVALUES / 2))
    sphalb0 = csalbr(tau_step)
    taur = tau * da.exp(-height / SCALEHEIGHT)
    rhoray, trdown, trup = chand(phi, muv, mus, taur)
    if isinstance(height, xr.DataArray):
        sphalb = da.map_blocks(_sphalb_index, (taur / taustep4sphalb + 0.5).astype(np.int32).data, sphalb0.compute(),
                               dtype=sphalb0.dtype)
    else:
        sphalb = sphalb0[(taur / taustep4sphalb + 0.5).astype(np.int32)]
    Ttotrayu = ((2 / 3. + muv) + (2 / 3. - muv) * trup) / (4 / 3. + taur)
    Ttotrayd = ((2 / 3. + mus) + (2 / 3. - mus) * trdown) / (4 / 3. + taur)
    TtotraytH2O = Ttotrayu * Ttotrayd * tH2O
    tOG = tO3 * tO2
    return sphalb, rhoray, TtotraytH2O, tOG


def get_atm_variables(mus, muv, phi, height, ah2o, bh2o, ao3, tau):
    air_mass = 1.0 / mus + 1 / muv
    air_mass = air_mass.where(air_mass <= MAXAIRMASS, -1.0)
    tO3 = 1.0
    tH2O = 1.0
    if ao3 != 0:
        tO3 = da.exp(-air_mass * UO3 * ao3)
    if bh2o != 0:
        if bUseV171:
            tH2O = da.exp(-da.exp(ah2o + bh2o * da.log(air_mass * UH2O)))
        else:
            tH2O = da.exp(-(ah2o * ((air_mass * UH2O) ** bh2o)))
    # Returns sphalb, rhoray, TtotraytH2O, tOG
    return atm_variables_finder(mus, muv, phi, height, tau, tO3, tH2O, TAUSTEP4SPHALB)


def get_atm_variables_abi(mus, muv, phi, height, G_O3, G_H2O, G_O2, ah2o, ao2, ao3, tau):
    tO3 = 1.0
    tH2O = 1.0
    if ao3 != 0:
        tO3 = da.exp(-G_O3 * ao3)
    if ah2o != 0:
        tH2O = da.exp(-G_H2O * ah2o)
    tO2 = da.exp(-G_O2 * ao2)
    # Returns sphalb, rhoray, TtotraytH2O, tOG.
    return atm_variables_finder(mus, muv, phi, height, tau, tO3, tH2O, TAUSTEP4SPHALB_ABI, tO2=tO2)


def G_calc(zenith, a_coeff):
    return (da.cos(da.deg2rad(zenith))+(a_coeff[0]*(zenith**a_coeff[1])*(a_coeff[2]-zenith)**a_coeff[3]))**-1


def _avg_elevation_index(avg_elevation, row, col):
    return avg_elevation[row, col]


def run_crefl(refl, coeffs,
              lon,
              lat,
              sensor_azimuth,
              sensor_zenith,
              solar_azimuth,
              solar_zenith,
              avg_elevation=None,
              percent=False,
              use_abi=False):
    """Run main crefl algorithm.

    All input parameters are per-pixel values meaning they are the same size
    and shape as the input reflectance data, unless otherwise stated.

    :param reflectance_bands: tuple of reflectance band arrays
    :param coefficients: tuple of coefficients for each band (see `get_coefficients`)
    :param lon: input swath longitude array
    :param lat: input swath latitude array
    :param sensor_azimuth: input swath sensor azimuth angle array
    :param sensor_zenith: input swath sensor zenith angle array
    :param solar_azimuth: input swath solar azimuth angle array
    :param solar_zenith: input swath solar zenith angle array
    :param avg_elevation: average elevation (usually pre-calculated and stored in CMGDEM.hdf)
    :param percent: True if input reflectances are on a 0-100 scale instead of 0-1 scale (default: False)

    """
    # FUTURE: Find a way to compute the average elevation before hand
    # Get digital elevation map data for our granule, set ocean fill value to 0
    if avg_elevation is None:
        LOG.debug("No average elevation information provided in CREFL")
        # height = np.zeros(lon.shape, dtype=np.float)
        height = 0.
    else:
        LOG.debug("Using average elevation information provided to CREFL")
        lat[(lat <= -90) | (lat >= 90)] = np.nan
        lon[(lon <= -180) | (lon >= 180)] = np.nan
        row = ((90.0 - lat) * avg_elevation.shape[0] / 180.0).astype(np.int32)
        col = ((lon + 180.0) * avg_elevation.shape[1] / 360.0).astype(np.int32)
        space_mask = da.isnull(lon) | da.isnull(lat)
        row[space_mask] = 0
        col[space_mask] = 0

        height = da.map_blocks(_avg_elevation_index, avg_elevation, row, col, dtype=avg_elevation.dtype)
        height = xr.DataArray(height, dims=['y', 'x'])
        # negative heights aren't allowed, clip to 0
        height = height.where((height >= 0.) & ~space_mask, 0.0)
        del lat, lon, row, col
    mus = da.cos(da.deg2rad(solar_zenith))
    mus = mus.where(mus >= 0)
    muv = da.cos(da.deg2rad(sensor_zenith))
    phi = solar_azimuth - sensor_azimuth

    if use_abi:
        LOG.debug("Using ABI CREFL algorithm")
        a_O3 = [268.45, 0.5, 115.42, -3.2922]
        a_H2O = [0.0311, 0.1, 92.471, -1.3814]
        a_O2 = [0.4567, 0.007, 96.4884, -1.6970]
        G_O3 = G_calc(solar_zenith, a_O3) + G_calc(sensor_zenith, a_O3)
        G_H2O = G_calc(solar_zenith, a_H2O) + G_calc(sensor_zenith, a_H2O)
        G_O2 = G_calc(solar_zenith, a_O2) + G_calc(sensor_zenith, a_O2)
        # Note: bh2o values are actually ao2 values for abi
        sphalb, rhoray, TtotraytH2O, tOG = get_atm_variables_abi(mus, muv, phi, height, G_O3, G_H2O, G_O2, *coeffs)
    else:
        LOG.debug("Using original VIIRS CREFL algorithm")
        sphalb, rhoray, TtotraytH2O, tOG = get_atm_variables(mus, muv, phi, height, *coeffs)

    del solar_azimuth, solar_zenith, sensor_zenith, sensor_azimuth
    # Note: Assume that fill/invalid values are either NaN or we are dealing
    # with masked arrays
    if percent:
        corr_refl = ((refl / 100.) / tOG - rhoray) / TtotraytH2O
    else:
        corr_refl = (refl / tOG - rhoray) / TtotraytH2O
    corr_refl /= (1.0 + corr_refl * sphalb)
    return corr_refl.clip(REFLMIN, REFLMAX)
