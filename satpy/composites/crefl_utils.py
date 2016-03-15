#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2011, 2012, 2014, 2015.

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   David Hoese <david.hoese@ssec.wisc.edu>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Shared utilities for correcting reflectance data using the 'crefl' algorithm.

Original code written by Ralph Kuehn with modifications by David Hoese and Martin Raspaud.
Ralph's code was originally based on the C crefl code distributed for VIIRS and MODIS.
"""
import os
import sys
import logging
import numpy as np

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
FILL_INT16 =32767

TAUSTEP4SPHALB = 0.0001
MAXNUMSPHALBVALUES = 4000    # with no aerosol taur <= 0.4 in all bands everywhere

def csalbr(tau):
    # Previously 3 functions csalbr fintexp1, fintexp3
    a= [ -.57721566, 0.99999193, -0.24991055, 0.05519968, -0.00976004, 0.00107857]
    xx = a[0] + a[1]*tau + a[2]*tau**2 + a[3]*tau**3 + a[4]*tau**4 + a[5]*tau**5

    # xx = a[0]
    # xftau = 1.0
    # for i in xrange(5):
    #     xftau = xftau*tau
    #     xx = xx + a[i] * xftau
    fintexp1 = xx-np.log(tau)
    fintexp3 = (np.exp(-tau) * (1.0 - tau) + tau**2 * fintexp1) / 2.0

    return (3.0 * tau - fintexp3 * (4.0 + 2.0 * tau) + 2.0 * np.exp(-tau)) / (4.0 + 3.0 * tau)

# From crefl.1.7.1
if bUseV171:
    aH2O =  np.array([-5.60723, -5.25251, 0, 0, -6.29824, -7.70944, -3.91877, 0, 0, 0, 0, 0, 0, 0, 0, 0 ])
    bH2O =  np.array([0.820175, 0.725159, 0, 0, 0.865732, 0.966947, 0.745342, 0, 0, 0, 0, 0, 0, 0, 0, 0 ])
    #const float aO3[Nbands]={ 0.0711,    0.00313, 0.0104,     0.0930,   0, 0, 0, 0.00244, 0.00383, 0.0225, 0.0663, 0.0836, 0.0485, 0.0395, 0.0119, 0.00263};*/
    aO3 =  np.array([0.0715289, 0, 0.00743232, 0.089691, 0, 0, 0, 0.001, 0.00383, 0.0225, 0.0663, 0.0836, 0.0485, 0.0395, 0.0119, 0.00263])
    #const float taur0[Nbands] = { 0.0507,  0.0164,  0.1915,  0.0948,  0.0036,  0.0012,  0.0004,  0.3109, 0.2375, 0.1596, 0.1131, 0.0994, 0.0446, 0.0416, 0.0286, 0.0155};*/
    taur0 = np.array([0.05100, 0.01631, 0.19325, 0.09536, 0.00366, 0.00123, 0.00043, 0.3139, 0.2375, 0.1596, 0.1131, 0.0994, 0.0446, 0.0416, 0.0286, 0.0155])
else:
    #From polar2grid cviirs.c
    aH2O =  np.array([0.000406601 ,0.0015933 , 0,1.78644e-05 ,0.00296457 ,0.000617252 , 0.000996563,0.00222253 ,0.00094005 , 0.000563288, 0, 0, 0, 0, 0, 0 ])
    bH2O =  np.array([0.812659,0.832931 , 1., 0.8677850, 0.806816 , 0.944958 ,0.78812 ,0.791204 ,0.900564 ,0.942907 , 0, 0, 0, 0, 0, 0 ])
    #/*const float aO3[Nbands]={ 0.0711,    0.00313, 0.0104,     0.0930,   0, 0, 0, 0.00244, 0.00383, 0.0225, 0.0663, 0.0836, 0.0485, 0.0395, 0.0119, 0.00263};*/
    aO3 =  np.array([ 0.0433461, 0.0,    0.0178299   ,0.0853012 , 0, 0, 0, 0.0813531,   0, 0, 0.0663, 0.0836, 0.0485, 0.0395, 0.0119, 0.00263])
    #/*const float taur0[Nbands] = { 0.0507,  0.0164,  0.1915,  0.0948,  0.0036,  0.0012,  0.0004,  0.3109, 0.2375, 0.1596, 0.1131, 0.0994, 0.0446, 0.0416, 0.0286, 0.0155};*/
    taur0 = np.array([0.04350, 0.01582, 0.16176, 0.09740,0.00369 ,0.00132 ,0.00033 ,0.05373 ,0.01561 ,0.00129, 0.1131, 0.0994, 0.0446, 0.0416, 0.0286, 0.0155])

# Map a range of wavelengths (min, nominal, wavelength) to the proper
# index inside the above coefficient arrays
# Indexes correspond to the original MODIS coefficients, starting with
# band 1 (index 0) and going to band 7
# Min and max wavelengths have been adjusted to include VIIRS and other
# instrument's bands
COEFF_INDEX_MAP = {
    (0.620, 0.6450, 0.672): 0,
    (0.841, 0.8585, 0.876): 1,
    (0.445, 0.4690, 0.488): 2,
    (0.545, 0.5550, 0.565): 3,
    (1.020, 1.2400, 1.250): 4,
    (1.600, 1.6400, 1.652): 5,
    (2.105, 2.1300, 2.255): 6,
}


def find_coefficient_index(nominal_wavelength):
    """Return index in to coefficient arrays for this band's wavelength.

    This function search through the `COEFF_INDEX_MAP` dictionary and
    finds the first key where `nominal_wavelength` falls between
    the minimum wavelength and maximum wavelength of the key.

    :param nominal_wavelength: float wavelength of the band being corrected
    :return: index in to coefficient arrays like `aH2O`, `aO3`, etc.
             None is returned if no matching wavelength is found
    """
    for (min_wl, nom_wl, max_wl), v in COEFF_INDEX_MAP.items():
        if min_wl <= nominal_wavelength <= max_wl:
            return v


def run_crefl(reflectance_bands, center_wavelengths,
              lon, lat,
              sensor_azimuth, sensor_zenith, solar_azimuth, solar_zenith,
              avg_elevation=None, percent=False):
    """Run main crefl algorithm.

    All input parameters are per-pixel values meaning they are the same size
    and shape as the input reflectance data, unless otherwise stated.

    :param reflectance_bands: tuple of reflectance band arrays
    :param center_wavelengths: tuple of nonimal wavelengths for the corresponding reflectance band
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
        height = 0.0
    else:
        row = np.int32((90.0 - lat) * avg_elevation.shape[0] / 180.0)
        col = np.int32((lon + 180.0) * avg_elevation.shape[1] / 360.0)
        height = np.float64(avg_elevation[row, col])
        height[height < 0.] = 0.0
        del lat, lon, row, col

    DEG2RAD = np.pi/180.0
    mus = np.cos(solar_zenith * DEG2RAD)
    muv = np.cos(sensor_zenith * DEG2RAD)
    phi = solar_azimuth - sensor_azimuth

    del solar_azimuth, solar_zenith, sensor_zenith, sensor_azimuth

    # From GetAtmVariables
    tau_step = np.linspace(TAUSTEP4SPHALB, MAXNUMSPHALBVALUES*TAUSTEP4SPHALB, MAXNUMSPHALBVALUES)
    sphalb0 = csalbr(tau_step)

    air_mass = 1.0/mus + 1/muv
    ii,jj = np.where(np.greater(air_mass,MAXAIRMASS))
    air_mass[ii,jj] = -1.0

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
           -6.777104e-02, 1.577425e-03, -1.240906e-02, 3.241678e-02, -3.503695e-02]
    as1 = [0.19666292, -5.439061e-02]
    as2 = [0.14545937, -2.910845e-02]
    #         float phios, xcos1, xcos2, xcos3;
    #         float xph1, xph2, xph3, xitm1, xitm2;
    #         float xlntaur, xitot1, xitot2, xitot3;
    #         int i, ib;

    phios = phi + 180.0
    xcos1 = 1.0
    xcos2 = np.cos(phios * DEG2RAD)
    xcos3 = np.cos(2.0 * phios * DEG2RAD)
    xph1 = 1.0 + (3.0 * mus * mus - 1.0) * (3.0 * muv * muv - 1.0) * xfd / 8.0
    xph2 = -xfd * xbeta2 * 1.5 * mus * muv * np.sqrt(1.0 - mus * mus) * np.sqrt(1.0 - muv * muv)
    xph3 = xfd * xbeta2 * 0.375 * (1.0 - mus * mus) * (1.0 - muv * muv)

    # pl[0] = 1.0
    # pl[1] = mus + muv
    # pl[2] = mus * muv
    # pl[3] = mus * mus + muv * muv
    # pl[4] = mus * mus * muv * muv

    fs01 = as0[0] + (mus + muv)*as0[1] + (mus * muv)*as0[2] + (mus * mus + muv * muv)*as0[3] + (mus * mus * muv * muv)*as0[4]
    fs02 = as0[5] + (mus + muv)*as0[6] + (mus * muv)*as0[7] + (mus * mus + muv * muv)*as0[8] + (mus * mus * muv * muv)*as0[9]
    #         for (i = 0; i < 5; i++) {
    #                 fs01 += (double) (pl[i] * as0[i]);
    #                 fs02 += (double) (pl[i] * as0[5 + i]);
    #         }

    odata = []
    for refl, center_wl in zip(reflectance_bands, center_wavelengths):
        ib = find_coefficient_index(center_wl)
        if ib is None:
            raise ValueError("Can't handle band with wavelength '{}'".format(center_wl))

        taur = taur0[ib] * np.exp(-height / SCALEHEIGHT)
        xlntaur = np.log(taur)
        fs0 = fs01 + fs02 * xlntaur
        fs1 = as1[0] + xlntaur * as1[1]
        fs2 = as2[0] + xlntaur * as2[1]
        del xlntaur
        trdown = np.exp(-taur / mus)
        trup= np.exp(-taur / muv)
        xitm1 = (1.0 - trdown * trup) / 4.0 / (mus + muv)
        xitm2 = (1.0 - trdown) * (1.0 - trup)
        xitot1 = xph1 * (xitm1 + xitm2 * fs0)
        xitot2 = xph2 * (xitm1 + xitm2 * fs1)
        xitot3 = xph3 * (xitm1 + xitm2 * fs2)
        rhoray = xitot1 * xcos1 + xitot2 * xcos2 * 2.0 + xitot3 * xcos3 * 2.0

        sphalb = sphalb0[np.int32(taur / TAUSTEP4SPHALB + 0.5)]
        Ttotrayu = ((2 / 3. + muv) + (2 / 3. - muv) * trup) / (4 / 3. + taur)
        Ttotrayd = ((2 / 3. + mus) + (2 / 3. - mus) * trdown) / (4 / 3. + taur)
        tO3 = 1.0
        tO2 = 1.0
        tH2O = 1.0

        if aO3[ib] != 0:
            tO3 = np.exp(-air_mass * UO3 * aO3[ib])
        if bH2O[ib] != 0:
            if bUseV171:
                tH2O = np.exp(-np.exp(aH2O[ib] + bH2O[ib] * np.log(air_mass * UH2O)))
            else:
                tH2O = np.exp(-(aH2O[ib]*(np.power((air_mass * UH2O),bH2O[ib]))))
        #t02 = exp(-m * aO2)
        TtotraytH2O = Ttotrayu * Ttotrayd * tH2O
        tOG = tO3 * tO2

        # Note: Assume that fill/invalid values are either NaN or we are dealing with masked arrays
        if percent:
            corr_refl = ((refl / 100.) / tOG - rhoray) / TtotraytH2O
        else:
            corr_refl = (refl / tOG - rhoray) / TtotraytH2O
        corr_refl /= (1.0 + corr_refl * sphalb)
        if hasattr(corr_refl, "mask"):
            corr_refl[corr_refl.mask] = 0
        odata.append(corr_refl)

    return odata
