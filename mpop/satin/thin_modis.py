#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010-2011.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Ronald Scheirer

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# mpop is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# mpop.  If not, see <http://www.gnu.org/licenses/>.

"""Interface to Modis level 1b format send through Eumetcast.
http://www.icare.univ-lille1.fr/wiki/index.php/MODIS_geolocation
http://www.sciencedirect.com/science?_ob=MiamiImageURL&_imagekey=B6V6V-4700BJP-3-27&_cdi=5824&_user=671124&_check=y&_orig=search&_coverDate=11%2F30%2F2002&view=c&wchp=dGLzVlz-zSkWz&md5=bac5bc7a4f08007722ae793954f1dd63&ie=/sdarticle.pdf
"""
import os.path
from ConfigParser import ConfigParser

import numpy as np
from pyhdf.SD import SD

from mpop import CONFIG_PATH


def load(satscene, *args, **kwargs):
    """Read data from file and load it into *satscene*.
    """    
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name+"-level2",
                                    raw = True):
        options[option] = value
    CASES[satscene.instrument_name](satscene, options)


def calibrate_refl(subdata, uncertainty):
    uncertainty_array = uncertainty.get()
    array = np.ma.MaskedArray(subdata.get(),
                              mask=(uncertainty_array >= 15))
    valid_range = subdata.attributes()["valid_range"]
    array = np.ma.masked_outside(array,
                                 valid_range[0],
                                 valid_range[1],
                                 copy = False)
    array = array * 1.0
    offsets = subdata.attributes()["reflectance_offsets"]
    scales = subdata.attributes()["reflectance_scales"]
    for i in range(len(scales)):
        array[i, :, :] = (array[i, :, :] - offsets[i]) * scales[i] * 100
    return array

def calibrate_tb(subdata, uncertainty):
    uncertainty_array = uncertainty.get()
    array = np.ma.MaskedArray(subdata.get(),
                              mask=(uncertainty_array >= 15))
    valid_range = subdata.attributes()["valid_range"]
    array = np.ma.masked_outside(array,
                                 valid_range[0],
                                 valid_range[1],
                                 copy = False)
    array = array * 1.0
    offsets = subdata.attributes()["radiance_offsets"]
    scales = subdata.attributes()["radiance_scales"]

    #- Planck constant (Joule second)
    h = 6.6260755e-34

    #- Speed of light in vacuum (meters per second)
    c = 2.9979246e+8

    #- Boltzmann constant (Joules per Kelvin)      
    k = 1.380658e-23

    #- Derived constants      
    c1 = 2.0 * h * c * c
    c2 = (h * c) / k


    #- Effective central wavenumber (inverse centimenters)
    cwn = np.array([\
    2.641775E+3, 2.505277E+3, 2.518028E+3, 2.465428E+3, \
    2.235815E+3, 2.200346E+3, 1.477967E+3, 1.362737E+3, \
    1.173190E+3, 1.027715E+3, 9.080884E+2, 8.315399E+2, \
    7.483394E+2, 7.308963E+2, 7.188681E+2, 7.045367E+2])

    #- Temperature correction slope (no units)
    tcs = np.array([\
    9.993411E-1, 9.998646E-1, 9.998584E-1, 9.998682E-1, \
    9.998819E-1, 9.998845E-1, 9.994877E-1, 9.994918E-1, \
    9.995495E-1, 9.997398E-1, 9.995608E-1, 9.997256E-1, \
    9.999160E-1, 9.999167E-1, 9.999191E-1, 9.999281E-1])

    #- Temperature correction intercept (Kelvin)
    tci = np.array([\
    4.770532E-1, 9.262664E-2, 9.757996E-2, 8.929242E-2, \
    7.310901E-2, 7.060415E-2, 2.204921E-1, 2.046087E-1, \
    1.599191E-1, 8.253401E-2, 1.302699E-1, 7.181833E-2, \
    1.972608E-2, 1.913568E-2, 1.817817E-2, 1.583042E-2])

    # Transfer wavenumber [cm^(-1)] to wavelength [m]
    cwn = 1. / (cwn * 1e2)

    # Due to the thin modis channels selection:
    available_channels = np.array([0, 3, 6, 7, 8, 10, 11, 12])
    cwn = cwn[available_channels]
    tcs = tcs[available_channels]
    tci = tci[available_channels]
    
    band_name = subdata.attributes()["band_names"].split(",")
    for i in range(len(scales)):
        tmp = (array[i, :, :] - offsets[i]) * scales[i]
        tmp = c2 / (cwn[i] * np.ma.log(c1 / (1.e6 * tmp * cwn[i] ** 5.) + 1.))
        array[i, :, :] = (tmp - tci[i]) / tcs[i]
    return array

def load_thin_modis(satscene, options):
    """Read modis data from file and load it into *satscene*.
    """
    filename = satscene.time_slot.strftime("thin_MOD021KM.P%Y%j.%H%M.hdf")
    filename = os.path.join(options["dir"], filename)
    
    data = SD(filename)

    datasets = ['EV_250_Aggr1km_RefSB',
                'EV_500_Aggr1km_RefSB',
                'EV_1KM_RefSB',
                'EV_1KM_Emissive']

    for dataset in datasets:
        subdata = data.select(dataset)
        band_names = subdata.attributes()["band_names"].split(",")
        valid_range = subdata.attributes()["valid_range"]
        if len(satscene.channels_to_load & set(band_names)) > 0:
            uncertainty = data.select(dataset+"_Uncert_Indexes")
            if dataset == 'EV_1KM_Emissive':
                array = calibrate_tb(subdata, uncertainty)
            else:
                array = calibrate_refl(subdata, uncertainty)
            for (i, band) in enumerate(band_names):
                if band in satscene.channels_to_load:
                    banddata = np.ma.masked_outside(array[i],
                                                    valid_range[0],
                                                    valid_range[1],
                                                    copy = False)
            
                    satscene[band] = banddata
def get_lat_lon(satscene, resolution):
    """Read lat and lon.
    """
    del resolution
    
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name+"-level2",
                                    raw = True):
        options[option] = value
        
    return LAT_LON_CASES[satscene.instrument_name](satscene, options)

def get_lat_lon_thin_modis(satscene, options):
    """Read lat and lon.
    """
    filename = satscene.time_slot.strftime("thin_MOD03.P%Y%j.%H%M.hdf")
    filename = os.path.join(options["dir"], filename)

    data = SD(filename)
    lat = data.select("Latitude")
    fill_value = lat.attributes()["_FillValue"]
    lat = np.ma.masked_equal(lat.get(), fill_value)
    lon = data.select("Longitude")
    fill_value = lon.attributes()["_FillValue"]
    lon = np.ma.masked_equal(lon.get(), fill_value)

    

    return lat, lon


CASES = {
    "modis": load_thin_modis
    }

LAT_LON_CASES = {
    "modis": get_lat_lon_thin_modis
    }
