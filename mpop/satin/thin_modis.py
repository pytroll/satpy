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
import glob
from ConfigParser import ConfigParser

import numpy as np
from pyhdf.SD import SD

from mpop import CONFIG_PATH
from mpop.satin.logger import LOG

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
    #array = np.ma.MaskedArray(subdata.get(),
    #                          mask=(uncertainty_array >= 15))
    array = subdata.get()
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
    #array = np.ma.MaskedArray(subdata.get(),
    #                          mask=(uncertainty_array >= 15))
    array = subdata.get()
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
    if satscene.satellite == "aqua":
        filename = satscene.time_slot.strftime("thin_MYD021KM.A%Y%j.%H%M.005.*NRT.hdf")
    elif(satscene.satellite == "terra"):
        filename = satscene.time_slot.strftime("thin_MOD021KM.A%Y%j.%H%M.005.*NRT.hdf")
    filenames = glob.glob(os.path.join(options["dir"], filename))
    if len(filenames) != 1:
        raise IOError("There should be one and only one " +
                      os.path.join(options["dir"], filename))
    filename = filenames[0]
    
    data = SD(filename)

    datasets = ['EV_250_Aggr1km_RefSB',
                'EV_500_Aggr1km_RefSB',
                'EV_1KM_RefSB',
                'EV_1KM_Emissive']

    for dataset in datasets:
        subdata = data.select(dataset)
        band_names = subdata.attributes()["band_names"].split(",")
        if len(satscene.channels_to_load & set(band_names)) > 0:
            uncertainty = data.select(dataset+"_Uncert_Indexes")
            if dataset == 'EV_1KM_Emissive':
                array = calibrate_tb(subdata, uncertainty)
            else:
                array = calibrate_refl(subdata, uncertainty)
            for (i, band) in enumerate(band_names):
                if band in satscene.channels_to_load:
                    satscene[band] = array[i]

    mda = data.attributes()["CoreMetadata.0"]
    orbit_idx = mda.index("ORBITNUMBER")
    satscene.orbit = mda[orbit_idx + 111:orbit_idx + 116]

    lat, lon = get_lat_lon(satscene, None)
    from pyresample import geometry
    satscene.area = geometry.SwathDefinition(lons=lon, lats=lat)

    # trimming out dead sensor lines
    if satscene.satname == "aqua":
        for band in ["6", "27"]:
            if not satscene[band].is_loaded() or satscene[band].data.mask.all():
                continue
            width = satscene[band].data.shape[1]
            height = satscene[band].data.shape[0]
            indices = satscene[band].data.mask.sum(1) < width
            if indices.sum() == height:
                continue
            satscene[band] = satscene[band].data[indices, :]
            satscene[band].area = geometry.SwathDefinition(
                lons=satscene.area.lons[indices,:],
                lats=satscene.area.lats[indices,:])
            satscene[band].area.area_id = ("swath_" + satscene.fullname + "_"
                                           + str(satscene.time_slot) + "_"
                                           + str(satscene[band].shape) + "_"
                                           + str(band))


    
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
    if satscene.satellite == "aqua":
        filename = satscene.time_slot.strftime("thin_MYD03.A%Y%j.%H%M.005.*NRT.hdf")
    elif(satscene.satellite == "terra"):
        filename = satscene.time_slot.strftime("thin_MOD03.A%Y%j.%H%M.005.*NRT.hdf")
    filenames = glob.glob(os.path.join(options["dir"], filename))
    if len(filenames) != 1:
        raise IOError("There should be one and only one " +
                      os.path.join(options["dir"], filename))
    filename = filenames[0]

    data = SD(filename)
    lat = data.select("Latitude")
    fill_value = lat.attributes()["_FillValue"]
    lat = np.ma.masked_equal(lat.get(), fill_value)
    lon = data.select("Longitude")
    fill_value = lon.attributes()["_FillValue"]
    lon = np.ma.masked_equal(lon.get(), fill_value)

    

    return lat, lon

def get_lonlat(satscene, row, col):
    """Estimate lon and lat.
    """

    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    path = conf.get("modis-level2", "dir")

    if satscene.satellite == "aqua":
        filename = satscene.time_slot.strftime("thin_MYD03.A%Y%j.%H%M.005.*NRT.hdf")
    elif(satscene.satellite == "terra"):
        filename = satscene.time_slot.strftime("thin_MOD03.A%Y%j.%H%M.005.*NRT.hdf")
    filenames = glob.glob(os.path.join(path, filename))
    if len(filenames) != 1:
        raise IOError("There should be one and only one " +
                      os.path.join(path, filename))
    filename = filenames[0]
    
    if(os.path.exists(filename) and
       (satscene.lon is None or satscene.lat is None)):
        data = SD(filename)
        lat = data.select("Latitude")
        fill_value = lat.attributes()["_FillValue"]
        satscene.lat = np.ma.masked_equal(lat.get(), fill_value)
        lon = data.select("Longitude")
        fill_value = lon.attributes()["_FillValue"]
        satscene.lon = np.ma.masked_equal(lon.get(), fill_value)

    estimate = True

    try:
        lon = satscene.lon[row, col]
        lat = satscene.lat[row, col]
        if satscene.lon.mask[row, col] == False and satscene.lat.mask[row, col] == False:
            estimate = False
    except TypeError:
        pass
    except IndexError:
        pass

    if not estimate:
        return lon, lat

    from mpop.saturn.two_line_elements import Tle

    tle = Tle(satellite=satscene.satname)
    track_start = tle.get_latlonalt(satscene.time_slot)
    track_end = tle.get_latlonalt(satscene.time_slot + satscene.granularity)

    # WGS84
    # flattening
    f = 1/298.257223563
    # semi_major_axis
    a = 6378137.0

    s, alpha12, alpha21 = vinc_dist(f, a,
                                    track_start[0], track_start[1],
                                    track_end[0], track_end[1])
    scanlines = satscene.granularity.seconds / satscene.span

    if row < scanlines/2:
        if row == 0:
            track_now = track_start
        else:
            track_now = vinc_pt(f, a, track_start[0], track_start[1], alpha12,
                                (s * row) / scanlines)
        lat_now = track_now[0]
        lon_now = track_now[1]
        
        s, alpha12, alpha21 = vinc_dist(f, a,
                                        lat_now, lon_now,
                                        track_end[0], track_end[1])
        fac = 1
    else:
        if scanlines - row - 1 == 0:
            track_now = track_end
        else:
            track_now = vinc_pt(f, a, track_end[0], track_end[1], alpha21,
                                (s * (scanlines - row - 1)) / scanlines)
        lat_now = track_now[0]
        lon_now = track_now[1]
        
        s, alpha12, alpha21 = vinc_dist(f, a,
                                        lat_now, lon_now,
                                        track_start[0], track_start[1])
        fac = -1

    if col < 1354/2:
        lat, lon, alp = vinc_pt(f, a, lat_now, lon_now, alpha12 + np.pi/2 * fac,
                                2340000.0 / 2 - (2340000.0/1354) * col)
    else:
        lat, lon, alp = vinc_pt(f, a, lat_now, lon_now, alpha12 - np.pi/2 * fac,
                                (2340000.0/1354) * col - 2340000.0 / 2)

    lon = np.rad2deg(lon)
    lat = np.rad2deg(lat)

    if lon > 180:
        lon -= 360
    if lon <= -180:
        lon += 360
    
    return lon, lat
        
    
import math

def vinc_dist(  f,  a,  phi1,  lembda1,  phi2,  lembda2 ) :
    """ 

    Returns the distance between two geographic points on the ellipsoid
    and the forward and reverse azimuths between these points.
    lats, longs and azimuths are in radians, distance in metres 

    Returns ( s, alpha12,  alpha21 ) as a tuple

    """

    if (abs( phi2 - phi1 ) < 1e-8) and ( abs( lembda2 - lembda1) < 1e-8 ) :
        return 0.0, 0.0, 0.0

    two_pi = 2.0*math.pi

    b = a * (1.0 - f)

    TanU1 = (1-f) * math.tan( phi1 )
    TanU2 = (1-f) * math.tan( phi2 )

    U1 = math.atan(TanU1)
    U2 = math.atan(TanU2)

    lembda = lembda2 - lembda1
    last_lembda = -4000000.0                # an impossibe value
    omega = lembda

    # Iterate the following equations, 
    #  until there is no significant change in lembda 

    while ( last_lembda < -3000000.0 or lembda != 0 and abs( (last_lembda - lembda)/lembda) > 1.0e-9 ) :

        sqr_sin_sigma = pow( math.cos(U2) * math.sin(lembda), 2) + \
                        pow( (math.cos(U1) * math.sin(U2) - \
                              math.sin(U1) *  math.cos(U2) * math.cos(lembda) ), 2 )

        Sin_sigma = math.sqrt( sqr_sin_sigma )
        
        Cos_sigma = math.sin(U1) * math.sin(U2) + math.cos(U1) * math.cos(U2) * math.cos(lembda)

        sigma = math.atan2( Sin_sigma, Cos_sigma )

        Sin_alpha = math.cos(U1) * math.cos(U2) * math.sin(lembda) / math.sin(sigma)
        alpha = math.asin( Sin_alpha )
      
        Cos2sigma_m = math.cos(sigma) - (2 * math.sin(U1) * math.sin(U2) / pow(math.cos(alpha), 2) )

        C = (f/16) * pow(math.cos(alpha), 2) * (4 + f * (4 - 3 * pow(math.cos(alpha), 2)))

        last_lembda = lembda
        
        lembda = omega + (1-C) * f * math.sin(alpha) * (sigma + C * math.sin(sigma) * \
                                                        (Cos2sigma_m + C * math.cos(sigma) * (-1 + 2 * pow(Cos2sigma_m, 2) )))


    u2 = pow(math.cos(alpha),2) * (a*a-b*b) / (b*b)

    A = 1 + (u2/16384) * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))

    B = (u2/1024) * (256 + u2 * (-128+ u2 * (74 - 47 * u2)))

    delta_sigma = B * Sin_sigma * (Cos2sigma_m + (B/4) * \
            (Cos_sigma * (-1 + 2 * pow(Cos2sigma_m, 2) ) - \
            (B/6) * Cos2sigma_m * (-3 + 4 * sqr_sin_sigma) * \
            (-3 + 4 * pow(Cos2sigma_m,2 ) )))

    s = b * A * (sigma - delta_sigma)

    alpha12 = math.atan2( (math.cos(U2) * math.sin(lembda)), \
            (math.cos(U1) * math.sin(U2) - math.sin(U1) * math.cos(U2) * math.cos(lembda)))

    alpha21 = math.atan2( (math.cos(U1) * math.sin(lembda)), \
            (-math.sin(U1) * math.cos(U2) + math.cos(U1) * math.sin(U2) * math.cos(lembda)))

    if ( alpha12 < 0.0 ) : 
        alpha12 =  alpha12 + two_pi
    if ( alpha12 > two_pi ) : 
        alpha12 = alpha12 - two_pi
        
    alpha21 = alpha21 + two_pi / 2.0
    if ( alpha21 < 0.0 ) : 
        alpha21 = alpha21 + two_pi
    if ( alpha21 > two_pi ) : 
        alpha21 = alpha21 - two_pi
            
    return s, alpha12,  alpha21 

# END of Vincenty's Inverse formulae 


#----------------------------------------------------------------------------
# Vincenty's Direct formulae                                                |
# Given: latitude and longitude of a point (phi1, lembda1) and              |
# the geodetic azimuth (alpha12)                                            |
# and ellipsoidal distance in metres (s) to a second point,                 |
#                                                                           |
# Calculate: the latitude and longitude of the second point (phi2, lembda2) |
# and the reverse azimuth (alpha21).                                        |
#                                                                           |
#----------------------------------------------------------------------------

def  vinc_pt( f, a, phi1, lembda1, alpha12, s ) :
    """

    Returns the lat and long of projected point and reverse azimuth
    given a reference point and a distance and azimuth to project.
    lats, longs and azimuths are passed in decimal degrees

    Returns ( phi2,  lambda2,  alpha21 ) as a tuple 

    """


    two_pi = 2.0*math.pi

    if ( alpha12 < 0.0 ) : 
        alpha12 = alpha12 + two_pi
    if ( alpha12 > two_pi ) : 
        alpha12 = alpha12 - two_pi


    b = a * (1.0 - f)

    TanU1 = (1-f) * math.tan(phi1)
    U1 = math.atan( TanU1 )
    sigma1 = math.atan2( TanU1, math.cos(alpha12) )
    Sinalpha = math.cos(U1) * math.sin(alpha12)
    cosalpha_sq = 1.0 - Sinalpha * Sinalpha

    u2 = cosalpha_sq * (a * a - b * b ) / (b * b)
    A = 1.0 + (u2 / 16384) * (4096 + u2 * (-768 + u2 * \
            (320 - 175 * u2) ) )
    B = (u2 / 1024) * (256 + u2 * (-128 + u2 * (74 - 47 * u2) ) )

    # Starting with the approximation
    sigma = (s / (b * A))

    last_sigma = 2.0 * sigma + 2.0  # something impossible

    # Iterate the following three equations 
    # until there is no significant change in sigma 

    # two_sigma_m , delta_sigma

    while ( abs( (last_sigma - sigma) / sigma) > 1.0e-9 ) :

        two_sigma_m = 2 * sigma1 + sigma

        delta_sigma = B * math.sin(sigma) * ( math.cos(two_sigma_m) \
                                              + (B/4) * (math.cos(sigma) * \
                                                         (-1 + 2 * math.pow( math.cos(two_sigma_m), 2 ) -  \
                                                          (B/6) * math.cos(two_sigma_m) * \
                                                          (-3 + 4 * math.pow(math.sin(sigma), 2 )) *  \
                                                          (-3 + 4 * math.pow( math.cos (two_sigma_m), 2 )))))
        last_sigma = sigma
        sigma = (s / (b * A)) + delta_sigma


    phi2 = math.atan2 ( (math.sin(U1) * math.cos(sigma) + math.cos(U1) * math.sin(sigma) * math.cos(alpha12) ), \
                        ((1-f) * math.sqrt( math.pow(Sinalpha, 2) +  \
                                            pow(math.sin(U1) * math.sin(sigma) - math.cos(U1) * math.cos(sigma) * math.cos(alpha12), 2))))


    lembda = math.atan2( (math.sin(sigma) * math.sin(alpha12 )), (math.cos(U1) * math.cos(sigma) -  \
                                                                  math.sin(U1) *  math.sin(sigma) * math.cos(alpha12)))

    C = (f/16) * cosalpha_sq * (4 + f * (4 - 3 * cosalpha_sq ))

    omega = lembda - (1-C) * f * Sinalpha *  \
            (sigma + C*math.sin(sigma)*(math.cos(two_sigma_m) +
                                        C * math.cos(sigma) *
                                        (-1 + 2 *
                                         math.pow(math.cos(two_sigma_m),2))))

    lembda2 = lembda1 + omega

    alpha21 = math.atan2 ( Sinalpha, (-math.sin(U1) * math.sin(sigma) +
                                      math.cos(U1) * math.cos(sigma) *
                                      math.cos(alpha12)))

    alpha21 = alpha21 + two_pi / 2.0
    if ( alpha21 < 0.0 ) :
        alpha21 = alpha21 + two_pi
    if ( alpha21 > two_pi ) :
        alpha21 = alpha21 - two_pi


    return phi2,  lembda2,  alpha21 

# END of Vincenty's Direct formulae




CASES = {
    "modis": load_thin_modis
    }

LAT_LON_CASES = {
    "modis": get_lat_lon_thin_modis
    }
