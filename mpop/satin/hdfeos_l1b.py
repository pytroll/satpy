#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010-2012.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Ronald Scheirer <ronald.scheirer@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

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
http://www.sciencedirect.com/science?_ob=MiamiImageURL&_imagekey=B6V6V-4700BJP-\
3-27&_cdi=5824&_user=671124&_check=y&_orig=search&_coverDate=11%2F30%2F2002&vie\
w=c&wchp=dGLzVlz-zSkWz&md5=bac5bc7a4f08007722ae793954f1dd63&ie=/sdarticle.pdf
"""
import os.path
from ConfigParser import ConfigParser

import numpy as np
from pyhdf.SD import SD

from mpop import CONFIG_PATH
from mpop.satin.logger import LOG

# load(["1", "11"], resolution=500)

def load(satscene, *args, **kwargs):
    """Read data from file and load it into *satscene*.
    """
    del args
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name+"-level2",
                                    raw = True):
        options[option] = value
    options["resolution"] = kwargs.get("resolution", 1000)
    CASES[satscene.instrument_name](satscene, options)


def calibrate_refl(subdata, uncertainty, indices):
    """Calibration for reflective channels.
    """
    del uncertainty
    #uncertainty_array = uncertainty.get()
    #array = np.ma.MaskedArray(subdata.get(),
    #                          mask=(uncertainty_array >= 15))

    # FIXME: The loading should not be done here. 

    
    array = np.vstack(np.expand_dims(subdata[idx, :, :], 0) for idx in indices)
    valid_range = subdata.attributes()["valid_range"]
    array = np.ma.masked_outside(array,
                                 valid_range[0],
                                 valid_range[1],
                                 copy = False)
    array = array * np.float32(1.0)
    offsets = np.array(subdata.attributes()["reflectance_offsets"],
                       dtype=np.float32)[indices]
    scales = np.array(subdata.attributes()["reflectance_scales"],
                      dtype=np.float32)[indices]
    dims = (len(indices), 1, 1)
    array = (array - offsets.reshape(dims)) * scales.reshape(dims) * 100
    return array

def calibrate_tb(subdata, uncertainty, indices, band_names):
    """Calibration for the emissive channels.
    """
    del uncertainty
    #uncertainty_array = uncertainty.get()
    #array = np.ma.MaskedArray(subdata.get(),
    #                          mask=(uncertainty_array >= 15))


    # FIXME: The loading should not be done here.
    
    array = np.vstack(np.expand_dims(subdata[idx, :, :], 0) for idx in indices)
    valid_range = subdata.attributes()["valid_range"]
    array = np.ma.masked_outside(array,
                                 valid_range[0],
                                 valid_range[1],
                                 copy = False)
    
    offsets = np.array(subdata.attributes()["radiance_offsets"],
                       dtype=np.float32)[indices]
    scales = np.array(subdata.attributes()["radiance_scales"],
                      dtype=np.float32)[indices]

    #- Planck constant (Joule second)
    h__ = np.float32(6.6260755e-34)

    #- Speed of light in vacuum (meters per second)
    c__ = np.float32(2.9979246e+8)

    #- Boltzmann constant (Joules per Kelvin)      
    k__ = np.float32(1.380658e-23)

    #- Derived constants      
    c_1 = 2 * h__ * c__ * c__
    c_2 = (h__ * c__) / k__


    #- Effective central wavenumber (inverse centimeters)
    cwn = np.array([\
    2.641775E+3, 2.505277E+3, 2.518028E+3, 2.465428E+3, \
    2.235815E+3, 2.200346E+3, 1.477967E+3, 1.362737E+3, \
    1.173190E+3, 1.027715E+3, 9.080884E+2, 8.315399E+2, \
    7.483394E+2, 7.308963E+2, 7.188681E+2, 7.045367E+2],
                   dtype=np.float32)

    #- Temperature correction slope (no units)
    tcs = np.array([\
    9.993411E-1, 9.998646E-1, 9.998584E-1, 9.998682E-1, \
    9.998819E-1, 9.998845E-1, 9.994877E-1, 9.994918E-1, \
    9.995495E-1, 9.997398E-1, 9.995608E-1, 9.997256E-1, \
    9.999160E-1, 9.999167E-1, 9.999191E-1, 9.999281E-1],
                   dtype=np.float32)

    #- Temperature correction intercept (Kelvin)
    tci = np.array([\
    4.770532E-1, 9.262664E-2, 9.757996E-2, 8.929242E-2, \
    7.310901E-2, 7.060415E-2, 2.204921E-1, 2.046087E-1, \
    1.599191E-1, 8.253401E-2, 1.302699E-1, 7.181833E-2, \
    1.972608E-2, 1.913568E-2, 1.817817E-2, 1.583042E-2],
                   dtype=np.float32)

    # Transfer wavenumber [cm^(-1)] to wavelength [m]
    cwn = 1 / (cwn * 100)

    # Some versions of the modis files do not contain all the bands.
    emmissive_channels = ["20", "21", "22", "23", "24", "25", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36"]
    current_channels = [i for i, band in enumerate(emmissive_channels)
                        if band in band_names]
    global_indices = list(np.array(current_channels)[indices])

    dims = (len(indices), 1, 1)
    cwn = cwn[global_indices].reshape(dims)
    tcs = tcs[global_indices].reshape(dims)
    tci = tci[global_indices].reshape(dims)
    
    tmp = (array - offsets.reshape(dims)) * scales.reshape(dims)
    tmp = c_2 / (cwn * np.ma.log(c_1 / (1000000 * tmp * cwn ** 5) + 1))
    array = (tmp - tci) / tcs
    return array

def load_modis(satscene, options):
    """Read modis data from file and load it into *satscene*.

    *resolution* parameters specifies in which resolution to load the data. If
     the specified resolution is not available for the channel, it is NOT
     loaded. If no resolution is specified, the 1km resolution (aggregated) is
     used.
    """
    import glob

    resolution = int(options["resolution"]) or 1000

    filenames = {}
    filename_tmpl = satscene.time_slot.strftime(options["filename"+
                                                        str(resolution)])
    file_list = glob.glob(os.path.join(options["dir"], filename_tmpl))
    if len(file_list) > 1:
        raise IOError("More than 1 file matching!")
    elif len(file_list) == 0:
        raise IOError("No EOS MODIS file matching " +
                      filename_tmpl + " in " +
                      options["dir"])

    load_generic(satscene, file_list[0], resolution)
    
def load_generic(satscene, filename, resolution):
    """Read modis data, generic part.
    """

    data = SD(filename)

    datadict = {
        1000: ['EV_250_Aggr1km_RefSB',
               'EV_500_Aggr1km_RefSB',
               'EV_1KM_RefSB',
               'EV_1KM_Emissive'],
        500: ['EV_250_Aggr500_RefSB',
              'EV_500_RefSB'],
        250: ['EV_250_RefSB']}

    datasets = datadict[resolution]

    
    
    # process by dataset, reflective and emissive datasets separately

    for dataset in datasets:
        subdata = data.select(dataset)
        band_names = subdata.attributes()["band_names"].split(",")
        if len(satscene.channels_to_load & set(band_names)) > 0:
            # get the relative indices of the desired channels
            indices = [i for i, band in enumerate(band_names)
                       if band in satscene.channels_to_load]
            uncertainty = data.select(dataset+"_Uncert_Indexes")
            if dataset.endswith('Emissive'):
                array = calibrate_tb(subdata, uncertainty, indices, band_names)
            else:
                array = calibrate_refl(subdata, uncertainty, indices)
            for (i, idx) in enumerate(indices):
                satscene[band_names[idx]] = array[i]
                # fix the resolution to match the loaded data.
                satscene[band_names[idx]].resolution = resolution


    # Get the orbit number
    mda = data.attributes()["CoreMetadata.0"]
    orbit_idx = mda.index("ORBITNUMBER")
    satscene.orbit = mda[orbit_idx + 111:orbit_idx + 116]
    

    # Get the geolocation
    if resolution != 1000:
        LOG.warning("Cannot load geolocation at this resolution (yet).")
        return
    
    lat, lon = get_lat_lon(satscene, resolution, filename)
    from pyresample import geometry
    satscene.area = geometry.SwathDefinition(lons=lon, lats=lat)

    # Trimming out dead sensor lines (detectors) on aqua:
    # (in addition channel 21 is noisy)
    if satscene.satname == "aqua":
        for band in ["6", "27", "36"]:
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

    # Trimming out dead sensor lines (detectors) on terra:
    # (in addition channel 27, 30, 34, 35, and 36 are nosiy)
    if satscene.satname == "terra":
        for band in ["29"]:
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


    
def get_lat_lon(satscene, resolution, filename):
    """Read lat and lon.
    """
    
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name+"-level2",
                                    raw = True):
        options[option] = value

    options["filename"] = filename
    options["resolution"] = resolution
    return LAT_LON_CASES[satscene.instrument_name](satscene, options)

def get_lat_lon_modis(satscene, options):
    """Read lat and lon.
    """
    import glob
    filename_tmpl = satscene.time_slot.strftime(options["geofile"])
    file_list = glob.glob(os.path.join(options["dir"], filename_tmpl))

    if len(file_list) > 1:
        raise IOError("More than 1 geolocation file matching!")
    elif len(file_list) == 0:
        LOG.warning("No geolocation file matching " + filename_tmpl
                    + " in " + options["dir"])
        LOG.debug("Using 5km geolocation and interpolating")
        filename = options["filename"]
        coarse_resolution = 5000
    else:
        filename = file_list[0]
        coarse_resolution = 1000

    resolution = options["resolution"]
    LOG.debug("Geolocation file = " + filename)
    
    data = SD(filename)
    lat = data.select("Latitude")
    fill_value = lat.attributes()["_FillValue"]
    lat = np.ma.masked_equal(lat.get(), fill_value)
    lon = data.select("Longitude")
    fill_value = lon.attributes()["_FillValue"]
    lon = np.ma.masked_equal(lon.get(), fill_value)

    if resolution == coarse_resolution:
        return lat, lon

    from geotiepoints import modis5kmto1km, modis1kmto500m, modis1kmto250m
    if coarse_resolution == 5000:
        lon, lat = modis5kmto1km(lon, lat)
    if resolution == 500:
        lon, lat = modis1kmto500m(lon, lat)
    if resolution == 250:
        lon, lat = modis1kmto250m(lon, lat)
    
    return lat, lon

def get_lonlat(satscene, row, col):
    """Estimate lon and lat.
    """
    import glob

    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    path = conf.get("modis-level2", "dir")
    geofile_tmpl = conf.get("modis-level2", "geofile")

    filename_tmpl = satscene.time_slot.strftime(geofile_tmpl)
    file_list = glob.glob(os.path.join(path, filename_tmpl))

    if len(file_list) > 1:
        raise IOError("More than 1 geolocation file matching!" + filename_tmpl)
    elif len(file_list) == 0:
        LOG.info("No MODIS geolocation file matching: " + filename_tmpl
                 + ", estimating")
        filename = ""
    else:
        filename = file_list[0]
        LOG.debug("Geolocation file = " + filename)

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
        if(satscene.lon.mask[row, col] == False and
           satscene.lat.mask[row, col] == False):
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
    f__ = 1/298.257223563
    # semi_major_axis
    a__ = 6378137.0

    s__, alpha12, alpha21 = vinc_dist(f__, a__,
                                      track_start[0], track_start[1],
                                      track_end[0], track_end[1])
    scanlines = satscene.granularity.seconds / satscene.span

    if row < scanlines/2:
        if row == 0:
            track_now = track_start
        else:
            track_now = vinc_pt(f__, a__, track_start[0], track_start[1],
                                alpha12, (s__ * row) / scanlines)
        lat_now = track_now[0]
        lon_now = track_now[1]
        
        s__, alpha12, alpha21 = vinc_dist(f__, a__,
                                        lat_now, lon_now,
                                        track_end[0], track_end[1])
        fac = 1
    else:
        if scanlines - row - 1 == 0:
            track_now = track_end
        else:
            track_now = vinc_pt(f__, a__, track_end[0], track_end[1], alpha21,
                                (s__ * (scanlines - row - 1)) / scanlines)
        lat_now = track_now[0]
        lon_now = track_now[1]
        
        s__, alpha12, alpha21 = vinc_dist(f__, a__,
                                        lat_now, lon_now,
                                        track_start[0], track_start[1])
        fac = -1

    if col < 1354/2:
        lat, lon, alp = vinc_pt(f__, a__, lat_now, lon_now,
                                alpha12 + np.pi/2 * fac,
                                2340000.0 / 2 - (2340000.0/1354) * col)
    else:
        lat, lon, alp = vinc_pt(f__, a__, lat_now, lon_now,
                                alpha12 - np.pi/2 * fac,
                                (2340000.0/1354) * col - 2340000.0 / 2)
        del alp
    lon = np.rad2deg(lon)
    lat = np.rad2deg(lat)

    if lon > 180:
        lon -= 360
    if lon <= -180:
        lon += 360
    
    return lon, lat
        
    
import math

def vinc_dist(f__, a__, phi1, lembda1, phi2, lembda2):
    """ 

    Returns the distance between two geographic points on the ellipsoid
    and the forward and reverse azimuths between these points.
    lats, longs and azimuths are in radians, distance in metres 

    Returns ( s__, alpha12,  alpha21 ) as a tuple

    """

    if (abs( phi2 - phi1 ) < 1e-8) and ( abs( lembda2 - lembda1) < 1e-8 ) :
        return 0.0, 0.0, 0.0

    two_pi = 2.0*math.pi

    b__ = a__ * (1.0 - f__)

    tan_u1 = (1 - f__) * math.tan(phi1)
    tan_u2 = (1 - f__) * math.tan(phi2)

    u_1 = math.atan(tan_u1)
    u_2 = math.atan(tan_u2)

    lembda = lembda2 - lembda1
    last_lembda = -4000000.0                # an impossibe value
    omega = lembda

    # Iterate the following equations, 
    #  until there is no significant change in lembda 

    while (last_lembda < -3000000.0 or
           lembda != 0 and
           abs( (last_lembda - lembda)/lembda) > 1.0e-9):

        sqr_sin_sigma = (pow( math.cos(u_2) * math.sin(lembda), 2) +
                         pow( (math.cos(u_1) * math.sin(u_2) -
                               math.sin(u_1) *  math.cos(u_2) *
                               math.cos(lembda) ), 2 ))

        sin_sigma = math.sqrt(sqr_sin_sigma)
        
        cos_sigma = (math.sin(u_1) * math.sin(u_2) +
                     math.cos(u_1) * math.cos(u_2) * math.cos(lembda))

        sigma = math.atan2(sin_sigma, cos_sigma)

        sin_alpha = (math.cos(u_1) * math.cos(u_2) * math.sin(lembda) /
                     math.sin(sigma))
        alpha = math.asin(sin_alpha)
      
        cos2sigma_m = (math.cos(sigma) -
                       (2 * math.sin(u_1) * math.sin(u_2) /
                        pow(math.cos(alpha), 2)))

        c__ = ((f__ / 16) * pow(math.cos(alpha), 2) *
               (4 + f__ * (4 - 3 * pow(math.cos(alpha), 2))))

        last_lembda = lembda
        
        lembda = (omega + (1-c__) * f__ * math.sin(alpha) *
                  (sigma + c__ * math.sin(sigma) * 
                   (cos2sigma_m + c__ * math.cos(sigma) *
                    (-1 + 2 * pow(cos2sigma_m, 2)))))


    u2_ = pow(math.cos(alpha), 2) * (a__ * a__- b__ * b__) / (b__ * b__)

    aa_ = 1 + (u2_/16384) * (4096 + u2_ * (-768 + u2_ * (320 - 175 * u2_)))

    bb_ = (u2_/1024) * (256 + u2_ * (-128+ u2_ * (74 - 47 * u2_)))

    delta_sigma = bb_ * sin_sigma * (cos2sigma_m + (bb_/4) * \
            (cos_sigma * (-1 + 2 * pow(cos2sigma_m, 2) ) - \
            (bb_/6) * cos2sigma_m * (-3 + 4 * sqr_sin_sigma) * \
            (-3 + 4 * pow(cos2sigma_m,2 ) )))

    s__ = b__ * aa_ * (sigma - delta_sigma)

    alpha12 = (math.atan2((math.cos(u_2) * math.sin(lembda)), 
                          (math.cos(u_1) * math.sin(u_2) -
                           math.sin(u_1) * math.cos(u_2) * math.cos(lembda))))

    alpha21 = (math.atan2( (math.cos(u_1) * math.sin(lembda)),
                           (-math.sin(u_1) * math.cos(u_2) +
                            math.cos(u_1) * math.sin(u_2) * math.cos(lembda))))

    if ( alpha12 < 0.0 ) : 
        alpha12 =  alpha12 + two_pi
    if ( alpha12 > two_pi ) : 
        alpha12 = alpha12 - two_pi
        
    alpha21 = alpha21 + two_pi / 2.0
    if ( alpha21 < 0.0 ) : 
        alpha21 = alpha21 + two_pi
    if ( alpha21 > two_pi ) : 
        alpha21 = alpha21 - two_pi
            
    return s__, alpha12,  alpha21 

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

def  vinc_pt( f__, a__, phi1, lembda1, alpha12, s__) :
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


    b__ = a__ * (1.0 - f__)

    tan_u1 = (1-f__) * math.tan(phi1)
    u_1 = math.atan( tan_u1 )
    sigma1 = math.atan2( tan_u1, math.cos(alpha12) )
    sinalpha = math.cos(u_1) * math.sin(alpha12)
    cosalpha_sq = 1.0 - sinalpha * sinalpha

    u_2 = cosalpha_sq * (a__ * a__ - b__ * b__ ) / (b__ * b__)
    aa_ = 1.0 + (u_2 / 16384) * (4096 + u_2 * (-768 + u_2 * \
            (320 - 175 * u_2) ) )
    bb_ = (u_2 / 1024) * (256 + u_2 * (-128 + u_2 * (74 - 47 * u_2) ) )

    # Starting with the approximation
    sigma = (s__ / (b__ * aa_))

    last_sigma = 2.0 * sigma + 2.0  # something impossible

    # Iterate the following three equations 
    # until there is no significant change in sigma 

    # two_sigma_m , delta_sigma

    while ( abs( (last_sigma - sigma) / sigma) > 1.0e-9 ) :

        two_sigma_m = 2 * sigma1 + sigma

        delta_sigma = (bb_ * math.sin(sigma) *
                       (math.cos(two_sigma_m)
                        + (bb_/4) * (math.cos(sigma) *
                                     (-1 + 2 * math.pow(math.cos(two_sigma_m),
                                                        2)
                                      - (bb_ / 6) * math.cos(two_sigma_m) *
                                      (-3 + 4 * math.pow(math.sin(sigma), 2 )) *
                                      (-3 + 4 * math.pow(math.cos (two_sigma_m),
                                                         2 ))))))
        last_sigma = sigma
        sigma = (s__ / (b__ * aa_)) + delta_sigma


    phi2 = math.atan2((math.sin(u_1) * math.cos(sigma) +
                       math.cos(u_1) * math.sin(sigma) * math.cos(alpha12)), 
                      ((1 - f__) * math.sqrt(math.pow(sinalpha, 2) +
                                             pow(math.sin(u_1) *
                                                 math.sin(sigma) -
                                                 math.cos(u_1) *
                                                 math.cos(sigma) *
                                                 math.cos(alpha12), 2))))


    lembda = math.atan2((math.sin(sigma) * math.sin(alpha12 )),
                        (math.cos(u_1) * math.cos(sigma) -
                         math.sin(u_1) *  math.sin(sigma) * math.cos(alpha12)))

    cc_ = (f__/16) * cosalpha_sq * (4 + f__ * (4 - 3 * cosalpha_sq ))

    omega = lembda - (1-cc_) * f__ * sinalpha *  \
            (sigma + cc_*math.sin(sigma)*(math.cos(two_sigma_m) +
                                        cc_ * math.cos(sigma) *
                                        (-1 + 2 *
                                         math.pow(math.cos(two_sigma_m),2))))

    lembda2 = lembda1 + omega

    alpha21 = math.atan2 ( sinalpha, (-math.sin(u_1) * math.sin(sigma) +
                                      math.cos(u_1) * math.cos(sigma) *
                                      math.cos(alpha12)))

    alpha21 = alpha21 + two_pi / 2.0
    if ( alpha21 < 0.0 ) :
        alpha21 = alpha21 + two_pi
    if ( alpha21 > two_pi ) :
        alpha21 = alpha21 - two_pi


    return phi2,  lembda2,  alpha21 

# END of Vincenty's Direct formulae




CASES = {
    "modis": load_modis
    }

LAT_LON_CASES = {
    "modis": get_lat_lon_modis
    }
