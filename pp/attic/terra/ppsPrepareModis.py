#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):

#   Ronald Scheirer <ronald.scheirer@smhi.se>
#   Martin Raspaud <martin.raspaud@smhi.se>

# This file is part of the mpop.

# mpop is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# mpop is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with mpop.  If not, see <http://www.gnu.org/licenses/>.

import _pyhl
import string
import sys
import math
import numpy as np


# select the amount of output (0 = nothing, 9 = an awful lot, I tell ya).
# This is the standard for the batch-mode which should be always set to 0. 
# For tracing errors on isolated program change value in the main part at
# the end of this file.
globals()["debug"] = 0

# Pick the channel indices you want to be saved here (starts with 0)
selection = np.array([0, 1, 5, 12, 13, 18, 21, 30, 32, 33])

# Names of available MODIS channels
CHNames = ("1","2","3","4","5","6","7","8","9","10","11","12","13hi",
                 "13lo","14hi","14lo","15","16","17","18","19","20","21",
                 "22","23","24","25","26","27","28","29","30","31","32","33",
                 "34","35","36")

FILE = ""
FILE2 = ""

# The maximum level of data-compression
COMPRESS_LVL = 6

# Some universal values
deg2rad = math.pi / 180.
rad2deg = 180. / math.pi

def short_scale(sca, off, tmp, err, u_i):
    """do the scaling needed in the shortwave range
    calculate channel by channel
    *sca* = scaling factor
    *off* = scaling offset
    *tmp* = scaled integer, representing the measured quantity
    *err* = error indicator
    *u_i*  = uncertainty index
    """
    
    tot = tmp.shape[1] * tmp.shape[2]
    res = tmp[:, :, :]
    
    if debug > 1:
        print
        print "** Function  short_scale **"
    i = 0

    while i < len(sca):

        tmp = res[i, :, :]
        res[i, :, :] = np.where(np.logical_or(
            np.logical_or(np.greater(err[i], 65535),# was 32767
                          np.equal(np.bitwise_and(u_i[i], 31), 31)),
            np.logical_or(np.greater(tmp, 65535), np.less(tmp, 0))),# was 32767
                                -99.9,(tmp - off[i]) * sca[i])
        i = i + 1
        if debug > 0:
            tmp = np.where(np.less(res[i-1, :, :], -9), 1, 0)
            count = sum(sum(tmp))
            proz = 100. * float(count) / float(tot)
            print i, '\t', count, '\t\t\t', proz
        
    # return scaled result
    return res


def calcRadi250Agg(scaled_ints, err, u_i):
    """Calculate shortwave radiances by linear scaling of the scaled integers.
    """

    if debug > 1:
        print
        print "** Function  calcRadi250Agg **"
    a = _pyhl.read_nodelist(FILE)
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_250_Aggr1km_RefSB/radiance_offsets")
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_250_Aggr1km_RefSB/radiance_scales")
    a.fetch()

    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_250_Aggr1km_RefSB/radiance_offsets")
    off = b.data()
    if debug > 2:
        print "Offset-list:"
        print off
    
    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_250_Aggr1km_RefSB/radiance_scales")
    sca = b.data()
    if debug > 2:
        print "Scaling-list:"
        print sca

    if debug > 0:
        print "-"*50
        print "Short Wave rad,250m (agg.),serial CH numbers"
        print "Ch   Missing pixels       \t%"

    scaled_ints = short_scale(sca, off, scaled_ints, err, u_i)

    return scaled_ints


def calcRadi500Agg(scaled_ints, err, u_i):
    """Calculate shortwave radiances by linear scaling of the scaled integers.
    """

    if debug > 1:
        print
        print "** Function  calcRadi500Agg **"
    a = _pyhl.read_nodelist(FILE)
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_500_Aggr1km_RefSB/radiance_offsets")
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_500_Aggr1km_RefSB/radiance_scales")
    a.fetch()

    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_500_Aggr1km_RefSB/radiance_offsets")
    off = b.data()
    if debug > 2:
        print "Offset-list:"
        print off
    
    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_500_Aggr1km_RefSB/radiance_scales")
    sca = b.data()
    if debug > 2:
        print "Scaling-list:"
        print sca

    if debug > 0:
        print "-"*50
        print "Short Wave rad,500m (agg.),serial CH numbers"
        print "Ch   Missing pixels       \t%"

    scaled_ints = short_scale(sca, off, scaled_ints, err, u_i)

    return scaled_ints


def calcRadi1000(scaled_ints, err, u_i):
    """Calculate shortwave radiances by linear scaling of the scaled integers.
    """

    if debug > 1:
        print
        print "** Function  calcRadi1000 **"
    a = _pyhl.read_nodelist(FILE)
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_RefSB/radiance_offsets")
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_RefSB/radiance_scales")
    a.fetch()

    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_RefSB/radiance_offsets")
    off = b.data()
    if debug > 2:
        print "Offset-list:"
        print off
    
    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_RefSB/radiance_scales")
    sca = b.data()
    if debug > 2:
        print "Scaling-list:"
        print sca

    if debug > 0:
        print "-"*50
        print "Short Wave rad,1000m,serial CH numbers"
        print "Ch   Missing pixels       \t%"

    scaled_ints = short_scale(sca, off, scaled_ints, err, u_i)

    return scaled_ints


def calcRefl250Agg(scaled_ints, err, u_i):
    """Calculate shortwave reflectances by linear scaling of the scaled
    integers.
    """

    if debug > 1:
        print
        print "** Function  calcRefl250Agg **"
    a = _pyhl.read_nodelist(FILE)
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_250_Aggr1km_RefSB/reflectance_offsets")
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_250_Aggr1km_RefSB/reflectance_scales")
    a.fetch()

    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_250_Aggr1km_RefSB/reflectance_offsets")
    off = b.data()
    if debug > 2:
        print "Offset-list:"
        print off
    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_250_Aggr1km_RefSB/reflectance_scales")
    sca = b.data()
    if debug > 2:
        print "Scaling-list:"
        print sca

    if debug > 0:
        print "-"*50
        print "Short Wave REFL, 250m (agg.), serial CH numbers"
        print "Ch   Missing pixels      \t%"

    scaled_ints = short_scale(sca, off, scaled_ints, err, u_i)

    return scaled_ints


def calcRefl500Agg(scaled_ints, err, u_i):
    """Calculate shortwave reflectances by linear scaling of the scaled
    integers.
    """

    if debug > 1:
        print
        print "** Function  calcRefl500Agg **"

    a = _pyhl.read_nodelist(FILE)
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_500_Aggr1km_RefSB/reflectance_offsets")
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_500_Aggr1km_RefSB/reflectance_scales")
    a.fetch()

    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_500_Aggr1km_RefSB/reflectance_offsets")
    off = b.data()
    if debug > 2:
        print "Offset-list:"
        print off

    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_500_Aggr1km_RefSB/reflectance_scales")
    sca = b.data()
    if debug > 2:
        print "Scaling-list:"
        print sca

    if debug > 0:
        print "-"*50
        print "Short Wave REFL, 500m (agg.), serial CH numbers"
        print "Ch   Missing pixels      \t%"

    scaled_ints = short_scale(sca, off, scaled_ints, err, u_i)

    return scaled_ints


def calcRefl1000(scaled_ints, err, u_i):
    """Calculate shortwave reflectances by linear scaling of the scaled
    integers.
    """

    if debug > 1:
        print
        print "** Function  calcRefl1000 **"

    a = _pyhl.read_nodelist(FILE)
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_RefSB/reflectance_offsets")
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_RefSB/reflectance_scales")
    a.fetch()

    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_RefSB/reflectance_offsets")
    off = b.data()
    if debug > 2:
        print "Offset-list:"
        print off

    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_RefSB/reflectance_scales")
    sca = b.data()
    if debug > 2:
        print "Scaling-list:"
        print sca

    if debug > 0:
        print "-"*50
        print "Short Wave REFL, 1000m, serial CH numbers"
        print "Ch   Missing pixels      \t%"

    scaled_ints = short_scale(sca, off, scaled_ints, err, u_i)

    return scaled_ints


def getSI250Agg(filename):
    """Read scaled integers (SI) and related data for further processing of
    250m (agg.) reflectivity channels.
    """
    
    if debug > 1:
        print
        print "** Function  getSI250Agg **"

    global FILE
    FILE = filename
    a = _pyhl.read_nodelist(filename)
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_250_Aggr1km_RefSB")
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_250_Aggr1km_RefSB_Uncert_Indexes")
    a.fetch()

    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_250_Aggr1km_RefSB")
    # keep original integers to allow for error control
    err = b.data()
    err = err.astype(np.uint16)
    # float-array for further calculations
    si = err * 1.
    if debug > 2:
        print "Shape of measurement array:"
        print si.shape

    # get uncertainty indices
    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_250_Aggr1km_RefSB_Uncert_Indexes")
    #print b.dims()
    u_i = b.data()
    if debug > 2:
        print "Shape of uncertainty index array:"
        print u_i.shape

    u_i = u_i.astype(np.uint8)


    # return scaled integers, error flag, and uncertaity indices
    return si, err, u_i


def getSI500Agg(filename):
    """Read scaled integers (SI) and related data for further processing of
    500m (agg.) reflectivity channels.
    """
    
    if debug > 1:
        print
        print "** Function  getSI500Agg **"

    global FILE
    FILE = filename
    a = _pyhl.read_nodelist(filename)
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_500_Aggr1km_RefSB")
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_500_Aggr1km_RefSB_Uncert_Indexes")
    a.fetch()

    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_500_Aggr1km_RefSB")
    # keep original integers to allow for error control
    err = b.data()
    err = err.astype(np.uint16)
    # float-array for further calculations
    si = err*1.
    if debug > 2:
        print "Shape of measurement array:"
        print si.shape

    # get uncertainty indices
    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_500_Aggr1km_RefSB_Uncert_Indexes")
    #print b.dims()
    u_i = b.data()
    if debug > 2:
        print "Shape of uncertainty index array:"
        print u_i.shape

    u_i = u_i.astype(np.uint8)


    # return scaled integers, error flag, and uncertaity indices
    return si, err, u_i


def getSI1000(filename):
    """Read scaled integers (SI) and related data for further processing of 1km
    reflectivity channels.
    """
    if debug > 1:
        print
        print "** Function  getSI1000 **"

    global FILE
    FILE = filename
    a = _pyhl.read_nodelist(filename)
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_RefSB")
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_RefSB_Uncert_Indexes")
    a.fetch()

    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_RefSB")
    # keep original integers to allow for error control
    err = b.data()
    err = err.astype(np.uint16)
    # float-array for further calculations
    si = err*1.
    if debug > 2:
        print "Shape of measurement array:"
        print si.shape

    # get uncertainty indices
    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_RefSB_Uncert_Indexes")
    #print b.dims()
    u_i = b.data()
    if debug > 2:
        print "Shape of uncertainty index array:"
        print u_i.shape

    u_i = u_i.astype(np.uint8)


    # return scaled integers, error flag, and uncertaity indices
    return si, err, u_i


def getREFL250Agg(filename):

    if debug > 1:
        print
        print "** Function  getREFL250Agg **"

    global FILE
    FILE = filename
    print "file is"
    print FILE
    # create dummy arrays
    rad = np.array([[0., 0.], [0., 0.]])
    refl = np.array([[0., 0.], [0., 0.]])
    # extract requested data from HDF file
    # and calculate corrected counts
    si, err, u_i = getSI250Agg(filename)
    if debug > 2:
        print "Shape of scaled integer array:"
        print si.shape

    # calculate shortwave radiance by linear scaling
    # uncomment the following line if radiances are needed
    #rad = calcRadi250Agg(si, err, u_i)
    if debug > 2:
        print "Shape of radiances array:"
        print rad.shape

    # calculate shortwave reflectance by linear scaling
    refl = calcRefl250Agg(si, err, u_i)
    if debug > 2:
        print "Shape of reflectivities array:"
        print refl.shape
    #print refl[0,:,:]   # test
    s250 = (rad, refl)
    # return radiances and reflectances
    return s250



def getREFL500Agg(filename):

    if debug > 1:
        print
        print "** Function  getREFL500Agg **"

    global FILE
    FILE = filename
    # create dummy arrays
    rad = np.array([[0., 0.], [0., 0.]])
    refl = np.array([[0., 0.], [0., 0.]])
    # extract requested data from HDF file
    # and calculate corrected counts
    si, err, u_i = getSI500Agg(filename)
    if debug > 2:
        print "Shape of scaled integer array:"
        print si.shape

    # calculate shortwave radiance by linear scaling
    # uncomment the following line if radiances are needed
    #rad = calcRadi500Agg(si, err, u_i)
    if debug > 2:
        print "Shape of radiances array:"
        print rad.shape

    # calculate shortwave reflectance by linear scaling
    refl = calcRefl500Agg(si, err, u_i)
    if debug > 2:
        print "Shape of reflectivities array:"
        print refl.shape

    s500 = (rad, refl)
    # return radiances and reflectances
    return s500



def getREFL1000(filename):

    if debug > 1:
        print
        print "** Function  getREFL1000 **"

    global FILE
    FILE = filename
    # create dummy arrays
    rad = np.array([[0., 0.], [0., 0.]])
    refl = np.array([[0., 0.], [0., 0.]])
    # extract requested data from HDF file
    # and calculate corrected counts
    si, err, u_i = getSI1000(filename)
    if debug > 2:
        print "Shape of scaled integer array:"
        print si.shape

    # calculate shortwave radiance by linear scaling
    # uncomment the following line if radiances are needed
    #rad = calcRadi1000(si, err, u_i)
    if debug > 2:
        print "Shape of radiances array:"
        print rad.shape

    # calculate shortwave reflectance by linear scaling
    refl = calcRefl1000(si, err, u_i)
    if debug > 2:
        print "Shape of reflectivities array:"
        print refl.shape

    s1000 = (rad, refl)
    # return radiances and reflectances
    return s1000



def getTB(filename):
    """Extract requested data from HDF file
    and calculate radiances.
    """
    if debug > 1:
        print
        print "** Function  getTB **"

    global FILE
    FILE = filename
    radiance, err, u_i = getLong(filename)
    # calculate TB from radiances
    temp_bright = estimate_tb(radiance, err, u_i)
    if debug > 2:
        print "Shape of brightness temperatures array:"
        print temp_bright.shape

    # return brightnesstemperature
    return temp_bright



def getLong(filename):
    # Read Emissive data from hdf file and calculate radiances

    if debug > 1:
        print
        print "** Function  getLong **"

    a = _pyhl.read_nodelist(filename)
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_Emissive")
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_Emissive/radiance_offsets")
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_Emissive/radiance_scales")
    a.selectNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_Emissive_Uncert_Indexes")
    a.fetch()
   
    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_Emissive")
    # keep original integers to allow for error control
    err = b.data()
    err = err.astype(np.uint16)
    # float-array for radiance calculation
    measure = err*1.
    if debug > 2:
        print "Shape of measured data array:"
        print measure.shape

    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_Emissive/radiance_offsets")
    off = b.data()
    if debug > 2:
        print "Offset-list:"
        print off

    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_Emissive/radiance_scales")
    sca = b.data()
    if debug > 2:
        print "Scaling-list:"
        print sca

    # get uncertainty indices
    b = a.getNode("/MODIS_SWATH_Type_L1B/Data Fields/EV_1KM_Emissive_Uncert_Indexes")
    u_i = b.data()
    if debug > 2:
        print "Shape of uncertainty index array:"
        print u_i.shape

    u_i = u_i.astype(np.uint8)

    # transfer from digital numbers to W/m^2/mum/sr
    i = 0
    while i < len(sca):
        measure[i, :, :] = (measure[i, :, :] - off[i]) * sca[i]
        i = i + 1

    # measure: measured radiances
    # err    : original integer-array for error control
    # u_i     : uncertainty index (drop data if the 4 MSBs are set

    # return measurements, error flag, and uncertaity indices
    return measure, err, u_i



def estimate_tb(rad, err, u_i):
    """ *rad*: measured radiances
    *err*: original integer-array for error control
    *u_i* : uncertainty index (drop data if the 4 MSBs are set
    
    Calculate brightnesstemperature by feeding measured radiances
    into an inverted Planckfkt. Doing this a monochromatic measurement
    is assumed implicitely. Because this is not true in the real world
    a correction (first-order fit) is applied to take the response
    functions into account.
    The fitting parameter and center wavelengths are taken from one of
    Liam Gumley's (Liam.Gumley@ssec.wisc.edu) IDL tools.
    These parameters are from March 2000, so if newer response functions
    appear this fitting should receive an update.
    """
    
    if debug > 1:
        print
        print "** Function  estimate_tb **"

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

    # calculate channel by channel
    i = 0

    if debug > 0:
        print "-"*50
        print "Long Wave TB, 1000m, serial CH numbers"
        print "Ch   Missing pixels        \t%"
        tot = rad.shape[1] * rad.shape[2]

    while i < len(cwn):

        # try to get rid of negative values
        # fill value is 99999.9
        tmp = np.where(np.logical_or(np.greater(err[i], 32767),
                                     np.equal(np.bitwise_and(u_i[i], 15), 15)),
                       99999.9, rad[i])
        # calculate equivalent temperature for monochromatic measurements
        # fill value is -99.9
        res = np.where(np.equal(tmp, 99999.9), -99.9,
                       c2 / (cwn[i] *
                             np.log(c1 / (1.e6 * tmp * cwn[i] ** 5.) + 1.)))
        
        # take response fkt. into account
        rad[i] = np.where(np.greater(res, -9.), (res - tci[i]) / tcs[i], -99.9)

        if debug > 0:
            tmp = np.where(np.less(res, -9), 1, 0)
            count = sum(sum(tmp))
            proz = 100. * float(count) / float(tot)
            print i + 1, '\t', count, '\t\t\t', proz
        
        i = i + 1

    # return brightnesstemperatures
    return rad


# -----------------------------------------------------------------------------------------------
#     Some GEO-Projection things. Maybe they should be excluded later...
# -----------------------------------------------------------------------------------------------

def cartesianDistance(B0,B1,L0,L1):
    """Transform 2 geographical coordinates into cartesian distance and
    direction.
    """

    # Earthradiun [m]
    r_earth = 6371000.
    
    # first lets make them useable...
    b0 = B0 * deg2rad
    b1 = B1 * deg2rad
    l0 = L0 * deg2rad
    l1 = L1 * deg2rad

    # a simple approach for a sphere
    bog = np.arccos(np.sin(b0) * np.sin(b1) +
                    np.cos(b0) * np.cos(b1) * np.cos(l1 - l0))
    dist = bog * r_earth

    # direction 0 -> 1
    direction = (rad2deg * np.arccos((np.sin(b1) - np.sin(b0) * np.cos(bog)) /
                                     (np.cos(b0) * np.sin(bog))))

    return dist, direction  # distance [m] and direction [deg]


def geodeticPosition(B0,L0,dist,Dire):
    """Calculate the final position on base of a start position and bearing
    after a certain distance (the direct geodetic problem).
    """
    # radiances, please...
    b0 = B0 * deg2rad
    l0 = L0 * deg2rad

    # some nasty formulas later
    b1 = 0.  # just dummies for now
    l1 = 0.  # just dummies for now

    # back to the geodetic system
    B1 = b1 * rad2deg
    L1 = l1 * rad2deg

    return B1, L1   # final position [degree]


# -----------------------------------------------------------------------------------------------
#     End of GEO-Projection things.
# -----------------------------------------------------------------------------------------------


def getGeo(filename):
    # Read geolocation data from hdf file 

    if debug > 1:
        print
        print "** Function  getGeo **"

    a = _pyhl.read_nodelist(filename)
    a.selectNode("/MODIS_SWATH_Type_L1B/Geolocation Fields/Latitude")
    a.selectNode("/MODIS_SWATH_Type_L1B/Geolocation Fields/Longitude")
    a.selectNode("/MODIS_SWATH_Type_L1B/Geolocation Fields/Latitude/_FillValue")
    a.selectNode("/MODIS_SWATH_Type_L1B/Geolocation Fields/Longitude/_FillValue")
    a.fetch()
   
    b = a.getNode("/MODIS_SWATH_Type_L1B/Geolocation Fields/Latitude")
    lat = b.data()
    b = a.getNode("/MODIS_SWATH_Type_L1B/Geolocation Fields/Longitude")
    lon = b.data()
    b = a.getNode("/MODIS_SWATH_Type_L1B/Geolocation Fields/Latitude/_FillValue")
    fillLat = b.data()
    b = a.getNode("/MODIS_SWATH_Type_L1B/Geolocation Fields/Longitude/_FillValue")
    fillLon = b.data()
    
    if debug > 2:
        print
        print "Fill value Lat: ", fillLat
        print "Fill value Lon: ", fillLon
        print "Shape of geo-arrays:"
        print "Lat: ", lat.shape
        print "Lon: ", lon.shape
        if debug > 8:
            print
            print "Locations:"
            print "Lat:"
            print lat
            print
            print "Lon:"
            print lon

    duma = np.array([0.])
    dumb = 0.
    geo = (lat, lon, fillLat, fillLon, duma, dumb, duma, dumb, duma, dumb, dumb)

    # Return geolocation data
    return geo


def getGeo2(filename):
    """Read geolocation data and requested angles from separat hdf file.
    """

    geogain = 0.001    # scaling factor for lat and lon
    anggain = 0.01     # scaling factor for angles

    if debug > 1:
        print
        print "** Function  getGeo2 **"

    a = _pyhl.read_nodelist(filename)
    a.selectNode("/MODIS_Swath_Type_GEO/Geolocation Fields/Latitude")
    a.selectNode("/MODIS_Swath_Type_GEO/Geolocation Fields/Longitude")
    a.selectNode("/MODIS_Swath_Type_GEO/Geolocation Fields/Latitude/_FillValue")
    a.selectNode("/MODIS_Swath_Type_GEO/Geolocation Fields/"
                 "Longitude/_FillValue")


    a.selectNode("/MODIS_Swath_Type_GEO/Data Fields/SolarZenith")
    a.selectNode("/MODIS_Swath_Type_GEO/Data Fields/SolarZenith/scale_factor")
    a.selectNode("/MODIS_Swath_Type_GEO/Data Fields/SolarZenith/_FillValue")

    a.selectNode("/MODIS_Swath_Type_GEO/Data Fields/SensorZenith")
    a.selectNode("/MODIS_Swath_Type_GEO/Data Fields/SensorZenith/scale_factor")
    a.selectNode("/MODIS_Swath_Type_GEO/Data Fields/SensorZenith/_FillValue")

    a.selectNode("/MODIS_Swath_Type_GEO/Data Fields/SolarAzimuth")
    a.selectNode("/MODIS_Swath_Type_GEO/Data Fields/SolarAzimuth/scale_factor")
    a.selectNode("/MODIS_Swath_Type_GEO/Data Fields/SolarAzimuth/_FillValue")

    a.selectNode("/MODIS_Swath_Type_GEO/Data Fields/SensorAzimuth")
    a.selectNode("/MODIS_Swath_Type_GEO/Data Fields/SensorAzimuth/scale_factor")
    a.selectNode("/MODIS_Swath_Type_GEO/Data Fields/SensorAzimuth/_FillValue")


    a.fetch()
   
    b = a.getNode("/MODIS_Swath_Type_GEO/Geolocation Fields/Latitude")
    lat = (b.data() / geogain).astype('int32')
    b = a.getNode("/MODIS_Swath_Type_GEO/Geolocation Fields/Longitude")
    lon = (b.data() / geogain).astype('int32')
    b = a.getNode("/MODIS_Swath_Type_GEO/Geolocation Fields/Latitude/_FillValue")
    fillLat = b.data()
    b = a.getNode("/MODIS_Swath_Type_GEO/Geolocation Fields/Longitude/_FillValue")
    fillLon = b.data()
    fillLat = int(fillLat/geogain)
    fillLon = int(fillLon/geogain)

    b = a.getNode("/MODIS_Swath_Type_GEO/Data Fields/SolarZenith")
    solzen = b.data()
    b = a.getNode("/MODIS_Swath_Type_GEO/Data Fields/SolarZenith/scale_factor")
    solzensca = b.data()
    b = a.getNode("/MODIS_Swath_Type_GEO/Data Fields/SolarZenith/_FillValue")
    solzenfil = b.data()

    b = a.getNode("/MODIS_Swath_Type_GEO/Data Fields/SensorZenith")
    senzen = b.data()
    b = a.getNode("/MODIS_Swath_Type_GEO/Data Fields/SensorZenith/scale_factor")
    senzensca = b.data()
    b = a.getNode("/MODIS_Swath_Type_GEO/Data Fields/SensorZenith/_FillValue")
    senzenfil = b.data()

    b = a.getNode("/MODIS_Swath_Type_GEO/Data Fields/SolarAzimuth")
    solazi = b.data()
    b = a.getNode("/MODIS_Swath_Type_GEO/Data Fields/SolarAzimuth/scale_factor")
    solazisca = b.data()
    b = a.getNode("/MODIS_Swath_Type_GEO/Data Fields/SolarAzimuth/_FillValue")
    solazifil = b.data()

    b = a.getNode("/MODIS_Swath_Type_GEO/Data Fields/SensorAzimuth")
    senazi = b.data()
    b = a.getNode("/MODIS_Swath_Type_GEO/Data Fields/SensorAzimuth/scale_factor")
    senazisca = b.data()
    b = a.getNode("/MODIS_Swath_Type_GEO/Data Fields/SensorAzimuth/_FillValue")
    senazifil = b.data()

    solzen = (solzen*solzensca/anggain).astype('int16')
    solzenfil = int(solzenfil*solzensca/anggain)
    senzen = (senzen*senzensca/anggain).astype('int16')
    senzenfil = int(senzenfil*senzensca/anggain)
    solazi = solazi*solazisca
    solazifil = solazifil*solazisca
    senazi = senazi*senazisca
    senazifil = senazifil*senazisca

    azidif = solazi - senazi
    azidif = np.where(np.logical_or(np.equal(solazi, solazifil),
                                    np.equal(senazi, senazifil)),
                      solazifil,
                      (np.arccos(np.cos((solazi - senazi) *
                                       deg2rad)) *
                       rad2deg))


    azidif = (azidif/anggain).astype('int16')
    solazifil = int(solazifil/anggain)
    senazifil = int(senazifil/anggain)

    solazi = (solazi/anggain).astype('int16')
    senazi = (senazi/anggain).astype('int16')

    
    if debug > 2:
        print
        print "Fill value Lat: ", fillLat
        print "Fill value Lon: ", fillLon
        print "Shape of geo-arrays:"
        print "Lat: ", lat.shape
        print "Lon: ", lon.shape
        print "Shape of angle-arrays:"
        print "Solar zenith: ", solzen.shape
        print "Sensor zenith: ", senzen.shape
        print "Azimuth difference: ", azidif.shape


        if debug > 8:
            print
            print "Locations:"
            print "Lat:"
            print lat
            print
            print "Lon:"
            print lon

    geo = (lat, lon, fillLat, fillLon, solzen, solzenfil, senzen, senzenfil,
           azidif, solazifil, geogain, anggain, solazi, senazi, senazifil)
    # last three parts in geo above are due to requirements of redundant
    # informations
    

    # Return geolocation data
    return geo



def writeAngles(GEO, outFile2, angpack):
    """Write down solar zenith, sensor zenith and azimuth difference angles to
    hdf file.
    """
    
    if debug > 1:
        print
        print "** Function  writeAngles **"


    lat = GEO[0]
    lon = GEO[1]
    latfill = GEO[2]
    lonfill = GEO[3]
    solzen = GEO[4]
    solzenfil = GEO[5]
    senzen = GEO[6]
    senzenfil = GEO[7]
    azidif = GEO[8]
    azidiffil = GEO[9]
    geogain = GEO[10]
    anggain = GEO[11]
    solazi  = GEO[12]
    senazi  = GEO[13]
    senazifil  = GEO[14]

    solazifil = azidiffil
    
    (datestr, timestr, datestr2, timestr2, nadirXSize, nadirYSize,
     scan1, scan2, platt, norbit, instrument, startEpoche,
     endEpoche, NumOfPixels, NumOfScans) = angpack

    nsets = 5 # solzen,senzen,azidiff,solazi,senazi


    azioff = 180.  # this azimuth offset is already within the the dataset. For
                   # good reasons


    #            ANGLES
    #=================================================================================

    # What
    # ====
    a = _pyhl.nodelist()
    b = _pyhl.node(_pyhl.GROUP_ID,"/what")
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/what/object")
    b.setScalarValue(-1,"SATP","string",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/what/sets")
    b.setScalarValue(-1,nsets,"int",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/what/version")
    b.setScalarValue(-1,"H5rad ?.?","string",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/what/date")
    b.setScalarValue(-1,datestr,"string",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/what/time")
    b.setScalarValue(-1,timestr,"string",-1)
    a.addNode(b)    

    # Where
    # =====
    b = _pyhl.node(_pyhl.GROUP_ID,"/where")
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/num_of_pixels")
    b.setScalarValue(-1,NumOfPixels,"int",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/num_of_lines")
    b.setScalarValue(-1,NumOfScans,"int",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/xscale")
    b.setScalarValue(-1,nadirXSize,"float",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/yscale")
    b.setScalarValue(-1,nadirYSize,"float",-1)
    a.addNode(b)
    # Geo Data:
    b = _pyhl.node(_pyhl.GROUP_ID,"/where/lon")
    a.addNode(b)
    b = _pyhl.node(_pyhl.DATASET_ID,"/where/lon/data")
    b.setArrayValue(1,lon.shape,lon,"int",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.GROUP_ID,"/where/lon/what")
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/lon/what/gain")
    b.setScalarValue(-1,geogain,"float",-1)
    a.addNode(b)    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/lon/what/offset")
    b.setScalarValue(-1,0.0,"float",-1)
    a.addNode(b)    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/lon/what/nodata")
    b.setScalarValue(-1,lonfill,"int",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/lon/what/missingdata")
    b.setScalarValue(-1,lonfill,"int",-1)
    a.addNode(b)    
    b = _pyhl.node(_pyhl.GROUP_ID,"/where/lat")
    a.addNode(b)
    b = _pyhl.node(_pyhl.DATASET_ID,"/where/lat/data")
    b.setArrayValue(1,lat.shape,lat,"int",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.GROUP_ID,"/where/lat/what")
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/lat/what/gain")
    b.setScalarValue(-1,geogain,"float",-1)
    a.addNode(b)    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/lat/what/offset")
    b.setScalarValue(-1,0.0,"float",-1)
    a.addNode(b)    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/lat/what/nodata")
    b.setScalarValue(-1,latfill,"int",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/lat/what/missingdata")
    b.setScalarValue(-1,latfill,"int",-1)
    a.addNode(b)    
    
    # How
    b = _pyhl.node(_pyhl.GROUP_ID,"/how")
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/how/startepochs")
    b.setScalarValue(-1,startEpoche,"int",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/how/endepochs")
    b.setScalarValue(-1,endEpoche,"int",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/how/platform")
    b.setScalarValue(-1,platt,"string",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/how/instrument")
    b.setScalarValue(-1,instrument,"string",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/how/orbit_number")
    b.setScalarValue(-1,int(norbit),"int",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/how/software")
    b.setScalarValue(-1,"NWCSAF/PPS","string",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/how/version")
    b.setScalarValue(-1,"2.0","string",-1)
    a.addNode(b)

    # Sun Zenith:
    image_number = 1
    b = _pyhl.node(_pyhl.GROUP_ID,"/image%d"%image_number)
    a.addNode(b)
    b = _pyhl.node(_pyhl.DATASET_ID,"/image%d/data"%image_number)
    b.setArrayValue(1,solzen.shape,solzen,"short",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/description"%(image_number))
    b.setScalarValue(-1,"sun zenith angle","string",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.GROUP_ID,"/image%d/what"%image_number)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/product"%image_number)
    b.setScalarValue(1,'SUNZ',"string",-1)
    a.addNode(b)        
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/quantity"%image_number)
    b.setScalarValue(1,"DEG","string",-1)
    a.addNode(b)        
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/startdate"%image_number)
    b.setScalarValue(-1,datestr,"string",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/starttime"%image_number)
    b.setScalarValue(-1,timestr,"string",-1)
    a.addNode(b)    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/enddate"%image_number)
    b.setScalarValue(-1,datestr2,"string",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/endtime"%image_number)
    b.setScalarValue(-1,timestr2,"string",-1)
    a.addNode(b)
    # Gain and Intercept:    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/gain"%image_number)
    b.setScalarValue(-1,anggain,"float",-1)
    a.addNode(b)    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/offset"%image_number)
    b.setScalarValue(-1,0.,"float",-1)
    a.addNode(b)    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/nodata"%image_number)
    b.setScalarValue(-1,solzenfil,"int",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/missingdata"%image_number)
    b.setScalarValue(-1,solzenfil,"int",-1)
    a.addNode(b)    

    # Satellite Zenith:
    image_number = 2
    b = _pyhl.node(_pyhl.GROUP_ID,"/image%d"%image_number)
    a.addNode(b)
    b = _pyhl.node(_pyhl.DATASET_ID,"/image%d/data"%image_number)
    b.setArrayValue(1,senzen.shape,senzen,"short",-1)                
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/description"%(image_number))
    b.setScalarValue(-1,"satellite zenith angle","string",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.GROUP_ID,"/image%d/what"%image_number)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/product"%image_number)
    b.setScalarValue(1,'SATZ',"string",-1)
    a.addNode(b)        
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/quantity"%image_number)
    b.setScalarValue(1,"DEG","string",-1)
    a.addNode(b)        
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/startdate"%image_number)
    b.setScalarValue(-1,datestr,"string",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/starttime"%image_number)
    b.setScalarValue(-1,timestr,"string",-1)
    a.addNode(b)    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/enddate"%image_number)
    b.setScalarValue(-1,datestr2,"string",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/endtime"%image_number)
    b.setScalarValue(-1,timestr2,"string",-1)
    a.addNode(b)
    # Gain and Intercept:    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/gain"%image_number)
    b.setScalarValue(-1,anggain,"float",-1)
    a.addNode(b)    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/offset"%image_number)
    b.setScalarValue(-1,0.,"float",-1)
    a.addNode(b)    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/nodata"%image_number)
    b.setScalarValue(-1,senzenfil,"int",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/missingdata"%image_number)
    b.setScalarValue(-1,senzenfil,"int",-1)
    a.addNode(b)    
    
    # Relative azimuth difference:
    image_number = 3
    b = _pyhl.node(_pyhl.GROUP_ID,"/image%d"%image_number)
    a.addNode(b)
    b = _pyhl.node(_pyhl.DATASET_ID,"/image%d/data"%image_number)
    b.setArrayValue(1,azidif.shape,azidif,"short",-1)                
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/description"%(image_number))
    b.setScalarValue(-1,"relative sun-satellite azimuth difference angle","string",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.GROUP_ID,"/image%d/what"%image_number)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/product"%image_number)
    b.setScalarValue(1,'SSAZD',"string",-1)
    a.addNode(b)        
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/quantity"%image_number)
    b.setScalarValue(1,"DEG","string",-1)
    a.addNode(b)        
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/startdate"%image_number)
    b.setScalarValue(-1,datestr,"string",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/starttime"%image_number)
    b.setScalarValue(-1,timestr,"string",-1)
    a.addNode(b)    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/enddate"%image_number)
    b.setScalarValue(-1,datestr2,"string",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/endtime"%image_number)
    b.setScalarValue(-1,timestr2,"string",-1)
    a.addNode(b)
    # Gain and Intercept:    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/gain"%image_number)
    b.setScalarValue(-1,anggain,"float",-1)
    a.addNode(b)    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/offset"%image_number)
    b.setScalarValue(-1,0.,"float",-1)
    a.addNode(b)    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/nodata"%image_number)
    b.setScalarValue(-1,azidiffil,"int",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/missingdata"%image_number)
    b.setScalarValue(-1,azidiffil,"int",-1)
    a.addNode(b)    

    # Solar azimuth:
    image_number = 4
    b = _pyhl.node(_pyhl.GROUP_ID,"/image%d"%image_number)
    a.addNode(b)
    b = _pyhl.node(_pyhl.DATASET_ID,"/image%d/data"%image_number)
    b.setArrayValue(1,solazi.shape,azidif,"short",-1)                
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/description"%(image_number))
    b.setScalarValue(-1,"solar azimuth angle","string",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.GROUP_ID,"/image%d/what"%image_number)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/product"%image_number)
    b.setScalarValue(1,'SUNA',"string",-1)
    a.addNode(b)        
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/quantity"%image_number)
    b.setScalarValue(1,"DEG","string",-1)
    a.addNode(b)        
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/startdate"%image_number)
    b.setScalarValue(-1,datestr,"string",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/starttime"%image_number)
    b.setScalarValue(-1,timestr,"string",-1)
    a.addNode(b)    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/enddate"%image_number)
    b.setScalarValue(-1,datestr2,"string",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/endtime"%image_number)
    b.setScalarValue(-1,timestr2,"string",-1)
    a.addNode(b)
    # Gain and Intercept:    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/gain"%image_number)
    b.setScalarValue(-1,anggain,"float",-1)
    a.addNode(b)    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/offset"%image_number)
    b.setScalarValue(-1,azioff,"float",-1)
    a.addNode(b)    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/nodata"%image_number)
    b.setScalarValue(-1,solazifil,"int",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/missingdata"%image_number)
    b.setScalarValue(-1,solazifil,"int",-1)
    a.addNode(b)    

    # Sensor azimuth:
    image_number = 5
    b = _pyhl.node(_pyhl.GROUP_ID,"/image%d"%image_number)
    a.addNode(b)
    b = _pyhl.node(_pyhl.DATASET_ID,"/image%d/data"%image_number)
    b.setArrayValue(1,senazi.shape,azidif,"short",-1)                
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/description"%(image_number))
    b.setScalarValue(-1,"satellite azimuth angle","string",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.GROUP_ID,"/image%d/what"%image_number)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/product"%image_number)
    b.setScalarValue(1,'SATA',"string",-1)
    a.addNode(b)        
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/quantity"%image_number)
    b.setScalarValue(1,"DEG","string",-1)
    a.addNode(b)        
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/startdate"%image_number)
    b.setScalarValue(-1,datestr,"string",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/starttime"%image_number)
    b.setScalarValue(-1,timestr,"string",-1)
    a.addNode(b)    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/enddate"%image_number)
    b.setScalarValue(-1,datestr2,"string",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/endtime"%image_number)
    b.setScalarValue(-1,timestr2,"string",-1)
    a.addNode(b)
    # Gain and Intercept:    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/gain"%image_number)
    b.setScalarValue(-1,anggain,"float",-1)
    a.addNode(b)    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/offset"%image_number)
    b.setScalarValue(-1,azioff,"float",-1)
    a.addNode(b)    
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/nodata"%image_number)
    b.setScalarValue(-1,senazifil,"int",-1)
    a.addNode(b)
    b = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/missingdata"%image_number)
    b.setScalarValue(-1,senazifil,"int",-1)
    a.addNode(b)    




    status2 = a.write(outFile2, COMPRESS_LVL)    

    # return the status
    return status2




def writeIt(s250, s500, s1000, TB, GEO, FILE):
    """This routine prepares the output and writes it in HDF5 format.
    *rad500*: Radiances, aggregated from 500m resolution (ch 3 to 7).
    *REFL500*: Reflectances, aggregated from 500m resolution (ch 3 to 7).
    *rad1*: Radiances, MODIS channels 8 to 19 and 26
    *REFL1*: Reflectances, MODIS channels 8 to 19 and 26
    *TB*: Brightness temperatures, ch 20 to 25 and 27 to 36
    """

    import time, datetime
    import string

    rad250 = s250[0]
    REFL250 = s250[1]
    rad500 = s500[0]
    REFL500 = s500[1]
    Rad1 = s1000[0]
    REFL1 = s1000[1]

    nsca = TB.shape[1]/10  # No of scans

    if debug > 1:
        print
        print "** Function  writeIt **"

    platt = "????"
    if string.find(FILE, "MOD") >= 0:
        platt = "eos1"
    if string.find(FILE, "MYD") >= 0:
        platt = "eos2"
       
    # get the orbitnumber out of the metadata
    a = _pyhl.read_nodelist(FILE)
    a.selectNode("/CoreMetadata.0_GLOSDS")
    a.fetch()
    b = a.getNode("/CoreMetadata.0_GLOSDS")
    meta = b.data()
    norbit = '?????'
    dot = string.find(meta, "ORBITNUMBER")
    if dot >= 0:
        dot = string.find(meta, "VALUE", dot)
        dot = string.find(meta, "=", dot)
        norbit = string.zfill(str(int(meta[dot+1:dot+10])),5)

    # ---------------------------------------
    # some hardwired values
    ss = 0  # five minute intervall should allways start at zero seconds...
    nadirXSize = 1000.
    nadirYSize = 1000.
    sc1 = 0             # first scanline
    instrument = "modis"
    # ---------------------------------------


    # Date and time...
    dot = string.find(FILE, "KM.")
    YYYY = int(FILE[dot+4:dot+8])
    DOY = int(FILE[dot+8:dot+11])    # Day Of Year
    hh = int(FILE[dot+12:dot+14])
    mm = int(FILE[dot+14:dot+16])
    first = datetime.datetime(YYYY, 1, 1, hh, mm)
    set1 = first + datetime.timedelta(days=DOY-1)
    datestr = string.zfill(str(set1.year),4) +\
              string.zfill(str(set1.month),2) +\
              string.zfill(str(set1.day),2)
    timestr = string.zfill(str(set1.hour),2) +\
              string.zfill(str(set1.minute),2) +\
              string.zfill(str(ss),2)
    set2 = set1 + datetime.timedelta(minutes=5)
    datestr2 = string.zfill(str(set2.year),4) +\
               string.zfill(str(set2.month),2) +\
               string.zfill(str(set2.day),2)
    timestr2 = string.zfill(str(set2.hour),2) +\
               string.zfill(str(set2.minute),2) +\
               string.zfill(str(ss),2)
    
    scan1 = string.zfill(str(sc1),5)
    scan2 = string.zfill(str(sc1 + nsca - 1),5)
    fill = -99.9   # fillvalue

    outFile = platt + "_" + datestr + "_" + timestr[0:4] + "_" + norbit + \
              "_satproj_" + scan1 + "_" + scan2 + "_" + instrument + ".h5"
    
    outFile2 = platt + "_" + datestr + "_" + timestr[0:4] + "_" + norbit + \
              "_satproj_" + scan1 + "_" + scan2 + "_" + "sunsatangles.h5"


    # Do this epoche-thing
    YYYY = int(datestr[0:4])
    MM = int(datestr[4:6])
    DD = int(datestr[6:])
    hh = int(timestr[0:2])
    mm = int(timestr[2:4])
    ss = int(timestr[4:])
    timeTuple = YYYY, MM, DD, hh, mm, ss, 0, 0, 0
    startEpoche = int(time.mktime(timeTuple) - time.timezone)
    endEpoche = startEpoche + (5*60)

    # calculate sun-earth distance correction factor
    # This is a bit odd, because the correction is done internally
    # and hidden in intercept and scaling factor.
    # However, the correction factor is needed by pps and this value
    # should be close to the one used by the MODIS team...
    corr = 1. - 0.0334*math.cos(2.*math.pi*(DOY - 2.)/365.25)

    NumOfPixels = len(TB[0,0,:])
    NumOfLines = len(TB[0,:,0])
    lat = GEO[0]
    lon = GEO[1]
    latfill = GEO[2]
    lonfill = GEO[3]
    geogain = GEO[10]

    # pack some data for angle-file...
    angpack = (datestr, timestr, datestr2, timestr2, nadirXSize, nadirYSize,\
               scan1, scan2, platt, norbit, instrument, startEpoche, endEpoche,\
               NumOfPixels, NumOfLines)


    status2 = writeAngles(GEO, outFile2, angpack)

    print
    print
    if status2 == None:
        write_log("INFO","Sunsatangle-File: OK",moduleid=MODULE_ID)
    else:
        write_log("WARNING","Sunsatangle-File: PROBLEMS",moduleid=MODULE_ID)
    print

    chList = []
    for ch in selection:
        chList.append(CHNames[ch])
    
    if debug > 0:
        print "-"*50
        print "Selected channels:"
        for ch in selection:
            print CHNames[ch]
        print
            
        print "Date:", datestr
        print "Time:", timestr

        if debug > 1:
            print "Number of pixels:", NumOfPixels
            print "Number of scans:", NumOfLines
            print "Nadir pixelsize (X):", nadirXSize
            print "Nadir pixelsize (Y):", nadirYSize
            print "Start (secs since 1970):", startEpoche
            print "End (secs since 1970):", endEpoche


    #            MEASUREMENTS
    #=================================================================================


    #----WHAT
    aList = _pyhl.nodelist()
    aNode = _pyhl.node(_pyhl.GROUP_ID,"/what")
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/what/object")
    aNode.setScalarValue(-1,"SATP","string",-1)
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/what/sets")
    aNode.setScalarValue(-1,len(selection),"int",-1)
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/what/version")
    aNode.setScalarValue(-1,"H5rad ?.?","string",-1)
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/what/date")
    aNode.setScalarValue(-1,datestr,"string",-1)
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/what/time")
    aNode.setScalarValue(-1,timestr,"string",-1)
    aList.addNode(aNode)

    #----WHERE
    aNode = _pyhl.node(_pyhl.GROUP_ID,"/where")
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/num_of_pixels")
    aNode.setScalarValue(-1,NumOfPixels,"int",-1)
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/num_of_lines")
    aNode.setScalarValue(-1,NumOfLines,"int",-1)
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/xscale")
    aNode.setScalarValue(-1,nadirXSize,"float",-1)
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/yscale")
    aNode.setScalarValue(-1,nadirYSize,"float",-1)
    aList.addNode(aNode)
    # Geo Data:
    # LON
    aNode = _pyhl.node(_pyhl.GROUP_ID,"/where/lon")
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.DATASET_ID,"/where/lon/data")
    aNode.setArrayValue(1,lon.shape,lon,"int",-1)
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.GROUP_ID,"/where/lon/what")
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/lon/what/gain")
    aNode.setScalarValue(-1,geogain,"float",-1)
    aList.addNode(aNode)    
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/lon/what/offset")
    aNode.setScalarValue(-1,0.0,"float",-1)
    aList.addNode(aNode)    
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/lon/what/nodata")
    aNode.setScalarValue(-1,lonfill,"int",-1)
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/lon/what/missingdata")
    aNode.setScalarValue(-1,lonfill,"int",-1)
    aList.addNode(aNode)
    # LAT
    aNode = _pyhl.node(_pyhl.GROUP_ID,"/where/lat")
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.DATASET_ID,"/where/lat/data")
    aNode.setArrayValue(1,lat.shape,lat,"int",-1)
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.GROUP_ID,"/where/lat/what")
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/lat/what/gain")
    aNode.setScalarValue(-1,geogain,"float",-1)
    aList.addNode(aNode)    
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/lat/what/offset")
    aNode.setScalarValue(-1,0.0,"float",-1)
    aList.addNode(aNode)    
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/lat/what/nodata")
    aNode.setScalarValue(-1,latfill,"int",-1)
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/where/lat/what/missingdata")
    aNode.setScalarValue(-1,latfill,"int",-1)
    aList.addNode(aNode)    
 
    #----HOW
    aNode = _pyhl.node(_pyhl.GROUP_ID,"/how")
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/how/startepochs")
    aNode.setScalarValue(-1,startEpoche,"int",-1)
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/how/endepochs")
    aNode.setScalarValue(-1,endEpoche,"int",-1)
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/how/platform")
    aNode.setScalarValue(-1,platt,"string",-1)
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/how/instrument")
    aNode.setScalarValue(-1,instrument,"string",-1)
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/how/orbit_number")
    aNode.setScalarValue(-1,int(norbit),"int",-1)
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/how/software")
    aNode.setScalarValue(-1,"NWCSAF/PPS","string",-1)
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/how/version")
    aNode.setScalarValue(-1,"2.0","string",-1)
    aList.addNode(aNode)
    aNode = _pyhl.node(_pyhl.DATASET_ID,"/channel_list")
    aNode.setArrayValue(1,(len(selection),),chList,"string",-1)
    aList.addNode(aNode)

    #----DATASETS
    image_number = 1
    if debug > 0:
        print
        print "Processing Channel:"
    for ch in selection:
        off = 0.
        aNode = _pyhl.node(_pyhl.GROUP_ID,"/image%d"%image_number)
        aList.addNode(aNode)
        chid  =  CHNames[ch]
        if debug > 0:
            print chid
        aNode = _pyhl.node(_pyhl.DATASET_ID,"/image%d/data"%image_number)
        if ch >= 0 and ch < 2:
            tmp = ch
            shape = REFL250[tmp,:,:].shape
            field =  where(equal(REFL250[tmp,:,:],fill),fill,REFL250[tmp,:,:]*100.)   # in %
            #gain = 0.0001
            gain = 0.01
            QT = "REFL"     # new quantity 
        if ch > 1 and ch < 7:
            tmp = ch - 2
            shape = REFL500[tmp,:,:].shape
            field =  where(equal(REFL500[tmp,:,:],fill),fill,REFL500[tmp,:,:]*100.)   # in %
            gain = 0.01
            QT = "REFL"     # new quantity 
        if ch > 6 and ch < 21:
            tmp = ch - 7
            shape = REFL1[tmp,:,:].shape
            field =  where(equal(REFL1[tmp,:,:],fill),fill,REFL1[tmp,:,:]*100.)   # in %
            gain = 0.01
            QT = "REFL"     # new quantity 
        if ch > 20 and ch < 27:
            tmp = ch - 21
            shape = TB[tmp,:,:].shape
            field = TB[tmp,:,:]
            off = 273.15
            gain = 0.01
            QT = "TB"     # new quantity 
        if ch == 27:
            tmp = 14
            shape = REFL1[tmp,:,:].shape
            field =  where(equal(REFL1[tmp,:,:],fill),fill,REFL1[tmp,:,:]*100.)   # in %
            gain = 0.01
            QT = "REFL"     # new quantity 
        if ch > 27:
            tmp = ch - 22
            shape = TB[tmp,:,:].shape
            field = TB[tmp,:,:]
            off = 273.15
            gain = 0.01
            QT = "TB"     # new quantity 
        #field = ((field-off)/gain).astype('int16')
        #fillval = int((fill-off)/gain)                     # test area...
        field = where(equal(field,fill),-9990,(field-off)/gain)
        field = field.astype('int16')
        fillval = -9990                                    # set final fillvalue
        aNode.setArrayValue(1,shape,field,"short",-1)
        aList.addNode(aNode)
        #fillval = int((fill-off)/gain)
        
        chname = chid
        aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/channel"%(image_number))
        aNode.setScalarValue(-1,chname,"string",-1)
        aList.addNode(aNode)
        #aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/description"%(image_number))
        #b.setScalarValue(-1,"AVHRR ch%s"%(chid.split("ch_")[1]),"string",-1)
        #a.addNode(b)

        if ch < 21 or ch == 27:      # solar channel
            aNode = _pyhl.node(_pyhl.GROUP_ID,"/image%d/how"%image_number)
            aList.addNode(aNode)
            aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/how/sun_earth_distance_correction_applied"%image_number)
            aNode.setScalarValue(1,'TRUE',"string",-1)
            aList.addNode(aNode)
            
            aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/how/sun_earth_distance_correction_factor"%image_number)
            aNode.setScalarValue(-1,corr,"float",-1)
            aList.addNode(aNode)    

        aNode = _pyhl.node(_pyhl.GROUP_ID,"/image%d/what"%image_number)
        aList.addNode(aNode)
        aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/product"%image_number)
        aNode.setScalarValue(1,'SATCH',"string",-1)
        aList.addNode(aNode)
        
        aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/quantity"%image_number)
        aNode.setScalarValue(1,QT,"string",-1)
        aList.addNode(aNode)
        
        aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/startdate"%image_number)
        aNode.setScalarValue(-1,datestr,"string",-1)
        aList.addNode(aNode)
        aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/starttime"%image_number)
        aNode.setScalarValue(-1,timestr,"string",-1)
        aList.addNode(aNode)    
        aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/enddate"%image_number)
        aNode.setScalarValue(-1,datestr2,"string",-1)
        aList.addNode(aNode)
        aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/endtime"%image_number)
        aNode.setScalarValue(-1,timestr2,"string",-1)
        aList.addNode(aNode)    
        aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/gain"%image_number)
        aNode.setScalarValue(-1,gain,"float",-1)
        aList.addNode(aNode)    
        aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/offset"%image_number)
        aNode.setScalarValue(-1,off,"float",-1)
        aList.addNode(aNode)    
        aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/nodata"%image_number)
        aNode.setScalarValue(-1,fillval,"int",-1)
        aList.addNode(aNode)
        aNode = _pyhl.node(_pyhl.ATTRIBUTE_ID,"/image%d/what/missingdata"%image_number)
        aNode.setScalarValue(-1,fillval,"int",-1)
        aList.addNode(aNode)    

        image_number = image_number + 1

    status1 = aList.write(outFile,COMPRESS_LVL)    

    print
    print
    if status1 == None:
        write_log("INFO","Satellite-File: OK",moduleid=MODULE_ID)
    else:
        write_log("WARNING","Satellite-File: PROBLEMS",moduleid=MODULE_ID)


    return status1, status2



def prepareInput():

    import glob

    FILE = None
    FILE2 = None
    
    if debug > 1:
        print
        print "** Function  prepareInput **"

    if len(sys.argv) != 5:
	write_log("INFO","Usage: %s <platform id> <year [YYYY]> <julian day of year [DDD]> <time [HHMM]>"\
                                     %sys.argv[0],moduleid=MODULE_ID)
	sys.exit(0)
    else:
        if sys.argv[1] == "eos1":
            pre1 = "MOD021KM"
            pre2 = "MOD03"
        elif sys.argv[1] == "eos2":
            pre1 = "MYD021KM"
            pre2 = "MYD03"
        else:
            write_log("INFO","Unsupported Platform '%s'. Allowed are: 'eos1' and 'eos2'"\
                                     %sys.argv[1],moduleid=MODULE_ID)
            sys.exit(0)

        # first try the MODIS filenam convention
        rest = ".A%s%s.%s.???.?????????????.h5"%(sys.argv[2],sys.argv[3],sys.argv[4])
        matchstr1 = pre1+rest
        matchstr2 = pre2+rest
        filel = glob.glob(matchstr1)
        if len(filel) == 0:
            write_log("INFO","No regular level 1b file %s found!"%matchstr1,moduleid=MODULE_ID)
            write_log("INFO","Try to find a 'reduced' one...",moduleid=MODULE_ID)
            # Try a more flexible one
            rest = "*%s%s*%s*.h5"%(sys.argv[2],sys.argv[3],sys.argv[4])
            matchstr1 = "*"+pre1+rest
            filel = glob.glob(matchstr1)
            if len(filel) == 0:
                write_log("ERROR","Even desperad search for %s did not succeed!"%matchstr1,moduleid=MODULE_ID)
                write_log("ERROR","Check filename and directory",moduleid=MODULE_ID)
                sys.exit(0)

        FILE = filel[0]
	write_log("INFO","Found level 1b file: %s "%FILE,moduleid=MODULE_ID)
        
        filel = glob.glob(matchstr2)
        if len(filel) == 0:
            write_log("WARNING","No geolocation file %s found! trying to extract data from level 1b file."\
                      %matchstr2,moduleid=MODULE_ID)
        else:
            FILE2 = filel[0]
            write_log("INFO","Found geolocation file: %s "%FILE2,moduleid=MODULE_ID)
            

    return FILE, FILE2


# ------------------------------------------------------------------------------------------------------
# Testing:
if __name__ == "__main__":

    # Amount of output (0:nothing, 9:alot).
    debug = 1
    
    if debug > 1:
        print
        print "** __main__ **"

    FILE, FILE2 = prepareInput()

    # do some Tests on GEO-Transformation
    #dist, dire = cartesianDistance(52.517,35.7,13.4,139.767)
    #print  dist, dire
    #sys.exit(0)

    #FILE  = "./MOD021KM.A2006017.1025.005.2006017235113.h5"
    #FILE2 = "./MOD03.A2006017.1025.005.2006017231847.h5"
    #FILE  = "./MOD021KM.A2005161.0920.005.2005161194747.h5"
    #FILE2 = "./MOD03.A2005161.0920.005.2005161192012.h5"

    #FILE  = "./MYD021KM.A2005187.1005.005.2005189074226.h5"
    #FILE2 = "./MYD03.A2005187.1005.005.2005189062314.h5"

    # Central USA
    #FILE  = "./MYD021KM.A2005175.1930.004.2005180015345.h5"
    #FILE2 = "./MYD03.A2005175.1930.004.2005180005149.h5"

    # Indic
    #FILE  = "./MYD021KM.A2005183.0830.005.2005361122955.h5"
    #FILE2 = "./MYD03.A2005183.0830.005.2005361120411.h5"

    # Test files
    #FILE  = "./MOD021KM.A2004216.0230.004.2004216112523.h5"
    #FILE2 = "./MOD03.A2004216.0230.004.2004216104751.h5"
    #FILE  = "./MOD021KM.A2008031.2055.005.2008032041408.h5"
    #FILE2 = "./MOD03.A2008031.2055.005.2008032035242.h5"
    #FILE  = "./MOD021KM.A2008022.0950.005.2008022223728.h5"
    #FILE2 = "./MOD03.A2008022.0950.005.2008022180503.h5"

    if FILE2 == None:
        #GEO = getGeo(FILE)          #  low resolution (needs an update but is
                                     #  probably not worth it)...
        write_log("ERROR","Extraction from level 1b file not supported yet!",moduleid=MODULE_ID)
        sys.exit(0)
    else:                                 
        GEO = getGeo2(FILE2)         # full resolution

    S250 = getREFL250Agg(FILE)
    S500 = getREFL500Agg(FILE)
    S1000 = getREFL1000(FILE)
    TB = getTB(FILE)

    status1, status2 = writeIt(S250, S500, S1000, TB, GEO, FILE)

    if debug > 0:
        print
        if status1 != None or status2 != None:
            print "Writing stati:",status1, status2

    print
    if status1 != None or status2 != None:
        write_log("ERROR","Something went wrong!",moduleid=MODULE_ID)
    else:
        write_log("INFO","MODIS preprocessing OK.",moduleid=MODULE_ID)


