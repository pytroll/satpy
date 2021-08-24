import seviri_native_util as snu
import numpy as np
from mpop.satin.hdfeos_l1b import vinc_pt
import logging
from pyorbital.orbital import Orbital
from pyorbital.orbital import OrbitElements
from pyorbital import tlefile
from mpop.satellites import GeostationaryFactory
from mpop.projector import get_area_def
import datetime
import matplotlib.pyplot as plt

from mpop.utils import debug_on
debug_on()

# Configure logger.
logger = logging.getLogger('parallax')
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - '
                           '%(message)s', level=logging.INFO)


def parallax_corr(h, azi, ele):
    """Function to calculate the distance correction due to parallax effects
    Keyword arguments:
        h    cloud top height
        azi    Viewing azimuth angle from observer position
        ele    Viewing elevation angle from observer position
               (90 - viewing zenith angle)
    """
    # Elevation displacement
    dz = h / np.tan(np.deg2rad(ele))
    # Azimuth components
    dx = -np.sin(np.deg2rad(azi)) * dz
    dy = -np.cos(np.deg2rad(azi)) * dz

    return(dx, dy, dz)

"""
pix_lat = 10  # replace with read array
pix_lon = 10  # replace with read array

X = 0  # replace with header-info
Y = 0  # replace with header info
Z = 42000  # replace with header info

h = 0.  # replace with srtm
"""


def vza_and_vaa(height, lat, lon, X, Y, Z):

    # Below are the values given by Gieske et. al. */

    # a = 6378.1370;
    # b = 6356.7523;

    # Instead, to be consistent, we used the values used for the line/column
    # <-> lat/lon functions. */
    a = 6378.1690
    b = 6356.5838

    D2R = (np.pi / 180.0)
    R2D = (180.0 / np.pi)

    cos_lat = np.cos(lat * D2R)
    sin_lat = np.sin(lat * D2R)
    cos_lon = np.cos(lon * D2R)
    sin_lon = np.sin(lon * D2R)

    e2 = 1. - (b * b) / (a * a)

    N = a / np.sqrt(1. - e2 * sin_lat * sin_lat)

    x = (N + height) * cos_lat * cos_lon
    y = (N + height) * cos_lat * sin_lon
    z = ((b * b) / (a * a) * N + height) * sin_lat

    qv = []
    qv.append(X - x)
    qv.append(Y - y)
    qv.append(Z - z)

    u = []
    u.append(-sin_lat * cos_lon * qv[0] + -sin_lat * sin_lon * qv[1] + cos_lat * qv[2])
    u.append(-sin_lon *           qv[0] +  cos_lon           * qv[1])
    u.append(cos_lat * cos_lon * qv[0] + -cos_lat * sin_lon * qv[1] + sin_lat * qv[2])

    vza = np.arccos(u[2] / np.sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2])) * R2D

    vaa = np.arctan2(u[1], u[0]) * R2D
    if (vaa < 0.):
        vaa += 360.
    return vza, vaa

# buff = vza_and_vaa(h, pix_lat, pix_lon, X, Y, Z)
# print buff


def get_parallaxed_coor(sat, tle, t, lat, lon, alt, h):
    """

    """
    # Setup orbital class with TLE file
    orbit = Orbital(sat, line1=tle.line1, line2=tle.line2)
    pos = orbit.get_position(t)
    # Calculate observer azimuth and elevation
    azi, ele = orbit.get_observer_look(t, lon, lat, alt)
    # Apply parallax correction for given heights
    x, y, z = parallax_corr(h, azi, ele)
    # WGS84 parameters
    f = 1 / 298.257223563
    a = 6378137.
    # Convert coordinates and azimuth to radians
    radlat = np.deg2rad(lat)
    radlon = np.deg2rad(lon)
    radazi = np.deg2rad(azi)
    # Calculate shifted point coordinates
    nlat, nlon, nazi = vinc_pt(f, a, radlat, radlon, radazi, z)
    # Reconvert to degree
    nlat = np.rad2deg(nlat)
    nlon = np.rad2deg(nlon)
    nazi = np.rad2deg(nazi)

    info = "\n--------------------------------------------------------" + \
           "\n      ----- Parallax correction summary -----" +\
           "\n--------------------------------------------------------" + \
           "\n Satellite Name: " + sat + \
           "\n Observation Time: " + \
           datetime.datetime.strftime(t, "%Y-%m-%d %H:%M:%S") + \
           "\n Satellite Position: " + str(pos[0]) + \
           "\n Satellite Velocity: " + str(pos[1]) + \
           "\n-----------------------------------" + \
           "\n Latitude: " + str(lat) + \
           "\n Longitude: " + str(lon) + \
           "\n Altitude: " + \
           str(alt) + "\n Height: " + str(h) + \
           "\n-----------------------------------" +\
           "\n Azimuth: " + str(azi) + \
           "\n Elevation: " + str(ele) + \
           "\n Parallax Distance: " + str(z) + \
           "\n New Latitude: " + str(nlat) + \
           "\n New Longitude: " + str(nlon) + \
           "\n--------------------------------------------------------"
    logger.debug(info)

    return(z, nlat, nlon)


def parallax_shift(map):
    """ Shift for a given map object and corresponding height information 
        high clouds due to the parallax effect.

    Keyword arguments:

    """
    

# Create TLE file for MSG 10
line1 = "1 38552U 12035B   16165.27110250 -.00000007  00000-0  00000+0 0  9998"
line2 = "2 38552   0.7862  86.5431 0001426 165.7440 107.6925  1.00282029 14565"
msgtle = tlefile.read('meteosat 10', line1=line1, line2=line2)
msgorb = Orbital('meteosat 10', line1=msgtle.line1, line2=msgtle.line2)

# Offenbach test location
t = datetime.datetime(2016, 4, 29, 12, 00)
lat = 50.0956
lon = 8.7761
alt = 109.
h = 8000.
get_parallaxed_coor('meteosat 10', msgtle, t, lat, lon, alt, h)

# Algeria test location
t = datetime.datetime(2016, 4, 29, 12, 00)
lat = 0.00
lon = 25.00
alt = 0.
h = 8000.
get_parallaxed_coor('meteosat 10', msgtle, t, lat, lon, alt, h)

# Example time slots
time_slot = datetime.datetime(2016, 4, 29, 12, 00)
time_slot2 = datetime.datetime(2016, 4, 29, 18, 00)
time_slot3 = datetime.datetime(2016, 4, 29, 00, 00)

# Import test data into pytroll
global_data = GeostationaryFactory.create_scene("meteosat", "10", "seviri",
                                                time_slot)
europe = get_area_def("EuropeCanary")
global_data.load([0.6, 3.9, 10.8], area_extent=europe.area_extent)
lonlats = global_data[10.8].area.get_lonlats()


# Loop for plot dijurnal changes of azimute and elevation
#hr = range(24)
#time = [datetime.datetime(2016, 4, 29, i, 00) for i in hr]
#obslist = [msgorb.get_observer_look(t, oflon, oflat, ofalt) for t in time]
#poslist = [msgorb.get_position(t) for t in time]

#azi = [obs[0] for obs in obslist]
#ele = [obs[1] for obs in obslist]
#fig = plt.subplot()
#plt.plot(hr, azi)
#plt.plot(hr, ele)
#plt.show()
# print(lonlats[1])
# print(snu.seviri_preproc.)
# t = snu.seviri_preproc.vza.__init__()