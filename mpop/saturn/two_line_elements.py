"""
Current Day Number: 11364.541666667 (10 February 2011, 13:00:00 UTC)

// SGP4 test

NORAD Catalog No: 25544
Satellite Name: ISS (ZARYA)
TLE Line 1: 1 25544U 98067A   11036.41885377  .00015023  00000-0  11493-3 0  7736
TLE Line 2: 2 25544  51.6463 116.5688 0003364 277.8443 128.8911 15.72389581700166

SDP4: 0
Period: 91.580357527185
Age: 5.1228128965013
Lat: -51.62128386358
Lon: 18.52912756499
Alt: 367.39281472872
Range: 10318.551827092
Range Rate: 1.2960949888636
Azi: 168.25676645006
Ele: -50.891962381116
Velocity: 7.6898969280099
Visible: 0
Eclipsed: 0
Eclipse Depth: -1.1299921383153
"""
import datetime
import pprint
import sys
import numpy as np


CK2 = 5.413080e-4
CK4 = 0.62098875e-6
E6A = 1.0e-6
QOMS2T = 1.88027916e-9
S = 1.01222928
XJ3 = -0.253881e-5
XKE = 0.743669161e-1
XKMPER = 6378.137
XMNPDA = 1440.0
AE = 1.0
# earth flattening
F = 1/298.257223563

TLE = """1 25544U 98067A   11036.41885377  .00015023  00000-0  11493-3 0  7736
2 25544  51.6463 116.5688 0003364 277.8443 128.8911 15.72389581700166"""

TLE = """1 27424U 02022A   11045.03153664  .00000197  00000-0  53804-4 0  8714
2 27424  98.2146 347.6229 0001550  78.4076 281.7310 14.57100380467139"""

import urllib2

if sys.version_info < (2, 5):
    import time
    def strptime(string, fmt=None):
        """This function is available in the datetime module only
        from Python >= 2.5.
        """
        
        return datetime.datetime(*time.strptime(string, fmt)[:6])
else:
    strptime = datetime.datetime.strptime
class Tle(object):

    def __init__(self, tle=None, satellite=None):
        self.tle = tle
        if satellite:
            tles_dict = {}

            import glob
            filelist = glob.glob("/data/24/saf/polar_in/tle2/tle-*.txt")
            if len(filelist) > 0:
                filelist.sort()
                tlef = open(filelist[-1])
                tles = [item.strip() for item in tlef]
                tlef.close()
                for i in xrange(0,len(tles)-2,3):
                    tles_dict[tles[i]] = tles[i+1]+"\n"+tles[i+2]
            else:
                for f in ["resource.txt", "weather.txt"]:
                    
                    tles = urllib2.urlopen("http://celestrak.com/NORAD/elements/" + f).readlines()
                    tles = [item.strip() for item in tles]

                    for i in xrange(0,len(tles)-2,3):
                        tles_dict[tles[i]] = tles[i+1]+"\n"+tles[i+2]

            self._read_tle(tles_dict[satellite.upper()])
            self._preprocess()


    def _read_tle(self, lines):

        def _read_tle_decimal(rep):
            num = int(rep[:-2]) * 1.0e-5
            exp = int(rep[-2:])
            return num * 10 ** exp

        tlist = lines.split()
        self.tle = {}
        self.tle["satnumber"] = tlist[1][:5]
        self.tle["classification"] = tlist[1][5:]
        self.tle["id_launch_year"] = tlist[2][:2]
        self.tle["id_launch_number"] = tlist[2][2:5]
        self.tle["id_launch_piece"] = tlist[2][5:]
        self.tle["epoch_year"] = int(tlist[3][:2])
        self.tle["epoch_day"] = float(tlist[3][2:])
        self.tle["epoch"] = (strptime(tlist[3][:2], "%y") +
                        datetime.timedelta(days=float(tlist[3][2:]) - 1))
        self.tle["mean_motion_derivative"] = float(tlist[4])
        self.tle["mean_motion_sec_derivative"] = _read_tle_decimal(tlist[5])
        self.tle["bstar"] = _read_tle_decimal(tlist[6])
        self.tle["ephemeris_type"] = int(tlist[7])
        self.tle["element_number"] = int(tlist[8][:-1])

        self.tle["inclination"] = float(tlist[11])
        self.tle["right_ascension"] = float(tlist[12])
        self.tle["excentricity"] = int(tlist[13]) * 10 ** -7
        self.tle["arg_perigee"] = float(tlist[14])
        self.tle["mean_anomaly"] = float(tlist[15])
        self.tle["mean_motion"] = float(tlist[16][:11])
        self.tle["orbit"] = int(tlist[16][11:-1])


    def _preprocess(self):
        self.tle["inclination"] = np.deg2rad(self.tle["inclination"])
        self.tle["right_ascension"] = np.deg2rad(self.tle["right_ascension"])
        self.tle["arg_perigee"] = np.deg2rad(self.tle["arg_perigee"])
        self.tle["mean_anomaly"] = np.deg2rad(self.tle["mean_anomaly"])

        self.tle["mean_motion"] *= (np.pi * 2 / XMNPDA)
        self.tle["mean_motion_derivative"] *= np.pi * 2 / XMNPDA ** 2
        self.tle["mean_motion_sec_derivative"] *= np.pi * 2 / XMNPDA ** 3
        self.tle["bstar"] *= AE

        n_0 = self.tle["mean_motion"]
        k_e = XKE
        k_2 = CK2
        i_0 = self.tle["inclination"]
        e_0 = self.tle["excentricity"]

        a_1 = (k_e / n_0) ** (2.0/3)
        delta_1 = ((3/2.0) * (k_2 / a_1**2) * ((3 * np.cos(i_0)**2 - 1) /
                                              (1 - e_0**2)**(2.0/3)))

        a_0 = a_1 * (1 - delta_1/3 - delta_1**2 - (134.0/81) * delta_1**3)

        delta_0 = ((3/2.0) * (k_2 / a_0**2) * ((3 * np.cos(i_0)**2 - 1) /
                                              (1 - e_0**2)**(2.0/3)))

        # original mean motion
        n_0pp = n_0 / (1 + delta_0)
        self.tle["original_mean_motion"] = n_0pp

        # semi major axis
        a_0pp = a_0 / (1 - delta_0)
        self.tle["semi_major_axis"] = a_0pp

        self.tle["period"] = np.pi * 2 / n_0pp

        self.tle["perigee"] = (a_0pp * (1 - e_0) / AE - AE) * XKMPER

        now = self.tle["epoch"]

        self.tle["right_ascension_lon"] = (self.tle["right_ascension"]
                                           - gmst(now))

        if self.tle["right_ascension_lon"] > np.pi:
            self.tle["right_ascension_lon"] -= 2 * np.pi

    def get_position(self, time):
        # for near earth orbits, period must be < 255 minutes


        perigee = self.tle["perigee"]
        a_0pp = self.tle["semi_major_axis"]
        e_0 = self.tle["excentricity"]
        i_0 = self.tle["inclination"]
        n_0pp = self.tle["original_mean_motion"]
        k_2 = CK2
        k_4 = CK4
        k_e = XKE
        bstar = self.tle["bstar"]
        w_0 = self.tle["arg_perigee"]
        M_0 = self.tle["mean_anomaly"]
        W_0 = self.tle["right_ascension"]
        t_0 = self.tle["epoch"]
        A30 = -XJ3 * AE**3

        if perigee < 98:
            s = 20/XKMPER + AE
            qoms2t = (QOMS2T ** 0.25 + S - s) ** 4
        elif perigee < 156:
            s = a_0pp * (1 - e_0) - S + AE 
            qoms2t = (QOMS2T ** 0.25 + S - s) ** 4
        else:
            qoms2t = QOMS2T
            s = S

        theta = np.cos(i_0)
        xi = 1 / (a_0pp - s)
        beta_0 = np.sqrt(1 - e_0 ** 2)
        eta = a_0pp * e_0 * xi

        C_2 = (qoms2t * xi**4 * n_0pp * (1 - eta**2)**(-3.5) *
               (a_0pp * (1 + 1.5 * eta**2 + 4 * e_0 * eta + e_0 * eta**3) +
                1.5 * (k_2 * xi) / (1 - eta**2) * (-0.5 + 1.5 * theta**2)*
                (8 + 24 * eta**2 + 3 * eta**4)))

        C_1 = bstar * C_2

        C_3 = (qoms2t * xi ** 5 * A30 * n_0pp * AE * np.sin(i_0) / (k_2 * e_0))

        coef = 2 * qoms2t * xi**4 * a_0pp * beta_0**2*(1-eta**2)**(-7/2.0)

        C_4 = (coef * n_0pp *
               ((2 * eta * (1 + e_0 * eta) + e_0/2.0 + (eta**3)/2.0) -
                2 * k_2 * xi / (a_0pp * (1 - eta**2)) *
                (3*(1-3*theta**2) * (1 + (3*eta**2)/2.0 - 2*e_0*eta - e_0*eta**3/2.0) +
                 3/4.0*(1-theta**2)*(2*eta**2 - e_0*eta - e_0*eta**3)*np.cos(2*w_0))))

        C_5 = coef * (1 + 11/4.0 * eta * (eta + e_0) + e_0 * eta**3)
        D_2 = 4 * a_0pp * xi * C_1**2
        D_3 = 4/3.0 * a_0pp * xi**2 * (17*a_0pp + s) * C_1**3
        D_4 = 2/3.0 * a_0pp * xi**3 * (221*a_0pp + 31*s) * C_1**4

        # Secular effects of atmospheric drag and gravitation
        dt = _days(time - t_0) * XMNPDA

        M_df = (M_0 + (1 +
                       3*k_2*(-1 + 3*theta**2)/(2*a_0pp**2 * beta_0**3) +
                       3*k_2**2*(13 - 78*theta**2 + 137*theta**4)/
                       (16*a_0pp**4*beta_0**7))*
                n_0pp*dt)
        w_df = (w_0 + (-3*k_2*(1 - 5*theta**2)/(2*a_0pp**2*beta_0**4) +
                       3 * k_2**2 * (7 - 114*theta**2 + 395*theta**4)/
                       (16*a_0pp*beta_0**8) +
                       5*k_4*(3-36*theta**2+49*theta**4)/(4*a_0pp**4*beta_0**8))*
                n_0pp*dt)
        W_df = (W_0 + (-3*k_2*theta/(a_0pp**2*beta_0**4) +
                       3*k_2**2*(4*theta- 19*theta**3)/(2*a_0pp**4*beta_0**8) +
                       5*k_4*theta*(3-7*theta**2)/(2*a_0pp**4*beta_0**8))*
                n_0pp*dt)
        deltaw = bstar * C_3 * np.cos(w_0)*dt
        deltaM = (-2/3.0 * qoms2t * bstar * xi**4 * AE / (e_0*eta) *
                  ((1 + eta * np.cos(M_df))**3 - (1 + eta * np.cos(M_0))**3))
        M_p = M_df + deltaw + deltaM
        w = w_df - deltaw - deltaM
        W = (W_df - 21/2.0 * (n_0pp * k_2 * theta)/(a_0pp**2 * beta_0**2) *
             C_1 * dt**2)

        e = (e_0 -
             bstar * C_4 * dt -
             bstar * C_5 * (np.sin(M_p) - np.sin(M_0)))

        a = a_0pp * (1 - C_1 * dt - D_2 * dt**2 - D_3 * dt**3 - D_4 * dt**4)**2
        L = M_p + w + W + n_0pp * (3/2.0 * C_1 * dt**2 +
                                   (D_2 + 2 * C_1 ** 2) * dt**3 +
                                   1/4.0 * (3*D_3 + 12*C_1*D_2 + 10*C_1**3)*dt**4 +
                                   1.0/5 * (3*D_4 + 12*C_1*D_3 + 6*D_2**2 +
                                            30*C_1**2*D_2 + 15*C_1**4)*dt**5)
        beta = np.sqrt(1 - e**2)
        n = k_e / (a ** (3/2.0))

        # Long-period periodic terms
        a_xN = e * np.cos(w)
        a_yNL = A30 * np.sin(i_0) / (4.0 * k_2 * a * beta**2)
        L_L = a_yNL/2 * a_xN * ((3 + 5 * theta) / (1 + theta))
        L_T = L + L_L
        a_yN = e * np.sin(w) + a_yNL

        U = (L_T - W) % (np.pi * 2)

        Epw = U
        for i in range(10):
            DeltaEpw = ((U - a_yN * np.cos(Epw) + a_xN  * np.sin(Epw) - Epw) /
                        (-a_yN * np.sin(Epw) - a_xN * np.cos(Epw) + 1))
            Epw = Epw + DeltaEpw
            if DeltaEpw < 10e-12:
                break

        # preliminary quantities for short-period periodics

        ecosE = a_xN * np.cos(Epw) + a_yN * np.sin(Epw)
        esinE = a_xN * np.sin(Epw) - a_yN * np.cos(Epw)

        e_L = (a_xN**2 + a_yN**2)**(0.5)
        p_L = a * (1 - e_L**2)
        r = a * (1 - ecosE)
        rdot = k_e * np.sqrt(a)/r * esinE
        rfdot = k_e * np.sqrt(p_L) / r
        cosu = a / r * (np.cos(Epw) - a_xN +
                        (a_yN * (esinE) / (1 + np.sqrt(1 - e_L**2))))
        sinu = a / r * (np.sin(Epw) - a_yN +
                        (a_xN * (esinE) / (1 + np.sqrt(1 - e_L**2))))
        u = np.arctan2(sinu, cosu)


        cos2u = np.cos(2*u)
        sin2u = np.sin(2*u)

        Deltar = k_2/(2*p_L) * (1 - theta**2) * cos2u
        Deltau = -k_2/(4*p_L**2) * (7*theta**2 - 1) * sin2u
        DeltaW = 3*k_2 * theta / (2 * p_L**2) * sin2u
        Deltai = 3*k_2 * theta / (2 * p_L**2) * cos2u * np.sin(i_0)
        Deltardot = - k_2 * n / p_L * (1 - theta**2) * sin2u
        Deltarfdot = k_2 * n / p_L * ((1 - theta**2) * cos2u -
                                      3/2.0 * (1 - 3*theta**2))

        # osculating quantities

        r_k = r * (1 - 3/2.0 * k_2 * np.sqrt(1 - e_L**2)/p_L**2 *
                   (3 * theta**2 - 1)) + Deltar
        u_k = u + Deltau
        W_k = W + DeltaW
        i_k = i_0 + Deltai
        rdot_k = rdot + Deltardot
        rfdot_k = rfdot + Deltarfdot

        M_x = -np.sin(W_k) * np.cos(i_k)
        M_y = np.cos(W_k) * np.cos(i_k)
        M_z = np.sin(i_k)

        N_x = np.cos(W_k)
        N_y = np.sin(W_k)
        N_z = 0

        U_x = M_x * np.sin(u_k) + N_x * np.cos(u_k)
        U_y = M_y * np.sin(u_k) + N_y * np.cos(u_k)
        U_z = M_z * np.sin(u_k) + N_z * np.cos(u_k)

        V_x = M_x * np.cos(u_k) - N_x * np.sin(u_k)
        V_y = M_y * np.cos(u_k) - N_y * np.sin(u_k)
        V_z = M_z * np.cos(u_k) - N_z * np.sin(u_k)


        r_x = r_k * U_x
        r_y = r_k * U_y
        r_z = r_k * U_z

        rdot_x = rdot_k * U_x + rfdot_k * V_x
        rdot_y = rdot_k * U_y + rfdot_k * V_y
        rdot_z = rdot_k * U_z + rfdot_k * V_z

        return r_x, r_y, r_z, rdot_x, rdot_y, rdot_z

    def get_latlonalt(self, time):
        pos_x, pos_y, pos_z, vel_x, vel_y, vel_z = self.get_position(time)
        del vel_x, vel_y, vel_z
        lon = ((np.arctan2(pos_y * XKMPER, pos_x * XKMPER) - gmst(time))
               % (2 * np.pi))

        if lon > np.pi:
            lon -= np.pi * 2
        if lon <= -np.pi:
            lon += np.pi * 2

        r = np.sqrt(pos_x ** 2 + pos_y ** 2)
        lat = np.arctan2(pos_z, r)
        e2 = F * (2 - F)
        while True:
            lat2 = lat
            c = 1/(np.sqrt(1 - e2 * (np.sin(lat2) ** 2)))
            lat = np.arctan2(pos_z + c * e2 *np.sin(lat2), r)
            if abs(lat - lat2) < 1e-10:
                break
        alt = r / np.cos(lat)- c;
        alt *= XKMPER
        return lat, lon, alt



def _jdays(time):
    dt = time - datetime.datetime(2000, 1, 1, 12, 0)
    return _days(dt)

def _days(dt):
    return (dt.days +
            (dt.seconds +
             dt.microseconds / (1000000.0)) / (24 * 3600.0))

def gmst(time):
    """Greenwich mean sidereal time, in radians.
    http://celestrak.com/columns/v02n02/
    """
    now = time
    #now = datetime.datetime(1995, 10, 1, 9, 0)
    now0 = datetime.datetime(now.year, now.month, now.day)
    epoch = datetime.datetime(2000, 1, 1, 12, 0)
    du2 = _days(now - epoch)
    du = _days(now0 - epoch)

    dus = (du2 - du) * 86400
    Tu = du / 36525.0
    theta_g_0 = (24110.54841 + Tu * (8640184.812866 +
                                     Tu * (0.093104 - Tu * 6.2 * 10e-6)))
    theta_g = (theta_g_0 + dus * 1.00273790934) % 86400
    return (theta_g / 86400.0) * 2 * np.pi


    
if __name__ == "__main__":
    
    tle = Tle(satellite="aqua")
    import pprint
    pprint.pprint(tle.tle)
    print np.rad2deg(tle.get_latlonalt(tle.tle["epoch"]))
