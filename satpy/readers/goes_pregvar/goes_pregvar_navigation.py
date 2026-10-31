"""GOES pre-GVAR Navigation.

A faithful Python port of the McIDAS GOES D-H earth-location algorithm,
which is itself a direct port of the original McIDAS-X FORTRAN
navigation codicil nvxgoes.dlm (NAVCOM/BETCOM/VASCOM/NAVINI/SATVEC).

Applies to SMS and GOES 1-7 VISSR/VAS AREA files, whose nav block starts
with the 4 ASCII bytes b"GOES" (word 1 of the nav codicil).

License note: the algorithm/formulas below are a derivative of
edu.wisc.ssec.mcidas.GOESnav, which is LGPL-2+ licensed by SSEC
(Tom Whittaker, Tommy Jasmin, Tom Rink, Don Murray, and others).
This port keeps the same terms.

References:
- `McIDAS Programmer's Manual`_, "GOES-7 AAA navigation block"
- SSEC VisAD `GOESnav.java`_

.. _McIDAS Programmer's Manual:
    https://www.ssec.wisc.edu/mcidas/doc/prog_man/2015/prog_man.pdf
.. _GOESnav.java:
    https://github.com/visad/visad/blob/master/core/src/edu/wisc/ssec/mcidas/GOESnav.java
"""

import collections
import datetime
import math

import numba
import numpy as np

GOES_MAGIC = b"GOES"

_RDPDG = np.pi / 180.0
_MCMISSING = -2139062144  # McIDAS "missing" (0x80808080 as signed int32), used for iarr[28]/skew

_EQUATORIAL_RADIUS_KM = 6378.137


def _packed_to_double(value):
    """Convert a packed decimal integer to double degrees (or hours).

    Mirrors McIDASUtil.mcPackedIntegerToDouble: SIGN DDD MM SS (or HH MM SS).
    """
    val = abs(int(value))
    dvalue = (val // 10000) + ((val // 100) % 100) / 60.0 + (val % 100) / 3600.0
    return -dvalue if value < 0 else dvalue


def _icon1(yymmdd):
    """YYMMDD -> YYDDD (year/day-of-year), per nvxgoes ICON1."""
    num = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    year = (yymmdd // 10000) % 100
    month = (yymmdd // 100) % 100
    day = yymmdd % 100
    if month < 1 or month > 12:
        month = 1
    julday = day + num[month - 1]
    if year % 4 == 0 and month > 2:
        julday += 1
    return 1000 * year + julday


def _leapyr(iy):
    return 366 - ((iy % 4) + 3) // 4


def _mc_daytime_to_secs(yrday, hms):
    """Convert a YYDDD day and HHMMSS time to seconds since the epoch.

    Matches McIDASUtil.mcDayTimeToSecs closely enough for the *difference*
    used in timdif() below (which is all this port needs -- it's only ever
    used as raha's time delta).
    """
    year = yrday // 1000
    year = 1900 + year if year >= 30 else 2000 + year
    day = yrday % 1000
    # days since a fixed reference (Jan 1 of `year`), matching Java's
    # Calendar-based day-of-year approach for the purpose of subtraction
    dt = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day - 1)
    seconds = _packed_to_double(hms) * 3600.0
    epoch = datetime.datetime(1970, 1, 1)
    return (dt - epoch).total_seconds() + seconds


def _timdif(yrday1, hms1, yrday2, hms2):
    return (_mc_daytime_to_secs(yrday2, hms2) - _mc_daytime_to_secs(yrday1, hms1)) / 60.0


def _check_magic(iarr, endian):
    expected = int.from_bytes(GOES_MAGIC, "big" if endian == ">" else "little")
    got = int(iarr[0]) & 0xFFFFFFFF
    if got != expected:
        raise ValueError(
            f"Not a GOES nav block (word 1 = {iarr[0]!r}, expected "
            f"{expected!r} for endian={endian!r}; if you're sure the "
            f"raw bytes are b'GOES', this is probably an endian "
            f"mismatch -- pass the same endian AreaFile detected)"
        )


# Scalar constants handed to _to_latlon_kernel() as one immutable bundle
# (numba treats a namedtuple of floats as a plain struct), instead of a
# ~45-argument call. Built by GOESNav._kernel_params().
_KernelParams = collections.namedtuple(
    "_KernelParams",
    [
        # scan / image geometry
        "numsen", "tmpscl", "pictim", "piclin", "radlin", "radele",
        "picele", "gamma", "gamdot",
        # spin-axis rotation matrix (b..) and attitude matrix (rotm..)
        "b11", "b12", "b13", "b21", "b22", "b23", "b31", "b32", "b33",
        "rotm11", "rotm13", "rotm21", "rotm23", "rotm31", "rotm33",
        # earth model and earth-rotation terms
        "bsq_over_asq", "asq", "bsq", "emega", "xref", "rdpdg",
        # Kepler orbit (see GOESNav._satvec_init())
        "tdife", "xmmc", "oeccen", "srome2", "epsiln",
        "px", "py", "pz", "qx", "qy", "qz",
    ],
)


def _double_to_packed(dvalue):
    """Convert double degrees (or hours) to a packed decimal integer."""
    dval = abs(dvalue)
    j = int(3600.0 * dval + 0.5)
    value = 10000 * (j // 3600) + 100 * ((j // 60) % 60) + j % 60
    return -value if dvalue < 0.0 else value


def _anomaly_hours(semima, oeccen, xmeana):
    """Return the time in hours that the orbit takes to reach the mean anomaly ``xmeana`` (degrees)."""
    re = 6378.388
    gracon = 0.07436574
    axmmc = gracon * (np.sqrt(re / semima)) ** 3
    xmanom = (np.pi / 180.0) * xmeana
    return (xmanom - oeccen * np.sin(xmanom)) / (60.0 * axmmc)


def _day_rollover(time):
    """Wrap a time in hours into [0, 24) and return it with the whole-day shift that was applied."""
    if time > 48.0:
        return time - 48.0, 2
    if time > 24.0:
        return time - 24.0, 1
    if time < -24.0:
        return time + 48.0, -2
    if time < 0.0:
        return time + 24.0, -1
    return time, 0


def _add_days(ietimy, iday):
    """Shift a YYDDD epoch by ``iday`` days, following nvxgoes' year rollover."""
    jyear = (ietimy // 1000) % 100 + 1000
    jday = ietimy % 1000 + iday
    if jday < 1:
        jyear -= 1
        jday += _leapyr(jyear)
    elif jday > _leapyr(jyear):
        jday += _leapyr(jyear)
        jyear += 1
    return 1000 * (jyear % 100) + jday


def _epoch(ietimy, ietimh, anomaly_hours):
    """Move the orbit epoch (YYDDD, HHMMSS) back to perigee, ``anomaly_hours`` earlier."""
    time, iday = _day_rollover(_packed_to_double(ietimh) - anomaly_hours)
    if iday != 0:
        ietimy = _add_days(ietimy, iday)
    return ietimy, _double_to_packed(time)


def _validate_nav_words(iarr):
    """Raise ValueError if the nav block lacks usable orbit, pointing or spin parameters."""
    if (iarr[6:12] <= 0).all():
        raise ValueError(
            "Invalid orbital parameters (all <= 0) -- this file's nav "
            "block has no real orbital elements."
        )
    if iarr[4] == 0:
        raise ValueError("Invalid orbit type")
    if not iarr[12:15].any():
        raise ValueError("Invalid ascension/declination parameters")
    if iarr[15] == 0:
        raise ValueError("Invalid spin period")


def _has_beta_scan_times(iss):
    """Tell whether satellite source ``iss`` records the first/last scan line times in its nav block."""
    return iss > 25 or iss == 12


@numba.njit
def _kepler_position(samtim, p):
    """Return the satellite position (satx, saty, satz) at time ``samtim``.

    Kepler's equation is solved by fixed-point iteration (see GOESNav._satvec()).
    """
    xmanom = p.xmmc * (p.tdife + samtim * 60.0)
    ecanm1 = xmanom
    ecanom = ecanm1
    for _ in range(20):
        ecanom = xmanom + p.oeccen * math.sin(ecanm1)
        if abs(ecanom - ecanm1) < p.epsiln:
            break
        ecanm1 = ecanom
    xomega = math.cos(ecanom) - p.oeccen
    yomega = p.srome2 * math.sin(ecanom)
    return (
        xomega * p.px + yomega * p.qx,
        xomega * p.py + yomega * p.qy,
        xomega * p.pz + yomega * p.qz,
    )


@numba.njit
def _scan_line_state(parlin, p):
    """Return everything that depends only on the scan-line group ``parlin``.

    That is the sample time, the satellite position, the attitude rotation
    angle and the cos/sin of the earth-rotation angle.
    """
    samtim = p.tmpscl * parlin + p.pictim
    sat = _kepler_position(samtim, p)
    xcor = p.b11 * sat[0] + p.b12 * sat[1] + p.b13 * sat[2]
    ycor = p.b21 * sat[0] + p.b22 * sat[1] + p.b23 * sat[2]
    rot = math.atan2(ycor, xcor) + math.pi
    angle = p.emega * samtim + p.xref
    return samtim, sat, rot, math.cos(angle), math.sin(angle)


@numba.njit
def _line_of_sight(ylin, yele, p):
    """Return the line-of-sight vector (elo, emo, eno) for the scan angles ``ylin`` / ``yele``."""
    coslin, sinlin = math.cos(ylin), math.sin(ylin)
    sinele, cosele = math.sin(yele), math.cos(yele)

    eli = p.rotm11 * coslin - p.rotm13 * sinlin
    emi = p.rotm21 * coslin - p.rotm23 * sinlin
    eni = p.rotm31 * coslin - p.rotm33 * sinlin
    temp = eli
    eli = cosele * eli + sinele * emi
    emi = -sinele * temp + cosele * emi

    return (
        p.b11 * eli + p.b21 * emi + p.b31 * eni,
        p.b12 * eli + p.b22 * emi + p.b32 * eni,
        p.b13 * eli + p.b23 * emi + p.b33 * eni,
    )


@numba.njit
def _intersect_earth(sat, los, p):
    """Intersect a line of sight with the earth ellipsoid.

    Returns ``(hit, point)``; ``hit`` is False (and ``point`` meaningless)
    where the line of sight misses the earth.
    """
    satx, saty, satz = sat
    elo, emo, eno = los
    onemsq = 1.0 - p.bsq_over_asq
    aq = p.bsq_over_asq + onemsq * eno * eno
    bq = 2.0 * ((elo * satx + emo * saty) * p.bsq_over_asq + eno * satz)
    cq = (satx * satx + saty * saty) * p.bsq_over_asq + satz * satz - p.bsq
    rad = bq * bq - 4.0 * aq * cq
    if rad < 1.0:
        return False, (0.0, 0.0, 0.0)
    s = -(bq + math.sqrt(rad)) / (2.0 * aq)
    return True, (satx + elo * s, saty + emo * s, satz + eno * s)


@numba.njit
def _geodetic_latlon(point, ct, st, p):
    """Rotate an earth-surface point into the earth-fixed frame and return (lat, lon), East positive."""
    x, y, z = point
    x1 = ct * x + st * y
    y1 = -st * x + ct * y
    if x1 == 0.0 and y1 == 0.0:
        return np.nan, np.nan
    a = math.atan(z / math.sqrt(x1 * x1 + y1 * y1))
    lat = math.atan2(p.asq * math.sin(a), p.bsq * math.cos(a)) / p.rdpdg
    lon = math.atan2(y1, x1) / p.rdpdg
    return lat, lon


@numba.njit
def _to_latlon_kernel(xlin, xele, p):
    """Convert flattened line/element image coordinates to (lat, lon) in degrees.

    Longitude is East positive.

    The satellite position and the quantities derived only from it
    (earth-rotation angle, attitude rotation) depend on the scan-line
    group (`parlin`) alone, so they are recomputed only when `parlin`
    changes from one pixel to the next. For a (line, element) grid
    flattened in C order -- lines varying slowest -- that is once per
    line group instead of once per pixel; any other ordering is still
    correct, just without the saving.
    """
    n = xlin.shape[0]
    lat = np.empty(n, dtype=np.float64)
    lon = np.empty(n, dtype=np.float64)

    have_line = False
    last_parlin = 0.0
    samtim = rot = ct = st = 0.0
    sat = (0.0, 0.0, 0.0)

    for i in range(n):
        parlin = (np.rint(xlin[i]) - 1.0) // p.numsen + 1.0
        if (not have_line) or parlin != last_parlin:
            have_line = True
            last_parlin = parlin
            samtim, sat, rot, ct, st = _scan_line_state(parlin, p)

        ylin = (xlin[i] - p.piclin) * p.radlin
        yele = (xele[i] - p.picele + p.gamma + p.gamdot * samtim) * p.radele - rot
        hit, point = _intersect_earth(sat, _line_of_sight(ylin, yele, p), p)
        if hit:
            lat[i], lon[i] = _geodetic_latlon(point, ct, st, p)
        else:
            lat[i] = np.nan
            lon[i] = np.nan
    return lat, lon


class GOESNav:
    """Port of edu.wisc.ssec.mcidas.GOESnav.

    Construct with the raw nav-block words (as signed 32-bit ints, 0-based
    array matching the Java `iarr` -- iarr[0] is nav word 1 == b"GOES").

    Also needs image-coordinate setup, equivalent to AREAnav's:
        setImageStart(dir_ul_line, dir_ul_element)
        setRes(dir_line_res, dir_elem_res)
        setStart(0, 0); setMag(1, 1)
    which is handled automatically from the AreaFile.dir dict passed in.
    """

    def __init__(self, iarr, area_dir, endian=">"):
        """Set up the navigation from a nav block and its AREA directory.

        Requires a nav block with real, populated orbital elements.

        Args:
            iarr: Raw nav-block words as signed 32-bit ints (0-based, like the
                Java `iarr`; ``iarr[0]`` is b"GOES").
            area_dir: The ``dir`` dict of the AreaFile, which supplies the
                image coordinate setup.
            endian: '>' or '<', matching whatever byte order the AreaFile
                detected and used to parse the nav words (AreaFile._endian).
                Needed because the "GOES" magic in word 1 is 4 ASCII bytes
                reinterpreted as a signed int32 in that same byte order --
                get this wrong and the magic check (and everything
                downstream) silently reads garbage.
        """
        iarr = np.asarray(iarr, dtype=np.int64)
        _check_magic(iarr, endian)
        _validate_nav_words(iarr)

        self.navday = int(iarr[1]) % 100000
        self._set_constants()
        self._set_orbit(iarr)
        self._set_scan_geometry(iarr)
        self._set_attitude(iarr)
        self._set_spin_axis(iarr)
        self._set_timing(iarr)
        self._set_image_coords(area_dir)
        self._satvec_init()

    def _set_constants(self):
        """Set the earth model and earth-rotation constants."""
        self.emega = 0.26251617
        self.ab = 40546851.22
        self.asq = 40683833.48
        self.bsq = 40410330.18
        self.r = 6371.221
        self.rdpdg = 1.745329252e-02
        self.bsq_over_asq = self.bsq / self.asq

    def _set_orbit(self, iarr):
        """Set the Kepler orbital elements and the orbit epoch from the nav block."""
        ietimy = _icon1(int(iarr[4]))
        ietimh = 100 * (int(iarr[5]) // 100) + round(0.6 * (int(iarr[5]) % 100))
        self.semima = iarr[6] / 100.0
        self.oeccen = iarr[7] / 1000000.0
        self.orbinc = iarr[8] / 1000.0
        xmeana = iarr[9] / 1000.0
        self.perhel = iarr[10] / 1000.0
        self.asnode = iarr[11] / 1000.0
        self.ietimy, self.ietimh = _epoch(ietimy, ietimh, _anomaly_hours(self.semima, self.oeccen, xmeana))

    def _set_scan_geometry(self, iarr):
        """Set the spin-scan geometry: picture center, spin rate and angles per line/element."""
        piclin = float(iarr[14])
        if iarr[14] >= 1000000:
            piclin = piclin / 10000.0
        self.piclin = piclin

        spinra = iarr[15] / 1000.0
        if spinra < 300.0:
            spinra = 60000.0 / spinra
        self.tmpscl = spinra / 3600000.0

        deglin = _packed_to_double(int(iarr[16]))
        lintot = int(iarr[17])
        degele = _packed_to_double(int(iarr[18]))
        ieltot = int(iarr[19])
        self.numsen = max((lintot // 100000) % 100, 1)
        totlin = self.numsen * (lintot % 100000)
        self.radlin = self.rdpdg * deglin / (totlin - 1.0)
        self.radele = self.rdpdg * degele / (ieltot - 1.0)
        self.picele = (1.0 + ieltot) / 2.0

    def _set_attitude(self, iarr):
        """Set the attitude rotation matrix from pitch, yaw, roll and skew."""
        cpitch = self.rdpdg * _packed_to_double(int(iarr[20]))
        cyaw = self.rdpdg * _packed_to_double(int(iarr[21]))
        croll = self.rdpdg * _packed_to_double(int(iarr[22]))
        skew = 0.0 if iarr[28] == _MCMISSING else iarr[28] / 100000.0

        pskew = np.arctan2(skew, self.radlin / self.radele)
        stp, ctp = np.sin(cpitch), np.cos(cpitch)
        sty, cty = np.sin(cyaw - pskew), np.cos(cyaw - pskew)
        str_, ctr = np.sin(croll), np.cos(croll)
        self.rotm11 = ctr * ctp
        self.rotm13 = sty * str_ * ctp + cty * stp
        self.rotm21 = -str_
        self.rotm23 = sty * ctr
        self.rotm31 = -ctr * stp
        self.rotm33 = cty * ctp - sty * str_ * stp

    def _set_spin_axis(self, iarr):
        """Set the spin-axis rotation matrix from the declination and right ascension."""
        dec = _packed_to_double(int(iarr[12])) * self.rdpdg
        sindec, cosdec = np.sin(dec), np.cos(dec)
        ras = _packed_to_double(int(iarr[13])) * self.rdpdg
        sinras, cosras = np.sin(ras), np.cos(ras)
        self.b11 = -sinras
        self.b12 = cosras
        self.b13 = 0.0
        self.b21 = -sindec * cosras
        self.b22 = -sindec * sinras
        self.b23 = cosdec
        self.b31 = cosdec * cosras
        self.b32 = cosdec * sinras
        self.b33 = sindec

    def _set_timing(self, iarr):
        """Set the earth-rotation reference, the picture time and the scan line times."""
        raha = _timdif(74001, 0, self.navday, 0) * 1.00273791 / 4.0 + 100.26467
        self.xref = (raha % 360.0) * self.rdpdg
        self.pictim = _packed_to_double(int(iarr[2]))
        self.gamma = float(iarr[38]) / 100.0
        self.gamdot = float(iarr[39]) / 100.0
        self._set_scan_times(iarr)

    def _set_scan_times(self, iarr):
        """Set the first/last scan line numbers and times."""
        if iarr[30] > 0 and _has_beta_scan_times(int(iarr[1]) // 100000):
            self.scan1 = float(iarr[30])
            self.time1 = _packed_to_double(int(iarr[31]))
            self.scan2 = float(iarr[34])
            self.time2 = _packed_to_double(int(iarr[35]))
        else:
            self.scan1 = 1.0
            self.time1 = self.pictim
            self.scan2 = float(int(iarr[17]) % 100000)
            self.time2 = self.time1 + self.scan2 * self.tmpscl

    def _set_image_coords(self, area_dir):
        """Set the image coordinates up like AREAnav does for AREA-file navigation.

        Equivalent to AREAnav's setImageStart/setRes/setStart/setMag.
        """
        self.start_image_line = area_dir["ul_line"]
        self.start_image_elem = area_dir["ul_element"]
        self.res_line = area_dir["line_res"]
        self.res_elem = area_dir["elem_res"]
        self.start_line = 0
        self.start_elem = 0
        self.mag_line = 1
        self.mag_elem = 1

    def _satvec_init(self):
        """Precompute the orbital-plane basis vectors used by _satvec().

        Only called for the normal Kepler-propagated constructor.
        """
        rdpdg = np.pi / 180.0
        re = 6378.388
        gracon = 0.07436574

        o = rdpdg * self.orbinc
        p = rdpdg * self.perhel
        a = rdpdg * self.asnode
        so, co = np.sin(o), np.cos(o)
        sp, cp = np.sin(p) * self.semima, np.cos(p) * self.semima
        sa, ca = np.sin(a), np.cos(a)
        self._px = cp * ca - sp * sa * co
        self._py = cp * sa + sp * ca * co
        self._pz = sp * so
        self._qx = -sp * ca - cp * sa * co
        self._qy = -sp * sa + cp * ca * co
        self._qz = cp * so
        self._srome2 = np.sqrt(1.0 - self.oeccen) * np.sqrt(1.0 + self.oeccen)
        self._xmmc = gracon * re * np.sqrt(re / self.semima) / self.semima

        iey = (self.ietimy // 1000) % 100
        ied = self.ietimy % 1000
        iefac = (iey - 1) // 4 + 1
        de = 365 * (iey - 1) + iefac + ied - 1
        te = 1440.0 * de + 60.0 * _packed_to_double(self.ietimh)

        inavy = (self.navday // 1000) % 100
        inavd = self.navday % 1000
        infac = (inavy - 1) // 4 + 1
        dnav = 365 * (inavy - 1) + infac + inavd - 1

        self._tdife = dnav * 1440.0 - te
        self._epsiln = 1.0e-8

    def _satvec(self, samtim):
        """Return the satellite position (x, y, z) at time ``samtim``.

        ``samtim`` may be a numpy array; the result is ECI-ish arrays
        (Earth-center-to-satellite, in the pre-earth-rotation frame that
        to_latlon()/get_subpoint() then rotate by (emega*samtim+xref)).

        Kepler's-equation solve from real orbital elements (fixed-point
        iteration, matching the Java's plain loop).
        """
        samtim = np.asarray(samtim, dtype=np.float64)

        timsam = samtim * 60.0
        diftim = self._tdife + timsam
        xmanom = self._xmmc * diftim
        ecanm1 = xmanom.copy()
        ecanom = ecanm1
        for _ in range(20):
            ecanom = xmanom + self.oeccen * np.sin(ecanm1)
            if np.all(np.abs(ecanom - ecanm1) < self._epsiln):
                ecanm1 = ecanom
                break
            ecanm1 = ecanom
        xomega = np.cos(ecanom) - self.oeccen
        yomega = self._srome2 * np.sin(ecanom)
        x = xomega * self._px + yomega * self._qx
        y = xomega * self._py + yomega * self._qy
        z = xomega * self._pz + yomega * self._qz
        return x, y, z

    def _nxyzll(self, x, y, z):
        """Convert earth-fixed x, y, z to geodetic (lat, lon) in degrees, with lon positive West."""
        with np.errstate(invalid="ignore", divide="ignore"):
            a = np.arctan(z / np.sqrt(x * x + y * y))
            lat = np.arctan2(self.asq * np.sin(a), self.bsq * np.cos(a)) / self.rdpdg
            lon = -np.arctan2(y, x) / self.rdpdg
        bad = (x == 0.0) & (y == 0.0)
        lat = np.where(bad, np.nan, lat)
        lon = np.where(bad, np.nan, lon)
        return lat, lon

    def _area_to_image(self, line, elem):
        """AREAnav.areaCoordToImageCoord, unflipped/unmagnified case."""
        img_line = self.start_image_line + (self.res_line * (line - self.start_line)) / self.mag_line
        img_elem = self.start_image_elem + (self.res_elem * (elem - self.start_elem)) / self.mag_elem
        return img_line, img_elem

    def _kernel_params(self):
        """Bundle the scalar constants _to_latlon_kernel() needs."""
        return _KernelParams(
            numsen=float(self.numsen), tmpscl=float(self.tmpscl),
            pictim=float(self.pictim), piclin=float(self.piclin),
            radlin=float(self.radlin), radele=float(self.radele),
            picele=float(self.picele), gamma=float(self.gamma),
            gamdot=float(self.gamdot),
            b11=float(self.b11), b12=float(self.b12), b13=float(self.b13),
            b21=float(self.b21), b22=float(self.b22), b23=float(self.b23),
            b31=float(self.b31), b32=float(self.b32), b33=float(self.b33),
            rotm11=float(self.rotm11), rotm13=float(self.rotm13),
            rotm21=float(self.rotm21), rotm23=float(self.rotm23),
            rotm31=float(self.rotm31), rotm33=float(self.rotm33),
            bsq_over_asq=float(self.bsq_over_asq), asq=float(self.asq),
            bsq=float(self.bsq), emega=float(self.emega),
            xref=float(self.xref), rdpdg=float(self.rdpdg),
            tdife=float(self._tdife), xmmc=float(self._xmmc),
            oeccen=float(self.oeccen), srome2=float(self._srome2),
            epsiln=float(self._epsiln),
            px=float(self._px), py=float(self._py), pz=float(self._pz),
            qx=float(self._qx), qy=float(self._qy), qz=float(self._qz),
        )

    def to_latlon(self, line, elem):
        """Convert line/element (AREA/file coordinates) to (lat, lon) in degrees.

        East positive / North positive. NaN where the line of sight misses
        the earth ellipsoid. Accepts any broadcastable pair of arrays and
        returns arrays of the broadcast shape.
        """
        line, elem = np.broadcast_arrays(
            np.asarray(line, dtype=np.float64),
            np.asarray(elem, dtype=np.float64),
        )
        shape = line.shape
        xlin, xele = self._area_to_image(line, elem)
        lat, lon = _to_latlon_kernel(
            np.ascontiguousarray(xlin).ravel(),
            np.ascontiguousarray(xele).ravel(),
            self._kernel_params(),
        )
        return lat.reshape(shape), lon.reshape(shape)

    def get_subpoint(self):
        """Return the satellite nadir (lat, lon) in degrees, east positive.

        Evaluated at time `time1` (nominal image start time, or first-beta
        scan time for VAS-era files). Used by get_nadir_sampling() to center
        the uniform-sampling grid, and a cheap cross-check of the orbital
        solution.
        """
        samtim = self.time1
        satx, saty, satz = self._satvec(np.asarray(samtim))
        ct = np.cos(self.emega * samtim + self.xref)
        st = np.sin(self.emega * samtim + self.xref)
        x1 = ct * satx + st * saty
        y1 = -st * satx + ct * saty
        lat, lon = self._nxyzll(x1, y1, satz)
        return float(lat), float(-lon)  # East positive

    def get_nadir_sampling(self):
        """Return the sub-satellite point, altitude and nadir pixel spacing.

        The spacing is that of the AREA grid this object was built for.

        Returns a dict with ``sub_lon`` / ``sub_lat`` (degrees, east/north
        positive), ``altitude`` (metres above the equatorial radius, at
        the nominal image time) and ``line_step`` / ``elem_step`` (metres
        between neighbouring AREA lines / elements at nadir). The steps
        come from the scan angle per full-resolution line/element
        (``radlin`` / ``radele``) scaled by the AREA directory's
        line/element resolution factors.
        """
        sub_lat, sub_lon = self.get_subpoint()
        satx, saty, satz = self._satvec(np.asarray(self.time1))
        altitude_m = (float(np.sqrt(satx ** 2 + saty ** 2 + satz ** 2)) - _EQUATORIAL_RADIUS_KM) * 1000.0
        return {
            "sub_lon": sub_lon,
            "sub_lat": sub_lat,
            "altitude": altitude_m,
            "line_step": altitude_m * self.radlin * self.res_line / self.mag_line,
            "elem_step": altitude_m * self.radele * self.res_elem / self.mag_elem,
        }
