"""Reader for GOES 8-15 imager data in netCDF format from NOAA CLASS

GOES Imager netCDF files contain geolocated detector counts. If ordering via
NOAA CLASS, select 16 bits/pixel. The instrument oversamples the viewed scene
in E-W direction by a factor of 1.75: IR/VIS pixels are 112/28 urad on a side,
but the instrument samples every 64/16 urad in E-W direction (see [BOOK-I] and
[BOOK-N]).

Important note: Some essential information are missing in the netCDF files,
which might render them inappropriate for certain applications. The unknowns
are:

    1. Subsatellite point
    2. Calibration coefficients
    3. Detector-scanline assignment, i.e. information about which scanline
       was recorded by which detector

Items 1. and 2. are not critical because the images are geo-located and NOAA
provides static calibration coefficients ([VIS], [IR]). The detector-scanline
assignment however cannot be reconstructed properly. This is where an
approximation has to be applied (see below).

Calibration
============

Calibration is performed according to [VIS] and [IR], but with an average
calibration coefficient applied to all detectors in a certain channel. The
reason for and impact of this approximation is described below.

The GOES imager simultaneously records multiple scanlines per sweep using
multiple detectors per channel. The VIS channel has 8 detectors, the IR
channels have 1-2 detectors (see e.g. Figures 3-5a/b, 3-6a/b and 3-7/a-b in
[BOOK-N]). Each detector has its own calibration coefficients, so in order to
perform an accurate calibration, the detector-scanline assignment is needed.

In theory it is known which scanline was recorded by which detector
(VIS: 5,6,7,8,1,2,3,4; IR: 1,2). However, the plate on which the detectors are
mounted flexes due to thermal gradients in the instrument which leads to a N-S
shift of +/- 8 visible or +/- 2 IR pixels. This shift is compensated in the
GVAR scan formation process, but in a way which is hard to reconstruct
properly afterwards. See [GVAR], section 3.2.1. for details.

Since the calibration coefficients of the detectors in a certain channel only
differ slightly, a workaround is to calibrate each scanline with the average
calibration coefficients. A worst case estimate of the introduced error can
be obtained by calibrating all possible counts with both the minimum and the
maximum calibration coefficients and computing the difference. The maximum
differences are:

GOES-8
======
00_7 0.0   %  # Counts are normalized
03_9 0.187 K
06_8 0.0   K  # only one detector
10_7 0.106 K
12_0 0.036 K

GOES-9
========
00_7 0.0   %  # Counts are normalized
03_9 0.0   K  # coefs identical
06_8 0.0   K  # only one detector
10_7 0.021 K
12_0 0.006 K

GOES-10
========
00_7 1.05  %
03_9 0.0   K  # coefs identical
06_8 0.0   K  # only one detector
10_7 0.013 K
12_0 0.004 K

GOES-11
========
00_7 1.25  %
03_9 0.0   K  # coefs identical
06_8 0.0   K  # only one detector
10_7 0.0   K  # coefs identical
12_0 0.065 K

GOES-12
========
00_7 0.8   %
03_9 0.0   K  # coefs identical
06_5 0.044 K
10_7 0.0   K  # coefs identical
13_3 0.0   K  # only one detector

GOES-13
========
00_7 1.31  %
03_9 0.0   K  # coefs identical
06_5 0.085 K
10_7 0.008 K
13_3 0.0   K  # only one detector

GOES-14
========
00_7 0.66  %
03_9 0.0   K  # coefs identical
06_5 0.043 K
10_7 0.006 K
13_3 0.003 K

GOES-15
========
00_7 0.86  %
03_9 0.0   K  # coefs identical
06_5 0.02  K
10_7 0.009 K
13_3 0.008 K


References:

[GVAR] https://goes.gsfc.nasa.gov/text/GVARRDL98.pdf
[BOOK-N] https://goes.gsfc.nasa.gov/text/GOES-N_Databook/databook.pdf
[BOOK-I] https://goes.gsfc.nasa.gov/text/databook/databook.pdf
[IR] https://www.ospo.noaa.gov/Operations/GOES/calibration/gvar-conversion.html
[VIS] https://www.ospo.noaa.gov/Operations/GOES/calibration/goes-vis-ch-calibration.html
[FAQ] https://www.ncdc.noaa.gov/sites/default/files/attachments/Satellite-Frequently-Asked-Questions_2.pdf
[SCHED-W] http://www.ospo.noaa.gov/Operations/GOES/west/imager-routine.html
[SCHED-E] http://www.ospo.noaa.gov/Operations/GOES/east/imager-routine.html
"""

from collections import namedtuple
from datetime import datetime, timedelta
import logging
import re

import dask.array as da
import numpy as np
import xarray as xr
import xarray.ufuncs as xu

import pyresample.geometry
from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.hrit_goes import (SPACECRAFTS, EQUATOR_RADIUS, POLE_RADIUS,
                                     ALTITUDE)
from satpy.readers.utils import bbox, get_geostationary_angle_extent

logger = logging.getLogger(__name__)

# Radiation constants. Source: [VIS]
C1 = 1.191066E-5  # [mW/(m2-sr-cm-4)]
C2 = 1.438833  # [K/cm-1]

# Calibration Coefficients
#
# VIS Channel
# ============
# slope, offset: Pre-Launch slope & offset for converting counts to radiance
#                (one per detector) [W m-2 um-1 sr-1].
# x0: Space count
# k: pi / (solar spectral irradiance averaged over the spectral response
#          function of the detector) [m2 sr um W-1]
#
#
# IR Channels
# ============
# scale, offset: Scale & offset for converting counts to radiance. Units:
#                [mW m-2 cm-1 sr-1], [1]. They are identical for all platforms.
# n: The channel's central wavenumber (one for each detector) [cm-1]
# a, b: Offset and slope for converting effective BT to actual BT (one per
#       detector). Units: [K], [1]
# btmin, btmax: Valid BT range [K]. Values outside this range will be masked.
#               Extracted from lookup tables provided in [IR].
SCALE_03_9 = 227.3889
OFFSET_03_9 = 68.2167
SCALE_06_8 = 38.8383
OFFSET_06_8 = 29.1287
SCALE_06_5 = 38.8383
OFFSET_06_5 = 29.1287
SCALE_10_7 = 5.2285
OFFSET_10_7 = 15.6854
SCALE_12_0 = 5.0273
OFFSET_12_0 = 15.3332
SCALE_13_3 = 5.5297
OFFSET_13_3 = 16.5892
CALIB_COEFS = {
    'GOES-15': {'00_7': {'slope': [5.851966E-1, 5.879772E-1, 5.856793E-1,
                                   5.854250E-1, 5.866992E-1, 5.836241E-1,
                                   5.846555E-1, 5.843753E-1],
                         'offset': [-16.9707, -17.0513, -16.9847, -16.9773,
                                    -17.0143, -16.9251, -16.9550, -16.9469],
                         'x0': 29,
                         'k': 1.88852E-3},
                '03_9': {'scale': SCALE_03_9,
                         'offset': OFFSET_03_9,
                         'n': [2562.7905, 2562.7905],
                         'a': [-1.5693377, -1.5693377],
                         'b': [1.0025034, 1.0025034],
                         'btmin': 205.0,
                         'btmax': 340.0},
                '06_5': {'scale': SCALE_06_8,
                         'offset': OFFSET_06_8,
                         'n': [1521.1988, 1521.5277],
                         'a': [-3.4706545, -3.4755568],
                         'b': [1.0093296, 1.0092838],
                         'btmin': 180.0,
                         'btmax': 340.0},
                '10_7': {'scale': SCALE_10_7,
                         'offset': OFFSET_10_7,
                         'n': [935.89417, 935.78158],
                         'a': [-0.36151367, -0.35316361],
                         'b': [1.0012715, 1.0012570],
                         'btmin': 180.0,
                         'btmax': 340.0},
                '13_3': {'scale': SCALE_13_3,
                         'offset': OFFSET_13_3,
                         'n': [753.72229, 753.93403],
                         'a': [-0.21475817, -0.24630068],
                         'b': [1.0006485, 1.0007178],
                         'btmin': 180.0,
                         'btmax': 340.0}
                },  # ITT RevH + STAR Correction
    'GOES-14': {'00_7': {'slope': [5.874693E-1, 5.865367E-1, 5.862807E-1,
                                   5.864086E-1, 5.857146E-1, 5.852004E-1,
                                   5.860814E-1, 5.841697E-1],
                         'offset': [-17.037, -17.010, -17.002,  -17.006,
                                    -16.986, -16.971, -16.996, -16.941],
                         'x0': 29,
                         'k': 1.88772E-3},
                '03_9': {'scale': SCALE_03_9,
                         'offset': OFFSET_03_9,
                         'n': [2577.3518, 2577.3518],
                         'a': [-1.5297091, -1.5297091],
                         'b': [1.0025608, 1.0025608],
                         'btmin': 205.0,
                         'btmax': 340.0},
                '06_5': {'scale': SCALE_06_8,
                         'offset': OFFSET_06_8,
                         'n': [1519.3488, 1518.5610],
                         'a': [-3.4647892, -3.4390527],
                         'b': [1.0093656, 1.0094427],
                         'btmin': 180.0,
                         'btmax': 340.0},
                '10_7': {'scale': SCALE_10_7,
                         'offset': OFFSET_10_7,
                         'n': [933.98541, 934.19579],
                         'a': [-0.29201763, -0.31824779],
                         'b': [1.0012018, 1.0012303],
                         'btmin': 180.0,
                         'btmax': 340.0},
                '13_3': {'scale': SCALE_13_3,
                         'offset': OFFSET_13_3,
                         'n': [752.88143, 752.82392],
                         'a': [-0.22508805, -0.21700982],
                         'b': [1.0006686, 1.0006503],
                         'btmin': 180.0,
                         'btmax': 340.0}
                },  # ITT RevH + STAR Correction
    'GOES-13': {'00_7': {'slope': [6.120196E-1, 6.118504E-1, 6.096360E-1,
                                   6.087055E-1, 6.132860E-1, 6.118208E-1,
                                   6.122307E-1, 6.066968E-1],
                         'offset': [-17.749, -17.744, -17.769, -17.653,
                                    -17.785, -17.743, -17.755, -17.594],
                         'x0': 29,
                         'k': 1.89544E-3},
                '03_9': {'scale': SCALE_03_9,
                         'offset': OFFSET_03_9,
                         'n': [2561.74, 2561.74],
                         'a': [-1.437204, -1.437204],
                         'b': [1.002562, 1.002562],
                         'btmin': 205.0,
                         'btmax': 340.0},
                '06_5': {'scale': SCALE_06_8,
                         'offset': OFFSET_06_8,
                         'n': [1522.52, 1521.66],
                         'a': [-3.625663, -3.607841],
                         'b': [1.010018, 1.010010],
                         'btmin': 180.0,
                         'btmax': 340.0},
                '10_7': {'scale': SCALE_10_7,
                         'offset': OFFSET_10_7,
                         'n': [937.23, 937.27],
                         'a': [-0.386043, -0.380113],
                         'b': [1.001298, 1.001285],
                         'btmin': 180.0,
                         'btmax': 340.0},
                '13_3': {'scale': SCALE_13_3,
                         'offset': OFFSET_13_3,
                         'n': [749.83],
                         'a': [-0.134801],
                         'b': [1.000482],
                         'btmin': 180.0,
                         'btmax': 340.0}  # Has only one detector on GOES-13
                },
    'GOES-12': {'00_7': {'slope': [5.771030E-1, 5.761764E-1, 5.775825E-1,
                                   5.790699E-1, 5.787051E-1, 5.755969E-1,
                                   5.753973E-1, 5.752099E-1],
                         'offset': [-16.736, -16.709, -16.750, -16.793,
                                    -16.782, -16.692, -16.687, -16.681],
                         'x0': 29,
                         'k': 1.97658E-3},
                '03_9': {'scale': SCALE_03_9,
                         'offset': OFFSET_03_9,
                         'n': [2562.45, 2562.45],
                         'a': [-0.650731, -0.650731],
                         'b': [1.001520, 1.001520],
                         'btmin': 205.0,
                         'btmax': 340.0},
                '06_5': {'scale': SCALE_06_8,
                         'offset': OFFSET_06_8,
                         'n': [1536.43, 1536.94],
                         'a': [-4.764728, -4.775517],
                         'b': [1.012420, 1.012403],
                         'btmin': 180.0,
                         'btmax': 340.0},
                '10_7': {'scale': SCALE_10_7,
                         'offset': OFFSET_10_7,
                         'n': [933.21, 933.21],
                         'a': [-0.360331, -0.360331],
                         'b': [1.001306, 1.001306],
                         'btmin': 180.0,
                         'btmax': 340.0},
                '13_3': {'scale': SCALE_13_3,
                         'offset': OFFSET_13_3,
                         'n': [751.91],
                         'a': [-0.253449],
                         'b': [1.000743],
                         'btmin': 180.0,
                         'btmax': 340.0}  # Has only one detector on GOES-12
                },
    'GOES-11': {'00_7': {'slope': [5.561568E-1, 5.552979E-1, 5.558981E-1,
                                   5.577627E-1, 5.557238E-1, 5.587978E-1,
                                   5.586530E-1, 5.528971E-1],
                         'offset': [-16.129, -16.104, -16.121, -16.175,
                                    -16.116, -16.205, -16.201, -16.034],
                         'x0': 29,
                         'k': 2.01524E-3},
                '03_9': {'scale': SCALE_03_9,
                         'offset': OFFSET_03_9,
                         'n': [2562.07, 2562.07],
                         'a': [-0.644790, -0.644790],
                         'b': [1.000775, 1.000775],
                         'btmin': 205.0,
                         'btmax': 340.0},
                '06_8': {'scale': SCALE_06_8,
                         'offset': OFFSET_06_8,
                         'n': [1481.53],
                         'a': [-0.543401],
                         'b': [1.001495],
                         'btmin': 180.0,
                         'btmax': 340.0},
                '10_7': {'scale': SCALE_10_7,
                         'offset': OFFSET_10_7,
                         'n': [931.76, 931.76],
                         'a': [-0.306809, -0.306809],
                         'b': [1.001274, 1.001274],
                         'btmin': 180.0,
                         'btmax': 340.0},
                '12_0': {'scale': SCALE_12_0,
                         'offset': OFFSET_12_0,
                         'n': [833.67, 833.04],
                         'a': [-0.333216, -0.315110],
                         'b': [1.001000, 1.000967],
                         'btmin': 180.0,
                         'btmax': 340.0}
                },
    'GOES-10': {'00_7': {'slope': [5.605602E-1, 5.563529E-1, 5.566574E-1,
                                   5.582154E-1, 5.583361E-1, 5.571736E-1,
                                   5.563135E-1, 5.613536E-1],
                         'offset': [-16.256, -16.134, -16.143, -16.188,
                                    -16.192, -16.158, -16.133, -16.279],
                         'x0': 29,
                         'k': 1.98808E-3},
                '03_9': {'scale': SCALE_03_9,
                         'offset': OFFSET_03_9,
                         'n': [2552.9845, 2552.9845],
                         'a': [-0.60584483, -0.60584483],
                         'b': [1.0011017, 1.0011017],
                         'btmin': 205.0,
                         'btmax': 340.0},
                '06_8': {'scale': SCALE_06_8,
                         'offset': OFFSET_06_8,
                         'n': [1486.2212],
                         'a': [-0.61653805],
                         'b': [1.0014011],
                         'btmin': 180.0,
                         'btmax': 340.0},
                '10_7': {'scale': SCALE_10_7,
                         'offset': OFFSET_10_7,
                         'n': [936.10260, 935.98981],
                         'a': [-0.27128884, -0.27064036],
                         'b': [1.0009674, 1.0009687],
                         'btmin': 180.0,
                         'btmax': 340.0},
                '12_0': {'scale': SCALE_12_0,
                         'offset': OFFSET_12_0,
                         'n': [830.88473, 830.89691],
                         'a': [-0.26505411, -0.26056452],
                         'b': [1.0009087, 1.0008962],
                         'btmin': 180.0,
                         'btmax': 340.0}
                },
    'GOES-9': {'00_7': {'slope': [0.5492361],
                        'offset': [-15.928],
                        'x0': 29,
                        'k': 1.94180E-3},
               '03_9': {'scale': SCALE_03_9,
                        'offset': OFFSET_03_9,
                        'n': [2555.18, 2555.18],
                        'a': [-0.579908, -0.579908],
                        'b': [1.000942, 1.000942],
                        'btmin': 205.0,
                        'btmax': 340.0},
               '06_8': {'scale': SCALE_06_8,
                        'offset': OFFSET_06_8,
                        'n': [1481.82],
                        'a': [-0.493016],
                        'b': [1.001076],
                        'btmin': 180.0,
                        'btmax': 340.0},
               '10_7': {'scale': SCALE_10_7,
                        'offset': OFFSET_10_7,
                        'n': [934.59, 934.28],
                        'a': [-0.384798, -0.363703],
                        'b': [1.001293, 1.001272],
                        'btmin': 180.0,
                        'btmax': 340.0},
               '12_0': {'scale': SCALE_12_0,
                        'offset': OFFSET_12_0,
                        'n': [834.02, 834.09],
                        'a': [-0.302995, -0.306838],
                        'b': [1.000941, 1.000948],
                        'btmin': 180.0,
                        'btmax': 340.0}
               },
    'GOES-8': {'00_7': {'slope': [0.5501873],
                        'offset': [-15.955],
                        'x0': 29,
                        'k': 1.92979E-3},
               '03_9': {'scale': SCALE_03_9,
                        'offset': OFFSET_03_9,
                        'n': [2556.71, 2558.62],
                        'a': [-0.578526, -0.581853],
                        'b': [1.001512, 1.001532],
                        'btmin': 205.0,
                        'btmax': 340.0},
               '06_8': {'scale': SCALE_06_8,
                        'offset': OFFSET_06_8,
                        'n': [1481.91],
                        'a': [-0.593903],
                        'b': [1.001418],
                        'btmin': 180.0,
                        'btmax': 340.0},
               '10_7': {'scale': SCALE_10_7,
                        'offset': OFFSET_10_7,
                        'n': [934.30, 935.38],
                        'a': [-0.322585, -0.351889],
                        'b': [1.001271, 1.001293],
                        'btmin': 180.0,
                        'btmax': 340.0},
               '12_0': {'scale': SCALE_12_0,
                        'offset': OFFSET_12_0,
                        'n': [837.06, 837.00],
                        'a': [-0.422571, -0.466954],
                        'b': [1.001170, 1.001257],
                        'btmin': 180.0,
                        'btmax': 340.0}
               }
}

# Angular sampling rates in radians. Source: [BOOK-I], [BOOK-N]
SAMPLING_EW_VIS = 16E-6
SAMPLING_NS_VIS = 28E-6
SAMPLING_EW_IR = 64E-6
SAMPLING_NS_IR = 112E-6

# Sector definitions. TODO: Add remaining sectors (PACUS, CONUS, ...)
FULL_DISC = 'Full Disc'
NORTH_HEMIS_EAST = 'Northern Hemisphere (GOES-East)'
SOUTH_HEMIS_EAST = 'Southern Hemisphere (GOES-East)'
NORTH_HEMIS_WEST = 'Northern Hemisphere (GOES-West)'
SOUTH_HEMIS_WEST = 'Southern Hemisphere (GOES-West)'
UNKNOWN_SECTOR = 'Unknown'

IR_SECTORS = {
    (2704, 5208): FULL_DISC,
    (1826, 3464): NORTH_HEMIS_EAST,
    (566, 3464): SOUTH_HEMIS_EAST,
    (1354, 3312): NORTH_HEMIS_WEST,
    (1062, 2760): SOUTH_HEMIS_WEST
}  # (nlines, ncols)

VIS_SECTORS = {
    (10819, 20800): FULL_DISC,
    (7307, 13852): NORTH_HEMIS_EAST,
    (2267, 13852): SOUTH_HEMIS_EAST,
    (5419, 13244): NORTH_HEMIS_WEST,
    (4251, 11044): SOUTH_HEMIS_WEST
}  # (nlines, ncols)

SCAN_DURATION = {
    FULL_DISC: timedelta(minutes=26),
    NORTH_HEMIS_WEST: timedelta(minutes=10, seconds=5),
    SOUTH_HEMIS_WEST: timedelta(minutes=6, seconds=54),
    NORTH_HEMIS_EAST: timedelta(minutes=14, seconds=15),
    SOUTH_HEMIS_EAST: timedelta(minutes=4, seconds=49)
}  # Source: [SCHED-W], [SCHED-E]


class GOESNCFileHandler(BaseFileHandler):
    """File handler for GOES Imager data in netCDF format"""
    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super(GOESNCFileHandler, self).__init__(filename, filename_info,
                                                filetype_info)
        self.nc = xr.open_dataset(filename,
                                  decode_cf=True,
                                  mask_and_scale=False,
                                  chunks={'xc': CHUNK_SIZE, 'yc': CHUNK_SIZE})
        self.sensor = 'goes_imager'
        self.nlines = self.nc.dims['yc']
        self.ncols = self.nc.dims['xc']
        self.platform_name = self._get_platform_name(
            self.nc.attrs['Satellite Sensor'])
        self.platform_shortname = self.platform_name.replace('-', '').lower()
        self.gvar_channel = int(self.nc['bands'].values)
        self.sector = self._get_sector(channel=self.gvar_channel,
                                       nlines=self.nlines,
                                       ncols=self.ncols)
        self._meta = None

    @staticmethod
    def _get_platform_name(ncattr):
        """Determine name of the platform"""
        match = re.match('G-(\d+)', ncattr)
        if match:
            return SPACECRAFTS.get(int(match.groups()[0]))

        return None

    def _get_sector(self, channel, nlines, ncols):
        """Determine which sector was scanned"""
        if self._is_vis(channel):
            margin = 100
            sectors_ref = VIS_SECTORS
        else:
            margin = 50
            sectors_ref = IR_SECTORS

        for (nlines_ref, ncols_ref), sector in sectors_ref.items():
            if np.fabs(ncols - ncols_ref) < margin and \
                    np.fabs(nlines - nlines_ref) < margin:
                return sector

        return UNKNOWN_SECTOR

    @staticmethod
    def _is_vis(channel):
        """Determine whether the given channel is a visible channel"""
        if isinstance(channel, str):
            return channel == '00_7'
        elif isinstance(channel, int):
            return channel == 1
        else:
            raise ValueError('Invalid channel')

    @staticmethod
    def _get_earth_mask(lat):
        """Identify earth/space pixels

        Returns:
            Mask (1=earth, 0=space)
        """
        logger.debug('Computing earth mask')
        return da.fabs(lat.values) <= 90

    @staticmethod
    def _get_lon0(lon, earth_mask, sector):
        """Estimate subsatellite point

        Since the actual subsatellite point is not stored in the netCDF files,
        try to estimate it using the longitude coordinates (full disc images
        only).

        Args:
            lon: Longitudes [degrees east]
            earth_mask: Mask identifying earth and space pixels
            sector: Specifies the scanned sector
        """
        if sector == FULL_DISC:
            logger.debug('Computing subsatellite point')

            # The earth is not centered in the image, compute bounding box
            # of the earth disc first
            rmin, rmax, cmin, cmax = bbox(earth_mask)

            # The subsatellite point is approximately at the centre of the
            # earth disk
            crow = rmin + (rmax - rmin) // 2
            ccol = cmin + (cmax - cmin) // 2
            return lon[crow, ccol]

        return None

    @staticmethod
    def _is_yaw_flip(lat, delta=10):
        """Determine whether the satellite is yaw-flipped ('upside down')"""
        logger.debug('Computing yaw flip flag')
        # In case of yaw-flip the data and coordinates in the netCDF files are
        # also flipped. Just check whether the latitude increases or decrases
        # with the line number.
        crow, ccol = np.array(lat.shape) // 2
        return (lat[crow+delta, ccol] - lat[crow, ccol]).values > 0

    def _get_area_def_uniform_sampling(self, lon0, channel, sector):
        """Get area definition with uniform sampling"""
        logger.debug('Computing area definition')

        if sector == FULL_DISC:
            # Define proj4 projection parameters
            proj_dict = {'a': EQUATOR_RADIUS,
                         'b': POLE_RADIUS,
                         'lon_0': lon0,
                         'h': ALTITUDE,
                         'proj': 'geos',
                         'units': 'm'}

            # Calculate maximum scanning angles
            xmax, ymax = get_geostationary_angle_extent(
                namedtuple('area', ['proj_dict'])(proj_dict))

            # Derive area extent using small angle approximation (maximum
            # scanning angle is ~8.6 degrees)
            llx, lly, urx, ury = ALTITUDE * np.array([-xmax, -ymax, xmax, ymax])
            area_extent = [llx, lly, urx, ury]

            # Original image is oversampled. Create pyresample area definition
            # with uniform sampling in N-S and E-W direction
            if self._is_vis(channel):
                sampling = SAMPLING_NS_VIS
            else:
                sampling = SAMPLING_NS_IR
            pix_size = ALTITUDE * sampling
            area_def = pyresample.geometry.AreaDefinition(
                area_id='goes_geos_uniform',
                name='{} geostationary projection (uniform sampling)'.format(
                    self.platform_name),
                proj_id='goes_geos_uniform',
                proj_dict=proj_dict,
                x_size=np.rint((urx - llx) / pix_size).astype(int),
                y_size=np.rint((ury - lly) / pix_size).astype(int),
                area_extent=area_extent)

            return area_def
        else:
            return None, None

    @property
    def start_time(self):
        """Start timestamp of the dataset"""
        dt = self.nc['time'].dt
        return datetime(year=dt.year, month=dt.month, day=dt.day,
                        hour=dt.hour, minute=dt.minute,
                        second=dt.second, microsecond=dt.microsecond)

    @property
    def end_time(self):
        """End timestamp of the dataset"""
        if self.sector is not None:
            return self.start_time + SCAN_DURATION[self.sector]

        return self.start_time

    def get_shape(self, key, info):
        """Get the shape of the data

        Returns:
            Number of lines, number of columns
        """
        return self.nlines, self.ncols

    @property
    def meta(self):
        """Derive metadata from the coordinates"""
        # Use buffered data if available
        if self._meta is None:
            lat = self.nc['lat']
            earth_mask = self._get_earth_mask(lat)
            yaw_flip = self._is_yaw_flip(lat)
            del lat

            lon = self.nc['lon'].values
            lon0 = self._get_lon0(lon=lon, earth_mask=earth_mask,
                                  sector=self.sector)
            area_def_uni = self._get_area_def_uniform_sampling(
                lon0=lon0, channel=self.gvar_channel, sector=self.sector)
            del lon

            self._meta = {'earth_mask': earth_mask,
                          'yaw_flip': yaw_flip,
                          'lon0': lon0,
                          'area_def_uni': area_def_uni}
        return self._meta

    def get_dataset(self, key, info, out=None, xslice=slice(None),
                    yslice=slice(None)):
        """Load dataset designated by the given key from file"""
        logger.debug('Reading dataset {}'.format(key.name))

        # Read data from file and calibrate if necessary
        if 'longitude' in key.name:
            data = self.nc['lon'][xslice, yslice]
        elif 'latitude' in key.name:
            data = self.nc['lat'][xslice, yslice]
        else:
            tic = datetime.now()
            data = self.calibrate(self.nc['data'].isel(time=0)[xslice, yslice],
                                  calibration=key.calibration,
                                  channel=key.name)
            logger.debug('Calibration time: {}'.format(datetime.now() - tic))

        # Mask space pixels
        data = data.where(self.meta['earth_mask'])

        # Set proper dimension names
        data = data.rename({'xc': 'x', 'yc': 'y'})

        # Update metadata
        data.attrs.update(info)
        data.attrs.update(
            {'platform_name': self.platform_name,
             'sensor': self.sensor,
             'sector': self.sector,
             'lon0': self.meta['lon0'],
             'area_def_uniform_sampling': self.meta['area_def_uni'],
             'yaw_flip': self.meta['yaw_flip']}
        )

        if out is None:
            return data
        else:
            out.data = data.data
            out.attrs.update(data.attrs)

    def calibrate(self, counts, calibration, channel):
        """Perform calibration"""
        # Convert 16bit counts from netCDF4 file to the original 10bit
        # GVAR counts by dividing by 32. See [FAQ].
        counts = counts / 32.

        coefs = CALIB_COEFS[self.platform_name][channel]
        if calibration == 'counts':
            return counts
        elif calibration in ['radiance', 'reflectance',
                             'brightness_temperature']:
            radiance = self._counts2radiance(counts=counts, coefs=coefs,
                                             channel=channel)
            if calibration == 'radiance':
                return radiance

            return self._calibrate(radiance=radiance, coefs=coefs,
                                   channel=channel, calibration=calibration)
        else:
            raise ValueError('Unsupported calibration for channel {}: {}'
                             .format(channel, calibration))

    def _counts2radiance(self, counts, coefs, channel):
        """Convert raw detector counts to radiance"""
        logger.debug('Converting counts to radiance')

        if self._is_vis(channel):
            # Since the scanline-detector assignment is unknown, use the average
            # coefficients for all scanlines.
            slope = np.array(coefs['slope']).mean()
            offset = np.array(coefs['offset']).mean()
            return self._viscounts2radiance(counts=counts, slope=slope,
                                            offset=offset)

        return self._ircounts2radiance(counts=counts, scale=coefs['scale'],
                                       offset=coefs['offset'])

    def _calibrate(self, radiance, coefs, channel, calibration):
        """Convert radiance to reflectance or brightness temperature"""
        if self._is_vis(channel):
            if not calibration == 'reflectance':
                raise ValueError('Cannot calibrate VIS channel to '
                                 '{}'.format(calibration))
            return self._calibrate_vis(radiance=radiance, k=coefs['k'])
        else:
            if not calibration == 'brightness_temperature':
                raise ValueError('Cannot calibrate IR channel to '
                                 '{}'.format(calibration))

            # Since the scanline-detector assignment is unknown, use the average
            # coefficients for all scanlines.
            mean_coefs = {'a': np.array(coefs['a']).mean(),
                          'b': np.array(coefs['b']).mean(),
                          'n': np.array(coefs['n']).mean(),
                          'btmin': coefs['btmin'],
                          'btmax': coefs['btmax']}
            return self._calibrate_ir(radiance=radiance, coefs=mean_coefs)

    @staticmethod
    def _ircounts2radiance(counts, scale, offset):
        """Convert IR counts to radiance

        Reference: [IR].

        Args:
            counts: Raw detector counts
            scale: Scale [mW-1 m2 cm sr]
            offset: Offset [1]

        Returns:
            Radiance [mW m-2 cm-1 sr-1]
        """
        rad = (counts - offset) / scale
        return rad.clip(min=0)

    @staticmethod
    def _calibrate_ir(radiance, coefs):
        """Convert IR radiance to brightness temperature

        Reference: [IR]

        Args:
            radiance: Radiance [mW m-2 cm-1 sr-1]
            coefs: Dictionary of calibration coefficients. Keys:
                   n: The channel's central wavenumber [cm-1]
                   a: Offset [K]
                   b: Slope [1]
                   btmin: Minimum brightness temperature threshold [K]
                   btmax: Maximum brightness temperature threshold [K]

        Returns:
            Brightness temperature [K]
        """
        logger.debug('Calibrating to brightness temperature')

        # Compute brightness temperature using inverse Planck formula
        n = coefs['n']
        bteff = C2 * n / xu.log(1 + C1 * n**3 / radiance.where(radiance > 0))
        bt = xr.DataArray(bteff * coefs['b'] + coefs['a'])

        # Apply BT threshold
        return bt.where(xu.logical_and(bt >= coefs['btmin'],
                                       bt <= coefs['btmax']))

    @staticmethod
    def _viscounts2radiance(counts, slope, offset):
        """Convert VIS counts to radiance

        References: [VIS]

        Args:
            counts: Raw detector counts
            slope: Slope [W m-2 um-1 sr-1]
            offset: Offset [W m-2 um-1 sr-1]
        Returns:
            Radiance [W m-2 um-1 sr-1]
        """
        rad = counts * slope + offset
        return rad.clip(min=0)

    @staticmethod
    def _calibrate_vis(radiance, k):
        """Convert VIS radiance to reflectance

        Note: Angle of incident radiation and annual variation of the
        earth-sun distance is not taken into account. A value of 100%
        corresponds to the radiance of a perfectly reflecting diffuse surface
        illuminated at normal incidence when the sun is at its annual-average
        distance from the Earth.

        TODO: Take angle of incident radiation (cos sza) and annual variation
        of the earth-sun distance into account.

        Reference: [VIS]

        Args:
            radiance: Radiance [mW m-2 cm-1 sr-1]
            k: pi / H, where H is the solar spectral irradiance at
               annual-average sun-earth distance, averaged over the spectral
               response function of the detector). Units of k: [m2 um sr W-1]
        Returns:
            Reflectance [%]
        """
        logger.debug('Calibrating to reflectance')
        refl = 100 * k * radiance
        return refl.clip(min=0)

    def __del__(self):
        try:
            self.nc.close()
        except (AttributeError, IOError, OSError):
            pass
