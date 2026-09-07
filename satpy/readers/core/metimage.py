
"""Utilities for the management of METimage (VII) products."""

PLATFORM_NAME_TRANSLATE = {
    "SGA1": "Metop-SG-A1",
    "SGA2": "Metop-SG-A2",
    "SGA3": "Metop-SG-A3"
}

# PLANCK COEFFICIENTS FOR CALIBRATION AS DEFINED BY EUMETSAT
C1 = 1.191062e+8   # [W/m2·sr-1·µm4]
C2 = 1.4387863e+4  # [K·µm]

# CONSTANTS DEFINING THE TIE POINTS
TIE_POINTS_FACTOR = 8    # Sub-sampling factor of tie points wrt pixel points
SCAN_ALT_TIE_POINTS = 4  # Number of tie points along the satellite track for each scan

# Number of pixel rows per instrument scan. The tie points of a scan bound
# ``SCAN_ALT_TIE_POINTS - 1`` intervals of ``TIE_POINTS_FACTOR`` pixels each.
ROWS_PER_SCAN = (SCAN_ALT_TIE_POINTS - 1) * TIE_POINTS_FACTOR

# MEAN EARTH RADIUS AS DEFINED BY IUGG
MEAN_EARTH_RADIUS = 6371008.7714  # [m]
