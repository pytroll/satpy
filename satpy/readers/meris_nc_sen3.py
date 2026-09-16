"""ENVISAT MERIS reader.

Sentinel 3 like format: https://earth.esa.int/eogateway/documents/20142/37627/MERIS-Sentinel-3-Like-L1-andL2-PFS.pdf

Default:
    scn = Scene(filenames=my_files, reader='meris_nc_sen3')

References:
    - :func:`xarray.open_dataset`
"""

import logging
from functools import reduce

import numpy as np

from satpy.readers.olci_nc import NCOLCI2, BitFlags, NCOLCIAngles, NCOLCIBase, NCOLCIMeteo

logger = logging.getLogger(__name__)


class NCMERISCal(NCOLCIBase):
    """Dummy class for calibration."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init the meris reader base."""
        super(NCMERISCal, self).__init__(filename, filename_info, filetype_info)
        self.sensor = "meris"


class NCMERISGeo(NCOLCIBase):
    """Dummy class for navigation."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init the meris reader base."""
        super(NCMERISGeo, self).__init__(filename, filename_info, filetype_info)
        self.sensor = "meris"


class NCMERIS2(NCOLCI2):
    """File handler for MERIS l2."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init the file handler."""
        super(NCMERIS2, self).__init__(filename, filename_info, filetype_info)
        self.sensor = "meris"
        self.reflectance_prefix = "M"
        self.reflectance_suffix = "_rho_w"

    def getbitmask(self, wqsf, items=None):
        """Get the bitmask. Experimental default mask."""
        items = items or ["SEA_ICE", "MEGLINT", "HIGHGLINT",
                          "HAZE_OVER_WATER", "WHITECAPS", "AC_FAIL", "WHITE_SCATT",
                          "LOWRW", "HIGHRW", "OUT_OF_RANGE_AAC", "OUT_OF_SCOPE_AAC",
                          "OUT_OF_RANGE_OC_NN", "OUT_OF_SCOPE_OC_NN",
                          "OUT_OF_RANGE_CHL_OC4ME_INPUT", "OUT_OF_RANGE_CHL_OC4ME"]
        bflags = BitFlags(
            wqsf,
            flag_list=["SEA_ICE", "MEGLINT", "HIGHGLINT", "CASE2_S", "CASE2_ANOM",
                       "HAZE_OVER_WATER", "WHITECAPS", "AC_FAIL", "BPAC_ON", "WHITE_SCATT",
                       "LOWRW", "HIGHRW", "OUT_OF_RANGE_AAC", "OUT_OF_SCOPE_AAC",
                       "OUT_OF_RANGE_OC_NN", "OUT_OF_SCOPE_OC_NN",
                       "OUT_OF_RANGE_CHL_OC4ME_INPUT", "OUT_OF_RANGE_CHL_OC4ME"],
        )
        return reduce(np.logical_or, [bflags[item] for item in items])


class NCMERISAngles(NCOLCIAngles):
    """File handler for the MERIS angles."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init the file handler."""
        super(NCMERISAngles, self).__init__(filename, filename_info, filetype_info)
        self.sensor = "meris"


class NCMERISMeteo(NCOLCIMeteo):
    """File handler for the MERIS meteo data."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init the file handler."""
        super(NCMERISMeteo, self).__init__(filename, filename_info, filetype_info)
        self.sensor = "meris"
