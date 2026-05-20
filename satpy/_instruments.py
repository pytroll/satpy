# Copyright (c) 2026 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Helpers for accessing and modifying instrument attributes."""

import logging
import warnings
from enum import StrEnum
from typing import Any

import satpy

logger = logging.getLogger(__name__)

def get_instruments_from_attrs(attrs: dict[str,Any], normalize: bool=False) -> set[str]:
    """Get instrument names from dataset attributes.

    String type attributes are converted to set. This can be
    removed once all file handlers provide instruments as a
    set.
    """
    legacy = attrs.get("sensor", set())
    instruments = attrs.get("instruments", legacy)
    if legacy:
        warnings.warn(
            "Satpy will ignore the 'sensor' attribute as of v1.1. "
            "Use the 'instruments' attribute instead.",
            DeprecationWarning,
            stacklevel=2
        )
    if isinstance(instruments, str):
        warnings.warn(
            "Converting 'instruments' attribute from string to set. "
            "This will result in an error in v1.1, when Satpy will require "
            "set type instruments attributes.",
            DeprecationWarning,
            stacklevel=2
        )
        instruments = set([instruments])
    if normalize:
        return {normalize_instrument_name(instrument) for instrument in instruments}
    return instruments



def normalize_instrument_name(instrument: str) -> str:
    """Normalize instrument name for internal usage."""
    sep_map = {
        "-": "",
        "(": "",
        ")": "",
        " ": "_",
        "/": "-"
    }
    sep_trans = str.maketrans(sep_map)
    return instrument.translate(sep_trans).lower()


def get_one_instrument_from_attrs(attrs: dict[str,Any]) -> str:
    """Get a single instrument name from dataset attributes."""
    instruments = get_instruments_from_attrs(attrs)
    if not instruments:
        raise KeyError("No 'instruments' in dataset attribute")
    if len(instruments) > 1:
        logger.warning(f"More than one instrument in dataset attributes, will use the first value: {instruments}")
    return list(instruments)[0]


def get_pyspectral_instrument_name(instrument: str) -> str:
    """Get instrument name expected by pyspectral."""
    return normalize_instrument_name(instrument)


def serialize_instruments(instruments: set[str]) -> str:
    """Serialize a set of instruments."""
    sep_map = {
        "-": "",
        "(": "",
        ")": "",
        " ": "",
        "/": ""
    }
    sep_trans = str.maketrans(sep_map)
    return "-".join(
        instr.translate(sep_trans).lower()
        for instr in sorted(instruments)
    )


def set_instruments_attr(attrs: dict[str,Any], instruments: set[str]|str) -> None:
    """Set 'instruments' dataset atrribute."""
    key = get_instruments_key()
    attrs[key] = instruments


def get_instruments_key():
    """Get key for instruments in dataset attributes."""
    return satpy.config.get("instruments_key")



class OSCAR(StrEnum):
    """WMO OSCAR instrument names."""
    ABI = "ABI"
    AHI = "AHI"
    AMSR_2 = "AMSR2"
    AMSU_A = "AMSU-A"
    AMSU_B = "AMSU-B"
    ATMS = "ATMS"
    AVHRR = "AVHRR"
    AVHRR_2 = "AVHRR/2"
    AVHRR_3 = "AVHRR/3"
    CRIS = "CrIS"
    EPIC = "EPIC"
    ETM_PLUS = "ETM+"
    FCI = "FCI"
    GLM = "GLM"
    GMI = "GMI"
    IASI = "IASI"
    IASI_NG = "IASI-NG"
    IMAGER_GOES_12_15 = "IMAGER (GOES 12-15)"
    IMAGER_GOES_8_11 = "IMAGER (GOES 8-11)"
    IMAGER_INSAT = "IMAGER (INSAT)"
    IMAGER_MTSAT_2 = "IMAGER (MTSAT-2)"
    JAMI = "JAMI"
    LI = "LI"
    MERIS = "MERIS"
    MERSI_1 = "MERSI-1"
    MERSI_2 = "MERSI-2"
    MERSI_3 = "MERSI-3"
    MERSI_LL = "MERSI-LL"
    MERSI_RM = "MERSI-RM"
    METIMAGE = "METimage"
    MHS = "MHS"
    MODIS = "MODIS"
    MSS = "MSS"
    MSU_GS = "MSU-GS"
    MSU_GS_A = "MSU-GS/A"
    MVIRI = "MVIRI"
    # OSCAR lists "MWR (Sterna)", "MWR (AWS)" etc.
    # But to avoid enhancement/composite duplication
    # we just use "MWR".
    MWR = "MWR"
    OCI = "OCI"
    OLCI = "OLCI"
    OLI = "OLI"
    SEAWIFS = "SeaWiFS"
    SEVIRI = "SEVIRI"
    SGLI = "SGLI"
    SLSTR = "SLSTR"
    SSMIS = "SSMIS"
    TIRS = "TIRS"
    TM = "TM"
    VIIRS = "VIIRS"
    VISSR = "VISSR"
    VISSR_HIMAWARI_5 = "VISSR (Himawari-5)"


def enum_to_str(instruments: set[StrEnum]) -> set[str]:
    """Convert OSCAR enums to string."""
    return {str(i) for i in instruments}


NORMALIZED_TO_WMO: dict[str, str] = {
    normalize_instrument_name(instrument): str(instrument)
    for instrument in OSCAR
}
