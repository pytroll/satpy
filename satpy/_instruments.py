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

def get_instruments_from_attrs(attrs: dict[str,Any], to_internal: bool=False) -> set[str]:
    """Get instrument names from dataset attributes.

    String type attributes are converted to set. This can be
    removed once all file handlers provide instruments as a
    set.
    """
    instruments = attrs.get("instruments", set())
    # 8< v1.0
    sensor = attrs.get("sensor", set())
    if sensor:
        warnings.warn(
            "Satpy will ignore the 'sensor' attribute as of v1.0. "
            "Use the 'instruments' attribute instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if not instruments:
            instruments = sensor
    if isinstance(instruments, str):
        warnings.warn(
            "Converting 'instruments' attribute from string to set. "
            "This will result in an error in v1.0, when Satpy will require "
            "set type instruments attributes.",
            DeprecationWarning,
            stacklevel=2
        )
        instruments = set([instruments])
    # >8 v1.0
    if to_internal:
        return {
            wmo_to_internal(inst) for inst in instruments
        }
    return instruments



def wmo_to_internal(instrument: str) -> str:
    """Convert WMO to internal instrument name."""
    sep_map = {
        "-": "-",
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
    return wmo_to_internal(instrument)


def join_instrument_names(instruments: set[str]) -> str:
    """Join a set of instrument names."""
    return "-".join(sorted(instruments))


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
    AGRI = "AGRI"
    AMI = "AMI"
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
    GHI = "GHI"
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
    MSI = "MSI"
    MSI_SENTINEL_2A = "MSI (Sentinel-2A)"
    MSS = "MSS"
    MSU_GS = "MSU-GS"
    MSU_GS_A = "MSU-GS/A"
    MVIRI = "MVIRI"
    MWR_AWS = "MWR (AWS)"
    MWR_STERNA = "MWR (Sterna)"
    OCI = "OCI"
    OLCI = "OLCI"
    OLI = "OLI"
    SAR_C_SENTINEL_1 = "SAR-C (Sentinel-1)"
    SEAWIFS = "SeaWiFS"
    SEVIRI = "SEVIRI"
    SGLI = "SGLI"
    SLSTR = "SLSTR"
    SSMIS = "SSMIS"
    TIRS = "TIRS"
    TM = "TM"
    TROPOMI = "TROPOMI"
    VIIRS = "VIIRS"
    VIRR_FY_3 = "VIRR (FY-3)"
    VISSR = "VISSR"
    VISSR_HIMAWARI_5 = "VISSR (Himawari-5)"


def enum_to_str(instruments: set[StrEnum]) -> set[str]:
    """Convert OSCAR enums to string."""
    return {str(i) for i in instruments}


_INTERNAL_TO_WMO = {
    wmo_to_internal(inst): str(inst)
    for inst in OSCAR
}

def internal_to_wmo(instrument: str) -> str:
    """Convert internal to WMO instrument name."""
    return _INTERNAL_TO_WMO.get(instrument, instrument)



# 8< v1.0
from pathlib import Path  # noqa

import satpy._instruments as inst_utils  # noqa
from satpy._config import PACKAGE_CONFIG_PATH  # noqa
from satpy._instruments import OSCAR  # noqa

RENAMED_ENH_INSTRUMENTS = {
    OSCAR.MSI_SENTINEL_2A: "sen2_msi",
    OSCAR.MWR_AWS: "mwr",
    OSCAR.MWR_STERNA: "mwr",
}
RENAMED_COMP_INSTRUMENTS = {
    OSCAR.IMAGER_GOES_8_11: "goes_imager",
    OSCAR.IMAGER_GOES_12_15: "goes_imager",
    OSCAR.IMAGER_INSAT: "insat3d_img",
    OSCAR.METIMAGE: "vii",
    OSCAR.MSI: "ec_msi",
    OSCAR.MSU_GS_A: "msu-gsa",
    OSCAR.MWR_AWS: "mwr",
    OSCAR.MWR_STERNA: "mwr",
    OSCAR.OLI: "oli_tirs",
    OSCAR.TIRS: "oli_tirs",
    OSCAR.VIRR_FY_3: "virr",
    OSCAR.SAR_C_SENTINEL_1: "sar-c",
    OSCAR.MSI_SENTINEL_2A: "sen2_msi",
}

def get_deprecated_instrument_aliases_for_enhancements(instruments: set[str]) -> set[str]:
    """Get deprecated instrument aliases for enhancements that were renamed."""
    return _get_deprecated_instrument_aliases(instruments, RENAMED_ENH_INSTRUMENTS)  # type: ignore


def get_deprecated_instrument_aliases_for_composites(instruments: set[str]) -> set[str]:
    """Get deprecated instrument aliases for composites that were renamed."""
    return _get_deprecated_instrument_aliases(instruments, RENAMED_COMP_INSTRUMENTS)  # type: ignore


def _get_deprecated_instrument_aliases(instruments: set[str], renamed_instruments: dict[str,str]) -> set[str]:
    old_names = [
        old_name
        for new_name in instruments
        if (old_name := renamed_instruments.get(new_name))
    ]
    return set(old_names)


def warn_if_deprecated_instrument_in_enhancement_filename(instrument: str, config_file: str) -> None:
    """Warn if enhancement filename contains a deprecated instrument name."""
    _warn_if_deprecated_instrument_in_filename(
        instrument,
        config_file,
        RENAMED_ENH_INSTRUMENTS,  # type: ignore
    )


def warn_if_deprecated_instrument_in_composite_filename(instrument: str, config_file: str) -> None:
    """Warn if composite filename contains a deprecated instrument name."""
    _warn_if_deprecated_instrument_in_filename(
        instrument,
        config_file,
        RENAMED_COMP_INSTRUMENTS,  # type: ignore
    )


def _warn_if_deprecated_instrument_in_filename(
        instrument: str, config_file: str, renamed_instruments: dict[str,str],
    ) -> None:
    is_old = instrument in renamed_instruments.values()
    is_user_file = Path(config_file).parent.parent != Path(PACKAGE_CONFIG_PATH)
    if is_old and is_user_file:
        new_names = [
            new_name
            for new_name, old_name in renamed_instruments.items()
            if old_name == instrument
        ]
        new_files = [
            inst_utils.wmo_to_internal(new_name) + ".yaml"
            for new_name in new_names
        ]
        msg = (f"Instrument '{instrument}' has been renamed to the official "
               f"WMO name '{'/'.join(new_names)}'. Your config file "
               f"{config_file} still uses the old instrument name. Rename "
               f"the file to '{'/'.join(new_files)}', otherwise it will be "
               f"ignored in Satpy v1.0.")
        warnings.warn(msg, DeprecationWarning, stacklevel=3)
