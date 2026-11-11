"""Demo data download for METimage L1B NetCDF files."""

import logging
import os
import tarfile
import tempfile

from satpy import config

from . import utils

logger = logging.getLogger(__name__)

ZENODO_TARBALL_URL = "https://zenodo.org/api/records/24093210/files/metimage_l1b_example.tar.gz/content"
FILENAMES_20261105_0926 = [
    "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-VII-1B-RAD_C_EUMT_20261105101551_G_O_20261105092558_20261105092659_C_N_T__.nc",
    "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-VII-1B-RAD_C_EUMT_20261105101547_G_O_20261105092659_20261105092759_C_N_T__.nc",
    "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-VII-1B-RAD_C_EUMT_20261105101740_G_O_20261105092759_20261105092900_C_N_T__.nc",
    "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-VII-1B-RAD_C_EUMT_20261105101811_G_O_20261105092900_20261105092958_C_N_T__.nc",
    "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-VII-1B-RAD_C_EUMT_20261105101923_G_O_20261105092958_20261105093059_C_N_T__.nc",
    "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-VII-1B-RAD_C_EUMT_20261105101921_G_O_20261105093059_20261105093159_C_N_T__.nc",
    "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-VII-1B-RAD_C_EUMT_20261105102142_G_O_20261105093159_20261105093258_C_N_T__.nc",
    "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-VII-1B-RAD_C_EUMT_20261105102241_G_O_20261105093258_20261105093359_C_N_T__.nc",
    "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-VII-1B-RAD_C_EUMT_20261105102240_G_O_20261105093359_20261105093459_C_N_T__.nc",
    "W_XX-EUMETSAT-Darmstadt,SAT,SGA1-VII-1B-RAD_C_EUMT_20261105102315_G_O_20261105093459_20261105093600_C_N_T__.nc",
]


def download_metimage_l1b_20261105_0926(base_dir=None):
    """Download Metop-SGA1 METimage L1B files for 2026-11-05 09:26 to 09:36 UTC.

    The ten uncompressed L1B radiance granules pass from Scandinavia over Europe
    to Northern Africa. They are pre-operational data from the EUMETSAT
    processing chain, intended only for testing and not for operational use.
    This dataset is published on Zenodo as a single gzipped tar file of about
    720MB, which unpacks to about 1.2GB: https://zenodo.org/records/24093210

    The tar file is downloaded to a temporary directory next to the extracted
    files and removed after extraction. If every granule already exists
    locally, nothing is downloaded.

    Args:
        base_dir: Directory to put the data in. A ``metimage_l1b/20261105_0926``
            subdirectory is created inside it. Defaults to the ``demo_data_dir``
            Satpy configuration value.

    Returns:
        List of paths to the ten granule files, sorted by observation start
        time.

    """
    base_dir = base_dir or config.get("demo_data_dir", ".")
    subdir = os.path.join(base_dir, "metimage_l1b", "20261105_0926")
    files = [os.path.join(subdir, fn) for fn in FILENAMES_20261105_0926]
    if all(os.path.isfile(fn) for fn in files):
        logger.info(f"METimage L1B demo files already exist in {subdir}, skipping download...")
        return files

    os.makedirs(subdir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=subdir) as td:
        tar_filename = os.path.join(td, "metimage_l1b_example.tar.gz")
        logger.info(f"Downloading METimage L1B demo data to {tar_filename}...")
        utils.download_url(ZENODO_TARBALL_URL, tar_filename)
        _unpack_tarfile_to(tar_filename, subdir)

    missing = [fn for fn in files if not os.path.isfile(fn)]
    if missing:
        raise RuntimeError(f"Downloaded METimage L1B demo data is missing expected files: {missing}")
    return files


def _unpack_tarfile_to(filename, subdir):
    """Unpack the tar file to subdir, rejecting members that are unsafe to extract."""
    with tarfile.open(filename, mode="r:gz") as tf:
        tf.extractall(path=subdir, filter="data")
