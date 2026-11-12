"""Demo FCI data download."""

import pathlib
import tarfile
import tempfile

from satpy import config

from . import utils

_fci_uncompressed_nominal = (
    "https://sftp.eumetsat.int/public/folder/UsCVknVOOkSyCdgpMimJNQ/"
    "User-Materials/Test-Data/MTG/MTG_FCI_L1C_Enhanced-NonN_TD-272_May2020/"
    "FCI_1C_UNCOMPRESSED_NOMINAL.tar.gz")


def download_fci_test_data(base_dir=None):
    """Download FCI test data.

    Download the nominal FCI test data from July 2020.
    """
    subdir = get_fci_test_data_dir(base_dir=base_dir)
    with tempfile.TemporaryDirectory() as td:
        nm = pathlib.Path(td) / "fci-test-data.tar.gz"
        utils.download_url(_fci_uncompressed_nominal, nm)
        return _unpack_tarfile_to(nm, subdir)


def get_fci_test_data_dir(base_dir=None):
    """Get directory for FCI test data."""
    base_dir = base_dir or config.get("demo_data_dir", ".")
    return pathlib.Path(base_dir) / "fci" / "test_data"


def _unpack_tarfile_to(filename, subdir):
    """Unpack content of tarfile in filename to subdir."""
    with tarfile.open(filename, mode="r:gz") as tf:
        contents = tf.getnames()
        tf.extractall(path=subdir, filter="data")
    return contents
