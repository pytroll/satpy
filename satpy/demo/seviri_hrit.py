"""Demo data download for SEVIRI HRIT files."""

import logging
import os.path

from satpy import config
from satpy.demo.utils import download_url

logger = logging.getLogger(__name__)

ZENODO_BASE_URL = "https://zenodo.org/api/files/dcc5ab29-d8a3-4fb5-ab2b-adc405d18c23/"
FILENAME = "H-000-MSG4__-MSG4________-{channel:_<9s}-{segment:_<9s}-201802281500-__"


def download_seviri_hrit_20180228_1500(base_dir=None, subset=None):
    """Download the SEVIRI HRIT files for 2018-02-28T15:00.

    *subset* is a dictionary with the channels as keys and granules to download
    as values, eg::

      {"HRV": [1, 2, 3], "IR_108": [1, 2], "EPI": None}

    """
    files = generate_subset_of_filenames(subset)

    base_dir = base_dir or config.get("demo_data_dir", ".")
    subdir = os.path.join(base_dir, "seviri_hrit", "20180228_1500")
    os.makedirs(subdir, exist_ok=True)
    targets = []
    for the_file in files:
        target = os.path.join(subdir, the_file)
        targets.append(target)
        if os.path.isfile(target):
            continue
        download_url(ZENODO_BASE_URL + the_file, target)
    return targets


def generate_subset_of_filenames(subset=None, base_dir=""):
    """Generate SEVIRI HRIT filenames."""
    if subset is None:
        subset = _create_full_set()
    pattern = os.path.join(base_dir, FILENAME)
    files = []
    for channel, segments in subset.items():
        new_files = _generate_filenames(pattern, channel, segments)
        files.extend(new_files)
    return files


def _generate_filenames(pattern, channel, segments):
    """Generate the filenames for *channel* and *segments*."""
    if channel in ["PRO", "EPI"]:
        new_files = [pattern.format(channel="", segment=channel)]
    else:
        new_files = (pattern.format(channel=channel, segment=f"{segment:06d}") for segment in segments)
    return new_files


def _create_full_set():
    """Create the full set dictionary."""
    subset = {"HRV": range(1, 25),
              "EPI": None,
              "PRO": None}
    channels = ["IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134",
                "VIS006", "VIS008",
                "WV_062", "WV_073"]
    for channel in channels:
        subset[channel] = range(1, 9)
    return subset
