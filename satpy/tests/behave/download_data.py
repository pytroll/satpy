#!/usr/bin/env python
"""Download test data and ancillary data for running this tutorial."""

import os
import math
import requests
from zipfile import ZipFile
from tqdm import tqdm

TUTORIAL_ROOT = os.path.dirname(os.path.abspath(__file__))


def download_pyspectral_luts():
    print("Downloading lookup tables used by pyspectral...")
    from pyspectral.utils import download_luts, download_rsr
    download_luts()
    download_rsr()
    return True


def _download_data_zip(url, output_filename):
    if os.path.isfile(output_filename):
        print("Data zip file already exists, won't re-download: {}".format(output_filename))
        return True

    print("Downloading {}".format(url))
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0
    with open(output_filename, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size), unit='KB', unit_scale=True):
            wrote += len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR: something went wrong downloading {}".format(url))
        return False
    return True


def _unzip(filename, output_dir):
    print("Extracting {}".format(filename))
    try:
        with ZipFile(filename, 'r') as zip_obj:
            zip_obj.extractall(output_dir)
    except (IOError, OSError):
        print("FAIL: Could not extract {}".format(filename))
        return False
    return True


def _download_and_unzip(url, output_dir):
    filename = os.path.basename(url)
    if _download_data_zip(url, filename):
        return _unzip(filename, output_dir)
    return False


def download_test_data():
    cwd = os.getcwd()
    os.chdir(TUTORIAL_ROOT)

    ret = _download_and_unzip(
        'https://bin.ssec.wisc.edu/pub/davidh/20180511_texas_fire_abi_l1b_conus.zip',
        os.path.join('data', 'abi_l1b')
    )
    ret &= _download_and_unzip(
        'https://bin.ssec.wisc.edu/pub/davidh/20180511_texas_fire_abi_l1b_meso.zip',
        os.path.join('data', 'abi_l1b')
    )
    ret &= _download_and_unzip(
        'https://bin.ssec.wisc.edu/pub/davidh/20180511_texas_fire_viirs_sdr.zip',
        os.path.join('data', 'viirs_sdr')
    )
    os.chdir(cwd)
    return ret


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download data necessary for the Satpy tutorial")
    parser.add_argument('--luts-only', action='store_true',
                        help="Only download LUTs for pyspectral operation")
    parser.add_argument('--data-only', action='store_true',
                        help="Only download test data")
    args = parser.parse_args()

    ret = True
    if not args.data_only:
        ret &= download_pyspectral_luts()
    if not args.luts_only:
        ret &= download_test_data()

    if ret:
        print("Downloaded `.zip` files can now be deleted.")
        print("SUCCESS")
    else:
        print("FAIL")
    return int(not ret)


if __name__ == "__main__":
    import sys
    sys.exit(main())