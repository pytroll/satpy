#!/usr/bin/env python
"""Test that the installation steps for this tutorial were successful.

1. Check that Satpy features are available and all dependencies are importable.
2. Check that data has been downloaded.

"""

import io
import os
from contextlib import redirect_stdout

try:
    from satpy.utils import check_satpy
except ImportError:
    print("FAIL: Satpy is not importable")
    raise


TUTORIAL_ROOT = os.path.dirname(os.path.abspath(__file__))


def check_satpy_features():
    print("Checking Satpy features...\n")
    readers = ['abi_l1b', 'viirs_sdr']
    writers = ['cf', 'geotiff', 'simple_image']
    extras = ['cartopy', 'geoviews']
    out = io.StringIO()
    with redirect_stdout(out):
        check_satpy(readers=readers, writers=writers, extras=extras)
    out_str = out.getvalue()
    print(out_str)

    for feature in readers + writers + extras:
        if feature + ":  ok" not in out_str:
            print("FAIL: Missing or corrupt Satpy dependency (see above for details).")
            return False
    return True


def check_data_download():
    print("Checking data directories...\n")

    # base_dirs
    abi_dir = os.path.join(TUTORIAL_ROOT, 'data', 'abi_l1b')
    viirs_dir = os.path.join(TUTORIAL_ROOT, 'data', 'viirs_sdr')

    # data case dirs
    conus_dir = os.path.join(abi_dir, '20180511_texas_fire_abi_l1b_conus')
    meso_dir = os.path.join(abi_dir, '20180511_texas_fire_abi_l1b_meso')
    viirs_dir = os.path.join(viirs_dir, '20180511_texas_fire_viirs_sdr')
    if not os.path.isdir(conus_dir):
        print("FAIL: Missing ABI L1B CONUS data: {}".format(conus_dir))
        return False
    if not os.path.isdir(meso_dir):
        print("FAIL: Missing ABI L1B Mesoscale data: {}".format(meso_dir))
        return False
    if not os.path.isdir(viirs_dir):
        print("FAIL: Missing VIIRS SDR data: {}".format(viirs_dir))
        return False

    # number of files
    if len(os.listdir(conus_dir)) != 16:
        print("FAIL: Expected 16 files in {}".format(conus_dir))
        return False
    if len(os.listdir(meso_dir)) != 1440:
        print("FAIL: Expected 1440 files in {}".format(meso_dir))
        return False
    if len(os.listdir(viirs_dir)) != 21:
        print("FAIL: Expected 21 files in {}".format(viirs_dir))
        return False

    return True


def main():
    ret = True
    ret &= check_satpy_features()
    ret &= check_data_download()
    if ret:
        print("SUCCESS")
    else:
        print("FAIL")
    return ret


if __name__ == "__main__":
    import sys
    sys.exit(main())