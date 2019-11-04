#!/usr/bin/env python
# Copyright (c) 2019 Satpy developers
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
"""Unittests for NWC SAF MSG (2013) reader."""

import unittest
import numpy as np
import tempfile
import os
import h5py
from collections import OrderedDict

try:
    from unittest import mock
except ImportError:
    import mock  # noqa


fake_ct = {
    "01-PALETTE": {
        "attrs": {
            "CLASS": b"PALETTE",
            "PAL_COLORMODEL": b"RGB",
            "PAL_TYPE": b"DIRECTINDEX",
        },
        "value": np.array(
            [
                [100, 100, 100],
                [0, 120, 0],
                [0, 0, 0],
                [250, 190, 250],
                [220, 160, 220],
                [255, 150, 0],
                [255, 100, 0],
                [255, 220, 0],
                [255, 180, 0],
                [255, 255, 140],
                [240, 240, 0],
                [250, 240, 200],
                [215, 215, 150],
                [255, 255, 255],
                [230, 230, 230],
                [0, 80, 215],
                [0, 180, 230],
                [0, 240, 240],
                [90, 200, 160],
                [200, 0, 200],
                [95, 60, 30],
            ],
            dtype=np.uint8,
        ),
    },
    "02-PALETTE": {
        "attrs": {
            "CLASS": b"PALETTE",
            "PAL_COLORMODEL": b"RGB",
            "PAL_TYPE": b"DIRECTINDEX",
        },
        "value": np.array(
            [[100, 100, 100], [255, 100, 0], [0, 80, 215], [95, 60, 30]], dtype=np.uint8
        ),
    },
    "CT": {
        "attrs": {
            "CLASS": b"IMAGE",
            "ID": b"CT",
            "IMAGE_COLORMODEL": b"RGB",
            "IMAGE_SUBCLASS": b"IMAGE_INDEXED",
            "IMAGE_VERSION": b"1.0",
            "N_COLS": 3712,
            "N_LINES": 1856,
            "OFFSET": 0.0,
            "PALETTE": "<HDF5 object reference> 01-PALETTE",
            "PRODUCT": b"CT__",
            "SCALING_FACTOR": 1.0,
        },
        "value": (np.random.rand(1856, 3712) * 255).astype(np.uint8),
    },
    "CT_PHASE": {
        "attrs": {
            "CLASS": b"IMAGE",
            "ID": b"CT_PHASE",
            "IMAGE_COLORMODEL": b"RGB",
            "IMAGE_SUBCLASS": b"IMAGE_INDEXED",
            "IMAGE_VERSION": b"1.0",
            "N_COLS": 3712,
            "N_LINES": 1856,
            "OFFSET": 0.0,
            "PALETTE": "<HDF5 object reference> 02-PALETTE",
            "PRODUCT": b"CT__",
            "SCALING_FACTOR": 1.0,
        },
        "value": (np.random.rand(1856, 3712) * 255).astype(np.uint8),
    },
    "CT_QUALITY": {
        "attrs": {
            "CLASS": b"IMAGE",
            "ID": b"CT_QUALITY",
            "IMAGE_COLORMODEL": b"RGB",
            "IMAGE_SUBCLASS": b"IMAGE_INDEXED",
            "IMAGE_VERSION": b"1.0",
            "N_COLS": 3712,
            "N_LINES": 1856,
            "OFFSET": 0.0,
            "PRODUCT": b"CT__",
            "SCALING_FACTOR": 1.0,
        },
        "value": (np.random.rand(1856, 3712) * 65535).astype(np.uint16),
    },
    "attrs": {
        "CFAC": 13642337,
        "COFF": 1856,
        "GP_SC_ID": 323,
        "IMAGE_ACQUISITION_TIME": b"201611090800",
        "LFAC": 13642337,
        "LOFF": 1856,
        "NB_PARAMETERS": 3,
        "NC": 3712,
        "NL": 1856,
        "NOMINAL_PRODUCT_TIME": b"201611090814",
        "PACKAGE": b"SAFNWC/MSG",
        "PRODUCT_ALGORITHM_VERSION": b"             2.2",
        "PRODUCT_NAME": b"CT__",
        "PROJECTION_NAME": b"GEOS<+000.0>",
        "REGION_NAME": b"MSG-N",
        "SAF": b"NWC",
        "SGS_PRODUCT_COMPLETENESS": 99,
        "SGS_PRODUCT_QUALITY": 79,
        "SPECTRAL_CHANNEL_ID": 0,
    },
}

fake_ct = OrderedDict(sorted(fake_ct.items(), key=lambda t: t[0]))

PROJ_KM = {
    "gdal_projection": "+proj=geos +a=6378.137000 +b=6356.752300 +lon_0=0.000000 +h=35785.863000",
    "gdal_xgeo_up_left": -5569500.0,
    "gdal_ygeo_up_left": 5437500.0,
    "gdal_xgeo_low_right": 5566500.0,
    "gdal_ygeo_low_right": 2653500.0,
}
PROJ = {
    "gdal_projection": "+proj=geos +a=6378137.000 +b=6356752.300 +lon_0=0.000000 +h=35785863.000",
    "gdal_xgeo_up_left": -5569500.0,
    "gdal_ygeo_up_left": 5437500.0,
    "gdal_xgeo_low_right": 5566500.0,
    "gdal_ygeo_low_right": 2653500.0,
}


class TestH5NWCSAF(unittest.TestCase):
    """Test the nwcsaf msg reader."""

    def setUp(self):
        """Set up the tests."""
        self.filename = os.path.join(
            tempfile.gettempdir(),
            "SAFNWC_MSG3_CT___201611090800_MSG-N_______.PLAX.CTTH.0.h5")

        def fill_h5(root, stuff):
            for key, val in stuff.items():
                if key in ["value", "attrs"]:
                    continue
                if "value" in val:
                    root[key] = val["value"]
                else:
                    grp = root.create_group(key)
                    fill_h5(grp, stuff[key])
                if "attrs" in val:
                    for attrs, val in val["attrs"].items():
                        if isinstance(val, str) and val.startswith("<HDF5 object reference>"):
                            root[key].attrs[attrs] = root[val[24:]].ref
                        else:
                            root[key].attrs[attrs] = val

        h5f = h5py.File(self.filename, mode="w")
        fill_h5(h5f, fake_ct)
        for attr, val in fake_ct["attrs"].items():
            h5f.attrs[attr] = val
        h5f.close()

    def test_get_dataset(self):
        """Retrieve datasets from a DNB file."""
        from satpy.readers.nwcsaf_msg2013_hdf5 import Hdf5NWCSAF
        from satpy import DatasetID

        filename_info = {}
        filetype_info = {}
        dsid = DatasetID(name='ct')
        test = Hdf5NWCSAF(self.filename, filename_info, filetype_info)
        ds = test.get_dataset(dsid, {'file_key': 'CT'})
        self.assertEqual(ds.shape, (1856, 3712))
        self.assertEqual(ds.dtype, np.uint8)

    def tearDown(self):
        """Destroy."""
        try:
            os.remove(self.filename)
        except OSError:
            pass


def suite():
    """Test suite for test_writers."""
    loader = unittest.TestLoader()
    my_suite = unittest.TestSuite()
    my_suite.addTest(loader.loadTestsFromTestCase(TestH5NWCSAF))

    return my_suite
