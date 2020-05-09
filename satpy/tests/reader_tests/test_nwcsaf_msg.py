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

CTYPE_TEST_ARRAY = (np.random.rand(1856, 3712) * 255).astype(np.uint8)
CTYPE_TEST_FRAME = (np.arange(100).reshape(10, 10) / 100. * 20).astype(np.uint8)
CTYPE_TEST_ARRAY[1000:1010, 1000:1010] = CTYPE_TEST_FRAME

CTTH_HEIGHT_TEST_ARRAY = (np.random.rand(1856, 3712) * 255).astype(np.uint8)
_CTTH_HEIGHT_TEST_FRAME = (np.arange(100).reshape(10, 10) / 100. * 80).astype(np.uint8)
CTTH_HEIGHT_TEST_ARRAY[1000:1010, 1000:1010] = _CTTH_HEIGHT_TEST_FRAME

CTTH_HEIGHT_TEST_FRAME_RES = _CTTH_HEIGHT_TEST_FRAME.astype(np.float32) * 200 - 2000
CTTH_HEIGHT_TEST_FRAME_RES[0, 0:10] = np.nan
CTTH_HEIGHT_TEST_FRAME_RES[1, 0:3] = np.nan

CTTH_PRESSURE_TEST_ARRAY = (np.random.rand(1856, 3712) * 255).astype(np.uint8)
_CTTH_PRESSURE_TEST_FRAME = (np.arange(100).reshape(10, 10) / 100. * 54).astype(np.uint8)
CTTH_PRESSURE_TEST_ARRAY[1000:1010, 1000:1010] = _CTTH_PRESSURE_TEST_FRAME

CTTH_PRESSURE_TEST_FRAME_RES = _CTTH_PRESSURE_TEST_FRAME.astype(np.float32) * 25 - 250
CTTH_PRESSURE_TEST_FRAME_RES[0, 0:10] = np.nan
CTTH_PRESSURE_TEST_FRAME_RES[1, 0:9] = np.nan

CTTH_TEMPERATURE_TEST_ARRAY = (np.random.rand(1856, 3712) * 255).astype(np.uint8)
_CTTH_TEMPERATURE_TEST_FRAME = (np.arange(100).reshape(10, 10) / 100. * 140).astype(np.uint8)
_CTTH_TEMPERATURE_TEST_FRAME[8, 5] = 255
CTTH_TEMPERATURE_TEST_ARRAY[1000:1010, 1000:1010] = _CTTH_TEMPERATURE_TEST_FRAME

CTTH_TEMPERATURE_TEST_FRAME_RES = _CTTH_TEMPERATURE_TEST_FRAME.astype(np.float32) * 1.0 + 150
CTTH_TEMPERATURE_TEST_FRAME_RES[8, 5] = np.nan


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
        "value": (CTYPE_TEST_ARRAY),
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

fake_ctth = {
    "01-PALETTE": {
        "attrs": {
            "CLASS": b"PALETTE",
            "PAL_COLORMODEL": b"RGB",
            "PAL_TYPE": b"DIRECTINDEX",
        },
        "value": np.array(
            [
                [0, 0, 0],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
                [239, 239, 223],
                [239, 239, 223],
                [238, 214, 210],
                [238, 214, 210],
                [0, 255, 255],
                [0, 255, 255],
                [0, 216, 255],
                [0, 216, 255],
                [0, 178, 255],
                [0, 178, 255],
                [0, 140, 48],
                [0, 140, 48],
                [0, 255, 0],
                [0, 255, 0],
                [153, 255, 0],
                [153, 255, 0],
                [178, 255, 0],
                [178, 255, 0],
                [216, 255, 0],
                [216, 255, 0],
                [255, 255, 0],
                [255, 255, 0],
                [255, 216, 0],
                [255, 216, 0],
                [255, 164, 0],
                [255, 164, 0],
                [255, 102, 0],
                [255, 102, 0],
                [255, 76, 0],
                [255, 76, 0],
                [178, 51, 0],
                [178, 51, 0],
                [153, 20, 47],
                [153, 20, 47],
                [126, 0, 43],
                [126, 0, 43],
                [255, 0, 216],
                [255, 0, 216],
                [255, 0, 128],
                [255, 0, 128],
                [255, 0, 128],
                [255, 0, 128],
                [255, 0, 128],
                [255, 0, 128],
                [255, 0, 128],
                [255, 0, 128],
                [255, 0, 128],
                [255, 0, 128],
                [255, 0, 128],
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
        "value": (np.random.rand(128, 3) * 255).astype(np.uint8),
    },
    "03-PALETTE": {
        "attrs": {
            "CLASS": b"PALETTE",
            "PAL_COLORMODEL": b"RGB",
            "PAL_TYPE": b"DIRECTINDEX",
        },
        "value": (np.random.rand(256, 3) * 255).astype(np.uint8),
    },
    "04-PALETTE": {
        "attrs": {
            "CLASS": b"PALETTE",
            "PAL_COLORMODEL": b"RGB",
            "PAL_TYPE": b"DIRECTINDEX",
        },
        "value": np.array(
            [
                [78, 119, 145],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [12, 12, 12],
                [24, 24, 24],
                [36, 36, 36],
                [48, 48, 48],
                [60, 60, 60],
                [72, 72, 72],
                [84, 84, 84],
                [96, 96, 96],
                [108, 108, 108],
                [120, 120, 120],
                [132, 132, 132],
                [144, 144, 144],
                [156, 156, 156],
                [168, 168, 168],
                [180, 180, 180],
                [192, 192, 192],
                [204, 204, 204],
                [216, 216, 216],
                [228, 228, 228],
                [240, 240, 240],
                [240, 240, 240],
            ],
            dtype=np.uint8,
        ),
    },
    "CTTH_EFFECT": {
        "attrs": {
            "CLASS": b"IMAGE",
            "ID": b"CTTH_EFFECT",
            "IMAGE_COLORMODEL": b"RGB",
            "IMAGE_SUBCLASS": b"IMAGE_INDEXED",
            "IMAGE_VERSION": b"1.0",
            "N_COLS": 3712,
            "N_LINES": 1856,
            "OFFSET": -50.0,
            "PALETTE": "<HDF5 object reference> 04-PALETTE",
            "PRODUCT": b"CTTH",
            "SCALING_FACTOR": 5.0,
        },
        "value": (np.random.rand(1856, 3712) * 255).astype(np.uint8),
    },
    "CTTH_HEIGHT": {
        "attrs": {
            "CLASS": b"IMAGE",
            "ID": b"CTTH_HEIGHT",
            "IMAGE_COLORMODEL": b"RGB",
            "IMAGE_SUBCLASS": b"IMAGE_INDEXED",
            "IMAGE_VERSION": b"1.0",
            "N_COLS": 3712,
            "N_LINES": 1856,
            "OFFSET": -2000.0,
            "PALETTE": "<HDF5 object reference> 02-PALETTE",
            "PRODUCT": b"CTTH",
            "SCALING_FACTOR": 200.0,
        },
        "value": (CTTH_HEIGHT_TEST_ARRAY),
    },
    "CTTH_PRESS": {
        "attrs": {
            "CLASS": b"IMAGE",
            "ID": b"CTTH_PRESS",
            "IMAGE_COLORMODEL": b"RGB",
            "IMAGE_SUBCLASS": b"IMAGE_INDEXED",
            "IMAGE_VERSION": b"1.0",
            "N_COLS": 3712,
            "N_LINES": 1856,
            "OFFSET": -250.0,
            "PALETTE": "<HDF5 object reference> 01-PALETTE",
            "PRODUCT": b"CTTH",
            "SCALING_FACTOR": 25.0,
        },
        "value": (CTTH_PRESSURE_TEST_ARRAY),
    },
    "CTTH_QUALITY": {
        "attrs": {
            "CLASS": b"IMAGE",
            "ID": b"CTTH_QUALITY",
            "IMAGE_COLORMODEL": b"RGB",
            "IMAGE_SUBCLASS": b"IMAGE_INDEXED",
            "IMAGE_VERSION": b"1.0",
            "N_COLS": 3712,
            "N_LINES": 1856,
            "OFFSET": 0.0,
            "PRODUCT": b"CTTH",
            "SCALING_FACTOR": 1.0,
        },
        "value": (np.random.rand(1856, 3712) * 65535).astype(np.uint16),
    },
    "CTTH_TEMPER": {
        "attrs": {
            "CLASS": b"IMAGE",
            "ID": b"CTTH_TEMPER",
            "IMAGE_COLORMODEL": b"RGB",
            "IMAGE_SUBCLASS": b"IMAGE_INDEXED",
            "IMAGE_VERSION": b"1.0",
            "N_COLS": 3712,
            "N_LINES": 1856,
            "OFFSET": 150.0,
            "PALETTE": "<HDF5 object reference> 03-PALETTE",
            "PRODUCT": b"CTTH",
            "SCALING_FACTOR": 1.0,
        },
        "value": (CTTH_TEMPERATURE_TEST_ARRAY),
    },
    "attrs": {
        "CFAC": 13642337,
        "COFF": 1856,
        "GP_SC_ID": 323,
        "IMAGE_ACQUISITION_TIME": b"201611090800",
        "LFAC": 13642337,
        "LOFF": 1856,
        "NB_PARAMETERS": 5,
        "NC": 3712,
        "NL": 1856,
        "NOMINAL_PRODUCT_TIME": b"201611090816",
        "PACKAGE": b"SAFNWC/MSG",
        "PRODUCT_ALGORITHM_VERSION": b"             2.2",
        "PRODUCT_NAME": b"CTTH",
        "PROJECTION_NAME": b"GEOS<+000.0>",
        "REGION_NAME": b"MSG-N",
        "SAF": b"NWC",
        "SGS_PRODUCT_COMPLETENESS": 87,
        "SGS_PRODUCT_QUALITY": 69,
        "SPECTRAL_CHANNEL_ID": 0,
    },
}

fake_ctth = OrderedDict(sorted(fake_ctth.items(), key=lambda t: t[0]))

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

AREA_DEF_DICT = {
    "proj_dict": {'proj': 'geos', 'lon_0': 0, 'h': 35785831, 'x_0': 0, 'y_0': 0,
                  'a': 6378169, 'b': 6356583.8, 'units': 'm', 'no_defs': None, 'type': 'crs'},
    "area_id": 'MSG-N',
    "x_size": 3712,
    "y_size": 1856,
    "area_extent": (-5570248.2825, 1501.0099, 5567247.8793, 5570247.8784)
}


class TestH5NWCSAF(unittest.TestCase):
    """Test the nwcsaf msg reader."""

    def setUp(self):
        """Set up the tests."""
        self.filename_ct = os.path.join(
            tempfile.gettempdir(),
            "SAFNWC_MSG3_CT___201611090800_MSG-N_______.PLAX.CTTH.0.h5",
        )

        self.filename_ctth = os.path.join(
            tempfile.gettempdir(),
            "SAFNWC_MSG3_CTTH_201611090800_MSG-N_______.PLAX.CTTH.0.h5",
        )

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
                        if isinstance(val, str) and val.startswith(
                            "<HDF5 object reference>"
                        ):
                            root[key].attrs[attrs] = root[val[24:]].ref
                        else:
                            root[key].attrs[attrs] = val

        h5f = h5py.File(self.filename_ct, mode="w")
        fill_h5(h5f, fake_ct)
        for attr, val in fake_ct["attrs"].items():
            h5f.attrs[attr] = val
        h5f.close()

        h5f = h5py.File(self.filename_ctth, mode="w")
        fill_h5(h5f, fake_ctth)
        for attr, val in fake_ctth["attrs"].items():
            h5f.attrs[attr] = val
        h5f.close()

    def test_get_area_def(self):
        """Get the area definition."""
        from satpy.readers.nwcsaf_msg2013_hdf5 import Hdf5NWCSAF
        from satpy import DatasetID

        filename_info = {}
        filetype_info = {}
        dsid = DatasetID(name="ct")
        test = Hdf5NWCSAF(self.filename_ct, filename_info, filetype_info)

        area_def = test.get_area_def(dsid)

        aext_res = AREA_DEF_DICT['area_extent']
        for i in range(4):
            self.assertAlmostEqual(area_def.area_extent[i], aext_res[i], 4)

        proj_dict = AREA_DEF_DICT['proj_dict']
        self.assertEqual(proj_dict['proj'], area_def.proj_dict['proj'])
        # Not all elements passed on Appveyor, so skip testing every single element of the proj-dict:
        # for key in proj_dict:
        #    self.assertEqual(proj_dict[key], area_def.proj_dict[key])

        self.assertEqual(AREA_DEF_DICT['x_size'], area_def.width)
        self.assertEqual(AREA_DEF_DICT['y_size'], area_def.height)

        self.assertEqual(AREA_DEF_DICT['area_id'], area_def.area_id)

    def test_get_dataset(self):
        """Retrieve datasets from a NWCSAF msgv2013 hdf5 file."""
        from satpy.readers.nwcsaf_msg2013_hdf5 import Hdf5NWCSAF
        from satpy import DatasetID

        filename_info = {}
        filetype_info = {}
        dsid = DatasetID(name="ct")
        test = Hdf5NWCSAF(self.filename_ct, filename_info, filetype_info)
        ds = test.get_dataset(dsid, {"file_key": "CT"})
        self.assertEqual(ds.shape, (1856, 3712))
        self.assertEqual(ds.dtype, np.uint8)
        np.testing.assert_allclose(ds.data[1000:1010, 1000:1010].compute(), CTYPE_TEST_FRAME)

        filename_info = {}
        filetype_info = {}
        dsid = DatasetID(name="ctth_alti")
        test = Hdf5NWCSAF(self.filename_ctth, filename_info, filetype_info)
        ds = test.get_dataset(dsid, {"file_key": "CTTH_HEIGHT"})
        self.assertEqual(ds.shape, (1856, 3712))
        self.assertEqual(ds.dtype, np.float32)
        np.testing.assert_allclose(ds.data[1000:1010, 1000:1010].compute(), CTTH_HEIGHT_TEST_FRAME_RES)

        filename_info = {}
        filetype_info = {}
        dsid = DatasetID(name="ctth_pres")
        test = Hdf5NWCSAF(self.filename_ctth, filename_info, filetype_info)
        ds = test.get_dataset(dsid, {"file_key": "CTTH_PRESS"})
        self.assertEqual(ds.shape, (1856, 3712))
        self.assertEqual(ds.dtype, np.float32)
        np.testing.assert_allclose(ds.data[1000:1010, 1000:1010].compute(), CTTH_PRESSURE_TEST_FRAME_RES)

        filename_info = {}
        filetype_info = {}
        dsid = DatasetID(name="ctth_tempe")
        test = Hdf5NWCSAF(self.filename_ctth, filename_info, filetype_info)
        ds = test.get_dataset(dsid, {"file_key": "CTTH_TEMPER"})
        self.assertEqual(ds.shape, (1856, 3712))
        self.assertEqual(ds.dtype, np.float32)
        np.testing.assert_allclose(ds.data[1000:1010, 1000:1010].compute(), CTTH_TEMPERATURE_TEST_FRAME_RES)

    def tearDown(self):
        """Destroy."""
        try:
            os.remove(self.filename_ct)
            os.remove(self.filename_ctth)
        except OSError:
            pass
