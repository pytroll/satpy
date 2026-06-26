#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2020 Satpy developers
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

"""SLSTR L1b reader."""

import datetime as dt
import logging
import os
import re
import warnings

import dask.array as da
import numpy as np
import xarray as xr

from satpy.readers.core.file_handlers import BaseFileHandler
from satpy.readers.core.slstr import CHANCALIB_FACTORS, PLATFORM_NAMES
from satpy.utils import get_legacy_chunk_size

logger = logging.getLogger(__name__)

CHUNK_SIZE = get_legacy_chunk_size()

class NCSLSTR1B(BaseFileHandler):
    """Filehandler for l1 SLSTR data.

    By default, the calibration factors recommended by EUMETSAT are applied.
    This is required as the SLSTR VIS channels are producing slightly incorrect
    radiances that require adjustment.
    Satpy uses the radiance corrections in S3.PN-SLSTR-L1.08, checked 11/03/2022.
    User-supplied coefficients can be passed via the `user_calibration` kwarg
    This should be a dict of channel names (such as `S1_nadir`, `S8_oblique`).

    For example::

        calib_dict = {'S1_nadir': 1.12}
        scene = satpy.Scene(filenames,
                            reader='slstr-l1b',
                            reader_kwargs={'user_calib': calib_dict})

    Will multiply S1 nadir radiances by 1.12.
    """

    def __init__(self, filename, filename_info, filetype_info,
                 user_calibration=None):
        """Initialize the SLSTR l1 data filehandler."""
        super(NCSLSTR1B, self).__init__(filename, filename_info,
                                        filetype_info)

        self.nc = xr.open_dataset(self.filename,
                                  decode_cf=True,
                                  mask_and_scale=True,
                                  chunks={"columns": CHUNK_SIZE,
                                          "rows": CHUNK_SIZE})
        self.nc = self.nc.rename({"columns": "x", "rows": "y"})
        self.baseline = filename_info["baseline"]
        self.channel = filename_info["dataset_name"]
        self.stripe = filename_info["stripe"]
        views = {"n": "nadir", "o": "oblique"}
        self.view = views[filename_info["view"]]
        cal_file = os.path.join(os.path.dirname(self.filename), "viscal.nc")
        self.cal = xr.open_dataset(cal_file,
                                   decode_cf=True,
                                   mask_and_scale=True,
                                   chunks={"views": CHUNK_SIZE})
        indices_file = os.path.join(os.path.dirname(self.filename),
                                    "indices_{}{}.nc".format(self.stripe, self.view[0]))
        self.indices = xr.open_dataset(indices_file,
                                       decode_cf=True,
                                       mask_and_scale=True,
                                       chunks={"columns": CHUNK_SIZE,
                                               "rows": CHUNK_SIZE})
        self.indices = self.indices.rename({"columns": "x", "rows": "y"})

        self.platform_name = PLATFORM_NAMES[filename_info["mission_id"]]
        self.sensor = "slstr"
        if isinstance(user_calibration, dict):
            self.usercalib = user_calibration
        else:
            self.usercalib = None

    def _apply_radiance_adjustment(self, radiances):
        """Adjust SLSTR radiances with default or user supplied values."""
        chan_name = self.channel + "_" + self.view
        adjust_fac = None
        if self.usercalib is not None:
            # If user supplied adjustment, use it.
            if chan_name in self.usercalib:
                adjust_fac = self.usercalib[chan_name]
        if adjust_fac is None:
            if chan_name in CHANCALIB_FACTORS:
                adjust_fac = CHANCALIB_FACTORS[chan_name]
            else:
                warnings.warn(
                    "Warning: No radiance adjustment supplied " +
                    "for channel " + chan_name,
                    stacklevel=3
                )
                return radiances
        return radiances * adjust_fac

    @staticmethod
    def _cal_rad(rad, didx, solar_flux=None):
        """Calibrate."""
        indices = np.isfinite(didx)
        rad[indices] /= solar_flux[didx[indices].astype(int)]
        return rad

    def get_dataset(self, key, info):
        """Load a dataset."""
        if (self.channel not in key["name"] or
                self.stripe != key["stripe"].name or
                self.view != key["view"].name):
            return
        logger.debug("Reading %s.", key["name"])
        chan_type = "BT" if key["calibration"] == "brightness_temperature" else "radiance"
        variable = self.nc[f"{self.channel}_{chan_type}_{self.stripe}{self.view[0]}"]
        # Processing baseline version 005 and above already include the radiance adjustment factor
        # Therefore, unless user supplies their own, do not apply here.
        if self.baseline < 5 or self.usercalib is not None:
            radiances = self._apply_radiance_adjustment(variable)
        else:
            radiances = variable
        units = variable.attrs["units"]
        if key["calibration"] == "reflectance":
            # TODO take into account sun-earth distance
            solar_flux = self.cal[re.sub("_[^_]*$", "", key["name"]) + "_solar_irradiances"]
            d_index = self.indices["detector_{}{}".format(self.stripe, self.view[0])]
            idx = 0 if self.view[0] == "n" else 1  # 0: Nadir view, 1: oblique (check).
            radiances.data = da.map_blocks(
                self._cal_rad, radiances.data, d_index.data, solar_flux=solar_flux[:, idx].values)
            radiances *= np.pi * 100
            units = "%"

        info = info.copy()
        info.update(radiances.attrs)
        info.update(key.to_dict())
        info.update(dict(units=units,
                         platform_name=self.platform_name,
                         sensor=self.sensor,
                         view=self.view))

        radiances.attrs = info
        return radiances

    @property
    def start_time(self):
        """Get the start time."""
        return dt.datetime.strptime(self.nc.attrs["start_time"], "%Y-%m-%dT%H:%M:%S.%fZ")

    @property
    def end_time(self):
        """Get the end time."""
        return dt.datetime.strptime(self.nc.attrs["stop_time"], "%Y-%m-%dT%H:%M:%S.%fZ")
