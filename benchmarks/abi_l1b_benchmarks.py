#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Satpy developers
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
"""Benchmark ABI L1B operations."""
from __future__ import annotations

import os

from pyspectral.rayleigh import check_and_download as download_luts
from pyspectral.rsr_reader import check_and_download as download_rsr

from benchmarks.utils import GeoBenchmarks, get_filenames


class ABIL1B(GeoBenchmarks):
    """Benchmark ABI L1B reading."""

    timeout = 600
    data_files: list[str] = []
    subdir = os.path.join("abi_l1b", "20190314_us_midlatitude_cyclone")
    reader = "abi_l1b"

    def setup_cache(self):
        """Fetch the data files."""
        try:
            from satpy.demo import get_us_midlatitude_cyclone_abi
            get_us_midlatitude_cyclone_abi()
        except ImportError:
            if len(get_filenames(self.subdir)) != 16:
                raise RuntimeError("Existing data files do not match the expected number of files.")
        download_rsr()
        download_luts(aerosol_type='rayleigh_only')

    def setup(self):
        """Set up the benchmarks."""
        import satpy
        self.data_files = get_filenames(self.subdir)
        satpy.CHUNK_SIZE = 2048

    def time_load_one_channel(self):
        """Time the loading of one channel."""
        self.compute_channel("C01")

    def peakmem_load_one_channel(self):
        """Check peak memory usage of loading one channel."""
        self.compute_channel("C01")

    def time_load_true_color(self):
        """Time the loading of the generation of true_color."""
        self.compute_composite("true_color")

    def peakmem_load_true_color(self):
        """Check peak memory usage of the generation of true_color."""
        self.compute_composite("true_color")

    def time_save_true_color_nocorr_to_geotiff(self):
        """Time the generation and saving of true_color_nocorr."""
        self.save_composite_as_geotiff("true_color_nocorr")

    def peakmem_save_true_color_to_geotiff(self):
        """Check peak memory usage of the generation and saving of true_color_nocorr."""
        self.save_composite_as_geotiff("true_color_nocorr")
