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
"""Benchmark SEVIRI HRIT operations."""

from pyspectral.rayleigh import check_and_download as download_luts
from pyspectral.rsr_reader import check_and_download as download_rsr

from benchmarks.utils import get_filenames, GeoBenchmarks
import os


class SEVIRIHRIT(GeoBenchmarks):
    """Benchmark SEVIRI HRIT reading."""

    timeout = 600
    data_files = []
    subdir = os.path.join("seviri_hrit", "20180228_1500")
    reader = "seviri_l1b_hrit"

    def setup_cache(self):
        """Fetch the data files."""
        try:
            from satpy.demo import download_seviri_hrit_20180228_1500
            download_seviri_hrit_20180228_1500()
        except ImportError:
            assert len(get_filenames(self.subdir)) == 114
        download_rsr()
        download_luts(aerosol_type='rayleigh_only')

    def setup(self):
        """Set up the benchmarks."""
        import satpy
        self.data_files = get_filenames(self.subdir)
        satpy.CHUNK_SIZE = 2048

    def time_load_one_channel(self):
        """Time the loading of one channel."""
        self.compute_vis006()

    def peakmem_load_one_channel(self):
        """Check peak memory usage of loading one channel."""
        self.compute_vis006()

    def time_load_overview(self):
        """Time the loading of the generation of overview."""
        self.compute_overview()

    def peakmem_load_overview(self):
        """Check peak memory usage of the generation of overview."""
        self.compute_overview()

    def time_save_overview_to_geotiff(self):
        """Time the generation and saving of overview."""
        self.save_overview_as_geotiff()

    def peakmem_save_overview_to_geotiff(self):
        """Check peak memory usage of the generation and saving of overview."""
        self.save_overview_as_geotiff()

    def compute_vis006(self):
        """Load and compute one channel."""
        composite = "VIS006"
        scn = self.load_no_padding(composite)
        scn[composite].compute()

    def compute_overview(self):
        """Compute an overview image."""
        composite = "overview"
        scn = self.load(composite)
        scn[composite].compute()

    def save_overview_as_geotiff(self):
        """Save a overview to disk as geotiff."""
        composite = "overview"
        scn = self.load(composite)
        scn.save_dataset(composite, filename='test.tif', tiled=True)
