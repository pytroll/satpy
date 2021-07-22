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
"""Benchmark AHI HSD operations.."""

from pyspectral.rayleigh import check_and_download as download_luts
from pyspectral.rsr_reader import check_and_download as download_rsr
import os

from benchmarks.utils import get_filenames, GeoBenchmarks


class HimawariHSD(GeoBenchmarks):
    """Benchmark Himawari HSD reading."""

    timeout = 600
    data_files = []
    subdir = os.path.join("ahi_hsd", "20210417_0500_random")
    reader = 'ahi_hsd'

    def setup_cache(self):
        """Fetch the data files."""
        try:
            from satpy.demo import download_typhoon_surigae_ahi
            download_typhoon_surigae_ahi(channels=[1, 2, 3, 4], segments=[4])
        except ImportError:
            assert len(get_filenames(self.subdir)) == 4
        download_rsr()
        download_luts(aerosol_type='rayleigh_only')

    def setup(self):
        """Set up the benchmarks."""
        import satpy
        self.data_files = get_filenames(self.subdir)
        satpy.CHUNK_SIZE = 2048

    def time_load_one_channel(self):
        """Time the loading of one channel."""
        self.compute_B01()

    def peakmem_load_one_channel(self):
        """Check peak memory usage of loading one channel."""
        self.compute_B01()

    def time_load_true_color(self):
        """Time the loading of the generation of true_color."""
        self.compute_true_color()

    def peakmem_load_true_color(self):
        """Check peak memory usage of the generation of true_color."""
        self.compute_true_color()

    def time_save_true_color_nocorr_to_geotiff(self):
        """Time the generation and saving of true_color_nocorr."""
        self.save_true_color_nocorr_as_geotiff()

    def peakmem_save_true_color_to_geotiff(self):
        """Check peak memory usage of the generation and saving of true_color_nocorr."""
        self.save_true_color_nocorr_as_geotiff()

    def load_and_native_resample(self, composite):
        """Load and native resample a composite."""
        scn = self.load_no_padding(composite)
        return scn.resample(resampler='native')

    def compute_B01(self):
        """Load and compute one channel."""
        composite = "B01"
        scn = self.load(composite)
        scn[composite].compute()

    def compute_true_color(self):
        """Compute a true color image."""
        composite = "true_color"
        lscn = self.load_and_native_resample(composite)
        lscn[composite].compute()

    def save_true_color_nocorr_as_geotiff(self):
        """Save a true_color_nocorr to disk as geotiff."""
        composite = "true_color_nocorr"
        lscn = self.load_and_native_resample(composite)
        lscn.save_dataset(composite, filename='test.tif', tiled=True)
