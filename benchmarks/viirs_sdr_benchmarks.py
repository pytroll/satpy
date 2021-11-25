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
"""Benchmark VIIRS SDR operations.."""

from __future__ import annotations

import glob
import os

from pyspectral.rayleigh import check_and_download as download_luts
from pyspectral.rsr_reader import check_and_download as download_rsr


class VIIRSSDRBenchmarkBase:
    """Shared methods for working with VIIRS SDR data."""

    timeout = 600
    data_files: list[str] = []

    def setup_cache(self):
        """Fetch the data files."""
        try:
            from satpy.demo import get_viirs_sdr_20170128_1229
            get_viirs_sdr_20170128_1229(
                channels=("I01", "M03", "M04", "M05"),
                granules=(2, 3, 4))
        except ImportError:
            assert len(self.get_filenames()) == 6 * 3
        download_rsr()
        download_luts(aerosol_type='rayleigh_only')

    def setup(self, name):
        """Set up the benchmarks."""
        import satpy
        self.data_files = self.get_filenames()
        satpy.CHUNK_SIZE = 2048

    def get_filenames(self):
        """Get the data filenames manually."""
        base_dir = os.environ.get("SATPY_DEMO_DATA_DIR", ".")
        return glob.glob(os.path.join(base_dir, "viirs_sdr", "20170128_1229", "*.h5"))

    def load(self, composite):
        """Load one composite."""
        from satpy import Scene
        scn = Scene(filenames=self.data_files, reader='viirs_sdr')
        scn.load([composite])
        return scn

    def load_and_native_resample(self, composite):
        """Load and native resample a composite."""
        scn = self.load(composite)
        lscn = scn.resample(resampler='native')
        return lscn


class VIIRSSDRReaderBenchmarks(VIIRSSDRBenchmarkBase):
    """Benchmark reading and writing VIIRS SDR data."""

    params = ["I01", "M03"]
    param_names = ["name"]

    def time_load_one_channel(self, name):
        """Time the loading of one channel."""
        self.compute_product(name)

    def peakmem_load_one_channel(self, name):
        """Check peak memory usage of loading one channel."""
        self.compute_product(name)

    def compute_product(self, name):
        """Load and compute one channel."""
        scn = self.load(name)
        scn[name].compute()


class VIIRSSDRCompositeBenchmarks(VIIRSSDRBenchmarkBase):
    """Benchmark generating and writing composites from VIIRS SDR data."""

    params = ["true_color", "true_color_crefl", "true_color_raw"]
    param_names = ["name"]

    def time_load_composite(self, name):
        """Time the loading of the generation of a composite."""
        self.compute_composite(name)

    def peakmem_load_composite(self, name):
        """Check peak memory usage of the generation of a composite."""
        self.compute_composite(name)

    def time_save_composite_to_geotiff(self, name):
        """Time the generation and saving of a composite."""
        self.save_composite_as_geotiff(name)

    def peakmem_save_composite_raw_to_geotiff(self, name):
        """Check peak memory usage of the generation and saving of a composite."""
        self.save_composite_as_geotiff(name)

    def compute_composite(self, name):
        """Compute a composite."""
        lscn = self.load_and_native_resample(name)
        lscn[name].compute()

    def save_composite_as_geotiff(self, name):
        """Save a composite to disk as geotiff."""
        lscn = self.load_and_native_resample(name)
        lscn.save_dataset(name, filename='test.tif', tiled=True)
