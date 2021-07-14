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

from pyspectral.rayleigh import check_and_download as download_luts
from pyspectral.rsr_reader import check_and_download as download_rsr


class VIIRSSDR:
    """Benchmark reading, compositing, and writing VIIRS SDR data."""

    timeout = 600
    data_files = []

    def setup_cache(self):
        """Fetch the data files."""
        try:
            from satpy.demo import get_viirs_sdr_20170128_1229
            get_viirs_sdr_20170128_1229(
                channels=("I01", "M03", "M04", "M05"),
                num_granules=3)
        except ImportError:
            assert len(self.get_filenames()) == 6 * 3
        download_rsr()
        download_luts(aerosol_type='rayleigh_only')

    def setup(self):
        """Set up the benchmarks."""
        import satpy
        self.data_files = self.get_filenames()
        satpy.CHUNK_SIZE = 2048

    def get_filenames(self):
        """Get the data filenames manually."""
        import os
        import glob
        base_dir = os.environ.get("SATPY_DEMO_DATA_DIR", ".")
        return glob.glob(os.path.join(base_dir, "viirs_sdr", "viirs_sdr_20170128_1229", "*.h5"))

    def time_load_one_i_channel(self):
        """Time the loading of one I-band resolution channel."""
        self.compute_I01()

    def peakmem_load_one_i_channel(self):
        """Check peak memory usage of loading one I-band resolution channel."""
        self.compute_I01()

    def time_load_one_m_channel(self):
        """Time the loading of one M-band resolution channel."""
        self.compute_M03()

    def peakmem_load_one_m_channel(self):
        """Check peak memory usage of loading one M-band resolution channel."""
        self.compute_M03()

    def time_load_true_color(self):
        """Time the loading of the generation of true_color."""
        self.compute_true_color()

    def peakmem_load_true_color(self):
        """Check peak memory usage of the generation of true_color."""
        self.compute_true_color()

    def time_load_true_color_crefl(self):
        """Time the loading of the generation of true_color_crefl."""
        self.compute_true_color_crefl()

    def peakmem_load_true_color_crefl(self):
        """Check peak memory usage of the generation of true_color_crefl."""
        self.compute_true_color_crefl()

    def time_save_true_color_raw_to_geotiff(self):
        """Time the generation and saving of true_color_raw."""
        self.save_true_color_raw_as_geotiff()

    def peakmem_save_true_color_raw_to_geotiff(self):
        """Check peak memory usage of the generation and saving of true_color_raw."""
        self.save_true_color_raw_as_geotiff()

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

    def compute_I01(self):
        """Load and compute one channel."""
        composite = "I01"
        scn = self.load(composite)
        scn[composite].compute()

    def compute_M03(self):
        """Load and compute one channel."""
        composite = "M03"
        scn = self.load(composite)
        scn[composite].compute()

    def compute_true_color(self):
        """Compute a true color image."""
        composite = "true_color"
        lscn = self.load_and_native_resample(composite)
        lscn[composite].compute()

    def compute_true_color_crefl(self):
        """Compute a true color crefl image."""
        composite = "true_color_crefl"
        lscn = self.load_and_native_resample(composite)
        lscn[composite].compute()

    def compute_true_color_raw(self):
        """Compute a true color raw image."""
        composite = "true_color_raw"
        lscn = self.load_and_native_resample(composite)
        lscn[composite].compute()

    def save_true_color_raw_as_geotiff(self):
        """Save a true_color_raw to disk as geotiff."""
        composite = "true_color_raw"
        lscn = self.load_and_native_resample(composite)
        lscn.save_dataset(composite, filename='test.tif', tiled=True)
