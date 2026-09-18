"""Benchmark AHI HSD operations.."""

from __future__ import annotations

import os

from pyspectral.rayleigh import check_and_download as download_luts
from pyspectral.rsr_reader import check_and_download as download_rsr

from benchmarks.utils import GeoBenchmarks, get_filenames


class HimawariHSD(GeoBenchmarks):
    """Benchmark Himawari HSD reading."""

    timeout = 600
    data_files: list[str] = []
    subdir = os.path.join("ahi_hsd", "20210417_0500_typhoon_surigae")
    reader = "ahi_hsd"

    def setup_cache(self):
        """Fetch the data files."""
        try:
            from satpy.demo import download_typhoon_surigae_ahi
            download_typhoon_surigae_ahi(channels=[1, 2, 3, 4], segments=[4])
        except ImportError:
            assert len(get_filenames(self.subdir)) == 4  # nosec
        download_rsr()
        download_luts(aerosol_types=["rayleigh_only"])

    def setup(self):
        """Set up the benchmarks."""
        import dask.config
        self.data_files = get_filenames(self.subdir)
        dask.config.set({"array.chunk-size": "32MiB"})

    def time_load_one_channel(self):
        """Time the loading of one channel."""
        self.compute_channel("B01")

    def peakmem_load_one_channel(self):
        """Check peak memory usage of loading one channel."""
        self.compute_channel("B01")

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
