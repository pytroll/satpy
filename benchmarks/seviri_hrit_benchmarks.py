"""Benchmark SEVIRI HRIT operations."""

from __future__ import annotations

import os

from pyspectral.rayleigh import check_and_download as download_luts
from pyspectral.rsr_reader import check_and_download as download_rsr

from benchmarks.utils import GeoBenchmarks, get_filenames


class SEVIRIHRIT(GeoBenchmarks):
    """Benchmark SEVIRI HRIT reading."""

    timeout = 600
    data_files: list[str] = []
    subdir = os.path.join("seviri_hrit", "20180228_1500")
    reader = "seviri_l1b_hrit"

    def setup_cache(self):
        """Fetch the data files."""
        try:
            from satpy.demo import download_seviri_hrit_20180228_1500
            download_seviri_hrit_20180228_1500()
        except ImportError:
            assert len(get_filenames(self.subdir)) == 114  # nosec
        download_rsr()
        download_luts(aerosol_types=["rayleigh_only"])

    def setup(self):
        """Set up the benchmarks."""
        import dask.config
        self.data_files = get_filenames(self.subdir)
        dask.config.set({"array.chunk-size": "32MiB"})

    def time_load_one_channel(self):
        """Time the loading of one channel."""
        self.compute_channel("VIS006")

    def peakmem_load_one_channel(self):
        """Check peak memory usage of loading one channel."""
        self.compute_channel("VIS006")

    def time_load_overview(self):
        """Time the loading of the generation of overview."""
        self.compute_composite("overview")

    def peakmem_load_overview(self):
        """Check peak memory usage of the generation of overview."""
        self.compute_composite("overview")

    def time_save_overview_to_geotiff(self):
        """Time the generation and saving of overview."""
        self.save_composite_as_geotiff("overview")

    def peakmem_save_overview_to_geotiff(self):
        """Check peak memory usage of the generation and saving of overview."""
        self.save_composite_as_geotiff("overview")
