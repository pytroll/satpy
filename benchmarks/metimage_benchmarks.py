"""Benchmark METimage.

Benchmarks for reading and processing EPS-SG METimage (VII) Level 1B radiance
data with the ``metimage_l1b_nc`` reader. Uses the pre-operational Metop-SGA1
example data that EUMETSAT published on Zenodo
(https://zenodo.org/records/24093210): ten uncompressed granules from
2026-11-05 09:26 to 09:36 UTC.

``setup_cache`` downloads the data (a 720MB gzipped tar file that unpacks to
about 1.2GB) with
:func:`satpy.demo.download_metimage_l1b_20261105_0926` into the
``metimage_l1b/20261105_0926`` subdirectory of the Satpy ``demo_data_dir``,
skipping the download when the files are already there. asv runs
``setup_cache`` in a temporary directory that it deletes afterwards, so set
``SATPY_DEMO_DATA_DIR`` to a persistent directory to avoid downloading the data
again on every run::

    SATPY_DEMO_DATA_DIR=/data/satpy_demo asv run -b metimage

"""

from __future__ import annotations

import glob
import os

import satpy
from satpy import Scene

DEMO_SUBDIR = os.path.join("metimage_l1b", "20261105_0926")
FILE_GLOB = "*-VII-1B-RAD_*.nc"
NUM_GRANULES = 10


class METimageL1B:
    """Benchmark METimage L1B reading."""

    timeout = 600
    reader = "metimage_l1b_nc"
    filenames: list[str] = []

    def setup_cache(self):
        """Fetch the data files."""
        try:
            from satpy.demo import download_metimage_l1b_20261105_0926
            download_metimage_l1b_20261105_0926()
        except ImportError:
            if len(self.get_filenames()) != NUM_GRANULES:
                raise RuntimeError("Existing data files do not match the expected number of files.")
    setup_cache.timeout = 3600  # type: ignore

    def setup(self, *args):
        """Set location of data files."""
        self.filenames = self.get_filenames()

    def get_filenames(self):
        """Get the filenames of the METimage demo data that is already available."""
        base_dir = satpy.config.get("demo_data_dir", ".")
        return sorted(glob.glob(os.path.join(base_dir, DEMO_SUBDIR, FILE_GLOB)))

    def time_create_scene(self, granules):
        """Time to create a scene."""
        self.create_scene(granules)
    time_create_scene.params = ["one", "all"]  # type: ignore
    time_create_scene.param_names = ["granules"]  # type: ignore

    def peakmem_create_scene(self, granules):
        """Peak RAM to create a scene."""
        self.create_scene(granules)
    peakmem_create_scene.params = time_create_scene.params  # type: ignore
    peakmem_create_scene.param_names = time_create_scene.param_names  # type: ignore

    def time_load(self, granules, loadable):
        """Time to create a scene and load one channel or composite."""
        self.load(granules, loadable)
    time_load.params = (time_create_scene.params,  # type: ignore
                        ["vii_10690", "true_color_uncorrected"])
    time_load.param_names = time_create_scene.param_names + ["dataset"]  # type: ignore

    def peakmem_load(self, granules, loadable):
        """Peak RAM to create a scene and load one channel or composite."""
        self.load(granules, loadable)
    peakmem_load.params = time_load.params  # type: ignore
    peakmem_load.param_names = time_load.param_names  # type: ignore

    def time_compute(self, granules, loadable):
        """Time to create a scene and load and compute one channel or composite."""
        self.compute(granules, loadable)
    time_compute.params = time_load.params  # type: ignore
    time_compute.param_names = time_load.param_names  # type: ignore

    def peakmem_compute(self, granules, loadable):
        """Peak RAM to create a scene and load and compute one channel or composite."""
        self.compute(granules, loadable)
    peakmem_compute.params = time_load.params  # type: ignore
    peakmem_compute.param_names = time_load.param_names  # type: ignore

    def create_scene(self, granules):
        """Create a scene from the selected granules."""
        return Scene(filenames=self._get_filename_selection(granules), reader=self.reader)

    def load(self, granules, loadable):
        """Create a scene and load one channel or composite."""
        scn = self.create_scene(granules)
        scn.load([loadable])
        return scn

    def compute(self, granules, loadable):
        """Create a scene, load one channel or composite and compute the whole array."""
        scn = self.load(granules, loadable)
        scn[loadable].compute()

    def _get_filename_selection(self, granules):
        if granules == "one":
            return self.filenames[:1]
        if granules == "all":
            return self.filenames
        raise ValueError(f"Expected granules 'one' or 'all', got {granules!r}")
