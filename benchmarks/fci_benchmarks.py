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
"""Benchmark FCI.

Benchmarks for reading and processing data from the Meteosat Third Generation
(MTG) Flexible Combined Imager (FCI).  Uses pre-launch simulated test data as
published by EUMETSAT in 2020.

Also includes some benchmarks trying different resamplers.
"""

from __future__ import annotations

import fnmatch
import os

import satpy
import satpy.demo.fci

from .utils import GeoBenchmarks


class FCI(GeoBenchmarks):
    """Benchmark FCI FDHSI test data reading."""

    timeout = 600
    region = "eurol"
    reader = "fci_l1c_nc"
    filenames: list[str] = []

    def setup_cache(self, *args):
        """Fetch the data files."""
        fns = self.get_filenames()
        cnt = len(fns)
        if cnt > 40:
            raise ValueError(f"Expected 41 files, found {cnt:d}")
        if cnt < 40:
            fns = satpy.demo.download_fci_test_data()

    def setup(self, *args):
        """Set location of data files."""
        self.filenames = self.get_filenames()

    def get_filenames(self):
        """Get filenames of FCI test data as already available."""
        p = satpy.demo.fci.get_fci_test_data_dir()
        g = p.glob("UNCOMPRESSED/NOMINAL/*-CHK-BODY-*.nc")
        return [os.fspath(fn) for fn in g]

    def time_create_scene(self, chunk):
        """Time to create a scene."""
        names = self._get_filename_selection(chunk)
        self.create_scene(names)
    time_create_scene.params = ["some", "all"]  # type: ignore
    time_create_scene.param_names = ["channel subset"]  # type: ignore

    def peakmem_create_scene(self, chunk):
        """Peak RAM to create a scene."""
        names = self._get_filename_selection(chunk)
        self.create_scene(names)
    peakmem_create_scene.params = time_create_scene.params  # type: ignore
    peakmem_create_scene.param_names = time_create_scene.param_names  # type: ignore

    def time_load(self, chunk, loadable):
        """Time to create a scene and load one channel or composite."""
        names = self._get_filename_selection(chunk)
        self.load_no_padding(loadable, names)
    time_load.params = (time_create_scene.params,  # type: ignore
                        ["ir_105", "natural_color_raw"])
    time_load.param_names = time_create_scene.param_names + ["dataset"]  # type: ignore

    def peakmem_load(self, chunk, loadable):
        """Peak RAM to create a scene and load one channel or composite."""
        names = self._get_filename_selection(chunk)
        self.load_no_padding(loadable, names)
    peakmem_load.params = time_load.params  # type: ignore
    peakmem_load.param_names = time_load.param_names  # type: ignore

    def time_compute(self, chunk, loadable):
        """Time to create a scene and load and compute one channel."""
        names = self._get_filename_selection(chunk)
        self.compute_channel(loadable, names)
    time_compute.params = time_load.params  # type: ignore
    time_compute.param_names = time_load.param_names  # type: ignore

    def peakmem_compute(self, chunk, loadable):
        """Peak memory for creating a scene and loading and computing one channel."""
        names = self._get_filename_selection(chunk)
        self.compute_channel(loadable, names)
    peakmem_compute.params = time_compute.params  # type: ignore
    peakmem_compute.param_names = time_compute.param_names  # type: ignore

    def time_load_resample_compute(self, chunk, loadable, mode):
        """Time to load all chunks, resample, and compute."""
        names = self._get_filename_selection(chunk)
        self.compute_composite(loadable, mode, self.region, names)
    time_load_resample_compute.params = time_load.params + (  # type: ignore
            ["nearest", "bilinear", "gradient_search"],)
    time_load_resample_compute.param_names = time_load.param_names + ["resampler"]  # type: ignore

    def peakmem_load_resample_compute(self, chunk, loadable, mode):
        """Peak memory to load all chunks, resample, and compute."""
        names = self._get_filename_selection(chunk)
        self.compute_composite(loadable, mode, self.region, names)
    peakmem_load_resample_compute.params = time_load_resample_compute.params  # type: ignore
    peakmem_load_resample_compute.param_names = time_load_resample_compute.param_names  # type: ignore

    def time_load_resample_save(self, chunk, loadable, mode):
        """Time to load all chunks, resample, and save."""
        names = self._get_filename_selection(chunk)
        self.save_composite_as_geotiff(loadable, mode, self.region, names)
    time_load_resample_save.params = time_load_resample_compute.params  # type: ignore
    time_load_resample_save.param_names = time_load_resample_compute.param_names  # type: ignore

    def peakmem_load_resample_save(self, chunk, loadable, mode):
        """Peak memory to load all chunks, resample, and save."""
        names = self._get_filename_selection(chunk)
        self.save_composite_as_geotiff(loadable, mode, self.region, names)
    peakmem_load_resample_save.params = time_load_resample_save.params  # type: ignore
    peakmem_load_resample_save.param_names = time_load_resample_save.param_names  # type: ignore

    def _get_filename_selection(self, selection):
        if selection == "some":
            return fnmatch.filter(self.filenames, "*3[0123].nc")
        if selection == "all":
            return self.filenames
        raise ValueError("Expected selection some or all, got " +
                         selection)
