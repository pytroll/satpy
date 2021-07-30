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

import fnmatch
import os

import satpy
import satpy.demo.fci


class FCI:
    """Benchmark FCI FDHSI test data reading."""

    timeout = 600
    data_files = []
    region = "maspalomas"

    def setup(self, *args):
        """Fetch the data files."""
        fns = self.get_filenames()
        cnt = len(fns)
        if cnt > 41:
            raise ValueError(f"Expected 41 files, found {cnt:d}")
        if cnt < 41:
            fns = satpy.demo.download_fci_test_data()
        self.filenames = fnmatch.filter(fns, "*-CHK-BODY-*")

    def get_filenames(self):
        """Get filenames of FCI test data as already available."""
        p = satpy.demo.fci._get_fci_test_data_dir()
        g = p.glob("UNCOMPRESSED/NOMINAL/*.nc")
        return [os.fspath(fn) for fn in g]

    def time_create_scene(self, chunk):
        """Time to create a scene."""
        self.create_scene(chunk)
    time_create_scene.params = ["some", "all"]

    def peakmem_create_scene(self, chunk):
        """Peak RAM to create a scene."""
        self.create_scene(chunk)
    peakmem_create_scene.params = time_create_scene.params

    def time_load(self, chunk, loadable):
        """Time to create a scene and load one channel or composite."""
        self.get_loaded_scene(chunk, loadable)
    time_load.params = (time_create_scene.params,
                        ["ir_105", "natural_color_raw"])

    def peakmem_load(self, chunk, loadable):
        """Peak RAM to create a scene and load one channel or composite."""
        self.get_loaded_scene(chunk, loadable)
    peakmem_load.params = time_load.params

    def time_compute(self, chunk, loadable):
        """Time to create a scene and load and compute one channel."""
        sc = self.get_loaded_scene(chunk, loadable)
        sc[loadable].compute()
    time_compute.params = time_load.params

    def peakmem_compute(self, chunk, loadable):
        """Peak memory for creating a scene and loading and computing one channel."""
        sc = self.get_loaded_scene(chunk, loadable)
        sc[loadable].compute()
    peakmem_compute.params = time_compute.params

    def time_load_resample_compute(self, chunk, loadable, mode):
        """Time to load all chunks, resample, and compute."""
        ls = self.get_resampled_scene(
                chunk, loadable, self.region, mode)
        ls[loadable].compute()
    time_load_resample_compute.params = time_load.params + (
            ["nearest", "bilinear", "gradient_search"],)

    def peakmem_load_resample_compute(self, chunk, loadable, mode):
        """Peak memory to load all chunks, resample, and compute."""
        ls = self.get_resampled_scene(
                chunk, loadable, self.region, mode)
        ls[loadable].compute()
    peakmem_load_resample_compute.params = time_load_resample_compute.params

    def time_load_resample_save(self, chunk, loadable, mode):
        """Time to load all chunks, resample, and save."""
        self.load_resample_save(chunk, loadable, self.region, mode)
    time_load_resample_save.params = time_load_resample_compute.params

    def peakmem_load_resample_save(self, chunk, loadable, mode):
        """Peak memory to load all chunks, resample, and save."""
        self.load_resample_save(chunk, loadable, self.region, mode)
    peakmem_load_resample_save.params = time_load_resample_save.params

    def create_scene(self, selection):
        """Create a scene with FCI, and return it."""
        if selection == "some":
            names = fnmatch.filter(self.filenames, "*3[0123].nc")
        elif selection == "all":
            names = self.filenames
        else:
            raise ValueError("Expected selection some or all, got " +
                             selection)
        return satpy.Scene(filenames=names, reader="fci_l1c_nc")

    def get_loaded_scene(self, selection, loadable):
        """Return a FCI scene with a loaded channel or composite."""
        sc = self.create_scene(selection)
        sc.load([loadable])
        return sc

    def get_resampled_scene(self, selection, loadable, area, resampler):
        """Load and resample an FCI scene with a composite."""
        sc = self.get_loaded_scene(selection, loadable)
        # if I don't put this here, computing fails with RuntimeError: NetCDF:
        # Not a valid ID.  Apparently the original scene object gets destroyed
        # and garbage collected.  I can't reproduce this in a MCVE, but it
        # happens when running through asv.
        self._sc = sc
        return sc.resample(area, resampler=resampler)

    def load_resample_save(self, selection, loadable, area, resampler):
        """Load, resample, and save FCI scene with composite."""
        ls = self.get_resampled_scene(selection, loadable, area, resampler)
        ls.save_datasets()
