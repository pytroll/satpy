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
"""

import fnmatch
import os

import satpy
import satpy.demo


class FCI:
    """Benchmark FCI FDHSI test data reading."""

    timeout = 600
    data_files = []

    def setup(self):
        """Fetch the data files."""
        fns = self.get_filenames()
        cnt = len(fns)
        if cnt > 41:
            raise ValueError(f"Expected 41 files, found {cnt:d}")
        elif cnt < 41:
            fns = satpy.demo.download_fci_test_data()
        self.filenames = fnmatch.filter(fns, "*-CHK-BODY-*")

    def get_filenames(self):
        """Get filenames of FCI test data as already available."""
        p = satpy.demo._get_fci_test_data_dir()
        g = p.glob("UNCOMPRESSED/NOMINAL/*.nc")
        return [os.fspath(fn) for fn in g]

    def time_create_scene_single_chunk(self):
        """Time to create a scene with a single chunk."""
        self.create_scene(slice(20, 21))

    def time_create_scene_all_chunks(self):
        """Time to create a scene with all chunks."""
        self.create_scene(slice(None))

    def peakmem_create_scene_single_chunk(self):
        """Peak memory usage to create a scene with a single chunk."""
        self.create_scene(slice(20, 21))

    def peakmem_create_scene_all_chunks(self):
        """Peak memory usage of creating a scene with all chunks."""
        self.create_scene(slice(None))

    def time_load_channel(self):
        """Time to create a scene and load one channel."""
        self.load_channel()

    def peakmem_load_channel(self):
        """Peak memory for creating a scene and loading one channel."""
        self.load_channel()

    def time_compute_channel(self):
        """Time to create a scene and load and compute one channel."""
        sc = self.load_channel()
        sc["ir_105"].compute()

    def peakmem_compute_channel(self):
        """Peak memory for creating a scene and loading and computing one channel."""
        sc = self.load_channel()
        sc["ir_105"].compute()

    def time_load_composite(self):
        """Time to create a scene and load a composite."""
        self.load_composite()

    def peakmem_load_composite(self):
        """Peak memory to create a scene and load a composite."""
        self.load_composite()

    def time_load_resample_save(self):
        """Time to load all chunks, resample, and save."""
        self.load_resample_save()

    def create_scene(self, selection):
        """Create a scene with FCI, and return it."""
        return satpy.Scene(filenames=self.filenames[selection], reader="fci_l1c_nc")

    def load_channel(self):
        """Return a FCI scene with a loaded channel."""
        sc = self.create_scene(slice(None))
        sc.load(["ir_105"])
        return sc

    def load_composite(self):
        """Return a FCI scene with a loaded composite."""
        sc = self.create_scene(slice(None))
        sc.load(["natural_color_raw"])
        return sc

    def load_resample_save(self):
        """Load, resample, and save FCI scene with composite."""
        sc = self.load_composite()
        ls = sc.resample("eurol")
        ls.save_datasets()
