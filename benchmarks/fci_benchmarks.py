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

import satpy
import satpy.demo


class FCI:
    """Benchmark FCI FDHSI test data reading."""

    timeout = 600
    data_files = []

    def setup_cache(self):
        """Fetch the data files."""
        fns = self.get_filenames()
        cnt = len(fns)
        if cnt == 41:
            self.filenames = fns
        elif cnt < 41:
            self.filenames = satpy.demo.download_fci_test_data()
        else:
            raise ValueError(f"Expected 41 files, found {cnt:d}")

    def get_filenames(self):
        """Get filenames of FCI test data as already available."""
        p = satpy.demo._get_fci_test_data_dir()
        g = p.glob("UNCOMPRESSED/NOMINAL/*.nc")
        return list(g)

    def time_create_scene(self):
        """Time to create a scene with all chunks."""
        self.create_scene()

    def peakmem_create_scene(self):
        """Peak memory usage of creating a scene."""
        self.create_scene()

    def time_load_channel(self):
        """Time to create a scene and load one channel."""
        self.load_channel()

    def peakmem_load_channel(self):
        """Peak memory for creating a scene and loading one channel."""
        self.load_channel()

    def time_load_composite(self):
        """Time to create a scene and load a composite."""
        self.load_composite()

    def peakmem_load_composite(self):
        """Peak memory to create a scene and load a composite."""
        self.load_composite()

    def time_load_resample_save(self):
        """Time to load, resample, and save."""
        self.load_resample_save()

    def create_scene(self):
        """Create a scene with FCI, and return it."""
        return satpy.Scene(filenames=self.filenames, reader="fci_l1c_nc")

    def load_channel(self):
        """Return a FCI scene with a loaded channel."""
        sc = self.create_scene()
        sc.load(["ir_105"])

    def load_composite(self):
        """Return a FCI scene with a loaded composite."""
        sc = self.create_scene()
        sc.load(["overview"])
        return sc

    def load_resample_save(self):
        """Load, resample, and save FCI scene with composite."""
        sc = self.load_composite()
        ls = sc.resample("eurol")
        ls.save_datasets()
