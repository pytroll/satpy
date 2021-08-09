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
"""Benchmark utilities."""

import os


def get_filenames(subdir):
    """Get the data filenames manually."""
    import glob
    base_dir = os.environ.get("SATPY_DEMO_DATA_DIR", ".")
    return glob.glob(os.path.join(base_dir, subdir, "*"))


class GeoBenchmarks:
    """Class for geo benchmarks."""

    def load_no_padding(self, composite):
        """Load one composite."""
        from satpy import Scene
        scn = Scene(filenames=self.data_files, reader=self.reader)
        scn.load([composite], pad_data=False)
        return scn
