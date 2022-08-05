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

    def create_scene(self, filenames=None):
        """Create a scene."""
        from satpy import Scene
        scn = Scene(filenames=filenames or self.data_files, reader=self.reader)
        return scn

    def load_no_padding(self, composite, filenames=None):
        """Load one composite or channel."""
        scn = self.create_scene(filenames=filenames)
        scn.load([composite], pad_data=False)
        return scn

    def load_and_native_resample(self, composite):
        """Load and native resample a composite or channel."""
        return self.load_and_resample(composite, "native")

    def load_and_resample(self, composite, resampler, area=None, filenames=None):
        """Load and resample a composite or channel with resampler and area."""
        scn = self.load_no_padding(composite, filenames=filenames)
        ls = scn.resample(area, resampler=resampler)
        ls._readers = scn._readers  # workaround for GH#1861
        return ls

    def compute_composite(self, composite, resampler="native",
                          area=None, filenames=None):
        """Compute a true color image."""
        lscn = self.load_and_resample(
                composite, resampler, area, filenames)
        lscn[composite].compute()

    def save_composite_as_geotiff(self, composite, resampler="native",
                                  area=None, filenames=None):
        """Save a composite to disk as geotiff."""
        lscn = self.load_and_resample(composite, resampler, area, filenames)
        lscn.save_dataset(composite, filename='test.tif', tiled=True)

    def compute_channel(self, channel, filenames=None):
        """Load and compute one channel."""
        scn = self.load_no_padding(channel, filenames=filenames)
        scn[channel].compute()
