# Copyright (c) 2024 Satpy developers
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

"""Script to create image testing references.

Script to create reference images for the automated image testing system.

create_reference.py <input-data> <output-directory> <satellite-name>

The input data directory must follow the data structure from the
image-comparison-tests repository with satellite_data/<satellite-name>.

This script is a work in progress and expected to change significantly.
It is absolutely not intended for any operational production of satellite
imagery.
"""

import sys
from glob import glob

from dask.diagnostics import ProgressBar

from satpy import Scene

ext_data_path = sys.argv[1]
outdir = sys.argv[2]
satellite = sys.argv[3]

filenames = glob(f"{ext_data_path}/satellite_data/{satellite}/*.nc")

scn = Scene(reader="abi_l1b", filenames=filenames)

composites = ["ash", "airmass"]
scn.load(composites)
ls = scn.resample(resampler="native")
with ProgressBar():
    ls.save_datasets(writer="simple_image", filename=outdir +
                      "/satpy-reference-image-{platform_name}-{sensor}-{start_time:%Y%m%d%H%M}-{area.area_id}-{name}.png")
