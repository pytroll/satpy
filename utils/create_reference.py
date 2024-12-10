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

import argparse
import pathlib
from glob import glob

from satpy import Scene


def generate_images(reader, filenames, area, composites, outdir):
    """Generate reference images for testing purposes."""
    from dask.diagnostics import ProgressBar
    scn = Scene(reader="abi_l1b", filenames=filenames)

    composites = ["ash", "airmass"]
    scn.load(composites)
    if area is None:
        ls = scn
    elif area == "native":
        ls = scn.resample(resampler="native")
    else:
        ls = scn.resample(area)

    with ProgressBar():
        ls.save_datasets(writer="simple_image", filename=outdir +
                          "/satpy-reference-image-{platform_name}-{sensor}-{start_time:%Y%m%d%H%M}-{area.area_id}-{name}.png")

def get_parser():
    """Return argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
            "satellite", action="store", type=str,
            help="Satellite name.")

    parser.add_argument(
            "reader", action="store", type=str,
            help="Reader name.")

    parser.add_argument(
            "area", action="store", type=str,
            help="Area name, 'null' (no resampling) or 'native' (native resampling)")

    parser.add_argument(
            "basedir", action="store", type=pathlib.Path,
            help="Root directory where reference input data are contained.")

    parser.add_argument(
            "outdir", action="store", type=pathlib.Path,
            help="Directory where to write resulting images.")

    return parser

def main():
    """Main function."""
    parsed = get_parser().parse_args()
    ext_data_path = parsed.basedir
    reader = parsed.reader
    area = parsed.area
    outdir = parsed.outdir
    satellite = parsed.satellite

    filenames = glob(f"{ext_data_path}/satellite_data/{satellite}/*")
    generate_images(reader, filenames, None if area.lower() == "null" else
                    area, ["airmass", "ash"], outdir)

if __name__ == "__main__":
    main()
