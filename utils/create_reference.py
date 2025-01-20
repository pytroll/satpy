# Copyright (c) 2024-2025 Satpy developers
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

The input data directory must follow the data structure from the
image-comparison-tests repository with satellite_data/<satellite-name>.

This script is a work in progress and expected to change significantly.

DO NOT USE FOR OPERATIONAL PRODUCTION!
"""

import argparse
import os
import pathlib

import hdf5plugin  # noqa: F401

from satpy import Scene


def generate_images(props):
    """Generate reference images for testing purposes.

    Args:
        props (namespace): Object with attributes corresponding to command line
        arguments as defined by :func:get_parser.
    """
    filenames = (props.basedir / "satellite_data" / props.satellite /
                 props.case).glob("*")

    scn = Scene(reader=props.reader, filenames=filenames)

    scn.load(props.composites)
    if props.area == "native":
        ls = scn.resample(resampler="native")
    elif props.area is not None:
        ls = scn.resample(props.area, resampler="gradient_search")
    else:
        ls = scn

    from dask.diagnostics import ProgressBar
    with ProgressBar():
        ls.save_datasets(
                writer="simple_image",
                filename=os.fspath(
                    props.basedir / "reference_images" /
                    "satpy-reference-image-{platform_name}-{sensor}-"
                    "{start_time:%Y%m%d%H%M}-{area.area_id}-{name}.png"))

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
            "case", help="case to generate", type=str)

    parser.add_argument(
            "-b", "--basedir", action="store", type=pathlib.Path,
            default=pathlib.Path("."),
            help="Base directory for reference data. "
                 "This must contain a subdirectories satellite_data and "
                 "reference_images.  The directory satellite_data must contain "
                 "input data in a subdirectory for the satellite and case. Output images "
                 "will be written to the subdirectory reference_images.")

    parser.add_argument(
            "-c", "--composites", nargs="+", help="composites to generate",
            type=str, default=["ash", "airmass"])

    parser.add_argument(
            "-a", "--area", action="store",
            default=None,
            help="Area name, or 'native' (native resampling)")

    return parser

def main():
    """Main function."""
    parsed = get_parser().parse_args()

    generate_images(parsed)

if __name__ == "__main__":
    main()
