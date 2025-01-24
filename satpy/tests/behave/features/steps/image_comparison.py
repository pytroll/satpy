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
"""Image comparison tests."""

import hdf5plugin  # noqa: F401  isort:skip
import os
import os.path
import warnings
from datetime import datetime
from glob import glob

import cv2
import dask
import numpy as np
from behave import given, then, when

import satpy
from satpy import Scene

ext_data_path = "/app/ext_data"
threshold = 2000

def before_all(context):
    """Define a before_all hook to create the timestamp and test results directory."""
    tm = datetime.now()
    context.timestamp = tm.strftime("%Y-%m-%d-%H-%M-%S")
    context.test_results_dir = f"{ext_data_path}/test_results/image_comparison/{context.timestamp}"
    os.makedirs(os.path.join(context.test_results_dir, "generated"), exist_ok=True)
    os.makedirs(os.path.join(context.test_results_dir, "difference"), exist_ok=True)

    # Write the timestamp to test_results.txt
    results_file = os.path.join(context.test_results_dir, "test_results.txt")
    with open(results_file, "a") as f:
        f.write(f"Test executed at {context.timestamp}.\n\n")

def setup_hooks():
    """Register the before_all hook."""
    from behave import use_fixture
    from behave.runner import Context

    use_fixture(before_all, Context)

setup_hooks()
@given("I have a {composite} reference image file from {satellite} resampled to {area}")
def step_given_reference_image(context, composite, satellite, area):
    """Prepare a reference image."""
    reference_image = f"satpy-reference-image-{satellite}-{composite}-{area}.png"
    context.reference_image = cv2.imread(f"{ext_data_path}/reference_images/{reference_image}")
    context.satellite = satellite
    context.composite = composite
    context.area = area


@when("I generate a new {composite} image file from {satellite} case {case} "
      "with {reader} for {area} resampling with {resampler} with clipping {clip}")
def step_when_generate_image(context, composite, satellite, case, reader, area, resampler, clip):
    """Generate test images."""
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["PYTROLL_CHUNK_SIZE"] = "1024"
    warnings.simplefilter("ignore")
    dask.config.set(scheduler="threads", num_workers=4)

    # Get the list of satellite files to open
    filenames = glob(f"{ext_data_path}/satellite_data/{satellite}/{case}/*.nc")


    if "," in reader:
        reader = reader.split(",")
    with satpy.config.set({"readers.clip_negative_radiances": False if clip == "null" else clip}):
        scn = Scene(reader=reader, filenames=filenames)
        scn.load([composite])

    if area == "null":
        ls = scn
    else:
        ls = scn.resample(area, resampler=resampler)

    # Save the generated image in the generated folder
    generated_image_path = os.path.join(context.test_results_dir, "generated",
                                        f"generated_{context.satellite}_{context.composite}_{context.area}.png")
    ls.save_datasets(writer="simple_image", filename=generated_image_path)

    # Save the generated image in the context
    context.generated_image = cv2.imread(generated_image_path)


@then("the generated image should be the same as the reference image")
def step_then_compare_images(context):
    """Compare test image to reference image."""
    # Load the images
    imageA = cv2.cvtColor(context.reference_image, cv2.COLOR_BGR2GRAY)
    imageB = cv2.cvtColor(context.generated_image, cv2.COLOR_BGR2GRAY)
    # Ensure both images have the same dimensions
    if imageA.shape != imageB.shape:
        raise ValueError("Both images must have the same dimensions")
    array1 = np.array(imageA)
    array2 = np.array(imageB)
    # Perform pixel-wise comparison
    result_matrix = (array1 != array2).astype(np.uint8) * 255

    # Save the resulting numpy array as an image in the difference folder
    diff_image_path = os.path.join(context.test_results_dir, "difference",
                                   f"diff_{context.satellite}_{context.composite}.png")
    cv2.imwrite(diff_image_path, result_matrix)

    # Count non-zero pixels in the result matrix
    non_zero_count = np.count_nonzero(result_matrix)

    # Write the results to a file in the test results directory
    results_file = os.path.join(context.test_results_dir, "test_results.txt")
    with open(results_file, "a") as f:
        f.write(f"Test for {context.satellite} - {context.composite}\n")
        f.write(f"Non-zero pixel differences: {non_zero_count}\n")
        if non_zero_count < threshold:
            f.write(f"Result: Passed - {non_zero_count} pixel differences.\n\n")
        else:
            f.write(f"Result: Failed - {non_zero_count} pixel differences exceed the threshold of {threshold}.\n\n")

    # Assert that the number of differences is below the threshold
    assert non_zero_count < threshold, (f"Images are not similar enough. "
            f"{non_zero_count} pixel differences exceed the threshold of "
            f"{threshold}.")
