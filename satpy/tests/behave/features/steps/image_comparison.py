import os
import warnings
from glob import glob
from PIL import Image
import cv2
import dask
import numpy as np
from behave import given, when, then
from satpy import Scene
from datetime import datetime
import pytz

ext_data_path = "/app/ext_data"
#ext_data_path = "/home/bildabgleich/pytroll-image-comparison-tests/data"
threshold = 2000

# Define a before_all hook to create the timestamp and test results directory
def before_all(context):
    berlin_time = datetime.now(pytz.timezone('Europe/Berlin'))
    context.timestamp = berlin_time.strftime("%Y-%m-%d-%H-%M-%S")
    context.test_results_dir = f"{ext_data_path}/test_results/image_comparison/{context.timestamp}"
    os.makedirs(os.path.join(context.test_results_dir, 'generated'), exist_ok=True)
    os.makedirs(os.path.join(context.test_results_dir, 'difference'), exist_ok=True)

    # Write the timestamp to test_results.txt
    results_file = os.path.join(context.test_results_dir, 'test_results.txt')
    with open(results_file, 'a') as f:
        f.write(f"Test executed at {context.timestamp}.\n\n")

# Register the before_all hook
def setup_hooks():
    from behave import use_fixture
    from behave.runner import Context

    use_fixture(before_all, Context)

setup_hooks()
@given('I have a {composite} reference image file from {satellite}')
def step_given_reference_image(context, composite, satellite):
    reference_image = f"reference_image_{satellite}_{composite}.png"
    context.reference_image = cv2.imread(f"./features/data/reference/{reference_image}")
    context.reference_different_image = cv2.imread(f"./features/data/reference_different/{reference_image}")
    context.satellite = satellite
    context.composite = composite


@when('I generate a new {composite} image file from {satellite}')
def step_when_generate_image(context, composite, satellite):
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['PYTROLL_CHUNK_SIZE'] = '1024'
    warnings.simplefilter('ignore')
    dask.config.set(scheduler='threads', num_workers=4)

    # Get the list of satellite files to open
    filenames = glob(f'{ext_data_path}/satellite_data/{satellite}/*.nc')

    scn = Scene(reader='abi_l1b', filenames=filenames)

    scn.load([composite])

    # Save the generated image in the generated folder
    generated_image_path = os.path.join(context.test_results_dir, 'generated',
                                        f'generated_{context.satellite}_{context.composite}.png')
    scn.save_datasets(writer='simple_image', filename=generated_image_path)

    # Save the generated image in the context
    context.generated_image = cv2.imread(generated_image_path)


@then('the generated image should be the same as the reference image')
def step_then_compare_images(context):
    # Load the images
    imageA = cv2.cvtColor(context.reference_image, cv2.COLOR_BGR2GRAY) # reference_different_image for testing only
    imageB = cv2.cvtColor(context.generated_image, cv2.COLOR_BGR2GRAY)
    # Ensure both images have the same dimensions
    if imageA.shape != imageB.shape:
        raise ValueError("Both images must have the same dimensions")
    array1 = np.array(imageA)
    array2 = np.array(imageB)
    # Perform pixel-wise comparison
    result_matrix = (array1 != array2).astype(np.uint8) * 255

    # Save the resulting numpy array as an image in the difference folder
    diff_image_path = os.path.join(context.test_results_dir, 'difference',
                                   f'diff_{context.satellite}_{context.composite}.png')
    cv2.imwrite(diff_image_path, result_matrix)

    # Count non-zero pixels in the result matrix
    non_zero_count = np.count_nonzero(result_matrix)

    # Write the results to a file in the test results directory
    results_file = os.path.join(context.test_results_dir, 'test_results.txt')
    with open(results_file, 'a') as f:
        f.write(f"Test for {context.satellite} - {context.composite}\n")
        f.write(f"Non-zero pixel differences: {non_zero_count}\n")
        if non_zero_count < threshold:
            f.write(f"Result: Passed - {non_zero_count} pixel differences.\n\n")
        else:
            f.write(f"Result: Failed - {non_zero_count} pixel differences exceed the threshold of {threshold}.\n\n")

    # Assert that the number of differences is below the threshold
    assert non_zero_count < threshold, f"Images are not similar enough. {non_zero_count} pixel differences exceed the threshold of {threshold}."
