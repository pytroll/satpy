"""Behave steps related to saving or showing datasets."""

from unittest.mock import patch

from behave import given, then, use_step_matcher, when

use_step_matcher("re")


@given("a dataset is available")
def step_impl_create_scene_one_dataset(context):
    """Create a Scene with a fake dataset for testing.

    Args:
        context (behave.runner.Context): Test context

    """
    from xarray import DataArray

    from satpy import Scene
    scn = Scene()
    scn["MyDataset"] = DataArray([[1, 2], [3, 4]], dims=["y", "x"])
    context.scene = scn


@when("the show command is called")
def step_impl_scene_show(context):
    """Call the Scene.show method.

    Args:
        context (behave.runner.Context): Test context

    """
    with patch("trollimage.xrimage.XRImage.show") as mock_show:
        context.scene.show("MyDataset")
        mock_show.assert_called_once_with()


@then("an image should pop up")
def step_impl_image_pop_up(context):
    """Check that a image window pops up (no-op currently).

    Args:
        context (behave.runner.Context): Test context

    """


@when("the save_dataset command is called")
def step_impl_save_dataset_to_png(context):
    """Run Scene.save_dataset to create a PNG image.

    Args:
        context (behave.runner.Context): Test context

    """
    context.filename = "/tmp/test_dataset.png"
    context.scene.save_dataset("MyDataset", context.filename)


@then("a file should be saved on disk")
def step_impl_file_exists_and_remove(context):
    """Check that a file exists on disk and then remove it.

    Args:
        context (behave.runner.Context): Test context

    """
    import os
    assert os.path.exists(context.filename)
    os.remove(context.filename)


@given("a bunch of datasets are available")
def step_impl_create_scene_two_datasets(context):
    """Create a Scene with two fake datasets for testing.

    Args:
        context (behave.runner.Context): Test context

    """
    from xarray import DataArray

    from satpy import Scene
    scn = Scene()
    scn["MyDataset"] = DataArray([[1, 2], [3, 4]], dims=["y", "x"])
    scn["MyDataset2"] = DataArray([[5, 6], [7, 8]], dims=["y", "x"])
    context.scene = scn


@when("the save_datasets command is called")
def step_impl_save_datasets(context):
    """Run Scene.save_datsets to create PNG images.

    Args:
        context (behave.runner.Context): Test context

    """
    context.scene.save_datasets(writer="simple_image", filename="{name}.png")


@then("a bunch of files should be saved on disk")
def step_impl_check_two_pngs_exist(context):
    """Check that two PNGs exist.

    Args:
        context (behave.runner.Context): Test context

    """
    import os
    for filename in ["MyDataset.png", "MyDataset2.png"]:
        assert os.path.exists(filename)
        os.remove(filename)
