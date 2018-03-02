from behave import given, when, then, use_step_matcher

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch


use_step_matcher("re")


@given("a dataset is available")  # noqa: F811
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    from satpy import Scene
    from xarray import DataArray
    scn = Scene()
    scn["MyDataset"] = DataArray([[1, 2], [3, 4]], dims=['y', 'x'])
    context.scene = scn


@when("the show command is called")  # noqa: F811
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    with patch('trollimage.xrimage.XRImage.show') as mock_show:
        context.scene.show("MyDataset")
        mock_show.assert_called_once_with()


@then("an image should pop up")  # noqa: F811
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    pass


@when("the save_dataset command is called")  # noqa: F811
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    context.filename = "/tmp/test_dataset.png"
    context.scene.save_dataset("MyDataset", context.filename)


@then("a file should be saved on disk")  # noqa: F811
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    import os
    assert(os.path.exists(context.filename))
    os.remove(context.filename)


@given("a bunch of datasets are available")  # noqa: F811
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    from satpy import Scene
    from xarray import DataArray
    scn = Scene()
    scn["MyDataset"] = DataArray([[1, 2], [3, 4]], dims=['y', 'x'])
    scn["MyDataset2"] = DataArray([[5, 6], [7, 8]], dims=['y', 'x'])
    context.scene = scn


@when("the save_datasets command is called")  # noqa: F811
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    context.scene.save_datasets(writer="simple_image", file_pattern="{name}.png")


@then("a bunch of files should be saved on disk")  # noqa: F811
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    import os
    for filename in ["MyDataset.png", "MyDataset2.png"]:
        assert(os.path.exists(filename))
        os.remove(filename)
