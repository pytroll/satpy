from behave import *

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch


use_step_matcher("re")


@given("a dataset is available")
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    from satpy.scene import Scene
    from datetime import datetime
    from satpy.dataset import Dataset
    scn = Scene(platform_name="Suomi-NPP", sensor="viirs",
                start_time=datetime(2015, 3, 11, 11, 20),
                end_time=datetime(2015, 3, 11, 11, 26))
    scn["MyDataset"] = Dataset([[1, 2], [3, 4]])
    context.scene = scn


@when("the show command is called")
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    with patch('trollimage.image.Image.show') as mock_show:
        context.scene.show("MyDataset")
        mock_show.assert_called_once_with()


@then("an image should pop up")
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    pass


@when("the save_dataset command is called")
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    context.filename = "/tmp/test_dataset.png"
    context.scene.save_dataset("MyDataset", context.filename)


@then("a file should be saved on disk")
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    import os
    assert(os.path.exists(context.filename))
    os.remove(context.filename)


@given("a bunch of datasets are available")
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    from satpy.scene import Scene
    from datetime import datetime
    from satpy.dataset import Dataset
    scn = Scene(platform_name="Suomi-NPP", sensor="viirs",
                start_time=datetime(2015, 3, 11, 11, 20),
                end_time=datetime(2015, 3, 11, 11, 26))
    scn["MyDataset"] = Dataset([[1, 2], [3, 4]])
    scn["MyDataset2"] = Dataset([[5, 6], [7, 8]])
    context.scene = scn



@when("the save_datasets command is called")
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    context.scene.save_datasets(writer="simple_image", file_pattern="{name}.png")



@then("a bunch of files should be saved on disk")
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    import os
    for filename in ["MyDataset.png", "MyDataset2.png"]:
        assert(os.path.exists(filename))
        os.remove(filename)
