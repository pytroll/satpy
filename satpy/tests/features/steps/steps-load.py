#!/usr/bin/python
# Copyright (c) 2015.
#

# Author(s):
#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""
"""

import os
import sys
from behave import use_step_matcher, given, when, then

if sys.version_info < (3, 0):
    from urllib2 import urlopen
else:
    from urllib.request import urlopen

use_step_matcher("re")


@given(u'data is available')
def step_impl(context):
    if not os.path.exists('/tmp/SVM02_npp_d20150311_t1122204_e1123446_b17451_c20150311113206961730_cspp_dev.h5'):
        response = urlopen(
            'https://zenodo.org/record/16355/files/SVM02_npp_d20150311_t1122204_e1123446_b17451_c20150311113206961730_cspp_dev.h5')
        with open('/tmp/SVM02_npp_d20150311_t1122204_e1123446_b17451_c20150311113206961730_cspp_dev.h5',
                  mode="w") as fp:
            fp.write(response.read())
    if not os.path.exists('/tmp/GMTCO_npp_d20150311_t1122204_e1123446_b17451_c20150311113205873710_cspp_dev.h5'):
        response = urlopen(
            'https://zenodo.org/record/16355/files/GMTCO_npp_d20150311_t1122204_e1123446_b17451_c20150311113205873710_cspp_dev.h5')
        with open('/tmp/GMTCO_npp_d20150311_t1122204_e1123446_b17451_c20150311113205873710_cspp_dev.h5',
                  mode="w") as fp:
            fp.write(response.read())


@when(u'user loads the data without providing a config file')
def step_impl(context):
    from satpy.scene import Scene
    from datetime import datetime
    os.chdir("/tmp/")
    scn = Scene(platform_name="Suomi-NPP", sensor="viirs",
                start_time=datetime(2015, 3, 11, 11, 20),
                end_time=datetime(2015, 3, 11, 11, 26))
    scn.load(["M02"])
    context.scene = scn


@then(u'the data is available in a scene object')
def step_impl(context):
    assert (context.scene["M02"] is not None)
    try:
        context.scene["M01"] is None
        assert (False)
    except KeyError:
        assert (True)


@when(u'some items are not available')
def step_impl(context):
    context.scene.load(["M01"])


@when(u'user wants to know what data is available')
def step_impl(context):
    from satpy.scene import Scene
    from datetime import datetime
    os.chdir("/tmp/")
    scn = Scene(platform_name="Suomi-NPP", sensor="viirs",
                start_time=datetime(2015, 3, 11, 11, 20),
                end_time=datetime(2015, 3, 11, 11, 26))
    context.available_dataset_ids = scn.available_dataset_ids()


@then(u'available datasets is returned')
def step_impl(context):
    assert (len(context.available_dataset_ids) >= 5)

@given("datasets with the same name")
def step_impl(context):
    """Datasets with the same name but different other ID parameters"""
    from satpy.scene import Scene
    from satpy.dataset import Dataset, DatasetID
    scn = Scene()
    scn[DatasetID('ds1', calibration='radiance')] = Dataset([[1, 2], [3, 4]])
    scn[DatasetID('ds1', resolution=500, calibration='reflectance')] = Dataset([[5, 6], [7, 8]])
    scn[DatasetID('ds1', resolution=250, calibration='reflectance')] = Dataset([[5, 6], [7, 8]])
    scn[DatasetID('ds1', resolution=1000, calibration='reflectance')] = Dataset([[5, 6], [7, 8]])
    scn[DatasetID('ds1', resolution=500, calibration='radiance', modifiers=('mod1',))] = Dataset([[5, 6], [7, 8]])
    scn[DatasetID('ds1', resolution=1000, calibration='radiance', modifiers=('mod1', 'mod2'))] = Dataset([[5, 6], [7, 8]])
    context.scene = scn

@when("a dataset is retrieved by name")
def step_impl(context):
    """Use the Scene's getitem method to get a dataset"""
    context.returned_dataset = context.scene['ds1']

@then("the least modified version of the dataset is returned")
def step_impl(context):
    """The dataset should be one of the least modified datasets"""
    assert(len(context.returned_dataset.id.modifiers) == 0)
