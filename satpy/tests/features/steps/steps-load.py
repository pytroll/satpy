#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018 Satpy developers
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
"""Behaviour steps for loading."""

import os
from urllib.request import urlopen

from behave import given, then, use_step_matcher, when

use_step_matcher("re")


@given(u'data is available')
def step_impl_data_available(context):
    """Make data available."""
    if not os.path.exists('/tmp/SVM02_npp_d20150311_t1122204_e1123446_b17451_c20150311113206961730_cspp_dev.h5'):
        response = urlopen('https://zenodo.org/record/16355/files/'
                           'SVM02_npp_d20150311_t1122204_e1123446_b17451_c20150311113206961730_cspp_dev.h5')
        with open('/tmp/SVM02_npp_d20150311_t1122204_e1123446_b17451_c20150311113206961730_cspp_dev.h5',
                  mode="w") as fp:
            fp.write(response.read())
    if not os.path.exists('/tmp/GMTCO_npp_d20150311_t1122204_e1123446_b17451_c20150311113205873710_cspp_dev.h5'):
        response = urlopen('https://zenodo.org/record/16355/files/'
                           'GMTCO_npp_d20150311_t1122204_e1123446_b17451_c20150311113205873710_cspp_dev.h5')
        with open('/tmp/GMTCO_npp_d20150311_t1122204_e1123446_b17451_c20150311113205873710_cspp_dev.h5',
                  mode="w") as fp:
            fp.write(response.read())


@when(u'user loads the data without providing a config file')
def step_impl_user_loads_no_config(context):
    """Load the data without a config."""
    from datetime import datetime

    from satpy import Scene, find_files_and_readers
    os.chdir("/tmp/")
    readers_files = find_files_and_readers(sensor='viirs',
                                           start_time=datetime(2015, 3, 11, 11, 20),
                                           end_time=datetime(2015, 3, 11, 11, 26))
    scn = Scene(filenames=readers_files)
    scn.load(["M02"])
    context.scene = scn


@then(u'the data is available in a scene object')
def step_impl_data_available_in_scene(context):
    """Check that the data is available in the scene."""
    assert context.scene["M02"] is not None
    assert context.scene.get("M01") is None


@when(u'some items are not available')
def step_impl_items_not_available(context):
    """Load some data."""
    context.scene.load(["M01"])


@when(u'user wants to know what data is available')
def step_impl_user_checks_availability(context):
    """Check availability."""
    from datetime import datetime

    from satpy import Scene, find_files_and_readers
    os.chdir("/tmp/")
    reader_files = find_files_and_readers(sensor="viirs",
                                          start_time=datetime(2015, 3, 11, 11, 20),
                                          end_time=datetime(2015, 3, 11, 11, 26))
    scn = Scene(filenames=reader_files)
    context.available_dataset_ids = scn.available_dataset_ids()


@then(u'available datasets are returned')
def step_impl_available_datasets_are_returned(context):
    """Check that datasets are returned."""
    assert (len(context.available_dataset_ids) >= 5)


@given("datasets with the same name")
def step_impl_datasets_with_same_name(context):
    """Datasets with the same name but different other ID parameters."""
    from xarray import DataArray

    from satpy import Scene
    from satpy.tests.utils import make_dataid
    scn = Scene()
    scn[make_dataid(name='ds1', calibration='radiance')] = DataArray([[1, 2], [3, 4]])
    scn[make_dataid(name='ds1', resolution=500, calibration='reflectance')] = DataArray([[5, 6], [7, 8]])
    scn[make_dataid(name='ds1', resolution=250, calibration='reflectance')] = DataArray([[5, 6], [7, 8]])
    scn[make_dataid(name='ds1', resolution=1000, calibration='reflectance')] = DataArray([[5, 6], [7, 8]])
    scn[make_dataid(name='ds1', resolution=500, calibration='radiance', modifiers=('mod1',))] = \
        DataArray([[5, 6], [7, 8]])
    ds_id = make_dataid(name='ds1', resolution=1000, calibration='radiance', modifiers=('mod1', 'mod2'))
    scn[ds_id] = DataArray([[5, 6], [7, 8]])
    context.scene = scn


@when("a dataset is retrieved by name")
def step_impl_dataset_retrieved_by_name(context):
    """Use the Scene's getitem method to get a dataset."""
    context.returned_dataset = context.scene['ds1']


@then("the least modified version of the dataset is returned")
def step_impl_least_modified_dataset_returned(context):
    """Check that the dataset should be one of the least modified datasets."""
    assert len(context.returned_dataset.attrs['modifiers']) == 0
