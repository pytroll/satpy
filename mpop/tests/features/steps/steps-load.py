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

# @given(u'data is available')
# def step_impl(context):
#     assert False
#
# @when(u'user loads the data without providing a config file')
# def step_impl(context):
#     assert False
#
# @then(u'scene is returned')
# def step_impl(context):
#     assert False
#
# @when(u'some items are no available')
# def step_impl(context):
#     assert False
#



@given(u'data is available')
def step_impl(context):
    import urllib2
    if not os.path.exists('/tmp/SVM02_npp_d20150311_t1122204_e1123446_b17451_c20150311113206961730_cspp_dev.h5'):
        response = urllib2.urlopen('https://zenodo.org/record/16355/files/SVM02_npp_d20150311_t1122204_e1123446_b17451_c20150311113206961730_cspp_dev.h5')
        with open('/tmp/SVM02_npp_d20150311_t1122204_e1123446_b17451_c20150311113206961730_cspp_dev.h5', mode="w") as fp:
            fp.write(response.read())
    if not os.path.exists('/tmp/GMTCO_npp_d20150311_t1122204_e1123446_b17451_c20150311113205873710_cspp_dev.h5'):
        response = urllib2.urlopen('https://zenodo.org/record/16355/files/GMTCO_npp_d20150311_t1122204_e1123446_b17451_c20150311113205873710_cspp_dev.h5')
        with open('/tmp/GMTCO_npp_d20150311_t1122204_e1123446_b17451_c20150311113205873710_cspp_dev.h5', mode="w") as fp:
            fp.write(response.read())


@when(u'user loads the data without providing a config file')
def step_impl(context):
    from mpop.scene import Scene
    from datetime import datetime
    os.chdir("/tmp/")
    scn = Scene(platform_name="Suomi-NPP", sensor="viirs",
                start_time=datetime(2015, 3, 11, 11, 20),
                end_time=datetime(2015, 3, 11, 11, 26))
    scn.load(["M02"])
    context.scene = scn

@then(u'the data is available in a scene object')
def step_impl(context):
    assert(context.scene["M02"] is not None)
    try:
        context.scene["M01"] is None
        assert(False)
    except KeyError:
        assert(True)

@when(u'some items are no available')
def step_impl(context):
    context.scene.load(["M01"])

#@when(u'user wants to know what data is available')
#def step_impl(context):
#    raise NotImplementedError(u'STEP: When user wants to know what data is available')

#@then(u'available datasets is returned')
#def step_impl(context):
#    raise NotImplementedError(u'STEP: Then available datasets is returned')
