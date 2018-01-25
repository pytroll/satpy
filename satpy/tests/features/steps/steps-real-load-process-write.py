#!/usr/bin/python
# Copyright (c) 2018 Pytroll developpers.
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
from glob import glob

from behave import use_step_matcher, given, when, then



import os
from contextlib import contextmanager
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image


def fft_proj_rms(a1, a2):
    """Compute the RMS of differences between FFT vectors of a1
    and projection of FFT vectors of a2.
    This metric is sensitive to large scale changes and image noise but
    insensitive to small rendering differences.
    """

    ms = 0

    # for i in range(a1.shape[-1]):
    fr1 = np.fft.rfftn(a1)
    fr2 = np.fft.rfftn(a2)

    ps1 = np.log10(fr1 * fr1.conj()).real
    ps2 = np.log10(fr2 * fr2.conj()).real

    p1 = np.arctan2(fr1.imag, fr1.real)
    p2 = np.arctan2(fr2.imag, fr2.real)

    theta = p2 - p1
    l = ps2 * np.cos(theta)
    ms += np.sum(((l - ps1) ** 2)) / float(ps1.size)

    rms = np.sqrt(ms)

    return rms


def fft_metric(data1, data2, max_value=0.1):
    """Execute FFT metric
    """

    rms = fft_proj_rms(data1, data2)
    return rms <= max_value


def assert_images_match(image1, image2, threshold=0.1):
    img1 = np.asarray(Image.open(image1))
    img2 = np.asarray(Image.open(image2))
    rms = fft_proj_rms(img1, img2)
    assert rms <= threshold, "Images {0} and {1} don't match: {2}".format(
        image1, image2, rms)

@given(u'{dformat} data is available')
def step_impl(context, dformat):
    data_path = os.path.join('test_data', dformat)
    data_available = os.path.exists(data_path)
    if not data_available:
        context.scenario.skip(reason="No {dformat} test data available".format(dformat=dformat))
    else:
        context.dformat = dformat
        context.data_path = data_path

@when(u'the user loads the {composite} composite')
def step_impl(context, composite):
    from satpy import Scene
    scn = Scene(reader=context.dformat,
                filenames=glob(os.path.join(context.data_path, 'data', '*')))
    scn.load([composite])
    context.scn = scn
    context.composite = composite

@when(u'the user resamples the data to {area}')
def step_impl(context, area):
    context.lscn = context.scn.resample(area)
    context.area = area

@when(u'the user saves the composite to disk')
def step_impl(context):
    with NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        context.lscn.save_dataset(context.composite, filename=tmp_file.name)
        context.new_filename = tmp_file.name

@then(u'the resulting image should match the reference image')
def step_impl(context):
    ref_filename = context.composite + "_" + context.area + ".png"
    ref_filename = os.path.join(context.data_path, "ref", ref_filename)
    assert(os.path.exists(ref_filename), "Missing reference file.")
    assert_images_match(ref_filename, context.new_filename)

def after_scenario(context, scenario):
    if hasattr(context, 'new_filename'):
        os.remove(context.new_filename)
