#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018-2021 Satpy developers
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
"""Step for the real load-process-write tests."""

import fnmatch
import os
from tempfile import NamedTemporaryFile

import numpy as np
from behave import given, then, when
from PIL import Image


def fft_proj_rms(a1, a2):
    """Compute the RMS of differences between two images.

    Compute the RMS of differences between two FFT vectors of a1
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
    l_factor = ps2 * np.cos(theta)
    ms += np.sum(((l_factor - ps1) ** 2)) / float(ps1.size)

    rms = np.sqrt(ms)

    return rms


def assert_images_match(image1, image2, threshold=0.1):
    """Assert that images are matching."""
    img1 = np.asarray(Image.open(image1))
    img2 = np.asarray(Image.open(image2))
    rms = fft_proj_rms(img1, img2)
    assert rms <= threshold, "Images {0} and {1} don't match: {2}".format(
        image1, image2, rms)


def get_all_files(directory, pattern):
    """Find all files matching *pattern* under ``directory``."""
    matches = []
    for root, _, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches


def before_all(context):
    """Enable satpy debugging."""
    if not context.config.log_capture:
        from satpy.utils import debug_on
        debug_on()


@given(u'{dformat} data is available')
def step_impl_input_files_exists(context, dformat):
    """Check that input data exists on disk."""
    data_path = os.path.join('test_data', dformat)
    data_available = os.path.exists(data_path)
    if not data_available:
        context.scenario.skip(reason="No test data available for " + dformat)
    else:
        context.dformat = dformat
        context.data_path = data_path


@when(u'the user loads the {composite} composite')
def step_impl_create_scene_and_load_single(context, composite):
    """Create a Scene and load a single composite."""
    from satpy import Scene
    scn = Scene(reader=context.dformat,
                filenames=get_all_files(os.path.join(context.data_path, 'data'),
                                        '*'))
    scn.load([composite])
    context.scn = scn
    context.composite = composite


@when(u'the user resamples the data to {area}')
def step_impl_resample_scene(context, area):
    """Resample the scene to an area or use the native resampler."""
    if area != '-':
        context.lscn = context.scn.resample(area)
    else:
        context.lscn = context.scn.resample(resampler='native')
    context.area = area


@when(u'the user saves the composite to disk')
def step_impl_save_to_png(context):
    """Call Scene.save_dataset to write a PNG image."""
    with NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        context.lscn.save_dataset(context.composite, filename=tmp_file.name)
        context.new_filename = tmp_file.name


@then(u'the resulting image should match the reference image')
def step_impl_compare_two_png_images(context):
    """Compare two PNG image files."""
    if context.area == '-':
        ref_filename = context.composite + ".png"
    else:
        ref_filename = context.composite + "_" + context.area + ".png"
    ref_filename = os.path.join(context.data_path, "ref", ref_filename)
    assert os.path.exists(ref_filename), "Missing reference file."
    assert_images_match(ref_filename, context.new_filename)
    os.remove(context.new_filename)
