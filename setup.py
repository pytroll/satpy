#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009-2015.

# SMHI,
# Folkborgsvägen 1,
# Norrköping,
# Sweden

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# mpop is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with mpop.  If not, see <http://www.gnu.org/licenses/>.

"""Setup file for mpop.
"""
import os.path
from setuptools import setup
import imp
import sys

version = imp.load_source('mpop.version', 'mpop/version.py')

BASE_PATH = os.path.sep.join(os.path.dirname(
    os.path.realpath(__file__)).split(os.path.sep))

requires = ['numpy >=1.4.1', 'pyresample', 'trollsift', 'trollimage', 'pykdtree', 'six']

if sys.version < '2.7':
    requires.append('ordereddict')


test_requires = []

if sys.version < '3.0':
    test_requires.append('mock')

try:
    from PIL import Image
except ImportError:
    requires.append("pillow")

NAME = 'mpop'

setup(name=NAME,
      version=version.__version__,
      description='Meteorological post processing package',
      author='Martin Raspaud',
      author_email='martin.raspaud@smhi.se',
      classifiers=["Development Status :: 5 - Production/Stable",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: GNU General Public License v3 " +
                   "or later (GPLv3+)",
                   "Operating System :: OS Independent",
                   "Programming Language :: Python",
                   "Topic :: Scientific/Engineering"],
      url="https://github.com/mraspaud/mpop",
      test_suite='mpop.tests.suite',
      packages=['mpop', 'mpop.instruments', 'mpop.readers',
                'mpop.satout', 'mpop.imageo', 'mpop.tests',
                'mpop.imageo.formats'],
      data_files=[(os.path.join('etc', 'pytroll'),
                   [os.path.join('etc', 'geo_image.cfg'),
                    os.path.join('etc', 'eps_avhrrl1b_6.5.xml')])],
      zip_safe=False,
      install_requires=requires,
      tests_require=test_requires,
      extras_require={'xRIT': ['mipp >= 0.6.0'],
                      'hdf_eos': ['pyhdf'],
                      'viirs': ['h5py'],
                      'nc': ['netCDF4'],
                      'proj': ['pyresample'],
                      'pyspectral': ['pyspectral'],
                      'pyorbital': ['pyorbital >= v0.2.3']}

      )
