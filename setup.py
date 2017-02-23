#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009-2016.

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>
#   David Hoese <david.hoese@ssec.wisc.edu>

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# satpy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Setup file for satpy."""

import imp
import os.path
import sys
from glob import glob

from setuptools import setup

version = imp.load_source('satpy.version', 'satpy/version.py')

BASE_PATH = os.path.sep.join(os.path.dirname(os.path.realpath(__file__)).split(
    os.path.sep))

requires = ['numpy >=1.4.1', 'pillow', 'pyresample', 'trollsift', 'trollimage',
            'pykdtree', 'six', 'pyyaml']

if sys.version < '2.7':
    requires.append('ordereddict')

test_requires = ['behave']

if sys.version < '3.0':
    test_requires.append('mock')


def _config_data_files(base_dirs, extensions=(".cfg", )):
    """Find all subdirectory configuration files.

    Searches each base directory relative to this setup.py file and finds
    all files ending in the extensions provided.

    :param base_dirs: iterable of relative base directories to search
    :param extensions: iterable of file extensions to include (with '.' prefix)
    :returns: list of 2-element tuples compatible with `setuptools.setup`
    """
    data_files = []
    pkg_root = os.path.realpath(os.path.dirname(__file__)) + "/"
    for base_dir in base_dirs:
        new_data_files = []
        for ext in extensions:
            configs = glob(os.path.join(pkg_root, base_dir, "*" + ext))
            configs = [c.replace(pkg_root, "") for c in configs]
            new_data_files.extend(configs)
        data_files.append((base_dir, new_data_files))

    return data_files


NAME = 'satpy'

setup(name=NAME,
      version=version.__version__,
      description='Meteorological post processing package',
      author='The Pytroll Team',
      author_email='pytroll@googlegroups.com',
      classifiers=["Development Status :: 5 - Production/Stable",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: GNU General Public License v3 " +
                   "or later (GPLv3+)",
                   "Operating System :: OS Independent",
                   "Programming Language :: Python",
                   "Topic :: Scientific/Engineering"],
      url="https://github.com/pytroll/satpy",
      test_suite='satpy.tests.suite',
      packages=['satpy', 'satpy.composites', 'satpy.readers',
                'satpy.writers', 'satpy.tests'],
      package_data={'satpy': [os.path.join('etc', 'geo_image.cfg'),
                              os.path.join('etc', 'areas.def'),
                              os.path.join('etc', 'satpy.cfg'),
                              os.path.join('etc', 'himawari-8.cfg'),
                              os.path.join('etc', 'eps_avhrrl1b_6.5.xml'),
                              os.path.join('etc', 'readers', '*.yaml'),
                              os.path.join('etc', 'writers', '*.cfg'),
                              os.path.join('etc', 'composites', '*.yaml'),
                              os.path.join('etc', 'enhancements', '*.cfg')]},
      zip_safe=False,
      install_requires=requires,
      tests_require=test_requires,
      extras_require={'xRIT': ['mipp >= 0.6.0'],
                      'hdf_eos': ['pyhdf'],
                      'viirs': ['h5py'],
                      'nc': ['netCDF4'],
                      'hrpt': ['pyorbital', 'pygac', 'python-geotiepoints'],
                      'proj': ['pyresample'],
                      'pyspectral': ['pyspectral'],
                      'pyorbital': ['pyorbital >= v0.2.3']}

      )
