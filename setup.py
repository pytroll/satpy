#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009, 2010, 2011.

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

from version import get_git_version


BASE_PATH = os.path.sep.join(os.path.dirname(
    os.path.realpath(__file__)).split(os.path.sep))

NAME = 'mpop'
EXTS = []

setup(name=NAME,
      version=get_git_version(),
      description='Meteorological post processing package',
      author='Martin Raspaud',
      author_email='martin.raspaud@smhi.se',
      packages=['mpop', 'mpop.satellites', 'mpop.instruments', 'mpop.satin',
                'mpop.satout', 'mpop.saturn', 'mpop.imageo'],
      data_files=[('etc',['etc/geo_image.cfg']),
                  ('etc',['etc/world_map.ascii']),
                  ('share/doc/'+NAME+'/',
                   ['doc/Makefile',
                    'doc/source/conf.py',
                    'doc/source/index.rst',
                    'doc/source/install.rst',
                    'doc/source/quickstart.rst',
                    'doc/source/image.rst',
                    'doc/source/pp.rst',
                    'doc/source/saturn.rst',
                    'doc/source/input.rst',
                    'doc/examples/geo_hrit.py',
                    'doc/examples/polar_aapp1b.py',
                    'doc/examples/polar_segments.py'])],
      zip_safe=False,
      ext_modules = EXTS,
      requires=['numpy (>=1.4.1)',
                'pyresample (>=0.7.1)',
                'mipp']
      )
