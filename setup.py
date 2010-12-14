#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009.

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

from setuptools import setup, Extension
import os.path
import ConfigParser

BASE_PATH = os.path.sep.join(os.path.dirname(
    os.path.realpath(__file__)).split(os.path.sep))

try:
    MODE = os.getenv("SMHI_MODE")
    if MODE is None:
        MODE = "offline"
    
    CONF = ConfigParser.ConfigParser()
    CONF.read(os.path.join(BASE_PATH, "etc", "meteosat09.cfg"))
    
    MSG_LIB = CONF.get(MODE, 'msg_lib')
    MSG_INC = CONF.get(MODE, 'msg_inc')
    
    CONF.read(os.path.join(BASE_PATH, "setup.cfg"))
    NUMPY_INC = CONF.get('numpy', 'numpy_inc')

    EXTS = [Extension('satin.pynwclib',
                      ['satin/pynwclib.c'],
                      include_dirs=[MSG_INC,
                                    NUMPY_INC],
                      libraries=['nwclib','msg','m'],
                      library_dirs=[MSG_LIB])]
except Exception, e:
    print e
    EXTS = []
    
NAME = 'mpop'

setup(name=NAME,
      version='0.10.0alpha1',
      description='Meteorological post processing package',
      author='Adam Dybbroe, Martin Raspaud',
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
                    'doc/source/satellites_h.rst',
                    'doc/source/geo_image.rst',
                    'doc/source/image.rst',
                    'doc/source/rs_images.rst'])],
      zip_safe=False,
      ext_modules = EXTS,
      requires=['acpg (>=2.03)',
                'numpy (>=1.4.1)',
                'pyresample (>=0.7.1)']
      )
