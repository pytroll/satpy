#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009-2017.

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

import os.path
import sys
from glob import glob
import versioneer

from setuptools import find_packages, setup

requires = ['numpy >=1.12', 'pillow', 'pyresample >=1.10.3', 'trollsift',
            'trollimage >=1.5.1', 'pykdtree', 'six', 'pyyaml', 'xarray >=0.10.1',
            'dask[array] >=0.17.1']

# pyhdf (conda) == python-hdf4 (pip)
test_requires = ['behave', 'h5py', 'netCDF4', 'pyhdf', 'imageio', 'libtiff',
                 'rasterio']

if sys.version < '3.0':
    test_requires.append('mock')


extras_require = {
    # Readers:
    'hdfeos_l1b': ['python-hdf4', 'python-geotiepoints >= 1.1.7'],
    'geocat': ['python-hdf4'],
    'acspo': ['netCDF4 >= 1.1.8'],
    'clavrx': ['netCDF4 >= 1.1.8'],
    'viirs_l1b': ['netCDF4 >= 1.1.8'],
    'viirs_sdr': ['h5py >= 2.7.0'],
    'viirs_compact': ['h5py >= 2.7.0'],
    'omps_edr': ['h5py >= 2.7.0'],
    'amsr2_l1b': ['h5py >= 2.7.0'],
    'hrpt': ['pyorbital >= 1.3.1', 'pygac', 'python-geotiepoints >= 1.1.7'],
    'proj': ['pyresample'],
    'pyspectral': ['pyspectral >= 0.7.0'],
    'pyorbital': ['pyorbital >= 1.3.1'],
    'hrit_msg': ['pytroll-schedule'],
    'nc_nwcsaf_msg': ['netCDF4 >= 1.1.8'],
    'sar_c': ['python-geotiepoints >= 1.1.7', 'gdal'],
    'abi_l1b': ['h5netcdf'],
    # Writers:
    'scmi': ['netCDF4 >= 1.1.8'],
    'geotiff': ['gdal', 'trollimage[geotiff]'],
    'mitiff': ['libtiff'],
    # MultiScene:
    'animations': ['imageio'],
    # Documentation:
    'doc': ['sphinx'],
}
all_extras = []
for extra_deps in extras_require.values():
    all_extras.extend(extra_deps)
extras_require['all'] = list(set(all_extras))


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
README = open('README.rst', 'r').read()

setup(name=NAME,
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='Python package for earth-observing satellite data processing',
      long_description=README,
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
      packages=find_packages(),
      package_data={'satpy': [os.path.join('etc', 'geo_image.cfg'),
                              os.path.join('etc', 'areas.yaml'),
                              os.path.join('etc', 'satpy.cfg'),
                              os.path.join('etc', 'himawari-8.cfg'),
                              os.path.join('etc', 'eps_avhrrl1b_6.5.xml'),
                              os.path.join('etc', 'readers', '*.yaml'),
                              os.path.join('etc', 'writers', '*.yaml'),
                              os.path.join('etc', 'composites', '*.yaml'),
                              os.path.join('etc', 'enhancements', '*.cfg'),
                              os.path.join('etc', 'enhancements', '*.yaml'),
                              ]},
      zip_safe=False,
      install_requires=requires,
      tests_require=test_requires,
      python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*',
      extras_require=extras_require,
      )
