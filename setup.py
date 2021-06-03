#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009-2020 Satpy developers
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
"""Setup file for satpy."""

import os.path
from glob import glob

from setuptools import find_packages, setup

try:
    # HACK: https://github.com/pypa/setuptools_scm/issues/190#issuecomment-351181286
    # Stop setuptools_scm from including all repository files
    import setuptools_scm.integration
    setuptools_scm.integration.find_files = lambda _: []
except ImportError:
    pass

requires = ['numpy >=1.13', 'pillow', 'pyresample >=1.11.0', 'trollsift',
            'trollimage >1.10.1', 'pykdtree', 'pyyaml', 'xarray >=0.10.1, !=0.13.0',
            'dask[array] >=0.17.1', 'pyproj>=2.2', 'zarr', 'donfig', 'appdirs',
            'pooch']

test_requires = ['behave', 'h5py', 'netCDF4', 'pyhdf', 'imageio', 'libtiff',
                 'rasterio', 'geoviews', 'trollimage', 'fsspec']

extras_require = {
    # Readers:
    'avhrr_l1b_gaclac': ['pygac >= 1.3.0'],
    'modis_l1b': ['pyhdf', 'python-geotiepoints >= 1.1.7'],
    'geocat': ['pyhdf'],
    'acspo': ['netCDF4 >= 1.1.8'],
    'clavrx': ['netCDF4 >= 1.1.8'],
    'viirs_l1b': ['netCDF4 >= 1.1.8'],
    'viirs_sdr': ['h5py >= 2.7.0'],
    'viirs_compact': ['h5py >= 2.7.0'],
    'omps_edr': ['h5py >= 2.7.0'],
    'amsr2_l1b': ['h5py >= 2.7.0'],
    'hrpt': ['pyorbital >= 1.3.1', 'pygac', 'python-geotiepoints >= 1.1.7'],
    'hrit_msg': ['pytroll-schedule'],
    'msi_safe': ['glymur'],
    'nc_nwcsaf_msg': ['netCDF4 >= 1.1.8'],
    'sar_c': ['python-geotiepoints >= 1.1.7', 'rasterio', 'rioxarray'],
    'abi_l1b': ['h5netcdf'],
    'seviri_l1b_hrit': ['pyorbital >= 1.3.1'],
    'seviri_l1b_native': ['pyorbital >= 1.3.1'],
    'seviri_l1b_nc': ['pyorbital >= 1.3.1', 'netCDF4 >= 1.1.8'],
    'seviri_l2_bufr': ['eccodes-python'],
    'seviri_l2_grib': ['eccodes-python'],
    'hsaf_grib': ['pygrib'],
    # Writers:
    'cf': ['h5netcdf >= 0.7.3'],
    'awips_tiled': ['netCDF4 >= 1.1.8'],
    'geotiff': ['rasterio', 'trollimage[geotiff]'],
    'mitiff': ['libtiff'],
    'ninjo': ['pyninjotiff', 'pint'],
    # Composites/Modifiers:
    'rayleigh': ['pyspectral >= 0.10.1'],
    'angles': ['pyorbital >= 1.3.1'],
    # MultiScene:
    'animations': ['imageio'],
    # Documentation:
    'doc': ['sphinx', 'sphinx_rtd_theme', 'sphinxcontrib-apidoc'],
    # Other
    'geoviews': ['geoviews'],
    'overlays': ['pycoast', 'pydecorate'],
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


entry_points = {
    'console_scripts': [
        'satpy_retrieve_all_aux_data=satpy.aux_download:retrieve_all_cmd',
    ],
}


NAME = 'satpy'
with open('README.rst', 'r') as readme:
    README = readme.read()

setup(name=NAME,
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
      packages=find_packages(),
      # Always use forward '/', even on Windows
      # See https://setuptools.readthedocs.io/en/latest/userguide/datafiles.html#data-files-support
      package_data={'satpy': ['etc/geo_image.cfg',
                              'etc/areas.yaml',
                              'etc/satpy.cfg',
                              'etc/himawari-8.cfg',
                              'etc/eps_avhrrl1b_6.5.xml',
                              'etc/readers/*.yaml',
                              'etc/writers/*.yaml',
                              'etc/composites/*.yaml',
                              'etc/enhancements/*.cfg',
                              'etc/enhancements/*.yaml',
                              'tests/etc/readers/*.yaml',
                              'tests/etc/composites/*.yaml',
                              'tests/etc/writers/*.yaml',
                              ]},
      zip_safe=False,
      use_scm_version={'write_to': 'satpy/version.py'},
      setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
      install_requires=requires,
      tests_require=test_requires,
      python_requires='>=3.7',
      extras_require=extras_require,
      entry_points=entry_points,
      )
