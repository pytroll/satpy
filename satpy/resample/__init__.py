# Copyright (c) 2015-2025 Satpy developers
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
"""Resampling in Satpy.

Satpy provides multiple resampling algorithms for resampling geolocated
data to uniform projected grids. The easiest way to perform resampling in
Satpy is through the :class:`~satpy.scene.Scene` object's
:meth:`~satpy.scene.Scene.resample` method. Additional utility functions are
also available to assist in resampling data. Below is more information on
resampling with Satpy as well as links to the relevant API documentation for
available keyword arguments.

Resampling algorithms
---------------------

.. csv-table:: Available Resampling Algorithms
    :header-rows: 1
    :align: center

    "Resampler", "Description", "Related"
    "nearest", "Nearest Neighbor", :class:`~satpy.resample.kdtree.KDTreeResampler`
    "ewa", "Elliptical Weighted Averaging", :class:`~pyresample.ewa.dask_ewa.DaskEWAResampler`
    "ewa_legacy", "Elliptical Weighted Averaging (Legacy)", \
    :class:`~pyresample.ewa._legacy_dask_ewa.LegacyDaskEWAResampler`
    "native", "Native", :class:`~satpy.resample.native.NativeResampler`
    "bilinear", "Bilinear", :class:`~satpy.resample.kdtree.BilinearResampler`
    "bucket_avg", "Average Bucket Resampling", :class:`~satpy.resample.bucket.BucketAvg`
    "bucket_sum", "Sum Bucket Resampling", :class:`~satpy.resample.bucket.BucketSum`
    "bucket_count", "Count Bucket Resampling", :class:`~satpy.resample.bucket.BucketCount`
    "bucket_fraction", "Fraction Bucket Resampling", :class:`~satpy.resample.bucket.BucketFraction`
    "gradient_search", "Gradient Search Resampling", :func:`~pyresample.gradient.create_gradient_search_resampler`

The resampling algorithm used can be specified with the ``resampler`` keyword
argument and defaults to ``nearest``:

.. code-block:: python

    >>> scn = Scene(...)
    >>> euro_scn = scn.resample('euro4', resampler='nearest')

.. warning::

    Some resampling algorithms expect certain forms of data. For example, the
    EWA resampling expects polar-orbiting swath data and prefers if the data
    can be broken in to "scan lines". See the API documentation for a specific
    algorithm for more information.

Resampling for comparison and composites
----------------------------------------

While all the resamplers can be used to put datasets of different resolutions
on to a common area, the 'native' resampler is designed to match datasets to
one resolution in the dataset's original projection. This is extremely useful
when generating composites between bands of different resolutions.

.. code-block:: python

    >>> new_scn = scn.resample(resampler='native')

By default this resamples to the
:meth:`highest resolution area <satpy.scene.Scene.finest_area>` (smallest footprint per
pixel) shared between the loaded datasets. You can easily specify the lowest
resolution area:

.. code-block:: python

    >>> new_scn = scn.resample(scn.coarsest_area(), resampler='native')

Providing an area that is neither the minimum or maximum resolution area
may work, but behavior is currently undefined.

Caching for geostationary data
------------------------------

Satpy will do its best to reuse calculations performed to resample datasets,
but it can only do this for the current processing and will lose this
information when the process/script ends. Some resampling algorithms, like
``nearest`` and ``bilinear``, can benefit by caching intermediate data on disk in the directory
specified by `cache_dir` and using it next time. This is most beneficial with
geostationary satellite data where the locations of the source data and the
target pixels don't change over time.

    >>> new_scn = scn.resample('euro4', cache_dir='/path/to/cache_dir')

See the documentation for specific algorithms to see availability and
limitations of caching for that algorithm.

Create custom area definition
-----------------------------

See :class:`pyresample.geometry.AreaDefinition` for information on creating
areas that can be passed to the resample method::

    >>> from pyresample.geometry import AreaDefinition
    >>> my_area = AreaDefinition(...)
    >>> local_scene = scn.resample(my_area)

Resize area definition in pixels
--------------------------------

Sometimes you may want to create a small image with fixed size in pixels.
For example, to create an image of (y, x) pixels :

    >>> small_scn = scn.resample(scn.finest_area().copy(height=y, width=x), resampler="nearest")


.. warning::

    Be aware that resizing with native resampling (``resampler="native"``) only
    works if the new size is an integer factor of the original input size. For example,
    multiplying the size by 2 or dividing the size by 2. Multiplying by 1.5 would
    not be allowed.


Create dynamic area definition
------------------------------

See :class:`pyresample.geometry.DynamicAreaDefinition` for more information.

Examples coming soon...

Store area definitions
----------------------

Area definitions can be saved to a custom YAML file (see
`pyresample's writing to disk <http://pyresample.readthedocs.io/en/stable/geometry_utils.html#writing-to-disk>`_)
and loaded using pyresample's utility methods
(`pyresample's loading from disk <http://pyresample.readthedocs.io/en/stable/geometry_utils.html#loading-from-disk>`_)::

    >>> from pyresample import load_area
    >>> my_area = load_area('my_areas.yaml', 'my_area')

Or using :func:`satpy.area.get_area_def`, which will search through all
``areas.yaml`` files in your ``SATPY_CONFIG_PATH``::

    >>> from satpy.area import get_area_def
    >>> area_eurol = get_area_def("eurol")

For examples of area definitions, see the file ``etc/areas.yaml`` that is
included with Satpy and where all the area definitions shipped with Satpy are
defined. The section below gives an overview of these area definitions.

Area definitions included in Satpy
----------------------------------

.. include:: /area_def_list.rst

"""
from __future__ import annotations

from typing import Any

from satpy.utils import _import_and_warn_new_location

IMPORT_PATHS = {
    "KDTreeResampler": "satpy.resample.kdtree",
    "BilinearResampler": "satpy.resample.kdtree",
    "NativeResampler": "satpy.resample.native",
    "BucketResamplerBase": "satpy.resample.bucket",
    "BucketAvg": "satpy.resample.bucket",
    "BucketSum": "satpy.resample.bucket",
    "BucketCount": "satpy.resample.bucket",
    "BucketFraction": "satpy.resample.bucket",
    "resample": "satpy.resample.base",
    "prepare_resampler": "satpy.resample.base",
    "resample_dataset": "satpy.resample.base",
    "get_area_file": "satpy.area",
    "get_area_def": "satpy.area",
    "add_xy_coords": "satpy.coords",
    "add_crs_xy_coords": "satpy.coords",
}

def __getattr__(name: str) -> Any:
    new_module = IMPORT_PATHS.get(name)

    if new_module is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    return _import_and_warn_new_location(new_module, name)
