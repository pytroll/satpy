==========
Resampling
==========




Resampling algorithms
=====================

The default resampling method in SatPy is nearest neighbor (``nearest``).
There are also ``bilinear`` and Elliptical Weighted Averaging (``ewa``)
available.

::

    >>> local_scene = global_scene.resample("euro4", resampler="bilinear")

To make resampling faster next time (when resampling geostationary satellite
data), it is possible to save the resampling coefficients and use more CPUs
when calculating the coefficients on the first go:

    >>> local_scene = global_scene.resample("euro4", resampler="bilinear",
    ...                                     nprocs=4, cache_dir="/var/tmp")

Create custom area definition
=============================

See :class:`pyresample.geometry.AreaDefinition` for information on creating
areas that can be passed to the resample method::

    >>> from pyresample.geometry import AreaDefinition
    >>> my_area = AreaDefinition(...)
    >>> local_scene = global_scene.resample(my_area)

Create dynamic area definition
==============================

See :class:`pyresample.geometry.DynamicAreaDefinition` for more information.

Examples coming soon...

Store area definitions
======================

Area definitions can be added to a custom YAML file (see
`pyresample's documentation <http://pyresample.readthedocs.io/en/stable/geo_def.html#pyresample-utils>`_
for more information)
and loaded using pyresample's utility methods::

    >>> from pyresample.utils import parse_area_file
    >>> my_area = parse_area_file('my_areas.yaml', 'my_area')[0]

Examples coming soon...
