============
Enhancements
============

Built-in enhancement methods
============================

stretch
-------

The most basic operation is to stretch the image so that the data fits
to the output format.  There are many different ways to stretch the
data, which are configured by giving them in `kwargs` dictionary, like
in the example above.  The default, if nothing else is defined, is to
apply a linear stretch.  For more details, see below.

linear
******

As the name suggests, linear stretch converts the input values to
output values in a linear fashion.  By default, 5% of the data is cut
on both ends of the scale, but these can be overridden with
``cutoffs=(0.005, 0.005)`` argument::

    - name: stretch
      method: !!python/name:satpy.enhancements.stretch
      kwargs:
        stretch: linear
        cutoffs: (0.003, 0.005)

.. note::

    This enhancement is currently not optimized for dask because it requires
    getting minimum/maximum information for the entire data array.

crude
*****

The crude stretching is used to limit the input values to a certain
range by clipping the data. This is followed by a linear stretch with
no cutoffs specified (see above). Example::

    - name: stretch
      method: !!python/name:satpy.enhancements.stretch
      kwargs:
        stretch: crude
        min_stretch: [0, 0, 0]
        max_stretch: [100, 100, 100]

It is worth noting that this stretch can also be used to _invert_ the
data by giving larger values to the min_stretch than to max_stretch.

histogram
*********

gamma
-----

invert
------

crefl_scaling
-------------

cira_stretch
------------

lookup
------

colorize
--------

palettize
---------

three_d_effect
--------------

The `three_d_effect` enhancement adds an 3D look to an image by
convolving with a 3x3 kernel.  User can adjust the strength of the
effect by determining the weight (default: 1.0).  Example::

    - name: 3d_effect
      method: !!python/name:satpy.enhancements.three_d_effect
      kwargs:
        weight: 1.0


btemp_threshold
---------------
