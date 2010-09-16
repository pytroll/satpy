.. NWCSAF/MPoP documentation master file, created by
   sphinx-quickstart on Fri Sep 25 16:58:28 2009.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==================================
 Welcome to MPoP's documentation!
==================================

The Meteorological Post-Processing package is a python library for generating
RGB products for meteorological remote sensing. As such it can create RGB
composites directly from satellite instrument channels, or take advantage of
precomputed PGEs.

It is designed to be easily extendable to support any meteorological satellite
by the creation of plugins. In the base distribution, we provide support for
Meteosat 7, 8, 9, MTSAT1R, MTSAT2, GOES 11, GOES 12, GOES 13 through the use of
mipp_, and Noaa 15, 16, 17, 18, 19, and Metop A through the use of aapp and
ahamap.

Reprojection of data is also available through the use of pyresample_.

.. _mipp: http://github.com/loerum/mipp.git
.. _pyresample: http://pyresample.googlecode.com

.. toctree::
   :maxdepth: 2

   install
   quickstart
   pp
   input
   runner
   organization
   satellites_h
   rs_images


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`




