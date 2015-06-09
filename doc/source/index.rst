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

Get to the project_ page, with source and downloads.

It is designed to be easily extendable to support any meteorological satellite
by the creation of plugins. In the base distribution, we provide support for
Meteosat-7, -8, -9, -10, Himawari-6 (MTSAT-1R), Himawari-7 (MTSAT-2), GOES-11, GOES-12, GOES-13 through the use of
mipp_, and NOAA-15, -16, -17, -18, -19, Metop-A and -B through the use of AAPP.

Reprojection of data is also available through the use of pyresample_.

.. _project: http://github.com/mraspaud/mpop
.. _mipp: http://github.com/loerum/mipp
.. _pyresample: http://pyresample.googlecode.com

.. toctree::
   :maxdepth: 2

   install
   quickstart
   pp
   input
   image

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`




