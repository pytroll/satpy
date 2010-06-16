.. NWCSAF/MPoP documentation master file, created by
   sphinx-quickstart on Fri Sep 25 16:58:28 2009.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==================================
 Welcome to MPoP's documentation!
==================================

Meteorological Post-Processing package is a python library for generating RGB
products for meteorological remote sensing. As such it can create RGB
composites directly from satellite instrument channels, or take advantage of
precomputed PGEs.

It is designed to be easily extendable to support any meteorological satellite
by the creation of plugins. In the base distribution, we provide support for
Meteosat 7, 8, 9, MTSAT1R, GOES 11, 12, Noaa 15, 16, 17, 18, 19, and Metop A
through the use of mipp_.

.. _mipp: http://github.com/loerum/mipp.git

.. toctree::
   :maxdepth: 2

   install
   quickstart
   organization
   satellites_h
   rs_images


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`




