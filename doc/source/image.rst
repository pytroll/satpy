.. _geographic-images:

===================
 Geographic images
===================

In order to build satellite composites, satpy has to handle images. We could
have used PIL, but we felt the need to use numpy masked arrays as base for our
image channels, and we had to handle geographically enriched images. Hence the
two following modules: :mod:`satpy.imageo.image` to handle simple images, and
:mod:`satpy.imageo.geo_image`.

Simple images
=============

.. automodule:: satpy.imageo.image
   :members:
   :undoc-members:

Geographically enriched images
==============================

.. automodule:: satpy.imageo.geo_image
   :members:
   :undoc-members:



