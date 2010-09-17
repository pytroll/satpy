.. _geographic-images:

===================
 Geographic images
===================

In order to build satellite composites, mpop has to handle images. We could
have used PIL, but we felt the need to use numpy masked arrays as base for our
image channels, and we had to handle geographically enriched images. Hence the
two following modules: :mod:`imageo.image` to handle simple images, and :mod:`imageo.geo_image`.

Simple images
=============

.. automodule:: imageo.image
   :members:
   :undoc-members:

Geographically enriched images
==============================

.. automodule:: imageo.geo_image
   :members:
   :undoc-members:



