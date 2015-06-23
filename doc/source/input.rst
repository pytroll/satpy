==============================================
 Input plugins: the :mod:`mpop.satin` package
==============================================

Available plugins and their requirements
========================================

mipp_xrit
---------

Reader for for hrit/lrit formats. Recommends numexpr and pyresample.

.. automodule:: mpop.satin.mipp_xrit
   :members:
   :undoc-members:


aapp1b
------

Reader for AAPP level 1b format. Requires numpy, recommends pyresample.

.. automodule:: mpop.satin.aapp1b
   :members:
   :undoc-members:


eps_l1b
-------
Reader for EPS level 1b format. Recommends pyresample.

.. automodule:: mpop.satin.eps_l1b
   :members:
   :undoc-members:



viirs_sdr
---------
Reader for the VIIRS SDR format. Requires h5py.

.. automodule:: mpop.satin.viirs_sdr
   :members:
   :undoc-members:

viirs_compact
-------------
Reader for the VIIRS compact format from EUMETSAT. Requires h5py.

.. automodule:: mpop.satin.viirs_compact
   :members:
   :undoc-members:

hdfeos_l1b
----------
Reader for Modis data format. Requires pyhdf.

.. automodule:: mpop.satin.hdfeos_l1b
   :members:
   :undoc-members:


msg_hdf
-------
Reader for MSG cloud products. Requires h5py, recommends acpg.

.. automodule:: mpop.satin.msg_hdf
   :members:
   :undoc-members:


pps_hdf
-------
Reader for PPS cloud products. Requires acpg.

.. automodule:: mpop.satin.pps_hdf
   :members:
   :undoc-members:


hrpt
----
Reader for level 0 hrpt format. Requires AAPP and pynav.

.. automodule:: mpop.satin.hrpt
   :members:
   :undoc-members:


eps1a
-----
Reader for level 1a Metop segments. Requires AAPP, kai and eugene.

.. automodule:: mpop.satin.eps1a
   :members:
   :undoc-members:


Interaction with reader plugins
===============================

The reader plugin instance used for a specific scene is accessible through a
scene attribute named after the plugin format. For example, the attribute for
the *foo* format would be called *foo_reader*.

This way the other methods present in the plugins are available through the
scene object.

The plugin API
==============

.. versionchanged:: 0.13.0
   New plugin API


.. automodule:: mpop.plugin_base
   :members:
   :undoc-members:

Adding a new plugin
===================

For now only reader and writer plugins base classes are defined.

To add one of those, just create a new class that subclasses the plugin.

The interface of any reader plugin must include the :meth:`load` method.

Take a look at the existing readers for more insight.
