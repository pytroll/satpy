=======================================
 Making use of the :mod:`mpop` package
=======================================

The :mod:`mpop` package is the heart of mpop: here are defined the core classes
which the user will then need to build satellite composites.

Conventions about satellite names
=================================

Throughout the document, we will use the following conventions: 

- *satellite name* will designate the name of the platform, e.g. "noaa" in
   satellite "noaa 19"

- *satellite number* will refer to the number of the satellite in the series,
   e.g. "19" for satellite "noaa 19"

- *variant* will be used to differentiate the same data (from the same
   satellite and instrument) coming in different flavours. For example, we use
   variant to distinguish data coming from the satellite metop 02 from direct
   readout, regional coverage or global coverage.

Creating a scene object
=======================

Creating a scene object can be done calling the `create_scene` function of a
factory, (for example :meth:`mpop.satellites.GenericFactory.create_scene`).

The reader is refered to the documentation of the
:meth:`mpop.scene.SatelliteInstrumentScene` for a description of the input
arguments.

Such a scene object is roughly a container for :class:`mpop.channel.Channel`
objects, which hold the actual data and information for each band.

Loading the data
================

Loading the data is done through the
:meth:`mpop.scene.SatelliteInstrumentScene.load` method. Calling it effectively
loads the data from disk into memory, so it can take a while depending on the
volume of data to load and the performance of the host computer. The channels
listed as arguments become loaded, and cannot be reloaded: a subsequent call to
the method will not reload the data from disk.

Re-projecting data
==================

Once the data is loaded, one might need to re-project the data. The scene
objects can be projected onto other areas if the pyresample_ software is
installed, thanks to the :meth:`mpop.scene.SatelliteInstrumentScene.project`
method. As input, this method takes either a Definition object (see
pyresample's documentation) or string identificator for the area. In the latter
case, the referenced region has to be defined in the area file. The name and
location of this file is defined in the `mpop.cfg` configuration file, itself
located in the directory pointed by the `PPP_CONFIG_DIR` environment variable.

For more information about the internals of the projection process, take a look
at the :mod:`mpop.projector` module.

.. _pyresample: http://googlecode.com/p/pyresample

Geo-localisation of the data
============================

Once the data is loaded, each channel should have an `area` attribute
containing a pyresample_ area object, if the pyresample_ package is
available. These area objects should implement the :meth:`get_lonlats` method,
returning the longitudes and latitudes of the channel data.  For more
information on this matter, the reader is then referred to the documentation_ of
the aforementioned package.

.. _documentation: http://pyresample.googlecode.com/svn/trunk/docs/build/html/index.html

Image composites
================

Methods building image composites are distributed in different modules, taking
advantage of the hierarchical structure offered by OOP.

The image composites common to all visir instruments are defined in the
:mod:`mpop.instruments.visir` module. Some instrument modules, like
:mod:`mpop.instruments.avhrr` or :mod:`mpop.instruments.seviri` overload these
methods to adapt better for the instrument at hand.

For instructions on how to write a new composites, see :ref:`geographic-images`.


Adding a new satellite: configuration file
==========================================

A satellite configuration file looks like the following (here meteosat 7, mviri
instrument):

.. literalinclude:: ../../../mpop-smhi/etc/meteosat07.cfg
   :language: ini
   :linenos:

The configuration file must hold a `satellite` section, the list of channels
for the needed instruments (here `mviri-n` sections), and how to read the
data in mipp (`mviri-level1`) and how to read it in mpop (`mviri-level2`).

Using this template we can define new satellite and instruments.

Adding a new satellite: python code
===================================

Another way of adding satellites and instruments to mpop is to write the
correponding python code.

Here are example of such code:

.. literalinclude:: ../../mpop/instruments/mviri.py
   :language: python
   :linenos:


The :mod:`mpop` API
===================

Satellite scenes
----------------

.. automodule:: mpop.scene
   :members:
   :undoc-members:

Instrument channels
-------------------

.. automodule:: mpop.channel
   :members:
   :undoc-members:

The VisIr instrument class
--------------------------

.. automodule:: mpop.instruments.visir
   :members:
   :undoc-members:

Projection facility
-------------------

.. automodule:: mpop.projector
   :members:
   :undoc-members:

Satellite class loader
----------------------

.. automodule:: mpop.satellites
   :members:
   :undoc-members:
