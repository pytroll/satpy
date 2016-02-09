=======================================
 Making use of the :mod:`satpy` package
=======================================

The :mod:`satpy` package is the heart of satpy: here are defined the core classes
which the user will then need to build satellite composites.

Conventions about satellite names
=================================

Throughout the document, we will use the following conventions: 

- *platform name* is the name of an individual satellite following the
  OSCAR_ naming scheme, e.g. "NOAA-19".

- *variant* will be used to differentiate the same data (from the same
   satellite and instrument) coming in different flavours. For example, we use
   variant to distinguish data coming from the satellite Metop-B from direct
   readout (no variant), regional coverage (EARS) or global coverage (GDS).

All the satellite configuration files in `PPP_CONFIG_DIR` should be
named `<variant><platform name>.cfg`, e.g. `NOAA-19.cfg` or
`GDSMetop-B.cfg`.

.. _OSCAR: http://www.wmo-sat.info/oscar/satellites/

Creating a scene object
=======================

Creating a scene object can be done calling the `create_scene` function of a
factory, (for example :meth:`satpy.satellites.GenericFactory.create_scene`).

The reader is refered to the documentation of the
:meth:`satpy.scene.SatelliteInstrumentScene` for a description of the input
arguments.

Such a scene object is roughly a container for :class:`satpy.channel.Channel`
objects, which hold the actual data and information for each band.

Loading the data
================

Loading the data is done through the
:meth:`satpy.scene.SatelliteInstrumentScene.load` method. Calling it effectively
loads the data from disk into memory, so it can take a while depending on the
volume of data to load and the performance of the host computer. The channels
listed as arguments become loaded, and cannot be reloaded: a subsequent call to
the method will not reload the data from disk.

Re-projecting data
==================

Once the data is loaded, one might need to re-project the data. The scene
objects can be projected onto other areas if the pyresample_ software is
installed, thanks to the :meth:`satpy.scene.SatelliteInstrumentScene.project`
method. As input, this method takes either a Definition object (see
pyresample's documentation) or string identificator for the area. In the latter
case, the referenced region has to be defined in the area file. The name and
location of this file is defined in the `satpy.cfg` configuration file, itself
located in the directory pointed by the `PPP_CONFIG_DIR` environment variable.

For more information about the internals of the projection process, take a look
at the :mod:`satpy.projector` module.

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
:mod:`satpy.instruments.visir` module. Some instrument modules, like
:mod:`satpy.instruments.avhrr` or :mod:`satpy.instruments.seviri` overload these
methods to adapt better for the instrument at hand.

For instructions on how to write a new composites, see :ref:`geographic-images`.


Adding a new satellite: configuration file
==========================================

A satellite configuration file looks like the following (here Meteosat-7, mviri
instrument):

.. literalinclude:: ../../../satpy-smhi/etc/Meteosat-7.cfg
   :language: ini
   :linenos:

The configuration file must hold a `satellite` section, the list of channels
for the needed instruments (here `mviri-n` sections), and how to read the
data in mipp (`mviri-level1`) and how to read it in satpy (`mviri-level2`).

Using this template we can define new satellite and instruments.

Adding a new satellite: python code
===================================

Another way of adding satellites and instruments to satpy is to write the
correponding python code.

Here are example of such code:

.. literalinclude:: ../../satpy/instruments/mviri.py
   :language: python
   :linenos:


The :mod:`satpy` API
===================

Satellite scenes
----------------

.. automodule:: satpy.scene
   :members:
   :undoc-members:

Instrument channels
-------------------

.. automodule:: satpy.channel
   :members:
   :undoc-members:

The VisIr instrument class
--------------------------

.. automodule:: satpy.instruments.visir
   :members:
   :undoc-members:

Projection facility
-------------------

.. automodule:: satpy.projector
   :members:
   :undoc-members:

Satellite class loader
----------------------

.. automodule:: satpy.satellites
   :members:
   :undoc-members:

Miscellaneous tools
-------------------

.. automodule:: satpy.tools
   :members:
   :undoc-members:
