=============================
 Organization of the package
=============================

The mpop package is organized as follows:

* the :mod:`satin` directory contains the input plugins to the different data
  formats. Available plugins include aapp1b, mipp, and msg_hrit.

* the :mod:`imageo` directory contains the image object and handling modules.

* the :mod:`pp` directory contains the different instruments and satellites
  implementations.

Writing a new input plugin
==========================

An input plugin has to implement the two following functions:

* :func:`load` that loads the calibrated data into the channels.

* :func:`get_lat_lon` which returns the latitude and longitude of the satellite
  view. This is needed only when the satellite view cannot be expressed
  analytically (swath).

Here is the example for the aapp1b reader plugin:

.. literalinclude:: ../../satin/aapp1b.py
   :linenos:

Adding an instrument
====================

In :mod:`pp.instrument` directory, one has to add a channel list and an
instrument name in a child class from visir. For example, for the AVHRR
instrument:

.. literalinclude:: ../../pp/instruments/avhrr.py
   :linenos:

Adding a satellite
==================

In the :mod:`pp.satellite` directory, add the definition of the satellite as a
child class of the instrument of interest. For example here is the noaa17
satellite definition:

.. literalinclude:: ../../pp/satellites/noaa17.py
   :linenos:


One should also link to the desired input format through a configuration file
named after the fullname of the satellite:

.. literalinclude:: ../../etc/noaa17.cfg
   :linenos:
