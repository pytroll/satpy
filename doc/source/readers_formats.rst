.. _specific-readers-and-formats:

============================
Specific Readers and Formats
============================

In this section you can find guidance for a selection of sensors or formats.

Implementation details for *all* readers including unique keyword arguments
can be found in the :doc:`Readers API documentation <api/satpy.readers>`. Note
that this documentation is for the Python modules and may be shared by
multiple reader instances. Most shared Python modules can be found in the
:doc:`readers "core" subpackage <api/satpy.readers.core>`.


SEVIRI L1.5 data
================

*The Spinning Enhanced Visible and InfraRed Imager (SEVIRI) is the primary
instrument on Meteosat Second Generation (MSG) and has the capacity to observe
the Earth in 12 spectral channels.*

*Level 1.5 corresponds to image data that has been corrected for all unwanted
radiometric and geometric effects, has been geolocated using a standardised
projection, and has been calibrated and radiance-linearised.*
(From the EUMETSAT documentation)

Satpy provides readers for various SEVIRI L1.5 data formats. For common
properties see :mod:`satpy.readers.core.seviri`. Format-specific
documentation can be found here:

- Native: :mod:`satpy.readers.seviri_l1b_native`
- HRIT: :mod:`satpy.readers.seviri_l1b_hrit`
- netCDF: :mod:`satpy.readers.seviri_l1b_nc`


HRIT format
===========

Satpy can read many variants of the HRIT format. Common functionality is implemented in
:mod:`satpy.readers.core.hrit`. Format-specific documentation can be found here:

- :mod:`SEVIRI HRIT <satpy.readers.seviri_l1b_hrit>`
- :mod:`GOES Imager HRIT <satpy.readers.goes_imager_hrit>`
- :mod:`Electro-L HRIT <satpy.readers.electrol_hrit>`
- :ref:`JMA HRIT <jma-hrit-readers>` (see below)


.. _jma-hrit-readers:

JMA HRIT readers
----------------

The JMA HRIT format is described in the `JMA HRIT - Mission Specific
Implementation`_. There are three readers for this format in Satpy:

- ``jami_hrit``: For data from the `JAMI` instrument on MTSAT-1R
- ``mtsat2-imager_hrit``: For data from the `Imager` instrument on MTSAT-2
- ``ahi_hrit``: For data from the `AHI` instrument on Himawari-8/9

Although the data format is identical, the instruments have different
characteristics, which is why there is a dedicated reader for each of them.
Documentation can be found in :mod:`satpy.readers.hrit_jma`.

Sample data is available here:

- `JAMI/Imager sample data`_
- `AHI sample data`_

.. _JMA HRIT - Mission Specific Implementation: http://www.jma.go.jp/jma/jma-eng/satellite/introduction/4_2HRIT.pdf
.. _JAMI/Imager sample data: https://www.data.jma.go.jp/mscweb/en/operation/hrit_sample.html
.. _AHI sample data: https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/sample_hrit.html
