=====================
Satpy's Documentation
=====================

Satpy is a python library for reading, manipulating, and writing data from
remote-sensing earth-observing satellite instruments. Satpy
provides users with readers that convert geophysical parameters from various
file formats to the common Xarray :class:`~xarray.DataArray` and
:class:`~xarray.Dataset` classes for easier interoperability with other
scientific python libraries. Satpy also provides interfaces for creating
RGB (Red/Green/Blue) images and other composite types by combining data
from multiple instrument bands or products. Various atmospheric corrections
and visual enhancements are provided for improving the usefulness and quality
of output images. Output data can be written to
multiple output file formats such as PNG, GeoTIFF, and CF standard NetCDF
files. Satpy also allows users to resample data to geographic projected grids
(areas). Satpy is maintained by the open source
`Pytroll <http://pytroll.github.io/>`_ group.

The Satpy library acts as a high-level abstraction layer on top of other
libraries maintained by the Pytroll group including:

- `pyresample <http://pyresample.readthedocs.io/en/latest/>`_
- `pyspectral <https://pyspectral.readthedocs.io/en/latest/>`_
- `trollimage <http://trollimage.readthedocs.io/en/latest/>`_
- `pycoast <https://pycoast.readthedocs.io/en/latest/>`_
- `pydecorate <https://pydecorate.readthedocs.io/en/latest/>`_
- `python-geotiepoints <https://python-geotiepoints.readthedocs.io/en/latest/>`_
- `pyninjotiff <https://github.com/pytroll/pyninjotiff>`_

Go to the Satpy project_ page for source code and downloads.

Satpy is designed to be easily extendable to support any earth observation
satellite by the creation of plugins (readers, compositors, writers, etc).
The table at the bottom of this page shows the input formats supported by
the base Satpy installation.

.. note::

    Satpy's interfaces are not guaranteed stable and may change until version
    1.0 when backwards compatibility will be a main focus.

.. versionchanged:: 0.20.0

    Dropped Python 2 support.

.. _project: http://github.com/pytroll/satpy

.. toctree::
    :maxdepth: 2

    overview
    install
    config
    data_download
    examples/index
    quickstart
    readers
    remote_reading
    composites
    resample
    enhancements
    writers
    multiscene
    dev_guide/index

.. toctree::
    :maxdepth: 1

    Satpy API <api/modules>
    faq
    Release Notes <https://github.com/pytroll/satpy/blob/main/CHANGELOG.md>
    Security Policy <https://github.com/pytroll/satpy/blob/main/SECURITY.md>

.. _reader_table:

.. include:: reader_table.rst

.. note::

    Status description:

    Defunct
        Most likely the reader is not functional. If it is there is a good chance of
        bugs and/or performance problems (e.g. not ported to dask/xarray yet). Future
        development is unclear. Users are encouraged to contribute (see section
        :doc:`dev_guide/CONTRIBUTING` and/or get help on Slack or by opening a Github issue).

    Alpha
        This denotes early development status. Reader is functional and implements some
        or all of the nominal features. There might be bugs. Exactness of results is
        not be guaranteed. Use at your own risk.

    Beta
        This denotes final developement status. Reader is functional and implements all
        nominal features. Results should be dependable but there might be bugs. Users
        are actively encouraged to test and report bugs.

    Nominal
        This denotes a finished status. Reader is functional and most likely no new
        features will be introduced. It has been tested and there are no known bugs.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
