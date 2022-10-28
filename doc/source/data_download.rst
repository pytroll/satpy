Downloading Data
================

One of the main features of Satpy is its ability to read various satellite
data formats. However, it currently only provides limited methods for
downloading data from remote sources and these methods are limited to demo
data for `Pytroll examples <https://github.com/pytroll/pytroll-examples>`_.
See the examples and the :mod:`~satpy.demo` API documentation for details.
Otherwise, Satpy assumes all data is available
through the local system, either as a local directory or network
mounted file systems. Certain readers that use ``xarray`` to open data files
may be able to load files from remote systems by using OpenDAP or similar
protocols.

As a user there are two options for getting access to data:

1. Download data to your local machine.
2. Connect to a remote system that already has access to data.

The most common case of a remote system having access to data is with a cloud
computing service like Google Cloud Platform (GCP) or Amazon Web
Services (AWS). Another possible case is an organization having direct
broadcast antennas where they receive data directly from the satellite or
satellite mission organization (NOAA, NASA, EUMETSAT, etc). In these cases
data is usually available as a mounted network file system and can be accessed
like a normal local path (with the added latency of network communications).

Below are some data sources that provide data that can be read by Satpy. If
you know of others please let us know by either creating a GitHub issue or
pull request.

NOAA GOES on Amazon Web Services
--------------------------------

* `Resource Description <https://registry.opendata.aws/noaa-goes/>`__
* `Data Browser <http://noaa-goes16.s3.amazonaws.com/index.html>`__
* Associated Readers: ``abi_l1b``

In addition to the pages above, Brian Blaylock's `GOES-2-Go <https://github.com/blaylockbk/goes2go>`_
python package is useful for downloading GOES data to your local machine.
Brian also prepared some instructions
for using the ``rclone`` tool for downloading AWS data to a local machine. The
instructions can be found
`here <https://github.com/blaylockbk/pyBKB_v3/blob/master/rclone_howto.md>`_.

NOAA GOES on Google Cloud Platform
----------------------------------

GOES-16
^^^^^^^

* `Resource Description <https://console.cloud.google.com/marketplace/details/noaa-public/goes-16>`__
* `Data Browser <https://console.cloud.google.com/storage/browser/gcp-public-data-goes-16>`__
* Associated Readers: ``abi_l1b``

GOES-17
^^^^^^^

* `Resource Description <https://console.cloud.google.com/marketplace/details/noaa-public/goes-17>`__
* `Data Browser <https://console.cloud.google.com/storage/browser/gcp-public-data-goes-17>`__
* Associated Readers: ``abi_l1b``

NOAA CLASS
----------

* `Data Ordering <https://www.class.ncdc.noaa.gov>`__
* Associated Readers: ``viirs_sdr``

NASA VIIRS Atmosphere SIPS
--------------------------

* `Resource Description <https://sips.ssec.wisc.edu/>`__
* Associated Readers: ``viirs_l1b``

EUMETSAT Data Center
--------------------

* `Data Ordering <https://eoportal.eumetsat.int>`__
