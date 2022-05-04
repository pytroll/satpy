====================
Reading remote files
====================

.. testsetup:: *
    >>> import sys
    >>> reload(sys)
    >>> sys.setdefaultencoding('utf8')


Using a single reader
=====================

Some of the readers in Satpy can read data directly over various transfer protocols. This is done
using `fsspec <https://filesystem-spec.readthedocs.io/en/latest/index.html>`_ and various packages
it is using underneath.

As an example, reading ABI data from public AWS S3 storage can be done in the following way::

    from satpy import Scene

    storage_options = {'anon': True}
    filenames = ['s3://noaa-goes16/ABI-L1b-RadC/2019/001/17/*_G16_s20190011702186*']
    scn = Scene(reader='abi_l1b', filenames=filenames, reader_kwargs={'storage_options': storage_options})
    scn.load(['true_color_raw'])

In addition to `fsspec` the `s3fs` library needs to be installed.

As an alternative, the storage options can be given using
`fsspec configuration <https://filesystem-spec.readthedocs.io/en/latest/features.html#configuration>`_.
For the above example, the configuration could be saved to `s3.json` in the `fsspec` configuration directory
(by default placed in `~/.config/fsspec/` directory in Linux)::

    {
        "s3": {
            "anon": "true"
        }
    }

For reference, reading SEVIRI HRIT data from a local S3 storage works the same way::

    filenames = [
        's3://satellite-data-eumetcast-seviri-rss/H-000-MSG3*202204260855*',
    ]
    storage_options = {
        "client_kwargs": {"endpoint_url": "https://PLACE-YOUR-SERVER-URL-HERE"},
        "secret": "VERYBIGSECRET",
        "key": "ACCESSKEY"
    }
    scn = Scene(reader='seviri_l1b_hrit', filenames=filenames, reader_kwargs={'storage_options': storage_options})
    scn.load(['WV_073'])

Using the `fsspec` configuration in `s3.json` the configuration would look like this::

    {
        "s3": {
            "client_kwargs": {"endpoint_url": "https://PLACE-YOUR-SERVER-URL-HERE"},
            "secret": "VERYBIGSECRET",
            "key": "ACCESSKEY"
        }
    }


Using multiple readers
======================

If multiple readers are used and the required credentials differ, the storage options are passed per reader like this::

    reader1_filenames = [...]
    reader2_filenames = [...]
    filenames = {
        'reader1': reader1_filenames,
        'reader2': reader2_filenames,
    }
    reader1_storage_options = {...}
    reader2_storage_options = {...}
    reader_kwargs = {
        'reader1': {
            'option1': 'foo',
            'storage_options': reader1_storage_options,
        },
        'reader2': {
            'option1': 'foo',
            'storage_options': reader1_storage_options,
        }
    }
    scn = Scene(filenames=filenames, reader_kwargs=reader_kwargs)


Resources
=========

See :class:`~satpy.readers.FSFile` for direct usage of `fsspec` with Satpy, and
`fsspec documentation <https://filesystem-spec.readthedocs.io/en/latest/index.html>`_ for more details on connection options
and detailes.


Supported readers
=================

.. _reader_table:

.. list-table:: Satpy Readers capable of reading remote files using `fsspec`
    :header-rows: 1
    :widths: 70 30

    * - Description
      - Reader name
    * - MSG (Meteosat 8 to 11) SEVIRI data in HRIT format
      - `seviri_l1b_hrit`
    * - GOES-R imager data in netcdf format
      - `abi_l1b`
    * - NOAA GOES-R ABI L2+ products in netcdf format
      - `abi_l2_nc`
    * - Sentinel-3 A and B OLCI Level 1B data in netCDF4 format
      - `olci_l1b`
    * - Sentinel-3 A and B OLCI Level 2 data in netCDF4 format
      - `olci_l2`
