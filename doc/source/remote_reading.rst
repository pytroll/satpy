====================
Reading remote files
====================

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

Reading from S3 as above requires the `s3fs` library to be installed in addition to `fsspec`.

As an alternative, the storage options can be given using
`fsspec configuration <https://filesystem-spec.readthedocs.io/en/latest/features.html#configuration>`_.
For the above example, the configuration could be saved to `s3.json` in the `fsspec` configuration directory
(by default placed in `~/.config/fsspec/` directory in Linux)::

    {
        "s3": {
            "anon": "true"
        }
    }

.. note::

    Options given in `reader_kwargs` override only the matching options given in configuration file and everythin else is left
    as-is. In case of problems in data access, remove the configuration file to see if that solves the issue.


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


Caching the remote files
========================

Caching the remote file locally can speedup the overall processing time significantly, especially if the data are re-used
for example when testing. The caching can be done by taking advantage of the `fsspec caching mechanism
<https://filesystem-spec.readthedocs.io/en/latest/features.html#caching-files-locally>`_::

    reader_kwargs = {
        'storage_options': {
            's3': {'anon': True},
            'simple': {
                'cache_storage': '/tmp/s3_cache',
            }
        }
    }

    filenames = ['simplecache::s3://noaa-goes16/ABI-L1b-RadC/2019/001/17/*_G16_s20190011702186*']
    scn = Scene(reader='abi_l1b', filenames=filenames, reader_kwargs=reader_kwargs)
    scn.load(['true_color_raw'])
    scn2 = scn.resample(scn.coarsest_area(), resampler='native')
    scn2.save_datasets(base_dir='/tmp/', tiled=True, blockxsize=512, blockysize=512, driver='COG', overviews=[])


The following table shows the timings for running the above code with different cache statuses::

.. _cache_timing_table:

.. list-table:: Processing times without and with caching
    :header-rows: 1
    :widths: 40 30 30

    * - Caching
      - Elapsed time
      - Notes
    * - No caching
      - 650 s
      - remove `reader_kwargs` and `simplecache::` from the code
    * - File cache
      - 66 s
      - Initial run
    * - File cache
      - 13 s
      - Second run

.. note::

    The cache is not cleaned by Satpy nor fsspec so the user should handle cleaning excess files from `cache_storage`.


.. note::

    Only `simplecache` is considered thread-safe, so using the other caching mechanisms may or may not work depending
    on the reader, Dask scheduler or the phase of the moon.


Resources
=========

See :class:`~satpy.readers.FSFile` for direct usage of `fsspec` with Satpy, and
`fsspec documentation <https://filesystem-spec.readthedocs.io/en/latest/index.html>`_ for more details on connection options
and detailes.
