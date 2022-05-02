====================
Reading remote files
====================

.. testsetup:: *
    >>> import sys
    >>> reload(sys)
    >>> sys.setdefaultencoding('utf8')


Some of the readers in Satpy can read data directly over various transfer protocols. This is done
using `fsspec <https://filesystem-spec.readthedocs.io/en/latest/index.html>`_ and various packages
it is using underneath. All the credential handling etc. are also done using
`fsspec configuration <https://filesystem-spec.readthedocs.io/en/latest/features.html#configuration>`_.
Simple example configs are shown below each code example.

As an example, reading ABI data from public AWS S3 storage can be done in the following way::

    from satpy import Scene

    filenames = ['s3://noaa-goes16/ABI-L1b-RadC/2019/001/17/*_G16_s20190011702186*']
    scn = Scene(reader='abi_l1b', filenames=filenames)
    scn.load(['true_color_raw'])

In addition to `fsspec` the `s3fs` library needs to be installed. The credentials and non-AWS end-point
are given in the `fsspec` configuration file for `s3` protocol, which is by default placed in
`~/.config/fsspec/s3.json` in Linux::

    {
        "s3": {
            "anon": "true"
        }
    }

For reference, reading SEVIRI HRIT data from a local S3 storage works the same way::

    filenames = [
        's3://satellite-data-eumetcast-seviri-rss/H-000-MSG3__-MSG3_RSS____-WV_073___-000006___-202204260855-__',
        's3://satellite-data-eumetcast-seviri-rss/H-000-MSG3__-MSG3_RSS____-WV_073___-000007___-202204260855-__',
        's3://satellite-data-eumetcast-seviri-rss/H-000-MSG3__-MSG3_RSS____-WV_073___-000008___-202204260855-__',
        's3://satellite-data-eumetcast-seviri-rss/H-000-MSG3__-MSG3_RSS____-_________-EPI______-202204260855-__',
        's3://satellite-data-eumetcast-seviri-rss/H-000-MSG3__-MSG3_RSS____-_________-PRO______-202204260855-__',
    ]
    scn = Scene(reader='seviri_l1b_hrit', filenames=filenames)
    scn.load(['WV_073'])

As this is a private resource, credentials and server end-point need to be configured::

    {
        "s3": {
            "client_kwargs": {"endpoint_url": "https://PLACE-YOUR-SERVER-URL-HERE"},
            "secret": "VERYBIGSECRET",
            "key": "ACCESSKEY"
        }
    }
