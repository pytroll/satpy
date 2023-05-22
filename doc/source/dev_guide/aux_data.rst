Auxiliary Data Download
=======================

Sometimes Satpy components need some extra data files to get their work
done properly. These include files like Look Up Tables (LUTs), coefficients,
or Earth model data (ex. elevations). This includes any file that would be too
large to be included in the Satpy python package; anything bigger than a small
text file. To help with this, Satpy includes utilities for downloading and
caching these files only when your component is used. This saves the user from
wasting time and disk space downloading files they may never use.
This functionality is made possible thanks to the
`Pooch library <https://www.fatiando.org/pooch/latest/>`_.

Downloaded files are stored in the directory configured by
:ref:`data_dir_setting`.

Adding download functionality
-----------------------------

The utility functions for data downloading include a two step process:

1. **Registering**: Tell Satpy what files might need to be downloaded and used
   later.
2. **Retrieving**: Ask Satpy to download and store the files locally.

Registering
^^^^^^^^^^^

Registering a file for downloading tells Satpy the remote URL for the file,
and an optional hash. The hash is used to verify a successful download.
Registering can also include a ``filename`` to tell Satpy what to name the
file when it is downloaded. If not provided it will be determined from the URL.
Once registered, Satpy can be told to retrieve the file (see below) by using a
"cache key". Cache keys follow the general scheme of
``<component_type>/<filename>`` (ex. ``readers/README.rst``).

Satpy includes a low-level function and a high-level Mixin class for
registering files. The higher level class is recommended for any Satpy
component like readers, writers, and compositors. The lower-level
:func:`~satpy.aux_download.register_file` function can be used for any other
use case.

The :class:`~satpy.aux_download.DataMixIn` class is automatically included
in the :class:`~satpy.readers.yaml_reader.FileYAMLReader` and
:class:`~satpy.writers.Writer` base classes. For any other component (like
a compositor) you should include it as another parent class:

.. code-block:: python

    from satpy.aux_download import DataDownloadMixin
    from satpy.composites import GenericCompositor

    class MyCompositor(GenericCompositor, DataDownloadMixin):
        """Compositor that uses downloaded files."""

        def __init__(self, name, url=None, known_hash=None, **kwargs):
            super().__init__(name, **kwargs)
            data_files = [{'url': url, 'known_hash': known_hash}]
            self.register_data_files(data_files)

However your code registers files, to be consistent it must do it during
initialization so that the :func:`~satpy.aux_download.find_registerable_files`.
If your component isn't a reader, writer, or compositor then this function
will need to be updated to find and load your registered files. See
:ref:`offline_aux_downloads` below for more information.

As mentioned, the mixin class is included in the base reader and writer class.
To register files in these cases, include a ``data_files`` section in your
YAML configuration file. For readers this would go under the ``reader``
section and for writers the ``writer`` section. This parameter is a list
of dictionaries including a ``url``, ``known_hash``, and optional
``filename``. For example::

    reader:
        name: abi_l1b
        short_name: ABI L1b
        long_name: GOES-R ABI Level 1b
        ... other metadata ...
        data_files:
          - url: "https://example.com/my_data_file.dat"
          - url: "https://raw.githubusercontent.com/pytroll/satpy/main/README.rst"
            known_hash: "sha256:5891286b63e7745de08c4b0ac204ad44cfdb9ab770309debaba90308305fa759"
          - url: "https://raw.githubusercontent.com/pytroll/satpy/main/RELEASING.md"
            filename: "satpy_releasing.md"
            known_hash: null

See the :class:`~satpy.aux_download.DataDownloadMixin` for more information.

Retrieving
^^^^^^^^^^

Files that have been registered (see above) can be retrieved by calling the
:func:`~satpy.aux_download.retrieve` function. This function expects a single
argument: the cache key. Cache keys are returned by registering functions, but
can also be pre-determined by following the scheme
``<component_type>/<filename>`` (ex. ``readers/README.rst``).
Retrieving a file will download it to local disk if needed and then return
the local pathname. Data is stored locally in the :ref:`data_dir_setting`.
It is up to the caller to then open the file.

.. _offline_aux_downloads:

Offline Downloads
-----------------

To assist with operational environments, Satpy includes a
:func:`~satpy.aux_download.retrieve_all` function that will try to find all
files that Satpy components may need to download in the future and download
them to the current directory specified by :ref:`data_dir_setting`.
This function allows you to specify a list of ``readers``, ``writers``, or
``composite_sensors`` to limit what components are checked for files to
download.

The ``retrieve_all`` function is also available through a command line script
called ``satpy_retrieve_all_aux_data``. Run the following for usage information.

.. code-block:: bash

    satpy_retrieve_all_aux_data --help

To make sure that no additional files are downloaded when running Satpy see
:ref:`download_aux_setting`.
