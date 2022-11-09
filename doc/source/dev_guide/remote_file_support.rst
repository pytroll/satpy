======================================
Adding remote file support to a reader
======================================

.. warning::
    This feature is currently very new and might improve and change
    in the future.

As of Satpy version 0.25.1 the possibility to search for files on remote
file systems (see :ref:`search_for_files`) as well as the possibility
for supported readers to read from remote filesystems has been added.

To add this feature to a reader the call to :func:`xarray.open_dataset`
has to be replaced by the function :func:`~satpy.readers.file_handlers.open_dataset`
included in Satpy which handles passing on the filename to be opened regardless
if it is a local file path or a :class:`~satpy.readers.FSFile` object which can wrap
:func:`fsspec.open` objects.

To be able to cache the ``open_dataset`` call which is favourable for remote files
it should be separated from the ``get_dataset`` method which needs to be implemented
in every reader. This could look like:

.. code-block:: python

    from satpy import CHUNK_SIZE
    from satpy._compat importe cached_property
    from satpy.readers.file_handlers import BaseFileHandler, open_dataset

    class Reader(BaseFileHandler):

        def __init__(self, filename, filename_info, filetype_info):
            super(Reader).__init__(filename, filename_info, filetype_info):

        @cached_property
        def nc(self):
            return open_dataset(self.filename, chunks=CHUNK_SIZE)

        def get_dataset(self):
            # Access the opened dataset
            data = self.nc["key"]


Any parameters allowed for :func:`xarray.open_dataset` can be passed as
keywords to :func:`~satpy.readers.file_handlers.open_dataset` if needed.

.. note::
    It is important to know that for remote files xarray might use a different
    backend to open the file than for local files (e.g. h5netcdf instead of netcdf4),
    which might result in some attributes being returned as arrays instead of scalars.
    This has to be accounted for when accessing attributes in the reader.
