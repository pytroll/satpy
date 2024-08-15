Advanced topics
===============

This section of the documentation contains advanced topics that are not relevant
for most users.

.. _preload:

Preloading segments for improved timeliness
-------------------------------------------

Normally, data need to exist before they can be read.  This requirement
impacts data processing timeliness.  For data arriving in segments,
Satpy can process each segment immediately as it comes in.
This feature currently only works with the :mod:`~satpy.readers.fci_l1c_nc` reader.
This is experimental and likely to be instable and might change.

Consider a near real time data reception situation where FCI segments are
delivered one by one.  Classically, to produce full disc imagery, users
would wait for all needed segments to arrive, before they start processing
any data by passing all segments to the :class:`~satpy.scene.Scene`.
For a more timely imagery production, users can create the Scene, load the
data, resample, and even call :meth:`~satpy.scene.Scene.save_datasets`
before the data are complete (:meth:`~satpy.scene.Scene.save_datasets` will wait until the data
are available, unless ``compute=False``).
Upon computation, much of the overhead in Satpy
internals has already been completed, and Satpy will process each segment
as it comes in.

To do so, Satpy caches a selection of data and metadata between segments
and between repeat cycles.  Caching between segments happens in-memory
and needs no preparation from the user, but where data are cached
between repeat cycles, the user needs to create this cache first from
a repeat cycle that is available completely::

  >>> from satpy.readers import create_preloadable_cache
  >>> create_preloadable_cache("fci_l1c_nc", fci_files)

This needs to be done only once as long as data or metadata cached
between repeat cycles does not change (for example, the rows at which
each repeat cycle starts and ends).  To make use of eager processing, set
the configuration variable ``readers.preload_segments``.  When creating
the scene, pass only the path to the first segment::

  >>> satpy.config.set({"readers.preload_segments": True})
  >>> sc = Scene(
  ...   filenames=[path_to_first_segment],
  ...   reader="fci_l1c_nc")

Satpy will figure out the names of the remaining segments and find them as
they come in.  If the data are already available, processing is similar to
the regular case.  If the data are not yet available, Satpy will wait during
the computation of the dask graphs until data become available.

For additional configuration parameters, see the :ref:`configuration documentation <preload_settings>`.

Known limitations as of Satpy 0.51:

- Mixing different file types for the same reader is not yet supported.
  For FCI, that means it is not yet possible to mix FDHSI and HRFI data.
- When segments are missing, processing times out and no image will be produced.
  There is currently no way to produce an incomplete image with the missing
  segment left out.
- Dask may not order the processing of the chunks optimally.  That means some
  dask workers may be waiting for chunks 33–40 as chunks 1–32 are coming in
  and are not being processed.  Possible workarounds:
  - Use as many workers are there are chunks (for example, 40).
  - Use the dask distributed scheduler using
    ``from dask.distributed import Client; Client()``.  This has only
    limited support in Satpy and is highly experimental.   It should be possible
    to read FCI L1C data, resample it
    using the gradient search resampler, and write the resulting data using the
    ``simple_image`` writer.  The nearest neighbour resampler or the GeoTIFF
    writer do not currently work (see https://github.com/pytroll/satpy/issues/1762).
    If you use this scheduler, set the configuration variable
    ``readers.preload_dask_distributed`` to True.
    This is not currently recommended in a production environment.  Any feedback
    is highly welcome.
- Currently, Satpy merely checks the existence of a file and not whether it
  has been completely written.  This may lead to incomplete files being read,
  which might lead to failures.

For more technical background reading including hints
on how this could be extended to other readers, see the API documentations for
:class:`~satpy.readers.netcdf_utils.PreloadableSegments` and
:class:`~satpy.readers.yaml_reader.GEOSegmentYAMLReader`.

.. versionadded:: 0.52

.. toctree::
    :hidden:
    :maxdepth: 1

    preloaded_reading
