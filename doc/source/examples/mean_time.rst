Tracking measurement time
=========================

.. versionadded:: v0.58

For some readers and writers, it is possible to keep track of pixel-level
measurement times and store the average measurement time in the metadata
for the resampled image.  This can be stored in the filename or the file
headers (or both).  It is only supported for selected readers and writers.

By default, Satpy does not keep track of measurement times.  To keep track
of measurement times, we must first tell the reader to add such times to
the metadata of each dataset.  With supported readers, this can be done
by passing ``reader_kwargs={"track_time": True}`` to :meth:`~satpy.scene.Scene`:

.. code-block:: python

   sc = Scene(filenames={"seviri_l1b_hrit": seviri_files}, reader_kwargs={"track_time": True})
   sc.load(["IR_108"])

The time is stored as a coordinate:

.. code-block:: python

   sc["IR_108"].coords["time"]

To retain it upon resampling, pass ``resample_coords=True`` to :meth:`~satpy.scene.Scene.resample`:

.. code-block:: python

   ls = sc.resample("eurol", resample_coords=True)

For supported writers, it can be stored in the headers by passing ``dynamic_fields={"mean_time"}``
to :meth:`~satpy.scene.Scene.save_datasets`.  Storing in the filename is not currently
supported, because the filename is normally constructed before any values are calculated,
and calculating the mean time would trigger an early dask computation.

Consult the documentation for specific writers for details on how the mean time may be written
to the headers.
