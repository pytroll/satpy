Tracking valid time
========================

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

For supported writers, it can be stored in the filename by passing ``dynamic_fields={"valid_time"}``
to :meth:`~satpy.scene.Scene.save_datasets`:

.. code-block:: python

   ls.save_datasets(
       writer="geotiff",
       filename="{platform_name}-{sensor}-{name}-{area.area_id}-{start_time:%Y%m%d%H%M}-{valid_time:%Y%m%d%H%M%S}.tif",
       dynamic_fields={"valid_time"})

For supported writers, valid time may also be written to the headers.  Consult
the documentation of your writer for details.
