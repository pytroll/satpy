MultiScene (Experimental)
=========================

Scene objects in Satpy are meant to represent a single geographic region at
a specific single instant in time or range of time. This means they are not
suited for handling multiple orbits of polar-orbiting satellite data,
multiple time steps of geostationary satellite data, or other special data
cases. To handle these cases Satpy provides the `MultiScene` class. The below
examples will walk through some basic use cases of the MultiScene.

.. warning::

    These features are still early in development and may change overtime as
    more user feedback is received and more features added.

Blending Scenes in MultiScene
-----------------------------
Scenes contained in a MultiScene can be combined in different ways.

Stacking scenes
***************

The code below uses the :meth:`~satpy.multiscene.MultiScene.blend` method of
the ``MultiScene`` object to stack two separate orbits from a VIIRS sensor. By
default the ``blend`` method will use the :func:`~satpy.multiscene.stack`
function which uses the first dataset as the base of the image and then
iteratively overlays the remaining datasets on top.

    >>> from satpy import Scene, MultiScene
    >>> from glob import glob
    >>> from pyresample.geometry import AreaDefinition
    >>> my_area = AreaDefinition(...)
    >>> scenes = [
    ...    Scene(reader='viirs_sdr', filenames=glob('/data/viirs/day_1/*t180*.h5')),
    ...    Scene(reader='viirs_sdr', filenames=glob('/data/viirs/day_2/*t180*.h5'))
    ... ]
    >>> mscn = MultiScene(scenes)
    >>> mscn.load(['I04'])
    >>> new_mscn = mscn.resample(my_area)
    >>> blended_scene = new_mscn.blend()
    >>> blended_scene.save_datasets()

Timeseries
**********

Using the :meth:`~satpy.multiscene.MultiScene.blend` method with the
:func:`~satpy.multiscene.timeseries` function will combine
multiple scenes from different time slots by time. A single `Scene` with each
dataset/channel extended by the time dimension will be returned. If used
together with the :meth:`~satpy.scene.Scene.to_geoviews` method, creation of
interactive timeseries Bokeh plots is possible.

    >>> from satpy import Scene, MultiScene
    >>> from satpy.multiscene import timeseries
    >>> from glob import glob
    >>> from pyresample.geometry import AreaDefinition
    >>> my_area = AreaDefinition(...)
    >>> scenes = [
    ...    Scene(reader='viirs_sdr', filenames=glob('/data/viirs/day_1/*t180*.h5')),
    ...    Scene(reader='viirs_sdr', filenames=glob('/data/viirs/day_2/*t180*.h5'))
    ... ]
    >>> mscn = MultiScene(scenes)
    >>> mscn.load(['I04'])
    >>> new_mscn = mscn.resample(my_area)
    >>> blended_scene = new_mscn.blend(blend_function=timeseries)
    >>> blended_scene['I04']
    <xarray.DataArray (time: 2, y: 1536, x: 6400)>
    dask.array<shape=(2, 1536, 6400), dtype=float64, chunksize=(1, 1536, 4096)>
    Coordinates:
      * time     (time) datetime64[ns] 2012-02-25T18:01:24.570942 2012-02-25T18:02:49.975797
    Dimensions without coordinates: y, x

Saving frames of an animation
-----------------------------

The MultiScene can take "frames" of data and join them together in a single
animation movie file. Saving animations requires the `imageio` python library
and for most available formats the ``ffmpeg`` command line tool suite should
also be installed. The below example saves a series of GOES-EAST ABI channel
1 and channel 2 frames to MP4 movie files. We can use the
:meth:`MultiScene.from_files <satpy.multiscene.MultiScene.from_files>` class
method to create a `MultiScene` from a series of files. This uses the
:func:`~satpy.readers.group_files` utility function to group files by start
time.

    >>> from satpy import Scene, MultiScene
    >>> from glob import glob
    >>> mscn = MultiScene.from_files(glob('/data/abi/day_1/*C0[12]*.nc'), reader='abi_l1b')
    >>> mscn.load(['C01', 'C02'])
    >>> mscn.save_animation('{name}_{start_time:%Y%m%d_%H%M%S}.mp4', fps=2)

.. versionadded:: 0.12

    The ``from_files`` and ``group_files`` functions were added in Satpy 0.12.
    See below for an alternative solution.

This will compute one video frame (image) at a time and write it to the MPEG-4
video file. For users with more powerful systems it is possible to use
the ``client`` and ``batch_size`` keyword arguments to compute multiple frames
in parallel using the dask ``distributed`` library (if installed).
See the :doc:`dask distributed <dask:setup/single-distributed>` documentation
for information on creating a ``Client`` object. If working on a cluster
you may want to use :doc:`dask jobqueue <jobqueue:index>` to take advantage
of multiple nodes at a time.

For older versions of Satpy we can manually create the `Scene` objects used.
The :func:`~glob.glob` function and for loops are used to group files into
Scene objects that, if used individually, could load the data we want. The
code below is equivalent to the ``from_files`` code above:

    >>> from satpy import Scene, MultiScene
    >>> from glob import glob
    >>> scene_files = []
    >>> for time_step in ['1800', '1810', '1820', '1830']:
    ...     scene_files.append(glob('/data/abi/day_1/*C0[12]*s???????{}*.nc'.format(time_step)))
    >>> scenes = [
    ...     Scene(reader='abi_l1b', filenames=files) for files in sorted(scene_files)
    ... ]
    >>> mscn = MultiScene(scenes)
    >>> mscn.load(['C01', 'C02'])
    >>> mscn.save_animation('{name}_{start_time:%Y%m%d_%H%M%S}.mp4', fps=2)

.. warning::

    GIF images, although supported, are not recommended due to the large file
    sizes that can be produced from only a few frames.

Saving multiple scenes
----------------------

The ``MultiScene`` object includes a
:meth:`~satpy.multiscene.MultiScene.save_datasets` method for saving the
data from multiple Scenes to disk. By default this will operate on one Scene
at a time, but similar to the ``save_animation`` method above this method can
accept a dask distributed ``Client`` object via the ``client`` keyword
argument to compute scenes in parallel (see documentation above). Note however
that some writers, like the ``geotiff`` writer, do not support multi-process
operations at this time and will fail when used with dask distributed. To save
multiple Scenes use:

    >>> from satpy import Scene, MultiScene
    >>> from glob import glob
    >>> mscn = MultiScene.from_files(glob('/data/abi/day_1/*C0[12]*.nc'), reader='abi_l1b')
    >>> mscn.load(['C01', 'C02'])
    >>> mscn.save_datasets(base_dir='/path/for/output')
