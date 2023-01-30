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

MultiScene Creation
-------------------
There are two ways to create a ``MultiScene``. Either by manually creating and
providing the scene objects,

    >>> from satpy import Scene, MultiScene
    >>> from glob import glob
    >>> scenes = [
    ...    Scene(reader='viirs_sdr', filenames=glob('/data/viirs/day_1/*t180*.h5')),
    ...    Scene(reader='viirs_sdr', filenames=glob('/data/viirs/day_2/*t180*.h5'))
    ... ]
    >>> mscn = MultiScene(scenes)
    >>> mscn.load(['I04'])

or by using the :meth:`MultiScene.from_files <satpy.multiscene.MultiScene.from_files>`
class method to create a ``MultiScene`` from a series of files. This uses the
:func:`~satpy.readers.group_files` utility function to group files by start
time or other filenames parameters.

   >>> from satpy import MultiScene
   >>> from glob import glob
   >>> mscn = MultiScene.from_files(glob('/data/abi/day_1/*C0[12]*.nc'), reader='abi_l1b')
   >>> mscn.load(['C01', 'C02'])

.. versionadded:: 0.12

    The ``from_files`` and ``group_files`` functions were added in Satpy 0.12.
    See below for an alternative solution.

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


Stacking scenes using weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is also possible to blend scenes together in a bit more sophisticated manner
using pixel based weighting instead of just stacking the scenes on top of each
other as described above. This can for instance be useful to make a cloud
parameter (cover, height, etc) composite combining cloud parameters derived
from both geostationary and polar orbiting satellite data close in time and
over a given area. This is useful for instance at high latitudes where
geostationary data degrade quickly with latitude and polar data are more
frequent.

This weighted blending can be accomplished via the use of the builtin
:func:`~functools.partial` function (see `Partial
<https://docs.python.org/3/library/functools.html#partial-objects>`_) and the
default :func:`~satpy.multiscene.stack` function. The
:func:`~satpy.multiscene.stack` function can take the optional argument
`weights` (`None` on default) which should be a sequence (of length equal to
the number of scenes being blended) of arrays with pixel weights.

The code below gives an example of how two cloud scenes can be blended using
the satellite zenith angles to weight which pixels to take from each of the two
scenes. The idea being that the reliability of the cloud parameter is higher
when the satellite zenith angle is small.

    >>> from satpy import Scene, MultiScene,  DataQuery
    >>> from functools import partial
    >>> from satpy.resample import get_area_def
    >>> areaid = get_area_def("myarea")
    >>> geo_scene = Scene(filenames=glob('/data/to/nwcsaf/geo/files/*nc'), reader='nwcsaf-geo')
    >>> geo_scene.load(['ct'])
    >>> polar_scene = Scene(filenames=glob('/data/to/nwcsaf/pps/noaa18/files/*nc'), reader='nwcsaf-pps_nc')
    >>> polar_scene.load(['cma', 'ct'])
    >>> mscn = MultiScene([geo_scene, polar_scene])
    >>> groups = {DataQuery(name='CTY_group'): ['ct']}
    >>> mscn.group(groups)
    >>> resampled = mscn.resample(areaid, reduce_data=False)
    >>> weights = [1./geo_satz, 1./n18_satz]
    >>> stack_with_weights = partial(stack, weights=weights)
    >>> blended = resampled.blend(blend_function=stack_with_weights)
    >>> blended_scene.save_dataset('CTY_group', filename='./blended_stack_weighted_geo_polar.nc')



Grouping Similar Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^

By default, ``MultiScene`` only operates on datasets shared by all scenes.
Use the :meth:`~satpy.multiscene.MultiScene.group` method to specify groups
of datasets that shall be treated equally by ``MultiScene``, even if their
names or wavelengths are different.

Example: Stacking scenes from multiple geostationary satellites acquired at
roughly the same time. First, create scenes and load datasets individually:

    >>> from satpy import Scene
    >>> from glob import glob
    >>> h8_scene = satpy.Scene(filenames=glob('/data/HS_H08_20200101_1200*'),
    ...                        reader='ahi_hsd')
    >>> h8_scene.load(['B13'])
    >>> g16_scene = satpy.Scene(filenames=glob('/data/OR_ABI*s20200011200*.nc'),
    ...                         reader='abi_l1b')
    >>> g16_scene.load(['C13'])
    >>> met10_scene = satpy.Scene(filenames=glob('/data/H-000-MSG4*-202001011200-__'),
    ...                           reader='seviri_l1b_hrit')
    >>> met10_scene.load(['IR_108'])

Now create a ``MultiScene`` and group the three similar IR channels together:

    >>> from satpy import MultiScene, DataQuery
    >>> mscn = MultiScene([h8_scene, g16_scene, met10_scene])
    >>> groups = {DataQuery('IR_group', wavelength=(10, 11, 12)): ['B13', 'C13', 'IR_108']}
    >>> mscn.group(groups)

Finally, resample the datasets to a common grid and blend them together:

    >>> from pyresample.geometry import AreaDefinition
    >>> my_area = AreaDefinition(...)
    >>> resampled = mscn.resample(my_area, reduce_data=False)
    >>> blended = resampled.blend()  # you can also use a custom blend function

You can access the results via ``blended['IR_group']``.


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
1 and channel 2 frames to MP4 movie files.

    >>> from satpy import Scene, MultiScene
    >>> from glob import glob
    >>> mscn = MultiScene.from_files(glob('/data/abi/day_1/*C0[12]*.nc'), reader='abi_l1b')
    >>> mscn.load(['C01', 'C02'])
    >>> mscn.save_animation('{name}_{start_time:%Y%m%d_%H%M%S}.mp4', fps=2)

This will compute one video frame (image) at a time and write it to the MPEG-4
video file. For users with more powerful systems it is possible to use
the ``client`` and ``batch_size`` keyword arguments to compute multiple frames
in parallel using the dask ``distributed`` library (if installed).
See the :doc:`dask distributed <dask:deploying-python>` documentation
for information on creating a ``Client`` object. If working on a cluster
you may want to use :doc:`dask jobqueue <jobqueue:index>` to take advantage
of multiple nodes at a time.

It is possible to add an overlay or decoration to each frame of an
animation.  For text added as a decoration, string substitution will be
applied based on the attributes of the dataset, for example:

    >>> mscn.save_animation(
    ...     "{name:s}_{start_time:%Y%m%d_%H%M}.mp4",
    ...     enh_args={
    ...     "decorate": {
    ...         "decorate": [
    ...             {"text": {
    ...                 "txt": "time {start_time:%Y-%m-%d %H:%M}",
    ...                 "align": {
    ...                     "top_bottom": "bottom",
    ...                     "left_right": "right"},
    ...                 "font": '/usr/share/fonts/truetype/arial.ttf',
    ...                 "font_size": 20,
    ...                 "height": 30,
    ...                 "bg": "black",
    ...                 "bg_opacity": 255,
    ...                 "line": "white"}}]}})

If your file covers ABI MESO data for an hour for channel 2 lasting
from 2020-04-12 01:00-01:59, then the output file will be called
``C02_20200412_0100.mp4`` (because the first dataset/frame corresponds to
an image that started to be taken at 01:00), consist of sixty frames (one
per minute for MESO data), and each frame will have the start time for
that frame floored to the minute blended into the frame.  Note that this
text is "burned" into the video and cannot be switched on or off later.

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

Combining multiple readers
--------------------------

.. versionadded:: 0.23

The :meth:`~satpy.multiscene.MultiScene.from_files` constructor allows to
automatically combine multiple readers into a single MultiScene.  It is no
longer necessary for the user to create the :class:`~satpy.scene.Scene`
objects themselves.  For example, you can combine Advanced Baseline
Imager (ABI) and Global Lightning Mapper (GLM) measurements.
Constructing a multi-reader MultiScene requires more parameters than a
single-reader MultiScene, because Satpy can poorly guess how to group
files belonging to different instruments.  For an example creating
a video with lightning superimposed on ABI channel 14 (11.2 µm)
using the built-in composite ``C14_flash_extent_density``,
which superimposes flash extent density from GLM (read with the
:class:`~satpy.readers.glm_l2.NCGriddedGLML2` or ``glm_l2`` reader) on ABI
channel 14 data (read with the :class:`~satpy.readers.abi_l1b.NC_ABI_L1B`
or ``abi_l1b`` reader), and therefore needs Scene objects that combine
both readers:

    >>> glm_dir = "/path/to/GLMC/"
    >>> abi_dir = "/path/to/ABI/"
    >>> ms = satpy.MultiScene.from_files(
    ...        glob.glob(glm_dir + "OR_GLM-L2-GLMC-M3_G16_s202010418*.nc") +
    ...        glob.glob(abi_dir + "C*/OR_ABI-L1b-RadC-M6C*_G16_s202010418*_e*_c*.nc"),
    ...        reader=["glm_l2", "abi_l1b"],
    ...        ensure_all_readers=True,
    ...        group_keys=["start_time"],
    ...        time_threshold=30)
    >>> ms.load(["C14_flash_extent_density"])
    >>> ms = ms.resample(ms.first_scene["C14"].attrs["area"])
    >>> ms.save_animation("/path/for/output/{name:s}_{start_time:%Y%m%d_%H%M}.mp4")

In this example, we pass to
:meth:`~satpy.multiscene.MultiScene.from_files` the additional parameters
``ensure_all_readers=True, group_keys=["start_time"], time_threshold=30``
so we only get scenes at times that both ABI and GLM have a file starting
within 30 seconds from each other, and ignore all other differences for
the purposes of grouping the two.  For this example, the ABI files occur
every 5 minutes but the GLM files (processed with glmtools) every minute.
Scenes where there is a GLM file without an ABI file starting within at
most ±30 seconds are skipped.  The ``group_keys`` and ``time_threshold``
keyword arguments are processed by the :func:`~satpy.readers.group_files`
function.  The heavy work of blending the two instruments together is
performed by the :class:`~satpy.composites.BackgroundCompositor` class
through the `"C14_flash_extent_density"` composite.
