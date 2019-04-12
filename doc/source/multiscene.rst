MultiScene (Experimental)
=========================

Scene objects in SatPy are meant to represent a single geographic region at
a specific single instant in time or range of time. This means they are not
suited for handling multiple orbits of polar-orbiting satellite data,
multiple time steps of geostationary satellite data, or other special data
cases. To handle these cases SatPy provides the `MultiScene` class. The below
examples will walk through some basic use cases of the MultiScene.

.. warning::

    These features are still early in development and may change overtime as
    more user feedback is received and more features added.

Stacking scenes
---------------

The MultiScene makes it easy to take multiple Scenes and stack them on top of
each other. The code below takes two separate orbits from a VIIRS sensor and
stacks them on top of each other.

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

Saving frames of an animation
-----------------------------

The MultiScene can take "frames" of data and join them together in a single
animation movie file. Saving animations required the `imageio` python library
and for most available formats the ``ffmpeg`` command line tool suite should
also be installed. The below example saves a series of GOES-EAST ABI channel
1 and channel 2 frames to MP4 movie files. Note that currently there is no
easy way to map files from multiple time steps/orbits in to individual Scene
objects. The `glob` function and for loops are used to group files into Scene
objects that, if used individually, could load the data we want.

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
