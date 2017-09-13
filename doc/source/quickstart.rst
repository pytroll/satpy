============
 Quickstart
============

Loading data
============

.. versionchanged:: 2.0.0-alpha.1
   New syntax

.. testsetup:: *
    >>> import sys
    >>> reload(sys)
    >>> sys.setdefaultencoding('utf8')

To work with weather satellite data, one has to create an instance of the :class:`Scene` class. In order for satpy to
get access to the data, either the current wording directory has to be set to the directory containing the data
files, or the `base_dir` keyword argument has to be provided on scene creation::

    >>> import os
    >>> os.chdir("/home/a001673/data/satellite/Meteosat-10/seviri/lvl1.5/2015/04/20/HRIT")
    >>> from satpy import Scene
    >>> from datetime import datetime
    >>> time_slot = datetime(2015, 4, 20, 10, 0)
    >>> global_scene = Scene(platform_name="Meteosat-10", sensor="seviri", reader="hrit_msg", start_time=datetime(2015, 4, 20, 10, 0))

or::

    >>> from satpy.scene import Scene
    >>> from datetime import datetime
    >>> time_slot = datetime(2015, 4, 20, 10, 0)
    >>> global_scene = Scene(platform_name="Meteosat-10", sensor="seviri", reader="hrit_msg", start_time=datetime(2015, 4, 20, 10, 0), base_dir="/home/a001673/data/satellite/Meteosat-10/seviri/lvl1.5/2015/04/20/HRIT") # doctest: +SKIP
    >>>

For some platforms, it might be necessary to also specify an `end_time`::

    >>> Scene(platform_name="SNPP", sensor="viirs", start_time=datetime(2015, 3, 11, 11, 20), end_time=datetime(2015, 3, 11, 11, 26)) # doctest: +SKIP

Loading weather satellite data with satpy is as simple as calling the  :meth:`Scene.load` method::

    >>> global_scene.load([0.6, 0.8, 10.8])
    >>> print global_scene

    seviri/IR_108:
            area: On-the-fly area
            start_time: 2015-04-20 10:00:00
            units: K
            wavelength_range: (9.8, 10.8, 11.8) μm
            shape: (3712, 3712)
    seviri/VIS006:
            area: On-the-fly area
            start_time: 2015-04-20 10:00:00
            units: %
            wavelength_range: (0.56, 0.635, 0.71) μm
            shape: (3712, 3712)
    seviri/VIS008:
            area: On-the-fly area
            start_time: 2015-04-20 10:00:00
            units: %
            wavelength_range: (0.74, 0.81, 0.88) μm
            shape: (3712, 3712)

As you can see, this loads the visible and IR channels provided as argument to the :meth:`load` method as a
list of wavelengths in micrometers. Another way to load the channels is to provide the names instead::

    >>> global_scene.load(["VIS006", "VIS008", "IR_108"])
    >>> print global_scene

To have a look at the available bands you should be able to load with your `Scene` object, you can call the
:meth:`available_datasets` method::

    >>> global_scene.available_dataset_names()

    [u'HRV',
     u'IR_108',
     u'IR_120',
     u'VIS006',
     u'WV_062',
     u'IR_039',
     u'IR_134',
     u'IR_097',
     u'IR_087',
     u'VIS008',
     u'IR_016',
     u'WV_073']


To access the loaded data::

    >>> print global_scene[0.6]

or::

    >>> print global_scene["VIS006"]

To visualize it::

    >>> global_scene.show(0.6)

To combine them::

    >>> global_scene["ndvi"] = (global_scene[0.8] - global_scene[0.6]) / (global_scene[0.8] + global_scene[0.6])
    >>> global_scene.show("ndvi")


Generating composites
=====================

The easiest way to generate composites is to `load` them::

    >>> global_scene.load(['overview'])
    >>> global_scene.show('overview')

To get a list of all available composites for the current scene::

    >>> global_scene.available_composites()

    [u'overview_sun',
     u'airmass',
     u'natural',
     u'night_fog',
     u'overview',
     u'green_snow',
     u'dust',
     u'fog',
     u'natural_sun',
     u'cloudtop',
     u'convection',
     u'ash']

To save a composite to disk::

    >>> global_scene.save_dataset('overview', 'my_nice_overview.png')

One can also specify which writer to use for filenames with non-standard extensions ::

    >>> global_scene.save_dataset('overview', 'my_nice_overview.stupidextension', writer='geotiff')


Resampling
==========

.. todo::
   Explain where and how to define new areas

Until now, we have used the channels directly as provided by the satellite,
that is in satellite projection. Generating composites thus produces views in
satellite projection, *i.e.* as viewed by the satellite.

Most often however, we will want to resample the data onto a specific area so
that only the area of interest is depicted in the RGB composites.

Here is how we do that::

    >>> local_scene = global_scene.resample("eurol")
    >>>

Now we have resampled channel data and composites onto the "eurol" area in the `local_scene` variable
and we can operate as before to display and save RGB composites::

    >>> local_scene.show('overview')
    >>> local_scene.save_dataset('overview', './local_overview.tif')

The image is automatically saved here in GeoTiff_ format.

.. _GeoTiff: http://trac.osgeo.org/geotiff/

The default resampling method is nearest neighbour. Also bilinear interpolation
is available, which can be used by adding `resampler="bilinear"` keyword:

    >>> local_scene = global_scene.resample("euro4", resampler="bilinear")
    >>>

To make resampling faster next time (when resampling geostationary satellite
data), it is possible to save the resampling coefficients and use more CPUs
when calculating the coefficients on the first go:

    >>> local_scene = global_scene.resample("euro4", resampler="bilinear",
    ...                                     nprocs=4, cache_dir="/var/tmp")
    >>>

Making custom composites
========================

Building custom composites makes use of the :class:`RGBCompositor` class. For example,
building an overview composite can be done manually with::

    >>> from satpy.composites import RGBCompositor
    >>> compositor = RGBCompositor("myoverview", "bla", "")
    >>> composite = compositor([local_scene[0.6],
    ...                         local_scene[0.8],
    ...                         local_scene[10.8]])
    >>> from satpy.writers import to_image
    >>> img = to_image(composite)
    >>> img.invert([False, False, True])
    >>> img.stretch("linear")
    >>> img.gamma(1.7)
    >>> img.show()


One important thing to notice is that there is an internal difference between a composite and an image. A composite
is defined as a special dataset which may have several bands (like R, G, B bands). However, the data isn't stretched,
or clipped or gamma filtered until an image is generated.


To save the custom composite, the following procedure can be used:

1. Create a custom directory for your custom configs.
2. Set it in the environment variable called PPP_CONFIG_DIR.
3. Write config files with your changes only (look at eg satpy/etc/composites/seviri.yaml for inspiration), pointing to the custom module containing your composites. Don't forget to add changes to the enhancement/generic.cfg file too.
4. Put your composites module on the python path.

With that, you should be able to load your new composite directly.


Colorizing and Palettizing using user-supplied colormaps
========================================================

It is possible to create single channel "composites" that are then colorized 
using users' own colormaps.  The colormaps are Numpy arrays with shape 
(num, 3), see the example below how to create the mapping file(s).

This example creates a 2-color colormap, and we interpolate the colors between 
the defined temperature ranges.  Beyond those limits the image clipped to 
the specified colors.

    >>> import numpy as np
    >>> from satpy.composites import BWCompositor
    >>> from satpy.enhancements import colorize
    >>> from satpy.writers import to_image
    >>> arr = np.array([[0, 0, 0], [255, 255, 255]])
    >>> np.save("/tmp/binary_colormap.npy", arr)
    >>> compositor = BWCompositor("test", standard_name="colorized_ir_clouds")
    >>> composite = compositor((local_scene[10.8], ))
    >>> img = to_image(composite)
    >>> kwargs = {"palettes": [{"filename": "/tmp/binary_colormap.npy",
    ...           "min_value": 223.15, "max_value": 303.15}]}
    >>> colorize(img, **kwargs)
    >>> img.show()

Similarly it is possible to use discreet values without color interpolation 
using `palettize()` instead of `colorize()`

You can define several colormaps and ranges in the `palettes` list and they 
are merged together.  See trollimage_ documentation for more information how 
colormaps and color ranges are merged.

The above example can be used in enhancements YAML config like this:

.. code-block:: yaml

  hot_or_cold:
    standard_name: hot_or_cold
    operations:
      - name: colorize
        method: &colorizefun !!python/name:satpy.enhancements.colorize ''
        kwargs:
          palettes:
            - {filename: /tmp/binary_colormap.npy, min_value: 223.15, max_value: 303.15}


.. _trollimage: http://trollimage.readthedocs.io/en/latest/


.. todo::
   How to save custom-made composites

.. todo::
   How to read cloud products from NWCSAF software.
