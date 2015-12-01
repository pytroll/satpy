.. -*- coding: utf-8 -*-

.. meta::
   :description: Reading Meteosat SEVIRI HRIT files with python
   :keywords: Meteosat, SEVIRI, LRIT, HRIT, reader, read, reading, python, pytroll


===========================
 Quickstart with MSG SEVIRI
===========================

For this tutorial, we will use the Meteosat data in the uncompressed EUMETSAT HRIT format, read it through mipp_ into
mpop_, resample it with pyresample_ and process it a bit. Install theses packages first.

Software to uncompress HRIT can be obtained from EUMETSAT (register and download
the `Public Wavelet Transform Decompression Library Software`_)

First example: Loading data
===========================
This example assumes uncompressed EUMETSAT HRIT data for 2015-04-20 10:00 exists in the current directory.

.. testsetup:: *
    >>> import sys
    >>> reload(sys)
    >>> sys.setdefaultencoding('utf8')
    >>> import os
    >>> os.chdir("/home/a001673/data/satellite/Meteosat-10/seviri/lvl1.5/2015/04/20/HRIT")

First, we will create a scene and load some full disk data.

    >>> from mpop.scene import Scene
    >>> from datetime import datetime
    >>> time_slot = datetime(2015, 4, 20, 10, 0)
    >>> global_scene = Scene(platform_name="Meteosat-10", sensor="seviri", start_time=datetime(2015, 4, 20, 10, 0))
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


In this example, we create an mpop_ scene object (:attr:`global_scene`) for the seviri instrument
onboard Meteosat-10, specifying the time of the scene of interest. The time is defined as a datetime object.

The next step is loading the data. This is done using mipp_ in the background, which takes care of
reading the HRIT data, and slicing the data so that we read just what is
needed. Calibration is also done with mipp_.


The :meth:`get_area_def` function reads an area definition from the configuration file  *area.def* in the PPP_CONFIG_DIR. The area definition is read into the variable :attr:`europe` which then gives access information about the area like projection and extent. 

Here we call the :meth:`load`function with a list of the wavelengths of the channels we are interested in, and the
area extent in satellite projection of the area of interest. Each retrieved channel is the closest in terms of
central wavelength, provided that the required wavelength is within the bounds of the channel. Note: If you have not
installed the numexpr_ package on your system you get the warning *"Module numexpr not found. Performance will be slower"*. This only affects the speed of loading SEVIRI data.

The wavelengths are given in micrometers and have to be given as a floating
point number (*i.e.*, don't type '1', but '1.0'). Using an integer number
instead returns a channel based on resolution, while using a string retrieves a
channels based on its name.

Retrieving the same channels based on channel name would be

    >>> europe = get_area_def("eurol")
    >>> global_scene.load([0.6, 0.8, 10.8], area_defs=europe)
    >>> global_scene.load(['VIS006', 'VIS008', 'IR_108'], area_extent=europe.area_extent)

The :attr:`area_extent` keyword argument in the :meth:`load` method specifies the subsection of the image to load in satellite projection coordinates. In this case the *EuropeCanary* is an area definition in the *geos* projection defined in the *area.def* file used by mpop_ (this area is provided in the mpop_ template *area.def*). If the :attr:`area_extent` keyword argument is not provided the full globe image is loaded.

Making RGB composites
=====================
The :meth:`load` functions return an mpop_ scene object (:attr:`global_scene`). The scene object is composed with an object named :attr:`image` which handles the creation of RGBs

    >>> from mpop.resample import get_area_def
    >>> from mpop.writers import get_enhanced_image
    >>> global_scene.load(["overview"])
    >>> img = get_enhanced_image(global_scene["overview"])
    >>> img.save("./myoverview.png")


.. image:: images/myoverview.png

Here we use the loaded data to generate an overview RGB composite image, and
save it as a png image. Instead of :meth:`save`, one could also use
:meth:`show` if the only purpose is to display the image on screen.

We want more !
==============

In the last example, the composite generation worked because the channels
needed for the overview (0.6, 0.8, 10.8 μm) were loaded. If we try to generate
a day natural color composite, which requires also the 1.6um channel, it will
result in an error::

    >>> global_scene.load(["natural"])
    >>> img = get_enhanced_image(global_scene["natural"])


So it means that we have to load the missing channel first. To do this we could
enter the channels list to load manually, as we did for the overview, but we
provide a way to get the list of channels needed by a given method using the
:attr:`prerequisites` method attribute::

    >>> print global_scene.compositors["natural"].prerequisites

Now you can save the image::

    >>> img.save("./mynaturalcolors.png")
    >>>

.. image:: images/mynaturalcolors.png

If you want to combine several prerequisites for channel loading, since
prerequisites are python sets, you can do::

    >>> global_scene.load(["overview", "natural"])


and add as many :attr:`| global_scene.image.mymethod.prerequisites` as needed.

A description of the available builtin composites for SEVIRI and VISIR derived sensors can be seen using::

    >>> print global_scene.compositors.keys()

The builtin composites are recommendations from the `MSG Interpretation Guide`_

Retrieving channels
===================

Retrieving channels is dead easy. From the center wavelength::

   >>> print global_scene[0.6]

    seviri/VIS006:
            area: On-the-fly area
            start_time: 2015-04-20 10:00:00
            units: %
            wavelength_range: (0.56, 0.635, 0.71) μm
            shape: (3712, 3712)


or from the channel name::

   >>> print global_scene["VIS006"]

    seviri/VIS006:
            area: On-the-fly area
            start_time: 2015-04-20 10:00:00
            units: %
            wavelength_range: (0.56, 0.635, 0.71) μm
            shape: (3712, 3712)

The printed lines consists of the following values:

* First the sensor and name is displayed,
* then the metadata is shown

The data of the channel is actually a subclass of a numpy masked array, so to view the data:
  
   >>> print global_scene[0.6]

   seviri/VIS006:
           area: On-the-fly area
           start_time: 2015-04-20 10:00:00
           units: %
           wavelength_range: (0.56, 0.635, 0.71) μm
           shape: (3712, 3712)
   [[-- -- -- ..., -- -- --]
    [-- -- -- ..., -- -- --]
    [-- -- -- ..., -- -- --]
    ...,
    [-- -- -- ..., -- -- --]
    [-- -- -- ..., -- -- --]
    [-- -- -- ..., -- -- --]]

Channels can be viewed with the :meth:`show` method of the image object::

  >>> get_enhanced_image(global_scene[0.6]).show()

.. image:: images/ch6.png
   

Channel arithmetics
===================

The common arithmetical operators are supported on channels, so that one can
run for example::

  >>> ndvi = (global_scene[0.6] - global_scene[0.8]) * (global_scene[0.8] + global_scene[0.6])

Projections
===========

Until now, we have used the channels directly as provided by the satellite,
that is in satellite projection. Generating composites thus produces views in
satellite projection, *i.e.* as viewed by the satellite.

Most often however, we will want to project the data onto a specific area so
that only the area of interest is depicted in the RGB composites.

Here is how we do that::

    >>> local_scene = global_scene.resample(europe)
    >>>

The area *euro_north* is defined in the *areas.def* file in PPP_CONFIG_DIR. In the sample *area.def* file this is a Stereographic projection of the european area.

Now we have projected data onto the *euro_north* area in the :attr:`local_scene` variable
and we can operate as before to generate and play with RGB composites::

    >>> writer = local_scene.get_writer("geotiff")
    >>> writer.save_sceneset(local_scene["overview"], filename="./local_overview.tif")

.. image:: images/euro_north.png

The image is saved here in GeoTiff_ format. 

Making custom composites
========================

Building custom composites makes use of the :mod:`imageo` module. For example,
building an overview composite can be done manually with::

    >>> from mpop.composites import RGBCompositor
    >>> compositor = RGBCompositor("myoverview", "bla", "")
    >>> composite = compositor([local_scene[0.6],
    ...                         local_scene[0.8],
    ...                         local_scene[10.8]])
    >>> from mpop.writers import to_image
    >>> img = to_image(composite)
    >>> img.invert([False, False, True])
    >>> img.stretch("linear")
    >>> img.show()

In order to have mpop automatically use the composites you create, it is
possible to write them in a python module which name has to be specified in the
`mpop.cfg` configuration file, under the :attr:`composites` section. Change the *mpop.cfg* file to have the following line::

  [composites]
  module=my_composites

Now create a file named *my_composites.py* in a local dir with the content::

  from mpop.imageo.geo_image import GeoImage

  def hr_visual(self):
      """Make a High Resolution visual BW image composite from Seviri
      channel.
      """
      self.check_channels("HRV")

      img = GeoImage(self["HRV"].data, self.area, self.time_slot,
                     fill_value=0, mode="L")
      img.enhance(stretch="crude")
      return img

  hr_visual.prerequisites = set(["HRV"])

  def hr_overview(self):
      """Make a High Resolution Overview RGB image composite from Seviri
      channels.
      """
      self.check_channels(0.635, 0.85, 10.8, "HRV")

      ch1 = self[0.635].check_range()
      ch2 = self[0.85].check_range()
      ch3 = -self[10.8].data

      img = GeoImage((ch1, ch2, ch3), self.area, self.time_slot,
                     fill_value=(0, 0, 0), mode="RGB")

      img.enhance(stretch="crude")
      img.enhance(gamma=[1.6, 1.6, 1.1])

      luminance = GeoImage((self["HRV"].data), self.area, self.time_slot,
                           crange=(0, 100), mode="L")

      luminance.enhance(gamma=2.0)

      img.replace_luminance(luminance.channels[0])

      return img

  hr_overview.prerequisites = set(["HRV", 0.635, 0.85, 10.8])

  seviri = [hr_visual, hr_overview] 

Note the :attr:`seviri` variable in the end. This means that the composites it
contains will be available to all scenes using the Seviri instrument. If we
replace this by::

  meteosat09seviri = [overview,
                      hr_visual]

then the composites will only be available for the Meteosat 9 satellite scenes.

In *my_composites.py* we have now defined 2 custom composites using the HRV channel. 
:attr:`hr_visual` makes an enhanced black and white image from the HRV channel alone. 
:attr:`hr_overview` is a more complex composite using the HRV channel as luminance for the overview image from the previous example. This creates the perception of higher resolution.

Add the dir containing *my_composites.py* to your PYTHONPATH. Now your new composites will be accessible on the :attr:`scene.image` object like the builtin composites::

    >>> from mpop.satellites import GeostationaryFactory
    >>> from mpop.projector import get_area_def
    >>> import datetime
    >>> time_slot = datetime.datetime(2009, 10, 8, 14, 30)
    >>> global_scene = GeostationaryFactory.create_scene("meteosat", "09", "seviri", time_slot)
    >>> msghrvn = get_area_def("MSGHRVN")
    >>> global_scene.load(global_scene.image.hr_overview.prerequisites, area_extent=msghrvn.area_extent)   
    >>> local_scene = global_scene.project("euro_north")
    >>> img = local_scene.image.hr_overview()
    >>> img.show()

.. image:: images/euro_north_hr.png


.. _GeoTiff: http://trac.osgeo.org/geotiff/
.. _mpop: http://www.github.com/mraspaud/mpop
.. _mipp: http://www.github.com/loerum/mipp
.. _pyresample: http://pyresample.googlecode.com
.. _numexpr: http://code.google.com/p/numexpr/
.. _Public Wavelet Transform Decompression Library Software: http://www.eumetsat.int/website/home/Data/DataDelivery/SupportSoftwareandTools/index.html
.. _MSG Interpretation Guide: http://oiswww.eumetsat.org/WEBOPS/msg_interpretation/index.php 
