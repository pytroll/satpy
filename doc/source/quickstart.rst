============
 Quickstart
============

The software uses OOP extensively, to allow higher level metaobject handling.

For this tutorial, we will use the Meteosat plugin and data.

Don’t forget to first source the `profile` file of interest located in the
source `etc` directory.

First example
=============

.. versionchanged:: 0.10.0
   The factory-based loading was added in 0.10.0


Ok, let's get it on::

    >>> from mpop.satellites import GeostationaryFactory
    >>> from mpop.projector import get_area_def
    >>> import datetime
    >>> time_slot = datetime.datetime(2009, 10, 8, 14, 30)
    >>> global_data = GeostationaryFactory.create_scene("Meteosat-9", "", "seviri", time_slot)
    >>> europe = get_area_def("EuropeCanary")
    >>> global_data.load([0.6, 0.8, 10.8], area_extent=europe.area_extent)
    >>> print global_data
    'IR_097: (9.380,9.660,9.940)μm, resolution 3000.40316582m, not loaded'
    'IR_016: (1.500,1.640,1.780)μm, resolution 3000.40316582m, not loaded'
    'VIS008: (0.740,0.810,0.880)μm, shape (1200, 3000), resolution 3000.40316582m'
    'VIS006: (0.560,0.635,0.710)μm, shape (1200, 3000), resolution 3000.40316582m'
    'WV_062: (5.350,6.250,7.150)μm, resolution 3000.40316582m, not loaded'
    'IR_120: (11.000,12.000,13.000)μm, resolution 3000.40316582m, not loaded'
    'WV_073: (6.850,7.350,7.850)μm, resolution 3000.40316582m, not loaded'
    'IR_087: (8.300,8.700,9.100)μm, resolution 3000.40316582m, not loaded'
    'IR_039: (3.480,3.920,4.360)μm, resolution 3000.40316582m, not loaded'
    'HRV: (0.500,0.700,0.900)μm, resolution 1000.13434887m, not loaded'
    'IR_134: (12.400,13.400,14.400)μm, resolution 3000.40316582m, not loaded'
    'IR_108: (9.800,10.800,11.800)μm, shape (1200, 3000), resolution 3000.40316582m'


In this example, we create a scene object for the seviri instrument onboard
Meteosat-9, specifying the time of the snapshot of interest. The time is
defined as a datetime object. 

The next step is loading the data. This is done using mipp, which takes care of
reading the HRIT data, and slicing the data so that we read just what is
needed. Calibration is also done with mipp. In order to slice the data, we
retreive the area we will work on, here set to variable *europe*.

Here we call the :meth:`load` function with a list of the wavelengths of the
channels we are interested in, and the area extent in satellite projection of
the area of interest. Each retrieved channel is the closest in terms of central
wavelength, provided that the required wavelength is within the bounds of the
channel.

The wavelengths are given in micrometers and have to be given as a floating
point number (*i.e.*, don't type '1', but '1.0'). Using an integer number
instead returns a channel based on resolution, while using a string retrieves a
channels based on its name.

    >>> img = global_data.image.overview()
    >>> img.save("./myoverview.png")
    >>>

Once the channels are loaded, we generate an overview RGB composite image, and
save it as a png image. Instead of :meth:`save`, one could also use
:meth:`show` if the only purpose is to display the image on screen.

Available composites are listed in the :mod:`mpop.satellites.visir` module
in the mpop documentation.

We want more!
==============

In the last example, the composite generation worked because the channels
needed for the overview (0.6, 0.8, 10.8 μm) were loaded. If we try to generate
a day natural color composite, which requires also the 1.6 μm channel, it will
result in an error::

   
    >>> img = global_data.image.natural()
    Traceback (most recent call last):
      ...
    NotLoadedError: Required channel 1.63 not loaded, aborting.

So it means that we have to load the missing channel first. To do this we could
enter the channels list to load manually, as we did for the overview, but we
provide a way to get the list of channels needed by a given method using the
`prerequisites` method attribute::

    >>> global_data.load(global_data.image.natural.prerequisites, area_extent=europe.area_extent)
    >>> img = global_data.image.natural()
    >>>

Now you can save the image::

    >>> img.save("./mynaturalcolors.png")
    >>>

If you want to combine several prerequisites for channel loading, since
prerequisites are python sets, you can do::

    >>> global_data.load(global_data.image.overview.prerequisites | 
    ...                  global_data.image.natural.prerequisites,
    ...                  area_extent=europe.area_extent)
    >>>

and add as many `| global_data.image.mymethod.prerequisites` as needed.

Retrieving channels
===================

Retrieving channels is dead easy. From the center wavelength::

   >>> print global_data[0.6]
   'VIS006: (0.560,0.635,0.710)μm, shape (1200, 3000), resolution 3000.40316582m'

or from the channel name::

   >>> print global_data["VIS006"]
   'VIS006: (0.560,0.635,0.710)μm, shape (1200, 3000), resolution 3000.40316582m'

or from the resolution::
 
   >>> print global_data[3000]
   'VIS006: (0.560,0.635,0.710)μm, shape (1200, 3000), resolution 3000.40316582m'

or more than one at the time::

   >>> print global_data[3000, 0.8]
   'VIS008: (0.740,0.810,0.880)μm, shape (1200, 3000), resolution 3000.40316582m'

The printed lines consists of the following values:

* First the name is displayed,
* then the triplet gives the min-, center-, and max-wavelength of the
  channel,
* follows the shape of the loaded data, or `None` if the data is not loaded,
* and finally the theoretical resolution of the channel is shown.

The data of the channel can be retrieved as an numpy (masked) array using the
data property::
  
   >>> print global_data[0.6].data
   [[-- -- -- ..., -- -- --]
    [-- -- -- ..., -- -- --]
    [-- -- -- ..., -- -- --]
    ..., 
    [7.37684259374 8.65549530999 6.58997938374 ..., 0.29507370375 0.1967158025
     0.1967158025]
    [7.18012679124 7.86863209999 6.19654777874 ..., 0.29507370375
     0.29507370375 0.29507370375]
    [5.80311617374 7.57355839624 6.88505308749 ..., 0.29507370375
     0.29507370375 0.29507370375]]

Channel arithmetics
===================

.. versionadded:: 0.10.0
   Channel arithmetics added.

The common arithmetical operators are supported on channels, so that one can
run for example::

  >>> cool_channel = (global_data[0.6] - global_data[0.8]) * global_data[10.8]

PGEs
====

From the satellite data PGEs [#f1]_ are generated by the accompanying program.
The loading procedure for PGEs is exactly the same as with regular channels::

    >>> global_data.area = "EuropeCanary"
    >>> global_data.load(["CTTH"])
    >>>
    
and they can be retrieved as simply as before::
    
    >>> print global_data["CTTH"] 
    'CTTH: shape (1200, 3000), resolution 3000.40316582m'

Making custom composites
========================

Building custom composites makes use of the :mod:`imageo` module. For example,
building an overview composite can be done manually with::

    >>> from mpop.imageo.geo_image import GeoImage
    >>> img = GeoImage((global_data[0.6].data, 
    ...                 global_data[0.8].data, 
    ...                 -global_data[10.8].data),
    ...                 "EuropeCanary",
    ...                 time_slot,
    ...                 mode = "RGB")
    >>> img.enhance(stretch="crude")
    >>> img.enhance(gamma=1.7)

.. versionadded:: 0.10.0
   Custom composites module added.

In order to have mpop automatically use the composites you create, it is
possible to write them in a python module which name has to be specified in the
`mpop.cfg` configuration file, under the *composites* section::

  [composites]
  module=mpop.smhi_composites

The module has to be importable (i.e. it has to be in the pythonpath). 
Here is an example of such a module::

  def overview(self):
      """Make an overview RGB image composite.
      """
      self.check_channels(0.635, 0.85, 10.8)

      ch1 = self[0.635].check_range()
      ch2 = self[0.85].check_range()
      ch3 = -self[10.8].data

      img = geo_image.GeoImage((ch1, ch2, ch3),
                               self.area,
                               self.time_slot,
                               fill_value=(0, 0, 0),
                               mode="RGB")

      img.enhance(stretch = (0.005, 0.005))

      return img

  overview.prerequisites = set([0.6, 0.8, 10.8])

  def hr_visual(self):
      """Make a High Resolution visual BW image composite from Seviri
      channel.
      """
      self.check_channels("HRV")

      img = geo_image.GeoImage(self["HRV"].data,
                               self.area,
                               self.time_slot,
                               fill_value=0,
                               mode="L")
      img.enhance(stretch="crude")
      return img

  hr_visual.prerequisites = set(["HRV"])

  seviri = [overview,
            hr_visual]

Projections
===========

Until now, we have used the channels directly as provided by the satellite,
that is in satellite projection. Generating composites thus produces views in
satellite projection, *i.e.* as viewed by the satellite.

Most often however, we will want to project the data onto a specific area so
that only the area of interest is depicted in the RGB composites.

Here is how we do that::

    >>> local_data = global_data.project("eurol")
    >>>

Now we have projected data onto the "eurol" area in the `local_data` variable
and we can operate as before to generate and play with RGB composites::

    >>> img = local_data.image.overview()
    >>> img.save("./local_overview.tif")
    >>>

The image is saved here in GeoTiff_ format. 

On projected images, one can also add contour overlay with the
:meth:`imageo.geo_image.add_overlay`.

.. _GeoTiff: http://trac.osgeo.org/geotiff/




.. rubric:: Footnotes

.. [#f1] PGEs in Meteosat : CloudType and CTTH
