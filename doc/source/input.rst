=========================================
 Input plugins: the :mod:`satin` package
=========================================

Available plugins
=================

The available input plugins are:

- *mipp* for hrit/lrit formats
- *aapp1b* for aapp level 1b format
- *eps_format* for eps level 1b format
- *hrpt* to read level 0 hrpt format (experimental)

Adding a new plugin
===================

The interface of any reader plugin must comprise the :func:`load` function and
optionally the :func:`get_lon_lat` function.

* :func:`load`: loads the calibrated data into the channels.

* :func:`get_lat_lon`: returns the latitude and longitude of the satellite
  view. This is needed only when the satellite view cannot be expressed
  analytically (swath).


Take a look at the existing readers for more insight.
