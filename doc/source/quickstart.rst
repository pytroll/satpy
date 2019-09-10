==========
Quickstart
==========

Loading and accessing data
==========================

.. testsetup:: *
    >>> import sys
    >>> reload(sys)
    >>> sys.setdefaultencoding('utf8')

To work with weather satellite data you must create a
:class:`~satpy.scene.Scene` object. Satpy does not currently provide an
interface to download satellite data, it assumes that the data is on a
local hard disk already. In order for Satpy to get access to the data the
Scene must be told what files to read and what
:ref:`Satpy Reader <reader_table>` should read them:

    >>> from satpy import Scene
    >>> from glob import glob
    >>> filenames = glob("/home/a001673/data/satellite/Meteosat-10/seviri/lvl1.5/2015/04/20/HRIT/*201504201000*")
    >>> global_scene = Scene(reader="seviri_l1b_hrit", filenames=filenames)

To load data from the files use the :meth:`Scene.load <satpy.scene.Scene.load>`
method. Printing the Scene object will list each of the
:class:`xarray.DataArray` objects currently loaded:

    >>> global_scene.load([0.6, 0.8, 10.8])
    >>> print(global_scene)
    <xarray.DataArray 'reshape-d66223a8e05819b890c4535bc7e74356' (y: 3712, x: 3712)>
    dask.array<shape=(3712, 3712), dtype=float32, chunksize=(464, 3712)>
    Coordinates:
      * x        (x) float64 5.567e+06 5.564e+06 5.561e+06 5.558e+06 5.555e+06 ...
      * y        (y) float64 -5.567e+06 -5.564e+06 -5.561e+06 -5.558e+06 ...
    Attributes:
        satellite_longitude:  0.0
        sensor:               seviri
        satellite_altitude:   35785831.0
        platform_name:        Meteosat-11
        standard_name:        brightness_temperature
        units:                K
        wavelength:           (9.8, 10.8, 11.8)
        satellite_latitude:   0.0
        start_time:           2018-02-28 15:00:10.814000
        end_time:             2018-02-28 15:12:43.956000
        area:                 Area ID: some_area_name\nDescription: On-the-fly ar...
        name:                 IR_108
        resolution:           3000.40316582
        calibration:          brightness_temperature
        polarization:         None
        level:                None
        modifiers:            ()
        ancillary_variables:  []
    <xarray.DataArray 'reshape-1982d32298aca15acb42c481fd74a629' (y: 3712, x: 3712)>
    dask.array<shape=(3712, 3712), dtype=float32, chunksize=(464, 3712)>
    Coordinates:
      * x        (x) float64 5.567e+06 5.564e+06 5.561e+06 5.558e+06 5.555e+06 ...
      * y        (y) float64 -5.567e+06 -5.564e+06 -5.561e+06 -5.558e+06 ...
    Attributes:
        satellite_longitude:  0.0
        sensor:               seviri
        satellite_altitude:   35785831.0
        platform_name:        Meteosat-11
        standard_name:        toa_bidirectional_reflectance
        units:                %
        wavelength:           (0.74, 0.81, 0.88)
        satellite_latitude:   0.0
        start_time:           2018-02-28 15:00:10.814000
        end_time:             2018-02-28 15:12:43.956000
        area:                 Area ID: some_area_name\nDescription: On-the-fly ar...
        name:                 VIS008
        resolution:           3000.40316582
        calibration:          reflectance
        polarization:         None
        level:                None
        modifiers:            ()
        ancillary_variables:  []
    <xarray.DataArray 'reshape-e86d03c30ce754995ff9da484c0dc338' (y: 3712, x: 3712)>
    dask.array<shape=(3712, 3712), dtype=float32, chunksize=(464, 3712)>
    Coordinates:
      * x        (x) float64 5.567e+06 5.564e+06 5.561e+06 5.558e+06 5.555e+06 ...
      * y        (y) float64 -5.567e+06 -5.564e+06 -5.561e+06 -5.558e+06 ...
    Attributes:
        satellite_longitude:  0.0
        sensor:               seviri
        satellite_altitude:   35785831.0
        platform_name:        Meteosat-11
        standard_name:        toa_bidirectional_reflectance
        units:                %
        wavelength:           (0.56, 0.635, 0.71)
        satellite_latitude:   0.0
        start_time:           2018-02-28 15:00:10.814000
        end_time:             2018-02-28 15:12:43.956000
        area:                 Area ID: some_area_name\nDescription: On-the-fly ar...
        name:                 VIS006
        resolution:           3000.40316582
        calibration:          reflectance
        polarization:         None
        level:                None
        modifiers:            ()
        ancillary_variables:  []

Satpy allows loading file data by wavelengths in micrometers (shown above) or by channel name::

    >>> global_scene.load(["VIS006", "VIS008", "IR_108"])

To have a look at the available channels for loading from your :class:`~satpy.scene.Scene` object use the
:meth:`~satpy.scene.Scene.available_dataset_names` method:

    >>> global_scene.available_dataset_names()
    ['HRV',
     'IR_108',
     'IR_120',
     'VIS006',
     'WV_062',
     'IR_039',
     'IR_134',
     'IR_097',
     'IR_087',
     'VIS008',
     'IR_016',
     'WV_073']


To access the loaded data use the wavelength or name:

    >>> print(global_scene[0.6])

Visualizing data                                                                                    
================                                                                                    

To visualize loaded data in a pop-up window:                                                        
                                                                                                    
    >>> global_scene.show(0.6)                                                                      
                                                                                                    
Alternatively if working in a Jupyter notebook the scene can be converted to
a `geoviews <http://geo.holoviews.org/index.html>`_ object using the
:meth:`~satpy.scene.Scene.to_geoviews` method. The geoviews package is not a
requirement of the base satpy install so in order to use this feature the user
needs to install the geoviews package himself.
                                                                                                    
    >>> import holoviews as hv                                                                      
    >>> import geoviews as gv                                                                       
    >>> import geoviews.feature as gf                                                               
    >>> gv.extension("bokeh", "matplotlib")                                                         
    >>> %opts QuadMesh Image [width=600 height=400 colorbar=True] Feature [apply_ranges=False]      
    >>> %opts Image QuadMesh (cmap='RdBu_r')                                                        
    >>> gview = global_scene.to_geoviews(vdims=[0.6])
    >>> gview[::5,::5] * gf.coastline * gf.borders                                                  
                                                                                                     
Creating new datasets                                                                               
=====================                                                                               

Calculations based on loaded datasets/channels can easily be assigned to a new dataset:

    >>> global_scene["ndvi"] = (global_scene[0.8] - global_scene[0.6]) / (global_scene[0.8] + global_scene[0.6])
    >>> global_scene.show("ndvi")

For more information on loading datasets by resolution, calibration, or other
advanced loading methods see the :doc:`readers` documentation.

Generating composites
=====================

Satpy comes with many composite recipes built-in and makes them loadable like any other dataset:

    >>> global_scene.load(['overview'])

To get a list of all available composites for the current scene:

    >>> global_scene.available_composite_names()
    ['overview_sun',
     'airmass',
     'natural',
     'night_fog',
     'overview',
     'green_snow',
     'dust',
     'fog',
     'natural_sun',
     'cloudtop',
     'convection',
     'ash']

Loading composites will load all necessary dependencies to make that composite and unload them after the composite
has been generated.

.. note::

    Some composite require datasets to be at the same resolution or shape. When this is the case the Scene object must
    be resampled before the composite can be generated (see below).

Resampling
==========

.. todo::

   Explain where and how to define new areas

In certain cases it may be necessary to resample datasets whether they come
from a file or are generated composites. Resampling is useful for mapping data
to a uniform grid, limiting input data to an area of interest, changing from
one projection to another, or for preparing datasets to be combined in a
composite (see above). For more details on resampling, different resampling
algorithms, and creating your own area of interest see the
:doc:`resample` documentation. To resample a Satpy Scene:

    >>> local_scene = global_scene.resample("eurol")

This creates a copy of the original ``global_scene`` with all loaded datasets
resampled to the built-in "eurol" area. Any composites that were requested,
but could not be generated are automatically generated after resampling. The
new ``local_scene`` can now be used like the original ``global_scene`` for
working with datasets, saving them to disk or showing them on screen:

    >>> local_scene.show('overview')
    >>> local_scene.save_dataset('overview', './local_overview.tif')

Saving to disk
==============

To save all loaded datasets to disk as geotiff images:

    >>> global_scene.save_datasets()

To save all loaded datasets to disk as PNG images:

    >>> global_scene.save_datasets(writer='simple_image')

Or to save an individual dataset:

    >>> global_scene.save_dataset('VIS006', 'my_nice_image.png')

Datasets are automatically scaled or "enhanced" to be compatible with the
output format and to provide the best looking image. For more information
on saving datasets and customizing enhancements see the documentation on
:doc:`writers`.

Troubleshooting
===============

When something goes wrong, a first step to take is check that the latest Version
of satpy and its dependencies are installed. Satpy drags in a few packages as
dependencies per default, but each reader and writer has it's own dependencies
which can be unfortunately easy to miss when just doing a regular `pip install`.
To check the missing dependencies for the readers and writers, a utility
function called `check_satpy` can be used:

  >>> from satpy.config import check_satpy
  >>> check_satpy()

Due to the way Satpy works, producing as many datasets as possible, there are
times that behavior can be unexpected but with no exceptions raised. To help
troubleshoot these situations log messages can be turned on. To do this run
the following code before running any other Satpy code:

    >>> from satpy.utils import debug_on
    >>> debug_on()
