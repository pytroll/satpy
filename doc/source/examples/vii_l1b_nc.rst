EPS-SG VII netCDF Example
===============================

Satpy includes a reader for the EPS-SG Visible and Infrared Imager (VII)
Level 1b data. The following Python code snippet shows an example on how to use
Satpy to read a channel and resample and save the image over the European area.

.. warning::

    This example is currently a work in progress. Some of the below code may
    not work with the currently released version of Satpy. Additional updates
    to this example will be coming soon.

.. code-block:: python

    import glob
    from satpy.scene import Scene

    # find the file/files to be read
    filenames = glob.glob('/path/to/VII/data/W_xx-eumetsat-darmstadt,SAT,SGA1-VII-1B-RAD_C_EUMT_20191007055100*')

    # create a VII scene from the selected granule(s)
    scn = Scene(filenames=filenames, reader='vii_l1b_nc')

    # print available dataset names for this scene
    print(scn.available_dataset_names())

    # load the datasets of interest
    # NOTE: only radiances are supported for test data
    scn.load(["vii_668"], calibration="radiance")

    # resample the scene to a specified area (e.g. "eurol1" for Europe in 1km resolution)
    eur = scn.resample("eurol", resampler='nearest', radius_of_influence=5000)

    # save the resampled data to disk
    eur.save_dataset("vii_668", filename='./vii_668_eur.png')
