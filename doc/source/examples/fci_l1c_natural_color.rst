MTG FCI - Natural Color Example
===============================

Satpy includes a reader for the Meteosat Third Generation (MTG) FCI Level 1c
data. The following Python code snippet shows an example on how to use Satpy
to generate a Natural Color RGB composite over the European area.

.. warning::

    This example is currently a work in progress. Some of the below code may
    not work with the currently released version of Satpy. Additional updates
    to this example will be coming soon.

.. note::

    For reading compressed data, a decompression library is
    needed. Either install the FCIDECOMP library (see the `FCI L1 Product User
    Guide <https://www.eumetsat.int/media/45923>`_, or the
    ``hdf5plugin`` package with::

        pip install hdf5plugin

    or::

        conda install hdf5plugin -c conda-forge

    If you use ``hdf5plugin``, make sure to add the line ``import hdf5plugin``
    at the top of your script.

.. code-block:: python

    from satpy.scene import Scene
    from satpy import find_files_and_readers

    # define path to FCI test data folder
    path_to_data = 'your/path/to/FCI/data/folder/'

    # find files and assign the FCI reader
    files = find_files_and_readers(base_dir=path_to_data, reader='fci_l1c_nc')

    # create an FCI scene from the selected files
    scn = Scene(filenames=files)

    # print available dataset names for this scene (e.g. 'vis_04', 'vis_05','ir_38',...)
    print(scn.available_dataset_names())

    # print available composite names for this scene (e.g. 'natural_color', 'airmass', 'convection',...)
    print(scn.available_composite_names())

    # load the datasets/composites of interest
    scn.load(['natural_color','vis_04'], upper_right_corner='NE')
    # note: the data inside the FCI files is stored upside down. The upper_right_corner='NE' argument
    # flips it automatically in upright position.

    # you can access the values of a dataset as a Numpy array with
    vis_04_values = scn['vis_04'].values

    # resample the scene to a specified area (e.g. "eurol1" for Europe in 1km resolution)
    scn_resampled = scn.resample("eurol", resampler='nearest', radius_of_influence=5000)

    # save the resampled dataset/composite to disk
    scn_resampled.save_dataset("natural_color", filename='./fci_natural_color_resampled.png')
