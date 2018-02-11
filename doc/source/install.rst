=========================
Installation Instructions
=========================

Pip-based Installation
======================

SatPy is available from the Python Packaging Index (PyPI). A sandbox
environment for `satpy` can be created using
`Virtualenv <http://pypi.python.org/pypi/virtualenv>`_.

To install the `satpy` package and the minimum amount of python dependencies:

.. code-block:: bash

    $ pip install satpy

Additional dependencies can be installed as "extras" and are grouped by
reader, writer, or feature added. Extras available can be found in the
`setup.py <https://github.com/pytroll/satpy/blob/master/setup.py>`_ file.
They can be installed individually:

.. code-block:: bash

    $ pip install satpy[viirs_sdr]

Or all at once, although this isn't recommended due to the large number of
dependencies:

.. code-block:: bash

    $ pip install satpy[all]

Conda-based Installation
========================

Currently SatPy is not available on any common conda environment. However, it
is possible to install SatPy in a conda environment with a combination of
the `conda-forge` channel and pip. A typical conda environment for SatPy can
be created with the following commands:

.. code-block:: bash

    $ conda config --add channels conda-forge
    $ conda env create -n satpy-env python=3.6 xarray dask pyresample netcdf4 h5py gdal
    $ source activate satpy-env
    $ pip install satpy

Using the `pip` commands as described above you should now have a
complete conda environment with a majority of SatPy's dependencies installed.
Activate the environment with ``source activate satpy-env`` to use the
environment in the future.

Ubuntu System Python Installation
=================================

To install SatPy on an Ubuntu system we recommend using virtual environments
to separate SatPy and its dependencies from the rest of the system. Note that
these instructions require using "sudo" privileges which may not be available
to all users and can be very dangerous. The following instructions attempt
to install some SatPy dependencies using the Ubuntu `apt` package manager to
ease installation. Replace `/path/to/pytroll-env` with the environment to be
created.

.. code-block:: bash

    $ sudo apt-get install python-pip python-gdal
    $ sudo pip install virtualenv
    $ virtualenv /path/to/pytroll-env
    $ source /path/to/pytroll-env/bin/activate
    $ pip install satpy


