=========================
Installation Instructions
=========================

Pip-based Installation
======================

Satpy is available from the Python Packaging Index (PyPI). A sandbox
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

Starting with version 0.9, Satpy is available from the conda-forge channel. If
you have not configured your conda environment to search conda-forge already
then do:

.. code-block:: bash

    $ conda config --add channels conda-forge

Then to install Satpy in to your current environment run:

.. code-block:: bash

    $ conda install satpy

.. note::

    Satpy only automatically installs the dependencies needed to process the
    most common use cases. Additional dependencies may need to be installed
    with conda or pip if import errors are encountered.

Ubuntu System Python Installation
=================================

To install Satpy on an Ubuntu system we recommend using virtual environments
to separate Satpy and its dependencies from the rest of the system. Note that
these instructions require using "sudo" privileges which may not be available
to all users and can be very dangerous. The following instructions attempt
to install some Satpy dependencies using the Ubuntu `apt` package manager to
ease installation. Replace `/path/to/pytroll-env` with the environment to be
created.

.. code-block:: bash

    $ sudo apt-get install python-pip python-gdal
    $ sudo pip install virtualenv
    $ virtualenv /path/to/pytroll-env
    $ source /path/to/pytroll-env/bin/activate
    $ pip install satpy


