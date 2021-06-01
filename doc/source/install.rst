=========================
Installation Instructions
=========================

Satpy is available from conda-forge (via conda), PyPI (via pip), or from
source (via pip+git). The below instructions show how to install stable
versions of Satpy. For a development/unstable version see :ref:`devinstall`.

Conda-based Installation
========================

Satpy can be installed into a conda environment by installing the package
from the conda-forge channel. If you do not already have access to a conda
installation, we recommend installing
`miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ for the smallest
and easiest installation.

The commands below will use ``-c conda-forge`` to make sure packages are
downloaded from the conda-forge channel. Alternatively, you can tell conda
to always use conda-forge by running:

.. code-block:: bash

    $ conda config --add channels conda-forge

We recommend creating a separate environment for your work with Satpy. If
you haven't created and activated one already, you can by running:

.. code-block:: bash

    $ conda create -c conda-forge -n my_satpy_env python
    $ conda activate my_satpy_env

The above will create a new environment with the latest version of Python
installed along with Satpy and all of its dependencies. The second command
will activate the environment so all future conda or python commands will
use this new environment.

Next to install Satpy into an existing activated environment run:

.. code-block:: bash

    $ conda install -c conda-forge satpy

.. note::

    Satpy only automatically installs the dependencies needed to process the
    most common use cases. Additional dependencies may need to be installed
    with conda or pip if import errors are encountered.

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
`setup.py <https://github.com/pytroll/satpy/blob/main/setup.py>`_ file.
They can be installed individually:

.. code-block:: bash

    $ pip install "satpy[viirs_sdr]"

Or all at once, although this isn't recommended due to the large number of
dependencies:

.. code-block:: bash

    $ pip install "satpy[all]"

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


