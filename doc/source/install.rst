=========================
Installation Instructions
=========================

Pip-based Installation
======================

SatPy is available from the Python Packaging Index (PyPI). A sandbox
environment for `satpy` can be created using
`Virtualenv <http://pypi.python.org/pypi/virtualenv>`_.

To install the `satpy` package and all of its python dependencies from PyPI:

.. code-block:: bash

    $ pip install satpy

Conda-based Installation
========================

The `satpy repository <https://github.com/pytroll/satpy>`_ contains an environment file
to install `satpy` and all its dependencies (including non python dependencies) via 
`conda <https://conda.io/docs/intro.html>`_.
For now there is only one file available to install `satpy` on python 2.7 and numpy 1.11
(`satpy-env_np111py27.yml`).
Environment files for other python or numpy versions can be created by adapting this file.

After downloading the file you can install `satpy` in an environment called `satpy-env`
using `conda`:

.. code-block:: bash

    $ conda env create -f satpy-env_np<xxx>py<yy>.yml

To activate this environment use     

.. code-block:: bash

    $ source activate satpy-env 

