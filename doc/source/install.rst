===========================
Installation instructions
===========================

Pip based installation
======================

`satpy` is available from pypi.
A sandbox environment for `satpy` can be created using `Virtualenv <http://pypi.python.org/pypi/virtualenv>`_.

Installation using pip:

.. code-block:: bash

    $ pip install satpy

This will install `satpy` and all it's python dependendencies from pypi.
However you have to asure, that all non python dependencies are installed
on your machine.

Installation based on conda
===========================

The `satpy repository <https://github.com/pytroll/satpy>`_ contains a file `satpy-environment.yml`
to install `satpy` and all its dependencies (including non python dependencies) via 
`conda <https://conda.io/docs/intro.html>`_.
The environment file uses python 2.7 and numpy 1.11 as a basis, environments for python 3 or other 
numpy versions can be created by adapting this file.

After downloading this file you can install `satpy` in an environment called `satpy-env`
using `conda`:

.. code-block:: bash

    $ conda env create -f satpy-environment.yml 

To activate this environment use     

.. code-block:: bash

    $ source activate satpy-env 

