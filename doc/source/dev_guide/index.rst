=================
Developer's Guide
=================

The below sections will walk through how to set up a development environment,
make changes to the code, and test that they work. See the
:doc:`CONTRIBUTING` section for more information on getting started and
contributor expectations. Additional information for developer's can be found
at the pages listed below.

.. toctree::
    :maxdepth: 1

    CONTRIBUTING
    xarray_migration
    custom_reader

Coding guidelines
=================

Satpy is part of `PyTroll <http://pytroll.github.io/>`_,
and all code should follow the
`PyTroll coding guidelines and best
practices <http://pytroll.github.io/guidelines.html>`_.

Satpy currently supports Python 2.7 and 3.4+. All code should be written to
be compatible with these versions.

Development installation
========================

See the :doc:`../install` section for basic installation instructions. When
it comes time to install Satpy it should be installed from a clone of the git
repository and in development mode so that local file changes are
automatically reflected in the python environment. We highly recommend making
a separate conda environment or virtualenv for development.

First, if you plan on contributing back to the project you should
`fork the repository <https://help.github.com/articles/fork-a-repo/>`_ and
clone your fork. The package can then be installed in development by doing::

    pip install -e .

Running tests
=============

Satpy tests are written using the python :mod:`unittest` module and the tests
can be executed by running::

    python setup.py test

Documentation
=============

Satpy's documentation is built using Sphinx. All documentation lives in the
``doc/`` directory of the project repository. After editing the source files
there the documentation can be generated locally::

    cd doc
    make html

The output of the make command should be checked for warnings and errors.
If code has been changed (new functions or classes) then the API documentation
files should be regenerated before running the above command::

    sphinx-apidoc -f -T -o source/api ../satpy ../satpy/tests
