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

Coding guidelines
=================

SatPy tries to follow
`PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ style guidelines for
all of its python code. We also try to limit lines of code to 80 characters
whenever possible and when it doesn't hurt readability. SatPy follows
`Google Style Docstrings <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_
for all code API documentation. When in doubt use the existing code as a
guide for how coding should be done.

SatPy currently supports Python 2.7 and 3.4+. All code should be written to
be compatible with these versions.

Development installation
========================

See the :doc:`../install` section for basic installation instructions. When
it comes time to install SatPy it should be installed from a clone of the git
repository and in development mode so that local file changes are
automatically reflected in the python environment. We highly recommend making
a separate conda environment or virtualenv for development.

First, if you plan on contributing back to the project you should
`fork the repository <https://help.github.com/articles/fork-a-repo/>`_ and
clone your fork. The package can then be installed in development by doing::

    pip install -e .

Running tests
=============

SatPy tests are written using the python :mod:`unittest` module and the tests
can be executed by running::

    python setup.py test

Documentation
=============

SatPy's documentation is built using Sphinx. All documentation lives in the
``doc/`` directory of the project repository. After editing the source files
there the documentation can be generated locally::

    cd doc
    make html

The output of the make command should be checked for warnings and errors.
If code has been changed (new functions or classes) then the API documentation
files should be regenerated before running the above command::

    sphinx-apidoc -f -T -o source/api ../satpy ../satpy/tests
