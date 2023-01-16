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
    remote_file_support
    plugins
    satpy_internals
    aux_data
    writing_tests

Coding guidelines
=================

Satpy is part of `Pytroll <http://pytroll.github.io/>`_,
and all code should follow the
`Pytroll coding guidelines and best
practices <http://pytroll.github.io/guidelines.html>`_.

Satpy is now Python 3 only and it is no longer needed to support Python 2.
Check ``setup.py`` for the current Python versions any new code needs
to support.

.. _devinstall:

Development installation
========================

See the :doc:`../install` section for basic installation instructions. When
it comes time to install Satpy it should be installed from a clone of the git
repository and in development mode so that local file changes are
automatically reflected in the python environment. We highly recommend making
a separate conda environment or virtualenv for development. For example, you
can do this using conda_::

  conda create -n satpy-dev python=3.8
  conda activate satpy-dev

.. _conda: https://conda.io/

This will create a new environment called "satpy-dev" with Python 3.8
installed. The second command will activate the environment so any future
conda, python, or pip commands will use this new environment.

If you plan on contributing back to the project you should first
`fork the repository <https://help.github.com/articles/fork-a-repo/>`_ and
clone your fork. The package can then be installed in development mode by doing::

    conda install --only-deps satpy
    pip install -e .

The first command will install all dependencies needed by the Satpy
conda-forge package, but won't actually install Satpy. The second command
should be run from the root of the cloned Satpy repository (where the
``setup.py`` is) and will install the actual package.

You can now edit the python files in your cloned repository and have them
immediately reflected in your conda environment.

Running tests
=============

Satpy tests are written using the third-party :doc:`pytest <pytest:index>`
package. There is usually no need to run all Satpy tests, but instead only
run the tests related to the component you are working on. All tests are
automatically run from the GitHub Pull Request using multiple versions of
Python, multiple operating systems, and multiple versions of dependency
libraries. If you want to run all Satpy tests you will need to install
additional dependencies that aren't needed for regular Satpy usage. To install
them run::

    pip install -e .[tests]

Satpy tests can be executed by running::

    pytest satpy/tests

You can also run a specific tests by specifying a sub-directory or module::

    pytest satpy/tests/reader_tests/test_abi_l1b.py

Running benchmarks
==================

Satpy benchmarks are written using the
`Airspeed Velocity <https://asv.readthedocs.io/en/stable/index.html>`_
package (:mod:`asv`).
The benchmarks can be run using::

    asv run

These are pretty computation intensive, and shouldn't be run unless you want to
diagnose some performance issue for example.

Once the benchmarks have run, you can use::

    asv publish
    asv preview

to have a look at the results. Again, have a look at the `asv` documentation for
more information.

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
