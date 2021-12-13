Satpy
=====

.. image:: https://github.com/pytroll/satpy/workflows/CI/badge.svg?branch=main
    :target: https://github.com/pytroll/satpy/actions?query=workflow%3A%22CI%22

.. image:: https://coveralls.io/repos/github/pytroll/satpy/badge.svg?branch=main
    :target: https://coveralls.io/github/pytroll/satpy?branch=main

.. image:: https://badge.fury.io/py/satpy.svg
    :target: https://badge.fury.io/py/satpy

.. image:: https://anaconda.org/conda-forge/satpy/badges/version.svg
   :target: https://anaconda.org/conda-forge/satpy/

.. image:: https://zenodo.org/badge/51397392.svg
   :target: https://zenodo.org/badge/latestdoi/51397392


The Satpy package is a python library for reading and manipulating
meteorological remote sensing data and writing it to various image and
data file formats. Satpy comes with the ability to make various RGB
composites directly from satellite instrument channel data or higher level
processing output. The
`pyresample <http://pyresample.readthedocs.io/en/latest/>`_ package is used
to resample data to different uniform areas or grids.

The documentation is available at
http://satpy.readthedocs.org/.

Installation
------------

Satpy can be installed from PyPI with pip:

.. code-block:: bash

    pip install satpy


It is also available from `conda-forge` for conda installations:

.. code-block:: bash

    conda install -c conda-forge satpy

Code of Conduct
---------------

Satpy follows the same code of conduct as the PyTroll project. For reference
it is copied to this repository in CODE_OF_CONDUCT.md_.

As stated in the PyTroll home page, this code of conduct applies to the
project space (GitHub) as well as the public space online and offline when
an individual is representing the project or the community. Online examples
of this include the PyTroll Slack team, mailing list, and the PyTroll twitter
account. This code of conduct also applies to in-person situations like
PyTroll Contributor Weeks (PCW), conference meet-ups, or any other time when
the project is being represented.

Any violations of this code of conduct will be handled by the core maintainers
of the project including David Hoese, Martin Raspaud, and Adam Dybbroe.
If you wish to report one of the maintainers for a violation and are
not comfortable with them seeing it, please contact one or more of the other
maintainers to report the violation. Responses to violations will be
determined by the maintainers and may include one or more of the following:

- Verbal warning
- Ask for public apology
- Temporary or permanent ban from in-person events
- Temporary or permanent ban from online communication (Slack, mailing list, etc)

For details see the official CODE_OF_CONDUCT.md_.

.. _CODE_OF_CONDUCT.md: ./CODE_OF_CONDUCT.md
