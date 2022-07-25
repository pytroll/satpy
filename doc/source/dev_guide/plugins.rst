===========================
Extending Satpy via plugins
===========================

.. warning::
    This feature is experimental and being modified without warnings.
    For now, it should not be used for anything else than toy examples and
    should not be relied on.

Satpy is able to load additional functionality outside of the builtin features
in the library. It does this by searching a series of configured paths for
additional configuration files for:

* readers
* composites and modifiers
* enhancements
* writers

For basic testing and temporary configuration changes, you can follow
the instructions in :ref:`component_configuration`. This will tell Satpy
where to look for your custom YAML configuration files and import any Python
code you'd like it to use for these components. However, this requires telling
Satpy of these paths on every execution (either as an environment variable or
by using ``satpy.config``).

Satpy also supports being told this information via setuptools "entry points".
Once your custom Python package with entry points is installed Satpy will
automatically discover it when searching for composites without the user
needing to explicitly import your package. This has the added
benefit of organizing your YAML configuration files and any custom python code
into a single python package. How to structure a package in this way is
described below.

An example project showing the usage of these entry points is available at
`this github repository <https://github.com/pytroll/satpy-composites-plugin-example>`_
where a custom compositor is created. This repository also includes common
configuration files and tools for writing clean code and automatically testing
your python code.

Plugin package structure
========================

The below sections will use the example package name ``satpy-myplugin``. This
is only an example and naming a plugin package with a ``satpy-`` prefix is not
required.

A plugin package should consist of three main parts:

1. ``pyproject.toml`` or ``setup.py``: These files define the metadata and
   entry points for your package. Only one of them is needed. With only a few
   exceptions it is recommended to use a ``pyproject.toml`` as this is the new
   and future way Python package configuration will be supported by the ``pip``
   package manager. See below for examples of the contents of this file.
2. ``mypkg/etc/``: A directory of Satpy-compatible component YAML files. These
   YAML files should be in ``readers/``, ``composites/``, ``enhancements/``,
   and ``writers/`` directories. These YAML files must follow the Satpy naming
   conventions for each component. For example, composites and enhancements
   allow for sensor-specific configuration files. Other directories can be
   added in this ``etc`` directory and will be ignored by Satpy. Satpy will
   collect all available YAML files from all installed plugins and merge them
   with those builtin to Satpy. The Satpy builtins will be used as a "base"
   configuration with all external YAML files applied after.
3. ``mypkg/``: The python package with any custom python code. This code should
   be based on or at least compatible with Satpy's base classes for each
   component or use utilities available from Satpy whenever possible.

   * readers: :class:`~satpy.readers.yaml_reader.FileYAMLReader` for any
     reader subclasses and
     :class:`~satpy.readers.file_handlers.BaseFileHandler` for any custom file
     handlers. See :doc:`custom_reader` for more information.
   * composites and modifiers: :class:`~satpy.composites.CompositeBase` for
     any generic compositor and :class:`~satpy.composites.GenericCompositor`
     for any composite that represents an image (RGB, L, etc). For modifiers,
     use :class:`~satpy.modifiers.ModifierBase`.
   * enhancements: Although not required, consider using
     :func:`satpy.enhancements.apply_enhancement`.
   * writers: :class:`~satpy.writers.Writer`

   Lastly, this directory should be structured like a standard python package.
   This primarily means a ``mypkg/__init__.py`` file should exist.

pyproject.toml
--------------

We recommend using a
`pyproject.toml <https://pip.pypa.io/en/stable/reference/build-system/pyproject-toml/>`_
file can be used to define the
metadata and configuration for a python package. With this file it is possible
to use package building tools to make an installable package. By using a
special feature called "entry points" we can configure our package to its
satpy features are automatically discovered by Satpy.

A ``pyproject.toml`` file is typically placed in the root of a project
repository and at the same level as the package (ex. ``satpy_myplugin/``
directory). An example for a package called ``satpy-myplugin`` with
custom composites is shown below.

.. code:: toml

    [project]
    name = "satpy-myplugin"
    description = "Example Satpy plugin package definition."
    version = "1.0.0"
    readme = "README.md"
    license = {text = "GPL-3.0-or-later"}
    requires-python = ">=3.8"
    dependencies = [
        "satpy",
    ]

    [tool.setuptools]
    packages = ["satpy_myplugin"]

    [build-system]
    requires = ["setuptools", "wheel"]
    build-backend = "setuptools.build_meta"

    [project.entry-points."satpy.composites"]
    example_composites = "satpy_myplugin"

This definition uses
`setuptools <https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html>`_
to build the resulting package (under ``build-system``). There are other
alternative tools (like `poetry <https://python-poetry.org/docs/pyproject/>`_)
that can be used.

Other custom components like readers and writers can be defined in the same
package by using additional entry points named ``satpy.readers`` for readers,
``satpy.writers`` for writers, and ``satpy.enhancements`` for enhancements.

Note the difference between the usage of the package name (``satpy-myplugin``)
which includes a hyphen and the package directory (``satpy_myplugin``) which uses
an underscore. Your package name does not need to have a separator (hyphen) in
it, but is used here due to the common practice of naming plugins this way.
Package directories can't use hyphens as this would be a syntax error when
trying to import the package. Underscores can't be used in package names as
this is not allowed by PyPI.

The first ``project`` section in this TOML file specifies metadata about the
package. This is most important if you plan on distributing your package on
PyPI or similar package repository. We specify that our package depends on
``satpy`` so if someone installs it Satpy will automatically be installed.
The second ``tools.setuptools`` section
tells the package building (via ``setuptools``) what directory the Python
code is in. The third section, ``build-system``, says what tool(s) should be
used for building the package and what extra requirements are needed during
this build process.

The last section, ``project.entry-points."satpy.composites"`` is the only
section specific to this package being a Satpy plugin. At the time of writing
the ``example_composites = "satpy_myplugin"`` portion is not actually used
by Satpy but is required to properly define the entry point in the plugin
package. Instead Satpy will assume that a package that defines the
``satpy.composites`` (or any of the other component types) entry point will
have a ``etc/`` directory in the root of the package structure. Even so,
for future compatibility, it is best to use the name of the package directory
on the right-hand side of the ``=``.

.. warning::

    Due to some limitations in setuptools you must also define a ``setup.py``
    file in addition to ``pyproject.toml`` if you'd like to use "editable"
    installations (``pip install -e .``). Once
    `this setuptools issue <https://github.com/pypa/setuptools/issues/2816>`_
    is resolved this won't be needed. For now this minimal ``setup.py`` will
    work:

    .. code-block:: python

        from setuptools import setup
        setup()

**Alternative: setup.py**

If you are more comfortable creating a ``setup.py``-based python package you
can use ``setup.py`` instead of ``pyproject.toml``. When used for custom
composites, in a package called ``satpy-myplugin`` it would look something like
this:

.. code:: python

    from setuptools import setup
    import os

    setup(
        name='satpy-myplugin',
        entry_points={
            'satpy.composites': [
                'example_composites = satpy_myplugin',
            ],
        },
        package_data={'satpy_myplugin': [os.path.join('etc', 'composites/*.yaml')]},
        install_requires=["satpy"],
    )

Note the difference between the usage of the package name (``satpy-plugin``)
which includes a hyphen and the package directory (``satpy_plugin``) which uses
an underscore. Your package name does not need to have a separator (hyphen) in
it, but is used here due to the common practice of naming plugins this way.
See the ``pyproject.toml`` information above for more information on what each
of these values means.

Licenses
--------

Disclaimer: We are not lawyers.

Satpy source code is under the GPLv3 license. This license requires any
derivative works to also be GPLv3 or GPLv3 compatible. It is our understanding
that importing a Python module could be considered "linking" that source code
to your own (thus being a derivative work) and would therefore require your
code to be licensed with a GPLv3-compatible license. It is currently only
possible to make a Satpy-compatible plugin without importing Satpy if it
contains only enhancements. Writers and compositors are possible without
subclassing, but are likely difficult to implement. Readers are even more
difficult to implement without using Satpy's base classes and utilities.
It is also our understanding that if your custom Satpy plugin code is not
publicly released then it does not need to be GPLv3.
