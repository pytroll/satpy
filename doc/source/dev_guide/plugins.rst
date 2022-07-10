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
   allow for sensor-specific configuration files. Satpy will collect all
   available YAML files from all installed plugins and merge them with those
   builtin to Satpy. The Satpy builtins will be used as a "base" configuration
   with all external YAML files applied after.
3. ``mypkg/``: The python package with any custom python code. This code should
   be based on Satpy's base classes for each component or use utilities
   available from Satpy whenever possible.

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

A ``pyproject.toml`` file can be used to define the metadata and configuration
for a python package. With this file it is possible to use package building
tools to make an installable package. By using a special feature called
"entry points" we can configure our package to its satpy features are
automatically discovered by Satpy.

A ``pyproject.toml`` file is typically placed in the root of a project
repository and at the same level as the package (ex. ``satpy_myplugin/``
directory). An example for a package called ``satpy_myplugin`` with
custom composites is shown below.

.. code:: toml

    TODO

Other custom components like readers and writers can be defined in the same
package by using additional entry points named TODO.

**Alternative: setup.py**

If you are more comfortable creating a ``setup.py``-based python package you
can use ``setup.py`` instead of ``pyproject.toml``. When used for custom
composites, in a package called ``satpy_cp`` it would look something like
this:

.. code:: python

    from setuptools import setup
    import os

    setup(
        name='satpy_cpe',
        entry_points={
            'satpy.composites': [
                'example_composites = satpy_cpe',
            ],
        },
        package_data={'satpy_cpe': [os.path.join('etc', 'composites/*.yaml')]},
    )
