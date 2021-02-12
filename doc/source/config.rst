Configuration
=============

Satpy has two levels of configuration that allow to control how Satpy and
its various components behave. There are a series of "settings" that change
the global Satpy behavior. There are also a series of "component
configuration" YAML files for controlling the complex functionality in readers,
compositors, writers, and other Satpy components that can't be controlled
with traditional keyword arguments.

Settings
--------

There are configuration parameters in Satpy that are not specific to one
component and control more global behavior of Satpy. These parameters can be
set in one of three ways:

1. Environment variable
2. YAML file
3. At runtime with ``satpy.config``

This functionality is provided by the :doc:`donfig <donfig:configuration>`
library. The currently available settings are described below.
Each option is available from all three methods. If specified as an
environment variable or specified in the YAML file on disk, it must be set
**before** Satpy is imported.

**YAML Configuration**

YAML files that include these parameters can be in any of the following
locations:

1. ``<python environment prefix>/etc/satpy/satpy.yaml``
2. ``<user_config_dir>/satpy.yaml`` (see below)
3. ``~/.satpy/satpy.yaml``
4. ``<SATPY_CONFIG_PATH>/satpy.yaml`` (see :ref:`config_path_setting` below)

The above ``user_config_dir`` is provided by the ``appdirs`` package and
differs by operating system. Typical user config directories are:

* Mac OSX: ``~/Library/Preferences/satpy``
* Unix/Linux: ``~/.config/satpy``
* Windows: ``C:\\Users\\<username>\\AppData\\Local\\pytroll\\satpy``

All YAML files found from the above paths will be merged into one
configuration object (accessed via ``satpy.config``).
The YAML contents should be a simple mapping of configuration key to its
value. For example:

.. code-block:: yaml

    cache_dir: "/tmp"
    data_dir: "/tmp"

Lastly, it is possible to specify an additional config path to the above
options by setting the environment variable ``SATPY_CONFIG``. The file
specified with this environment variable will be added last after all of the
above paths have been merged together.

**At runtime**

After import, the values can be customized at runtime by doing:

.. code-block:: python

    import satpy
    satpy.config.set(cache_dir="/my/new/cache/path")
    # ... normal satpy code ...

Or for specific blocks of code:

.. code-block:: python

    import satpy
    with satpy.config.set(cache_dir="/my/new/cache/path"):
        # ... some satpy code ...
    # ... code using the original cache_dir

Similarly, if you need to access one of the values you can
use the ``satpy.config.get`` method.

Cache Directory
^^^^^^^^^^^^^^^

* **Environment variable**: ``SATPY_CACHE_DIR``
* **YAML/Config Key**: ``cache_dir``
* **Default**: See below

Directory where any files cached by Satpy will be stored. This
directory is not necessarily cleared out by Satpy, but is rarely used without
explicitly being enabled by the user. This
defaults to a different path depending on your operating system following
the `appdirs <https://github.com/ActiveState/appdirs#some-example-output>`_
"user cache dir".

.. _config_path_setting:

Component Configuration Path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Environment variable**: ``SATPY_CONFIG_PATH``
* **YAML/Config Key**: ``config_path``
* **Default**: ``[]``

Base directory, or directories, where Satpy component YAML configuration files
are stored. Satpy expects configuration files for specific component types to
be in appropriate subdirectories (ex. ``readers``, ``writers``, etc), but
these subdirectories should not be included in the ``config_path``.
For example, if you have custom composites configured in
``/my/config/dir/etc/composites/visir.yaml``, then ``config_path`` should
include ``/my/config/dir/etc`` for Satpy to find this configuration file
when searching for composites. This option replaces the legacy
``PPP_CONFIG_DIR`` environment variable.

Note that this value must be a list. In Python, this could be set by doing:

.. code-block:: python

    satpy.config.set(config_path=['/path/custom1', '/path/custom2'])

If setting an environment variable then it must be a
colon-separated string and must be set **before** calling/importing Satpy.
If the environment variable is a single path it will be converted to a list
when Satpy is imported.

.. code-block:: bash

    export SATPY_CONFIG_PATH="/path/custom1:/path/custom2"

Satpy will always include the builtin configuration files that it
is distributed with regardless of this setting. When a component supports
merging of configuration files, they are merged in reverse order. This means
"base" configuration paths should be at the end of the list and custom/user
paths should be at the beginning of the list.

.. _data_dir_setting:

Data Directory
^^^^^^^^^^^^^^

* **Environment variable**: ``SATPY_DATA_DIR``
* **YAML/Config Key**: ``data_dir``
* **Default**: See below

Directory where any data Satpy needs to perform certain operations will be
stored. This replaces the legacy ``SATPY_ANCPATH`` environment variable. This
defaults to a different path depending on your operating system following the
`appdirs <https://github.com/ActiveState/appdirs#some-example-output>`_
"user data dir".

.. _download_aux_setting:

Download Auxiliary Data
^^^^^^^^^^^^^^^^^^^^^^^

* **Environment variable**: ``SATPY_DOWNLOAD_AUX``
* **YAML/Config Key**: ``download_aux``
* **Default**: True

Whether to allow downloading of auxiliary files for certain Satpy operations.
See :doc:`dev_guide/aux_data` for more information. If ``True`` then Satpy
will download and cache any necessary data files to :ref:`data_dir_setting`
when needed. If ``False`` then pre-downloaded files will be used, but any
other files will not be downloaded or checked for validity.

.. _component_configuration:

Component Configuration
-----------------------

Much of the functionality of Satpy comes from the various components it
uses, like readers, writers, compositors, and enhancements. These components
are configured for reuse from YAML files stored inside Satpy or in custom user
configuration files. Custom directories can be provided by specifying the
:ref:`config_path setting <config_path_setting>` mentioned above.

To create and use your own custom component configuration you should:

1. Create a directory to store your new custom YAML configuration files.
   The files for each component will go in a subdirectory specific to that
   component (ex. ``composites``, ``enhancements``, ``readers``, ``writers``).
2. Set the Satpy :ref:`config_path <config_path_setting>` to point to your new
   directory. This could be done by setting the environment variable
   ``SATPY_CONFIG_PATH`` to your custom directory (don't include the
   component sub-directory) or one of the other methods for setting this path.
3. Create YAML configuration files with your custom YAML files. In most cases
   there is no need to copy configuration from the builtin Satpy files as
   these will be merged with your custom files.
4. If your custom configuration uses custom Python code, this code must be
   importable by Python. This means your code must either be installed in your
   Python environment or you must set your ``PYTHONPATH`` to the location of
   the modules.
5. Run your Satpy code and access your custom components like any of the
   builtin components.
