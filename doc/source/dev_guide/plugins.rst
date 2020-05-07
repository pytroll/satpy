================================================
 Adding new functionality to Satpy via plugins
================================================

.. warning::
    This feature is experimental and being modified without warnings.
    For now, it should not be used for anything else than toy examples and
    should not be relied on.

Satpy has the capability of using plugins. At the moment, new composites can be
added to satpy through external plugins. Plugins for reader and writers may be
added at a later date (PRs are welcome!).

Here is an
`example <https://github.com/mraspaud/satpy-composites-plugin-example>`_ of a
composites plugin.

The key is to use the same configuration directory structure as satpy and add
a `satpy.composites` entry point in the setup.py file of the plugin:

.. code: python

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
