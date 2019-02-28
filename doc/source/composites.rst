==========
Composites
==========

Documentation coming soon...

Modifiers
=========

Making custom composites
========================

.. note::

    These features will be added to the ``Scene`` object in the future.

Building custom composites makes use of the :class:`GenericCompositor` class. For example,
building an overview composite can be done manually with::

    >>> from satpy.composites import GenericCompositor
    >>> compositor = GenericCompositor("myoverview", "bla", "")
    >>> composite = compositor([local_scene[0.6],
    ...                         local_scene[0.8],
    ...                         local_scene[10.8]])
    >>> from satpy.writers import to_image
    >>> img = to_image(composite)
    >>> img.invert([False, False, True])
    >>> img.stretch("linear")
    >>> img.gamma(1.7)
    >>> img.show()


One important thing to notice is that there is an internal difference between a composite and an image. A composite
is defined as a special dataset which may have several bands (like R, G, B bands). However, the data isn't stretched,
or clipped or gamma filtered until an image is generated.


To save the custom composite, the following procedure can be used:

1. Create a custom directory for your custom configs.
2. Set it in the environment variable called PPP_CONFIG_DIR.
3. Write config files with your changes only (look at eg satpy/etc/composites/seviri.yaml for inspiration), pointing to the custom module containing your composites. Don't forget to add changes to the enhancement/generic.cfg file too.
4. Put your composites module on the python path.

With that, you should be able to load your new composite directly.

.. todo::

    How to save custom-made composites
