===========================
 Installation instructions
===========================

Getting the files
=================

Getting files from Git repository
---------------------------------

This is probably the best way if you plan to hack the package::

  git clone git://github.com/mraspaud/mpop.git

After this you will be in the master branch. This is what is in production at
the moment. You might be fine with this, but if you want to be up-to-date,
there are two more branches you have to be aware of: *test* and *unstable*.

*test* receives bugfixes during freezing periods, while *unstable* has all the
latest great features, but is, as it's name suggests, potentially unstable.

To retrieve these branches from the repository, just do::

  git checkout -b test origin/test

and/or::

  git checkout -b unstable origin/unstable

Now that you got the source files, you can proceed to the configuration
section.

Getting files from a tarball
----------------------------

If you are in good terms with us, you might have gotten a tarball source
distribution. You then only need to depackage it::

  tar zxvf tarball.tar.gz

Configuration
=============

Main configuration
------------------

Several files define the configuration of MPPP.

First have a look at `setup.cfg` in the root directory. It should contain the
following lines::

  [install]
  prefix = /local_disk/opt/MPoP/current

  [numpy]
  numpy_inc = /usr/lib64/python2.5/site-packages/numpy/core/include

`prefix` defines the directory to install to. It is the directory that will
then be in your `PYTHONPATH` and that contains the compiled libs.

`numpy_inc` is the location of your numpy C headers. This is needed for the
Meteosat plugin only.

Then look at the `etc/offline.profile` file. Set up the first `PYTHONPATH` to
reflect your prefix, the second section to reflect your ACPG installation and
the third section to reflect your AHAMAP installation. The third section is
just needed for the Noaa plugin.

Plugin configurations
---------------------

Meteosat 09 plugin
******************

The `msg_dirs` section of the `etc/meteosat09.cfg` file has to reflect your MSG
installation.

Noaa 17 plugin
**************

The `dir` variable in the `etc/noaa17.cfg` file has to be set to the PPS
directory containing level 1B data.

Installing
==========

Just run::

  python setup.py install

or if you want to specify a different prefix::

  python setup.py install --prefix=/my/path/

Then don’t forget to source the `etc/offline.profile` file before you start
playing.

Happy hacking !

Distributing
============

After you hack the hell out of MPPP, you might want to package it and sell it
to make money. We can tell you how to do the first part.

To build a source tarball, and output it to the `dist` directory::

  python setup.py sdist

To build a binary tarball, and output it to the `dist` directory::

  python setup.py bdist

To build rpms, and output them to the `dist` directory::

  python setup.py bdist_rpm

Don’t forget to send us a bottle of wine or two if you become rich with this :)

