===========================
 Installation instructions
===========================

Getting the files and installing them
=====================================

First you need to get the files from github::

  cd /path/to/my/source/directory/
  git clone git://github.com/mraspaud/mpop.git

You can also retreive a tarball from there if you prefer, then run::
  
  tar zxvf tarball.tar.gz

Then you need to install mpop on you computer::

  cd mpop
  python setup.py install [--prefix=/my/custom/installation/directory]

You can also install it in develop mode to make it easier to hack::

  python setup.py develop [--prefix=/my/custom/installation/directory]


Configuration
=============

Environment variables
---------------------

Environment variables which are needed for mpop are the `PYTHONPATH` of course,
and the `PPP_CONFIG_DIR`, which is the directory where the configuration files
are to be found. If the latter is not defined, the `etc` directory of the mpop
installation is used.

Input data directories
----------------------

The input data directories are setup in the satellite configuration files,
which can be found in the `PPP_CONFIG_DIR` directory (some template files are
provided with mpop in the `etc` directory):

.. code-block:: ini

   [seviri-level1]
   format = 'xrit/MSG'
   dir='/data/geo_in'
   filename='H-000-MSG?__-MSG?________-%(channel)s-%(segment)s-%Y%m%d%H%M-__'
   filename_pro='H-000-MSG?__-MSG?________-_________-%(segment)s-%Y%m%d%H%M-__'
   filename_epi='H-000-MSG?__-MSG?________-_________-%(segment)s-%Y%m%d%H%M-__'
        

   [seviri-level2]
   format='mipp_xrit'


The different levels indicate different steps of the reading. The `level2`
section gives at least the plugin to read the data with. In some cases, the
data is first read from another level, as is this case with HRIT/LRIT data when
we use mipp_: there we use the `level1` section.

The data location is generally dealt in to parts: the directory and the
filename. There can also be additional filenames depending on the reader
plugin: here, mipp needs also the filename for prologue and epilogue files.

Note that the section starts with the name of the instrument. This is important
in the case where several instruments are available for the same satellite.
Note also that the filename can contain wildcards (`*` and `?`) and optional
values (here channel, segment, and time markers). It is up to the input plugin
to handle these constructs if needed.


.. _mipp: http://www.github.com/loerum/mipp
