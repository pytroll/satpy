FAQ
===

Below you'll find frequently asked questions, performance tips, and other
topics that don't really fit in to the rest of the SatPy documentation.

If you have any other questions that aren't answered here feel free to make
an issue on GitHub or talk to us on the Slack team or mailing list. See the
:ref:`contributing <dev_help>` documentation for more information.

.. contents:: Topics
    :depth: 1
    :local:

Why is SatPy slow on my powerful machine?
-----------------------------------------

SatPy depends heavily on the dask library for its performance. However,
on some systems dask's default settings can actually hurt performance.
By default dask will create a "worker" for each logical core on your
system. In most systems you have twice as many logical cores
(also known as threaded cores) as physical cores. Managing and communicating
with all of these workers can slow down dask, especially when they aren't all
being used by most SatPy calculations. One option is to limit the number of
workers by doing the following at the **top** of your python code:

.. code-block:: python

    import dask
    from multiprocessing.pool import ThreadPool
    dask.config.set(pool=ThreadPool(8))
    # all other SatPy imports and code

This will limit dask to using 8 workers. Typically numbers between 4 and 8
are good starting points.

Similarly, if you have many workers processing large chunks of data you may
be using much more memory than you expect. If you limit the number of workers
*and* the size of the data chunks being processed by each worker you can
reduce the overall memory usage. Default chunk size can be configured in SatPy
by setting the following environment variable:

.. code-block:: bash

    export PYTROLL_CHUNK_SIZE=2048

This could also be set inside python using ``os.environ``, but must be set
**before** SatPy is imported. This value defaults to 4096, meaning each
chunk of data will be 4096 rows by 4096 columns. In the future setting this
value will change to be easier to set in python.

How do I avoid memory errors?
-----------------------------

If your environment is using many dask workers, it may be using more memory
than it needs to be using. See the "Why is SatPy slow on my powerful machine?"
question above for more information on changing SatPy's memory usage.
