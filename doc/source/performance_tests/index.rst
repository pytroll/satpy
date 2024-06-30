=================
Performance Tests
=================

For better performace tweaks on specific readers, here're the results from a series
of tests involving ``DASK_ARRAY__CHUNK_SIZE``, ``DASK_NUM_WORKERS`` and other options
mentioned in :doc:`FAQ <../faq>`.

.. toctree::
    :maxdepth: 1

    abi_l1b_tests


Test platform
-------------
+----------+--------------------------------------+
| CPU      | 1x 8-core, 8-thread i7-9700k @4.6GHz |
+----------+--------------------------------------+
| Memory   | 2x 32GB DDR4                         |
+----------+--------------------------------------+
| SSD      | 1x Samsung 980 Pro PCI-E 2TB         |
+----------+--------------------------------------+
| OS       | Windows 11 23H2 Workstation Pro      |
+----------+--------------------------------------+


Conda environment
-----------------
+------------+--------------------------------------+
| Channel    | conda-forge                          |
+------------+--------------------------------------+
| Python     | 3.12.3                               |
+------------+--------------------------------------+
| dask       | 2024.6.2                             |
+------------+--------------------------------------+
| numpy      | 2.0.0                                |
+------------+--------------------------------------+
| satpy      | 0.49                                 |
+------------+--------------------------------------+
| pyresample | 1.28.3                               |
+------------+--------------------------------------+
| pyspectral | 0.13.1                               |
+------------+--------------------------------------+
| psutil     | 6.0.0                                |
+------------+--------------------------------------+


Test procedure
--------------
- Each round will go through 5 scenes to calculate average.

- The composite will usually be the default ``true_color`` which requires heavy computation like atmospheric corrections.

- A new monitor thread using ``psutil`` will record the CPU and memory usage synchronously. The sample rate is around 0.5 seconds.

- When the current round finished, the machine will take a 2-min rest to let the CPU cool down.

- After that, reboot will clear the system cache and prevent the test program from taking advantage of it.


Test conditions
---------------
+------------------------------------+--------------------------------------------------------+
| DASK_ARRAY__CHUNK_SIZE (in MiB)    | 16, 32, 64, 96, 128                                    |
+------------------------------------+--------------------------------------------------------+
| DASK_ARRAY__CHUNK_SIZE (in arrays) | 512x512, 1024x1024, 2048x2048, 3072x3072, 4096x4096    |
+------------------------------------+--------------------------------------------------------+
| DASK_NUM_WORKERS                   | 8, 12, 16                                              |
+------------------------------------+--------------------------------------------------------+
| OMP_NUM_THREADS                    | 8                                                      |
+------------------------------------+--------------------------------------------------------+
| generate=False                     | Used when the composite requires different resolutions |
+------------------------------------+--------------------------------------------------------+
| nprocs=8                           | Used on ``nearest`` or ``bilinear`` resampling         |
+------------------------------------+--------------------------------------------------------+
| resampling cache                   | Used on ``nearest`` or ``bilinear`` resampling         |
+------------------------------------+--------------------------------------------------------+

General conclusions
-------------------