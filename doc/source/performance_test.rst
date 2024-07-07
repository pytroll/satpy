=================
Performance Tests
=================

For better performace tweaks on specific readers, a tool ``performance_test`` under ``benchmarks`` is introduced here.
It involves ``DASK_ARRAY__CHUNK_SIZE``, ``DASK_NUM_WORKERS``, ``OMP_NUM_THREADS`` and other options mentioned in
:doc:`FAQ <../faq>`. This tool can loop through conditions defined by these values finally give a report in HTML.
The report contains tables and charts for better understanding. It has two types of tests: ``simple_test``
and ``resampler_test``.


How it works?
=============
- The core is just a regular satpy script -- find datasets, load the composite, resample it if needed and
  save it as geotiff.
- A monitor thread using ``psutil`` records the CPU and memory usage synchronously. The sample rate is around
  0.5 seconds. Any errors during the test will also be recorded.
- Each round will have one single condition tested. The result is stored in a csv file. After that, the machine will
  take a 1-min rest to let the CPU cool down.
- After all the tests finished, it collects all the result csv files, visualizing and summarizing theme into the HTML
  report.


Preparations
============
1. Additional packages required
-------------------------------
- **psutil:** Record CPU/memory usage
- **pandas:** Analyze test result
- **matplotlib**: Plot the charts for report.
- **py-cpuinfo**: Get the CPU model for report.


2. Choose the composite and get corresponding datasets
------------------------------------------------------
Although one scene is enough to run the test, 3-5 scenes would be better to get the average.

- For geostationary satellites, it is recommended to get those around **solar noon** under **full-disk** scan mode.
- For polar satellites, scenes should be around the **same area** so the intensities of the computation are similar.


3. Organize the datasets
------------------------
One scene per folder. All the dataset folders should have the same naming patterns, e.g.:

.. code-block:: batch

    2024/06/29  09:06   <DIR>   G16_s20241691700214_e20241691709522_FLDK
    2024/06/29  09:06   <DIR>   G16_s20241701700215_e20241701709523_FLDK
    2024/06/29  09:06   <DIR>   G16_s20241711700217_e20241711709525_FLDK
    2024/06/29  09:06   <DIR>   G16_s20241721700219_e20241721709527_FLDK
    2024/06/29  09:06   <DIR>   G16_s20241731700220_e20241731709528_FLDK


4. Do I have enough swap memory?
--------------------------------
Some conditions or resamplers will consume a hell of memory and will need swap. When both are at their limits,
the OS may just kill the test process without any warnings or errors recorded.


5. Arrange your time and work
-----------------------------
The whole test progress may last hours long depending on the conditions. Keep the machine free during this period.
Turn off all the unnecessary background jobs such as software update.


Usage
=====
Initialize
----------

.. code-block:: python

    from performance_test import SatpyPerformanceTest
    tester = SatpyPerformanceTest(work_dir="C:/Users/ABC/Downloads/Sat/Geo/ABI pef test",
                                  folder_pattern="G16_s*_e*_FLDK",
                                  reader_name="abi_l1b",
                                  composite="true_color",
                                  chunk_size_opts=[16, 64],
                                  worker_opts=[8, 16])


``simple_test``
---------------
This will test the reader in dataset's original projection.

