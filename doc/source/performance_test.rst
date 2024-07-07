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
- Each round has one single condition tested. The result is stored in a csv file. After that, the machine will
  take a 1-min rest to let the CPU cool down.
- After all the tests finished, it collects all the result csv files, summarizing and visualizing them into the HTML
  report.


Preparations
============
1. Additional packages required
-------------------------------
- **psutil:** Record CPU/memory usage.
- **pandas:** Analyze test result.
- **matplotlib**: Plot the charts for report.
- **py-cpuinfo**: Get the CPU model for report.


2. Choose the composite and get corresponding datasets
------------------------------------------------------
Usually the composite should be the one involving a lot of computation like atmospheric correction. For most of the
earth observing satellites, this could be ``true_color`` or something like that.

Although one scene is enough to run the test, 3-5 scenes would be better to get the average.

- For geostationary satellites, it is recommended to get those around **solar noon** under **full-disk** scan mode.
- For polar orbit satellites, scenes should be around the **same area** so the intensities of the computation are similar.


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
Some conditions or resamplers may consume a hell of physical memory and then swap. When both are at their limits,
the OS may just kill the test process without any warnings or errors recorded.


5. Arrange your time and work
-----------------------------
The whole test progress could last hours long depending on the conditions. Keep the machine free during this period.
Avoid any unnecessary background jobs like software update.


Usage
=====
.. note::

    Both ``simple_test`` and ``resampler_test`` collect all the results under ``work_dir`` and produce the report
    in the same format. So if you already have some previous tests, just keep them in the same directory and the
    test will merge them into one automatically.

Initialize
----------
.. autofunction:: performance_test.SatpyPerformanceTest.__init__

.. code-block:: python

    from performance_test import SatpyPerformanceTest
    tester = SatpyPerformanceTest(work_dir="C:/Users/ABC/Downloads/Sat/Geo/ABI pef test",
                                  folder_pattern="G16_s*_e*_FLDK",
                                  reader_name="abi_l1b",
                                  composite="true_color",
                                  chunk_size_opts=[16, 64],
                                  worker_opts=[8, 16])

simple_test
-----------
.. autofunction:: performance_test.SatpyPerformanceTest.simple_test

.. code-block:: python

    # You can set some system environments related to satpy before running the test.
    os.environ["PSP_CONFIG_FILE"] = "D:/satpy_config/pyspectral/pyspectral.yaml"

    tester.simple_test(diff_res=True)

resampler_test
--------------
.. autofunction:: performance_test.SatpyPerformanceTest.resampler_test

.. code-block:: python

    from pyresample.geometry import AreaDefinition

    proj = "+proj=lcc +lon_0=-96 +lat_1=20 +lat_2=60 +datum=WGS84 +ellps=WGS84"
    width = 8008
    height = 8008
    area_extent = (-106000, 2635000, 3898000, 6639000)
    nprocs=8

    area_def = AreaDefinition(area_id="NorthAmerica", proj_id="lcc", description="na",
                              projection=proj, width=width, height=height, area_extent=area_extent, nprocs=nprocs)
    new_tester.resampler_test(resamplers=["bilinear", "ewa"],
                              area_def=area_def,
                              resampler_kwargs={
                              "bilinear": {"cache_dir": "C:/Users/45107/Downloads/Sat/Geo/ABI pef test/cache"},
                              "ewa": {"weight_delta_max": 40, "weight_distance_max": 2},
                              })
.. note::

    When you test ``bilinear`` or ``nearest`` resampler on geostationary datasets and want to both accelerate the test
    and exclude the impact of resampling cache, it is recommended to pre-build the cache with just one scene and
    one condition. And by that, you can also have a chance to tell how big the difference is between with and
    without cache (Sometimes, it's VERY, especially for ``bilinear``).

How to test ``OMP_NUM_THREADS``?
--------------------------------
``OMP_NUM_THREADS`` should be set outside the python script. In **Linux**, you can do it temporarily by

.. code-block:: shell

    OMP_NUM_THREADS=4 python your_test_script.py

In **Windows**:

.. code-block:: batch

    set OMP_NUM_THREADS=4 && python your_test_script.py

You can also choose not to set it. Normally the program will use as many logic cores as available. Either way, the test
will pick up the correct value and pass it to the report.
