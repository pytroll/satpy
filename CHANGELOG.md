## Version 0.49.0 (2024/06/05)

### Issues Closed

* [Issue 2790](https://github.com/pytroll/satpy/issues/2790) - VIIRS L1B DNB_SENZ file_key ([PR 2791](https://github.com/pytroll/satpy/pull/2791) by [@wjsharpe](https://github.com/wjsharpe))
* [Issue 2781](https://github.com/pytroll/satpy/issues/2781) - [Question] Sun Zenith Correction
* [Issue 2765](https://github.com/pytroll/satpy/issues/2765) - abi_l2_nc reader Key Error 'calibration' when trying to load Mask from fire Hot Spot ([PR 2794](https://github.com/pytroll/satpy/pull/2794) by [@djhoese](https://github.com/djhoese))

In this release 3 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 2804](https://github.com/pytroll/satpy/pull/2804) - Fix LI L2 accumulated products `'with_area_definition': False` 1-d coordinates computation
* [PR 2794](https://github.com/pytroll/satpy/pull/2794) - Fix ABI L2 datasets when unitless and no calibration ([2765](https://github.com/pytroll/satpy/issues/2765))
* [PR 2791](https://github.com/pytroll/satpy/pull/2791) - fixed DNB_SENZ file_key ([2790](https://github.com/pytroll/satpy/issues/2790))

#### Features added

* [PR 2807](https://github.com/pytroll/satpy/pull/2807) - Update the vii_l1b-reader, for new testdata format of VII
* [PR 2801](https://github.com/pytroll/satpy/pull/2801) - Replace pytest-lazyfixture with pytest-lazy-fixtures
* [PR 2800](https://github.com/pytroll/satpy/pull/2800) - Add numpy rules to ruff
* [PR 2799](https://github.com/pytroll/satpy/pull/2799) - Add netcdf4 to goci2 optional dependency in `pyproject.toml`
* [PR 2795](https://github.com/pytroll/satpy/pull/2795) - Add support for MERSI-1 on FY-3A/B/C
* [PR 2789](https://github.com/pytroll/satpy/pull/2789) - Activate LI L2 accumulated products gridding by default
* [PR 2787](https://github.com/pytroll/satpy/pull/2787) - Fix datetime imports
* [PR 2778](https://github.com/pytroll/satpy/pull/2778) - Add the reader for the fci L1C Africa files
* [PR 2776](https://github.com/pytroll/satpy/pull/2776) - Add option to choose start time to MSI SAFE reader
* [PR 2727](https://github.com/pytroll/satpy/pull/2727) - Refactor Sentinel-1 SAR-C reader

#### Documentation changes

* [PR 2789](https://github.com/pytroll/satpy/pull/2789) - Activate LI L2 accumulated products gridding by default

#### Backward incompatible changes

* [PR 2789](https://github.com/pytroll/satpy/pull/2789) - Activate LI L2 accumulated products gridding by default

#### Refactoring

* [PR 2787](https://github.com/pytroll/satpy/pull/2787) - Fix datetime imports

#### Clean ups

* [PR 2797](https://github.com/pytroll/satpy/pull/2797) - Add missing coverage configuration section to pyproject.toml
* [PR 2784](https://github.com/pytroll/satpy/pull/2784) - Fix various issues in unstable CI

In this release 18 pull requests were closed.


## Version 0.48.0 (2024/04/22)

### Issues Closed

* [Issue 2782](https://github.com/pytroll/satpy/issues/2782) - Documentation points to missing setup.py ([PR 2786](https://github.com/pytroll/satpy/pull/2786) by [@mraspaud](https://github.com/mraspaud))
* [Issue 2771](https://github.com/pytroll/satpy/issues/2771) - Load data in another datatype rather than float64
* [Issue 2759](https://github.com/pytroll/satpy/issues/2759) - 'defusedxml' missing in "msi_safe" extras ([PR 2761](https://github.com/pytroll/satpy/pull/2761) by [@fwfichtner](https://github.com/fwfichtner))
* [Issue 2749](https://github.com/pytroll/satpy/issues/2749) - [Question] Resample of mesoscale data gives blank data
* [Issue 2747](https://github.com/pytroll/satpy/issues/2747) - Cannot load from MTG FCI L1C data
* [Issue 2729](https://github.com/pytroll/satpy/issues/2729) - Add Viirs L2 Reader + Enhancments ([PR 2740](https://github.com/pytroll/satpy/pull/2740) by [@wjsharpe](https://github.com/wjsharpe))
* [Issue 2695](https://github.com/pytroll/satpy/issues/2695) - Improvements for BackgroundCompositor ([PR 2696](https://github.com/pytroll/satpy/pull/2696) by [@yukaribbba](https://github.com/yukaribbba))

In this release 7 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 2786](https://github.com/pytroll/satpy/pull/2786) - Remove doc references to setup.py ([2782](https://github.com/pytroll/satpy/issues/2782))
* [PR 2779](https://github.com/pytroll/satpy/pull/2779) - Convert Sentinel-2 MSI sensor name to lowercase in the reader YAML config file and add support for "counts" calibration
* [PR 2774](https://github.com/pytroll/satpy/pull/2774) - Fix the viirs EDR tests for newer xarray
* [PR 2761](https://github.com/pytroll/satpy/pull/2761) - Add missing defusedxml ([2759](https://github.com/pytroll/satpy/issues/2759))
* [PR 2754](https://github.com/pytroll/satpy/pull/2754) - Bugfix vgac reader
* [PR 2701](https://github.com/pytroll/satpy/pull/2701) - Ici reader tiepoints bugfix
* [PR 2696](https://github.com/pytroll/satpy/pull/2696) - Add double alpha channel support and improve metadata behaviours for BackgroundCompositor ([2695](https://github.com/pytroll/satpy/issues/2695))

#### Features added

* [PR 2780](https://github.com/pytroll/satpy/pull/2780) - Add new (Eumetrain) FCI RGB composites
* [PR 2767](https://github.com/pytroll/satpy/pull/2767) - Use flags from file when available in OLCI NC reader
* [PR 2763](https://github.com/pytroll/satpy/pull/2763) - Replace setup with pyproject.toml
* [PR 2762](https://github.com/pytroll/satpy/pull/2762) - Add support for EO-SIP AVHRR LAC data
* [PR 2753](https://github.com/pytroll/satpy/pull/2753) - Add fsspec support to `li_l2_nc` reader
* [PR 2740](https://github.com/pytroll/satpy/pull/2740) - Add VIIRS L2 Reader ([2729](https://github.com/pytroll/satpy/issues/2729))
* [PR 2696](https://github.com/pytroll/satpy/pull/2696) - Add double alpha channel support and improve metadata behaviours for BackgroundCompositor ([2695](https://github.com/pytroll/satpy/issues/2695))
* [PR 2595](https://github.com/pytroll/satpy/pull/2595) - VGAC decode the time variable

#### Documentation changes

* [PR 2786](https://github.com/pytroll/satpy/pull/2786) - Remove doc references to setup.py ([2782](https://github.com/pytroll/satpy/issues/2782))
* [PR 2766](https://github.com/pytroll/satpy/pull/2766) - Add Data Store to EUMETSAT part
* [PR 2750](https://github.com/pytroll/satpy/pull/2750) - Add missing `h` docstring information to _geos_area.py

In this release 18 pull requests were closed.


## Version 0.47.0 (2024/02/21)

### Issues Closed

* [Issue 2734](https://github.com/pytroll/satpy/issues/2734) - Using a static image alters time information ([PR 2737](https://github.com/pytroll/satpy/pull/2737) by [@pnuu](https://github.com/pnuu))
* [Issue 2723](https://github.com/pytroll/satpy/issues/2723) - MODIS Satpy scene Don't know how to open the following files: {'MOD021KM.A2017131.1325.061.2017314123114.hdf'}
* [Issue 2719](https://github.com/pytroll/satpy/issues/2719) - Add lat lon to Seviri plots
* [Issue 2718](https://github.com/pytroll/satpy/issues/2718) - Set invert as a modifier when do composite
* [Issue 2712](https://github.com/pytroll/satpy/issues/2712) - mitiff writer add config option to add or not to add the size of a pixel in the proj string ([PR 2713](https://github.com/pytroll/satpy/pull/2713) by [@TAlonglong](https://github.com/TAlonglong))
* [Issue 2710](https://github.com/pytroll/satpy/issues/2710) - scene.save_datasets() outputs different values for AHI_HSD reader with calibration="brightness_temperature"
* [Issue 2708](https://github.com/pytroll/satpy/issues/2708) - this is regarding slstr_l1b geometry
* [Issue 2703](https://github.com/pytroll/satpy/issues/2703) - read swath in loop
* [Issue 2680](https://github.com/pytroll/satpy/issues/2680) - satpy_cf_nc reader cannot read FCI file written with cf writer
* [Issue 2672](https://github.com/pytroll/satpy/issues/2672) - Changes in NWC SAF GEO v2021 data ([PR 2673](https://github.com/pytroll/satpy/pull/2673) by [@pnuu](https://github.com/pnuu))
* [Issue 2630](https://github.com/pytroll/satpy/issues/2630) - wrong start_time with BackgroundCompositor ([PR 2737](https://github.com/pytroll/satpy/pull/2737) by [@pnuu](https://github.com/pnuu))
* [Issue 2447](https://github.com/pytroll/satpy/issues/2447) - add more options to time handling in combine_metadata ([PR 2737](https://github.com/pytroll/satpy/pull/2737) by [@pnuu](https://github.com/pnuu))
* [Issue 2446](https://github.com/pytroll/satpy/issues/2446) - combine metadata in `MultiFiller` ([PR 2737](https://github.com/pytroll/satpy/pull/2737) by [@pnuu](https://github.com/pnuu))
* [Issue 2427](https://github.com/pytroll/satpy/issues/2427) - Wrong start_time, end_time attributes after MultiScene.blend(blend_function=timeseries) ([PR 2737](https://github.com/pytroll/satpy/pull/2737) by [@pnuu](https://github.com/pnuu))
* [Issue 2319](https://github.com/pytroll/satpy/issues/2319) - slstr_l2.yaml points to deleted slstr_l2.py ([PR 2731](https://github.com/pytroll/satpy/pull/2731) by [@djhoese](https://github.com/djhoese))
* [Issue 1921](https://github.com/pytroll/satpy/issues/1921) - Standardize dataset information for SEVIRI and FCI L2 products
* [Issue 1174](https://github.com/pytroll/satpy/issues/1174) - combine_metadata only supports the average of time attrs ([PR 2737](https://github.com/pytroll/satpy/pull/2737) by [@pnuu](https://github.com/pnuu))

In this release 17 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 2743](https://github.com/pytroll/satpy/pull/2743) - Fix nominal time attributes in SEVIRI HRIT ([](https://github.com/`nominal_start/issues/))
* [PR 2742](https://github.com/pytroll/satpy/pull/2742) - Fix nominal end time in AHI HSD
* [PR 2737](https://github.com/pytroll/satpy/pull/2737) - Change `start_time` and `end_time` handling in `combine_metadata` ([2734](https://github.com/pytroll/satpy/issues/2734), [2630](https://github.com/pytroll/satpy/issues/2630), [2447](https://github.com/pytroll/satpy/issues/2447), [2446](https://github.com/pytroll/satpy/issues/2446), [2427](https://github.com/pytroll/satpy/issues/2427), [1174](https://github.com/pytroll/satpy/issues/1174))
* [PR 2731](https://github.com/pytroll/satpy/pull/2731) - Remove slstr_l2 reader in favor of ghrsst_l2 ([2319](https://github.com/pytroll/satpy/issues/2319))
* [PR 2730](https://github.com/pytroll/satpy/pull/2730) - Pin pytest to fix CI
* [PR 2726](https://github.com/pytroll/satpy/pull/2726) - Fix AGRI L1 C07 having a valid LUT value for its fill value ([565](https://github.com/ssec/polar2grid/issues/565))
* [PR 2713](https://github.com/pytroll/satpy/pull/2713) - Add kwargs config option to turn off mitiff corner correction ([2712](https://github.com/pytroll/satpy/issues/2712))
* [PR 2711](https://github.com/pytroll/satpy/pull/2711) - Add support for NOAA-21 in MiRS limb correction
* [PR 2707](https://github.com/pytroll/satpy/pull/2707) - Fix 'viirs_edr' renaming two sets of dimensions to the same names
* [PR 2700](https://github.com/pytroll/satpy/pull/2700) - Fix eps_l1b reader Delayed usage causing docs failures

#### Features added

* [PR 2746](https://github.com/pytroll/satpy/pull/2746) - Fix concurrency group in ci
* [PR 2745](https://github.com/pytroll/satpy/pull/2745) - Sort reader table by name + diverse fixes
* [PR 2744](https://github.com/pytroll/satpy/pull/2744) - Fix cutoffs for night_ir_alpha and bump up trollimage version
* [PR 2737](https://github.com/pytroll/satpy/pull/2737) - Change `start_time` and `end_time` handling in `combine_metadata` ([2734](https://github.com/pytroll/satpy/issues/2734), [2630](https://github.com/pytroll/satpy/issues/2630), [2447](https://github.com/pytroll/satpy/issues/2447), [2446](https://github.com/pytroll/satpy/issues/2446), [2427](https://github.com/pytroll/satpy/issues/2427), [1174](https://github.com/pytroll/satpy/issues/1174))
* [PR 2728](https://github.com/pytroll/satpy/pull/2728) - Update asv dependencies
* [PR 2720](https://github.com/pytroll/satpy/pull/2720) - Add support for the MERSI-RM instrument on FY-3G
* [PR 2714](https://github.com/pytroll/satpy/pull/2714) - Add QC-based filtering to the VIIRS EDR AOD550 product
* [PR 2675](https://github.com/pytroll/satpy/pull/2675) - Make CF encoding of dataset attributes public
* [PR 2673](https://github.com/pytroll/satpy/pull/2673) - Add NWC SAF GEO v2021 ASIIF-TF and ASII-GW dataset names ([2672](https://github.com/pytroll/satpy/issues/2672))
* [PR 2534](https://github.com/pytroll/satpy/pull/2534) - Add fsspec functionality to `viirs_sdr` reader
* [PR 2441](https://github.com/pytroll/satpy/pull/2441) - Add channel aliases to the CLAVRx reader to facilitate composites

#### Documentation changes

* [PR 2700](https://github.com/pytroll/satpy/pull/2700) - Fix eps_l1b reader Delayed usage causing docs failures

#### Backward incompatible changes

* [PR 2731](https://github.com/pytroll/satpy/pull/2731) - Remove slstr_l2 reader in favor of ghrsst_l2 ([2319](https://github.com/pytroll/satpy/issues/2319))

#### Refactoring

* [PR 2699](https://github.com/pytroll/satpy/pull/2699) - Move Scene.to_hvplot internals to _scene_converters

#### Clean ups

* [PR 2711](https://github.com/pytroll/satpy/pull/2711) - Add support for NOAA-21 in MiRS limb correction
* [PR 2700](https://github.com/pytroll/satpy/pull/2700) - Fix eps_l1b reader Delayed usage causing docs failures
* [PR 2689](https://github.com/pytroll/satpy/pull/2689) - Fix/supress warnings in reader tests ([](https://github.com/and/issues/))
* [PR 2665](https://github.com/pytroll/satpy/pull/2665) - FCI L2 CF harmonization
* [PR 2597](https://github.com/pytroll/satpy/pull/2597) - Update CI to test Python 3.12

In this release 29 pull requests were closed.


## Version 0.46.0 (2023/12/18)

### Issues Closed

* [Issue 2668](https://github.com/pytroll/satpy/issues/2668) - FCI HRFI true_color unavailable even after native resampling if upper_right_corner is used ([PR 2690](https://github.com/pytroll/satpy/pull/2690) by [@djhoese](https://github.com/djhoese))
* [Issue 2664](https://github.com/pytroll/satpy/issues/2664) - Cannot generate day-night composites
* [Issue 2654](https://github.com/pytroll/satpy/issues/2654) - Unable to read radiance with AVHRR EPS ([PR 2655](https://github.com/pytroll/satpy/pull/2655) by [@mraspaud](https://github.com/mraspaud))
* [Issue 2647](https://github.com/pytroll/satpy/issues/2647) - Preservation of input data dtype in processing FCI data
* [Issue 2618](https://github.com/pytroll/satpy/issues/2618) - GCOM-C Support (Continued) ([PR 1094](https://github.com/pytroll/satpy/pull/1094) by [@mraspaud](https://github.com/mraspaud))
* [Issue 2588](https://github.com/pytroll/satpy/issues/2588) - FCI chunks/segments out of order if pad_data=False ([PR 2692](https://github.com/pytroll/satpy/pull/2692) by [@ameraner](https://github.com/ameraner))
* [Issue 2263](https://github.com/pytroll/satpy/issues/2263) - VIIRS day composite 'snow_age' does not work with Satpy 0.37.1
* [Issue 1496](https://github.com/pytroll/satpy/issues/1496) - Improve error reporting of satpy.utils.get_satpos
* [Issue 1086](https://github.com/pytroll/satpy/issues/1086) - Add a reader for GCOM-C Level 1 data ([PR 1094](https://github.com/pytroll/satpy/pull/1094) by [@mraspaud](https://github.com/mraspaud))

In this release 9 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 2694](https://github.com/pytroll/satpy/pull/2694) - Match all projectables in `NDVIHybridGreen.__call__` to avoid coordinate mismatch errors ([2668](https://github.com/pytroll/satpy/issues/2668), [2668](https://github.com/pytroll/satpy/issues/2668))
* [PR 2692](https://github.com/pytroll/satpy/pull/2692) - Anticipate filehandler sorting in `GEOSegmentYAMLReader` to have sorted handlers also with `pad_data=False` ([2588](https://github.com/pytroll/satpy/issues/2588))
* [PR 2690](https://github.com/pytroll/satpy/pull/2690) - Fix composites failing on non-aligned geolocation coordinates ([2668](https://github.com/pytroll/satpy/issues/2668))
* [PR 2682](https://github.com/pytroll/satpy/pull/2682) - Update AHI HSD reader to correctly handle singleton arrays.
* [PR 2674](https://github.com/pytroll/satpy/pull/2674) - Update xarray version in CF writer tests for compression kwarg
* [PR 2671](https://github.com/pytroll/satpy/pull/2671) - Workaround AWIPS bug not handling integers properly in "awips_tiled" writer
* [PR 2669](https://github.com/pytroll/satpy/pull/2669) - Fix RealisticColors compositor upcasting data to float64
* [PR 2655](https://github.com/pytroll/satpy/pull/2655) - Fix missing radiance units in eps l1b ([2654](https://github.com/pytroll/satpy/issues/2654))

#### Features added

* [PR 2683](https://github.com/pytroll/satpy/pull/2683) - Fci/l2/amv/reader
* [PR 2679](https://github.com/pytroll/satpy/pull/2679) - Update MiRS reader coefficient files to newer version
* [PR 2677](https://github.com/pytroll/satpy/pull/2677) - Add remaining JPSS satellite platform aliases to "mirs" reader ([665](https://github.com/ssec/polar2grid/issues/665))
* [PR 2669](https://github.com/pytroll/satpy/pull/2669) - Fix RealisticColors compositor upcasting data to float64
* [PR 2660](https://github.com/pytroll/satpy/pull/2660) - Update tropomi_l2 reader with "_reduced" file patterns
* [PR 2557](https://github.com/pytroll/satpy/pull/2557) - Add baseline for GeoColor composite including FCI, AHI and ABI recipes
* [PR 2106](https://github.com/pytroll/satpy/pull/2106) - Add Scene function to use Hvplot backend visualization
* [PR 1094](https://github.com/pytroll/satpy/pull/1094) - Add Gcom-C sgli reader ([2618](https://github.com/pytroll/satpy/issues/2618), [1086](https://github.com/pytroll/satpy/issues/1086))

#### Backward incompatible changes

* [PR 2684](https://github.com/pytroll/satpy/pull/2684) - Get rid of warnings in compositor tests

#### Clean ups

* [PR 2691](https://github.com/pytroll/satpy/pull/2691) - Reduce the number of warnings in writer tests
* [PR 2690](https://github.com/pytroll/satpy/pull/2690) - Fix composites failing on non-aligned geolocation coordinates ([2668](https://github.com/pytroll/satpy/issues/2668))
* [PR 2684](https://github.com/pytroll/satpy/pull/2684) - Get rid of warnings in compositor tests
* [PR 2681](https://github.com/pytroll/satpy/pull/2681) - Get rid of warnings in resampler tests
* [PR 2676](https://github.com/pytroll/satpy/pull/2676) - Convert times in SEVIRI readers to nanosecond precision to silence warnings
* [PR 2658](https://github.com/pytroll/satpy/pull/2658) - Update unstable version of h5py in CI

In this release 23 pull requests were closed.


## Version 0.45.0 (2023/11/29)

### Issues Closed

* [Issue 2646](https://github.com/pytroll/satpy/issues/2646) - satpy/tests/scene_tests/test_resampling.py is using called_once in assertions rather than assert_called_once, causing test failures on Python 3.12 ([PR 2648](https://github.com/pytroll/satpy/pull/2648) by [@ArrayBolt3](https://github.com/ArrayBolt3))
* [Issue 2643](https://github.com/pytroll/satpy/issues/2643) - SunZenithReducer defaults make True Color FCI imagery too dark at high solar zenith angles ([PR 2653](https://github.com/pytroll/satpy/pull/2653) by [@ameraner](https://github.com/ameraner))
* [Issue 2638](https://github.com/pytroll/satpy/issues/2638) - Update AVHRR EPS reader to read cloud flags information ([PR 2639](https://github.com/pytroll/satpy/pull/2639) by [@fwfichtner](https://github.com/fwfichtner))
* [Issue 2619](https://github.com/pytroll/satpy/issues/2619) - NDVI hybrid green correction triggers early dask computations ([PR 2623](https://github.com/pytroll/satpy/pull/2623) by [@pnuu](https://github.com/pnuu))
* [Issue 2614](https://github.com/pytroll/satpy/issues/2614) - DayNightCompositor triggers early dask computation ([PR 2617](https://github.com/pytroll/satpy/pull/2617) by [@pnuu](https://github.com/pnuu))
* [Issue 2613](https://github.com/pytroll/satpy/issues/2613) - modifier NIREmissivePartFromReflectance triggers early dask computation
* [Issue 2604](https://github.com/pytroll/satpy/issues/2604) - grid_mapping attrs lead to failure of cf writer
* [Issue 2601](https://github.com/pytroll/satpy/issues/2601) - Is the 31(32)-band read by the modis_l1b reader converted to bright temperature by default?

In this release 8 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 2648](https://github.com/pytroll/satpy/pull/2648) - Fix assert_called_once usage in resample tests ([2646](https://github.com/pytroll/satpy/issues/2646))
* [PR 2635](https://github.com/pytroll/satpy/pull/2635) - Fix nwcsaf_geo start time to be nominal time
* [PR 2627](https://github.com/pytroll/satpy/pull/2627) - Fix ABI readers using wrong dtype for resolution-based chunks
* [PR 2625](https://github.com/pytroll/satpy/pull/2625) - Cleanup various warnings encountered during tests
* [PR 2623](https://github.com/pytroll/satpy/pull/2623) - Fix unnecessary Dask `compute()`s in `NDVIHybridGreen` compositor ([2619](https://github.com/pytroll/satpy/issues/2619))
* [PR 2617](https://github.com/pytroll/satpy/pull/2617) - Reduce Dask computations in `DayNightCompositor` ([2614](https://github.com/pytroll/satpy/issues/2614))
* [PR 2608](https://github.com/pytroll/satpy/pull/2608) - Fix ABI L2 to only convert reflectances to percentages
* [PR 2607](https://github.com/pytroll/satpy/pull/2607) - Fix ABI L2 reader to produce reflectances as percentages
* [PR 2606](https://github.com/pytroll/satpy/pull/2606) - Change platform name for EPIC (DSCOVR) to upper case.
* [PR 2585](https://github.com/pytroll/satpy/pull/2585) - Make caching warn if some of the args are unhashable

#### Features added

* [PR 2653](https://github.com/pytroll/satpy/pull/2653) - Update Sun-zenith reducer defaults ([2643](https://github.com/pytroll/satpy/issues/2643))
* [PR 2652](https://github.com/pytroll/satpy/pull/2652) - Add file pattern for CRRPh of NWC SAF GEO v2021
* [PR 2642](https://github.com/pytroll/satpy/pull/2642) - Set dtype for get_lonlats() in NIR reflectance calculation
* [PR 2640](https://github.com/pytroll/satpy/pull/2640) - Keep original dtype in DayNightCompositor
* [PR 2639](https://github.com/pytroll/satpy/pull/2639) - Update AVHRR EPS reader to read cloud flags information ([2638](https://github.com/pytroll/satpy/issues/2638))
* [PR 2637](https://github.com/pytroll/satpy/pull/2637) - Keep FCI data as 32-bit floats
* [PR 2632](https://github.com/pytroll/satpy/pull/2632) - Add reader for OSI SAF L3 products
* [PR 2631](https://github.com/pytroll/satpy/pull/2631) - Add a reader for MODIS Level 3 files in CMG format.
* [PR 2623](https://github.com/pytroll/satpy/pull/2623) - Fix unnecessary Dask `compute()`s in `NDVIHybridGreen` compositor ([2619](https://github.com/pytroll/satpy/issues/2619))
* [PR 2621](https://github.com/pytroll/satpy/pull/2621) - Add resolution-based chunking to ABI L1b reader
* [PR 2610](https://github.com/pytroll/satpy/pull/2610) - Remove legacy resampler code

#### Clean ups

* [PR 2648](https://github.com/pytroll/satpy/pull/2648) - Fix assert_called_once usage in resample tests ([2646](https://github.com/pytroll/satpy/issues/2646))
* [PR 2641](https://github.com/pytroll/satpy/pull/2641) - Add "A" and "D" checks to ruff config
* [PR 2634](https://github.com/pytroll/satpy/pull/2634) - Remove duplicate entries of required netcdf variables in FCI reader
* [PR 2625](https://github.com/pytroll/satpy/pull/2625) - Cleanup various warnings encountered during tests
* [PR 2624](https://github.com/pytroll/satpy/pull/2624) - Replace assertRaises with pytest.raises
* [PR 2621](https://github.com/pytroll/satpy/pull/2621) - Add resolution-based chunking to ABI L1b reader
* [PR 2612](https://github.com/pytroll/satpy/pull/2612) - Remove tests for removed and deprecated functionality
* [PR 2610](https://github.com/pytroll/satpy/pull/2610) - Remove legacy resampler code
* [PR 2586](https://github.com/pytroll/satpy/pull/2586) - Replace flake8 with ruff in pre-commit and ci linting
* [PR 2524](https://github.com/pytroll/satpy/pull/2524) - Refactor CFWriter utility into CF directory

In this release 31 pull requests were closed.


## Version 0.44.0 (2023/10/17)

### Issues Closed

* [Issue 2593](https://github.com/pytroll/satpy/issues/2593) - FY4A REGC data resampling return all nan
* [Issue 2591](https://github.com/pytroll/satpy/issues/2591) -  Is there a corresponding reader for S3A_SL_2_WST?
* [Issue 2581](https://github.com/pytroll/satpy/issues/2581) - Can reader 'modis_l1b' correct MODIS Bow Tie Effect?
* [Issue 2580](https://github.com/pytroll/satpy/issues/2580) - Does python3.8 and below seem to fail to install via the command line "conda install -c conda-forge satpy"?
* [Issue 2571](https://github.com/pytroll/satpy/issues/2571) - Add Calibration by Meirink et al for SEVIRI ([PR 2589](https://github.com/pytroll/satpy/pull/2589) by [@pdebuyl](https://github.com/pdebuyl))
* [Issue 2549](https://github.com/pytroll/satpy/issues/2549) - setuptools-scm-git-archive is obsolete -- use setuptools-scm >= 7 ([PR 2598](https://github.com/pytroll/satpy/pull/2598) by [@pdebuyl](https://github.com/pdebuyl))
* [Issue 2266](https://github.com/pytroll/satpy/issues/2266) - AGRI data fails with `native` resampling`

In this release 7 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 2598](https://github.com/pytroll/satpy/pull/2598) - remove setuptools_scm_git_archive from requirement ([2549](https://github.com/pytroll/satpy/issues/2549), [2549](https://github.com/pytroll/satpy/issues/2549))
* [PR 2579](https://github.com/pytroll/satpy/pull/2579) - Adapt satpy to numpy 2
* [PR 2575](https://github.com/pytroll/satpy/pull/2575) - Remove use of deprecated setuptools_scm_git_archive build package

#### Features added

* [PR 2600](https://github.com/pytroll/satpy/pull/2600) - Add some global EPSG 4326 gridded lat/lon areas.
* [PR 2589](https://github.com/pytroll/satpy/pull/2589) - Add meirink calib ([2571](https://github.com/pytroll/satpy/issues/2571))
* [PR 2584](https://github.com/pytroll/satpy/pull/2584) - Convert AHI HSD dask chunking to be based on band resolution
* [PR 2574](https://github.com/pytroll/satpy/pull/2574) - Rename ABI "night_microphysics_abi" composite to "night_microphysics"
* [PR 2572](https://github.com/pytroll/satpy/pull/2572) - Add reader for GERB high-resolution HDF5 files
* [PR 2558](https://github.com/pytroll/satpy/pull/2558) - New reader for Himawari L2 NOAA enterprise cloud products.
* [PR 2556](https://github.com/pytroll/satpy/pull/2556) - Implement modifier for reducing signal as a function of sunz angle
* [PR 2554](https://github.com/pytroll/satpy/pull/2554) - Implement non-linear scaling for NDVI hybrid green correction
* [PR 2488](https://github.com/pytroll/satpy/pull/2488) - Add a blend method to create temporal RGB from MultiScene
* [PR 2052](https://github.com/pytroll/satpy/pull/2052) - Add resolution dependent chunk sizing to 'modis_l1b' reader

#### Documentation changes

* [PR 2582](https://github.com/pytroll/satpy/pull/2582) - Add mastodon link
* [PR 2517](https://github.com/pytroll/satpy/pull/2517) - Add documentation on putting text onto images

#### Backward incompatible changes

* [PR 2574](https://github.com/pytroll/satpy/pull/2574) - Rename ABI "night_microphysics_abi" composite to "night_microphysics"

#### Clean ups

* [PR 2587](https://github.com/pytroll/satpy/pull/2587) - Remove libnetcdf specific build from CI env
* [PR 2578](https://github.com/pytroll/satpy/pull/2578) - Remove unneeded performance tracker in seviri reader
* [PR 2575](https://github.com/pytroll/satpy/pull/2575) - Remove use of deprecated setuptools_scm_git_archive build package

In this release 19 pull requests were closed.


## Version 0.43.0 (2023/07/03)

### Issues Closed

* [Issue 2519](https://github.com/pytroll/satpy/issues/2519) - MSG Dust RGB adding coastilnes and grid to the image
* [Issue 2506](https://github.com/pytroll/satpy/issues/2506) - Add xarray_kwargs capability to the geocat reader ([PR 2507](https://github.com/pytroll/satpy/pull/2507) by [@joleenf](https://github.com/joleenf))
* [Issue 2502](https://github.com/pytroll/satpy/issues/2502) - Cropping S3 image not working
* [Issue 2494](https://github.com/pytroll/satpy/issues/2494) - avhrr_l1b_gaclac fails to read most files from NOAA CLASS ([PR 2501](https://github.com/pytroll/satpy/pull/2501) by [@sfinkens](https://github.com/sfinkens))
* [Issue 2490](https://github.com/pytroll/satpy/issues/2490) - ninjogeotiff writer adds offset/scale factor when this is not meaningful ([PR 2491](https://github.com/pytroll/satpy/pull/2491) by [@gerritholl](https://github.com/gerritholl))
* [Issue 2483](https://github.com/pytroll/satpy/issues/2483) - Cacheing doesn't work with `scn.crop` ([PR 2485](https://github.com/pytroll/satpy/pull/2485) by [@djhoese](https://github.com/djhoese))
* [Issue 2465](https://github.com/pytroll/satpy/issues/2465) - Possibility of dual licensing: GPL-3.0 & MIT
* [Issue 2464](https://github.com/pytroll/satpy/issues/2464) - MITIFF writer using pillow: turn off compression due to rowsperstrip issues
* [Issue 2463](https://github.com/pytroll/satpy/issues/2463) - seviri_l1b_native reader issue with reading remote files (azure)
* [Issue 2409](https://github.com/pytroll/satpy/issues/2409) - Inconsistent behavior of time attributes in EUM L1 GEO readers ([PR 2420](https://github.com/pytroll/satpy/pull/2420) by [@YouvaEUMex](https://github.com/YouvaEUMex))
* [Issue 1749](https://github.com/pytroll/satpy/issues/1749) - Load from blended scene
* [Issue 859](https://github.com/pytroll/satpy/issues/859) - Doesn't recognize MODIS L2 file

In this release 12 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 2522](https://github.com/pytroll/satpy/pull/2522) - Fix CF tests due to new xarray release
* [PR 2516](https://github.com/pytroll/satpy/pull/2516) - Fix SEVIRI native reader failing when missing main header
* [PR 2510](https://github.com/pytroll/satpy/pull/2510) - Fix warnings from NWCSAF reader
* [PR 2507](https://github.com/pytroll/satpy/pull/2507) - Fix HDF4 support in geocat reader with hardcoded engine ([2506](https://github.com/pytroll/satpy/issues/2506))
* [PR 2492](https://github.com/pytroll/satpy/pull/2492) - Fix xarray version for cf tests
* [PR 2491](https://github.com/pytroll/satpy/pull/2491) - Change logic for ninjogeotiff gradient/axisintercept tags ([2490](https://github.com/pytroll/satpy/issues/2490))
* [PR 2485](https://github.com/pytroll/satpy/pull/2485) - Fix angle caching not handling a specific type of irregular chunking ([2483](https://github.com/pytroll/satpy/issues/2483))
* [PR 2481](https://github.com/pytroll/satpy/pull/2481) - Fix NWCSAF reading for NOAA-21

#### Features added

* [PR 2521](https://github.com/pytroll/satpy/pull/2521) - Add a median filter modifier
* [PR 2508](https://github.com/pytroll/satpy/pull/2508) - Add support for OLCI L2 files which are missing Frame_IDs
* [PR 2504](https://github.com/pytroll/satpy/pull/2504) - Improve flexibility of olci level2 reader
* [PR 2501](https://github.com/pytroll/satpy/pull/2501) - Add Pygac reference to avhrr_l1b_gaclac documentation ([2494](https://github.com/pytroll/satpy/issues/2494))
* [PR 2499](https://github.com/pytroll/satpy/pull/2499) - Add option to clip negative ABI radiances
* [PR 2497](https://github.com/pytroll/satpy/pull/2497) - Enable to pass a custom function to Scene.aggregate
* [PR 2489](https://github.com/pytroll/satpy/pull/2489) - Add "neutral_resolution_band" kwarg to RatioSharpenedRGB/SelfSharpenedRGB
* [PR 2480](https://github.com/pytroll/satpy/pull/2480) - Add helper-function for reading SEVIRI L1.5 Native header.
* [PR 2449](https://github.com/pytroll/satpy/pull/2449) - Generalise the `true_color_reproduction` composite and enhancement
* [PR 2420](https://github.com/pytroll/satpy/pull/2420) - Fix inconsistent behavior of time attributes in EUM L1 GEO readers ([2409](https://github.com/pytroll/satpy/issues/2409))
* [PR 2259](https://github.com/pytroll/satpy/pull/2259) - Refactor `CFWriter.save_datasets` and enable retrieval of equivalent xr.Dataset with `scn.to_xarray()`
* [PR 2117](https://github.com/pytroll/satpy/pull/2117) - Add reader for GMS-5 VISSR data

#### Documentation changes

* [PR 2514](https://github.com/pytroll/satpy/pull/2514) - Fix argument name in DayNightComposite example document
* [PR 2501](https://github.com/pytroll/satpy/pull/2501) - Add Pygac reference to avhrr_l1b_gaclac documentation ([2494](https://github.com/pytroll/satpy/issues/2494))
* [PR 2478](https://github.com/pytroll/satpy/pull/2478) - Fix eccodes package names in setup.py, update documentation for setting up development environment.
* [PR 2474](https://github.com/pytroll/satpy/pull/2474) - Reorganize seviri_l2_grib.yaml file and add more documentation to seviri_l1b_native.py

#### Clean ups

* [PR 2523](https://github.com/pytroll/satpy/pull/2523) - Convert CF Writer tests to pytest
* [PR 2486](https://github.com/pytroll/satpy/pull/2486) - Fix leftover deprecated nosetest teardown methods
* [PR 2478](https://github.com/pytroll/satpy/pull/2478) - Fix eccodes package names in setup.py, update documentation for setting up development environment.
* [PR 2474](https://github.com/pytroll/satpy/pull/2474) - Reorganize seviri_l2_grib.yaml file and add more documentation to seviri_l1b_native.py

In this release 28 pull requests were closed.


## Version 0.42.2 (2023/05/10)

### Issues Closed

* [Issue 2471](https://github.com/pytroll/satpy/issues/2471) - Missing dependencies in setup.py for running all the tests ([PR 2472](https://github.com/pytroll/satpy/pull/2472) by [@sjoro](https://github.com/sjoro))
* [Issue 2469](https://github.com/pytroll/satpy/issues/2469) - Problem in reprojecting MSG SEVIRI data

In this release 2 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 2473](https://github.com/pytroll/satpy/pull/2473) - Fix rayleigh correction not handling angles as required inputs
* [PR 2472](https://github.com/pytroll/satpy/pull/2472) - Add missing test dependencies and update dev environment documentation ([2471](https://github.com/pytroll/satpy/issues/2471))

#### Documentation changes

* [PR 2472](https://github.com/pytroll/satpy/pull/2472) - Add missing test dependencies and update dev environment documentation ([2471](https://github.com/pytroll/satpy/issues/2471))

#### Clean ups

* [PR 2472](https://github.com/pytroll/satpy/pull/2472) - Add missing test dependencies and update dev environment documentation ([2471](https://github.com/pytroll/satpy/issues/2471))
* [PR 2453](https://github.com/pytroll/satpy/pull/2453) - Fix various warnings encountered when running tests

In this release 5 pull requests were closed.


## Version 0.42.1 (2023/05/02)

### Issues Closed

* [Issue 2458](https://github.com/pytroll/satpy/issues/2458) - Fail to install satpy despite the version

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 2456](https://github.com/pytroll/satpy/pull/2456) - Fix Adaptive DNB composite bug introduced from refactoring

In this release 1 pull request was closed.


## Version 0.42.0 (2023/04/28)

### Issues Closed

* [Issue 2455](https://github.com/pytroll/satpy/issues/2455) - Highlights blacked-out in MODIS false-color image
* [Issue 2204](https://github.com/pytroll/satpy/issues/2204) - CLAVRx Reader for AHI data misses CLAVRx files with naming convention that retains full AHI original name. ([PR 2314](https://github.com/pytroll/satpy/pull/2314) by [@joleenf](https://github.com/joleenf))
* [Issue 1944](https://github.com/pytroll/satpy/issues/1944) - satpy v0.33 seems to run slower than v0.30 when I project modis l1b data

In this release 3 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 2454](https://github.com/pytroll/satpy/pull/2454) - Change optional rayleigh prerequisites to required for MODIS
* [PR 2451](https://github.com/pytroll/satpy/pull/2451) - Fix assumption that arrays have 2+ dimensions in CF writer ([74](https://github.com/foua-pps/level1c4pps/issues/74))
* [PR 2440](https://github.com/pytroll/satpy/pull/2440) - Fix nwcsaf pps palettes
* [PR 2437](https://github.com/pytroll/satpy/pull/2437) - Update MODIS composites with specific channel for Rayleigh correction
* [PR 2432](https://github.com/pytroll/satpy/pull/2432) - Remove usage of open_rasterio in some readers
* [PR 2417](https://github.com/pytroll/satpy/pull/2417) - Fix NUCAPS reader compatibility with new versions of xarray
* [PR 2394](https://github.com/pytroll/satpy/pull/2394) - Add weighted blended stacking to MultiScene (fixes multi-band handling)
* [PR 2168](https://github.com/pytroll/satpy/pull/2168) - Fix coordinate names in Ocean Color CCI reader.

#### Features added

* [PR 2439](https://github.com/pytroll/satpy/pull/2439) - Add support for unit conversion in image writers
* [PR 2438](https://github.com/pytroll/satpy/pull/2438) - Deprecate PYTROLL_CHUNK_SIZE and fallback to array.chunk-size config from dask
* [PR 2423](https://github.com/pytroll/satpy/pull/2423) - Support reading from S3 for AHI-HSD
* [PR 2422](https://github.com/pytroll/satpy/pull/2422) - Update condition of xarray version in CF writer tests
* [PR 2421](https://github.com/pytroll/satpy/pull/2421) - Add MODIS 05 L2 datasets to `modis_l2` reader
* [PR 2416](https://github.com/pytroll/satpy/pull/2416) - Drop support for Python 3.8 and add 3.11 tests
* [PR 2407](https://github.com/pytroll/satpy/pull/2407) - VIIRS VGAC reader
* [PR 2394](https://github.com/pytroll/satpy/pull/2394) - Add weighted blended stacking to MultiScene (fixes multi-band handling)
* [PR 2379](https://github.com/pytroll/satpy/pull/2379) - Add reader for IASI L2 CDR in NetCDF format
* [PR 2314](https://github.com/pytroll/satpy/pull/2314) - Update `clavrx` reader file patterns and flag_meanings handling ([2204](https://github.com/pytroll/satpy/issues/2204))
* [PR 2124](https://github.com/pytroll/satpy/pull/2124) - Add reader for FY-3E / MERSI-LL

#### Documentation changes

* [PR 2448](https://github.com/pytroll/satpy/pull/2448) - Add section to FAQ to describe how `generate=False` may speed up RGBs
* [PR 2431](https://github.com/pytroll/satpy/pull/2431) - Add codescene badge to README

#### Backward incompatible changes

* [PR 2416](https://github.com/pytroll/satpy/pull/2416) - Drop support for Python 3.8 and add 3.11 tests

#### Clean ups

* [PR 2450](https://github.com/pytroll/satpy/pull/2450) - Remove unneeded stickler config
* [PR 2429](https://github.com/pytroll/satpy/pull/2429) - Cleanup and separate Scene unit tests
* [PR 2416](https://github.com/pytroll/satpy/pull/2416) - Drop support for Python 3.8 and add 3.11 tests
* [PR 2394](https://github.com/pytroll/satpy/pull/2394) - Add weighted blended stacking to MultiScene (fixes multi-band handling)

In this release 26 pull requests were closed.


## Version 0.41.1 (2023/03/15)

### Issues Closed

* [Issue 2414](https://github.com/pytroll/satpy/issues/2414) - Certain composite images not plotting correctly using imshow - but they are fine using `scene.show()`

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 2415](https://github.com/pytroll/satpy/pull/2415) - Fix dataid attribute access in JMA HRIT readers

In this release 1 pull request was closed.


## Version 0.41.0 (2023/03/14)

### Issues Closed

* [Issue 2399](https://github.com/pytroll/satpy/issues/2399) - Latitude and Longitude for GK2A AMI products
* [Issue 2395](https://github.com/pytroll/satpy/issues/2395) - DataID.__getattr__ calls nonexisting method on parent class ([PR 2396](https://github.com/pytroll/satpy/pull/2396) by [@gerritholl](https://github.com/gerritholl))
* [Issue 2392](https://github.com/pytroll/satpy/issues/2392) - The cloud_top_height composite broken for PPS ([PR 2403](https://github.com/pytroll/satpy/pull/2403) by [@gerritholl](https://github.com/gerritholl))
* [Issue 2386](https://github.com/pytroll/satpy/issues/2386) - Docs incorrect for cf writer compression options ([PR 2390](https://github.com/pytroll/satpy/pull/2390) by [@sfinkens](https://github.com/sfinkens))
* [Issue 2288](https://github.com/pytroll/satpy/issues/2288) - Add support for reducing Rayleigh correction at large solar zenith angles ([PR 2380](https://github.com/pytroll/satpy/pull/2380) by [@yukaribbba](https://github.com/yukaribbba))
* [Issue 2244](https://github.com/pytroll/satpy/issues/2244) - Bug in CFWriter with netcdf4>=1.60 because of changing behaviour of zlib encoding specification ([PR 2390](https://github.com/pytroll/satpy/pull/2390) by [@sfinkens](https://github.com/sfinkens))

In this release 6 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 2412](https://github.com/pytroll/satpy/pull/2412) - Repair CF writer tests failing with libnetcdf >= 4.9.0
* [PR 2411](https://github.com/pytroll/satpy/pull/2411) - Fix warning when CF-writing a Scene with SwathDefinition area
* [PR 2408](https://github.com/pytroll/satpy/pull/2408) - Remove DataID attribute access in seviri_l2_grib reader
* [PR 2405](https://github.com/pytroll/satpy/pull/2405) - MWI v2 test dataset has BT calibration factors for each channel
* [PR 2403](https://github.com/pytroll/satpy/pull/2403) - Fix failing NWCSAF/PPS composites ([2392](https://github.com/pytroll/satpy/issues/2392))
* [PR 2390](https://github.com/pytroll/satpy/pull/2390) - Drop default compression in CF Writer ([2386](https://github.com/pytroll/satpy/issues/2386), [2244](https://github.com/pytroll/satpy/issues/2244))

#### Features added

* [PR 2396](https://github.com/pytroll/satpy/pull/2396) - Remove DataID attribute access ([2395](https://github.com/pytroll/satpy/issues/2395))
* [PR 2391](https://github.com/pytroll/satpy/pull/2391) - Increase warnings stacklevel
* [PR 2388](https://github.com/pytroll/satpy/pull/2388) - Fix gap between day/night-only composites
* [PR 2380](https://github.com/pytroll/satpy/pull/2380) - Add an interface for reducing Rayleigh correction at high solar zenith angles in composites YAML ([2288](https://github.com/pytroll/satpy/issues/2288))

#### Documentation changes

* [PR 2390](https://github.com/pytroll/satpy/pull/2390) - Drop default compression in CF Writer ([2386](https://github.com/pytroll/satpy/issues/2386), [2244](https://github.com/pytroll/satpy/issues/2244))

#### Clean ups

* [PR 2404](https://github.com/pytroll/satpy/pull/2404) - Convert TestYAMLFiles to pytest
* [PR 2397](https://github.com/pytroll/satpy/pull/2397) - Don't use deprecated distutils module.

In this release 13 pull requests were closed.


## Version 0.40.0 (2023/02/09)

### Issues Closed

* [Issue 2385](https://github.com/pytroll/satpy/issues/2385) - Tests needed for PSPRayleighReflectance
* [Issue 2381](https://github.com/pytroll/satpy/issues/2381) - Remote files reading example does not work. ([PR 2382](https://github.com/pytroll/satpy/pull/2382) by [@djhoese](https://github.com/djhoese))
* [Issue 2377](https://github.com/pytroll/satpy/issues/2377) - Bug remapping SEVIRI data - bounding box too small
* [Issue 2376](https://github.com/pytroll/satpy/issues/2376) - Fails displaying the NWCSAF/Geo cloud products correctly
* [Issue 2374](https://github.com/pytroll/satpy/issues/2374) - Question: How to understand "satpy.enhancements.ahi.jma_true_color_reproduction"?
* [Issue 2371](https://github.com/pytroll/satpy/issues/2371) - cloud_top_height colormap is broken ([PR 2373](https://github.com/pytroll/satpy/pull/2373) by [@gerritholl](https://github.com/gerritholl))
* [Issue 2368](https://github.com/pytroll/satpy/issues/2368) - Fails to resample RSS in native format
* [Issue 2366](https://github.com/pytroll/satpy/issues/2366) - Problem caching angles derivation: `TypeError: Object of type float32 is not JSON serializable` ([PR 2367](https://github.com/pytroll/satpy/pull/2367) by [@adybbroe](https://github.com/adybbroe))
* [Issue 2363](https://github.com/pytroll/satpy/issues/2363) - Runtime deprecation warning is thrown for compositor that is not used ([PR 2384](https://github.com/pytroll/satpy/pull/2384) by [@strandgren](https://github.com/strandgren))
* [Issue 2357](https://github.com/pytroll/satpy/issues/2357) - Alpha band improvement for DayNightCompositor ([PR 2358](https://github.com/pytroll/satpy/pull/2358) by [@yukaribbba](https://github.com/yukaribbba))
* [Issue 2349](https://github.com/pytroll/satpy/issues/2349) - A bug not detected by tests introduced when porting mitiff writer to PIL ([PR 2350](https://github.com/pytroll/satpy/pull/2350) by [@TAlonglong](https://github.com/TAlonglong))
* [Issue 2343](https://github.com/pytroll/satpy/issues/2343) - HTTPSConnectionPool（host='zenodo.org'， port=443）
* [Issue 2339](https://github.com/pytroll/satpy/issues/2339) - `numpy.bool` type deprecated, new release needed
* [Issue 2330](https://github.com/pytroll/satpy/issues/2330) - ancillary variables does not get resampled ([PR 2336](https://github.com/pytroll/satpy/pull/2336) by [@gerritholl](https://github.com/gerritholl))
* [Issue 2329](https://github.com/pytroll/satpy/issues/2329) - resampling replaces ancillary variable ([PR 2336](https://github.com/pytroll/satpy/pull/2336) by [@gerritholl](https://github.com/gerritholl))
* [Issue 2308](https://github.com/pytroll/satpy/issues/2308) - Unify colormap creations ([PR 2313](https://github.com/pytroll/satpy/pull/2313) by [@gerritholl](https://github.com/gerritholl))
* [Issue 2298](https://github.com/pytroll/satpy/issues/2298) - There is a bug in 'find_files_and_readers(reader='modis_l2)'' ([PR 2299](https://github.com/pytroll/satpy/pull/2299) by [@springfieldpsk](https://github.com/springfieldpsk))
* [Issue 2286](https://github.com/pytroll/satpy/issues/2286) - satpy_cf_nc reader to support variables with same name but different attributes ([PR 2290](https://github.com/pytroll/satpy/pull/2290) by [@TAlonglong](https://github.com/TAlonglong))
* [Issue 2095](https://github.com/pytroll/satpy/issues/2095) - Problem with merged composites for VIIRS SDR and NWCSAF ([PR 2101](https://github.com/pytroll/satpy/pull/2101) by [@ninahakansson](https://github.com/ninahakansson))
* [Issue 1844](https://github.com/pytroll/satpy/issues/1844) - Improve support for mode P images ([PR 2301](https://github.com/pytroll/satpy/pull/2301) by [@gerritholl](https://github.com/gerritholl))
* [Issue 690](https://github.com/pytroll/satpy/issues/690) - native python readers for GOES GVAR and McIDAS Area format

In this release 21 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 2382](https://github.com/pytroll/satpy/pull/2382) - Fix Scene storage options not being used if no other reader_kwargs are passed ([2381](https://github.com/pytroll/satpy/issues/2381))
* [PR 2373](https://github.com/pytroll/satpy/pull/2373) - Fix palette images for float data ([2371](https://github.com/pytroll/satpy/issues/2371))
* [PR 2367](https://github.com/pytroll/satpy/pull/2367) - Fix bug hashing the angles derivation using the NWCSAF reader ([2366](https://github.com/pytroll/satpy/issues/2366))
* [PR 2360](https://github.com/pytroll/satpy/pull/2360) - Fix "p_id" to be compatible with Himawari-9
* [PR 2350](https://github.com/pytroll/satpy/pull/2350) - fix write multiple channels mitiff writer ([2349](https://github.com/pytroll/satpy/issues/2349))
* [PR 2336](https://github.com/pytroll/satpy/pull/2336) - Fix ancillary variable confusion after resampling ([2330](https://github.com/pytroll/satpy/issues/2330), [2329](https://github.com/pytroll/satpy/issues/2329))
* [PR 2309](https://github.com/pytroll/satpy/pull/2309) - Fix amsub_l1c_aapp.yaml frequency handling pointers
* [PR 2299](https://github.com/pytroll/satpy/pull/2299) - Fix filter in modis_l2.yaml ([2298](https://github.com/pytroll/satpy/issues/2298))
* [PR 2290](https://github.com/pytroll/satpy/pull/2290) - satpy_cf_nc reader to support reading variables with the same name with different resolutions from two files ([2286](https://github.com/pytroll/satpy/issues/2286))
* [PR 2101](https://github.com/pytroll/satpy/pull/2101) - Drop nwcsaf y coord ([2095](https://github.com/pytroll/satpy/issues/2095))
* [PR 1719](https://github.com/pytroll/satpy/pull/1719) - Make WavelengthRange importable from satpy.dataset

#### Features added

* [PR 2384](https://github.com/pytroll/satpy/pull/2384) - Improve handing of deprecation warnings for deprecated YAML composite recipes ([2363](https://github.com/pytroll/satpy/issues/2363))
* [PR 2378](https://github.com/pytroll/satpy/pull/2378) - Adapt unit test between categorical and continuous
* [PR 2372](https://github.com/pytroll/satpy/pull/2372) - add flash_area composite for LI L2
* [PR 2358](https://github.com/pytroll/satpy/pull/2358) - Add an argument to the compositor to switch alpha band on/off in DayNightCompositor ([2357](https://github.com/pytroll/satpy/issues/2357))
* [PR 2355](https://github.com/pytroll/satpy/pull/2355) - Make the nwcsaf nc readers tests use real files
* [PR 2353](https://github.com/pytroll/satpy/pull/2353) - Add some project urls for pypi display
* [PR 2345](https://github.com/pytroll/satpy/pull/2345) - Add time coverage attrs for TROPOMI L2
* [PR 2341](https://github.com/pytroll/satpy/pull/2341) - Seviri l2 cf harmonization
* [PR 2337](https://github.com/pytroll/satpy/pull/2337) - Add ATMS SDR HDF5 format reader
* [PR 2327](https://github.com/pytroll/satpy/pull/2327) - Update himawari8/9 JMA True Color Reproduction composite
* [PR 2313](https://github.com/pytroll/satpy/pull/2313) - Refactor colormap creation ([2308](https://github.com/pytroll/satpy/issues/2308))
* [PR 2312](https://github.com/pytroll/satpy/pull/2312) - Add config option for temporary directory
* [PR 2305](https://github.com/pytroll/satpy/pull/2305) - Remote file reading for FCI L1c
* [PR 2304](https://github.com/pytroll/satpy/pull/2304) - Add additional OLCI L2 datasets.
* [PR 2301](https://github.com/pytroll/satpy/pull/2301) - Improve support for p-mode images ([1844](https://github.com/pytroll/satpy/issues/1844))
* [PR 2285](https://github.com/pytroll/satpy/pull/2285) - Add a reader for H-SAF h10 Snow Cover (SC) in hdf5
* [PR 2282](https://github.com/pytroll/satpy/pull/2282) - Add the 'Rocket Plume' RGB to satpy.
* [PR 2281](https://github.com/pytroll/satpy/pull/2281) - group save_datasets result by file
* [PR 2275](https://github.com/pytroll/satpy/pull/2275) - Multiscene blend with weights

#### Documentation changes

* [PR 2369](https://github.com/pytroll/satpy/pull/2369) - Add Getting Help section to documentation
* [PR 2356](https://github.com/pytroll/satpy/pull/2356) - Add tmp_path to the list of fixtures use in test_readers.py
* [PR 2348](https://github.com/pytroll/satpy/pull/2348) - Add an initial section on writing tests
* [PR 2342](https://github.com/pytroll/satpy/pull/2342) - Fix typos in Changelog Isse -> Issue

#### Clean ups

* [PR 2361](https://github.com/pytroll/satpy/pull/2361) - Fix Sar lon/lat interpolator for scipy deprecation
* [PR 2359](https://github.com/pytroll/satpy/pull/2359) - Speed up `find_files_and_readers`
* [PR 2354](https://github.com/pytroll/satpy/pull/2354) - Stop creating dummy files in cwd when testing
* [PR 2344](https://github.com/pytroll/satpy/pull/2344) - Remove unneeded code from olci reader

In this release 38 pull requests were closed.


## Version 0.39.0 (2022/12/30)

### Issues Closed

* [Issue 2326](https://github.com/pytroll/satpy/issues/2326) - satpy installation failed with pip
* [Issue 2325](https://github.com/pytroll/satpy/issues/2325) - VIIRS SDR DNB units and file_units are conflicting
* [Issue 2323](https://github.com/pytroll/satpy/issues/2323) - use area def raise value different when process himawari datasets
* [Issue 2311](https://github.com/pytroll/satpy/issues/2311) - scatsat1_l2b and caliop_l2_cloud readers fail to initialise due to failed `Dataset` import
* [Issue 2289](https://github.com/pytroll/satpy/issues/2289) - Resampling from MSG Satellite projection to Dutch weather radar projection, area_extent missing
* [Issue 2273](https://github.com/pytroll/satpy/issues/2273) - Add support for HRFI imagery in the FCI L1c reader ([PR2287](https://github.com/pytroll/satpy/pull/2287) by [@ameraner](https://github.com/ameraner))
* [Issue 2270](https://github.com/pytroll/satpy/issues/2270) - Missing SEVIRI service mode for new IODC sub-satellite longitude of MET09
* [Issue 2269](https://github.com/pytroll/satpy/issues/2269) - Occasional errors when closing FCI L2 NetCDF file
* [Issue 2264](https://github.com/pytroll/satpy/issues/2264) - Can't load static image composite multiple times ([PR2292](https://github.com/pytroll/satpy/pull/2292) by [@djhoese](https://github.com/djhoese))
* [Issue 2248](https://github.com/pytroll/satpy/issues/2248) - Test_scene failure on debian sid
* [Issue 2245](https://github.com/pytroll/satpy/issues/2245) - `native` resampler fails for some chunk sizes ([PR2291](https://github.com/pytroll/satpy/pull/2291) by [@djhoese](https://github.com/djhoese))
* [Issue 2213](https://github.com/pytroll/satpy/issues/2213) - Update `li_l2`-reader to read MTG LI L2 test data ([PR2271](https://github.com/pytroll/satpy/pull/2271) by [@seenno](https://github.com/seenno))
* [Issue 2186](https://github.com/pytroll/satpy/issues/2186) - FCI L1c reader is single threaded ([PR2237](https://github.com/pytroll/satpy/pull/2237) by [@ameraner](https://github.com/ameraner))
* [Issue 1595](https://github.com/pytroll/satpy/issues/1595) - Native resampler to coarsest_area fails for HRV channel due to indivisible chunk sizes
* [Issue 1482](https://github.com/pytroll/satpy/issues/1482) - Add option to turn off masking in seviri_l1b_hrit ([PR1693](https://github.com/pytroll/satpy/pull/1693) by [@BENR0](https://github.com/BENR0))
* [Issue 1381](https://github.com/pytroll/satpy/issues/1381) - fails to load all composites for abi_l1b
* [Issue 658](https://github.com/pytroll/satpy/issues/658) - MTG LI reader is not Python 3 compatible
* [Issue 650](https://github.com/pytroll/satpy/issues/650) - MTG LI reader fails
* [Issue 499](https://github.com/pytroll/satpy/issues/499) - Graceful handling of saturated values in MODIS data
* [Issue 370](https://github.com/pytroll/satpy/issues/370) - Make hdfeos_l1b geo interpolation use dask arrays

In this release 20 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR2335](https://github.com/pytroll/satpy/pull/2335) - Fix Scene modifying user-provided reader_kwargs
* [PR 2318](https://github.com/pytroll/satpy/pull/2318) - Remove pylibtiff dependencies
* [PR 2317](https://github.com/pytroll/satpy/pull/2317) - Fix for bug regarding masking of good scan lines with seviri hrit reader introduced by #1693
* [PR 2297](https://github.com/pytroll/satpy/pull/2297) - Replace np.bool by np.bool_
* [PR 2292](https://github.com/pytroll/satpy/pull/2292) - Fix Scene not handling DataArrays with 'sensor' set to None ([2264](https://github.com/pytroll/satpy/issues/2264))
* [PR 2291](https://github.com/pytroll/satpy/pull/2291) - Fix native resampler not working for some chunk sizes ([2245](https://github.com/pytroll/satpy/issues/2245))
* [PR 2276](https://github.com/pytroll/satpy/pull/2276) - Update AGRI for fill values and cailbration LUTs.

#### Features added

* [PR 2315](https://github.com/pytroll/satpy/pull/2315) - Port mitiff writer to use pillow
* [PR 2287](https://github.com/pytroll/satpy/pull/2287) - Feature: add support for HRFI imagery in the FCI L1c reader ([2273](https://github.com/pytroll/satpy/issues/2273))
* [PR 2280](https://github.com/pytroll/satpy/pull/2280) - Add NDVI-scaled hybrid green correction
* [PR 2271](https://github.com/pytroll/satpy/pull/2271) - Add LI L2 reader ([2213](https://github.com/pytroll/satpy/issues/2213))
* [PR 2265](https://github.com/pytroll/satpy/pull/2265) - Add a reader for insat 3d imager data (`insat3d_img_l1b_h5`)
* [PR 2237](https://github.com/pytroll/satpy/pull/2237) - Optimise the FCI L1c/netcdf_utils by introducing on-demand variables collection and caching ([2186](https://github.com/pytroll/satpy/issues/2186))
* [PR 1693](https://github.com/pytroll/satpy/pull/1693) - Add masking of data with line quality flags to seviri nc reader ([1482](https://github.com/pytroll/satpy/issues/1482))
* [PR 1678](https://github.com/pytroll/satpy/pull/1678) - Add a reader for Meris L2 data

#### Documentation changes

* [PR 2306](https://github.com/pytroll/satpy/pull/2306) - fix description for 500m resolution
* [PR 2295](https://github.com/pytroll/satpy/pull/2295) - Fix documentation reference links to EUMeTrain webpages
* [PR 2274](https://github.com/pytroll/satpy/pull/2274) - Documentation correction: Not all built-in compositors derive from GenericCompositor

In this release 18 pull requests were closed.


## Version 0.38.0 (2022/11/11)

### Issues Closed

* [Issue 2258](https://github.com/pytroll/satpy/issues/2258) - VIIRS day_microphysics array broadcast error ([PR 2260](https://github.com/pytroll/satpy/pull/2260) by [@djhoese](https://github.com/djhoese))
* [Issue 2250](https://github.com/pytroll/satpy/issues/2250) - Is reader modis_l1b available?
* [Issue 2249](https://github.com/pytroll/satpy/issues/2249) - ahi-hsd raise keyerror when load B08 in dataquery
* [Issue 2241](https://github.com/pytroll/satpy/issues/2241) - Slow loading speed of TROPOMI L2 product
* [Issue 2234](https://github.com/pytroll/satpy/issues/2234) - Scene `modifiers` keyword argument has no effect ([PR 2235](https://github.com/pytroll/satpy/pull/2235) by [@djhoese](https://github.com/djhoese))
* [Issue 2233](https://github.com/pytroll/satpy/issues/2233) - 'cached_property' from 'functools' - seems incompatible with python3.7
* [Issue 2228](https://github.com/pytroll/satpy/issues/2228) - Question: Why the ellipsoid of a geostationary satellite image is not typical WGS84?
* [Issue 2227](https://github.com/pytroll/satpy/issues/2227) - CF writer output wrong for area with geographic CRS ([PR 2236](https://github.com/pytroll/satpy/pull/2236) by [@gerritholl](https://github.com/gerritholl))
* [Issue 2215](https://github.com/pytroll/satpy/issues/2215) - Abi l2 nc reader can't handle AOD product ([PR 2216](https://github.com/pytroll/satpy/pull/2216) by [@mraspaud](https://github.com/mraspaud))
* [Issue 2208](https://github.com/pytroll/satpy/issues/2208) - hy2_scat_l2b_h5 reader does not work any more due to space in `valid range` attribute ([PR 2268](https://github.com/pytroll/satpy/pull/2268) by [@TAlonglong](https://github.com/TAlonglong))
* [Issue 2206](https://github.com/pytroll/satpy/issues/2206) - If you do `python -c "import scipy.sparse"` do you get a similar error?
* [Issue 2202](https://github.com/pytroll/satpy/issues/2202) - AttributeError: 'AreaDefinition' object has no attribute 'crs'
* [Issue 2192](https://github.com/pytroll/satpy/issues/2192) - Available Readers Problem
* [Issue 2189](https://github.com/pytroll/satpy/issues/2189) - Supported readers page does not include many readers ([PR 2191](https://github.com/pytroll/satpy/pull/2191) by [@BENR0](https://github.com/BENR0))
* [Issue 2183](https://github.com/pytroll/satpy/issues/2183) - Reading bzipped Seviri HRIT segment crashes when reading the data from disk ([PR 2185](https://github.com/pytroll/satpy/pull/2185) by [@mraspaud](https://github.com/mraspaud))
* [Issue 2170](https://github.com/pytroll/satpy/issues/2170) - satpy_cf_nc Reader Fails to Read Data Written by cf Writer ([PR 2176](https://github.com/pytroll/satpy/pull/2176) by [@mraspaud](https://github.com/mraspaud))
* [Issue 2154](https://github.com/pytroll/satpy/issues/2154) - module 'ntpath' has no attribute 'sep'"
* [Issue 2111](https://github.com/pytroll/satpy/issues/2111) - Archived GOES datasets not loading. AttributeError: 'area' object has no attribute 'crs'
* [Issue 1929](https://github.com/pytroll/satpy/issues/1929) - Two test failures in test_goes_imager_nc.py with Python 3.10
* [Issue 1672](https://github.com/pytroll/satpy/issues/1672) - Add AreaDefinition support to the 'satpy_cf_nc' reader ([PR 1695](https://github.com/pytroll/satpy/pull/1695) by [@BENR0](https://github.com/BENR0))

In this release 20 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 2262](https://github.com/pytroll/satpy/pull/2262) - Fix ratio sharpening not sharing invalid mask between bands ([556](https://github.com/ssec/polar2grid/issues/556))
* [PR 2260](https://github.com/pytroll/satpy/pull/2260) - Fix VIIRS L1B I-band angle names being inconsistent with VIIRS SDR ([2258](https://github.com/pytroll/satpy/issues/2258))
* [PR 2257](https://github.com/pytroll/satpy/pull/2257) - Fix failure creating directory if it already exists.
* [PR 2246](https://github.com/pytroll/satpy/pull/2246) - Fix MODIS reader tests failing with new geotiepoints
* [PR 2239](https://github.com/pytroll/satpy/pull/2239) - Fix incorrect rows_per_scan in 'acspo' reader ([498](https://github.com/ssec/polar2grid/issues/498))
* [PR 2236](https://github.com/pytroll/satpy/pull/2236) - CF-compliant storage for lon/lat case ([2227](https://github.com/pytroll/satpy/issues/2227))
* [PR 2235](https://github.com/pytroll/satpy/pull/2235) - Fix Scene.load modifiers keyword argument having no effect ([2234](https://github.com/pytroll/satpy/issues/2234))
* [PR 2232](https://github.com/pytroll/satpy/pull/2232) - Make longitude masker & filler single band
* [PR 2221](https://github.com/pytroll/satpy/pull/2221) - Fix ici after mws merge
* [PR 2220](https://github.com/pytroll/satpy/pull/2220) - Fix CLAVR-x configuration in 'awips_tiled' writer to be backwards compatible
* [PR 2216](https://github.com/pytroll/satpy/pull/2216) - Fix coord renaming for AOD product in 'abi_l2_nc' reader ([2215](https://github.com/pytroll/satpy/issues/2215))
* [PR 2210](https://github.com/pytroll/satpy/pull/2210) - Fix VIIRS EDR Active Fires reader for new format and fix fine/coarse 1D swath handling ([458](https://github.com/ssec/polar2grid/issues/458))
* [PR 2190](https://github.com/pytroll/satpy/pull/2190) - Fix some enhancements producing dask arrays wrapped in dask arrays
* [PR 2185](https://github.com/pytroll/satpy/pull/2185) - Refactor HRIT readers to be smarter about compression and reading data ([2183](https://github.com/pytroll/satpy/issues/2183))
* [PR 2177](https://github.com/pytroll/satpy/pull/2177) - Fix plugins not working with new versions of setuptools
* [PR 2176](https://github.com/pytroll/satpy/pull/2176) - Fix cf write-read roundtrip ([2170](https://github.com/pytroll/satpy/issues/2170))
* [PR 2166](https://github.com/pytroll/satpy/pull/2166) - Correct the sun azimuth angle range within satpy.

#### Features added

* [PR 2230](https://github.com/pytroll/satpy/pull/2230) - Add support for compressed FSFiles to HRIT readers
* [PR 2209](https://github.com/pytroll/satpy/pull/2209) - Update seadas_l2 reader to handle alternative NetCDF file format ([457](https://github.com/ssec/polar2grid/issues/457))
* [PR 2207](https://github.com/pytroll/satpy/pull/2207) - Add SEVIRI level 2 AMV BUFR
* [PR 2203](https://github.com/pytroll/satpy/pull/2203) - Fix experimental dependency stdlibc++ issues in CI
* [PR 2198](https://github.com/pytroll/satpy/pull/2198) - Add warning for SEVIRI native reader in case of bad data
* [PR 2187](https://github.com/pytroll/satpy/pull/2187) - adding a reader for ATMS level1b data
* [PR 2185](https://github.com/pytroll/satpy/pull/2185) - Refactor HRIT readers to be smarter about compression and reading data ([2183](https://github.com/pytroll/satpy/issues/2183))
* [PR 2175](https://github.com/pytroll/satpy/pull/2175) - Add utility function to compute the relative azimuth angle.
* [PR 2164](https://github.com/pytroll/satpy/pull/2164) - Add low level moisture composite
* [PR 2125](https://github.com/pytroll/satpy/pull/2125) - Add reader for FY-4B / GHI data
* [PR 2120](https://github.com/pytroll/satpy/pull/2120) - Add reader for MWS onboard EPS-SG-A
* [PR 2118](https://github.com/pytroll/satpy/pull/2118) - Add a reader for EPS-SG Ice Cloud Imager
* [PR 1695](https://github.com/pytroll/satpy/pull/1695) - Add `get_area_def` to cf reader ([1672](https://github.com/pytroll/satpy/issues/1672))

#### Documentation changes

* [PR 2247](https://github.com/pytroll/satpy/pull/2247) - Document behaviour on default enhancement
* [PR 2225](https://github.com/pytroll/satpy/pull/2225) - Update writer table docs
* [PR 2200](https://github.com/pytroll/satpy/pull/2200) - Remove mention of fallback to GDAL in geotiff writer
* [PR 2195](https://github.com/pytroll/satpy/pull/2195) - Add additional logging information about enhancements being used
* [PR 2191](https://github.com/pytroll/satpy/pull/2191) - Fix automatic reader table not listing readers with missing dependencies ([2189](https://github.com/pytroll/satpy/issues/2189))

#### Clean ups

* [PR 2268](https://github.com/pytroll/satpy/pull/2268) - Cleanup hy2 reader ([2208](https://github.com/pytroll/satpy/issues/2208))
* [PR 2252](https://github.com/pytroll/satpy/pull/2252) - Create dependabot.yml
* [PR 2240](https://github.com/pytroll/satpy/pull/2240) - Refactor RGB ratio sharpening again for better performance
* [PR 2205](https://github.com/pytroll/satpy/pull/2205) - Update URL to rasterio repository in CI

In this release 39 pull requests were closed.


## Version 0.37.1 (2022/08/15)

### Issues Closed

* [Issue 2173](https://github.com/pytroll/satpy/issues/2173) - MetopC script fails after update to SatPy 0.37 ([PR 2174](https://github.com/pytroll/satpy/pull/2174) by [@mraspaud](https://github.com/mraspaud))

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 2174](https://github.com/pytroll/satpy/pull/2174) - Fix 3d effect enhancement ([2173](https://github.com/pytroll/satpy/issues/2173))

In this release 1 pull request was closed.


## Version 0.37.0 (2022/08/05)

### Issues Closed

* [Issue 2163](https://github.com/pytroll/satpy/issues/2163) - ValueError: time data '2022-07-11T00:30:01Z' does not match format '%Y-%m-%dT%H:%M:%S.%fZ' ([PR 2165](https://github.com/pytroll/satpy/pull/2165) by [@simonrp84](https://github.com/simonrp84))
* [Issue 2161](https://github.com/pytroll/satpy/issues/2161) - Plotting Scene with Cartopy gives correct borders but red background
* [Issue 2155](https://github.com/pytroll/satpy/issues/2155) - AVHRR CLASS Filename prefix prevents reading with Scene. ([PR 2157](https://github.com/pytroll/satpy/pull/2157) by [@djhoese](https://github.com/djhoese))
* [Issue 2145](https://github.com/pytroll/satpy/issues/2145) - Could not calculate destination definition resolution
* [Issue 2143](https://github.com/pytroll/satpy/issues/2143) - Save geotiff with keep_palette still add an alpha band
* [Issue 2139](https://github.com/pytroll/satpy/issues/2139) - Hostname neo.sci.gsfc.nasa.gov not reachable anymore
* [Issue 2135](https://github.com/pytroll/satpy/issues/2135) - Images are slightly different between scn.crop() and original one
* [Issue 2134](https://github.com/pytroll/satpy/issues/2134) - scene.coarsest_area and scene.resample not working on MSG1/MSG2 in satpy 0.29
* [Issue 2130](https://github.com/pytroll/satpy/issues/2130) - Can't install satpy with pip
* [Issue 2127](https://github.com/pytroll/satpy/issues/2127) - Raised RuntimeError when trying to make HIMAWARI-8 true color image ([PR 2128](https://github.com/pytroll/satpy/pull/2128) by [@mherbertson](https://github.com/mherbertson))
* [Issue 2112](https://github.com/pytroll/satpy/issues/2112) - Trying to run scn.load(['true_color']) on GOES-16 ABI_L1b causes ValueError
* [Issue 2093](https://github.com/pytroll/satpy/issues/2093) - Multiscene blend does not work well.
* [Issue 2089](https://github.com/pytroll/satpy/issues/2089) - MultiScene.group doesn't work for differing identifier properties ([PR 2099](https://github.com/pytroll/satpy/pull/2099) by [@sfinkens](https://github.com/sfinkens))
* [Issue 1996](https://github.com/pytroll/satpy/issues/1996) - FCI dataset attributes are missing orbital parameters ([PR 2110](https://github.com/pytroll/satpy/pull/2110) by [@ameraner](https://github.com/ameraner))
* [Issue 1949](https://github.com/pytroll/satpy/issues/1949) - keep `delta_time` instead of renaming to `offset_time`
* [Issue 1865](https://github.com/pytroll/satpy/issues/1865) - navigation available for all L1b readers
* [Issue 1845](https://github.com/pytroll/satpy/issues/1845) - Add parallax correction ([PR 1904](https://github.com/pytroll/satpy/pull/1904) by [@gerritholl](https://github.com/gerritholl))
* [Issue 1699](https://github.com/pytroll/satpy/issues/1699) - Remove `compression` argument from CF writer `save_datasets`
* [Issue 1638](https://github.com/pytroll/satpy/issues/1638) - satpy_cf_nc reader not working with files written from seviri_l1b readers
* [Issue 1348](https://github.com/pytroll/satpy/issues/1348) - 'AHIHSDFileHandler' object has no attribute 'area'
* [Issue 1308](https://github.com/pytroll/satpy/issues/1308) - Error with yaml files during tutorial need example file I think
* [Issue 1015](https://github.com/pytroll/satpy/issues/1015) - Add compute method to Scene ([PR 1017](https://github.com/pytroll/satpy/pull/1017) by [@BENR0](https://github.com/BENR0))

In this release 22 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 2165](https://github.com/pytroll/satpy/pull/2165) - Update AGRI reader to deal with invalid `valid_range` HDF attribute ([2163](https://github.com/pytroll/satpy/issues/2163))
* [PR 2151](https://github.com/pytroll/satpy/pull/2151) - Change default type of AWIPS tiled writer to int16 + _Unsigned
* [PR 2150](https://github.com/pytroll/satpy/pull/2150) - Fix typos in reader table
* [PR 2142](https://github.com/pytroll/satpy/pull/2142) - Fix MODIS readers applying add_offset incorrectly
* [PR 2141](https://github.com/pytroll/satpy/pull/2141) - Fix expected test results for pyorbital 1.7.2
* [PR 2137](https://github.com/pytroll/satpy/pull/2137) - Fix GPM IMERG reader.
* [PR 2128](https://github.com/pytroll/satpy/pull/2128) - Fix AHI source file segment order after decompressing. ([2127](https://github.com/pytroll/satpy/issues/2127))
* [PR 2123](https://github.com/pytroll/satpy/pull/2123) - Fix negative channel 3A slope calibration coefficient in AVHRR reader
* [PR 2122](https://github.com/pytroll/satpy/pull/2122) - Fix yaml files for some seviri/abi/ahi BlackMarble background composites
* [PR 2115](https://github.com/pytroll/satpy/pull/2115) - Update 'viirs_edr_active_fires' to work with newer Active Fires output
* [PR 2114](https://github.com/pytroll/satpy/pull/2114) - Fix ACSPO 'sensor' attribute not being lowercase
* [PR 2107](https://github.com/pytroll/satpy/pull/2107) - Fix 'ahi_hsd' reader crashing when 'observation_timeline' was invalid
* [PR 2103](https://github.com/pytroll/satpy/pull/2103) - Update SEVIRI ICARE reader to properly use dask.
* [PR 2100](https://github.com/pytroll/satpy/pull/2100) - Fix handling of non-existent reflectance bands in 'viirs_l1b' reader
* [PR 2099](https://github.com/pytroll/satpy/pull/2099) - Fix MultiScene.group in case of differing identifier properties ([2089](https://github.com/pytroll/satpy/issues/2089))
* [PR 2098](https://github.com/pytroll/satpy/pull/2098) - Fix Scene.coarsest/finest_area not returning consistent results
* [PR 1877](https://github.com/pytroll/satpy/pull/1877) - Update SEVIRI native reader with 'time_parameters' metadata

#### Features added

* [PR 2160](https://github.com/pytroll/satpy/pull/2160) - Add reader for ESA's Ocean Color CCI data
* [PR 2157](https://github.com/pytroll/satpy/pull/2157) - Add filename pattern for CLASS subscription files ([2155](https://github.com/pytroll/satpy/issues/2155))
* [PR 2156](https://github.com/pytroll/satpy/pull/2156) - Added filename pattern for CLASS subscription files.
* [PR 2147](https://github.com/pytroll/satpy/pull/2147) - added the CRR-Ph files to the nwcsaf geo yaml file
* [PR 2146](https://github.com/pytroll/satpy/pull/2146) - Update Metimage reader for L2 test data
* [PR 2140](https://github.com/pytroll/satpy/pull/2140) - Add Natural Color / IR composites
* [PR 2133](https://github.com/pytroll/satpy/pull/2133) - Rewrite 'apply_enhancement' as individual decorators to allow for easier dask map_blocks usage
* [PR 2110](https://github.com/pytroll/satpy/pull/2110) - Add orbital_parameters to fci_l1c_nc reader ([1996](https://github.com/pytroll/satpy/issues/1996))
* [PR 2105](https://github.com/pytroll/satpy/pull/2105) - Enable solar zenith angle caching for the DayNightCompositor
* [PR 2102](https://github.com/pytroll/satpy/pull/2102) - Add more products to fci_l2_nc reader
* [PR 2097](https://github.com/pytroll/satpy/pull/2097) - [pre-commit.ci] pre-commit autoupdate
* [PR 2096](https://github.com/pytroll/satpy/pull/2096) - Convert remote files to FSFile objects automatically
* [PR 1919](https://github.com/pytroll/satpy/pull/1919) - Implement adaptive FCI chunks padding and create a new GEOVariableSegmentYAMLReader class
* [PR 1904](https://github.com/pytroll/satpy/pull/1904) - Add parallax correction via new `ParallaxCorrectionModifier` ([1845](https://github.com/pytroll/satpy/issues/1845), []([bug](https://github.com/pytroll/satpy/pull/1904/issues/))
* [PR 1769](https://github.com/pytroll/satpy/pull/1769) - Add new composite (true_color_with_night_fires) to GOES/ABI: True color (day) with fires (night)
* [PR 1547](https://github.com/pytroll/satpy/pull/1547) - Add support for  fsspec files to seviri_l1b_nc reader
* [PR 1017](https://github.com/pytroll/satpy/pull/1017) - Add pass through of xr compute, persist and chunk to Scene ([1015](https://github.com/pytroll/satpy/issues/1015))

#### Documentation changes

* [PR 2153](https://github.com/pytroll/satpy/pull/2153) - Document alternative for deprecated get_min/max_area
* [PR 2138](https://github.com/pytroll/satpy/pull/2138) - Add plugin functionality for readers, writers, and enhancements
* [PR 2108](https://github.com/pytroll/satpy/pull/2108) - Functions to automatically generate reader table for documentation
* [PR 2104](https://github.com/pytroll/satpy/pull/2104) - Improvements in custom reader documentation
* [PR 2091](https://github.com/pytroll/satpy/pull/2091) - Fix link to rad2refl document in SEVIRI base reader
* [PR 1886](https://github.com/pytroll/satpy/pull/1886) - Update quickstart documentation so that HRV channel is not loaded

In this release 40 pull requests were closed.


## Version 0.36.0 (2022/04/14)

### Issues Closed

* [Issue 2082](https://github.com/pytroll/satpy/issues/2082) - Some composite are produced with the wrong colors
* [Issue 2073](https://github.com/pytroll/satpy/issues/2073) - Creating scene with SEVIRI HRIT reader fails with UnicodeDecodeError ([PR 2077](https://github.com/pytroll/satpy/pull/2077) by [@pdebuyl](https://github.com/pdebuyl))
* [Issue 2066](https://github.com/pytroll/satpy/issues/2066) - RGBs should never have units, but some do ([PR 2068](https://github.com/pytroll/satpy/pull/2068) by [@gerritholl](https://github.com/gerritholl))
* [Issue 2062](https://github.com/pytroll/satpy/issues/2062) - Can make Trollimage colorbar in scene
* [Issue 1975](https://github.com/pytroll/satpy/issues/1975) - bunzip regular seviri hrit segments ([PR 2060](https://github.com/pytroll/satpy/pull/2060) by [@pdebuyl](https://github.com/pdebuyl))
* [Issue 1954](https://github.com/pytroll/satpy/issues/1954) - Cloud Phase/Type/Phase Distinction RGBs for VIIRS and FCI ([PR 1957](https://github.com/pytroll/satpy/pull/1957) by [@gerritholl](https://github.com/gerritholl))
* [Issue 1702](https://github.com/pytroll/satpy/issues/1702) - Resampling not working with SLSTR ancillary datasets

In this release 7 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 2084](https://github.com/pytroll/satpy/pull/2084) - Fix CREFL using incorrect coefficients for MODIS
* [PR 2083](https://github.com/pytroll/satpy/pull/2083) - Fix VIIRS L1B reader sensor not matching VIIRS SDR reader
* [PR 2080](https://github.com/pytroll/satpy/pull/2080) - Ignore alpha when adding luminance in Sandwich compositor
* [PR 2079](https://github.com/pytroll/satpy/pull/2079) - Remove marine_clean_aerosol from default AHI rayleigh_corrected modifier
* [PR 2077](https://github.com/pytroll/satpy/pull/2077) - Fix missing 'rb' mode for opening files ([2073](https://github.com/pytroll/satpy/issues/2073))
* [PR 2070](https://github.com/pytroll/satpy/pull/2070) - Fix lru_cache memory leaks and other linting errors
* [PR 2048](https://github.com/pytroll/satpy/pull/2048) - Fix CMIC CRE product in nwcsaf yaml reader
* [PR 2016](https://github.com/pytroll/satpy/pull/2016) - Fix the sensor name for msu_gsa_l1b reader
* [PR 1410](https://github.com/pytroll/satpy/pull/1410) - Fix osisaf SST reader

#### Features added

* [PR 2086](https://github.com/pytroll/satpy/pull/2086) - Update FCI reader for new test data release and add patches for IDPF data
* [PR 2078](https://github.com/pytroll/satpy/pull/2078) - Add possibility to define the dataset rectification longitude in seviri_l2_bufr reader
* [PR 2076](https://github.com/pytroll/satpy/pull/2076) - Support reading FSFiles in SEVIRI HRIT reader.
* [PR 2068](https://github.com/pytroll/satpy/pull/2068) - Make sure RGBs do not have units attributes. ([2066](https://github.com/pytroll/satpy/issues/2066))
* [PR 2065](https://github.com/pytroll/satpy/pull/2065) - Add filename to YAML for NASA NRT VIIRS files with creation date/time.
* [PR 2060](https://github.com/pytroll/satpy/pull/2060) - Allow reading Bz2 hrit segments ([1975](https://github.com/pytroll/satpy/issues/1975))
* [PR 2057](https://github.com/pytroll/satpy/pull/2057) - Add option to replace saturated MODIS L1b values with max valid value
* [PR 1980](https://github.com/pytroll/satpy/pull/1980) - Adapt AAPP reader for generic chunk size
* [PR 1957](https://github.com/pytroll/satpy/pull/1957) - Add RGBs for cloud phase (distinction) and type ([1954](https://github.com/pytroll/satpy/issues/1954))
* [PR 1410](https://github.com/pytroll/satpy/pull/1410) - Fix osisaf SST reader

#### Documentation changes

* [PR 2075](https://github.com/pytroll/satpy/pull/2075) - Add documentation on how the colorize enhancement can be used
* [PR 2071](https://github.com/pytroll/satpy/pull/2071) - Add example to the documentation using multiple readers

#### Refactoring

* [PR 2087](https://github.com/pytroll/satpy/pull/2087) - Refactor HRIT/LRIT format reader.

In this release 22 pull requests were closed.


## Version 0.35.0 (2022/03/16)

### Issues Closed

* [Issue 2063](https://github.com/pytroll/satpy/issues/2063) - Unable to commit changes due to bandit (use of subprocess module)
* [Issue 2037](https://github.com/pytroll/satpy/issues/2037) - Why the lon/lat is interpolated to 1km while data are still 5km for MOD06 product
* [Issue 2012](https://github.com/pytroll/satpy/issues/2012) - Define time metadata options and usage ([PR 2031](https://github.com/pytroll/satpy/pull/2031) by [@djhoese](https://github.com/djhoese))
* [Issue 1973](https://github.com/pytroll/satpy/issues/1973) - Using cached geolocation and angles results in an error if chunk size not appropriately set. ([PR 2041](https://github.com/pytroll/satpy/pull/2041) by [@djhoese](https://github.com/djhoese))
* [Issue 1842](https://github.com/pytroll/satpy/issues/1842) - Update needed for vii_l1b_nc reader to match a change to the Test Data and processor ([PR 1979](https://github.com/pytroll/satpy/pull/1979) by [@pepephillips](https://github.com/pepephillips))
* [Issue 1110](https://github.com/pytroll/satpy/issues/1110) - NWCSAF reader does not support GOES or HIMAWARI
* [Issue 1022](https://github.com/pytroll/satpy/issues/1022) - Factorize area def computation in goes_imager_hrit ([PR 1934](https://github.com/pytroll/satpy/pull/1934) by [@sfinkens](https://github.com/sfinkens))
* [Issue 956](https://github.com/pytroll/satpy/issues/956) - UnboundLocalError when passing "empty" generator as filenames
* [Issue 723](https://github.com/pytroll/satpy/issues/723) - Passing multiple readers fails if `generic_image` is among them
* [Issue 684](https://github.com/pytroll/satpy/issues/684) - Gracefully handle pykdtree's use of OpenMP (OMP_NUM_THREADS) with dask

In this release 10 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 2054](https://github.com/pytroll/satpy/pull/2054) - Fix DifferenceCompositor not using metadata from YAML
* [PR 2049](https://github.com/pytroll/satpy/pull/2049) - Fix dataset attribute typo and reduce amount of categorical dataset filtering in fci_l2_nc reader
* [PR 2044](https://github.com/pytroll/satpy/pull/2044) - Fix unit handling in ERF DNB normalization's saturation correction
* [PR 2041](https://github.com/pytroll/satpy/pull/2041) - Fix angle generation caching not working with irregular chunks ([1973](https://github.com/pytroll/satpy/issues/1973))
* [PR 2032](https://github.com/pytroll/satpy/pull/2032) - Fix various metadata bugs in 'awips_tiled' writer ([417](https://github.com/ssec/polar2grid/issues/417))
* [PR 1933](https://github.com/pytroll/satpy/pull/1933) - Change tested Python versions to 3.8, 3.9 and 3.10

#### Features added

* [PR 2056](https://github.com/pytroll/satpy/pull/2056) - Update SLSTR calibration coefficients
* [PR 2055](https://github.com/pytroll/satpy/pull/2055) - Skip dataset flipping in GEOFlippableFileYAMLReader in case of SwathDefinition data
* [PR 2047](https://github.com/pytroll/satpy/pull/2047) - Add missing GOES-18 support to glm_l2 reader
* [PR 2034](https://github.com/pytroll/satpy/pull/2034) - Update angle generation to prefer "actual" satellite position
* [PR 2033](https://github.com/pytroll/satpy/pull/2033) - Remove use of legacy satellite position attributes
* [PR 2031](https://github.com/pytroll/satpy/pull/2031) - Update AHI HSD reader with observation/scheduled times and nominal satellite position ([2012](https://github.com/pytroll/satpy/issues/2012))
* [PR 2030](https://github.com/pytroll/satpy/pull/2030) - Add 'preference' option to 'get_satpos' utility
* [PR 2028](https://github.com/pytroll/satpy/pull/2028) - Add 'colormap_tag' keyword argument to geotiff writer
* [PR 1993](https://github.com/pytroll/satpy/pull/1993) - Add 'l2_flags' quality filtering to 'seadas_l2' reader
* [PR 1979](https://github.com/pytroll/satpy/pull/1979) - Update VII reader for test data v2 ([1842](https://github.com/pytroll/satpy/issues/1842))
* [PR 1933](https://github.com/pytroll/satpy/pull/1933) - Change tested Python versions to 3.8, 3.9 and 3.10
* [PR 1927](https://github.com/pytroll/satpy/pull/1927) - Add support for more FCI L2  products and datasets

#### Refactoring

* [PR 2040](https://github.com/pytroll/satpy/pull/2040) - Refactor composite generation to avoid unneeded warnings
* [PR 1934](https://github.com/pytroll/satpy/pull/1934) - Factorize area computation in goes_imager_hrit ([1022](https://github.com/pytroll/satpy/issues/1022))

In this release 20 pull requests were closed.


## Version 0.34.0 (2022/02/18)

### Issues Closed

* [Issue 2026](https://github.com/pytroll/satpy/issues/2026) - Missing units in avhrr_l1b_eps reader ([PR 2027](https://github.com/pytroll/satpy/pull/2027) by [@gerritholl](https://github.com/gerritholl))
* [Issue 2024](https://github.com/pytroll/satpy/issues/2024) - Allow to skip unit conversion in ninjotiff writer ([PR 2025](https://github.com/pytroll/satpy/pull/2025) by [@gerritholl](https://github.com/gerritholl))
* [Issue 2023](https://github.com/pytroll/satpy/issues/2023) - Allow to keep units in composite
* [Issue 2022](https://github.com/pytroll/satpy/issues/2022) - save_dataset changes dataset in-place
* [Issue 2018](https://github.com/pytroll/satpy/issues/2018) - Wrong AxisIntercept (add_offset) when writing °C temperature units with ninjogeotiff writer
* [Issue 2014](https://github.com/pytroll/satpy/issues/2014) - Problem in converting VIIRS hdf to geotif
* [Issue 2010](https://github.com/pytroll/satpy/issues/2010) - AHI HSD true_color incorrect with cache_sensor_angles  ([PR 2013](https://github.com/pytroll/satpy/pull/2013) by [@djhoese](https://github.com/djhoese))
* [Issue 2008](https://github.com/pytroll/satpy/issues/2008) - abi_l1b reader leaks memory in Python-3.7 ([PR 2011](https://github.com/pytroll/satpy/pull/2011) by [@sfinkens](https://github.com/sfinkens))
* [Issue 2004](https://github.com/pytroll/satpy/issues/2004) - Configure image type returned by MaskingCompositor ([PR 2005](https://github.com/pytroll/satpy/pull/2005) by [@gerritholl](https://github.com/gerritholl))
* [Issue 2001](https://github.com/pytroll/satpy/issues/2001) - Failed to load AVHRR LAC data
* [Issue 1999](https://github.com/pytroll/satpy/issues/1999) - Reader for Арктика-М (Arktika-M) МСУ-ГС (MSU-GS) data ([PR 2000](https://github.com/pytroll/satpy/pull/2000) by [@simonrp84](https://github.com/simonrp84))
* [Issue 1998](https://github.com/pytroll/satpy/issues/1998) - Add reader for Arctica M N-1 hdf5 data
* [Issue 1995](https://github.com/pytroll/satpy/issues/1995) - AttributeError when cropping data for VIIRS
* [Issue 1959](https://github.com/pytroll/satpy/issues/1959) - Unittest failure in test_modifiers.py
* [Issue 1948](https://github.com/pytroll/satpy/issues/1948) - Contribute to Satpy
* [Issue 1945](https://github.com/pytroll/satpy/issues/1945) - Wrong dtype of `uint32` array saved by the cf_writer
* [Issue 1943](https://github.com/pytroll/satpy/issues/1943) - sza_check from trollflow2 fails with KeyError: 'start_time'
* [Issue 1883](https://github.com/pytroll/satpy/issues/1883) - Test failure on i386 and armhf ([PR 1966](https://github.com/pytroll/satpy/pull/1966) by [@djhoese](https://github.com/djhoese))
* [Issue 1384](https://github.com/pytroll/satpy/issues/1384) - AHI HRIT reader has gotten slower ([PR 1986](https://github.com/pytroll/satpy/pull/1986) by [@pnuu](https://github.com/pnuu))
* [Issue 1099](https://github.com/pytroll/satpy/issues/1099) - `find_files_and_readers` read unneeded files

In this release 20 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 2027](https://github.com/pytroll/satpy/pull/2027) - Include units with AVHRR EPS metadata ([2026](https://github.com/pytroll/satpy/issues/2026))
* [PR 2017](https://github.com/pytroll/satpy/pull/2017) - Fix ABI rayleigh_corrected_crefl modifier using deprecated DEM specifier
* [PR 2015](https://github.com/pytroll/satpy/pull/2015) - Fix various dask array bugs in CREFL modifier
* [PR 2013](https://github.com/pytroll/satpy/pull/2013) - Fix angle generation caching occassionally swapping results ([2010](https://github.com/pytroll/satpy/issues/2010))
* [PR 2011](https://github.com/pytroll/satpy/pull/2011) - Fix memory leak in cached_property backport ([2008](https://github.com/pytroll/satpy/issues/2008), [2008](https://github.com/pytroll/satpy/issues/2008))
* [PR 2006](https://github.com/pytroll/satpy/pull/2006) - Fix Scene not being serializable
* [PR 2002](https://github.com/pytroll/satpy/pull/2002) - Update tests to be more flexible to CRS and enhancement changes
* [PR 1991](https://github.com/pytroll/satpy/pull/1991) - Update reference to dask distributed setup page
* [PR 1988](https://github.com/pytroll/satpy/pull/1988) - Update geometry.py docstring from compositor to modifier
* [PR 1987](https://github.com/pytroll/satpy/pull/1987) - Check that time is not already a coordinate in CF writer
* [PR 1983](https://github.com/pytroll/satpy/pull/1983) - More general filename filter for ascat soil moisture, allowing for Metop-B and Metop-C
* [PR 1982](https://github.com/pytroll/satpy/pull/1982) - Fix ninjotiff writer from erraneous K to C conversion

#### Features added

* [PR 2025](https://github.com/pytroll/satpy/pull/2025) - Allow skipping unit conversion in NinJoTIFF ([2024](https://github.com/pytroll/satpy/issues/2024))
* [PR 2007](https://github.com/pytroll/satpy/pull/2007) - Update abi_l2_nc to include filename metadata similar to abi_l1b
* [PR 2005](https://github.com/pytroll/satpy/pull/2005) - Add flag to MaskingCompositor to return RGBA for single-band input ([2004](https://github.com/pytroll/satpy/issues/2004))
* [PR 2000](https://github.com/pytroll/satpy/pull/2000) - Add a reader for the MSU-GS/A + Arctica-M1 data ([1999](https://github.com/pytroll/satpy/issues/1999))
* [PR 1992](https://github.com/pytroll/satpy/pull/1992) - Add support for CMIC product from PPSv2021
* [PR 1989](https://github.com/pytroll/satpy/pull/1989) - read the "elevation" variable in slstr_l1b
* [PR 1986](https://github.com/pytroll/satpy/pull/1986) - Add reader kwarg to 'ahi_hrit' to disable exact start_time ([1384](https://github.com/pytroll/satpy/issues/1384))
* [PR 1967](https://github.com/pytroll/satpy/pull/1967) - Add ability to read comma-separated colormaps during enhancement
* [PR 1966](https://github.com/pytroll/satpy/pull/1966) - Reduce MODIS L1b/L2 test case size for better test performance ([1883](https://github.com/pytroll/satpy/issues/1883))
* [PR 1962](https://github.com/pytroll/satpy/pull/1962) - Use a dependency matrix for benchmarking

#### Documentation changes

* [PR 2020](https://github.com/pytroll/satpy/pull/2020) - Clarify documentation regarding attributes used in get_angles
* [PR 1991](https://github.com/pytroll/satpy/pull/1991) - Update reference to dask distributed setup page
* [PR 1988](https://github.com/pytroll/satpy/pull/1988) - Update geometry.py docstring from compositor to modifier
* [PR 1969](https://github.com/pytroll/satpy/pull/1969) - Improve modifier documentation
* [PR 1968](https://github.com/pytroll/satpy/pull/1968) - Improve API documentation in CompositeBase
* [PR 1961](https://github.com/pytroll/satpy/pull/1961) - Update documentation to refer to all EO satellite data
* [PR 1960](https://github.com/pytroll/satpy/pull/1960) - Add release notes and security policy to documentation
* [PR 1950](https://github.com/pytroll/satpy/pull/1950) - Fix formatting in configuration documentation

In this release 30 pull requests were closed.


## Version 0.33.1 (2021/12/17)

### Issues Closed

* [Issue 1937](https://github.com/pytroll/satpy/issues/1937) - Add SECURITY.md
* [Issue 1932](https://github.com/pytroll/satpy/issues/1932) - warnings of `invalid value encountered in true_divide` and `invalid value encountered in double_scalars` in
* [Issue 1903](https://github.com/pytroll/satpy/issues/1903) - MPEF Product Header record definition , in seviri_base.py, needs to be updated
* [Issue 1799](https://github.com/pytroll/satpy/issues/1799) - Deprecate Scene.attrs property
* [Issue 1192](https://github.com/pytroll/satpy/issues/1192) - Harmonize SEVIRI area definitions

In this release 5 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 1946](https://github.com/pytroll/satpy/pull/1946) - Fix angle generation not working for StackedAreaDefinitions
* [PR 1942](https://github.com/pytroll/satpy/pull/1942) - Fix dynamic_dnb composite converting NaNs to 0s
* [PR 1941](https://github.com/pytroll/satpy/pull/1941) - Fix SAFE SAR azimuth noise array construction
* [PR 1918](https://github.com/pytroll/satpy/pull/1918) - Fix geo interpolation for aapp data

#### Features added

* [PR 1674](https://github.com/pytroll/satpy/pull/1674) - Feature add support for AHI True Color Reproduction

In this release 5 pull requests were closed.


## Version 0.33.0 (2021/12/10)

### Issues Closed

* [Issue 1930](https://github.com/pytroll/satpy/issues/1930) - ninjogeotiff writer produces file with ninjo_TransparentPixel=None ([PR 1931](https://github.com/pytroll/satpy/pull/1931) by [@gerritholl](https://github.com/gerritholl))
* [Issue 1902](https://github.com/pytroll/satpy/issues/1902) - High memory usage generating composites from ABI/AHI

In this release 2 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 1931](https://github.com/pytroll/satpy/pull/1931) - When no fill value is used, write TransparentPixel=-1 in ninjogeotiff headers ([1930](https://github.com/pytroll/satpy/issues/1930))
* [PR 1926](https://github.com/pytroll/satpy/pull/1926) - Update seadas_l2 chlor_a enhancement to use new log10 stretch
* [PR 1922](https://github.com/pytroll/satpy/pull/1922) - Fix ABI cloud_phase composite recipe and enhancement

#### Features added

* [PR 1917](https://github.com/pytroll/satpy/pull/1917) - Add support to read and visualize NOAA GOESR L2+ cloud mask products
* [PR 1912](https://github.com/pytroll/satpy/pull/1912) - Add Frequency range
* [PR 1908](https://github.com/pytroll/satpy/pull/1908) - Update AHI HSD calibration coefficients
* [PR 1905](https://github.com/pytroll/satpy/pull/1905) - Updated mpef product header to include new fields
* [PR 1882](https://github.com/pytroll/satpy/pull/1882) - Update GDAL_OPTIONS with driver= and COG-specific options
* [PR 1370](https://github.com/pytroll/satpy/pull/1370) - Add support for reading AAPP level-1c MHS/AMSU-B data

#### Refactoring

* [PR 1910](https://github.com/pytroll/satpy/pull/1910) - Refactor SZA and cos(SZA) generation to reduce duplicate computations

In this release 10 pull requests were closed.


## Version 0.32.0 (2021/12/01)

### Issues Closed

* [Issue 1900](https://github.com/pytroll/satpy/issues/1900) - Load composites mixed from files or provided data ([PR1901](https://github.com/pytroll/satpy/pull/1901) by [@djhoese](https://github.com/djhoese))
* [Issue 1898](https://github.com/pytroll/satpy/issues/1898) - Loading composites without file handlers fails with KeyError ([PR1899](https://github.com/pytroll/satpy/pull/1899) by [@erritholl](https://github.com/gerritholl))
* [Issue 1893](https://github.com/pytroll/satpy/issues/1893) - Download and install Satpy for raspberry pi
* [Issue 1889](https://github.com/pytroll/satpy/issues/1889) - Question: How to release loaded data from memory?
* [Issue 1880](https://github.com/pytroll/satpy/issues/1880) - Add area definitions corresponding to geostationary imager fields of regard ([PR1881](https://github.com/pytroll/satpy/pull/1881) by [@erritholl](https://github.com/gerritholl))
* [Issue 1879](https://github.com/pytroll/satpy/issues/1879) - How to use histogram enhancement in yaml files?
* [Issue 1749](https://github.com/pytroll/satpy/issues/1749) - Load from blended scene ([PR 1797](https://github.com/pytroll/satpy/pull/1797) by [@djhoese](https://github.com/djhoese))
* [Issue 1747](https://github.com/pytroll/satpy/issues/1747) - Load composites without file handlers. ([PR 1797](https://github.com/pytroll/satpy/pull/1797) by [@djhoese](https://github.com/djhoese))
* [Issue 1456](https://github.com/pytroll/satpy/issues/1456) - Default cache directory should respect XDG Base Directory Specification.
* [Issue 583](https://github.com/pytroll/satpy/issues/583) - PPP_CONFIG_DIR set locally does not include the global dir for the eps_l1b reader

In this release 10 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 1899](https://github.com/pytroll/satpy/pull/1899) - Fix loading multi-sensor composites for manually added data ([1898](https://github.com/pytroll/satpy/issues/1898))
* [PR 1891](https://github.com/pytroll/satpy/pull/1891) - Fix file handlers improperly matching some file types
* [PR 1884](https://github.com/pytroll/satpy/pull/1884) - Fix nucaps reader failing when given multiple input files

#### Features added

* [PR 1901](https://github.com/pytroll/satpy/pull/1901) - Update Scene.sensor_names to include sensors from readers and contained data ([1900](https://github.com/pytroll/satpy/issues/1900))
* [PR 1897](https://github.com/pytroll/satpy/pull/1897) - Update AHI gridded reader to use HTTP instead of FTP
* [PR 1894](https://github.com/pytroll/satpy/pull/1894) - Add 'seadas_l2' reader for 'chlor_a' product
* [PR 1892](https://github.com/pytroll/satpy/pull/1892) - Add new pre-commit checks
* [PR 1888](https://github.com/pytroll/satpy/pull/1888) - Optimize composite YAML loading
* [PR 1885](https://github.com/pytroll/satpy/pull/1885) - Add optional on-disk zarr caching to sensor angle generation
* [PR 1881](https://github.com/pytroll/satpy/pull/1881) - Add area definitions for GOES ABI FOR ([1880](https://github.com/pytroll/satpy/issues/1880))
* [PR 1797](https://github.com/pytroll/satpy/pull/1797) - Allow loading of composites after Scene resampling ([1752](https://github.com/pytroll/satpy/issues/1752), [1749](https://github.com/pytroll/satpy/issues/1749), [1747](https://github.com/pytroll/satpy/issues/1747))

#### Documentation changes

* [PR 1873](https://github.com/pytroll/satpy/pull/1873) - Fix a typo in the ninjogeotiff documentation

In this release 12 pull requests were closed.


## Version 0.31.0 (2021/11/03)

### Issues Closed

* [Issue 1866](https://github.com/pytroll/satpy/issues/1866) - Data Type of AHI NetCDF Output
* [Issue 1859](https://github.com/pytroll/satpy/issues/1859) - Yaml UnsafeLoader ImportErrror on colab.google ([PR1860](https://github.com/pytroll/satpy/pull/1860) by [@arammer](https://github.com/abrammer))
* [Issue 1853](https://m/pytroll/satpy/pull/1864) by [@djhoese](https://github.com/djhoese))

In this release 12 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR1868](https://github.com/pytroll/satpy/pull/1868) - Fix MiRS reader not working with new versions of dask
* [PR1860](https://github.com/pytroll/satpy/pull/1860) - Catch ImportError on UnsafeLoader in composites/config_loader ([1859](https://github.com/pytroll/satpy/issues/1859))
* [PR1855](https://github.com/pytroll/satpy/pull/1855) - Fix 'acspo' reader producing non-y/x dimension names
* [PR1854](https://github.com/pytroll/satpy/pull/1854) - Fix 'awips_tiled' writer doing unnecessary attribute formatting
* [PR1849](https://github.com/pytroll/satpy/pull/1849) - Update AGRI reader to ensure that angles can be correctly loaded.

#### Features added

* [PR1850](https://github.com/pytroll/satpy/pull/1850) - Adapt msi-safe to the new product format ([1847](https://github.com/pytroll/satpy/issues/1847))
* [PR1839](https://github.com/pytroll/satpy/pull/1839) - Add ninjogeotiff writer to write GeoTIFFs including necessary NinJo tags in GDALMetadata ([1838](https://github.com/pytroll/satpy/issues/1838))
* [PR1743](https://github.com/pytroll/satpy/pull/1743) - Add option to configure group_files behaviour on empty groups in case of multiple readers ([1742](https://github.com/pytroll/satpy/issues/1742))

#### Documentation changes

* [PR1867](https://github.com/pytroll/satpy/pull/1867) - Update PDF metadata for sphinx documentation
* [PR1864](https://github.com/pytroll/satpy/pull/1864) - Update Scene.save_datasets to clarify what will be saved ([1138](https://github.com/pytroll/satpy/issues/1138))
* [PR1862](https://github.com/pytroll/satpy/pull/1862) - Correct phrasing of upside-down
* [PR1852](https://github.com/pytroll/satpy/pull/1852) - Fix reference to dask distributed setup page

In this release 12 pull requests were closed.


## Version 0.30.1 (2021/09/28)

### Issues Closed

* [Issue 1835](https://github.com/pytroll/satpy/issues/1835) - scipy module error?
* [Issue 1832](https://github.com/pytroll/satpy/issues/1832) - variable from python to composite
* [Issue 1831](https://github.com/pytroll/satpy/issues/1831) - example yml files for other readers
* [Issue 1829](https://github.com/pytroll/satpy/issues/1829) - pytest satpy/tests does not work ([PR1830](https://github.com/pytroll/satpy/pull/1830) by [@djhoese](https://github.com/djhoese))
* [Issue 1828](https://github.com/pytroll/satpy/issues/1828) - Error occurred plotting Himawari-8
* [Issue 1484](https://github.com/pytroll/satpy/issues/1484) - Broken links to new EUMETSAT website ([PR1827](https://github.com/pytroll/satpy/pull/1827) by [@pdeyl](https://github.com/pdebuyl))

In this release 6 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR1837](https://github.com/pytroll/satpy/pull/1837) - Fix config path handling on Windows
* [PR1827](https://github.com/pytroll/satpy/pull/1827) - Fix eumetsat urls in satpy/readers ([1484](https://github.com/pytroll/satpy/issues/1484))

#### Documentation changes

* [PR1837](https://github.com/pytroll/satpy/pull/1837) - Fix config path handling on Windows
* [PR1830](https://github.com/pytroll/satpy/pull/1830) - Move tests_require to special "tests" extra for easier installation ([1829](https://github.com/pytroll/satpy/issues/1829))
* [PR1827](https://github.com/pytroll/satpy/pull/1827) - Fix eumetsat urls in satpy/readers ([1484](https://github.com/pytroll/satpy/issues/1484))

In this release 5 pull requests were closed.


## Version 0.30.0 (2021/09/17)

### Issues Closed

* [Issue 1821](https://github.com/pytroll/satpy/issues/1821) - Resampling to `true_color_with_night_ir_hires` no longer works. ([PR 1823](https://github.com/pytroll/satpy/pull/1823))
* [Issue 1803](https://github.com/pytroll/satpy/issues/1803) - how to xRITDecompress files for using satpy
* [Issue 1796](https://github.com/pytroll/satpy/issues/1796) - Extend use of bz2 compression for input files for seviri_l1b_hrit ([PR 1798](https://github.com/pytroll/satpy/pull/1798))
* [Issue 1794](https://github.com/pytroll/satpy/issues/1794) - ir_overview vs cloudtop
* [Issue 1793](https://github.com/pytroll/satpy/issues/1793) - Different `y_bounds` and `x_bounds` shapes of TROPOMI MultiScene
* [Issue 1791](https://github.com/pytroll/satpy/issues/1791) - Memory usage has increased drastically
* [Issue 1786](https://github.com/pytroll/satpy/issues/1786) - The `viirs_sdr` reader does not function correctly with `GMODO` geolocation. ([PR 1787](https://github.com/pytroll/satpy/pull/1787))
* [Issue 1783](https://github.com/pytroll/satpy/issues/1783) - Metadata name problem in HY-2B L2B reader ([PR 1785](https://github.com/pytroll/satpy/pull/1785))
* [Issue 1780](https://github.com/pytroll/satpy/issues/1780) - What shoud I do if I only want to keep the day part of DayNightCompositor? ([PR 1816](https://github.com/pytroll/satpy/pull/1816))
* [Issue 1779](https://github.com/pytroll/satpy/issues/1779) - piecewise_linear_stretch didn't work properly on GK-2A AMI data
* [Issue 1773](https://github.com/pytroll/satpy/issues/1773) - [Question] Geolocation information of FengYun4A (FY-4A) AGRI L1B data ([PR 1782](https://github.com/pytroll/satpy/pull/1782))
* [Issue 1759](https://github.com/pytroll/satpy/issues/1759) - Ask For Help: How to operate SunZenithCorrector manually?
* [Issue 1750](https://github.com/pytroll/satpy/issues/1750) - MultiScene.blend does not document the interface for the blend function ([PR 1751](https://github.com/pytroll/satpy/pull/1751))
* [Issue 1745](https://github.com/pytroll/satpy/issues/1745) - Resampling MODIS Level 1B data
* [Issue 1738](https://github.com/pytroll/satpy/issues/1738) - available_dataset_names omits composites depending on more than one reader
* [Issue 1730](https://github.com/pytroll/satpy/issues/1730) - geotiff writer ignores dtype argument, always writes float if enhance=False ([PR 1733](https://github.com/pytroll/satpy/pull/1733))
* [Issue 1728](https://github.com/pytroll/satpy/issues/1728) - Unable to read HY-2B SCA L2B file
* [Issue 1727](https://github.com/pytroll/satpy/issues/1727) - 'NoData' area is not black(clean) in the Sentinel-2 MSI output ([PR 1628](https://github.com/pytroll/satpy/pull/1628))
* [Issue 1722](https://github.com/pytroll/satpy/issues/1722) - 'ModuleNotFoundError' when processing Sentinel-2 MSI data ([PR 1723](https://github.com/pytroll/satpy/pull/1723))
* [Issue 1718](https://github.com/pytroll/satpy/issues/1718) - Raw metadata handling impacts performance ([PR 1795](https://github.com/pytroll/satpy/pull/1795))
* [Issue 1661](https://github.com/pytroll/satpy/issues/1661) - Support for clavrx netcdf files ([PR 1716](https://github.com/pytroll/satpy/pull/1716))
* [Issue 1625](https://github.com/pytroll/satpy/issues/1625) - Part of Sentinel-2 images missing when atmospheric corrected ([PR 1628](https://github.com/pytroll/satpy/pull/1628))
* [Issue 1584](https://github.com/pytroll/satpy/issues/1584) - to_xarray_dataset on empty scene fails with TypeError ([PR 1698](https://github.com/pytroll/satpy/pull/1698))

In this release 23 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 1823](https://github.com/pytroll/satpy/pull/1823) - Fix unify_chunks usage in compositors and fix image mode in BackgroundCompositor ([1821](https://github.com/pytroll/satpy/issues/1821))
* [PR 1814](https://github.com/pytroll/satpy/pull/1814) - Add missing metadata to MODIS L1b and L2 readers
* [PR 1813](https://github.com/pytroll/satpy/pull/1813) - Fix composites failing when inputs are different chunk sizes
* [PR 1808](https://github.com/pytroll/satpy/pull/1808) - Fix ReflectanceCorrector (crefl) for MODIS data
* [PR 1804](https://github.com/pytroll/satpy/pull/1804) - Fix consistency with nucaps sensor metadata (set/lowercase)
* [PR 1802](https://github.com/pytroll/satpy/pull/1802) - Add warning in 'awips_tiled' writer when 'units' are too long
* [PR 1800](https://github.com/pytroll/satpy/pull/1800) - Fix for missing attributes when requesting 'counts' calibration from ABI L1B reader.
* [PR 1792](https://github.com/pytroll/satpy/pull/1792) - Maintain categorical clavrx data as integer arrays
* [PR 1787](https://github.com/pytroll/satpy/pull/1787) - Fix 'viirs_sdr' repeating data when TC geolocation was not available ([1786](https://github.com/pytroll/satpy/issues/1786))
* [PR 1784](https://github.com/pytroll/satpy/pull/1784) - Fix ABI readers not assigning 'platform_name' for GOES-18/19
* [PR 1782](https://github.com/pytroll/satpy/pull/1782) - Update AGRI/L1 geolocation ([1773](https://github.com/pytroll/satpy/issues/1773))
* [PR 1777](https://github.com/pytroll/satpy/pull/1777) - Fix mviri l1b fiduceo reader compatibility with newer xarray
* [PR 1776](https://github.com/pytroll/satpy/pull/1776) - Fix 'awips_tiled' writer producing an invalid y coordinate
* [PR 1774](https://github.com/pytroll/satpy/pull/1774) - Fix the seviri benchmarks
* [PR 1771](https://github.com/pytroll/satpy/pull/1771) - Fix VIIRS SDR reader not handling multi-granule files with fewer scans
* [PR 1770](https://github.com/pytroll/satpy/pull/1770) - Fix CLAVR-x reader and 'awips_tiled' writer to produce AWIPS-compatible output
* [PR 1744](https://github.com/pytroll/satpy/pull/1744) - Fix VIRR reader handling valid_range when it is a numpy array
* [PR 1734](https://github.com/pytroll/satpy/pull/1734) - Remove valid_range from attributes in VIRR L1b reader
* [PR 1733](https://github.com/pytroll/satpy/pull/1733) - Fix geotiff writer ignoring dtype argument ([1730](https://github.com/pytroll/satpy/issues/1730), [1730](https://github.com/pytroll/satpy/issues/1730))
* [PR 1724](https://github.com/pytroll/satpy/pull/1724) - Replace doc references to PPP_CONFIG_DIR ([1724](https://github.com/pytroll/satpy/issues/1724))
* [PR 1723](https://github.com/pytroll/satpy/pull/1723) - Fix package dependencies for the `msi_safe` reader ([1722](https://github.com/pytroll/satpy/issues/1722))
* [PR 1698](https://github.com/pytroll/satpy/pull/1698) - Fix error when calling to_xarray_dataset on an empty scene ([1584](https://github.com/pytroll/satpy/issues/1584))
* [PR 1628](https://github.com/pytroll/satpy/pull/1628) - Fix for transposed angles in safe-msi reader ([1727](https://github.com/pytroll/satpy/issues/1727), [1625](https://github.com/pytroll/satpy/issues/1625))

#### Features added

* [PR 1824](https://github.com/pytroll/satpy/pull/1824) - Add additional ACSPO reader file patterns
* [PR 1817](https://github.com/pytroll/satpy/pull/1817) - Fix ninjotiff writer for mode P
* [PR 1816](https://github.com/pytroll/satpy/pull/1816) - Add 'day_night' flag to DayNightCompositor for day-only or night-only results ([1780](https://github.com/pytroll/satpy/issues/1780))
* [PR 1815](https://github.com/pytroll/satpy/pull/1815) - Add MODIS L2 products produced by IMAPP
* [PR 1805](https://github.com/pytroll/satpy/pull/1805) - Add 'reader' name to all produced DataArrays
* [PR 1801](https://github.com/pytroll/satpy/pull/1801) - added link to the GOES-2-go package in the docs as a download source.
* [PR 1798](https://github.com/pytroll/satpy/pull/1798) - Add on-the-fly bz2 decompression for HRIT MSG PRO and EPI files ([1796](https://github.com/pytroll/satpy/issues/1796))
* [PR 1790](https://github.com/pytroll/satpy/pull/1790) - Add ABI L1B benchmarks
* [PR 1785](https://github.com/pytroll/satpy/pull/1785) - Feature handle data from HY-2B SCAT files directly from NSOAS  ([1783](https://github.com/pytroll/satpy/issues/1783))
* [PR 1772](https://github.com/pytroll/satpy/pull/1772) - Add access point to global_attrs to netCDF4FileHandler
* [PR 1760](https://github.com/pytroll/satpy/pull/1760) - Add benchmarks for seviri hrit
* [PR 1720](https://github.com/pytroll/satpy/pull/1720) - Add a test to ensure seviri hrv has priority over vis008 when requesting 0.8µm
* [PR 1717](https://github.com/pytroll/satpy/pull/1717) - Add low resolution file patterns for AHI HSD reader
* [PR 1716](https://github.com/pytroll/satpy/pull/1716) - Update Clavrx reader for netcdf files ([1661](https://github.com/pytroll/satpy/issues/1661))
* [PR 1692](https://github.com/pytroll/satpy/pull/1692) - Add raw 'counts' calibration to 'abi_l1b' reader
* [PR 1297](https://github.com/pytroll/satpy/pull/1297) - Add support for MCMIP GOES ABI L2 files ([1162](https://github.com/pytroll/satpy/issues/1162))

#### Documentation changes

* [PR 1819](https://github.com/pytroll/satpy/pull/1819) - Fix invalid YAML syntax in enhancement documentation
* [PR 1801](https://github.com/pytroll/satpy/pull/1801) - added link to the GOES-2-go package in the docs as a download source.
* [PR 1765](https://github.com/pytroll/satpy/pull/1765) - Add missing demo data directory entry to config documentation
* [PR 1751](https://github.com/pytroll/satpy/pull/1751) - Improve documentation for MultiScene.blend ([1750](https://github.com/pytroll/satpy/issues/1750))
* [PR 1726](https://github.com/pytroll/satpy/pull/1726) - Point out get_area_def in resample documentation ([1726](https://github.com/pytroll/satpy/issues/1726))
* [PR 1724](https://github.com/pytroll/satpy/pull/1724) - Replace doc references to PPP_CONFIG_DIR ([1724](https://github.com/pytroll/satpy/issues/1724))

In this release 45 pull requests were closed.


## Version 0.29.0 (2021/06/04)

### Issues Closed

* [Issue 1714](https://github.com/pytroll/satpy/issues/1714) - Plotting day night composite satellite image
* [Issue 1689](https://github.com/pytroll/satpy/issues/1689) - BackgroundCompositor using IR Sandwich (masked so only coldest clouds are visible) and True Color as inputs ([PR 1690](https://github.com/pytroll/satpy/pull/1690))
* [Issue 1684](https://github.com/pytroll/satpy/issues/1684) - Rename fci_l1c_fdhsi to fci_l1c_nc ([PR 1712](https://github.com/pytroll/satpy/pull/1712))
* [Issue 1293](https://github.com/pytroll/satpy/issues/1293) - DOC: broken link for geoview ([PR 1697](https://github.com/pytroll/satpy/pull/1697))
* [Issue 1120](https://github.com/pytroll/satpy/issues/1120) - Broken-off sentence in `cf_writer` module documentation: "If a non-dimensional coordinate is identical for" ([PR 1697](https://github.com/pytroll/satpy/pull/1697))
* [Issue 1104](https://github.com/pytroll/satpy/issues/1104) - NUCAPS reader uses incorrect _FillValue ([PR 1710](https://github.com/pytroll/satpy/pull/1710))
* [Issue 1097](https://github.com/pytroll/satpy/issues/1097) - Deprecate satpy.readers.utils.get_area_slices
* [Issue 1085](https://github.com/pytroll/satpy/issues/1085) - Add tonemapping modifiers for truecolor images
* [Issue 1060](https://github.com/pytroll/satpy/issues/1060) - Reorder installation instructions to put conda before PyPI ([PR 1711](https://github.com/pytroll/satpy/pull/1711))
* [Issue 1028](https://github.com/pytroll/satpy/issues/1028) - Mitiff tests failing on python 3.7 travis environments
* [Issue 990](https://github.com/pytroll/satpy/issues/990) - Documentation on storing area definitions has a broken fragment identifier link to pyresample ([PR 1697](https://github.com/pytroll/satpy/pull/1697))
* [Issue 973](https://github.com/pytroll/satpy/issues/973) - For VIIRS composite there are two composites with the same name.
* [Issue 936](https://github.com/pytroll/satpy/issues/936) - Swap names for Vis/IR default natural_color and natural_color_sun composites
* [Issue 722](https://github.com/pytroll/satpy/issues/722) - Standardise self.mda for SEVIRI attributes
* [Issue 608](https://github.com/pytroll/satpy/issues/608) - Update to fix deprecation warning from dask regarding atop
* [Issue 566](https://github.com/pytroll/satpy/issues/566) - Add AbstractScene class
* [Issue 500](https://github.com/pytroll/satpy/issues/500) - Add ability to add proper references to published algorithms
* [Issue 495](https://github.com/pytroll/satpy/issues/495) - Update tests to skip tests if dependencies are missing
* [Issue 425](https://github.com/pytroll/satpy/issues/425) - Add DART compatible observation writer
* [Issue 346](https://github.com/pytroll/satpy/issues/346) - lat-lon as the default dimensions
* [Issue 334](https://github.com/pytroll/satpy/issues/334) - Add 'Performance Tips' section to documentation
* [Issue 164](https://github.com/pytroll/satpy/issues/164) - Should enhancers know the data type beforehand
* [Issue 102](https://github.com/pytroll/satpy/issues/102) - Fix meteosat 10 area
* [Issue 100](https://github.com/pytroll/satpy/issues/100) - Add background color option to simple image writer
* [Issue 99](https://github.com/pytroll/satpy/issues/99) - Adding coastlines does not preserve transparency
* [Issue 92](https://github.com/pytroll/satpy/issues/92) - Merge area definition files
* [Issue 9](https://github.com/pytroll/satpy/issues/9) - Convert mpop readers to satpy yaml readers

In this release 27 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 1710](https://github.com/pytroll/satpy/pull/1710) - Fix NUCAPS reader having incorrect _FillValue attribute ([1104](https://github.com/pytroll/satpy/issues/1104))
* [PR 1706](https://github.com/pytroll/satpy/pull/1706) - Update SLSTR reader to choose correct file for interpolated angles
* [PR 1691](https://github.com/pytroll/satpy/pull/1691) - Fix reference to sector_id global key in 'awips_tiled' writer YAML
* [PR 1690](https://github.com/pytroll/satpy/pull/1690) - Fix SandwichCompositor modifying input data ([1689](https://github.com/pytroll/satpy/issues/1689))
* [PR 1679](https://github.com/pytroll/satpy/pull/1679) - Remove extra attributes tag and fix indentation

#### Features added

* [PR 1715](https://github.com/pytroll/satpy/pull/1715) - Fix benchmarks to run with older commits
* [PR 1701](https://github.com/pytroll/satpy/pull/1701) - Add pending deprecation reader names check
* [PR 1680](https://github.com/pytroll/satpy/pull/1680) - Implement reading of index map and auxiliary data in FCI L1c reader

#### Documentation changes

* [PR 1711](https://github.com/pytroll/satpy/pull/1711) - Rewrite installation instructions to make conda use clearer ([1060](https://github.com/pytroll/satpy/issues/1060))
* [PR 1697](https://github.com/pytroll/satpy/pull/1697) - Solve various documentation issues ([990](https://github.com/pytroll/satpy/issues/990), [1293](https://github.com/pytroll/satpy/issues/1293), [1120](https://github.com/pytroll/satpy/issues/1120))

In this release 10 pull requests were closed.


## Version 0.28.1 (2021/05/18)

### Issues Closed

* [Issue 1676](https://github.com/pytroll/satpy/issues/1676) - New config feature does not support a subprocess call to another script which uses satpy too. ([PR 1677](https://github.com/pytroll/satpy/pull/1677))
* [Issue 1647](https://github.com/pytroll/satpy/issues/1647) - Bucket resamplers AttributeError in logging message ([PR 1648](https://github.com/pytroll/satpy/pull/1648))
* [Issue 1145](https://github.com/pytroll/satpy/issues/1145) - satpy to support reading of satpy generated netcdf cf files
* [Issue 1016](https://github.com/pytroll/satpy/issues/1016) - Add reader for netcdf datasets written with Satpy
* [Issue 604](https://github.com/pytroll/satpy/issues/604) - test_generic_image.py failure: "projection not named"
* [Issue 562](https://github.com/pytroll/satpy/issues/562) - Undocumented dependency packages ([PR 1673](https://github.com/pytroll/satpy/pull/1673))

In this release 6 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 1677](https://github.com/pytroll/satpy/pull/1677) - Fix SATPY_CONFIG_PATH being unusable when imported in a subprocess ([1676](https://github.com/pytroll/satpy/issues/1676))
* [PR 1671](https://github.com/pytroll/satpy/pull/1671) - Improve MiRS reader handling of missing metadata
* [PR 1670](https://github.com/pytroll/satpy/pull/1670) - Fix combination of raw metadata (again)
* [PR 1666](https://github.com/pytroll/satpy/pull/1666) - Ensure that orbital parameters are in a dict
* [PR 1648](https://github.com/pytroll/satpy/pull/1648) - Fix bucket resamplers trying to print non-existent name ([1647](https://github.com/pytroll/satpy/issues/1647))
* [PR 1639](https://github.com/pytroll/satpy/pull/1639) - Fix MultiScene writer handling of multiple delayed objects
* [PR 1499](https://github.com/pytroll/satpy/pull/1499) - Fix default dtype in geotiff writer if enhance=False

In this release 7 pull requests were closed.


## Version 0.28.0 (2021/05/14)

### Issues Closed

* [Issue 1669](https://github.com/pytroll/satpy/issues/1669) - Cropping a country from an earth image using latitude and longitude coordinate
* [Issue 1667](https://github.com/pytroll/satpy/issues/1667) - Extracting data/ reading data from .DAT file
* [Issue 1664](https://github.com/pytroll/satpy/issues/1664) - Nan values when resample with Kompsat
* [Issue 1656](https://github.com/pytroll/satpy/issues/1656) - Cannot load datasets of multiple SEVIRI native files  ([PR 1663](https://github.com/pytroll/satpy/pull/1663))
* [Issue 1650](https://github.com/pytroll/satpy/issues/1650) - wrong gamma for red beam of cira_fire_temperature RGB ([PR 1662](https://github.com/pytroll/satpy/pull/1662))
* [Issue 1641](https://github.com/pytroll/satpy/issues/1641) - UnicodeDecodeError and ValueError when passing local FSFile to abi_l1b
* [Issue 1635](https://github.com/pytroll/satpy/issues/1635) - The `crop` function is no longer working.
* [Issue 1633](https://github.com/pytroll/satpy/issues/1633) - Auxiliary offline download doesn't work for modifiers ([PR 1634](https://github.com/pytroll/satpy/pull/1634))
* [Issue 1632](https://github.com/pytroll/satpy/issues/1632) - Can't resample GOES Meso data when using night IR composite ([PR 1643](https://github.com/pytroll/satpy/pull/1643))
* [Issue 1626](https://github.com/pytroll/satpy/issues/1626) - problem with read  UMETSAT
* [Issue 1601](https://github.com/pytroll/satpy/issues/1601) - Allow MiRS reader to apply limb correction optionally ([PR 1621](https://github.com/pytroll/satpy/pull/1621))
* [Issue 1594](https://github.com/pytroll/satpy/issues/1594) - slstr_l2: Failed to filter out correct files using find_files_and_readers() with start_time and end_time
* [Issue 1562](https://github.com/pytroll/satpy/issues/1562) - Improve Scene.copy wishlist handling when datasets to copy are specified ([PR 1630](https://github.com/pytroll/satpy/pull/1630))
* [Issue 1495](https://github.com/pytroll/satpy/issues/1495) - Values of reflectance

In this release 14 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 1665](https://github.com/pytroll/satpy/pull/1665) - Fix fci l2 tests on windows
* [PR 1663](https://github.com/pytroll/satpy/pull/1663) - Ignore raw metadata when combining metadata ([1656](https://github.com/pytroll/satpy/issues/1656))
* [PR 1662](https://github.com/pytroll/satpy/pull/1662) - Fix cira fire temperature and green snow ([1650](https://github.com/pytroll/satpy/issues/1650))
* [PR 1655](https://github.com/pytroll/satpy/pull/1655) - Apply valid_range in MiRS reader when present
* [PR 1644](https://github.com/pytroll/satpy/pull/1644) - Add id for GOMS3/Electro-l n3
* [PR 1643](https://github.com/pytroll/satpy/pull/1643) - Fix combine_metadata not handling lists of different sizes ([1632](https://github.com/pytroll/satpy/issues/1632))
* [PR 1640](https://github.com/pytroll/satpy/pull/1640) - Fix AAPP l1b reader for negative slope on channel 2 ([332](https://github.com/ssec/polar2grid/issues/332))
* [PR 1634](https://github.com/pytroll/satpy/pull/1634) - Fix offline aux download not working for modifiers ([1633](https://github.com/pytroll/satpy/issues/1633))
* [PR 1631](https://github.com/pytroll/satpy/pull/1631) - Fix satellite altitude being in kilometers in ABI L2 reader
* [PR 1630](https://github.com/pytroll/satpy/pull/1630) - Fix Scene.copy not preserving wishlist properly ([1562](https://github.com/pytroll/satpy/issues/1562))
* [PR 1578](https://github.com/pytroll/satpy/pull/1578) - Fix nightly/unstable CI URL

#### Features added

* [PR 1659](https://github.com/pytroll/satpy/pull/1659) - Add SEVIRI + NWC SAF GEO VIS/IR cloud overlay composite
* [PR 1657](https://github.com/pytroll/satpy/pull/1657) - Add parallax-corrected file patterns to the nwcsaf-geo reader
* [PR 1646](https://github.com/pytroll/satpy/pull/1646) - Add new piecewise_linear_stretch enhancement method
* [PR 1636](https://github.com/pytroll/satpy/pull/1636) - Add first benchmarks (uses asv)
* [PR 1623](https://github.com/pytroll/satpy/pull/1623) - Add the reinhard enhancements
* [PR 1621](https://github.com/pytroll/satpy/pull/1621) - Add `limb_correction` keyword argument to MiRS reader ([1601](https://github.com/pytroll/satpy/issues/1601))
* [PR 1620](https://github.com/pytroll/satpy/pull/1620) - Add feature to StaticImageCompositor to allow filenames relative to Satpy 'data_dir'
* [PR 1560](https://github.com/pytroll/satpy/pull/1560) - Allow custom dataset names in 'generic_image' reader and fix nodata handling

In this release 19 pull requests were closed.


## Version 0.27.0 (2021/03/26)

### Issues Closed

* [Issue 1616](https://github.com/pytroll/satpy/issues/1616) - Thermal channels NinJoTIFF writing fail with AttributeError due to lost attributes ([PR 1617](https://github.com/pytroll/satpy/pull/1617))
* [Issue 1614](https://github.com/pytroll/satpy/issues/1614) - Saving AAPP-processed NOAA HRPT to NinJoTIFF fails with AttributeError ([PR 1615](https://github.com/pytroll/satpy/pull/1615))
* [Issue 1608](https://github.com/pytroll/satpy/issues/1608) - SEVIRI L1.5 native reader does not support files not including 0100 in the file name ([PR 1609](https://github.com/pytroll/satpy/pull/1609))
* [Issue 1605](https://github.com/pytroll/satpy/issues/1605) - Reading FSFile fails with TypeError ([PR 1606](https://github.com/pytroll/satpy/pull/1606))
* [Issue 1604](https://github.com/pytroll/satpy/issues/1604) - group_files does not support FSFile objects (TypeError: unhasshable type 'FSFile') ([PR 1606](https://github.com/pytroll/satpy/pull/1606))
* [Issue 1493](https://github.com/pytroll/satpy/issues/1493) - Failed to save tropomi nc file with specific variables loaded ([PR 1588](https://github.com/pytroll/satpy/pull/1588))

In this release 6 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 1617](https://github.com/pytroll/satpy/pull/1617) - Fix ninjotiff convert units attributes ([1616](https://github.com/pytroll/satpy/issues/1616))
* [PR 1615](https://github.com/pytroll/satpy/pull/1615) - Fix and improve unit conversion when writing NinJoTIFF ([1614](https://github.com/pytroll/satpy/issues/1614))
* [PR 1613](https://github.com/pytroll/satpy/pull/1613) - Standardize vii dims
* [PR 1610](https://github.com/pytroll/satpy/pull/1610) - Fix auxiliary download script not using provided data directory
* [PR 1609](https://github.com/pytroll/satpy/pull/1609) - Fix file pattern matching in SEVIRI Native reader ([1608](https://github.com/pytroll/satpy/issues/1608))
* [PR 1606](https://github.com/pytroll/satpy/pull/1606) - Make FSFile hashable again ([1605](https://github.com/pytroll/satpy/issues/1605), [1604](https://github.com/pytroll/satpy/issues/1604))
* [PR 1603](https://github.com/pytroll/satpy/pull/1603) - Update slstr_l2.yaml
* [PR 1600](https://github.com/pytroll/satpy/pull/1600) - When setting `upper_right_corner` make sure that all dataset coordinates are flipped
* [PR 1588](https://github.com/pytroll/satpy/pull/1588) - Bugfix of link_coords ([1493](https://github.com/pytroll/satpy/issues/1493))

#### Features added

* [PR 1618](https://github.com/pytroll/satpy/pull/1618) - Update VIIRS/MODIS ReflecanceCorrector modifier to download DEM data at runtime
* [PR 1612](https://github.com/pytroll/satpy/pull/1612) - Add support for SEVIRI Native files without archive header
* [PR 1602](https://github.com/pytroll/satpy/pull/1602) - Fix missing VIIRS SDR DNB solar and lunar azimuth angle datasets
* [PR 1468](https://github.com/pytroll/satpy/pull/1468) - Harmonize SEVIRI auxiliary data

#### Documentation changes

* [PR 1599](https://github.com/pytroll/satpy/pull/1599) - Add MiRS and mimicTPW2_comp readers to sphinx documentation

In this release 14 pull requests were closed.


## Version 0.26.0 (2021/03/15)

### Issues Closed

* [Issue 1587](https://github.com/pytroll/satpy/issues/1587) - Don't allow auxiliary downloads during tests ([PR 1591](https://github.com/pytroll/satpy/pull/1591))
* [Issue 1581](https://github.com/pytroll/satpy/issues/1581) - FSFile object compares unequal when all properties equal ([PR 1582](https://github.com/pytroll/satpy/pull/1582))
* [Issue 1573](https://github.com/pytroll/satpy/issues/1573) - Crash when reaching warnings.DeprecationWarning ([PR 1576](https://github.com/pytroll/satpy/pull/1576))
* [Issue 1572](https://github.com/pytroll/satpy/issues/1572) - Satpy Github issue template example code fails with ModuleNotFoundError ([PR 1575](https://github.com/pytroll/satpy/pull/1575))
* [Issue 1550](https://github.com/pytroll/satpy/issues/1550) - Scene metadata overwriting composite metadata and handling sets in filename generation ([PR 1551](https://github.com/pytroll/satpy/pull/1551))
* [Issue 1549](https://github.com/pytroll/satpy/issues/1549) - Satpy problems with MODIS ([PR 1556](https://github.com/pytroll/satpy/pull/1556))
* [Issue 1538](https://github.com/pytroll/satpy/issues/1538) - modifier API documentation not included with sphinx-generated API documentation
* [Issue 1536](https://github.com/pytroll/satpy/issues/1536) - Can't resample mscn to GridDefinition
* [Issue 1532](https://github.com/pytroll/satpy/issues/1532) - Loading SLSTR composite doesn't respect the `view` ([PR 1533](https://github.com/pytroll/satpy/pull/1533))
* [Issue 1530](https://github.com/pytroll/satpy/issues/1530) - Improve documentation/handling of string input for config_path  ([PR 1534](https://github.com/pytroll/satpy/pull/1534))
* [Issue 1520](https://github.com/pytroll/satpy/issues/1520) - Test failure if SATPY_CONFIG_PATH set ([PR 1521](https://github.com/pytroll/satpy/pull/1521))
* [Issue 1518](https://github.com/pytroll/satpy/issues/1518) - satpy_cf_nc reader fails to read satpy cf writer generated netcdf files where variables start with a number. ([PR 1525](https://github.com/pytroll/satpy/pull/1525))
* [Issue 1517](https://github.com/pytroll/satpy/issues/1517) - Scene.load error on conflicting 'y' values with MSG example.
* [Issue 1516](https://github.com/pytroll/satpy/issues/1516) - FSFile should support any PathLike objects ([PR 1519](https://github.com/pytroll/satpy/pull/1519))
* [Issue 1510](https://github.com/pytroll/satpy/issues/1510) - Seviri L1b native Solar zenith angle
* [Issue 1509](https://github.com/pytroll/satpy/issues/1509) - Replace pkg_resources usage with version.py file ([PR 1512](https://github.com/pytroll/satpy/pull/1512))
* [Issue 1508](https://github.com/pytroll/satpy/issues/1508) - Add sphinx building to GitHub Actions
* [Issue 1507](https://github.com/pytroll/satpy/issues/1507) - FCI Level2 OCA Data - error parameters have a parameter name change in the latest version of the test data ([PR 1524](https://github.com/pytroll/satpy/pull/1524))
* [Issue 1477](https://github.com/pytroll/satpy/issues/1477) - seviri l2 grib add file names from Eumetsat datastore ([PR 1503](https://github.com/pytroll/satpy/pull/1503))
* [Issue 1362](https://github.com/pytroll/satpy/issues/1362) - Feature request: download tif's if needed in a composite ([PR 1513](https://github.com/pytroll/satpy/pull/1513))
* [Issue 894](https://github.com/pytroll/satpy/issues/894) - SCMI Writer can produce un-ingestable AWIPS files
* [Issue 628](https://github.com/pytroll/satpy/issues/628) - Use 'donfig' package for global configuration settings ([PR 1501](https://github.com/pytroll/satpy/pull/1501))
* [Issue 367](https://github.com/pytroll/satpy/issues/367) - Add 'to_xarray_dataset' method to Scene
* [Issue 175](https://github.com/pytroll/satpy/issues/175) - Cannot read AVHRR in HRPT format (geoloc dtype error) ([PR 1531](https://github.com/pytroll/satpy/pull/1531))

In this release 24 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 1596](https://github.com/pytroll/satpy/pull/1596) - Fix bug in finest_area and coarsest_area logic for originally flipped SEVIRI data
* [PR 1592](https://github.com/pytroll/satpy/pull/1592) - Fix tests where xarray was unable to guess backend engine
* [PR 1589](https://github.com/pytroll/satpy/pull/1589) - Delete unnecessary coordinates in tropomi reader
* [PR 1582](https://github.com/pytroll/satpy/pull/1582) - Ensure FSFile objects compare equal when they should ([1581](https://github.com/pytroll/satpy/issues/1581))
* [PR 1579](https://github.com/pytroll/satpy/pull/1579) - Fix AHI HSD reader not having access to the AreaDefinition on load
* [PR 1574](https://github.com/pytroll/satpy/pull/1574) - Fix, correct usage of data returned by pyspectral AtmosphericalCorrection
* [PR 1567](https://github.com/pytroll/satpy/pull/1567) - Redesign awips_tiled writer to avoid xarray/dask deadlocks
* [PR 1564](https://github.com/pytroll/satpy/pull/1564) - Fix DifferenceCompositor ignoring YAML metadata
* [PR 1558](https://github.com/pytroll/satpy/pull/1558) - Fix dependency tree CompositorNode not retaining properties on copy
* [PR 1556](https://github.com/pytroll/satpy/pull/1556) - Fix the dataid sorting ([1549](https://github.com/pytroll/satpy/issues/1549))
* [PR 1551](https://github.com/pytroll/satpy/pull/1551) - Fix composite metadata overwriting and 'sensor' filename formatting ([1550](https://github.com/pytroll/satpy/issues/1550))
* [PR 1548](https://github.com/pytroll/satpy/pull/1548) - Add 'environment_prefix' to AWIPS tiled writer for flexible filenames
* [PR 1546](https://github.com/pytroll/satpy/pull/1546) - Make viirs-compact datasets compatible with dask distributed
* [PR 1545](https://github.com/pytroll/satpy/pull/1545) - Fix deprecated sphinx html_context usage in conf.py
* [PR 1542](https://github.com/pytroll/satpy/pull/1542) - Fix compression not being applied in awips_tiled writer
* [PR 1541](https://github.com/pytroll/satpy/pull/1541) - Fix swath builtin coordinates not being used
* [PR 1537](https://github.com/pytroll/satpy/pull/1537) - Add static scale_factor/add_offset/_FillValue to awips_tiled GLM config
* [PR 1533](https://github.com/pytroll/satpy/pull/1533) - Fix SLSTR composites for oblique view ([1532](https://github.com/pytroll/satpy/issues/1532))
* [PR 1531](https://github.com/pytroll/satpy/pull/1531) - Update the HRPT reader to latest satpy api ([175](https://github.com/pytroll/satpy/issues/175))
* [PR 1524](https://github.com/pytroll/satpy/pull/1524) - Fixed issue with reading fci oca error data and added fci toz product ([1507](https://github.com/pytroll/satpy/issues/1507))
* [PR 1521](https://github.com/pytroll/satpy/pull/1521) - Fix config test when user environment variables are set ([1520](https://github.com/pytroll/satpy/issues/1520))
* [PR 1519](https://github.com/pytroll/satpy/pull/1519) - Allow to pass pathlike-objects to FSFile ([1516](https://github.com/pytroll/satpy/issues/1516))
* [PR 1514](https://github.com/pytroll/satpy/pull/1514) - Correct the pdict a_name of agri_l1 reader
* [PR 1503](https://github.com/pytroll/satpy/pull/1503) - Fix issue with reading MSG GRIB products from the eumetsat datastore ([1477](https://github.com/pytroll/satpy/issues/1477))

#### Features added

* [PR 1597](https://github.com/pytroll/satpy/pull/1597) - add file_patterns in file_types with resolution type for satpy_cf_nc reader
* [PR 1591](https://github.com/pytroll/satpy/pull/1591) - Disallow tests from downloading files while running tests ([1587](https://github.com/pytroll/satpy/issues/1587))
* [PR 1586](https://github.com/pytroll/satpy/pull/1586) - Update GRIB reader for greater flexibility.
* [PR 1580](https://github.com/pytroll/satpy/pull/1580) - Sar-c reader optimization
* [PR 1577](https://github.com/pytroll/satpy/pull/1577) - New compositors: MultiFiller and LongitudeMaskingCompositor
* [PR 1570](https://github.com/pytroll/satpy/pull/1570) - Add the SAR Ice Log composite
* [PR 1565](https://github.com/pytroll/satpy/pull/1565) - Rename min_area() and max_area() methods
* [PR 1563](https://github.com/pytroll/satpy/pull/1563) - Allow 'glm_l2' reader to accept arbitrary filename prefixes
* [PR 1555](https://github.com/pytroll/satpy/pull/1555) - Add altitude in the list of dataset for OLCI.nc
* [PR 1554](https://github.com/pytroll/satpy/pull/1554) - Enable showing DeprecationWarning in debug_on and add unit test ([1554](https://github.com/pytroll/satpy/issues/1554))
* [PR 1544](https://github.com/pytroll/satpy/pull/1544) - Read wavelength ranges from netcdf
* [PR 1539](https://github.com/pytroll/satpy/pull/1539) - Fix args of bucket_sum and bucket_avg resampler
* [PR 1525](https://github.com/pytroll/satpy/pull/1525) - When saving to CF prepend datasets starting with a digit by CHANNEL_ ([1518](https://github.com/pytroll/satpy/issues/1518))
* [PR 1522](https://github.com/pytroll/satpy/pull/1522) - Switch to 'ewa' and 'ewa_legacy' resamplers from pyresample
* [PR 1513](https://github.com/pytroll/satpy/pull/1513) - Add auxiliary data download API ([1362](https://github.com/pytroll/satpy/issues/1362))
* [PR 1505](https://github.com/pytroll/satpy/pull/1505) - Ascat soilmoisture reader
* [PR 1501](https://github.com/pytroll/satpy/pull/1501) - Add central configuration object ([628](https://github.com/pytroll/satpy/issues/628))

#### Documentation changes

* [PR 1559](https://github.com/pytroll/satpy/pull/1559) - Fix geotiff writer FAQ link
* [PR 1545](https://github.com/pytroll/satpy/pull/1545) - Fix deprecated sphinx html_context usage in conf.py
* [PR 1543](https://github.com/pytroll/satpy/pull/1543) - Switch to sphinxcontrib.apidoc for automatically updating API docs ([1540](https://github.com/pytroll/satpy/issues/1540))
* [PR 1534](https://github.com/pytroll/satpy/pull/1534) - Clarify usage of config 'config_path' option ([1530](https://github.com/pytroll/satpy/issues/1530))

#### Backward incompatible changes

* [PR 1565](https://github.com/pytroll/satpy/pull/1565) - Rename min_area() and max_area() methods
* [PR 1561](https://github.com/pytroll/satpy/pull/1561) - Remove deprecated VIIRSFog compositor in favor of DifferenceCompositor
* [PR 1501](https://github.com/pytroll/satpy/pull/1501) - Add central configuration object ([628](https://github.com/pytroll/satpy/issues/628))

In this release 48 pull requests were closed.


## Version 0.25.1 (2021/01/06)

### Issues Closed

* [Issue 1500](https://github.com/pytroll/satpy/issues/1500) - Cannot create a scene for OLCI data

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 1502](https://github.com/pytroll/satpy/pull/1502) - Fix the linting error of test_agri_l1
* [PR 1459](https://github.com/pytroll/satpy/pull/1459) - Remove unnecessary string decode in agri_l1 reader

In this release 2 pull requests were closed.


## Version 0.25.0 (2021/01/04)

### Issues Closed

* [Issue 1494](https://github.com/pytroll/satpy/issues/1494) - geolocation problem with MODIS LAADS data
* [Issue 1489](https://github.com/pytroll/satpy/issues/1489) - The reader "viirs_l1b" cannot read the VIIRS L1B data
* [Issue 1488](https://github.com/pytroll/satpy/issues/1488) - Resampling with bucket resamplers drops coords from xr.DataArray ([PR 1491](https://github.com/pytroll/satpy/pull/1491))
* [Issue 1460](https://github.com/pytroll/satpy/issues/1460) - VIIl1b reader fails for testdata ([PR 1462](https://github.com/pytroll/satpy/pull/1462))
* [Issue 1453](https://github.com/pytroll/satpy/issues/1453) - Small error in documentation ([PR 1473](https://github.com/pytroll/satpy/pull/1473))
* [Issue 1449](https://github.com/pytroll/satpy/issues/1449) - Encoding of wavelength range ([PR 1466](https://github.com/pytroll/satpy/pull/1466))
* [Issue 1446](https://github.com/pytroll/satpy/issues/1446) - Resample
* [Issue 1443](https://github.com/pytroll/satpy/issues/1443) - Loading and resampling composites sometimes discards their dependencies ([PR 1351](https://github.com/pytroll/satpy/pull/1351))
* [Issue 1440](https://github.com/pytroll/satpy/issues/1440) - Error reading SEVIRI native file from EUMETSAT API ([PR 1438](https://github.com/pytroll/satpy/pull/1438))
* [Issue 1437](https://github.com/pytroll/satpy/issues/1437) - HSD / HRIT projection question
* [Issue 1436](https://github.com/pytroll/satpy/issues/1436) - 'str' object has no attribute 'decode' during Sentinel-2 MSI processing
* [Issue 1187](https://github.com/pytroll/satpy/issues/1187) - Areas claiming to view "full globe" should be labelled "full disk" instead ([PR 1485](https://github.com/pytroll/satpy/pull/1485))

In this release 12 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 1491](https://github.com/pytroll/satpy/pull/1491) - Fix missing coordinates for bucket resamplers ([1488](https://github.com/pytroll/satpy/issues/1488))
* [PR 1481](https://github.com/pytroll/satpy/pull/1481) - Remove x/y coordinates in mviri_l1b_fiduceo_nc
* [PR 1473](https://github.com/pytroll/satpy/pull/1473) - Fix '::' erroneous for dicts syntax in docstrings ([1453](https://github.com/pytroll/satpy/issues/1453), [1453](https://github.com/pytroll/satpy/issues/1453))
* [PR 1466](https://github.com/pytroll/satpy/pull/1466) - Fix wavelength range print out to use regular nbsp ([1449](https://github.com/pytroll/satpy/issues/1449))
* [PR 1447](https://github.com/pytroll/satpy/pull/1447) - Fix handling of modifiers in satpy-cf reader

#### Features added

* [PR 1485](https://github.com/pytroll/satpy/pull/1485) - Harmonise AreaDefinition namings in EUM geos readers and sort geos areas in areas.yaml ([1187](https://github.com/pytroll/satpy/issues/1187))
* [PR 1478](https://github.com/pytroll/satpy/pull/1478) - Improve FCI geolocation computation, harmonize area_id, add geolocation tests
* [PR 1476](https://github.com/pytroll/satpy/pull/1476) - Add support for multiple values in the DecisionTree used for enhancements
* [PR 1474](https://github.com/pytroll/satpy/pull/1474) - Fix EUMGACFDR reader so that all datasets can be read.
* [PR 1465](https://github.com/pytroll/satpy/pull/1465) - Updates to FCI reader to include CT, CTTH, GII and the latest filenam…
* [PR 1457](https://github.com/pytroll/satpy/pull/1457) - Harmonize calibration in SEVIRI readers
* [PR 1442](https://github.com/pytroll/satpy/pull/1442) - Switch ci coverage to xml for codecov compatibility
* [PR 1441](https://github.com/pytroll/satpy/pull/1441) - Add github workflow
* [PR 1439](https://github.com/pytroll/satpy/pull/1439) - Add support for s3 buckets in OLCI and ABI l1 readers
* [PR 1438](https://github.com/pytroll/satpy/pull/1438) - Full disk padding feature for SEVIRI Native data ([1440](https://github.com/pytroll/satpy/issues/1440))
* [PR 1427](https://github.com/pytroll/satpy/pull/1427) - Add reader for FIDUCEO MVIRI FCDR data
* [PR 1421](https://github.com/pytroll/satpy/pull/1421) - Add reader for AMSR2 Level 2 data produced by GAASP software (amsr2_l2_gaasp)
* [PR 1402](https://github.com/pytroll/satpy/pull/1402) - Add ability to create complex tiled AWIPS NetCDF files (formerly SCMI writer)
* [PR 1393](https://github.com/pytroll/satpy/pull/1393) - Fix sar-c calibration and add support for dB units
* [PR 1380](https://github.com/pytroll/satpy/pull/1380) - Add arbitrary filename suffix to ABI L1B reader
* [PR 1351](https://github.com/pytroll/satpy/pull/1351) - Refactor Scene loading and dependency tree ([1443](https://github.com/pytroll/satpy/issues/1443))
* [PR 937](https://github.com/pytroll/satpy/pull/937) - Add GLM + ABI highlight composite

#### Documentation changes

* [PR 1473](https://github.com/pytroll/satpy/pull/1473) - Fix '::' erroneous for dicts syntax in docstrings ([1453](https://github.com/pytroll/satpy/issues/1453), [1453](https://github.com/pytroll/satpy/issues/1453))
* [PR 1448](https://github.com/pytroll/satpy/pull/1448) - DOC: add explanation to the way x and y work in aggregate

#### Refactoring

* [PR 1402](https://github.com/pytroll/satpy/pull/1402) - Add ability to create complex tiled AWIPS NetCDF files (formerly SCMI writer)
* [PR 1351](https://github.com/pytroll/satpy/pull/1351) - Refactor Scene loading and dependency tree ([1443](https://github.com/pytroll/satpy/issues/1443))

In this release 26 pull requests were closed.


## Version 0.24.0 (2020/11/16)

### Issues Closed

* [Issue 1412](https://github.com/pytroll/satpy/issues/1412) - Mimic reader fails when multiple times are provided to Scene object
* [Issue 1409](https://github.com/pytroll/satpy/issues/1409) - "Unexpected number of scanlines!" when reading AVHRR GAC data
* [Issue 1399](https://github.com/pytroll/satpy/issues/1399) - Customes Scene creation from MultiScene.from_files ([PR 1400](https://github.com/pytroll/satpy/pull/1400))
* [Issue 1396](https://github.com/pytroll/satpy/issues/1396) - reader_kwargs should differentiate between different readers ([PR 1397](https://github.com/pytroll/satpy/pull/1397))
* [Issue 1389](https://github.com/pytroll/satpy/issues/1389) - Can't load angle data from msi_safe in version 0.23 ([PR 1391](https://github.com/pytroll/satpy/pull/1391))
* [Issue 1387](https://github.com/pytroll/satpy/issues/1387) - NUCAPS time format of data from CLASS ([PR 1388](https://github.com/pytroll/satpy/pull/1388))
* [Issue 1371](https://github.com/pytroll/satpy/issues/1371) - MIMIC reader available_dataset_names returns 1d lat/lon fields ([PR 1392](https://github.com/pytroll/satpy/pull/1392))
* [Issue 1343](https://github.com/pytroll/satpy/issues/1343) - Feature Request: available_readers to return alphabetical order
* [Issue 1224](https://github.com/pytroll/satpy/issues/1224) - GRIB-2/ICON geolocation unknown or invalid for western hemisphere ([PR 1296](https://github.com/pytroll/satpy/pull/1296))

In this release 9 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 1435](https://github.com/pytroll/satpy/pull/1435) - Fix tests for GEOFlippableFileYAMLReader after AreaDefinition.area_extent being immutable
* [PR 1433](https://github.com/pytroll/satpy/pull/1433) - Fix cloud-free pixels in cloudtop height composite
* [PR 1432](https://github.com/pytroll/satpy/pull/1432) - Fix enhance2dataset to support P-mode datasets
* [PR 1431](https://github.com/pytroll/satpy/pull/1431) - Fix crash when TLE files are missing
* [PR 1430](https://github.com/pytroll/satpy/pull/1430) - Fix infer_mode not using the band coordinate
* [PR 1428](https://github.com/pytroll/satpy/pull/1428) - Bugfix NWC SAF GEO v2016 area definition
* [PR 1422](https://github.com/pytroll/satpy/pull/1422) - Fix HDF5 utility file handler not decoding byte arrays consistently
* [PR 1413](https://github.com/pytroll/satpy/pull/1413) - Fix pyspectral link in the main doc page
* [PR 1407](https://github.com/pytroll/satpy/pull/1407) - Fix mersi 2 angles reading
* [PR 1392](https://github.com/pytroll/satpy/pull/1392) - Remove 1-D lat/lon variables from mimic reader's available datasets ([1371](https://github.com/pytroll/satpy/issues/1371))
* [PR 1391](https://github.com/pytroll/satpy/pull/1391) - Fix the MSI / Sentinel-2 reader so it uses new DataID ([1389](https://github.com/pytroll/satpy/issues/1389))
* [PR 1388](https://github.com/pytroll/satpy/pull/1388) - Fix handling of new date string formats in NUCAPS reader ([1387](https://github.com/pytroll/satpy/issues/1387))
* [PR 1382](https://github.com/pytroll/satpy/pull/1382) - Fixed bug getting name to the calibration in mitiff writer
* [PR 1296](https://github.com/pytroll/satpy/pull/1296) - Fix grib reader handling for data on 0-360 longitude ([1224](https://github.com/pytroll/satpy/issues/1224))

#### Features added

* [PR 1420](https://github.com/pytroll/satpy/pull/1420) - Add support for Near-realtime VIIRS L1b data.
* [PR 1411](https://github.com/pytroll/satpy/pull/1411) - Added MERSI-2 file pattern for data from NMSC
* [PR 1406](https://github.com/pytroll/satpy/pull/1406) - Handle bilinear caching in Pyresample
* [PR 1405](https://github.com/pytroll/satpy/pull/1405) - Add FIR product to seviri_l2_grib reader
* [PR 1401](https://github.com/pytroll/satpy/pull/1401) - Add function to the SLSTR L1 reader to enable correction of VIS radiances.
* [PR 1400](https://github.com/pytroll/satpy/pull/1400) - Improve customisation in multiscene creation ([1399](https://github.com/pytroll/satpy/issues/1399))
* [PR 1397](https://github.com/pytroll/satpy/pull/1397) - Allow different kwargs for different readers ([1396](https://github.com/pytroll/satpy/issues/1396))
* [PR 1394](https://github.com/pytroll/satpy/pull/1394) - Add satpy cf-reader and eumetsat gac reader ([1205](https://github.com/pytroll/satpy/issues/1205))
* [PR 1390](https://github.com/pytroll/satpy/pull/1390) - Add support to Pyspectral NIRReflectance masking limit
* [PR 1378](https://github.com/pytroll/satpy/pull/1378) - Alphabetize available_readers method and update documentation

#### Documentation changes

* [PR 1415](https://github.com/pytroll/satpy/pull/1415) - Update Code of Conduct contact email to groups.io address
* [PR 1413](https://github.com/pytroll/satpy/pull/1413) - Fix pyspectral link in the main doc page
* [PR 1374](https://github.com/pytroll/satpy/pull/1374) - DOC: add conda-forge badge

#### Backward incompatible changes

* [PR 1360](https://github.com/pytroll/satpy/pull/1360) - Create new ModifierBase class and move existing modifiers to satpy.modifiers

#### Refactoring

* [PR 1360](https://github.com/pytroll/satpy/pull/1360) - Create new ModifierBase class and move existing modifiers to satpy.modifiers

In this release 29 pull requests were closed.


## Version 0.23.0 (2020/09/18)

### Issues Closed

* [Issue 1372](https://github.com/pytroll/satpy/issues/1372) - fix typo in developer instructions for conda install ([PR 1373](https://github.com/pytroll/satpy/pull/1373))
* [Issue 1367](https://github.com/pytroll/satpy/issues/1367) - AVHRR lat/lon grids incorrect size ([PR 1368](https://github.com/pytroll/satpy/pull/1368))
* [Issue 1355](https://github.com/pytroll/satpy/issues/1355) - ir product
* [Issue 1350](https://github.com/pytroll/satpy/issues/1350) - pip install[complete] vs pip install[all]
* [Issue 1344](https://github.com/pytroll/satpy/issues/1344) - scn.load('C01') gives - TypeError
* [Issue 1339](https://github.com/pytroll/satpy/issues/1339) - hrv composites for global scene
* [Issue 1336](https://github.com/pytroll/satpy/issues/1336) - Problem with making MODIS L1 images
* [Issue 1334](https://github.com/pytroll/satpy/issues/1334) - SEVIRI reader doesn't include Earth-Sun distance in the rad->refl calibration ([PR 1341](https://github.com/pytroll/satpy/pull/1341))
* [Issue 1330](https://github.com/pytroll/satpy/issues/1330) - AAPP AVHRR level 1 reader raises a Value error when a channel is missing ([PR 1333](https://github.com/pytroll/satpy/pull/1333))
* [Issue 1292](https://github.com/pytroll/satpy/issues/1292) - Feature Request: update to Quickstart to use data from the demo module
* [Issue 1291](https://github.com/pytroll/satpy/issues/1291) - get_us_midlatitude_cyclone_abi in satpy.demo fails  ([PR 1295](https://github.com/pytroll/satpy/pull/1295))
* [Issue 1289](https://github.com/pytroll/satpy/issues/1289) - update _makedirs in satpy.demo ([PR 1295](https://github.com/pytroll/satpy/pull/1295))
* [Issue 1279](https://github.com/pytroll/satpy/issues/1279) - MultiScene.blend(blend_function=timeseries) results in incorrect start_time, end_time
* [Issue 1278](https://github.com/pytroll/satpy/issues/1278) - Trying to get Earth's semimajor and semiminor axis size from HRIT files
* [Issue 1271](https://github.com/pytroll/satpy/issues/1271) - Test failures in MERSI and VIIRS readers after fixing bugs in test routines ([PR 1270](https://github.com/pytroll/satpy/pull/1270))
* [Issue 1268](https://github.com/pytroll/satpy/issues/1268) - Support multiple readers in MultiScene.from_files ([PR 1269](https://github.com/pytroll/satpy/pull/1269))
* [Issue 1261](https://github.com/pytroll/satpy/issues/1261) - Reading the SEVIRI HRV channel with seviri_l1b_native returns a numpy array ([PR 1272](https://github.com/pytroll/satpy/pull/1272))
* [Issue 1258](https://github.com/pytroll/satpy/issues/1258) - Saving true color GOES image requires double-resampling if calibration='radiance' ([PR 1088](https://github.com/pytroll/satpy/pull/1088))
* [Issue 1252](https://github.com/pytroll/satpy/issues/1252) - Incorrect error message when calibration key unknown
* [Issue 1243](https://github.com/pytroll/satpy/issues/1243) - Wrong data type of orbital_parameters in FY4A AGRI reader ([PR 1244](https://github.com/pytroll/satpy/pull/1244))
* [Issue 1191](https://github.com/pytroll/satpy/issues/1191) - cf_writer should append to Convention global attribute if given header_attr ([PR 1204](https://github.com/pytroll/satpy/pull/1204))
* [Issue 1149](https://github.com/pytroll/satpy/issues/1149) - GLM data LCFA from Class
* [Issue 299](https://github.com/pytroll/satpy/issues/299) - Missing HRV-channel StackedAreaDefinition for native_msg-reader

In this release 23 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 1368](https://github.com/pytroll/satpy/pull/1368) - Fix wrong number of scanlines in eps reader ([1367](https://github.com/pytroll/satpy/issues/1367))
* [PR 1366](https://github.com/pytroll/satpy/pull/1366) - Fixing a few typos in slstr_l1b yaml reader
* [PR 1365](https://github.com/pytroll/satpy/pull/1365) - Fix leftovers from module splitting
* [PR 1358](https://github.com/pytroll/satpy/pull/1358) - Daskify Earth-Sun distance correction.
* [PR 1357](https://github.com/pytroll/satpy/pull/1357) - Only add longitude/latitude variables in cf_writer if they are not included already.
* [PR 1354](https://github.com/pytroll/satpy/pull/1354) - Update name for gridded AHI reader
* [PR 1353](https://github.com/pytroll/satpy/pull/1353) - Add_band workaround for dask bug
* [PR 1341](https://github.com/pytroll/satpy/pull/1341) - Add Sun-Earth distance corrector utility and apply in SEVIRI readers ([1334](https://github.com/pytroll/satpy/issues/1334))
* [PR 1338](https://github.com/pytroll/satpy/pull/1338) - Fix exception to catch when new namedtuple syntax is used
* [PR 1333](https://github.com/pytroll/satpy/pull/1333) - Fix aapp_l1b reader to behave nicely on missing datasets ([1330](https://github.com/pytroll/satpy/issues/1330))
* [PR 1320](https://github.com/pytroll/satpy/pull/1320) - Fix 'viirs_sdr' reader not scaling DNB data properly
* [PR 1319](https://github.com/pytroll/satpy/pull/1319) - Fix NIRReflectance passing None as sunz_threshold
* [PR 1318](https://github.com/pytroll/satpy/pull/1318) - Fix time extraction from filenames in yaml for SEVIRI Native and NetCDF readers
* [PR 1315](https://github.com/pytroll/satpy/pull/1315) - Fix tests on i386
* [PR 1313](https://github.com/pytroll/satpy/pull/1313) - Fix true colors generation for AHI HSD data and refactor the dep tree code
* [PR 1311](https://github.com/pytroll/satpy/pull/1311) - Make colorize compositor dask-compatible
* [PR 1309](https://github.com/pytroll/satpy/pull/1309) - Refactor the combine_metadata function and allow numpy arrays to be combined
* [PR 1303](https://github.com/pytroll/satpy/pull/1303) - Fix nucaps reader failing when kwargs are passed
* [PR 1302](https://github.com/pytroll/satpy/pull/1302) - Fix numpy scalars considered arrays in combine_metadata
* [PR 1295](https://github.com/pytroll/satpy/pull/1295) - Fix ABI mid-latitude cyclone demo downloading wrong number of files ([1291](https://github.com/pytroll/satpy/issues/1291), [1289](https://github.com/pytroll/satpy/issues/1289))
* [PR 1262](https://github.com/pytroll/satpy/pull/1262) - Fix handling of HRV channel navigation for RSS data in seviri_l1b_native reader
* [PR 1259](https://github.com/pytroll/satpy/pull/1259) - Update safe_msi for new pyproj compatibility
* [PR 1247](https://github.com/pytroll/satpy/pull/1247) - Fix time reading in vaisala_gld360 reader

#### Features added

* [PR 1352](https://github.com/pytroll/satpy/pull/1352) - Reintroduce support for pyproj 1.9.6 in cf_writer
* [PR 1342](https://github.com/pytroll/satpy/pull/1342) - Update seviri icare tests
* [PR 1327](https://github.com/pytroll/satpy/pull/1327) - Refactor reader configuration loading to remove redundant code
* [PR 1312](https://github.com/pytroll/satpy/pull/1312) - Add reader for gridded AHI data
* [PR 1304](https://github.com/pytroll/satpy/pull/1304) - DOC: add create vm instructions
* [PR 1294](https://github.com/pytroll/satpy/pull/1294) - Add ability to supply radiance correction coefficients to AHI HSD and AMI readers
* [PR 1284](https://github.com/pytroll/satpy/pull/1284) - add more RGB to FY4A
* [PR 1269](https://github.com/pytroll/satpy/pull/1269) - Support multiple readers in group_files and MultiScene.from_files ([1268](https://github.com/pytroll/satpy/issues/1268))
* [PR 1263](https://github.com/pytroll/satpy/pull/1263) - Add generic filepatterns for mersi2 reader
* [PR 1257](https://github.com/pytroll/satpy/pull/1257) - Add per-frame decoration to MultiScene ([1257](https://github.com/pytroll/satpy/issues/1257))
* [PR 1255](https://github.com/pytroll/satpy/pull/1255) - Add test utility to make a scene.
* [PR 1254](https://github.com/pytroll/satpy/pull/1254) - Preserve chunks in CF Writer
* [PR 1251](https://github.com/pytroll/satpy/pull/1251) - Add ABI Fire Temperature, Day Convection, and Cloud Type composites.
* [PR 1241](https://github.com/pytroll/satpy/pull/1241) - Add environment variables handeling to static image compositor
* [PR 1237](https://github.com/pytroll/satpy/pull/1237) - More flexible way of passing avhrr_l1b_gaclac reader kwargs to pygac
* [PR 1204](https://github.com/pytroll/satpy/pull/1204) - Alter the way cf_writer handle hardcoded global attributes ([1191](https://github.com/pytroll/satpy/issues/1191))
* [PR 1088](https://github.com/pytroll/satpy/pull/1088) - Make the metadata keys that uniquely identify a DataArray (DataID) configurable per reader ([1258](https://github.com/pytroll/satpy/issues/1258))
* [PR 564](https://github.com/pytroll/satpy/pull/564) - Add new ABI composites

#### Documentation changes

* [PR 1373](https://github.com/pytroll/satpy/pull/1373) - Fix word order error in conda install instructions ([1372](https://github.com/pytroll/satpy/issues/1372))
* [PR 1346](https://github.com/pytroll/satpy/pull/1346) - DOC: put pip install with extra dependency in quotation
* [PR 1332](https://github.com/pytroll/satpy/pull/1332) - Remove reference to datasetid in tests.utils.
* [PR 1331](https://github.com/pytroll/satpy/pull/1331) - Fix auxiliary files for releasing and pr template
* [PR 1325](https://github.com/pytroll/satpy/pull/1325) - Use nbviewer for linking notebooks.
* [PR 1317](https://github.com/pytroll/satpy/pull/1317) - Fix typo in variable names in resample documentation
* [PR 1314](https://github.com/pytroll/satpy/pull/1314) - Remove use of YAML Anchors for easier understanding
* [PR 1304](https://github.com/pytroll/satpy/pull/1304) - DOC: add create vm instructions
* [PR 1264](https://github.com/pytroll/satpy/pull/1264) - Fix "see above" reference at start of enhance docs
* [PR 1088](https://github.com/pytroll/satpy/pull/1088) - Make the metadata keys that uniquely identify a DataArray (DataID) configurable per reader ([1258](https://github.com/pytroll/satpy/issues/1258))

#### Backward incompatible changes

* [PR 1327](https://github.com/pytroll/satpy/pull/1327) - Refactor reader configuration loading to remove redundant code
* [PR 1300](https://github.com/pytroll/satpy/pull/1300) - Refactor scene to privatize some attributes and methods

#### Refactoring

* [PR 1341](https://github.com/pytroll/satpy/pull/1341) - Add Sun-Earth distance corrector utility and apply in SEVIRI readers ([1334](https://github.com/pytroll/satpy/issues/1334))
* [PR 1327](https://github.com/pytroll/satpy/pull/1327) - Refactor reader configuration loading to remove redundant code
* [PR 1313](https://github.com/pytroll/satpy/pull/1313) - Fix true colors generation for AHI HSD data and refactor the dep tree code
* [PR 1309](https://github.com/pytroll/satpy/pull/1309) - Refactor the combine_metadata function and allow numpy arrays to be combined
* [PR 1301](https://github.com/pytroll/satpy/pull/1301) - Split DependencyTree from Node and DatasetDict
* [PR 1300](https://github.com/pytroll/satpy/pull/1300) - Refactor scene to privatize some attributes and methods
* [PR 1088](https://github.com/pytroll/satpy/pull/1088) - Make the metadata keys that uniquely identify a DataArray (DataID) configurable per reader ([1258](https://github.com/pytroll/satpy/issues/1258))

In this release 60 pull requests were closed.


## Version 0.22.0 (2020/06/10)

### Issues Closed

* [Issue 1232](https://github.com/pytroll/satpy/issues/1232) - Add link to documentation for VII L1b-reader. ([PR 1236](https://github.com/pytroll/satpy/pull/1236))
* [Issue 1229](https://github.com/pytroll/satpy/issues/1229) - FCI reader can read pixel_quality flags only after reading corresponding channel data ([PR 1230](https://github.com/pytroll/satpy/pull/1230))
* [Issue 1215](https://github.com/pytroll/satpy/issues/1215) - FCI reader fails to load composites due to metadata issues ([PR 1216](https://github.com/pytroll/satpy/pull/1216))
* [Issue 1201](https://github.com/pytroll/satpy/issues/1201) - Incorrect error message when some but not all readers found ([PR 1202](https://github.com/pytroll/satpy/pull/1202))
* [Issue 1198](https://github.com/pytroll/satpy/issues/1198) - Let NetCDF4FileHandler cache variable dimension names ([PR 1199](https://github.com/pytroll/satpy/pull/1199))
* [Issue 1190](https://github.com/pytroll/satpy/issues/1190) - Unknown dataset, solar_zenith_angle
* [Issue 1172](https://github.com/pytroll/satpy/issues/1172) - find_files_and_readers is slow ([PR 1178](https://github.com/pytroll/satpy/pull/1178))
* [Issue 1171](https://github.com/pytroll/satpy/issues/1171) - Add reading of pixel_quality variable to FCI FDHSI reader ([PR 1177](https://github.com/pytroll/satpy/pull/1177))
* [Issue 1168](https://github.com/pytroll/satpy/issues/1168) - Add more versatile options for masking datasets ([PR 1175](https://github.com/pytroll/satpy/pull/1175))
* [Issue 1167](https://github.com/pytroll/satpy/issues/1167) - saving sentinel-2 image as jpg
* [Issue 1164](https://github.com/pytroll/satpy/issues/1164) - Question about license
* [Issue 1162](https://github.com/pytroll/satpy/issues/1162) - abi_l2_nc reader unable to read MCMIP files
* [Issue 1156](https://github.com/pytroll/satpy/issues/1156) - dealing with 1D array output from data assimilation
* [Issue 1154](https://github.com/pytroll/satpy/issues/1154) - MERSI-2 250meters corrected refl.
* [Issue 1153](https://github.com/pytroll/satpy/issues/1153) - tropomi reader: scene attributes and data array attributes are different ([PR 1155](https://github.com/pytroll/satpy/pull/1155))
* [Issue 1151](https://github.com/pytroll/satpy/issues/1151) - amsr2 l1b reader also match amsr2 l2 products ([PR 1152](https://github.com/pytroll/satpy/pull/1152))
* [Issue 1144](https://github.com/pytroll/satpy/issues/1144) - Documentation bug: group_files keyword argument reader doc has sentence consisting of only the word "This" ([PR 1147](https://github.com/pytroll/satpy/pull/1147))
* [Issue 1143](https://github.com/pytroll/satpy/issues/1143) - save_datasets doesn't work for tropomi_l2 data ([PR 1139](https://github.com/pytroll/satpy/pull/1139))
* [Issue 1132](https://github.com/pytroll/satpy/issues/1132) - Add area definitions for the FCI FDHSI L1c grids ([PR 1188](https://github.com/pytroll/satpy/pull/1188))
* [Issue 1050](https://github.com/pytroll/satpy/issues/1050) - Return counts from avhrr_l1b_gaclac reader  ([PR 1051](https://github.com/pytroll/satpy/pull/1051))
* [Issue 1014](https://github.com/pytroll/satpy/issues/1014) - The fci_l1c_fdhsi reader should proved the `platform_name` in the attributes ([PR 1176](https://github.com/pytroll/satpy/pull/1176))
* [Issue 958](https://github.com/pytroll/satpy/issues/958) - Add a CMSAF reader ([PR 720](https://github.com/pytroll/satpy/pull/720))
* [Issue 680](https://github.com/pytroll/satpy/issues/680) - Expose `overviews` from Trollimage for saving (geo)tiff images

In this release 23 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 1230](https://github.com/pytroll/satpy/pull/1230) - FCI: fix areadef when only pixel quality asked ([1229](https://github.com/pytroll/satpy/issues/1229), [1229](https://github.com/pytroll/satpy/issues/1229))
* [PR 1216](https://github.com/pytroll/satpy/pull/1216) - Make combine_arrays understand non-numpy arrays ([1215](https://github.com/pytroll/satpy/issues/1215), [1215](https://github.com/pytroll/satpy/issues/1215))
* [PR 1213](https://github.com/pytroll/satpy/pull/1213) - Remove invalid valid_range metadata from abi readers
* [PR 1211](https://github.com/pytroll/satpy/pull/1211) - Fix "rows_per_scan" not being available from VIIRS SDR readers
* [PR 1202](https://github.com/pytroll/satpy/pull/1202) - Fix bad error message when Scene was given a bad reader name ([1201](https://github.com/pytroll/satpy/issues/1201))
* [PR 1195](https://github.com/pytroll/satpy/pull/1195) - Fix accessing uncached root group variable when using NetCDF4FileHandler in caching mode ([1195](https://github.com/pytroll/satpy/issues/1195))
* [PR 1170](https://github.com/pytroll/satpy/pull/1170) - Fix cf writing of 3d arrays
* [PR 1155](https://github.com/pytroll/satpy/pull/1155) - Lowercase sensor of tropomi_l2 ([1153](https://github.com/pytroll/satpy/issues/1153))
* [PR 1139](https://github.com/pytroll/satpy/pull/1139) - Keep int type and fix scale_factor/dim bug in tropomi_l2 reader ([1143](https://github.com/pytroll/satpy/issues/1143))

#### Features added

* [PR 1227](https://github.com/pytroll/satpy/pull/1227) - Delete kdtree after saving cache
* [PR 1226](https://github.com/pytroll/satpy/pull/1226) - Add a feature for handling scheduled_time in ahi_hsd reader.
* [PR 1219](https://github.com/pytroll/satpy/pull/1219) - Add VII L2 netCDF-reader.
* [PR 1218](https://github.com/pytroll/satpy/pull/1218) - Add VII L1b netCDF-reader.
* [PR 1212](https://github.com/pytroll/satpy/pull/1212) - Add file pattern for NWCSAF input file names to 'grib' reader ([1212](https://github.com/pytroll/satpy/issues/1212))
* [PR 1199](https://github.com/pytroll/satpy/pull/1199) - Cache dimension per variable ([1198](https://github.com/pytroll/satpy/issues/1198))
* [PR 1189](https://github.com/pytroll/satpy/pull/1189) - Add option to supply sunz-threshold applied in Pyspectral
* [PR 1188](https://github.com/pytroll/satpy/pull/1188) - Add areas for FCI ([1132](https://github.com/pytroll/satpy/issues/1132))
* [PR 1186](https://github.com/pytroll/satpy/pull/1186) - Fix SEVIRI native reader flipping
* [PR 1185](https://github.com/pytroll/satpy/pull/1185) - Add scanline acquisition times to hrit_jma
* [PR 1183](https://github.com/pytroll/satpy/pull/1183) - Add options for creating geotiff overviews
* [PR 1181](https://github.com/pytroll/satpy/pull/1181) - Add more explicit error message when string is passed to Scene.load
* [PR 1180](https://github.com/pytroll/satpy/pull/1180) - Migrate FCI tests to pytest
* [PR 1178](https://github.com/pytroll/satpy/pull/1178) - Optimize readers searching for matching filenames ([1172](https://github.com/pytroll/satpy/issues/1172))
* [PR 1177](https://github.com/pytroll/satpy/pull/1177) - Add support for reading pixel_quality ancillary variables, FCI reader no longer logs warnings ([1171](https://github.com/pytroll/satpy/issues/1171))
* [PR 1176](https://github.com/pytroll/satpy/pull/1176) - Provide platform_name in FCI L1C FDHSI reader. ([1014](https://github.com/pytroll/satpy/issues/1014))
* [PR 1175](https://github.com/pytroll/satpy/pull/1175) - Add more flexible masking ([1168](https://github.com/pytroll/satpy/issues/1168))
* [PR 1173](https://github.com/pytroll/satpy/pull/1173) - Check whether time dimension exists for timeseries
* [PR 1169](https://github.com/pytroll/satpy/pull/1169) - Implement remote file search
* [PR 1165](https://github.com/pytroll/satpy/pull/1165) - Add missing_ok option to find_files_and_readers ([1165](https://github.com/pytroll/satpy/issues/1165))
* [PR 1163](https://github.com/pytroll/satpy/pull/1163) - Add TROPOMI NO2 LEVEL2 composites
* [PR 1161](https://github.com/pytroll/satpy/pull/1161) - Add Effective_Pressure to NUCAPS reader
* [PR 1152](https://github.com/pytroll/satpy/pull/1152) - amsr2 reader for l2 ssw product ([1151](https://github.com/pytroll/satpy/issues/1151))
* [PR 1142](https://github.com/pytroll/satpy/pull/1142) - add filepatterns S-HSAF-h03B and S-HSAF-h05B to hsaf_grib.yaml
* [PR 1141](https://github.com/pytroll/satpy/pull/1141) - Add night lights composites for ABI, AHI and AMI
* [PR 1135](https://github.com/pytroll/satpy/pull/1135) - Fix reflectance and BT calibration in FCI FDHSI reader
* [PR 1100](https://github.com/pytroll/satpy/pull/1100) - Add support for GPM IMERG data
* [PR 1051](https://github.com/pytroll/satpy/pull/1051) - Return counts from satpy/avhrr_l1b_gaclac reader ([1050](https://github.com/pytroll/satpy/issues/1050))
* [PR 983](https://github.com/pytroll/satpy/pull/983) - Add group method to MultiScene
* [PR 812](https://github.com/pytroll/satpy/pull/812) - Add MOD06 support to 'modis_l2' reader ([1200](https://github.com/pytroll/satpy/issues/1200))
* [PR 720](https://github.com/pytroll/satpy/pull/720) - CMSAF CLAAS v2. reader ([958](https://github.com/pytroll/satpy/issues/958))

#### Documentation changes

* [PR 1223](https://github.com/pytroll/satpy/pull/1223) - Add FCI Natural Color example page to sphinx docs
* [PR 1203](https://github.com/pytroll/satpy/pull/1203) - Add link to MTSAT sample data
* [PR 1147](https://github.com/pytroll/satpy/pull/1147) - Fix incomplete group_files docstring ([1144](https://github.com/pytroll/satpy/issues/1144))

In this release 43 pull requests were closed.


## Version 0.21.0 (2020/04/06)

### Issues Closed

* [Issue 1124](https://github.com/pytroll/satpy/issues/1124) - Crop scene of visual spectrum of the sentinel 2 satellite ([PR 1125](https://github.com/pytroll/satpy/pull/1125))
* [Issue 1112](https://github.com/pytroll/satpy/issues/1112) - Loading both abi and nwcsaf-geo confuses satpy into sometimes trying the wrong composite ([PR 1113](https://github.com/pytroll/satpy/pull/1113))
* [Issue 1096](https://github.com/pytroll/satpy/issues/1096) - Saving an image with NinjoTIFFWriter is broken in satpy v.0.20.0 ([PR 1098](https://github.com/pytroll/satpy/pull/1098))
* [Issue 1092](https://github.com/pytroll/satpy/issues/1092) - Avhrr l1b eps reader changes values of angles after reading ([PR 1101](https://github.com/pytroll/satpy/pull/1101))
* [Issue 1087](https://github.com/pytroll/satpy/issues/1087) - Saving each scene in a separate image file
* [Issue 1075](https://github.com/pytroll/satpy/issues/1075) - SEVIRI L1b netCDF reader not dask-compliant ([PR 1109](https://github.com/pytroll/satpy/pull/1109))
* [Issue 1059](https://github.com/pytroll/satpy/issues/1059) - test against xarray master ([PR 1095](https://github.com/pytroll/satpy/pull/1095))
* [Issue 1013](https://github.com/pytroll/satpy/issues/1013) - Fails to load solar_zenith_angle from SLSTR l1b data
* [Issue 883](https://github.com/pytroll/satpy/issues/883) - satpy resample call -> numby.ndarray deepcopy error ([PR 1126](https://github.com/pytroll/satpy/pull/1126))
* [Issue 840](https://github.com/pytroll/satpy/issues/840) - MTG-FCI-FDHSI reader has wrong projection ([PR 845](https://github.com/pytroll/satpy/pull/845))
* [Issue 630](https://github.com/pytroll/satpy/issues/630) - Converting hdf5 attributes to string containing h5py.Reference of size 1 causes a AttributeError ([PR 1126](https://github.com/pytroll/satpy/pull/1126))

In this release 11 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 1131](https://github.com/pytroll/satpy/pull/1131) - Fix geostationary utilities assuming a/b radii are always available
* [PR 1129](https://github.com/pytroll/satpy/pull/1129) - Make the viirs_sdr reader return float32s
* [PR 1125](https://github.com/pytroll/satpy/pull/1125) - Fix Scene.crop using PROJ definition to create target area definition ([1124](https://github.com/pytroll/satpy/issues/1124))
* [PR 1118](https://github.com/pytroll/satpy/pull/1118) - Fix supported Python version in devguide
* [PR 1116](https://github.com/pytroll/satpy/pull/1116) - Make an alias for the snow composite in viirs
* [PR 1115](https://github.com/pytroll/satpy/pull/1115) - Fix mitiff writer to support sensors as a set
* [PR 1113](https://github.com/pytroll/satpy/pull/1113) - Add sensor-name property to NWCSAF readers ([1112](https://github.com/pytroll/satpy/issues/1112), [1111](https://github.com/pytroll/satpy/issues/1111))
* [PR 1107](https://github.com/pytroll/satpy/pull/1107) - Raise an error if data and angle shapes don't match in NIRReflectance
* [PR 1106](https://github.com/pytroll/satpy/pull/1106) - Scale valid range if available.
* [PR 1101](https://github.com/pytroll/satpy/pull/1101) - Fix eps l1b angles computation returning non deterministic results ([1092](https://github.com/pytroll/satpy/issues/1092))
* [PR 1098](https://github.com/pytroll/satpy/pull/1098) - Fix ninjotiff writer tests failing when pyninjotiff is installed ([1096](https://github.com/pytroll/satpy/issues/1096))
* [PR 1089](https://github.com/pytroll/satpy/pull/1089) - Make sunz correction use available sunz dataset
* [PR 1038](https://github.com/pytroll/satpy/pull/1038) - Switch to pyproj for projection to CF NetCDF grid mapping ([1029](https://github.com/pytroll/satpy/issues/1029), [1029](https://github.com/pytroll/satpy/issues/1029))

#### Features added

* [PR 1128](https://github.com/pytroll/satpy/pull/1128) - Add tm5_constant_a and tm5_constant_b for tropomi_l2
* [PR 1126](https://github.com/pytroll/satpy/pull/1126) - Update omps edr reader and hdf5_utils to handle OMPS SO2 data from FMI ([883](https://github.com/pytroll/satpy/issues/883), [630](https://github.com/pytroll/satpy/issues/630))
* [PR 1121](https://github.com/pytroll/satpy/pull/1121) - HY-2B scatterometer l2b hdf5 reader
* [PR 1117](https://github.com/pytroll/satpy/pull/1117) - Add support for satpy.composites entry points
* [PR 1113](https://github.com/pytroll/satpy/pull/1113) - Add sensor-name property to NWCSAF readers ([1112](https://github.com/pytroll/satpy/issues/1112), [1111](https://github.com/pytroll/satpy/issues/1111))
* [PR 1109](https://github.com/pytroll/satpy/pull/1109) - Fix dask and attribute issue in seviri_l1b_nc reader ([1075](https://github.com/pytroll/satpy/issues/1075))
* [PR 1095](https://github.com/pytroll/satpy/pull/1095) - Switch to pytest in CI and add unstable dependency environment ([1059](https://github.com/pytroll/satpy/issues/1059))
* [PR 1091](https://github.com/pytroll/satpy/pull/1091) - Add assembled_lat_bounds, assembled_lon_bounds and time variables
* [PR 1071](https://github.com/pytroll/satpy/pull/1071) - Add SEVIRI L2 GRIB reader
* [PR 1044](https://github.com/pytroll/satpy/pull/1044) - Set travis and appveyor numpy version back to 'stable'
* [PR 845](https://github.com/pytroll/satpy/pull/845) - MTG: get projection and extent information from file ([840](https://github.com/pytroll/satpy/issues/840), [840](https://github.com/pytroll/satpy/issues/840))
* [PR 606](https://github.com/pytroll/satpy/pull/606) - Add enhanced (more natural) version of natural colors composite

#### Documentation changes

* [PR 1130](https://github.com/pytroll/satpy/pull/1130) - Add note about datatype in custom reader documentation
* [PR 1118](https://github.com/pytroll/satpy/pull/1118) - Fix supported Python version in devguide


## Version 0.20.0 (2020/02/25)

### Issues Closed

* [Issue 1077](https://github.com/pytroll/satpy/issues/1077) - Tropomi l2 reader needs to handle more filenames ([PR 1078](https://github.com/pytroll/satpy/pull/1078))
* [Issue 1076](https://github.com/pytroll/satpy/issues/1076) - Metop level 2 EUMETCAST BUFR reader ([PR 1079](https://github.com/pytroll/satpy/pull/1079))
* [Issue 1004](https://github.com/pytroll/satpy/issues/1004) - Computing the lons and lats of metop granules from the eps_l1b reader is painfully slow ([PR 1063](https://github.com/pytroll/satpy/pull/1063))
* [Issue 1002](https://github.com/pytroll/satpy/issues/1002) - Resampling of long passes of metop l1b eps data gives strange results
* [Issue 928](https://github.com/pytroll/satpy/issues/928) - Satpy Writer 'geotiff' exists but could not be loaded
* [Issue 924](https://github.com/pytroll/satpy/issues/924) - eps_l1b reader does not accept more than 1 veadr element ([PR 1063](https://github.com/pytroll/satpy/pull/1063))
* [Issue 809](https://github.com/pytroll/satpy/issues/809) - Update avhrr_l1b_aapp reader ([PR 811](https://github.com/pytroll/satpy/pull/811))
* [Issue 112](https://github.com/pytroll/satpy/issues/112) - Python 2 Cruft ([PR 1047](https://github.com/pytroll/satpy/pull/1047))

In this release 8 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 1084](https://github.com/pytroll/satpy/pull/1084) - Add latitude_bounds and longitude_bounds to tropomi_l2
* [PR 1078](https://github.com/pytroll/satpy/pull/1078) - Tropomi l2 reader to handle more types of products ([1077](https://github.com/pytroll/satpy/issues/1077))
* [PR 1072](https://github.com/pytroll/satpy/pull/1072) - Fix the omerc-bb area to use a sphere as ellps
* [PR 1066](https://github.com/pytroll/satpy/pull/1066) - Rename natural_color_sun to natural_color in generic VIS/IR RGB recipes
* [PR 1063](https://github.com/pytroll/satpy/pull/1063) - Fix eps infinite loop ([924](https://github.com/pytroll/satpy/issues/924), [1004](https://github.com/pytroll/satpy/issues/1004))
* [PR 1058](https://github.com/pytroll/satpy/pull/1058) - Work around changes in xarray 0.15
* [PR 1057](https://github.com/pytroll/satpy/pull/1057) - lowercase the sensor name
* [PR 1055](https://github.com/pytroll/satpy/pull/1055) - Fix sst standard name
* [PR 1049](https://github.com/pytroll/satpy/pull/1049) - Fix handling of paths with forward slashes on Windows
* [PR 1048](https://github.com/pytroll/satpy/pull/1048) - Fix AMI L1b reader incorrectly grouping files
* [PR 1045](https://github.com/pytroll/satpy/pull/1045) - Update hrpt.py for new pygac syntax
* [PR 1043](https://github.com/pytroll/satpy/pull/1043) - Update seviri icare reader that handles differing dataset versions
* [PR 1042](https://github.com/pytroll/satpy/pull/1042) - Replace a unicode hyphen in the glm_l2 reader
* [PR 1041](https://github.com/pytroll/satpy/pull/1041) - Unify Dataset attribute naming in SEVIRI L2 BUFR-reader

#### Features added

* [PR 1082](https://github.com/pytroll/satpy/pull/1082) - Update SLSTR composites
* [PR 1079](https://github.com/pytroll/satpy/pull/1079) - Metop level 2 EUMETCAST BUFR reader ([1076](https://github.com/pytroll/satpy/issues/1076))
* [PR 1067](https://github.com/pytroll/satpy/pull/1067) - Add GOES-17 support to the 'geocat' reader
* [PR 1065](https://github.com/pytroll/satpy/pull/1065) - Add AHI airmass, ash, dust, fog, and night_microphysics RGBs
* [PR 1064](https://github.com/pytroll/satpy/pull/1064) - Adjust default blending in DayNightCompositor
* [PR 1061](https://github.com/pytroll/satpy/pull/1061) - Add support for NUCAPS Science EDRs
* [PR 1052](https://github.com/pytroll/satpy/pull/1052) - Delegate dask delays to pyninjotiff
* [PR 1047](https://github.com/pytroll/satpy/pull/1047) - Remove deprecated abstractproperty usage ([112](https://github.com/pytroll/satpy/issues/112))
* [PR 1020](https://github.com/pytroll/satpy/pull/1020) - Feature Sentinel-3 Level-2 SST
* [PR 988](https://github.com/pytroll/satpy/pull/988) - Remove py27 tests and switch to py38
* [PR 964](https://github.com/pytroll/satpy/pull/964) - Update SEVIRI L2 BUFR reader to handle BUFR products from EUMETSAT Data Centre
* [PR 839](https://github.com/pytroll/satpy/pull/839) - Add support of colorbar
* [PR 811](https://github.com/pytroll/satpy/pull/811) - Daskify and test avhrr_l1b_aapp reader ([809](https://github.com/pytroll/satpy/issues/809))

#### Documentation changes

* [PR 1068](https://github.com/pytroll/satpy/pull/1068) - Fix a typo in writer 'filename' documentation
* [PR 1056](https://github.com/pytroll/satpy/pull/1056) - Fix name of natural_color composite in quickstart

#### Backwards incompatible changes

* [PR 1066](https://github.com/pytroll/satpy/pull/1066) - Rename natural_color_sun to natural_color in generic VIS/IR RGB recipes
* [PR 988](https://github.com/pytroll/satpy/pull/988) - Remove py27 tests and switch to py38

In this release 31 pull requests were closed.


## Version 0.19.1 (2020/01/10)

### Issues Closed

* [Issue 1030](https://github.com/pytroll/satpy/issues/1030) - Geostationary padding results in wrong area definition for AHI mesoscale sectors. ([PR 1037](https://github.com/pytroll/satpy/pull/1037))
* [Issue 1029](https://github.com/pytroll/satpy/issues/1029) - NetCDF (CF) writer doesn't include semi_minor_axis/semi_major_axis for new versions of pyproj ([PR 1040](https://github.com/pytroll/satpy/pull/1040))
* [Issue 1023](https://github.com/pytroll/satpy/issues/1023) - RTD "Edit on Github" broken in "latest" documentation

In this release 3 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 1040](https://github.com/pytroll/satpy/pull/1040) - Fix geostationary axis handling in CF writer ([1029](https://github.com/pytroll/satpy/issues/1029))
* [PR 1037](https://github.com/pytroll/satpy/pull/1037) - Fix segment handling for non-FLDK sectors in the AHI HSD reader ([1030](https://github.com/pytroll/satpy/issues/1030))
* [PR 1036](https://github.com/pytroll/satpy/pull/1036) - Fix ABI L1b/L2 time dimension causing issues with newer xarray
* [PR 1034](https://github.com/pytroll/satpy/pull/1034) - Fix AMI geolocation being off by 1 pixel
* [PR 1033](https://github.com/pytroll/satpy/pull/1033) - Fix avhrr_l1b_aapp reader not including standard_name metadata
* [PR 1031](https://github.com/pytroll/satpy/pull/1031) - Fix tropomi_l2 reader not using y and x dimension names

#### Features added

* [PR 1035](https://github.com/pytroll/satpy/pull/1035) - Add additional Sentinel 3 OLCI 2 datasets
* [PR 1027](https://github.com/pytroll/satpy/pull/1027) - Update SCMI writer and VIIRS EDR Flood reader to work for pre-tiled data

#### Documentation changes

* [PR 1032](https://github.com/pytroll/satpy/pull/1032) - Add documentation about y and x dimensions for custom readers

In this release 9 pull requests were closed.


## Version 0.19.0 (2019/12/30)

### Issues Closed

* [Issue 996](https://github.com/pytroll/satpy/issues/996) - In the sar-c_safe reader, add platform_name to the attribute. ([PR 998](https://github.com/pytroll/satpy/pull/998))
* [Issue 991](https://github.com/pytroll/satpy/issues/991) - Secondary file name patterns aren't used if the first doesn't match
* [Issue 975](https://github.com/pytroll/satpy/issues/975) - Add HRV navigation to `seviri_l1b_native`-reader ([PR 985](https://github.com/pytroll/satpy/pull/985))
* [Issue 972](https://github.com/pytroll/satpy/issues/972) - MTG-FCI-FDHSI reader is slow, apparently not actually dask-aware ([PR 981](https://github.com/pytroll/satpy/pull/981))
* [Issue 970](https://github.com/pytroll/satpy/issues/970) - Pad all geostationary L1 data to full disk area ([PR 977](https://github.com/pytroll/satpy/pull/977))
* [Issue 960](https://github.com/pytroll/satpy/issues/960) - Factorize area def computation in jma_hrit ([PR 978](https://github.com/pytroll/satpy/pull/978))
* [Issue 957](https://github.com/pytroll/satpy/issues/957) - Rayleigh correction in bands l2 of the ABI sensor
* [Issue 954](https://github.com/pytroll/satpy/issues/954) - Mask composites using cloud products ([PR 982](https://github.com/pytroll/satpy/pull/982))
* [Issue 949](https://github.com/pytroll/satpy/issues/949) - Make a common function for geostationnary area_extent computation ([PR 952](https://github.com/pytroll/satpy/pull/952))
* [Issue 807](https://github.com/pytroll/satpy/issues/807) - Add a MIMIC-TPW2 reader ([PR 858](https://github.com/pytroll/satpy/pull/858))
* [Issue 782](https://github.com/pytroll/satpy/issues/782) - Update custom reader documentation to mention coordinates and available datasets ([PR 1019](https://github.com/pytroll/satpy/pull/1019))
* [Issue 486](https://github.com/pytroll/satpy/issues/486) - Add GMS series satellite data reader

In this release 12 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 1021](https://github.com/pytroll/satpy/pull/1021) - Fix padding of segmented geostationary images
* [PR 1010](https://github.com/pytroll/satpy/pull/1010) - Fix missing part in ahi_hrit file pattern
* [PR 1007](https://github.com/pytroll/satpy/pull/1007) - Fix `ahi_hrit` expected segments
* [PR 1006](https://github.com/pytroll/satpy/pull/1006) - Rename standard_name for various readers to be consistent
* [PR 993](https://github.com/pytroll/satpy/pull/993) - Fix VIIRS EDR Flood file patterns not working for AOI files ([243](https://github.com/ssec/polar2grid/issues/243))
* [PR 989](https://github.com/pytroll/satpy/pull/989) - Fix generation of solar and satellite angles when lon/lats are invalid
* [PR 976](https://github.com/pytroll/satpy/pull/976) - CF Writer Improvements
* [PR 974](https://github.com/pytroll/satpy/pull/974) - Fix available_composite_names including night_background static images ([239](https://github.com/ssec/polar2grid/issues/239))
* [PR 969](https://github.com/pytroll/satpy/pull/969) - Fix HDF4 handling of scalar attributes
* [PR 966](https://github.com/pytroll/satpy/pull/966) - Add the fire temperature products to AHI
* [PR 931](https://github.com/pytroll/satpy/pull/931) - Update coord2area_def.py

#### Features added

* [PR 1012](https://github.com/pytroll/satpy/pull/1012) - Implement a small cviirs speedup
* [PR 1011](https://github.com/pytroll/satpy/pull/1011) - Provide only dask arrays to pyspectral's nir reflectance computation
* [PR 1009](https://github.com/pytroll/satpy/pull/1009) - Add support for SEVIRI data from icare
* [PR 1005](https://github.com/pytroll/satpy/pull/1005) - Remove unused reader xslice/yslice keyword arguments
* [PR 1003](https://github.com/pytroll/satpy/pull/1003) - Update copyright header in readers. Add and fix docstrings.
* [PR 998](https://github.com/pytroll/satpy/pull/998) - Add platform name to attributes of sar_c_safe reader ([996](https://github.com/pytroll/satpy/issues/996))
* [PR 997](https://github.com/pytroll/satpy/pull/997) - Add check if prerequisites is used
* [PR 994](https://github.com/pytroll/satpy/pull/994) - Add LAC support to the avhrr-gac-lac reader
* [PR 992](https://github.com/pytroll/satpy/pull/992) - Add hrv_clouds, hrv_fog and natural_with_night_fog composites to seviri.yaml
* [PR 987](https://github.com/pytroll/satpy/pull/987) - scene.aggregate will now handle a SwathDefinition
* [PR 985](https://github.com/pytroll/satpy/pull/985) - Add HRV full disk navigation for `seviri_l1b_native`-reader ([975](https://github.com/pytroll/satpy/issues/975))
* [PR 984](https://github.com/pytroll/satpy/pull/984) - Add on-the-fly decompression to the AHI HSD reader
* [PR 982](https://github.com/pytroll/satpy/pull/982) - Add simple masking compositor ([954](https://github.com/pytroll/satpy/issues/954))
* [PR 981](https://github.com/pytroll/satpy/pull/981) - Optionally cache small data variables and file handles ([972](https://github.com/pytroll/satpy/issues/972))
* [PR 980](https://github.com/pytroll/satpy/pull/980) - Read the meta_data dictionary from pygac
* [PR 978](https://github.com/pytroll/satpy/pull/978) - Factorize area computation in hrit_jma ([960](https://github.com/pytroll/satpy/issues/960))
* [PR 977](https://github.com/pytroll/satpy/pull/977) - Add a YAMLReader to pad segmented geo data ([970](https://github.com/pytroll/satpy/issues/970))
* [PR 976](https://github.com/pytroll/satpy/pull/976) - CF Writer Improvements
* [PR 966](https://github.com/pytroll/satpy/pull/966) - Add the fire temperature products to AHI
* [PR 962](https://github.com/pytroll/satpy/pull/962) - add support for meteo file in OLCI L1B reader
* [PR 961](https://github.com/pytroll/satpy/pull/961) - Fix default radius_of_influence for lon/lat AreaDefintions
* [PR 952](https://github.com/pytroll/satpy/pull/952) - Adds a common function for geostationary projection / area definition calculations ([949](https://github.com/pytroll/satpy/issues/949))
* [PR 920](https://github.com/pytroll/satpy/pull/920) - Transverse Mercator section added in cf writer
* [PR 908](https://github.com/pytroll/satpy/pull/908) - Add interface to pyresample gradient resampler
* [PR 858](https://github.com/pytroll/satpy/pull/858) - Mimic  TPW Reader ([807](https://github.com/pytroll/satpy/issues/807))
* [PR 854](https://github.com/pytroll/satpy/pull/854) - Add GOES-R GLM L2 Gridded product reader and small ABI L1b changes

#### Documentation changes

* [PR 1025](https://github.com/pytroll/satpy/pull/1025) - Switch to configuration file for readthedocs
* [PR 1019](https://github.com/pytroll/satpy/pull/1019) - Add more information about creating custom readers ([782](https://github.com/pytroll/satpy/issues/782))
* [PR 1018](https://github.com/pytroll/satpy/pull/1018) - Add information to Quickstart on basics of getting measurement values and navigation
* [PR 1008](https://github.com/pytroll/satpy/pull/1008) - Add documentation for combine_metadata function
* [PR 1003](https://github.com/pytroll/satpy/pull/1003) - Update copyright header in readers. Add and fix docstrings.
* [PR 1001](https://github.com/pytroll/satpy/pull/1001) - Get travis badge from master branch
* [PR 999](https://github.com/pytroll/satpy/pull/999) - Add FCI L1C reader short and long name metadata
* [PR 968](https://github.com/pytroll/satpy/pull/968) - Add information about multi-threaded compression with geotiff creation

In this release 45 pull requests were closed.


## Version 0.18.1 (2019/11/07)

### Pull Requests Merged

#### Bugs fixed

* [PR 959](https://github.com/pytroll/satpy/pull/959) - Fix `grid` argument handling in overlaying

In this release 1 pull request was closed.


## Version 0.18.0 (2019/11/06)

### Issues Closed

* [Issue 944](https://github.com/pytroll/satpy/issues/944) - Multiple errors when processing OLCI data. ([PR 945](https://github.com/pytroll/satpy/pull/945))
* [Issue 940](https://github.com/pytroll/satpy/issues/940) - Loading of DNB data from VIIRS compact SDR is slow ([PR 941](https://github.com/pytroll/satpy/pull/941))
* [Issue 922](https://github.com/pytroll/satpy/issues/922) - Clarify orbital_parameters metadata ([PR 950](https://github.com/pytroll/satpy/pull/950))
* [Issue 888](https://github.com/pytroll/satpy/issues/888) - Unintended/wrong behaviour of getitem method in HDF5FileHandler? ([PR 886](https://github.com/pytroll/satpy/pull/886))
* [Issue 737](https://github.com/pytroll/satpy/issues/737) - Add reader for GEO-KOMPSAT AMI ([PR 911](https://github.com/pytroll/satpy/pull/911))

In this release 5 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 953](https://github.com/pytroll/satpy/pull/953) - Encode header attributes in CF writer
* [PR 945](https://github.com/pytroll/satpy/pull/945) - Fix bug in OLCI reader that caused multiple error messages to print ([944](https://github.com/pytroll/satpy/issues/944))
* [PR 942](https://github.com/pytroll/satpy/pull/942) - Fix VIIRS EDR Active Fires not assigning a _FillValue to confidence_pct
* [PR 939](https://github.com/pytroll/satpy/pull/939) - Fix MERSI-2 natural_color composite using the wrong band for sharpening
* [PR 938](https://github.com/pytroll/satpy/pull/938) - Fix MultiScene.save_animation to work with new dask.distributed versions
* [PR 914](https://github.com/pytroll/satpy/pull/914) - Cleaning up and adding MERSI-2 RGB composites

#### Features added

* [PR 955](https://github.com/pytroll/satpy/pull/955) - Code clean-up for SEVIRI L2 BUFR-reader
* [PR 953](https://github.com/pytroll/satpy/pull/953) - Encode header attributes in CF writer
* [PR 948](https://github.com/pytroll/satpy/pull/948) - Add the possibility to include scale and offset in geotiffs
* [PR 947](https://github.com/pytroll/satpy/pull/947) - Feature mitiff palette
* [PR 941](https://github.com/pytroll/satpy/pull/941) - Speed up cviirs tiepoint interpolation ([940](https://github.com/pytroll/satpy/issues/940))
* [PR 935](https://github.com/pytroll/satpy/pull/935) - Adapt avhrr_l1b_gaclac to recent pygac changes
* [PR 934](https://github.com/pytroll/satpy/pull/934) - Update add_overlay to make use of the full pycoast capabilities
* [PR 911](https://github.com/pytroll/satpy/pull/911) - Add GK-2A AMI L1B Reader ([737](https://github.com/pytroll/satpy/issues/737))
* [PR 886](https://github.com/pytroll/satpy/pull/886) - Reader for NWCSAF/MSG 2013 format ([888](https://github.com/pytroll/satpy/issues/888))
* [PR 769](https://github.com/pytroll/satpy/pull/769) - Added initial version of an MSG BUFR reader and TOZ product yaml file
* [PR 586](https://github.com/pytroll/satpy/pull/586) - Update handling of reading colormaps from files in enhancements

#### Documentation changes

* [PR 950](https://github.com/pytroll/satpy/pull/950) - Clarify documentation of orbital_parameters metadata ([922](https://github.com/pytroll/satpy/issues/922))
* [PR 943](https://github.com/pytroll/satpy/pull/943) - Fix sphinx docs generation after setuptools_scm migration

In this release 19 pull requests were closed.


## Version 0.17.1 (2019/10/08)

### Issues Closed

* [Issue 918](https://github.com/pytroll/satpy/issues/918) - satpy 0.17 does not work with pyresample 1.11 ([PR 927](https://github.com/pytroll/satpy/pull/927))
* [Issue 902](https://github.com/pytroll/satpy/issues/902) - background compositor with colorized ir_clouds and static image problem ([PR 917](https://github.com/pytroll/satpy/pull/917))
* [Issue 853](https://github.com/pytroll/satpy/issues/853) - scene.available_composite_names() return a composite even the dependency is not fullfilled ([PR 921](https://github.com/pytroll/satpy/pull/921))
* [Issue 830](https://github.com/pytroll/satpy/issues/830) - generic_image reader doesn't read area from .yaml file? ([PR 925](https://github.com/pytroll/satpy/pull/925))

In this release 4 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 925](https://github.com/pytroll/satpy/pull/925) - Fix area handling in StaticImageCompositor ([830](https://github.com/pytroll/satpy/issues/830))
* [PR 923](https://github.com/pytroll/satpy/pull/923) - Make the olci l2 mask a bool array instead of floats
* [PR 921](https://github.com/pytroll/satpy/pull/921) - Fix Scene.available_composite_names showing unavailable composites ([853](https://github.com/pytroll/satpy/issues/853))
* [PR 917](https://github.com/pytroll/satpy/pull/917) - Fix BackgroundCompositor not retaining input metadata ([902](https://github.com/pytroll/satpy/issues/902))

#### Features added

* [PR 927](https://github.com/pytroll/satpy/pull/927) - Fix resampler imports ([918](https://github.com/pytroll/satpy/issues/918))

#### Backwards incompatible changes

* [PR 921](https://github.com/pytroll/satpy/pull/921) - Fix Scene.available_composite_names showing unavailable composites ([853](https://github.com/pytroll/satpy/issues/853))

In this release 6 pull requests were closed.


## Version 0.17.0 (2019/10/01)

### Issues Closed

* [Issue 896](https://github.com/pytroll/satpy/issues/896) - Satpy built-in composite for dust RGB (MSG/SEVIRI data) does not generate expected color pattern
* [Issue 893](https://github.com/pytroll/satpy/issues/893) - Resampling data read with generic image reader corrupts data
* [Issue 876](https://github.com/pytroll/satpy/issues/876) - Update reader configuration with human-readable long names ([PR 887](https://github.com/pytroll/satpy/pull/887))
* [Issue 865](https://github.com/pytroll/satpy/issues/865) - Himawari-8 B13 image is negative?
* [Issue 863](https://github.com/pytroll/satpy/issues/863) - Record what the values from MODIS cloud mask represent
* [Issue 852](https://github.com/pytroll/satpy/issues/852) -  No module named geotiepoints.modisinterpolator
* [Issue 851](https://github.com/pytroll/satpy/issues/851) - Scene(reader, filenames = [radiance, geoloc]) expects filenames to be in a specific format
* [Issue 850](https://github.com/pytroll/satpy/issues/850) - group_files function returns only one dictionary ([PR 855](https://github.com/pytroll/satpy/pull/855))
* [Issue 848](https://github.com/pytroll/satpy/issues/848) - FCI composites not loadable ([PR 849](https://github.com/pytroll/satpy/pull/849))
* [Issue 846](https://github.com/pytroll/satpy/issues/846) - Segmentation fault calculating overlay projection with MTG
* [Issue 762](https://github.com/pytroll/satpy/issues/762) - Add x and y coordinates to all loaded gridded DataArrays
* [Issue 735](https://github.com/pytroll/satpy/issues/735) - Bilinear interpolation doesn't work with `StackedAreaDefinitions`
* [Issue 678](https://github.com/pytroll/satpy/issues/678) - Consider using setuptools-scm instead of versioneer ([PR 856](https://github.com/pytroll/satpy/pull/856))
* [Issue 617](https://github.com/pytroll/satpy/issues/617) - Update 'generic_image' reader to use rasterio for area creation ([PR 847](https://github.com/pytroll/satpy/pull/847))
* [Issue 603](https://github.com/pytroll/satpy/issues/603) - Support FY-4A hdf data ([PR 751](https://github.com/pytroll/satpy/pull/751))

In this release 15 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 915](https://github.com/pytroll/satpy/pull/915) - Fix CRS object being recreated when adding CRS coordinate
* [PR 905](https://github.com/pytroll/satpy/pull/905) - Fix ABI L2 reader not scaling and masking data
* [PR 901](https://github.com/pytroll/satpy/pull/901) - Fix compact viirs angle interpolation at the poles
* [PR 891](https://github.com/pytroll/satpy/pull/891) - Fix HDF4 reading utility using dtype classes instead of instances
* [PR 890](https://github.com/pytroll/satpy/pull/890) - Fix MERSI-2 and VIRR readers being recognized by pyspectral
* [PR 889](https://github.com/pytroll/satpy/pull/889) - Fix the ninjotiff writer to provide correct scale and offset
* [PR 884](https://github.com/pytroll/satpy/pull/884) - Update mersi2_l1b sensor name to mersi-2 to match pyspectral
* [PR 882](https://github.com/pytroll/satpy/pull/882) - Bug in mitiff writer; calibration information is not written in the imagedescription
* [PR 877](https://github.com/pytroll/satpy/pull/877) - Fix standard_name and units for T4/T13 in viirs_edr_active_fires reader
* [PR 875](https://github.com/pytroll/satpy/pull/875) - Fix error in hncc_dnb composite test
* [PR 871](https://github.com/pytroll/satpy/pull/871) - Fix FY-4 naming to follow WMO Oscar naming
* [PR 869](https://github.com/pytroll/satpy/pull/869) - Fix the nwcsaf-nc reader to drop scale and offset once data is scaled
* [PR 867](https://github.com/pytroll/satpy/pull/867) - Fix attribute datatypes in CF Writer
* [PR 837](https://github.com/pytroll/satpy/pull/837) - Fix Satpy tests to work with new versions of pyresample
* [PR 790](https://github.com/pytroll/satpy/pull/790) - Modify the  SLSTR file pattern to support stripe and frame products

#### Features added

* [PR 910](https://github.com/pytroll/satpy/pull/910) - Add near real-time and reprocessed file patterns to TROPOMI L1b reader
* [PR 907](https://github.com/pytroll/satpy/pull/907) - Handle bad orbit coefficients in SEVIRI HRIT header
* [PR 906](https://github.com/pytroll/satpy/pull/906) - Avoid xarray 0.13.0
* [PR 903](https://github.com/pytroll/satpy/pull/903) - Fix HRV area definition tests
* [PR 898](https://github.com/pytroll/satpy/pull/898) - Add night lights compositor and SEVIRI day/night composite
* [PR 897](https://github.com/pytroll/satpy/pull/897) - Cache slicing arrays in bilinear resampler
* [PR 895](https://github.com/pytroll/satpy/pull/895) - Add the possibility to pad the HRV in the seviri hrit reader
* [PR 892](https://github.com/pytroll/satpy/pull/892) - Update coefficients for FY-3B VIRR reflectance calibration
* [PR 890](https://github.com/pytroll/satpy/pull/890) - Fix MERSI-2 and VIRR readers being recognized by pyspectral
* [PR 881](https://github.com/pytroll/satpy/pull/881) - Make it possible to reverse a built-in colormap in enhancements
* [PR 880](https://github.com/pytroll/satpy/pull/880) -  Replace Numpy files with zarr for resampling LUT caching
* [PR 874](https://github.com/pytroll/satpy/pull/874) - Hardcoding of mersi2 l1b reader valid_range for channel 24 and 25 as these are wrong in the HDF data
* [PR 873](https://github.com/pytroll/satpy/pull/873) - Add mersi2 level 1b ears data file names to the reader
* [PR 872](https://github.com/pytroll/satpy/pull/872) - Fix ABI L1B coordinates to be equivalent at all resolutions
* [PR 856](https://github.com/pytroll/satpy/pull/856) - Switch to setuptools_scm for automatic version numbers from git tags ([678](https://github.com/pytroll/satpy/issues/678))
* [PR 849](https://github.com/pytroll/satpy/pull/849) - Make composites available to FCI FDHSI L1C  ([848](https://github.com/pytroll/satpy/issues/848))
* [PR 847](https://github.com/pytroll/satpy/pull/847) - Update 'generic_image' reader to use rasterio for area creation ([617](https://github.com/pytroll/satpy/issues/617))
* [PR 767](https://github.com/pytroll/satpy/pull/767) - Add a reader for NOAA GOES-R ABI L2+ products (abi_l2_nc)
* [PR 751](https://github.com/pytroll/satpy/pull/751) - Add a reader for FY-4A AGRI level 1 data ([603](https://github.com/pytroll/satpy/issues/603))
* [PR 672](https://github.com/pytroll/satpy/pull/672) - Add CIMSS True Color (Natural Color) RGB recipes

#### Documentation changes

* [PR 916](https://github.com/pytroll/satpy/pull/916) - Update orbit coefficient docstrings in seviri_l1b_hrit
* [PR 887](https://github.com/pytroll/satpy/pull/887) - Add more reader metadata like long_name and description ([876](https://github.com/pytroll/satpy/issues/876))
* [PR 878](https://github.com/pytroll/satpy/pull/878) - Add Suyash458 to AUTHORS.md

#### Backwards incompatible changes

* [PR 890](https://github.com/pytroll/satpy/pull/890) - Fix MERSI-2 and VIRR readers being recognized by pyspectral

In this release 39 pull requests were closed.


## Version 0.16.1 (2019/07/04)

### Issues Closed

* [Issue 835](https://github.com/pytroll/satpy/issues/835) - modis_l2 reader is not working properly.
* [Issue 829](https://github.com/pytroll/satpy/issues/829) - Citing satpy ([PR 833](https://github.com/pytroll/satpy/pull/833))
* [Issue 826](https://github.com/pytroll/satpy/issues/826) - SEVIRI channels loaded from netcdf in Scene object appear to have wrong names and calibration ([PR 827](https://github.com/pytroll/satpy/pull/827))
* [Issue 823](https://github.com/pytroll/satpy/issues/823) - Netcdf produced with the satpy CF writer don't pass cf-checker ([PR 825](https://github.com/pytroll/satpy/pull/825))
* [Issue 398](https://github.com/pytroll/satpy/issues/398) - Add AUTHORS file to replace individual copyright authors

In this release 5 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 843](https://github.com/pytroll/satpy/pull/843) - Remove Invalid Metadata From ACSPO Reader
* [PR 841](https://github.com/pytroll/satpy/pull/841) - Temporarily remove longitude/latitude 2D xarray coordinates
* [PR 838](https://github.com/pytroll/satpy/pull/838) - Fix 'abi_l1b' reader keeping _Unsigned attribute
* [PR 836](https://github.com/pytroll/satpy/pull/836) - Fix composites not being recorded with desired resolution in deptree
* [PR 831](https://github.com/pytroll/satpy/pull/831) - Fix EWA resampling tests not properly testing caching
* [PR 828](https://github.com/pytroll/satpy/pull/828) - Fix delayed generation of composites and composite resolution
* [PR 827](https://github.com/pytroll/satpy/pull/827) - Corrected nc_key for channels WV_062, WV_073, IR_087 ([826](https://github.com/pytroll/satpy/issues/826))
* [PR 825](https://github.com/pytroll/satpy/pull/825) - Fix the cf writer for better CF compliance ([823](https://github.com/pytroll/satpy/issues/823))

#### Features added

* [PR 842](https://github.com/pytroll/satpy/pull/842) - Fix cviirs reader to be more dask-friendly
* [PR 832](https://github.com/pytroll/satpy/pull/832) - Add pre-commit configuration

#### Documentation changes

* [PR 813](https://github.com/pytroll/satpy/pull/813) - Add some documentation to modis readers similar to hrit

#### Backwards incompatible changes

* [PR 844](https://github.com/pytroll/satpy/pull/844) - Change default CF writer engine to follow xarray defaults

In this release 12 pull requests were closed.


## Version 0.16.0 (2019/06/18)

### Issues Closed

* [Issue 795](https://github.com/pytroll/satpy/issues/795) - Composites delayed in the presence of non-dimensional coordinates ([PR 796](https://github.com/pytroll/satpy/pull/796))
* [Issue 753](https://github.com/pytroll/satpy/issues/753) - seviri l1b netcdf reader needs to be updated due to EUM fixing Attribute  Issue ([PR 791](https://github.com/pytroll/satpy/pull/791))
* [Issue 734](https://github.com/pytroll/satpy/issues/734) - Add a compositor that can use static images ([PR 804](https://github.com/pytroll/satpy/pull/804))
* [Issue 670](https://github.com/pytroll/satpy/issues/670) - Refine Satellite Position
* [Issue 640](https://github.com/pytroll/satpy/issues/640) - question: save geotiff without modifying pixel value
* [Issue 625](https://github.com/pytroll/satpy/issues/625) - Fix inconsistency  between save_dataset and save_datasets ([PR 803](https://github.com/pytroll/satpy/pull/803))
* [Issue 460](https://github.com/pytroll/satpy/issues/460) - Creating day/night composites ([PR 804](https://github.com/pytroll/satpy/pull/804))

In this release 7 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 805](https://github.com/pytroll/satpy/pull/805) - Fix 3a3b transition in the aapp l1b reader
* [PR 803](https://github.com/pytroll/satpy/pull/803) - Fix save_datasets always using geotiff writer regardless of filename ([625](https://github.com/pytroll/satpy/issues/625), [625](https://github.com/pytroll/satpy/issues/625))
* [PR 802](https://github.com/pytroll/satpy/pull/802) - Factorize and improve modis reader's interpolation
* [PR 800](https://github.com/pytroll/satpy/pull/800) - Fix 'virr_l1b' reader when slope attribute is 0
* [PR 796](https://github.com/pytroll/satpy/pull/796) - Drop non-dimensional coordinates in Compositor ([795](https://github.com/pytroll/satpy/issues/795), [795](https://github.com/pytroll/satpy/issues/795))
* [PR 792](https://github.com/pytroll/satpy/pull/792) - Bug mitiff writer when only one channel is to be written with calibration information
* [PR 791](https://github.com/pytroll/satpy/pull/791) - Fix handling of file attributes in seviri_l1b_nc reader ([753](https://github.com/pytroll/satpy/issues/753))

#### Features added

* [PR 821](https://github.com/pytroll/satpy/pull/821) - Remove warning about unused kwargs in YAML reader
* [PR 820](https://github.com/pytroll/satpy/pull/820) - Add support for NWCSAF GEO v2018, retain support for v2016
* [PR 818](https://github.com/pytroll/satpy/pull/818) - Add TLEs to dataset attributes in avhrr_l1b_gaclac
* [PR 816](https://github.com/pytroll/satpy/pull/816) - Add grouping parameters for the 'viirs_sdr' reader
* [PR 814](https://github.com/pytroll/satpy/pull/814) - Reader for Hydrology SAF precipitation products
* [PR 806](https://github.com/pytroll/satpy/pull/806) - Add flag_meanings and flag_values to 'viirs_edr_active_fires' categories
* [PR 805](https://github.com/pytroll/satpy/pull/805) - Fix 3a3b transition in the aapp l1b reader
* [PR 804](https://github.com/pytroll/satpy/pull/804) - Add compositor for adding an image as a background ([734](https://github.com/pytroll/satpy/issues/734), [460](https://github.com/pytroll/satpy/issues/460))
* [PR 794](https://github.com/pytroll/satpy/pull/794) - Add 'orbital_parameters' metadata to all geostationary satellite readers
* [PR 788](https://github.com/pytroll/satpy/pull/788) - Add new 'crs' coordinate variable when pyproj 2.0+ is installed
* [PR 779](https://github.com/pytroll/satpy/pull/779) - Add TROPOMI L2 reader (tropomi_l2)
* [PR 736](https://github.com/pytroll/satpy/pull/736) - CF Writer: Attribute encoding, groups and non-dimensional coordinates. Plus: Raw SEVIRI HRIT metadata
* [PR 687](https://github.com/pytroll/satpy/pull/687) - Add Vaisala GLD360-reader.

#### Documentation changes

* [PR 797](https://github.com/pytroll/satpy/pull/797) - Sort AUTHORS.md file by last name

#### Backwards incompatible changes

* [PR 822](https://github.com/pytroll/satpy/pull/822) - Deprecate old reader names so that they are no longer recognized ([598](https://github.com/pytroll/satpy/issues/598))
* [PR 815](https://github.com/pytroll/satpy/pull/815) - Remove legacy GDAL-based geotiff writer support

In this release 23 pull requests were closed.

## Version 0.15.2 (2019/05/22)

### Issues Closed

* [Issue 785](https://github.com/pytroll/satpy/issues/785) - Loading cache for resampling scene fails with numpy 1.16.3 ([PR 787](https://github.com/pytroll/satpy/pull/787))
* [Issue 777](https://github.com/pytroll/satpy/issues/777) - Log warning and error messages are not printed to console ([PR 778](https://github.com/pytroll/satpy/pull/778))
* [Issue 776](https://github.com/pytroll/satpy/issues/776) - africa projection yields CRSError when saving dataset ([PR 780](https://github.com/pytroll/satpy/pull/780))
* [Issue 774](https://github.com/pytroll/satpy/issues/774) - ABI Level 1b long_name when reflectances and brightness temperatures are calculated
* [Issue 766](https://github.com/pytroll/satpy/issues/766) - MODIS l1b reader seems to switch latitude and longitude for 500m data ([PR 781](https://github.com/pytroll/satpy/pull/781))
* [Issue 742](https://github.com/pytroll/satpy/issues/742) - GOES16/17 netcdf reader fails with rasterio installed
* [Issue 649](https://github.com/pytroll/satpy/issues/649) - Make MTG-I reader work ([PR 755](https://github.com/pytroll/satpy/pull/755))
* [Issue 466](https://github.com/pytroll/satpy/issues/466) - Fix deprecation warnings with xarray, dask, and numpy
* [Issue 449](https://github.com/pytroll/satpy/issues/449) - Adding coastlines to single channel not working

In this release 9 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 787](https://github.com/pytroll/satpy/pull/787) - Loading resample cache with numpy 1.16.3 ([785](https://github.com/pytroll/satpy/issues/785))
* [PR 781](https://github.com/pytroll/satpy/pull/781) - Fix longitude/latitude being swapped in modis readers ([766](https://github.com/pytroll/satpy/issues/766))
* [PR 780](https://github.com/pytroll/satpy/pull/780) - Fix builtin areas to be compatible with rasterio ([776](https://github.com/pytroll/satpy/issues/776))
* [PR 778](https://github.com/pytroll/satpy/pull/778) - Fix NullHandler not allowing warning/error logs to be printed to console ([777](https://github.com/pytroll/satpy/issues/777))
* [PR 775](https://github.com/pytroll/satpy/pull/775) - Fix 'abi_l1b' reader not updating long_name when calibrating
* [PR 770](https://github.com/pytroll/satpy/pull/770) - Fix typo for mersi2/abi/ahi using bidirection instead of bidirectional
* [PR 763](https://github.com/pytroll/satpy/pull/763) - Fix AVHRR tests importing external mock on Python 3
* [PR 760](https://github.com/pytroll/satpy/pull/760) - Avoid leaking file objects in NetCDF4FileHandler

#### Features added

* [PR 759](https://github.com/pytroll/satpy/pull/759) - Fix the avhrr_l1b_gaclac to support angles, units and avhrr variants
* [PR 755](https://github.com/pytroll/satpy/pull/755) - Update MTG FCI FDHSI L1C reader for latest data format ([649](https://github.com/pytroll/satpy/issues/649))
* [PR 470](https://github.com/pytroll/satpy/pull/470) - Switched `xarray.unfuncs` to `numpy`

#### Documentation changes

* [PR 773](https://github.com/pytroll/satpy/pull/773) - Improve Scene.show documentation
* [PR 771](https://github.com/pytroll/satpy/pull/771) - Update pull request template to include AUTHORS and flake8 changes

In this release 13 pull requests were closed.


## Version 0.15.1 (2019/05/10)

### Pull Requests Merged

#### Bugs fixed

* [PR 761](https://github.com/pytroll/satpy/pull/761) - Fix mersi2_l1b reader setting sensor as a set object

In this release 1 pull request was closed.


## Version 0.15.0 (2019/05/10)

### Issues Closed

* [Issue 758](https://github.com/pytroll/satpy/issues/758) - RuntimeError with NetCDF4FileHandler
* [Issue 730](https://github.com/pytroll/satpy/issues/730) - Rewrite introduction paragraph in documentation ([PR 747](https://github.com/pytroll/satpy/pull/747))
* [Issue 725](https://github.com/pytroll/satpy/issues/725) - Update 'viirs_edr_active_fires' reader to read newest algorithm output ([PR 733](https://github.com/pytroll/satpy/pull/733))
* [Issue 706](https://github.com/pytroll/satpy/issues/706) - Add reader for FY3D MERSI2 L1B data ([PR 740](https://github.com/pytroll/satpy/pull/740))
* [Issue 434](https://github.com/pytroll/satpy/issues/434) - Allow readers to filter the available datasets configured in YAML ([PR 739](https://github.com/pytroll/satpy/pull/739))

In this release 5 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 757](https://github.com/pytroll/satpy/pull/757) - Fix MODIS L1B and L2 readers not reading geolocation properly
* [PR 754](https://github.com/pytroll/satpy/pull/754) - Fix optional modifier dependencies being unloaded for delayed composites
* [PR 750](https://github.com/pytroll/satpy/pull/750) - Add missing warnings import to geotiff writer

#### Features added

* [PR 752](https://github.com/pytroll/satpy/pull/752) - Add scanline timestamps to seviri_l1b_hrit
* [PR 740](https://github.com/pytroll/satpy/pull/740) - Add FY-3D MERSI-2 L1B Reader (mersi2_l1b) ([706](https://github.com/pytroll/satpy/issues/706))
* [PR 739](https://github.com/pytroll/satpy/pull/739) - Refactor available datasets logic to be more flexible ([434](https://github.com/pytroll/satpy/issues/434))
* [PR 738](https://github.com/pytroll/satpy/pull/738) - Remove unused area slice-based filtering in the base reader
* [PR 733](https://github.com/pytroll/satpy/pull/733) - Update VIIRS EDR Active Fires ([725](https://github.com/pytroll/satpy/issues/725))
* [PR 728](https://github.com/pytroll/satpy/pull/728) - Add VIIRS Fire Temperature rgb
* [PR 711](https://github.com/pytroll/satpy/pull/711) - Replace usage of deprecated get_proj_coords_dask
* [PR 611](https://github.com/pytroll/satpy/pull/611) - Add MODIS L2 reader
* [PR 580](https://github.com/pytroll/satpy/pull/580) - Allow colormaps to be saved with geotiff writer
* [PR 532](https://github.com/pytroll/satpy/pull/532) - Add enhancement for VIIRS flood reader

#### Documentation changes

* [PR 747](https://github.com/pytroll/satpy/pull/747) - Update index page introduction ([730](https://github.com/pytroll/satpy/issues/730))

In this release 14 pull requests were closed.


## Version 0.14.2 (2019/04/25)

### Issues Closed

* [Issue 679](https://github.com/pytroll/satpy/issues/679) - Cannot save a multiscene animation - imagio:ffmpeg warning

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 731](https://github.com/pytroll/satpy/pull/731) - Fix viirs sdr reader to allow ivcdb files in the sdr directory
* [PR 726](https://github.com/pytroll/satpy/pull/726) - Bugfixes in the Electro-L reader ([](https://groups.google.com/forum//issues/))

#### Features added

* [PR 729](https://github.com/pytroll/satpy/pull/729) - Add "extras" checks to check_satpy utility function

#### Documentation changes

* [PR 724](https://github.com/pytroll/satpy/pull/724) - Add codeowners

In this release 4 pull requests were closed.


## Version 0.14.1 (2019/04/12)

### Issues Closed

* [Issue 716](https://github.com/pytroll/satpy/issues/716) - Reading the EUMETSAT compact viirs format returns wrong platform name (J01 instead of NOAA-20) ([PR 717](https://github.com/pytroll/satpy/pull/717))
* [Issue 710](https://github.com/pytroll/satpy/issues/710) - Question (maybe a bug): Why does RBG array exported with scn.save_dataset contain values greater than 255 ?

In this release 2 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 721](https://github.com/pytroll/satpy/pull/721) - Consistent platform id attribute across NAT + HRIT SEVIRI readers
* [PR 719](https://github.com/pytroll/satpy/pull/719) - Fix VIIRS 'night_fog' RGB composite recipe to use M12 instead of M14
* [PR 718](https://github.com/pytroll/satpy/pull/718) - Fix 'seviri_l1b_hrit' reader's area creation for pyproj 2.0+
* [PR 717](https://github.com/pytroll/satpy/pull/717) - Fix 'viirs_compact' and 'viirs_l1b' readers to return WMO/Oscar platform name ([716](https://github.com/pytroll/satpy/issues/716))
* [PR 715](https://github.com/pytroll/satpy/pull/715) - Fix hurricane florence demo download to only include M1 files
* [PR 712](https://github.com/pytroll/satpy/pull/712) - Fix 'mitiff' writer not clipping enhanced data before scaling to 8 bit values
* [PR 709](https://github.com/pytroll/satpy/pull/709) - Fix datetime64 use in 'seviri_l1b_hrit' reader for numpy < 1.15
* [PR 708](https://github.com/pytroll/satpy/pull/708) - Fix 'seviri_0deg' and 'seviri_iodc' builtin areas (areas.yaml) not matching reader areas

#### Documentation changes

* [PR 713](https://github.com/pytroll/satpy/pull/713) - Add links to source from API documentation

In this release 9 pull requests were closed.


## Version 0.14.0 (2019/04/09)

### Issues Closed

* [Issue 698](https://github.com/pytroll/satpy/issues/698) - Read WKT geotiff
* [Issue 692](https://github.com/pytroll/satpy/issues/692) - sdr_viirs_l1b reader fails in 0.13, recent master, Works with version 0.12.0 ([PR 693](https://github.com/pytroll/satpy/pull/693))
* [Issue 683](https://github.com/pytroll/satpy/issues/683) - Question: Change image size when saving with satpy.save_dataset ([PR 691](https://github.com/pytroll/satpy/pull/691))
* [Issue 681](https://github.com/pytroll/satpy/issues/681) - incorrect data offset in HSD files ([PR 689](https://github.com/pytroll/satpy/pull/689))
* [Issue 666](https://github.com/pytroll/satpy/issues/666) - Add drawing of lat lon graticules when saving dataset ([PR 668](https://github.com/pytroll/satpy/pull/668))
* [Issue 646](https://github.com/pytroll/satpy/issues/646) - Add 'demo' subpackage for accessing example data ([PR 686](https://github.com/pytroll/satpy/pull/686))
* [Issue 528](https://github.com/pytroll/satpy/issues/528) - Support dask version of PySpectral ([PR 529](https://github.com/pytroll/satpy/pull/529))
* [Issue 511](https://github.com/pytroll/satpy/issues/511) - Add/update documentation about composites and compositors ([PR 705](https://github.com/pytroll/satpy/pull/705))

In this release 8 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 700](https://github.com/pytroll/satpy/pull/700) - Mask out invalid values in the precipitation probability product
* [PR 693](https://github.com/pytroll/satpy/pull/693) - Fix VIIRS SDR reading of visible channels at nighttime ([692](https://github.com/pytroll/satpy/issues/692))
* [PR 689](https://github.com/pytroll/satpy/pull/689) - Fix Himawari HSD reader's incorrect header information ([681](https://github.com/pytroll/satpy/issues/681))
* [PR 688](https://github.com/pytroll/satpy/pull/688) - Fix offset correction in seviri_l1b_hrit
* [PR 685](https://github.com/pytroll/satpy/pull/685) - Fix bug in Scene.resample causing AssertionError
* [PR 677](https://github.com/pytroll/satpy/pull/677) - Fix MultiScene save_animation when distributed isn't installed
* [PR 675](https://github.com/pytroll/satpy/pull/675) - Do not pass `filter_parameters` to the filehandler creation

#### Features added

* [PR 691](https://github.com/pytroll/satpy/pull/691) - Add Scene.aggregate method (python 3 only) ([683](https://github.com/pytroll/satpy/issues/683))
* [PR 686](https://github.com/pytroll/satpy/pull/686) - Add demo subpackage to simplify test data download ([646](https://github.com/pytroll/satpy/issues/646))
* [PR 676](https://github.com/pytroll/satpy/pull/676) - Feature add nightfog modis
* [PR 674](https://github.com/pytroll/satpy/pull/674) - Use platform ID to choose the right reader for AVHRR GAC data
* [PR 671](https://github.com/pytroll/satpy/pull/671) - Add satellite position to dataset attributes (seviri_l1b_hrit)
* [PR 669](https://github.com/pytroll/satpy/pull/669) - Add ocean-color for viirs and modis
* [PR 668](https://github.com/pytroll/satpy/pull/668) - Add grid/graticules to add_overlay function. ([666](https://github.com/pytroll/satpy/issues/666))
* [PR 665](https://github.com/pytroll/satpy/pull/665) - Add reader for VIIRS Active Fires
* [PR 645](https://github.com/pytroll/satpy/pull/645) - Reader for the SAR OCN L2 wind product in SAFE format.
* [PR 565](https://github.com/pytroll/satpy/pull/565) - Add reader for FY-3 VIRR (virr_l1b)
* [PR 529](https://github.com/pytroll/satpy/pull/529) - Add dask support to NIRReflectance modifier ([528](https://github.com/pytroll/satpy/issues/528))

#### Documentation changes

* [PR 707](https://github.com/pytroll/satpy/pull/707) - Add ABI Meso demo data case and clean up documentation
* [PR 705](https://github.com/pytroll/satpy/pull/705) - Document composites ([511](https://github.com/pytroll/satpy/issues/511))
* [PR 701](https://github.com/pytroll/satpy/pull/701) - Clarify release instructions
* [PR 699](https://github.com/pytroll/satpy/pull/699) - Rename SatPy to Satpy throughout documentation
* [PR 673](https://github.com/pytroll/satpy/pull/673) - Add information about GDAL_CACHEMAX to FAQ

In this release 23 pull requests were closed.


## Version 0.13.0 (2019/03/18)

### Issues Closed

* [Issue 641](https://github.com/pytroll/satpy/issues/641) - After pip upgrade to satpy 0.12 and pyproj 2.0.1 got pyproj.exceptions.CRSError
* [Issue 626](https://github.com/pytroll/satpy/issues/626) - Issue loading MODIS Aqua data ([PR 648](https://github.com/pytroll/satpy/pull/648))
* [Issue 620](https://github.com/pytroll/satpy/issues/620) - Add FAQ about controlling number of threads for pykdtree and blas ([PR 621](https://github.com/pytroll/satpy/pull/621))
* [Issue 521](https://github.com/pytroll/satpy/issues/521) - Interactively set the Calibration Mode when creating the Scene Object ([PR 543](https://github.com/pytroll/satpy/pull/543))

In this release 4 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 664](https://github.com/pytroll/satpy/pull/664) - Fix Scene.crop with RGBs and multidimensional data
* [PR 662](https://github.com/pytroll/satpy/pull/662) - Fix masked resampling when dataset dtype is integer
* [PR 661](https://github.com/pytroll/satpy/pull/661) - Fix CTTH composite not to mark invalid data as cloud-free
* [PR 660](https://github.com/pytroll/satpy/pull/660) - Fix seviri_l1b_hrit prologue/epilogue readers
* [PR 655](https://github.com/pytroll/satpy/pull/655) - Fix yaml load to be compatible with pyyaml 5.1
* [PR 652](https://github.com/pytroll/satpy/pull/652) - Fix resampling of ancillary variables when also first class datasets
* [PR 648](https://github.com/pytroll/satpy/pull/648) - Add wrapped line support for metadata in modis_l1b reader ([626](https://github.com/pytroll/satpy/issues/626))
* [PR 644](https://github.com/pytroll/satpy/pull/644) - Fix the modis overview not to sun normalize the IR channel
* [PR 633](https://github.com/pytroll/satpy/pull/633) - Fix VIIRS HNCC composite passing xarray objects to dask
* [PR 632](https://github.com/pytroll/satpy/pull/632) - Fixing start and end times when missing in the CF writer

#### Features added

* [PR 647](https://github.com/pytroll/satpy/pull/647) - Switch python-hdf4 dependencies to pyhdf
* [PR 643](https://github.com/pytroll/satpy/pull/643) - In cira_strech clip values less or equal to 0 to avoid nans and -inf.
* [PR 642](https://github.com/pytroll/satpy/pull/642) - Bugfix pps2018 cpp products
* [PR 638](https://github.com/pytroll/satpy/pull/638) - Add processing-mode and disposition-mode to the avhrr-l1b-eps file name
* [PR 636](https://github.com/pytroll/satpy/pull/636) - Facilitate selection of calibration coefficients in seviri_l1b_hrit
* [PR 635](https://github.com/pytroll/satpy/pull/635) - Add local caching of slicing for data reduction
* [PR 627](https://github.com/pytroll/satpy/pull/627) - Add DNB satellite angles (DNB_SENZ, DNB_SENA) to VIIRS SDR reader
* [PR 557](https://github.com/pytroll/satpy/pull/557) - Improve the SAR-C reading and Ice composite
* [PR 543](https://github.com/pytroll/satpy/pull/543) - Calibration mode can now be passed via a keyword argument ([521](https://github.com/pytroll/satpy/issues/521))
* [PR 538](https://github.com/pytroll/satpy/pull/538) - Support CLASS packed viirs files in viirs_sdr reader

#### Documentation changes

* [PR 659](https://github.com/pytroll/satpy/pull/659) - DOC: Refer to PyTroll coding guidelines
* [PR 653](https://github.com/pytroll/satpy/pull/653) - DOC: Fix small typos in documentation
* [PR 651](https://github.com/pytroll/satpy/pull/651) - Rename changelog for releases before 0.9.0
* [PR 621](https://github.com/pytroll/satpy/pull/621) - Add FAQ items on number of workers and threads ([620](https://github.com/pytroll/satpy/issues/620))

In this release 24 pull requests were closed.


## Version 0.12.0 (2019/02/15)

### Issues Closed

* [Issue 601](https://github.com/pytroll/satpy/issues/601) - MultiScene 'save_animation' fails if "datasets=" isn't provided ([PR 602](https://github.com/pytroll/satpy/pull/602))
* [Issue 310](https://github.com/pytroll/satpy/issues/310) - Create MultiScene from list of files ([PR 576](https://github.com/pytroll/satpy/pull/576))

In this release 2 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 616](https://github.com/pytroll/satpy/pull/616) - Fix geotiff writer being unimportable if gdal isn't installed
* [PR 615](https://github.com/pytroll/satpy/pull/615) - Fix confusing error in abi_l1b reader when file fails to open
* [PR 607](https://github.com/pytroll/satpy/pull/607) - Fix VIIRS 'histogram_dnb' compositor not returning new data
* [PR 605](https://github.com/pytroll/satpy/pull/605) - Fix enhancements using dask delayed on internal functions
* [PR 602](https://github.com/pytroll/satpy/pull/602) - Fix MultiScene save_animation not using dataset IDs correctly ([601](https://github.com/pytroll/satpy/issues/601), [601](https://github.com/pytroll/satpy/issues/601))
* [PR 600](https://github.com/pytroll/satpy/pull/600) - Fix resample reduce_data bug introduced in #582

#### Features added

* [PR 614](https://github.com/pytroll/satpy/pull/614) - Support for reduced resolution OLCI data
* [PR 613](https://github.com/pytroll/satpy/pull/613) - Add 'crop' and 'save_datasets' to MultiScene
* [PR 609](https://github.com/pytroll/satpy/pull/609) - Add ability to use dask distributed when generating animation videos
* [PR 582](https://github.com/pytroll/satpy/pull/582) - Add 'reduce_data' keyword argument to disable cropping before resampling
* [PR 576](https://github.com/pytroll/satpy/pull/576) - Add group_files and from_files utility functions for creating Scenes from multiple files ([310](https://github.com/pytroll/satpy/issues/310))
* [PR 567](https://github.com/pytroll/satpy/pull/567) - Add utility functions for generating GeoViews plots ([541](https://github.com/pytroll/satpy/issues/541))

In this release 12 pull requests were closed.


## Version 0.11.2 (2019/01/28)

### Issues Closed

* [Issue 584](https://github.com/pytroll/satpy/issues/584) - DayNightCompositor does not work with eg overview_sun as the day part ([PR 593](https://github.com/pytroll/satpy/pull/593))
* [Issue 577](https://github.com/pytroll/satpy/issues/577) - Creation of composites using `sunz_corrected` modifier fails with VIIRS SDR data
* [Issue 569](https://github.com/pytroll/satpy/issues/569) - Can not show or save ABI true color image (RuntimeWarning: invalid value encountered in log)
* [Issue 531](https://github.com/pytroll/satpy/issues/531) - Mask space pixels in AHI HSD reader ([PR 592](https://github.com/pytroll/satpy/pull/592))
* [Issue 106](https://github.com/pytroll/satpy/issues/106) - Warnings

In this release 5 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 594](https://github.com/pytroll/satpy/pull/594) - Fix VIIRS L1B reader not using standard 'y' and 'x' dimension names
* [PR 593](https://github.com/pytroll/satpy/pull/593) - Fix sunz_corrected modifier adding unnecessary x and y coordinates ([587](https://github.com/pytroll/satpy/issues/587), [584](https://github.com/pytroll/satpy/issues/584))
* [PR 592](https://github.com/pytroll/satpy/pull/592) - Fix masking of AHI HSD space pixels ([531](https://github.com/pytroll/satpy/issues/531))
* [PR 589](https://github.com/pytroll/satpy/pull/589) - Fix dask not importing sharedict automatically in dask 1.1+
* [PR 588](https://github.com/pytroll/satpy/pull/588) - Fix start_time type in seviri_l1b_nc reader
* [PR 585](https://github.com/pytroll/satpy/pull/585) - Fix geotiff writer not using fill_value from writer YAML config
* [PR 572](https://github.com/pytroll/satpy/pull/572) - Fix VIIRS SDR masking and distracting colors in composites
* [PR 570](https://github.com/pytroll/satpy/pull/570) - Fix CF epoch for xarray compat
* [PR 563](https://github.com/pytroll/satpy/pull/563) - Fix StopIteration and python 3.7 compatibility issue in MultiScene
* [PR 554](https://github.com/pytroll/satpy/pull/554) - Fix AreaDefinition usage to work with newer versions of pyresample

#### Features added

* [PR 561](https://github.com/pytroll/satpy/pull/561) - Add AHI HRIT B07 files for high resolution night data

#### Documentation changes

* [PR 590](https://github.com/pytroll/satpy/pull/590) - Add FAQ page to docs
* [PR 575](https://github.com/pytroll/satpy/pull/575) - Add page for data download resources
* [PR 574](https://github.com/pytroll/satpy/pull/574) - Add code of conduct

In this release 14 pull requests were closed.


## Version 0.11.1 (2018/12/27)

### Pull Requests Merged

#### Bugs fixed

* [PR 560](https://github.com/pytroll/satpy/pull/560) - Fix available_composite_ids including inline comp dependencies

In this release 1 pull request was closed.


## Version 0.11.0 (2018/12/21)

### Issues Closed

* [Issue 555](https://github.com/pytroll/satpy/issues/555) - GOES-16 geolocation seems off when saving as TIFF
* [Issue 552](https://github.com/pytroll/satpy/issues/552) - GOES Composites failling ([PR 553](https://github.com/pytroll/satpy/pull/553))
* [Issue 534](https://github.com/pytroll/satpy/issues/534) - Support GOES-15 in netcdf format from Eumetcast (`nc_goes` reader) ([PR 530](https://github.com/pytroll/satpy/pull/530))
* [Issue 527](https://github.com/pytroll/satpy/issues/527) - [SEP] Reader naming conventions ([PR 546](https://github.com/pytroll/satpy/pull/546))
* [Issue 518](https://github.com/pytroll/satpy/issues/518) - Make bilinear interpolation dask/xarray friendly ([PR 519](https://github.com/pytroll/satpy/pull/519))
* [Issue 467](https://github.com/pytroll/satpy/issues/467) - Flake8-ify all of satpy ([PR 515](https://github.com/pytroll/satpy/pull/515))
* [Issue 459](https://github.com/pytroll/satpy/issues/459) - How to colorize images
* [Issue 449](https://github.com/pytroll/satpy/issues/449) - Adding coastlines to single channel not working ([PR 551](https://github.com/pytroll/satpy/pull/551))
* [Issue 337](https://github.com/pytroll/satpy/issues/337) - Plot true color by using VIIRS SDR
* [Issue 333](https://github.com/pytroll/satpy/issues/333) - `available_readers` to detail unavailable items
* [Issue 263](https://github.com/pytroll/satpy/issues/263) - How to get the available dataset names from the reader
* [Issue 147](https://github.com/pytroll/satpy/issues/147) - SEVIRI HRIT reading: More userfriendly warning when no EPI/PRO files are present ([PR 452](https://github.com/pytroll/satpy/pull/452))

In this release 12 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 556](https://github.com/pytroll/satpy/pull/556) - Fix turning off enhancements in writers for float data
* [PR 553](https://github.com/pytroll/satpy/pull/553) - Fix DifferenceCompositor and other compositors when areas are incompatible ([552](https://github.com/pytroll/satpy/issues/552), [552](https://github.com/pytroll/satpy/issues/552))
* [PR 550](https://github.com/pytroll/satpy/pull/550) - Fix AHI HRIT file patterns so area's ID is correct
* [PR 548](https://github.com/pytroll/satpy/pull/548) - Fix ratio sharpening compositors when the ratio is negative
* [PR 547](https://github.com/pytroll/satpy/pull/547) - Fix EWA resampling for new versions of pyresample
* [PR 542](https://github.com/pytroll/satpy/pull/542) - Fix palette application for pps 2018 products
* [PR 508](https://github.com/pytroll/satpy/pull/508) - Fix the cf_writer to accept single-valued time coordinate variable

#### Features added

* [PR 558](https://github.com/pytroll/satpy/pull/558) - Make counts available in ahi_hsd
* [PR 551](https://github.com/pytroll/satpy/pull/551) - Fix image overlays for single band data (requires trollimage 1.6+) ([449](https://github.com/pytroll/satpy/issues/449))
* [PR 549](https://github.com/pytroll/satpy/pull/549) - Fix nwcpps ct palette from v2018 to be backwards compatible
* [PR 546](https://github.com/pytroll/satpy/pull/546) - Rename readers to meet new reader naming scheme ([527](https://github.com/pytroll/satpy/issues/527))
* [PR 545](https://github.com/pytroll/satpy/pull/545) - Add configurable parameters to solar zenith correctors
* [PR 530](https://github.com/pytroll/satpy/pull/530) - Add reader for Goes15 netcdf Eumetsat format ([534](https://github.com/pytroll/satpy/issues/534))
* [PR 519](https://github.com/pytroll/satpy/pull/519) - Add xarray/dask bilinear resampling ([518](https://github.com/pytroll/satpy/issues/518))
* [PR 507](https://github.com/pytroll/satpy/pull/507) - Change default enhancement for reflectance data to gamma 1.5
* [PR 452](https://github.com/pytroll/satpy/pull/452) - Improve handling of missing file requirements in readers ([147](https://github.com/pytroll/satpy/issues/147))

#### Documentation changes

* [PR 533](https://github.com/pytroll/satpy/pull/533) - Fix copy/paste error in readers table for viirs_l1b
* [PR 515](https://github.com/pytroll/satpy/pull/515) - Fix all flake8 errors in satpy package code ([467](https://github.com/pytroll/satpy/issues/467))

#### Backwards incompatible changes

* [PR 546](https://github.com/pytroll/satpy/pull/546) - Rename readers to meet new reader naming scheme ([527](https://github.com/pytroll/satpy/issues/527))
* [PR 507](https://github.com/pytroll/satpy/pull/507) - Change default enhancement for reflectance data to gamma 1.5

In this release 20 pull requests were closed.


## Version 0.10.0 (2018/11/23)

### Issues Closed

* [Issue 491](https://github.com/pytroll/satpy/issues/491) - Area definition of incomplete SEVIRI images
* [Issue 487](https://github.com/pytroll/satpy/issues/487) - Resampling a User Defined Scene
* [Issue 465](https://github.com/pytroll/satpy/issues/465) - Native resampler fails with 3D DataArrays ([PR 468](https://github.com/pytroll/satpy/pull/468))
* [Issue 464](https://github.com/pytroll/satpy/issues/464) - Drawing coastlines/borders with save_datasets ([PR 469](https://github.com/pytroll/satpy/pull/469))
* [Issue 453](https://github.com/pytroll/satpy/issues/453) - Review subclasses of BaseFileHander ([PR 455](https://github.com/pytroll/satpy/pull/455))
* [Issue 450](https://github.com/pytroll/satpy/issues/450) - Allow readers to accept pathlib.Path instances ([PR 451](https://github.com/pytroll/satpy/pull/451))
* [Issue 445](https://github.com/pytroll/satpy/issues/445) - Readthedocs builds are failing
* [Issue 439](https://github.com/pytroll/satpy/issues/439) - KeyError when creating true_color for ABI
* [Issue 417](https://github.com/pytroll/satpy/issues/417) - Add custom string formatter for lower/upper support
* [Issue 414](https://github.com/pytroll/satpy/issues/414) - Inconsistent units of geostationary radiances ([PR 490](https://github.com/pytroll/satpy/pull/490))
* [Issue 405](https://github.com/pytroll/satpy/issues/405) - Angle interpolation for MODIS data missing ([PR 430](https://github.com/pytroll/satpy/pull/430))
* [Issue 397](https://github.com/pytroll/satpy/issues/397) - Add README to setup.py description ([PR 443](https://github.com/pytroll/satpy/pull/443))
* [Issue 369](https://github.com/pytroll/satpy/issues/369) - Mitiff writer is broken ([PR 480](https://github.com/pytroll/satpy/pull/480))

In this release 13 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 510](https://github.com/pytroll/satpy/pull/510) - Make sure a discrete data type is preserved through resampling
* [PR 506](https://github.com/pytroll/satpy/pull/506) - Remove dependency on nc_nwcsaf_msg
* [PR 504](https://github.com/pytroll/satpy/pull/504) - Change unnecessary warning messages to debug
* [PR 496](https://github.com/pytroll/satpy/pull/496) - Add more descriptive names to AHI readers AreaDefinition names
* [PR 492](https://github.com/pytroll/satpy/pull/492) - Fix thinned modis reading in 'hdfeos_l1b' reader
* [PR 480](https://github.com/pytroll/satpy/pull/480) - Fix 'mitiff' writer to use 'base_dir' properly ([369](https://github.com/pytroll/satpy/issues/369))
* [PR 476](https://github.com/pytroll/satpy/pull/476) - Fix handling of navigation in a grib file with lons greater than 180
* [PR 473](https://github.com/pytroll/satpy/pull/473) - Change combine_metadata to average any 'time' fields
* [PR 471](https://github.com/pytroll/satpy/pull/471) - Fix offset between VIS+IR and HRV navigation for hrit seviri
* [PR 469](https://github.com/pytroll/satpy/pull/469) - Fix attributes not being preserved when adding overlays or decorations ([464](https://github.com/pytroll/satpy/issues/464))
* [PR 468](https://github.com/pytroll/satpy/pull/468) - Fix native resampling when RGBs are resampled ([465](https://github.com/pytroll/satpy/issues/465))
* [PR 458](https://github.com/pytroll/satpy/pull/458) - Fix the slstr reader for consistency and tir view
* [PR 456](https://github.com/pytroll/satpy/pull/456) - Fix SCMI writer not writing fill values properly
* [PR 448](https://github.com/pytroll/satpy/pull/448) - Fix saving a dataset with a prerequisites attrs to netcdf
* [PR 447](https://github.com/pytroll/satpy/pull/447) - Fix masking in DayNightCompositor when composites have partial missing data
* [PR 446](https://github.com/pytroll/satpy/pull/446) - Fix nc_nwcsaf_msg reader's handling of projection units

#### Features added

* [PR 503](https://github.com/pytroll/satpy/pull/503) - Add two luminance sharpening compositors
* [PR 498](https://github.com/pytroll/satpy/pull/498) - Make it possible to configure in-line composites
* [PR 488](https://github.com/pytroll/satpy/pull/488) - Add the check_satpy function to find missing dependencies
* [PR 481](https://github.com/pytroll/satpy/pull/481) - Refactor SCMI writer to be dask friendly
* [PR 478](https://github.com/pytroll/satpy/pull/478) - Allow writers to create output directories if they don't exist
* [PR 477](https://github.com/pytroll/satpy/pull/477) - Add additional metadata to ABI L1B DataArrays
* [PR 474](https://github.com/pytroll/satpy/pull/474) - Improve handling of dependency loading when reader has multiple matches
* [PR 463](https://github.com/pytroll/satpy/pull/463) - MSG Level1.5 NetCDF Reader (code and yaml file) for VIS/IR Channels
* [PR 455](https://github.com/pytroll/satpy/pull/455) - Ensure file handlers all use filenames as strings ([453](https://github.com/pytroll/satpy/issues/453))
* [PR 451](https://github.com/pytroll/satpy/pull/451) - Allow readers to accept pathlib.Path instances as filenames. ([450](https://github.com/pytroll/satpy/issues/450))
* [PR 442](https://github.com/pytroll/satpy/pull/442) - Replace areas.def with areas.yaml
* [PR 441](https://github.com/pytroll/satpy/pull/441) - Fix metop reader
* [PR 438](https://github.com/pytroll/satpy/pull/438) - Feature new olcil2 datasets
* [PR 436](https://github.com/pytroll/satpy/pull/436) - Allow on-the-fly decompression of xRIT files in xRIT readers
* [PR 430](https://github.com/pytroll/satpy/pull/430) - Implement fast modis lon/lat and angles interpolation ([405](https://github.com/pytroll/satpy/issues/405))

#### Documentation changes

* [PR 501](https://github.com/pytroll/satpy/pull/501) - Add DOI role and reference to Zinke DNB method
* [PR 489](https://github.com/pytroll/satpy/pull/489) - Add a first version on how to write a custom reader
* [PR 444](https://github.com/pytroll/satpy/pull/444) - Fix the readers table in the sphinx docs so it wraps text
* [PR 443](https://github.com/pytroll/satpy/pull/443) - Add long_description to setup.py ([397](https://github.com/pytroll/satpy/issues/397))
* [PR 440](https://github.com/pytroll/satpy/pull/440) - Fix CI badges in README

#### Backwards incompatible changes

* [PR 485](https://github.com/pytroll/satpy/pull/485) - Deprecate 'enhancement_config' keyword argument in favor of 'enhance'

In this release 37 pull requests were closed.


## Version 0.9.4 (2018/09/29)

### Pull Requests Merged

#### Bugs fixed

* [PR 433](https://github.com/pytroll/satpy/pull/433) - Fix native_msg readers standard_names to match other satpy readers
* [PR 432](https://github.com/pytroll/satpy/pull/432) - Fix reader config loading so it raises exception for bad reader name
* [PR 428](https://github.com/pytroll/satpy/pull/428) - Fix start_time and end_time being lists in native_msg reader
* [PR 426](https://github.com/pytroll/satpy/pull/426) - Fix hrit_jma reader not having satellite lon/lat/alt info
* [PR 423](https://github.com/pytroll/satpy/pull/423) - Fixed that save_dataset does not propagate fill_value
* [PR 421](https://github.com/pytroll/satpy/pull/421) - Fix masking and simplify avhrr_aapp_l1b reader
* [PR 413](https://github.com/pytroll/satpy/pull/413) - Fix calculating solar zenith angle in eps_l1b reader
* [PR 412](https://github.com/pytroll/satpy/pull/412) - Fix platform_name and sensor not being added by avhrr eps l1b reader

#### Features added

* [PR 415](https://github.com/pytroll/satpy/pull/415) - Add hrit_jma file patterns that don't include segments

In this release 9 pull requests were closed.


## Version 0.9.3 (2018/09/10)

### Issues Closed

* [Issue 336](https://github.com/pytroll/satpy/issues/336) - Scene crop does not compare all dataset areas ([PR 406](https://github.com/pytroll/satpy/pull/406))

In this release 1 issue was closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 409](https://github.com/pytroll/satpy/pull/409) - Fix viirs_sdr reading of aggregated files
* [PR 406](https://github.com/pytroll/satpy/pull/406) - Fix Scene crop so new areas are consistent with resolution ([336](https://github.com/pytroll/satpy/issues/336))

In this release 2 pull requests were closed.


## Version 0.9.2 (2018/08/23)

### Pull Requests Merged

#### Bugs fixed

* [PR 402](https://github.com/pytroll/satpy/pull/402) - Fix 'platform_name' metadata in ACSPO and CLAVR-x readers
* [PR 401](https://github.com/pytroll/satpy/pull/401) - Wrap solar and satellite angles in xarray in AVHRR AAPP reader

In this release 2 pull requests were closed.


## Version 0.9.1 (2018/08/19)

### Issues Closed

* [Issue 388](https://github.com/pytroll/satpy/issues/388) - SCMI Writer raises exception with lettered grids ([PR 389](https://github.com/pytroll/satpy/pull/389))
* [Issue 385](https://github.com/pytroll/satpy/issues/385) - No platform_name and sensor in dataset metadata for avhrr_aapp_l1b reader ([PR 386](https://github.com/pytroll/satpy/pull/386))
* [Issue 379](https://github.com/pytroll/satpy/issues/379) - Data is not masked when loading calibrated GOES HRIT data ([PR 380](https://github.com/pytroll/satpy/pull/380))
* [Issue 377](https://github.com/pytroll/satpy/issues/377) - Unmasked data when using DayNightCompositor ([PR 378](https://github.com/pytroll/satpy/pull/378))
* [Issue 372](https://github.com/pytroll/satpy/issues/372) - "find_files_and_readers" doesn't work on Windows ([PR 373](https://github.com/pytroll/satpy/pull/373))
* [Issue 364](https://github.com/pytroll/satpy/issues/364) - Unable to load individual channels from VIIRS_SDR data.
* [Issue 350](https://github.com/pytroll/satpy/issues/350) - Creating a Scene object with NOAA-15/18 data
* [Issue 347](https://github.com/pytroll/satpy/issues/347) - No image is shown in Jupyter notebook via scene.show()
* [Issue 345](https://github.com/pytroll/satpy/issues/345) - Future warning - xarray ([PR 352](https://github.com/pytroll/satpy/pull/352))

In this release 9 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 395](https://github.com/pytroll/satpy/pull/395) - Fix DayNightCompositor not checking inputs areas
* [PR 391](https://github.com/pytroll/satpy/pull/391) - Fix native resampler using SwathDefinition as an AreaDefinition
* [PR 387](https://github.com/pytroll/satpy/pull/387) - Fix enhancement config loading when yaml file is empty
* [PR 386](https://github.com/pytroll/satpy/pull/386) - Add platform_name and sensor in avhrr_aapp_l1b reader ([385](https://github.com/pytroll/satpy/issues/385))
* [PR 381](https://github.com/pytroll/satpy/pull/381) - Fix keyword arguments not being properly passed to writers
* [PR 362](https://github.com/pytroll/satpy/pull/362) - Replace np.ma.mean by np.nanmean for pixel aggregation
* [PR 361](https://github.com/pytroll/satpy/pull/361) - Remove Rayleigh correction from abi natural composite
* [PR 360](https://github.com/pytroll/satpy/pull/360) - Fix lookup table enhancement for multi-band datasets
* [PR 339](https://github.com/pytroll/satpy/pull/339) - fixed meteosat native georeferencing

#### Documentation changes

* [PR 359](https://github.com/pytroll/satpy/pull/359) - Add examples from pytroll-examples to documentation

In this release 10 pull requests were closed.


## Version 0.9.0 (2018/07/02)

### Issues Closed

* [Issue 344](https://github.com/pytroll/satpy/issues/344) - find_files_and_reader does not seem to care about start_time! ([PR 349](https://github.com/pytroll/satpy/pull/349))
* [Issue 338](https://github.com/pytroll/satpy/issues/338) - Creating a Scene object with Terra MODIS data
* [Issue 332](https://github.com/pytroll/satpy/issues/332) - Non-requested datasets are saved when composites fail to generate ([PR 342](https://github.com/pytroll/satpy/pull/342))

In this release 3 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 355](https://github.com/pytroll/satpy/pull/355) - Fix ABI L1B reader losing file variable attributes
* [PR 353](https://github.com/pytroll/satpy/pull/353) - Fix multiscene memory issues by adding an optional batch_size
* [PR 351](https://github.com/pytroll/satpy/pull/351) - Fix AMSR-2 L1B reader loading bytes incorrectly
* [PR 349](https://github.com/pytroll/satpy/pull/349) - Fix datetime-based file selection when filename only has a start time ([344](https://github.com/pytroll/satpy/issues/344))
* [PR 348](https://github.com/pytroll/satpy/pull/348) - Fix freezing of areas before resampling even as strings
* [PR 343](https://github.com/pytroll/satpy/pull/343) - Fix shape assertion after resampling
* [PR 342](https://github.com/pytroll/satpy/pull/342) - Fix Scene save_datasets to only save datasets from the wishlist ([332](https://github.com/pytroll/satpy/issues/332))
* [PR 341](https://github.com/pytroll/satpy/pull/341) - Fix ancillary variable loading when anc var is already loaded
* [PR 340](https://github.com/pytroll/satpy/pull/340) - Cut radiances array depending on number of scans

In this release 9 pull requests were closed.


## Version 0.9.0b0 (2018/06/26)

### Issues Closed

* [Issue 328](https://github.com/pytroll/satpy/issues/328) - hrit reader bugs ([PR 329](https://github.com/pytroll/satpy/pull/329))
* [Issue 323](https://github.com/pytroll/satpy/issues/323) - "Manual" application of corrections
* [Issue 320](https://github.com/pytroll/satpy/issues/320) - Overview of code layout
* [Issue 279](https://github.com/pytroll/satpy/issues/279) - Add 'level' to DatasetID ([PR 283](https://github.com/pytroll/satpy/pull/283))
* [Issue 272](https://github.com/pytroll/satpy/issues/272) - How to save region of interest from Band 3 Himawari Data as png image ([PR 276](https://github.com/pytroll/satpy/pull/276))
* [Issue 267](https://github.com/pytroll/satpy/issues/267) - Missing dependency causes strange error during unit tests ([PR 273](https://github.com/pytroll/satpy/pull/273))
* [Issue 244](https://github.com/pytroll/satpy/issues/244) - Fix NUCAPS reader for NUCAPS EDR v2 files ([PR 326](https://github.com/pytroll/satpy/pull/326))
* [Issue 236](https://github.com/pytroll/satpy/issues/236) - scene.resample(cache_dir=) fails with TypeError: Unicode-objects must be encoded before hashing
* [Issue 233](https://github.com/pytroll/satpy/issues/233) - IOError: Unable to read attribute (no appropriate function for conversion path)
* [Issue 211](https://github.com/pytroll/satpy/issues/211) - Fix OLCI and other readers' file patterns to work on Windows
* [Issue 207](https://github.com/pytroll/satpy/issues/207) - Method not fully documented in terms of possible key word arguments
* [Issue 199](https://github.com/pytroll/satpy/issues/199) - Reading Modis file produce a double image
* [Issue 168](https://github.com/pytroll/satpy/issues/168) - Cannot read MODIS data
* [Issue 167](https://github.com/pytroll/satpy/issues/167) - KeyError 'v' using Scene(base_dir=, reader=) ([PR 325](https://github.com/pytroll/satpy/pull/325))
* [Issue 165](https://github.com/pytroll/satpy/issues/165) - HRIT GOES reader is broken ([PR 303](https://github.com/pytroll/satpy/pull/303))
* [Issue 160](https://github.com/pytroll/satpy/issues/160) - Inconsistent naming of optional datasets in composite configs and compositors
* [Issue 157](https://github.com/pytroll/satpy/issues/157) - Add animation example ([PR 322](https://github.com/pytroll/satpy/pull/322))
* [Issue 156](https://github.com/pytroll/satpy/issues/156) - Add cartopy example
* [Issue 146](https://github.com/pytroll/satpy/issues/146) - Add default null log handler
* [Issue 123](https://github.com/pytroll/satpy/issues/123) - NetCDF writer doesn't work ([PR 307](https://github.com/pytroll/satpy/pull/307))
* [Issue 114](https://github.com/pytroll/satpy/issues/114) - Print a list of available sensors/readers
* [Issue 82](https://github.com/pytroll/satpy/issues/82) - Separate file discovery from Scene init
* [Issue 61](https://github.com/pytroll/satpy/issues/61) - Creating composites post-load
* [Issue 10](https://github.com/pytroll/satpy/issues/10) - Optimize CREFL for memory

In this release 24 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 331](https://github.com/pytroll/satpy/pull/331) - Adapt slstr reader to xarray&dask
* [PR 329](https://github.com/pytroll/satpy/pull/329) - issue#328: fixed bugs loading JMA HRIT files ([328](https://github.com/pytroll/satpy/issues/328))
* [PR 326](https://github.com/pytroll/satpy/pull/326) - Fix nucaps reader for NUCAPS EDR v2 files ([244](https://github.com/pytroll/satpy/issues/244), [244](https://github.com/pytroll/satpy/issues/244))
* [PR 325](https://github.com/pytroll/satpy/pull/325) - Fix exception when Scene is given reader and base_dir ([167](https://github.com/pytroll/satpy/issues/167))
* [PR 319](https://github.com/pytroll/satpy/pull/319) - Fix msi reader delayed
* [PR 318](https://github.com/pytroll/satpy/pull/318) - Fix nir reflectance to use XArray
* [PR 312](https://github.com/pytroll/satpy/pull/312) - Allow custom regions in ahi-hsd file patterns
* [PR 311](https://github.com/pytroll/satpy/pull/311) - Allow valid_range to be a tuple for cloud product colorization
* [PR 303](https://github.com/pytroll/satpy/pull/303) - Fix hrit goes to support python 3 ([165](https://github.com/pytroll/satpy/issues/165))
* [PR 288](https://github.com/pytroll/satpy/pull/288) - Fix hrit-goes reader
* [PR 192](https://github.com/pytroll/satpy/pull/192) - Clip day and night composites after enhancement

#### Features added

* [PR 315](https://github.com/pytroll/satpy/pull/315) - Add slicing to Scene
* [PR 314](https://github.com/pytroll/satpy/pull/314) - Feature mitiff writer
* [PR 307](https://github.com/pytroll/satpy/pull/307) - Fix projections in cf writer ([123](https://github.com/pytroll/satpy/issues/123))
* [PR 305](https://github.com/pytroll/satpy/pull/305) - Add support for geolocation and angles to msi reader
* [PR 302](https://github.com/pytroll/satpy/pull/302) - Workaround the LinearNDInterpolator thread-safety issue for Sentinel 1 SAR geolocation
* [PR 301](https://github.com/pytroll/satpy/pull/301) - Factorize header definitions between hrit_msg and native_msg. Fix a bug in header definition.
* [PR 298](https://github.com/pytroll/satpy/pull/298) - Implement sentinel 2 MSI reader
* [PR 294](https://github.com/pytroll/satpy/pull/294) - Add the ocean color product to olci
* [PR 153](https://github.com/pytroll/satpy/pull/153) - [WIP] Improve compatibility of cf_writer with CF-conventions

In this release 20 pull requests were closed.


## Version 0.9.0a2 (2018/05/14)

### Issues Closed

* [Issue 286](https://github.com/pytroll/satpy/issues/286) - Proposal: search automatically for local config-files/readers
* [Issue 278](https://github.com/pytroll/satpy/issues/278) - msg native reader fails on full disk image
* [Issue 277](https://github.com/pytroll/satpy/issues/277) - msg_native reader fails when order number has a hyphen in it ([PR 282](https://github.com/pytroll/satpy/pull/282))
* [Issue 270](https://github.com/pytroll/satpy/issues/270) - How to find the value at certain latitude and longtitude
* [Issue 269](https://github.com/pytroll/satpy/issues/269) - How to intepret the parameter values in  AreaDefinition
* [Issue 268](https://github.com/pytroll/satpy/issues/268) - How to find the appropriate values of parameters in Scene.resample() function using Himawari Data
* [Issue 241](https://github.com/pytroll/satpy/issues/241) - reader native_msg using `np.str`
* [Issue 218](https://github.com/pytroll/satpy/issues/218) - Resampling to EPSG:4326 produces unexpected results
* [Issue 189](https://github.com/pytroll/satpy/issues/189) - Error when reading MSG native format
* [Issue 62](https://github.com/pytroll/satpy/issues/62) - msg_native example
* [Issue 33](https://github.com/pytroll/satpy/issues/33) - Load metadata without loading data

In this release 11 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 290](https://github.com/pytroll/satpy/pull/290) - Fix unicode-named data loading
* [PR 285](https://github.com/pytroll/satpy/pull/285) - Fix native_msg calibration bug
* [PR 282](https://github.com/pytroll/satpy/pull/282) - Fix native_msg reader for ROI input and multi-part order file patterns ([277](https://github.com/pytroll/satpy/issues/277))
* [PR 280](https://github.com/pytroll/satpy/pull/280) - Fix CLAVR-x reader to work with xarray
* [PR 274](https://github.com/pytroll/satpy/pull/274) - Convert ahi hsd reader to dask and xarray
* [PR 265](https://github.com/pytroll/satpy/pull/265) - Bugfix msg native reader
* [PR 262](https://github.com/pytroll/satpy/pull/262) - Fix dependency tree to find the best dependency when multiple matches occur
* [PR 260](https://github.com/pytroll/satpy/pull/260) - Fix ABI L1B reader masking data improperly

#### Features added

* [PR 293](https://github.com/pytroll/satpy/pull/293) - Switch to netcdf4 as engine for nc nwcsaf reading
* [PR 292](https://github.com/pytroll/satpy/pull/292) - Use pyresample's boundary classes
* [PR 291](https://github.com/pytroll/satpy/pull/291) - Allow datasets without areas to be concatenated
* [PR 289](https://github.com/pytroll/satpy/pull/289) - Fix so UMARF files (with extention .nat) are found as well
* [PR 287](https://github.com/pytroll/satpy/pull/287) - Add production configuration for NWCSAF RDT, ASII products by Marco Sassi
* [PR 283](https://github.com/pytroll/satpy/pull/283) - Add GRIB Reader ([279](https://github.com/pytroll/satpy/issues/279))
* [PR 281](https://github.com/pytroll/satpy/pull/281) - Port the maia reader to dask/xarray
* [PR 276](https://github.com/pytroll/satpy/pull/276) - Support reducing data for geos areas ([272](https://github.com/pytroll/satpy/issues/272))
* [PR 273](https://github.com/pytroll/satpy/pull/273) - Msg readers cleanup ([267](https://github.com/pytroll/satpy/issues/267))
* [PR 271](https://github.com/pytroll/satpy/pull/271) - Add appveyor and use ci-helpers for CI environments
* [PR 264](https://github.com/pytroll/satpy/pull/264) - Add caching at the scene level, and handle saving/loading from disk
* [PR 262](https://github.com/pytroll/satpy/pull/262) - Fix dependency tree to find the best dependency when multiple matches occur

In this release 20 pull requests were closed.


## Version 0.9.0a1 (2018/04/22)

### Issues Closed

* [Issue 227](https://github.com/pytroll/satpy/issues/227) - Issue Reading MSG4
* [Issue 225](https://github.com/pytroll/satpy/issues/225) - Save Datasets using SCMI ([PR 228](https://github.com/pytroll/satpy/pull/228))
* [Issue 215](https://github.com/pytroll/satpy/issues/215) - Change `Scene.compute` to something else ([PR 220](https://github.com/pytroll/satpy/pull/220))
* [Issue 208](https://github.com/pytroll/satpy/issues/208) - Strange behaviour when trying to load data to a scene object after having worked with it ([PR 214](https://github.com/pytroll/satpy/pull/214))
* [Issue 200](https://github.com/pytroll/satpy/issues/200) - Different mask handling when saving to PNG or GeoTIFF ([PR 201](https://github.com/pytroll/satpy/pull/201))
* [Issue 176](https://github.com/pytroll/satpy/issues/176) - Loading viirs natural_color composite fails ([PR 177](https://github.com/pytroll/satpy/pull/177))

In this release 6 issues were closed.

### Pull Requests Merged

#### Bugs fixed

* [PR 259](https://github.com/pytroll/satpy/pull/259) - Fix writer and refactor so bad writer name raises logical exception
* [PR 257](https://github.com/pytroll/satpy/pull/257) - Fix geotiff and png writers to save to a temporary directory
* [PR 256](https://github.com/pytroll/satpy/pull/256) - Add 'python_requires' to setup.py to specify python support
* [PR 253](https://github.com/pytroll/satpy/pull/253) - Fix ABI L1B reader to use 64-bit scaling factors for X/Y variables
* [PR 250](https://github.com/pytroll/satpy/pull/250) - Fix floating point geotiff saving in dask geotiff writer
* [PR 249](https://github.com/pytroll/satpy/pull/249) - Fix float geotiff saving on 0.8
* [PR 248](https://github.com/pytroll/satpy/pull/248) - Fix unloading composite deps when one of them has incompatible areas
* [PR 243](https://github.com/pytroll/satpy/pull/243) - Remove ABI composite reducerX modifiers

#### Features added

* [PR 252](https://github.com/pytroll/satpy/pull/252) - Use rasterio to save geotiffs when available
* [PR 239](https://github.com/pytroll/satpy/pull/239) - Add CSPP Geo (geocat) AHI reading support

In this release 10 pull requests were closed.


## Version 0.9.0a0 (2018-03-20)

#### Bugs fixed

* [Issue 179](https://github.com/pytroll/satpy/issues/179) - Cannot read AVHRR in AAPP format
* [PR 234](https://github.com/pytroll/satpy/pull/234) - Bugfix sar reader
* [PR 231](https://github.com/pytroll/satpy/pull/231) - Bugfix palette based compositor concatenation
* [PR 230](https://github.com/pytroll/satpy/pull/230) - Fix dask angle calculations of rayleigh corrector
* [PR 229](https://github.com/pytroll/satpy/pull/229) - Fix bug in dep tree when modifier deps are modified wavelengths
* [PR 228](https://github.com/pytroll/satpy/pull/228) - Fix 'platform' being used instead of 'platform_name'
* [PR 224](https://github.com/pytroll/satpy/pull/224) - Add helper method for checking areas in compositors
* [PR 222](https://github.com/pytroll/satpy/pull/222) - Fix resampler caching by source area
* [PR 221](https://github.com/pytroll/satpy/pull/221) - Fix Scene loading and resampling when generate=False
* [PR 220](https://github.com/pytroll/satpy/pull/220) - Rename Scene's `compute` to `generate_composites`
* [PR 219](https://github.com/pytroll/satpy/pull/219) - Fixed native_msg calibration problem and added env var to change the …
* [PR 214](https://github.com/pytroll/satpy/pull/214) - Fix Scene not being copied properly during resampling
* [PR 210](https://github.com/pytroll/satpy/pull/210) - Bugfix check if lons and lats should be masked before resampling
* [PR 206](https://github.com/pytroll/satpy/pull/206) - Fix optional dependencies not being passed to modifiers with opts only
* [PR 187](https://github.com/pytroll/satpy/pull/187) - Fix reader configs having mismatched names between filename and config
* [PR 185](https://github.com/pytroll/satpy/pull/185) - Bugfix nwcsaf_pps reader for file discoverability
* [PR 177](https://github.com/pytroll/satpy/pull/177) - Bugfix viirs loading - picked from (xarray)develop branch
* [PR 163](https://github.com/pytroll/satpy/pull/163) - Bugfix float geotiff

#### Features added

* [PR 232](https://github.com/pytroll/satpy/pull/232) - Add ABI L1B system tests
* [PR 226](https://github.com/pytroll/satpy/pull/226) - EARS NWCSAF products reading
* [PR 217](https://github.com/pytroll/satpy/pull/217) - Add xarray/dask support to DayNightCompositor
* [PR 216](https://github.com/pytroll/satpy/pull/216) - Fix dataset writing so computations are shared between tasks
* [PR 213](https://github.com/pytroll/satpy/pull/213) - [WIP] Reuse same resampler for similar datasets
* [PR 212](https://github.com/pytroll/satpy/pull/212) - Improve modis reader to support dask
* [PR 209](https://github.com/pytroll/satpy/pull/209) - Fix enhancements to work with xarray
* [PR 205](https://github.com/pytroll/satpy/pull/205) - Fix ABI 'natural' and 'true_color' composites to work with xarray
* [PR 204](https://github.com/pytroll/satpy/pull/204) - Add 'native' resampler
* [PR 203](https://github.com/pytroll/satpy/pull/203) - [WIP] Feature trollimage xarray
* [PR 195](https://github.com/pytroll/satpy/pull/195) - Add ABI-specific configs for Airmass composite
* [PR 186](https://github.com/pytroll/satpy/pull/186) - Add missing nodata tiff tag
* [PR 180](https://github.com/pytroll/satpy/pull/180) - Replace BW and RGBCompositor with a more generic one

#### Documentation changes

* [PR 155](https://github.com/pytroll/satpy/pull/155) - Add contributing and developers guide documentation

In this release 1 issue and 31 pull requests were closed.
