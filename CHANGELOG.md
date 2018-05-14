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
