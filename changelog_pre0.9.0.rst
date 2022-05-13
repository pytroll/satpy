Changelog
=========


v0.8.1 (2018-01-19)
-------------------

Fix
~~~
- Bugfix: Fix so the Himawari platform name is a string and not a numpy
  array. [Adam.Dybbroe]
- Bugfix: The satellite azimuth returned by PyOrbital is not in the
  range -180 to 180 as was expected. [Adam.Dybbroe]

Other
~~~~~
- Update changelog. [Martin Raspaud]
- Bump version: 0.8.0 → 0.8.1. [Martin Raspaud]
- Merge pull request #162 from pytroll/bugfix-pyorbital-azimuth-
  difference. [Martin Raspaud]

  Bugfix: The satellite azimuth returned by PyOrbital is not in the ran…
- Merge pull request #154 from pytroll/bugfix-viirs-truecolor-
  ratiosharpening. [Martin Raspaud]

  Add a rayleigh_correction modifier for I-bands,
- Add a rayleigh_correction modifier for I-bands, which is refered to in
  the ratio-sharpened true color and natural_color RGBs. [Adam.Dybbroe]
- Fix backwards compatibility with scene instantiation. [Martin Raspaud]


v0.8.0 (2018-01-11)
-------------------

Fix
~~~
- Bugfix: Explicitly set the resolution for sun-satellite geometry for
  the Rayleigh correction modifiers needed for True Color imagery.
  [Adam.Dybbroe]

Other
~~~~~
- Update changelog. [Martin Raspaud]
- Bump version: 0.7.8 → 0.8.0. [Martin Raspaud]
- Merge pull request #152 from pytroll/bugfix-truecolor-viirs. [Martin
  Raspaud]

  Bugfix: Explicitly set the resolution for sun-satellite geometry
- Bugfix viirs_sdr reader: Use correct sunz corrector for ibands.
  [Adam.Dybbroe]
- Merge pull request #91 from pytroll/feature-discover-utility. [Martin
  Raspaud]

  Separate find files utility
- Merge branch 'develop' into feature-discover-utility. [David Hoese]
- Refactor all of the documentation and fix various docstrings. [davidh-
  ssec]
- Update documentation index and installation instructions. [davidh-
  ssec]
- Merge branch 'develop' into feature-discover-utility. [davidh-ssec]

  # Conflicts:
  #	satpy/readers/mipp_xrit.py
  #	satpy/tests/test_readers.py
  #	satpy/utils.py

- Add filename filtering and tests for find_files_and_readers. [davidh-
  ssec]
- Remove unused strftime function. [davidh-ssec]
- Fix behavior tests and other necessary changes to fix file discovery.
  [davidh-ssec]
- Update Scene and reader loading docstrings. [davidh-ssec]
- Move reader start_time and end_time to filter_parameters. [davidh-
  ssec]

  Includes a first attempt at updating mipp_xrit to work with this

- Fix `load_readers` tests after changing from ReaderFinder. [davidh-
  ssec]
- Remove 'sensor' functionality from Scene init and clean reader
  loading. [davidh-ssec]
- Fix behavior tests. [davidh-ssec]
- Move file finding functionality to a separate utility function.
  [davidh-ssec]
- Move ABI simulated green calculation to a separate function. [davidh-
  ssec]
- Merge pull request #149 from pytroll/truecolor-red-channel-corr.
  [Martin Raspaud]

  Truecolor uses red channel as base for rayleigh correction
- Fix indentation error in viirs.yaml. [Martin Raspaud]
- Merge branch 'develop' into truecolor-red-channel-corr. [Martin
  Raspaud]
- Remove marine-clean true color recipe, as it was the same as the
  standard recipe. [Adam.Dybbroe]
- Bugfix abi true color recipes. [Adam.Dybbroe]
- Apply consistency in true color imagery across sensors. Adding for
  land and sea variants. [Adam.Dybbroe]
- Use the red band in the damping of the atm correction over reflective
  targets. [Adam.Dybbroe]


v0.7.8 (2018-01-11)
-------------------
- Update changelog. [Martin Raspaud]
- Bump version: 0.7.7 → 0.7.8. [Martin Raspaud]
- Merge pull request #148 from pytroll/feature-utils. [Martin Raspaud]

   Fix platform name reading for ahi hsd reader in py3
- Fix platform name reading for ahi hsd reader in py3. [Martin Raspaud]

  This patch also factorizes some code to a np2str function that takes care of converting np.string_ to str
- Merge pull request #130 from pytroll/ahi_truecolor. [Martin Raspaud]

  Use the cira stretch also for the true_color_ahi_default
- Use consistent standard_name naming. [Adam.Dybbroe]
- Fix for Himawari true colors at different resolutions. [Adam.Dybbroe]
- Use the cira stretch also for the true_color_ahi_default.
  [Adam.Dybbroe]
- Merge pull request #141 from pytroll/pep8. [Martin Raspaud]

  Remove unused imports and use pep8-ify
- Remove unused imports and use pep8-ify. [Adam.Dybbroe]
- Merge pull request #145 from pytroll/fix-refl37-rgbs. [Martin Raspaud]

  Add snow RGB, add r37-based and natural RGB recipes specific to SEVIRI, and fix sun-zenith correction
- When doing atm correction with pass the band name rather than the
  wavelength to Pyspectral, as the latter may be ambigous.
  [Adam.Dybbroe]
- Explain how the 3.x reflectance needs to be derived before getting the
  emissive part. [Adam.Dybbroe]
- Removing the two protected internal variables: self._nir and
  self._tb11. [Adam.Dybbroe]
- Add new recipes for daytime-cloudtop RGBs using Pyspectral to remove
  the reflective part of the 3.x signal. [Adam.Dybbroe]
- Add method initiating the reflectance/emissive calculations.
  [Adam.Dybbroe]
- Update __init__.py. [Adam Dybbroe]

  Replaced "dummy" with "_"
- Add a NIR (3.x micron band) emissive RGB provided by new pyspectral.
  [Adam.Dybbroe]
- Adapt method call to latest pyspectral. [Adam.Dybbroe]
- Fix so it is possible to derive 3.7 micron reflective RGBs from both
  VIIRS I- and M-bands. [Adam.Dybbroe]
- Add snow RGBs for VIIRS for both M- and I-bands. [Adam.Dybbroe]
- Add snow RGB, add r37-based and natural RGB recipes specific to
  SEVIRI, and fix sun-zenith correction. [Adam.Dybbroe]
- Merge pull request #143 from pytroll/noaa-20-platform-naming. [Martin
  Raspaud]

  Fix platform_name for NOAA-20 and -21
- Fix platform_name for NOAA-20 and -21. [Adam.Dybbroe]


v0.7.7 (2017-12-21)
-------------------
- Update changelog. [davidh-ssec]
- Bump version: 0.7.6 → 0.7.7. [davidh-ssec]
- Merge pull request #140 from pytroll/bugfix-scmi-signed. [David Hoese]

  Bugfix scmi signed integer data variables
- Add ipython tab completion for scene keys. [davidh-ssec]
- Fix SCMI writer because AWIPS doesn't like unsigned integers. [davidh-
  ssec]

  Using the entire 16-bit unsigned integer space displays fine in AWIPS
  but it doesn't handle them correctly when adding derived parameters.
  Meaning once the data goes in to a python script and gets converted to
  a signed interger...yeah. This change makes it so data is a signed
  16-bit integer that only uses the positive half of the bit space.

- Merge pull request #138 from pytroll/bugfix-modis-reader. [David
  Hoese]

  WIP: Fix readers not returning the highest resolution dataset IDs
- Add more file patterns to hdfeos_l1b reader. [davidh-ssec]
- Fix requesting a specific resolution from a reader. [davidh-ssec]
- Merge remote-tracking branch 'origin/fix-resolution' into bugfix-
  modis-reader. [davidh-ssec]
- Allow providing resolution when loading a composite. [Martin Raspaud]
- Fix hdfeos_l1b reader not knowing what resolution of datasets it had.
  [davidh-ssec]
- Fix interpolation problem at 250m resolution. [Martin Raspaud]
- Fix readers not returning the highest resolution dataset IDs. [davidh-
  ssec]
- Merge pull request #139 from pytroll/bugfix-viirs-l1b. [David Hoese]

  Fix VIIRS L1B to work with JPSS-1 and new NASA filenames
- Fix VIIRS L1B to work with JPSS-1 and new NASA filenames. [davidh-
  ssec]
- Clean up style. [Martin Raspaud]
- Fix lon/lat caching in hdfeos_l1b for different resolutions. [Martin
  Raspaud]

  Fixes #132
- Merge pull request #137 from pytroll/logging_corrupted_file. [Martin
  Raspaud]

  When opening/reading a nc or hdf file fails, be verbose telling which file it is that fails
- When opening/reading a file fails, be verbose telling which file it is
  that fails. [Adam.Dybbroe]
- Merge pull request #134 from howff/hdfeos_l1b_ipopp_filenames. [Martin
  Raspaud]

  Added IPOPP-style MODIS-L1b filenames
- Update doc re. IMAPP and IPOPP. [Andrew Brooks]
- Added IPOPP-style MODIS-L1b filenames. [Andrew Brooks]


v0.7.6 (2017-12-19)
-------------------
- Update changelog. [Martin Raspaud]
- Bump version: 0.7.5 → 0.7.6. [Martin Raspaud]
- Merge pull request #135 from pytroll/viirs_truecolor_config_error.
  [Martin Raspaud]

  Replace effective_solar_pathlength_corrected with the standard sunz-corrected
- Replace effective_solar_pathlength_corrected witn the standard sunz-
  correction. VIIRS data are already sun-zenith corrected.
  [Adam.Dybbroe]
- Update documentation to add hrit_goes. [Martin Raspaud]
- Fix GOES navigation. [Martin Raspaud]
- Finalize GOES LRIT reader. [Martin Raspaud]
- Merge pull request #39 from howff/develop. [Martin Raspaud]

  Reader for GOES HRIT, WIP
- Fix available_composite_names in doc. [Andrew Brooks]
- Merge branch 'develop' of https://github.com/pytroll/satpy into
  develop. [Andrew Brooks]
- Start of reader for GOES HRIT. [howff]
- Update PULL_REQUEST_TEMPLATE.md. [Martin Raspaud]

  This hides the comments when the PR is previewed and reminds user to provide a description for the PR.
- Merge pull request #122 from eysteinn/scatsat1. [Martin Raspaud]

  Add reader for ScatSat1 Level 2B wind speed data, HDF5 format
- Read end_time info correctly. [Eysteinn]
- Add reader for ScatSat1 Level 2B wind speed data. [Eysteinn]
- Merge pull request #129 from pytroll/viirs_rgbs. [Martin Raspaud]

  Use the Pyspectral atm correction as the default.
- Use the Pyspectral atm correction as the default. Add a high-res
  overview RGB, use the hncc-dnb in the night-microphysics and use the
  effective_solar_pathlength_corrected for all true color RGBs.
  [Adam.Dybbroe]
- Merge pull request #128 from pytroll/atm_corrections. [Martin Raspaud]

  Atm corrections
- Pep8 cosmetics. [Adam.Dybbroe]
- Pep8 cosmetics. [Adam.Dybbroe]
- Pep8 editorial, and fixing copyright. [Adam.Dybbroe]
- Add some pre-defined atm/rayleigh corrections to appply over land and
  sea. [Adam.Dybbroe]
- Merge pull request #131 from pytroll/bugfix-hrit-jma. [Martin Raspaud]

  Bugfix hrit_jma
- Bugfix hrit_jma. [Martin Raspaud]
- Use a more appropriate and shorter link to the MSG native format pdf
  doc. [Adam.Dybbroe]
- Merge pull request #126 from pytroll/feature_ahi_stretch. [Martin
  Raspaud]

  Improvemements to AHI True color imagery
- Use marine_clean and us-standard for atm correction, and improve
  stretch at low sun elevation. [Adam.Dybbroe]
- Use the CIRA stretch for True color imagery. [Adam.Dybbroe]


v0.7.5 (2017-12-11)
-------------------
- Update changelog. [davidh-ssec]
- Bump version: 0.7.4 → 0.7.5. [davidh-ssec]
- Remove unused legacy .cfg files. [davidh-ssec]
- Merge branch 'master' into develop. [davidh-ssec]
- Merge pull request #118 from mitkin/master. [Martin Raspaud]

  Add file pattern for MODIS L1B from LAADS WEB
- Add file pattern for MODIS L1B from LAADS WEB. [Mikhail Itkin]

  NASA's LAADS WEB pattern is slightly different

- Remove old and unused mipp_xrit reader. [davidh-ssec]
- Fix SCMI writer not overwriting data from previous tiles. [davidh-
  ssec]
- Merge pull request #121 from pytroll/fix-ir-modifiers. [Martin
  Raspaud]

  Remove VIIRS SDR IR modifiers
- Remove sun zenith angle correction from IR channels. [Panu Lahtinen]
- Add github templates for issues and PRs. [Martin Raspaud]
- Bugfix epsl1b reader. [Martin Raspaud]
- Merge pull request #107 from pytroll/fix-nwcsaf-proj4. [David Hoese]

  Convert NWC SAF MSG projection string to meters
- Merge branch 'fix-nwcsaf-proj4' of https://github.com/pytroll/satpy
  into fix-nwcsaf-proj4. [Panu Lahtinen]
- Merge branch 'fix-nwcsaf-proj4' of https://github.com/pytroll/satpy
  into fix-nwcsaf-proj4. [Panu Lahtinen]
- Read attributes "flag_meanings", "flag_values" and "long_name" [Panu
  Lahtinen]
- Configure more datasets. [Panu Lahtinen]
- Fix also area extents. [Panu Lahtinen]
- Add unit tests for utils.proj_units_to_meters() [Panu Lahtinen]
- Move proj_units_to_meters() to satpy.utils. [Panu Lahtinen]
- Convert projection parameters from kilometers to meters. [Panu
  Lahtinen]
- Read attributes "flag_meanings", "flag_values" and "long_name" [Panu
  Lahtinen]
- Configure more datasets. [Panu Lahtinen]
- Fix also area extents. [Panu Lahtinen]
- Add unit tests for utils.proj_units_to_meters() [Panu Lahtinen]
- Move proj_units_to_meters() to satpy.utils. [Panu Lahtinen]
- Convert projection parameters from kilometers to meters. [Panu
  Lahtinen]
- Move proj_units_to_meters() to satpy.utils. [Panu Lahtinen]
- Convert projection parameters from kilometers to meters. [Panu
  Lahtinen]
- Read attributes "flag_meanings", "flag_values" and "long_name" [Panu
  Lahtinen]
- Configure more datasets. [Panu Lahtinen]
- Fix also area extents. [Panu Lahtinen]
- Add unit tests for utils.proj_units_to_meters() [Panu Lahtinen]
- Move proj_units_to_meters() to satpy.utils. [Panu Lahtinen]
- Convert projection parameters from kilometers to meters. [Panu
  Lahtinen]
- Merge pull request #111 from eysteinn/sentinel1-reproject. [David
  Hoese]

  Fixed area information to safe_sar_c reader to allow for resampling
- Added coordinates to sar_c.yaml to allow for reprojection. [Eysteinn]
- Merge pull request #108 from TAlonglong/feature-decorate. [Martin
  Raspaud]

  Feature decorate
- __init__.py docstring in a few add pydecorate features. [Trygve
  Aspenes]
- Satpy/writers/__init__.py implement more general way of handling
  pydecorate calls from satpy save_dataset. Instead of logo and text
  separate, use decorate. This needs to be a list to keep the order of
  alignment available in pydecorate. Since the argument to add_decorate
  needs to be a mapping it may look like this:
  decorate={'decorate':[{'logo':{...}},{'text':{...}},...]} [Trygve
  Aspenes]
- Merge branch 'develop' into develop-fork. [Trygve Aspenes]
- Satpy/writers/__init__.py added add_text function. This is meant to be
  used when calling save_dataset to add text to an image using
  pydecorate. eg save_dataset(...., text_overlay={'text': 'THIS IS THE
  TEXT TO BE ADDED', 'align':{'top_bottom':'bottom',
  'left_right':'right'},
  'font':'/usr/share/fonts/truetype/msttcorefonts/Arial.ttf',
  'font_size':25, 'height':30, 'bg':'black', 'bg_opacity':255,
  'line':'white'}). Not all options available as style in pydecorate are
  implemented. This is left TODO. This PR is dependent on
  https://github.com/pytroll/pydecorate/pull/3 to be completed. [Trygve
  Aspenes]
- Adding to more options to add_overlay. This to better control which
  levels of coast(GSHHS) and borders (WDB_II) are put on the plot.
  [Trygve Aspenes]
- Merge pull request #88 from pytroll/feature-3d-enhancement. [Panu
  Lahtinen]

  Add 3D enhancement, fix BWCompositor
- Merge branch 'feature-3d-enhancement' of
  https://github.com/pytroll/satpy into feature-3d-enhancement. [Panu
  Lahtinen]
- Add example of composite with 3D effect. [Panu Lahtinen]
- Fix BWCompositor to handle info correctly. [Panu Lahtinen]
- Add 3D effect enhancement. [Panu Lahtinen]
- Remove rebase comments. [Panu Lahtinen]
- Add example of composite with 3D effect. [Panu Lahtinen]
- Fix BWCompositor to handle info correctly. [Panu Lahtinen]
- Add 3D effect enhancement. [Panu Lahtinen]
- Merge pull request #87 from pytroll/feature-IASI-L2-reader. [Panu
  Lahtinen]

  Add IASI L2 reader
- Merge branch 'feature-IASI-L2-reader' of
  https://github.com/pytroll/satpy into feature-IASI-L2-reader. [Panu
  Lahtinen]
- Merge branch 'feature-IASI-L2-reader' of
  https://github.com/pytroll/satpy into feature-IASI-L2-reader. [Panu
  Lahtinen]
- Fix unit of time. [Panu Lahtinen]
- Remove un-needed '' from the reader init line. [Panu Lahtinen]
- Merge branch 'develop' into feature-IASI-L2-reader. [Panu Lahtinen]
- Add mapping from M03 to Metop-C. [Panu Lahtinen]
- Add subsatellite resolution to datasets. [Panu Lahtinen]
- Fix typos, make read_dataset() and read_geo() functions instead of
  methods. [Panu Lahtinen]
- Add initial version of IASI L2 reader. [Panu Lahtinen]
- Fix unit of time. [Panu Lahtinen]
- Remove un-needed '' from the reader init line. [Panu Lahtinen]
- Add mapping from M03 to Metop-C. [Panu Lahtinen]
- Add subsatellite resolution to datasets. [Panu Lahtinen]
- Fix typos, make read_dataset() and read_geo() functions instead of
  methods. [Panu Lahtinen]
- Add initial version of IASI L2 reader. [Panu Lahtinen]
- Fix unit of time. [Panu Lahtinen]
- Remove un-needed '' from the reader init line. [Panu Lahtinen]
- Add mapping from M03 to Metop-C. [Panu Lahtinen]
- Add subsatellite resolution to datasets. [Panu Lahtinen]
- Fix typos, make read_dataset() and read_geo() functions instead of
  methods. [Panu Lahtinen]
- Add initial version of IASI L2 reader. [Panu Lahtinen]
- Merge pull request #96 from eysteinn/create_colormap. [David Hoese]

  Create colormap
- Make colorizing/palettizing more flexible. [Eysteinn]
- Merge pull request #4 from pytroll/develop. [Eysteinn Sigurðsson]

  Develop
- Merge pull request #3 from pytroll/develop. [Eysteinn Sigurðsson]

  Develop
- Merge pull request #109 from pytroll/bugfix-scmi. [David Hoese]

  Fix SCMI writer and add more tiled grids
- Fix SCMI writer writing masked geolocation to netcdf files. [davidh-
  ssec]
- Add additional GOES SCMI grids. [davidh-ssec]
- Allow adding overlay for L and LA images. [Martin Raspaud]
- Merge pull request #101 from pytroll/bugfix-scmi3. [David Hoese]

  Fix python 3 compatibility in scmi writer
- Add more SCMI writer tests for expected failures. [davidh-ssec]
- Fix python 3 compatibility in scmi writer. [davidh-ssec]

  Includes fix for X/Y coordinate precision which affects GOES-16 data

- Merge pull request #105 from howff/doc-fix. [Martin Raspaud]

  fix available_composite_names in doc
- Fix available_composite_names in doc. [Andrew Brooks]


v0.7.4 (2017-11-13)
-------------------
- Update changelog. [davidh-ssec]
- Bump version: 0.7.3 → 0.7.4. [davidh-ssec]
- Update changelog. [davidh-ssec]
- Fix physical_element for VIIRS M07 in SCMI writer. [davidh-ssec]
- Merge pull request #97 from pytroll/feature-optimize-scmi. [David
  Hoese]

  Optimize SCMI writer to reuse results of tile calculations
- Fix area id in SCMI writer to be more specific. [davidh-ssec]
- Optimize SCMI writer to reuse results of tile calculations. [davidh-
  ssec]

  It uses a little bit more memory, but speeds up the processing by quite
  a bit when tested under the Polar2Grid equivalent.

- Fix floating point saving for geotiff. [Martin Raspaud]
- Merge pull request #93 from pytroll/bugfix-user-enhancements. [David
  Hoese]

  Fix enhancement config loading when user configs are present
- Fix enhancement config loading when user configs are present. [davidh-
  ssec]


v0.7.3 (2017-10-24)
-------------------
- Update changelog. [davidh-ssec]
- Bump version: 0.7.2 → 0.7.3. [davidh-ssec]
- Merge branch 'develop' into new_release. [davidh-ssec]
- Fix mock import in unittest. [davidh-ssec]

  mock should come from the unittest package in python 3+

- Merge pull request #90 from pytroll/bugfix-scmi-writer. [David Hoese]

  Fix SCMI writer to use newest version of pyresample
- Fix SCMI writer to use newest version of pyresample. [davidh-ssec]
- Adjust extents to kilometers. [Panu Lahtinen]
- Merge pull request #86 from pytroll/bugfix-resample-setitem. [David
  Hoese]

  Fix resampling when a dataset was added via setitem and a test for it
- Fix resampling when a dataset was added via setitem and a test for it.
  [davidh-ssec]

  Includes removing python 3.3 from travis tests

- Merge pull request #84 from eysteinn/composite-snowage-fix. [Martin
  Raspaud]

  Composite snowage fix
- Expand the dynamic of the channels up to 255 before to combine them:
  (0,1.6) => (0,255) [Eysteinn]
- Merge pull request #2 from pytroll/develop. [Eysteinn Sigurðsson]

  Develop
- Merge pull request #85 from pytroll/feature-fullres-abi-tc. [David
  Hoese]

  Feature fullres abi tc
- Fix geocat tests. [davidh-ssec]
- Fix bug in geocat reader and SCMI writer. [davidh-ssec]

  Caused incorrect H8 and GOES-16 geolocation

- Fix reader metaclass with newer versions of six. [davidh-ssec]
- Fix metadata in ABI true color. [davidh-ssec]
- Fix ABI true color averaging. [davidh-ssec]
- Fix DatasetID comparison in python 3 and add test for it. [davidh-
  ssec]
- Fix super call in ABI true color 2km class. [davidh-ssec]
- Add writers yaml files to setup.py. [davidh-ssec]
- Create sharpened full resolution ABI true color. [davidh-ssec]
- Merge pull request #81 from loreclem/develop. [Martin Raspaud]

  Develop
- Added some doc. [lorenzo clementi]
- Fixed missing import. [lorenzo clementi]
- Bugfix (typo) [lorenzo clementi]
- First working version of ninjo converter. [lorenzo clementi]
- Improved generic reader, removed useles bitmap composite. [lorenzo
  clementi]
- Bugfix in the generic image reader. [lorenzo clementi]
- Draft generic image reader. [lorenzo clementi]
- Merge pull request #80 from pytroll/solar-pathlength-correction.
  [Martin Raspaud]

  Solar pathlength correction and Rayleigh correction interface
- Fix anti pattern: Not using get() to return a default value from a
  dict. [Adam.Dybbroe]
- Introduce an alternative sun-zenith correction algorithm, and fix
  rayleigh/aerosol correction so atmosphere and aerosol type can be
  specified in the config files. [Adam.Dybbroe]
- Merge branch 'develop' into solar-pathlength-correction.
  [Adam.Dybbroe]
- Maia reader (#79) [roquetp]

  * not finalised version : problem with standard name
  * Fix maia reader for simple loading
  * working version with CM and CT
  * add Datasets and fix the problem with end_time.
  * Add a exemple for read MAIA files
  * Add maia reader
  * fix on maia name
  * add reference on the test case
  * autopep8 on the example polar_maia.py and add the reference of the data
  test case
  * maia-reader : clean and pep8
  * add reference documentation



v0.7.2 (2017-09-18)
-------------------

Fix
~~~
- Bugfix: Get the solar zenith angle. [Adam.Dybbroe]

Other
~~~~~
- Update changelog. [davidh-ssec]
- Bump version: 0.7.1 → 0.7.2. [davidh-ssec]
- Merge pull request #67 from pytroll/feature-scmi-writer. [David Hoese]

  Feature scmi writer
- Fix SCMI lettered grid test to not create huge arrays. [davidh-ssec]
- Fix SCMI test so it actually uses lettered grids. [davidh-ssec]
- Add more SCMI writer tests and documentation. [davidh-ssec]
- Fix geocat reader for better X/Y coordinate estimation. [davidh-ssec]
- Add really basic SCMI writer test. [davidh-ssec]
- Fix SCMI debug tile generation. [davidh-ssec]
- Add debug tile creation to SCMI writer. [davidh-ssec]
- Fix SCMI writer for lettered grids. [davidh-ssec]
- Fix numbered tile counts for SCMI writer. [davidh-ssec]
- Add initial SCMI writer. [davidh-ssec]

  WIP: Multiple tiles, lettered tiles, debug images

- Separate EnhancementDecisionTree in to base DecisionTree and subclass.
  [davidh-ssec]
- Add 'goesr' as possible platform in geocat reader. [davidh-ssec]
- Add SCMI and geotiff writer extras to setup.py. [davidh-ssec]
- Add GOES-16 filename to geocat config. [davidh-ssec]
- Merge pull request #69 from pytroll/modis-viewing-geometry-and-atm-
  correction. [Martin Raspaud]

  Modis viewing geometry and atm correction
- Modis true_color atm corrected with pyspectral. [Adam.Dybbroe]
- Merge branch 'develop' into modis-viewing-geometry-and-atm-correction.
  [Adam.Dybbroe]
- Merge pull request #73 from pytroll/cira-stretch-numpy-1-13-issue.
  [Martin Raspaud]

  Add unittest for cira_stretch and fix it for numpy >=1.13
- Bugfix unittest suite. [Adam.Dybbroe]
- Fix cira_stretch to work despite broken numpy (numpy issue 9687)
  [Adam.Dybbroe]
- Smaller unittest example, and fixed. Works for numpy < 1.13 only
  though. [Adam.Dybbroe]
- Add unittest for cira_stretch and fix it for numpy >=1.13.
  [Adam.Dybbroe]
- Merge pull request #75 from pytroll/feature_realistic_colors. [Martin
  Raspaud]

  Realistic colors composite for SEVIRI
- Merge branch 'develop' into feature_realistic_colors. [Martin Raspaud]
- Merge branch 'develop' into feature_realistic_colors. [Martin Raspaud]
- Add RealisticColors compositor for SEVIRI. [Panu Lahtinen]
- Use array shape instead of possibly non-existent lon array shape.
  [Panu Lahtinen]
- Adjust mask size when number of channels is changed when enhancing.
  [Panu Lahtinen]
- Merge pull request #71 from eysteinn/composite-snowage. [Martin
  Raspaud]

  added snow_age viirs composite & lookup table enhancement
- Merge branch 'develop' into composite-snowage. [Martin Raspaud]
- Ch out is explicit. [Eysteinn]
- Allows any number of channels. [Eysteinn]
- Allows any number of channels. [Eysteinn]
- Fixed satpy/etc/enhancements/generic.yaml. [Eysteinn]
- Added snow_age viirs composite & lookup table enhancement. [Eysteinn]
- Merge pull request #72 from pytroll/feature_day-night_compositor.
  [Martin Raspaud]

  Add DayNightCompositor
- Add DayNightCompositor and example composite and enhancement configs.
  [Panu Lahtinen]
- Merge pull request #74 from eysteinn/composite-seviri. [Martin
  Raspaud]

  Composite seviri
- .changed night_overview to ir_overview. [Eysteinn]
- Added night_overview to seviri. [Eysteinn]
- Added night_microphysics to visir. [Eysteinn]
- Merge pull request #68 from pytroll/feature_palette_enhancement. [Panu
  Lahtinen]

  Merged.
- Update with palettize() and clarify usage. [Panu Lahtinen]
- Refactor using _merge_colormaps() instead of dupplicate code. [Panu
  Lahtinen]
- Add palettize() [Panu Lahtinen]
- Fix typo. [Panu Lahtinen]
- Add user palette colorization to quickstart documentation. [Panu
  Lahtinen]
- Add palettize enhancement and colormap creation from .npy files. [Panu
  Lahtinen]
- Add sun-sat viewing angles and support for atm correction.
  [Adam.Dybbroe]
- Bugfix atm correction. [Adam.Dybbroe]
- Merge pull request #65 from pytroll/feature_bwcompositor. [Martin
  Raspaud]

  Feature bwcompositor
- Undo line wrapping done by autopep8. [Panu Lahtinen]
- Add single channel compositor. [Panu Lahtinen]
- Merge pull request #66 from loreclem/master. [Martin Raspaud]

  Added test to check the  1.5 km georeferencing shift
- Added test to check whether to apply the  1.5 km georeferencing
  correction or not. [lorenzo clementi]
- Add ir atm correction, and new airmass composite using this
  correction. [Adam.Dybbroe]
- Change writer configs from INI (.cfg) to YAML (#63) [David Hoese]

  * Change writer configs from INI (.cfg) to YAML

  * Add very simple writer tests and fix writer load from Scene
- Merge pull request #59 from pytroll/feature-geocat-reader. [David
  Hoese]

  Add geocat reader
- Add CLAVR-x reader to documentation. [davidh-ssec]
- Add geocat reader to documentation. [davidh-ssec]
- Fix a few styling issues in geocat reader. [davidh-ssec]
- Add python-hdf4 and HDF4 C library to travis dependencies. [davidh-
  ssec]
- Add HDF4 utils tests. [davidh-ssec]
- Add geocat unit tests. [davidh-ssec]
- Add geocat reader. [davidh-ssec]


v0.7.1 (2017-08-29)
-------------------
- Update changelog. [Martin Raspaud]
- Bump version: 0.7.0 → 0.7.1. [Martin Raspaud]
- Fix style. [Martin Raspaud]
- Fix hdf4 lib name in dependencies. [Martin Raspaud]
- Rename optional dependencies for hdfeos to match reader name. [Martin
  Raspaud]
- Rename mda with metadata in hdfeos_l1b reader. [Martin Raspaud]
- Add overview composite for modis. [Martin Raspaud]
- Do not guess end time when filtering a filename. [Martin Raspaud]
- Add optional dependencies for viirs_compact. [Martin Raspaud]
- Fix abi_l1b test again. [Martin Raspaud]
- Fix abi_l1b tests. [Martin Raspaud]
- Fix sweep axis parameter reading in py3 for abi_l1b. [Martin Raspaud]
- Support py3 in abi_l1b. [Martin Raspaud]
- Add optional dependencies for abi_l1b. [Martin Raspaud]
- Merge pull request #58 from pytroll/metadata-filtering. [Martin
  Raspaud]

  Metadata filtering
- Fix filehandler unit test to use filename_info as a dict. [Martin
  Raspaud]
- Implement suggested style changes. [Martin Raspaud]

  See conversation in PR #58
- Finish fixing 0° Service to 0DEG. [Martin Raspaud]
- Fix Meteosat numbers to remove leading 0. [Martin Raspaud]
- Change HRIT base service to 0DEG. [Martin Raspaud]
- Change HRIT MSG patterns to explicit `service` [Martin Raspaud]
- Correct unit tests for metadata filtering compatibility. [Martin
  Raspaud]
- Add metadata filtering of filehandlers. [Martin Raspaud]
- Replace filter by list comprehension for py3 compatibility. [Martin
  Raspaud]
- Check area compatibility before merging channels in RGBCompositor.
  [Martin Raspaud]
- Add overview for ABI. [Martin Raspaud]
- Add EUM file patterns for ABI. [Martin Raspaud]
- Avoid crash when pattern matching on file crashes. [Martin Raspaud]
- Fix clavrx reader when filenames don't have end_time. [davidh-ssec]
- Add optional dependencies for sar_c. [Martin Raspaud]
- Fix h5py py3 issues with byte arrays as strings. [Martin Raspaud]
- Add optional dependency for the nc_nwcsaf_msg reader. [Martin Raspaud]
- Fix hrit_msg reading for py3. [Martin Raspaud]
- Add optional dependency for the hrit_msg reader. [Martin Raspaud]
- Add platform_name and service to msg metadata. [Martin Raspaud]
- Bugfix in MSG acquisition time metadata. [Martin Raspaud]
- Fix xRIT end time to follow specifications. [Martin Raspaud]


v0.7.0 (2017-08-15)
-------------------
- Update changelog. [Martin Raspaud]
- Bump version: 0.6.2 → 0.7.0. [Martin Raspaud]
- Fix support for OMPS EDRs from other NASA sources. [davidh-ssec]

  Fix #57

- Change 'ncc_zinke' composite name to 'hncc_dnb' [davidh-ssec]

  Includes changes to code to make sure that things we think are floats
  actually are floats.

- Fix major bug that stopped certain composites from being loadable.
  [davidh-ssec]

  If a composite modified (added information) to the DatasetID of its
  returned Dataset then the wishlist was not properly modified. This
  resulted in the Dataset being unloaded and seen as "unneeded". There
  was a test for this, but it wasn't working as expected.

- Update ABI scale factors to be 64-bit floats to improve X/Y
  calculations. [davidh-ssec]

  In other applications I have noticed that the in-file 32-bit
  factor and offset produce a noticeable drift in the per-pixel X/Y
  values. When converted to 64-bit to force 64-bit arithmetic the results
  are closer to the advertised pixel resolution of the instrument.

- Add 'reader' name metadata to all reader datasets. [davidh-ssec]
- Add flag_meanings to clavrx reader. [davidh-ssec]

  Includes addition of /dtype to hdf4/hdf5/netcdf file handlers

- Fix area unit conversion. [Martin Raspaud]
- Fix the path to the doc to test. [Martin Raspaud]
- Fix some documentation. [Martin Raspaud]
- Fix area hashing in resample caching. [davidh-ssec]
- Add better error when provided enhancement config doesn't exist.
  [davidh-ssec]
- Simple workaround for printing a dataset with no-name areas. [davidh-
  ssec]
- Fix `get_config_path` to return user files before package provided.
  [davidh-ssec]
- Fix bug in geotiff writer where gdal options were ignored. [davidh-
  ssec]
- Merge pull request #53 from pytroll/feature-clavrx-reader. [David
  Hoese]

  Add CLAVR-x reader
- Update setuptools before installing on travis. [davidh-ssec]
- Fix enhancement configs in setup.py. [davidh-ssec]

  Includes fixing of hdf4 dependency to python-hdf4

- Add CLAVR-x reader. [davidh-ssec]
- Merge pull request #54 from tparker-usgs/writerTypo. [David Hoese]

  Correct typo in writer
- Correct typo. [Tom Parker]


v0.6.2 (2017-05-22)
-------------------
- Update changelog. [davidh-ssec]
- Bump version: 0.6.1 → 0.6.2. [davidh-ssec]
- Fix NUCAPS reader when used with multiple input granules. [davidh-
  ssec]

  Includes extra fix for the scene when missing datasets need to be
  printed/logged.

- Work on projections for cf-writer. [Martin Raspaud]
- Cosmetic fixes. [Martin Raspaud]
- Improve cf write including grid mappings. [Martin Raspaud]
- Bugfix eps_l1b. [Martin Raspaud]
- Pass kwargs to dataset saving. [Martin Raspaud]
- Add ninjotiff writer. [Martin Raspaud]
- Avoid crashing when resampling  datasets without area. [Martin
  Raspaud]
- Add reducer8 compositor. [Martin Raspaud]
- Merge pull request #51 from pytroll/common-nwcsaf-readers. [Martin
  Raspaud]

  Add reader for NWCSAF/PPS which can also be used by NWCSAF/MSG
- Add support for PPS/CPP cloud phase and effective radius.
  [Adam.Dybbroe]
- Harmonize composite names between PPS and MSG, and try handle the odd
  PPS palette in CTTH-height. [Adam.Dybbroe]
- Added more PPS products - CPP parameters still missing. [Adam.Dybbroe]
- Add modis support for pps reader. [Adam.Dybbroe]
- Comment out get_shape method. [Adam.Dybbroe]
- Add reader for NWCSAF/PPS which can also be used by NWCSAF/MSG.
  [Adam.Dybbroe]
- Add initial enhancer tests. [davidh-ssec]


v0.6.1 (2017-04-24)
-------------------
- Update changelog. [Martin Raspaud]
- Bump version: 0.6.0 → 0.6.1. [Martin Raspaud]
- Change branch for landscape badge. [Martin Raspaud]
- Fix badge to point to develop. [Martin Raspaud]
- Add a couple of badges to the readme. [Martin Raspaud]
- Remove imageo subpackage and related tests. [davidh-ssec]
- Add test for ReaderFinder. [davidh-ssec]

  Required fixing all reader tests that had improper patching of base file handlers.

- Add NUCAPS reader tests. [davidh-ssec]
- Fix OMPS EDR valid_min comparison. [davidh-ssec]
- Add OMPS EDR tests. [davidh-ssec]
- Add shape checking to AMSR2 L1B tests. [davidh-ssec]
- Attempt to fix AMSR2 L1B reader tests. [davidh-ssec]
- Add AMSR2 L1B tests. [davidh-ssec]
- Fix loading of failed datasets. [davidh-ssec]

  Fix #42

- Fix viirs sdr loading when dataset's file type isn't loaded. [davidh-
  ssec]
- Add a ColorizeCompositor vs PaletteCompositor. [Martin Raspaud]
- Fix viirs sdr tests for python 3. [davidh-ssec]
- Add ability for VIIRS SDRs to load geolocation files from N_GEO_Ref.
  [davidh-ssec]

  Also fixed tests and fixed dfilter not working in VIIRS SDRs when
  key was a DatasetID

- Clean up styling for coordinates check. [davidh-ssec]

  Quantified code complained about duplicate if statements

- Raise ValueError instead of IOError when standard_name is missing in
  coordinates. [Adam.Dybbroe]
- Use previously unused cache dict to hold cached geolocation data.
  [Adam.Dybbroe]
- Remove redundant import. [Adam.Dybbroe]
- Raise an IOError when (lon,lat) coordinates doesn't have a
  standard_name. [Adam.Dybbroe]
- Add warning when sensor is not supported by any readers. [davidh-ssec]

  Fix #32



v0.6.0 (2017-04-18)
-------------------

Fix
~~~
- Bugfix: Masking data and apply vis-calibration. [Adam.Dybbroe]
- Bugfix: Add wavelength to the DatasetID. [Adam.Dybbroe]
- Bugfix: Add wavelength to the dataset info object, so pyspectral
  interface works. [Adam.Dybbroe]

Other
~~~~~
- Update changelog. [Martin Raspaud]
- Bump version: 0.5.0 → 0.6.0. [Martin Raspaud]
- Fix pyresample link in README. [davidh-ssec]
- Update documentation and readme to be more SatPy-y. [davidh-ssec]
- Add ACSPO reader to documentation. [davidh-ssec]
- Reduce redundant code in netcdf4 based tests. [davidh-ssec]
- Add ACSPO reader tests. [davidh-ssec]
- Force minimum version of netcdf4-python. [davidh-ssec]
- Update pip on travis before installing dependencies. [davidh-ssec]
- Install netcdf4 from source tarball on travis instead of from wheel.
  [davidh-ssec]

  netCDF4-python seems to be broken on travis when installed from a wheel.
  This tries installing it from a source tarball.

- Replace netcdf4 with h5netcdf in netcdf4 file handler tests. [davidh-
  ssec]

  Travis has a library issue with netcdf4 so trying h5netcdf instead

- Install cython via apt for travis tests. [davidh-ssec]
- Add tests for NetCDF4 File Handler utility class. [davidh-ssec]
- Add tests for HDF5 File Handler utility class. [davidh-ssec]
- Update VIIRS L1B tests to work with python 3. [davidh-ssec]

  Includes installing netcdf4 apt packages on travis

- Add netCDF4 library to travis tests. [davidh-ssec]
- Add VIIRS L1B tests. [davidh-ssec]
- Change YAML reader to only provide datasets that are requested.
  [davidh-ssec]

  Includes changes to mask any data slices when data can't be loaded from
  one or more file handlers. Raises an error if all file handlers fail.

- Clean up style. [Martin Raspaud]
- Add behave test for returned least modified dataset. [davidh-ssec]
- Merge pull request #48 from pytroll/feature_bilinear. [David Hoese]

  Bilinear interpolation
- Merge pull request #49 from pytroll/fix_ewa. [David Hoese]

  Fix EWA resampling
- Remove data copy from EWA resampling. [davidh-ssec]
- Send copy of the data to fornav() [Panu Lahtinen]
- Merge branch 'fix_ewa' of https://github.com/pytroll/satpy into
  fix_ewa. [Panu Lahtinen]
- Send copy of data to fornav() [Panu Lahtinen]

  - Fixes EWA resampling

- Remove unused import. [Panu Lahtinen]
- Discard masks from cache data. [Panu Lahtinen]
- Start fixing EWA; single channels work, multichannels yield bad
  images. [Panu Lahtinen]
- Add example using bilinear interpolation, caching and more CPUs. [Panu
  Lahtinen]
- Handle datasets with multiple channels. [Panu Lahtinen]
- Reorganize code. [Panu Lahtinen]

  - move caches to base class attribute
  - move cache reading to base class
  - move cache updating to base class

- Add bilinear resampling, separate lonlat masking to a function. [Panu
  Lahtinen]
- Merge pull request #50 from pytroll/feature-acspo-reader. [David
  Hoese]

  Add ACSPO SST Reader
- Add more documentation methods in ACSPO reader. [davidh-ssec]
- Fix ACSPO reader module docstring. [davidh-ssec]
- Add ACSPO SST Reader. [davidh-ssec]
- Cleanup code based on quantifiedcode. [davidh-ssec]
- Add test to make sure least modified datasets are priorities in
  getitem. [davidh-ssec]
- Change DatasetID sorting to be more pythonic. [davidh-ssec]
- Fix incorrect usage of setdefault. [davidh-ssec]
- Change DatasetIDs to be sortable and sort them in DatasetDict.keys()
  [davidh-ssec]
- Make failing test more deterministic. [davidh-ssec]

  Planning to change how requested datasets are loaded/discovered so this test will need to get updated in the future anyway.

- Fix DatasetDict.__getitem__ being slightly non-deterministic. [davidh-
  ssec]

  __getitem__ was depending on the output and order of .keys() which is
  not guaranteed to be the same every time. If more than one key was found
  to match the `item` then the first in a list based on .keys() was
  returned. The first element in this list was not always the same.

- Fix Scene loading or computing datasets multiple times. [davidh-ssec]
- Add filename filtering for start and end time. [davidh-ssec]
- Fix Scene loading datasets multiple times. [davidh-ssec]

  Fix #45

- Fix setup.py's usage of find_packages. [davidh-ssec]
- Fix deleting an item from the Scene if it wasn't in the wishlist.
  [davidh-ssec]

  If a user specified `unload=False` then there may be something in the Scene that isn't needed later.

- Use setuptool's find_packages in setup.py. [davidh-ssec]
- Use only h5py for compact viirs reading. [Martin Raspaud]
- Remove hanging print statements. [Martin Raspaud]
- Add night overview composite for viirs. [Martin Raspaud]
- Add area def for MSG HRV. [Martin Raspaud]
- Merge pull request #47 from pytroll/feature-yaml-enhancements. [Martin
  Raspaud]

  Switch enhancements to yaml format
- Switch enhancements to yaml format. [Martin Raspaud]
- Fix missed Projectable use in composites. [davidh-ssec]
- Add support for segmented geostationary data. [Martin Raspaud]
- Merge pull request #43 from pytroll/msg-native. [Martin Raspaud]

  Msg native
- Possible fix for python 3.5. [Adam.Dybbroe]
- Fix for python 3.5. [Adam.Dybbroe]
- Change from relative to absolute import. [Adam.Dybbroe]
- Merge branch 'develop' into msg-native. [Adam.Dybbroe]
- Handle (nastily) cases where channel data are not available in the
  file. Add unittests. [Adam.Dybbroe]
- Merge branch 'develop' into msg-native. [Adam.Dybbroe]
- Add unittests for count to radiance calibration. [Adam.Dybbroe]
- Use 10 to 16 bit conversion function that was copied from mipp.
  [Adam.Dybbroe]
- Handle subset of SEVIRI channels Full disk supported only.
  [Adam.Dybbroe]
- Make file reading numpy 1.12 compatible. [Sauli Joro]
- Remove dependency on mipp. [Adam.Dybbroe]
- Merge branch 'develop' into msg-native. [Adam.Dybbroe]

  Conflicts:
  	satpy/readers/__init__.py
  	satpy/readers/hrit_msg.py
- Fix IR and VIS calibration. [Adam.Dybbroe]
- Pep8 and editorial (header) updates. [Adam.Dybbroe]
- Adding the native msg header record definitions. [Adam.Dybbroe]
- Semi-stable native reader version. Calibration unfinished.
  [Adam.Dybbroe]
- Unfinished msg native reader. [Adam.Dybbroe]
- Merge pull request #38 from bmu/develop. [Martin Raspaud]

  conda based install
- Reformulated the documentation again. [bmu]
- Corrected channel preferences of conda requirement file. [bmu]
- Corrected file name in documentation. [bmu]
- Renamed requirement file to reflect python and numpy version. [bmu]
- Added installation section to the docs. [bmu]
- Add vi swp files to gitignore. [bmu]
- Added environment file for conda installations. [bmu]
- Merge pull request #40 from m4sth0/develop. [Martin Raspaud]

  Add area slicing support for MTG-LI filehandler
- Add workaround for area slicing issue. [m4sth0]

  Choosing an sub area for data import in a scene objects like
  EuropeCanary results in a wrong area slice due to wrong area
  interpolation. If the lat lon values of a sub area are invalid
  (e.g. in space) the slicing gets incorrect.
  This commit will bypass this by calculating the slices directly
  without interpolation for two areas with the same projection (geos)

- Add area slicing support for MTG-LI filehandler. [m4sth0]
- Merge pull request #41 from meteoswiss-mdr/develop. [Martin Raspaud]

  Pytroll workshop --> new NWCSAF v2016 products
- Pytroll workshop --> new NWCSAF v2016 products. [sam]
- Change table of supported data types. [Adam.Dybbroe]
- Add column "shortcomings" to table of supported readers, and add row
  for native reader. [Adam.Dybbroe]
- Do not compute resampling mask for AreaDefintions. [Martin Raspaud]
- Add support for LRIT 8 bits. [Martin Raspaud]
- Cleanup HRIT readers. [Martin Raspaud]
- Add ABI composite module. [Martin Raspaud]
- Update list of supported formats. [Martin Raspaud]
- Remove uneeded code for electro reader. [Martin Raspaud]
- Add HRIT JMA reader. [Martin Raspaud]
- Merge pull request #35 from m4sth0/develop. [Martin Raspaud]

  Fix MTG-FCI and LI readers
- Fix MTG-FCI and LI readers. [m4sth0]
- Fix area extent for MSG segments. [Martin Raspaud]
- Add very basic tests for the VIIRS SDR file reader. [davidh-ssec]
- Test some utility functions. [Martin Raspaud]
- Fix tutorial. [Martin Raspaud]


v0.5.0 (2017-03-27)
-------------------
- Update changelog. [Martin Raspaud]
- Bump version: 0.4.3 → 0.5.0. [Martin Raspaud]
- Make sure calibration order is respected. [Martin Raspaud]
- Fix angles interpolation in olci reader. [Martin Raspaud]
- Fix some py3 tests. [Martin Raspaud]
- Test BaseFileHandler. [Martin Raspaud]
- Add some reader tests. [Martin Raspaud]
- Work on ABI true color. [Martin Raspaud]
- Add more VIIRS SDR tests. [davidh-ssec]
- Add a missing docstring. [Martin Raspaud]
- Refactor and test yaml_reader. [Martin Raspaud]
- Add basic VIIRS SDR file handler tests. [davidh-ssec]
- Add h5netcdf to travis. [Martin Raspaud]
- Add the ABI reader tests to main test suite. [Martin Raspaud]
- Optimize and test ABI l1b calibration functions. [Martin Raspaud]
- Add Zinke NCC algorithm to viirs DNB. [Martin Raspaud]
- Fix lunar angles names in viirs sdr. [Martin Raspaud]
- Add lunar angles support in compact viirs. [Martin Raspaud]


v0.4.3 (2017-03-07)
-------------------
- Update changelog. [Martin Raspaud]
- Bump version: 0.4.2 → 0.4.3. [Martin Raspaud]
- Add more tests to yaml_reader. [Martin Raspaud]
- Document what the Scene accepts better. [davidh-ssec]
- Remove unused FileKey class. [davidh-ssec]
- Add more tests for Scene object. [davidh-ssec]
- Fix ABI L1B area again. [davidh-ssec]
- Add Electro-L N2 HRIT reader. [Martin Raspaud]
- Fix off by one error on calculating ABI L1B pixel resolution. [davidh-
  ssec]
- Add sweep PROJ.4 parameter to ABI L1B reader. [davidh-ssec]
- Fix geos bbox to rotate in the right direction. [Martin Raspaud]
- Fix ABI L1B file patterns not working for mesos. [davidh-ssec]
- Fix tests to handle reader_kwargs and explicit sensor keyword
  argument. [davidh-ssec]
- Add reader_kwargs to Scene to pass to readers. [davidh-ssec]
- Fix yaml reader start/end time with multiple file types. [davidh-ssec]
- Allow `Scene.all_composite_ids` to return even if no sensor composite
  config. [davidh-ssec]


v0.4.2 (2017-02-27)
-------------------
- Update changelog. [Martin Raspaud]
- Bump version: 0.4.1 → 0.4.2. [Martin Raspaud]
- Merge branch 'develop' [Martin Raspaud]
- Fix area coverage test for inmporterror. [Martin Raspaud]
- Add two more tests for yaml_reader. [Martin Raspaud]
- Add more datasets for NUCAPS reader. [davidh-ssec]
- Add missing_datasets property to Scene. [davidh-ssec]

  Includes fix for trying to compute datasets after resampling that previously failed to load from readers

- Make 'view' a variable in SLSTR reader. [Martin Raspaud]
- Test available_datasets in yaml_reader. [Martin Raspaud]
- Remove NotImplementedError in abstactmethods. [Martin Raspaud]
- Test filering yaml filehandlers by area. [Martin Raspaud]
- Add yamlreader test. [Martin Raspaud]
- Fix reader test of all_dataset_ids. [davidh-ssec]
- Fix unit conversion for ABI L1B reader. [davidh-ssec]
- Fix python3 tests. [Martin Raspaud]
- Test all datasets ids and names. [Martin Raspaud]
- Fix ABI Reader to work with non-CONUS images. [davidh-ssec]
- Add unit conversion to ABI reader so generic composites work better.
  [davidh-ssec]
- Fix ABI reader area definition and file type definitions. [davidh-
  ssec]
- Change default start_time from file handler filename info. [davidh-
  ssec]
- Add `get` method to hdf5 and netcdf file handlers. [davidh-ssec]
- Fix interpolation of slstr angles. [Martin Raspaud]
- Merge pull request #31 from mitkin/feature_caliop-reader. [Martin
  Raspaud]

  Add CALIOP v3 HDF4 reader
- PEP8 fixes. [Mikhail Itkin]
- Read end_time from file metadata. [Mikhail Itkin]
- Functional CALIOP V3 HDF4 file handler. [Mikhail Itkin]
- Merge branch 'develop' of https://github.com/pytroll/satpy into
  feature_caliop-reader. [Mikhail Itkin]
- CALIOP reader WIP. [Mikhail Itkin]
- Update to caliop reader. [Mikhail Itkin]
- Add CALIOP reader (non functional yet) [Mikhail Itkin]
- Work on slstr reader. [Martin Raspaud]
- Fix small style error. [davidh-ssec]
- Change swath definition name to be more unique. [davidh-ssec]
- Fix style. [Martin Raspaud]
- Create on-the-fly name for swath definitions. [Martin Raspaud]
- Do some style cleanup. [Martin Raspaud]
- Add simple tests for scene dunder-methods and others. [davidh-ssec]

  Fix bugs that these tests encountered

- Remove osx from travis testing environments. [davidh-ssec]
- Fix amsr2 l1b reader coordinates. [davidh-ssec]
- Update link to satpy's repository. [Mikhail Itkin]

  Used to be under `mraspaud`, now `pytroll`


v0.4.1 (2017-02-21)
-------------------
- Update changelog. [davidh-ssec]
- Bump version: 0.4.0 → 0.4.1. [davidh-ssec]
- Remove forgotten print statement in tests. [davidh-ssec]
- Fix wavelength comparison when there are mixed types. [davidh-ssec]
- Remove old files. [Martin Raspaud]
- Merge pull request #30 from pytroll/feature-get-dataset-key-refactor.
  [David Hoese]

  Refactor get_dataset_key
- Merge branch 'develop' into feature-get-dataset-key-refactor. [Martin
  Raspaud]
- Rename ds id search function. [Martin Raspaud]
- Added some test to get_dataset_key refactor. [Martin Raspaud]
- Refactor get_dataset_key. [Martin Raspaud]
- Use dfilter in node. [Martin Raspaud]
- Refactor get_dataset_key wip. [Martin Raspaud]
- Use wavelength instead of channel name for NIR refl computation.
  [Martin Raspaud]
- Update contact info. [Martin Raspaud]


v0.4.0 (2017-02-21)
-------------------
- Update changelog. [davidh-ssec]
- Bump version: 0.3.1 → 0.4.0. [davidh-ssec]
- Fix composite loading when prereqs are delayed. [davidh-ssec]
- Remove randomness altogether. [Martin Raspaud]
- Reduce range of randomness for helper tests. [Martin Raspaud]
- Make PSPRayleigh modifier fail if dataset shapes don't match. [Martin
  Raspaud]
- Replace compositor name by id in log message. [Martin Raspaud]
- Remove unnecessary print statement. [Martin Raspaud]
- Remove plotting from helper_functions. [Martin Raspaud]
- Add some randomness in helper_function tests. [Martin Raspaud]
- Refactor and test helper functions for geostationary areas. [Martin
  Raspaud]
- Add masking of space pixels in AHI hsd reader. [Martin Raspaud]
- Add tests when datasets fail to load. [davidh-ssec]
- Remove redundant container specification in certain reader configs.
  [davidh-ssec]

  Now that Areas are set by coordinates and Projectables are now Datasets there is no need to customize the container a dataset uses to define it as "metadata".

- Fix composite loading when the compositor adds more information to the
  DatasetID. [davidh-ssec]
- Add new composites for AHI. [Martin Raspaud]
- Remove fast finish and py26 from travis config. [davidh-ssec]
- Fix duplicate or incorrect imports from Projectable/DatasetID
  refactor. [davidh-ssec]
- Remove Projectable class to use Dataset everywhere instead. [davidh-
  ssec]
- Merge pull request #28 from pytroll/feature-remove-id. [David Hoese]

  Remove 'id' from the info attribute in datasets and composites
- Remove to_trimmed_dict, add a kw to to_dict instead. [Martin Raspaud]
- Add id attribute to Dataset. [Martin Raspaud]
- Fix tests.utils to work with the id attribute. [Martin Raspaud]
- Remove id from infodict, wip. [Martin Raspaud]
- Fix style. [Martin Raspaud]
- Use getattr instead of if-else construct in apply_modifier_info.
  [Martin Raspaud]
- Use wavelength instead of channel name for NIR refl computation.
  [Martin Raspaud]
- Fix modifier info getting applied. [davidh-ssec]

  Now the modifiers DatasetID gets updated along with any information that can be gathered from the source

- Fix loading modified datasets that change resolution. [davidh-ssec]
- Add more Scene loading tests for composites that use wavelengths
  instead of names. [davidh-ssec]
- Fix rows_per_scan for VIIRS L1B reader and the sharpened RGB
  compositor. [davidh-ssec]
- Fix scene loading when reader dataset failed to load. [davidh-ssec]
- Add day microphysics composite to slstr. [Martin Raspaud]
- Fix reading angles for SLSTR (S3) [Martin Raspaud]
- Fix test by using DATASET_KEYS instead of DatasetID's as_dict. [Martin
  Raspaud]
- Correct some metadata in viirs_sdr. [Martin Raspaud]
- Refactor and test get_dataset_by* [Martin Raspaud]
- Merge pull request #27 from davidh-ssec/develop. [David Hoese]

  Refactor Scene dependency tree
- Add some docstrings to new deptree and compositor handling. [davidh-
  ssec]
- Fix intermittent bug where requested dataset/comp wasn't "kept" after
  loading. [davidh-ssec]

  This would happen when a composite depended on a dataset that was also requested by the user. If the composite was processed first then the dependency wasn't reprocessed, but this was incorrectly not replacing the requested `name` in the wishlist with the new `DatasetID`.

- Add tests for Scene loading. [davidh-ssec]

  Includes a few fixes for bugs that were discovered including choosing the best dataset from a DatasetDict when there are multiple matching Datasets.

- Add very basic Scene loading tests. [davidh-ssec]
- Fix behavior tests for python 3 and composite dependencies. [davidh-
  ssec]
- Move dependency logic to DependencyTree class. [davidh-ssec]
- Fix dependency tree when scene is resampled. [davidh-ssec]
- Refactor compositor loading to better handle modified
  datasets/composites. [davidh-ssec]

  Includes assigning DatasetIDs to every compositor and renaming some missed references to wavelength_range which should be wavelength.

- Fix DatasetID hashability in python 3. [davidh-ssec]

  In python 3 if __eq__ is defined then the object is automatically unhashable. I don't think we should run in to problems with a more flexible __eq__ than the hash function.

- Fix loading composite by DatasetID. [davidh-ssec]

  Includes some clean up of dependency tree, including changes to Node. Also includes adding comparison methods to the DatasetID class

- Fix `available_modifiers` [davidh-ssec]

  Required changes to how a deptree is created. Includes adding name attribute to Node class.

- Refactor name and wavelength comparison functions to top of readers
  module. [davidh-ssec]

  So they can be used outside of DatasetDict

- Added some tests for yaml_reader generic functions. [Martin Raspaud]
- Add true_color_lowres to viirs (no pan sharpening) [Martin Raspaud]
- Provide blue band to psp rayleigh correction. [Martin Raspaud]
- Add MODIS composite config. [Martin Raspaud]
- Add ABI composite config. [Martin Raspaud]
- Cleanup style in yaml_reader. [Martin Raspaud]
- Implement slicing for hrit. [Martin Raspaud]
- Cleanup abi_l1b reader. [Martin Raspaud]
- Allow get_dataset to raise KeyError to signal missing dataset in file.
  [Martin Raspaud]
- Fix geostationary boundingbox. [Martin Raspaud]
- Fill in correct wavelength for olci. [Martin Raspaud]
- Add lon and lan info for hrpt. [Martin Raspaud]
- Remove redundant file opening in hdfeos. [Martin Raspaud]
- Add forgoten unit. [Martin Raspaud]
- Fix wrong standard_name and add "overview" recipe. [Adam.Dybbroe]
- Fix NIRReflectance modifier. [Martin Raspaud]
- Update standard names and mda for hrit_msg. [Martin Raspaud]
- Add another modis filepattern. [Nina.Hakansson]
- Add python 3.6 to travis testing. [davidh-ssec]
- Update travis config to finish as soon as required environments
  finish. [davidh-ssec]
- Fix h5py reading of byte strings on python 3. [davidh-ssec]

  Was handling scalar arrays of str objects, but in python 3 they are bytes objects and weren't detected in the previous condition.

- Cleanup test_yaml_reader.py. [Martin Raspaud]
- Add tests for file selection. [Martin Raspaud]
- Document how to save custom composites. [Martin Raspaud]
- Fix VIIRS L1B reader for reflectances on v1.1+ level 1 processing
  software. [davidh-ssec]
- Fix bug in FileYAMLReader when filenames are provided. [davidh-ssec]
- Add a reader for Sentinel-2 MSI L1C data. [Martin Raspaud]
- Remove unnecessary arguments in sar-c reader. [Martin Raspaud]


v0.3.1 (2017-01-16)
-------------------
- Update changelog. [Martin Raspaud]
- Bump version: 0.3.0 → 0.3.1. [Martin Raspaud]
- Cleanup SAR-C. [Martin Raspaud]
- Add annotations loading for sar-c. [Martin Raspaud]
- Merge pull request #22 from mitkin/feature-sar-geolocation. [Martin
  Raspaud]

  Feature SAFE (Sentinel 1) SAR geolocation
- Refactor coordinates computation. [Mikhail Itkin]

  Refactor changes for pull request #22

- Merge branch 'develop' of https://github.com/mitkin/satpy into
  feature-sar-geolocation. [Mikhail Itkin]
- Make Sentinel 1 (SAFE) reader able to read coordinates. [Mikhail
  Itkin]

  Add latitude and longitude dictionaries to the `sar_c.yaml` reader
  and make the `safe_sar_c.py` reader compute coordinate arrays from
  a collection of GCPs provided in the measurement files.

  NB: each polarization has it's set of longitudes and latitudes.

- Restore reducers to their original values. [Martin Raspaud]
- Add alternatives for true color on ahi. [Martin Raspaud]

  Thanks balt
- Add name to the dataset attributes when writing nc files. [Martin
  Raspaud]
- Improve documentation. [Martin Raspaud]
- Add proper enhancements for nwcsaf images. [Martin Raspaud]
- Refactor hrit msg area def computation. [Martin Raspaud]
- Perform som PEP8 cleanup. [Martin Raspaud]
- Fix nwcsaf reader and its area definition. [Martin Raspaud]
- Merge pull request #21 from mitkin/develop. [David Hoese]

  Mock pyresample.ewa
- Mock pyresample.ewa. [Mikhail Itkin]

  Mock pyresample.ewa to prevent sphinx from importing the module.
- Add NWCSAF MSG nc reader and composites. [Martin Raspaud]
- Add gamma to the sarice composite. [Martin Raspaud]
- Cleanup the sar composite. [Martin Raspaud]
- Add the sar-ice composite. [Martin Raspaud]
- Clean up the safe sar-c reader. [Martin Raspaud]
- Finalize MSG HRIT calibration. [Martin Raspaud]
- Fix abi reader copyright. [Martin Raspaud]
- Refactor yaml_reader's create_filehandlers. [Martin Raspaud]
- Rename function. [Martin Raspaud]
- Add a composite file for slstr. [Martin Raspaud]
- Add a noaa GAC/LAC reader using PyGAC. [Martin Raspaud]
- Implement a mipp-free HRIT reader. [Martin Raspaud]

  WIP, supports only MSG, no calibration yet.
- Concatenate area_def through making new AreaDefinition. [Martin
  Raspaud]

  This makes the concatenation independent of the AreaDefinition
  implementation.
- Allow stacking area_def from bottom-up. [Martin Raspaud]
- Fix yaml_reader testing. [Martin Raspaud]
- Add support for filetype requirements. [Martin Raspaud]
- Remove print statement in slstr reader. [Martin Raspaud]
- Remove deprecated helper functions. [Martin Raspaud]
- Refactor select_files, yaml_reader. [Martin Raspaud]
- Editorials. [Adam.Dybbroe]
- Add coastline overlay capability. [Martin Raspaud]
- Move the Node class to its own module. [Martin Raspaud]
- Initialize angles in epsl1b reader. [Martin Raspaud]
- Add angles reading to eps reader. [Martin Raspaud]


v0.3.0 (2016-12-13)
-------------------
- Update changelog. [Martin Raspaud]
- Bump version: 0.2.1 → 0.3.0. [Martin Raspaud]
- Fix NUCAPS reader to work with latlon datasets. [davidh-ssec]

  This required changing yaml_reader to work with 1D arrays since NUCAPS is all 1D (both swath data and metadata).

- Refactor yaml_reader's load method. [Martin Raspaud]
- Merge branch 'develop' into feature-lonlat-datasets. [Martin Raspaud]
- Fix VIIRS L1B reader to work with xslice/yslice and fix geolocation
  dataset names. [davidh-ssec]
- Fix netcdf wrapper to work better with older and newer versions of
  netcdf4-python. [davidh-ssec]
- Make ahi reader use correct default slicing. [Martin Raspaud]
- Bugfix sliced reading. [Martin Raspaud]
- Put slice(None) as default for reading. [Martin Raspaud]
- Allow readers not supporting slices. [Martin Raspaud]
- Refactor scene's init. [Martin Raspaud]
- Convert nucaps to coordinates. [Martin Raspaud]
- Adapt viirs_l1b to coordinates. [Martin Raspaud]
- Convert omps reader to coordinates. [Martin Raspaud]
- Reinstate viirs_sdr.yaml for coordinates, add standard_names. [Martin
  Raspaud]
- Adapt compact viirs reader to coordinates. [Martin Raspaud]
- Add first version of S1 Sar-c reader. [Martin Raspaud]
- Adapt olci reader to coordinates. [Martin Raspaud]
- Add S3 slstr reader. [Martin Raspaud]
- Add standard_names to hdfeos navigation. [Martin Raspaud]
- Fix epsl1b reader for lon/lat standard_name. [Martin Raspaud]
- Adapt amsr2 reader for coordinates. [Martin Raspaud]
- Fix aapp1b reader. [Martin Raspaud]
- Use standard name for lon and lat identification. [Martin Raspaud]
- Merge branch 'develop' into feature-lonlat-datasets. [Martin Raspaud]

  Conflicts:
  	satpy/readers/ahi_hsd.py

- Area loading for ahi_hsd. [Martin Raspaud]
- Fix python3 syntax incompatibility. [Martin Raspaud]
- Implement area-based loading. [Martin Raspaud]
- Add get_bounding_box for area-based file selection. [Martin Raspaud]
- Fix ahi area extent. [Martin Raspaud]
- Merge remote-tracking branch 'origin/feature-lonlat-datasets' into
  feature-lonlat-datasets. [Martin Raspaud]
- Convert VIIRS SDR reader to coordinates. [davidh-ssec]
- Fix viirs_sdr i bands to work with coordinates. [davidh-ssec]
- Support different path separators in patterns. [Martin Raspaud]
- Move area def loading to its own function. [Martin Raspaud]
- Merge branch 'develop' into feature-lonlat-datasets. [Martin Raspaud]

  Conflicts:
  	satpy/readers/yaml_reader.py
- Merge branch 'develop' into feature-lonlat-datasets. [Martin Raspaud]

  Conflicts:
  	satpy/readers/yaml_reader.py
- Pass down the calibration, polarization and resolution from main load.
  [Martin Raspaud]
- Fix typo in sunzenith correction description. Default is 88 deg, not
  80. [Adam.Dybbroe]
- Fix sun zenith key for caching. [Martin Raspaud]
- Move helper functions to readers directory. [Martin Raspaud]
- Adapt hrpt reader to coordinates. [Martin Raspaud]
- Fix resample to work when the area has no name. [Martin Raspaud]
- Adapt aapp_l1b and hdfeos to coordinates. [Martin Raspaud]
- Change remove arguments from get_area_def signature. [Martin Raspaud]
- Adapt eps_l1b to 'coordinates' [Martin Raspaud]
- Navigation is now handled thru 'coordinates' [Martin Raspaud]

  Here we make longitude and latitudes usual datasets, and the keyword
  called 'coordinates' in the config specifies the coordinates to use for
  the dataset at hand.


v0.2.1 (2016-12-08)
-------------------
- Update changelog. [Martin Raspaud]
- Bump version: 0.2.0 → 0.2.1. [Martin Raspaud]
- Move ghrsst_osisaf.yaml to new location. [Martin Raspaud]
- Remove old mpop legacy files. [Martin Raspaud]
- Move etc to satpy, use package_data for default config files. [Martin
  Raspaud]
- Merge pull request #19 from adybbroe/osisaf_sst_reader. [Martin
  Raspaud]

  Add OSISAF SST GHRSST reader
- Add OSISAF SST GHRSST reader. [Adam.Dybbroe]
- Replace memmap with fromfile in ahi hsd reading. [Martin Raspaud]
- Merge branch 'develop' of github.com:pytroll/satpy into develop.
  [Adam.Dybbroe]
- Merge pull request #18 from northaholic/develop. [Martin Raspaud]

  improve FCI reader readability. fix FCI reader config for WV channels.
- Improve FCI reader readability. fix FCI reader config for WV channels.
  [Sauli Joro]
- Merge pull request #17 from m4sth0/develop. [Martin Raspaud]

  Add MTG LI reader
- Add MTG-LI L2 reader for preliminary test data. [m4sth0]
- Merge branch 'develop' of https://github.com/pytroll/satpy into
  develop. [m4sth0]
- Merge branch 'develop' of https://github.com/pytroll/satpy into
  develop. [m4sth0]
- Solve compatibility problem with older netCDF4 versions.
  [Adam.Dybbroe]
- Fix style in abi reader. [Martin Raspaud]
- Add ABI reader + YAML. [Guido Della Bruna]
- Merge pull request #15 from m4sth0/develop. [Martin Raspaud]

  Develop
- Merge branch 'develop' of https://github.com/pytroll/satpy into
  develop. [m4sth0]
- Fixed FCI channel calibration method. [m4sth0]
- Fix VIIRS L1B moon illumination fraction for L1B v2.0. [davidh-ssec]

  In NASA Level 1 software version <2.0 the fraction was a global attribute, now in v2.0 it is a per-pixel swath variable

- Fix DNB SZA and LZA naming to match viirs composite configs. [davidh-
  ssec]
- Fix start_time/end_time creation in Scene when no readers found.
  [davidh-ssec]
- Merge pull request #14 from m4sth0/develop. [Martin Raspaud]

  Add calibration functions for FCI
- Add calibration functions for FCI. [m4sth0]
- Bugfix. [Adam.Dybbroe]
- Bugfix. [Adam.Dybbroe]
- Editorial pep8/pylint. [Adam.Dybbroe]
- Merge pull request #13 from m4sth0/develop. [Martin Raspaud]

  Add MTG-FCI Level 1C netCDF reader
- Add MTG-FCI Level 1C netCDF reader The test dataset from EUMETSAT for
  the FCI Level 1C Format Familiarisation is used to implement the
  reader in satpy. Limitations due to missing meta data for satellite
  georeferencing and calibration. [m4sth0]
- Pass down the calibration, polarization and resolution from main load.
  [Martin Raspaud]
- Fix typo in sunzenith correction description. Default is 88 deg, not
  80. [Adam.Dybbroe]
- Move helper functions to readers directory. [Martin Raspaud]
- Fix Scene sensor metadata when it is a string instead of a list.
  [davidh-ssec]
- Fix start_time/end_time properties on Scene object after resampling.
  [davidh-ssec]

  These properties were dependent on scn.readers which doesn't exist after resampling creates a new "copy" of the original Scene. Now these values are part of the metadata in .info and set on init.

- Replace errors with warnings when loading dependencies. [davidh-ssec]


v0.2.0 (2016-11-21)
-------------------

Fix
~~~
- Bugfix: converted MSG products should be saveable. [Martin Raspaud]
- Bugfix: satellite name in msg_hdf now supports missing number. [Martin
  Raspaud]
- Bugfix: misspelling. [Martin Raspaud]
- Bugfix: mipp_xrit: do not crash on unknown channels, just warn and
  skip. [Martin Raspaud]
- Bugfix: changed reference from composites.cfg to
  composites/generic.cfg. [Martin Raspaud]
- Bugfix: works now for file auto discovery. [Martin Raspaud]
- Bugfix: get_filename wants a reader_instance and cleanup. [Martin
  Raspaud]
- Bugfix: setup.py includes now eps xml format description. [Martin
  Raspaud]
- Close all h5files in viirs_sdr, not only the last one.
  [Martin.Raspaud]
- Bugfix: close h5 files when done. [Martin Raspaud]

  Prior to h5py 3.0, the h5 files open with h5py are not closed upon
  deletion, so we have to do it ourselves...
- Bugfix: area.id doesn't exist, use area.area_id. [Martin Raspaud]
- Bugfix: return when each file has been loaded independently. [Martin
  Raspaud]
- Bugfix: Do not crash on multiple non-nwc files. [Martin Raspaud]
- Bugfix: check start and end times from loaded channels only. [Martin
  Raspaud]
- Bugfix: viirs start and end times not relying on non-existant channels
  anymore. [Martin Raspaud]
- Bugfix: type() doesn't support unicode, cast to str. [Martin Raspaud]
- Bugfix: allow more than one "-" in section names. [Martin Raspaud]
- Bugfix: read aqua/terra orbit number from file only if not already
  defined. [Martin Raspaud]
- Bugfix: fixed unittest case for wavelengths as lists. [Martin Raspaud]
- Bugfix: remove deprecated mviri testcases. [Martin Raspaud]
- Bugfix: backward compatibility with netcdf files. [Martin Raspaud]
- Bugfix: removed the old mviri compositer. [Martin Raspaud]
- Bugfix: When assembling, keep track of object, not just lon/lats.
  [Martin Raspaud]
- Bugfix: assembling scenes would unmask some lon/lats... [Martin
  Raspaud]
- Bugfix: handling of channels with different resolutions in
  assemble_segments. [Martin Raspaud]
- Bugfix: Runner crashed if called with an area not in product list.
  [Martin Raspaud]
- Bugfix: the nwcsaf_pps reader was crashing if no file was found...
  [Martin Raspaud]
- Bugfix: pynav is not working in some cases, replace with pyorbital.
  [Martin Raspaud]
- Bugfix: can now add overlay in monochromatic images. [Martin Raspaud]
- Bugfix: swath scene projection takes forever from the second time.
  [Martin Raspaud]

  The swath scene, when projected more than once would recompute the nearest neighbours for every channel.

- Bugfix: importing geotiepoints. [Martin Raspaud]
- Bugfix: hdfeos was not eumetcast compliant :( [Martin Raspaud]
- Bugfix: Do not raise exception on loading failure (nwcsaf_pps) [Martin
  Raspaud]
- Bugfix: fixed misc bugs. [Martin Raspaud]
- Bugfix: comparing directories with samefile is better than ==. [Martin
  Raspaud]
- Bugfix: updating old eps_l1b interface. [Martin Raspaud]
- Bugfix: Fixed typo in gatherer. [Martin Raspaud]
- Bugfix: taking satscene.area into consideration for get_lonlat.
  [Martin Raspaud]
- Bugfix: mipp required version to 0.6.0. [Martin Raspaud]
- Bugfix: updating unittest and setup for new mipp release. [Martin
  Raspaud]
- Bugfix: for eps l1b, get_lonlat did not return coherent values since
  the introduction of pyresample. [Martin Raspaud]
- Bugfix: mipp to mipp_xrit namechange. [Martin Raspaud]
- Bugfix: better detection of needed channels in aapp1b. [Martin
  Raspaud]
- Bugfix: support for other platforms. [Martin Raspaud]
- Bugfix: Support python 2.4 in mipp plugin. [Martin Raspaud]
- Bugfix: masked arrays should be conserved by scene.__setitem__ [Martin
  Raspaud]
- Bugfix: Don't make area and time_slot static in compositer. [Martin
  Raspaud]
- Bugfix: reinit channels_to_load and messages for no loading. [Martin
  Raspaud]

  - When the loading process is interrupted, the channels_to_load attribute was not reinitialized.
  - Added a message when loading for a given level did not load anything.

- Bugfix: Give an informative message when area is missing for msg's hdf
  reader. [Martin Raspaud]
- Bugfix: update satpos file retrieval for hrpt and eps1a. [Martin
  Raspaud]
- Bugfix: fixed unittests for new plugin system. [Martin Raspaud]
- Bugfix: Do not load plugins automatically... [Martin Raspaud]
- Bugfix: satellite vs satname again. [Martin Raspaud]
- Bugfix: don't crash if msg hdf can't be loaded. [Martin Raspaud]
- Bugfix: project now chooses mode automatically by default. [Martin
  Raspaud]
- Bugfix: eps_avhrr adapted to new plugin format. [Martin Raspaud]
- Bugfix: loading in msg_hdf adapted to new plugin system. [Martin
  Raspaud]
- Bugfix: loading plugins should fail on any exception. [Martin Raspaud]
- Bugfix: stupid syntax error. [Martin Raspaud]
- Bugfix: mistook satname for satellite. [Martin Raspaud]
- Bugfix: move to jenkins. [Martin Raspaud]
- Bugfix: affecting area to channel_image. [Martin Raspaud]
- Bugfix: Better handling of alpha channel. [Martin Raspaud]
- Bugfix: filewatcher would wait a long time if no new file has come.
  [Martin Raspaud]
- Bugfix: netcdf saving didn't record lat and lon correctly. [Martin
  Raspaud]
- Bugfix: netcdf saving didn't work if only one value was available.
  [Martin Raspaud]
- Bugfix: test_mipp had invalid proj parameters. [Martin Raspaud]
- Bugfix: satellite vs satname again. [Martin Raspaud]
- Bugfix: project now chooses mode automatically by default. [Martin
  Raspaud]
- Bugfix: move to jenkins. [Martin Raspaud]
- Bugfix: fixed unit test for projector reflecting the new mode
  handling. [Martin Raspaud]
- Bugfix: fixed None mode problem in projector. [Martin Raspaud]
- Bugfix: The default projecting mode now take into account the types of
  the in and out areas. [Martin Raspaud]
- Bugfix: forgot the argument to wait in filewatcher. [Martin Raspaud]
- Bugfix: tags and gdal_options were class attributes, they should be
  instance attributes. [Martin Raspaud]
- Bugfix: 0 reflectances were masked in aapp1b loader. [Martin Raspaud]
- Bugfix: corrected parallax values as no_data in msg products reading.
  [Martin Raspaud]
- Bugfix: tags and gdal_options were class attributes, they should be
  instance attributes. [Martin Raspaud]
- Bugfix: Compatibility with nordrad was broken. [Martin Raspaud]
- Bugfix: forgot the argument to wait in filewatcher. [Martin Raspaud]
- Bugfix: forgot strptime = datetime.strptime when python > 2.5. [Martin
  Raspaud]
- Bugfix: corrected parallax values as no_data in msg products reading.
  [Martin Raspaud]
- Bugfix: individual channel areas are preserved when assembled
  together. [Martin Raspaud]
- Bugfix: cleanup tmp directory when convertion to lvl 1b is done.
  [Martin Raspaud]
- Bugfix: remove hardcoded pathes in hrpt and eps lvl 1a. [Martin
  Raspaud]
- Bugfix: use mpop's main config path. [Martin Raspaud]
- Bugfix: added python 2.4 compatibility. [Martin Raspaud]
- Bugfix: allow all masked array as channel data. [Martin Raspaud]
- Better support for channel-bound areas. [Martin Raspaud]
- Bugfix: 0 reflectances were masked in aapp1b loader. [Martin Raspaud]
- Bugfix: tags and gdal_options were class attributes, they should be
  instance attributes. [Martin Raspaud]
- Bugfix: error checking on area_extent for loading. [Martin Raspaud]
- Bugfix: non loaded channels should not induce computation of
  projection. [Martin Raspaud]
- Bugfix: thin modis didn't like area extent and was locked in 2010...
  [Martin Raspaud]
- Bugfix: Compatibility with nordrad was broken. [Martin Raspaud]
- Bugfix: fixed matching in git command for version numbering. [Martin
  Raspaud]
- Bugfix: Negative temperatures (in K) should not be valid data when
  reading aapp1b files. [Martin Raspaud]
- Bugfix: remove hudson from tags when getting version. [Martin Raspaud]
- Bugfix: fixed hdf inconstistencies with the old pyhl reading of msg
  ctype and ctth files. [Martin Raspaud]
- Bugfix: Updated code and tests to validate unittests. [Martin Raspaud]
- Bugfix: data reloaded even if the load_again flag was False. [Martin
  Raspaud]
- Bugfix: updated tests for disapearance of avhrr.py. [Martin Raspaud]
- Bugfix: access to CompositerClass would fail if using the old
  interface. [Martin Raspaud]
- Bugfix: typesize for msg's ctth didn't please pps... [Martin Raspaud]
- Bugfix: fixed data format (uint8) in msg_hdf. [Martin Raspaud]
- Bugfix: wrong and forgotten instanciations. [Martin Raspaud]
- Bugfix: crashing on missing channels in mipp loading. [Martin Raspaud]
- Bugfix: forgot to pass along area_extent in mipp loader. [Martin
  Raspaud]
- Bugfix: fixing integration test (duck typing). [Martin Raspaud]
- Bugfix: pyresample.geometry is loaded lazily for area building.
  [Martin Raspaud]
- Bugfix: Updated unit tests. [Martin Raspaud]
- Bugfix: Last change introduced empty channel list for meteosat 09.
  [Martin Raspaud]
- Bugfix: Last change introduced empty channel list for meteosat 09.
  [Martin Raspaud]
- Bugfix: update unittests for new internal implementation. [Martin
  Raspaud]
- Bugfix: compression argument was wrong in
  satelliteinstrumentscene.save. [Martin Raspaud]
- Bugfix: adapted mpop to new equality operation in pyresample. [Martin
  Raspaud]
- Bugfix: More robust config reading in projector and test_projector.
  [Martin Raspaud]
- Bugfix: updated the msg_hrit (nwclib based) reader. [Martin Raspaud]
- Bugfix: swath processing was broken, now fixed. [Martin Raspaud]
- Bugfix: corrected the smaller msg globe area. [Martin Raspaud]
- Bugfix: Erraneous assumption on the position of the 0,0 lon lat in the
  seviri frame led to many wrong things. [Martin Raspaud]
- Bugfix: introduced bugs in with last changes. [Martin Raspaud]
- Bugfix: new area extent for EuropeCanary. [Martin Raspaud]
- Bugfix: Updated setup.py to new structure. [Martin Raspaud]
- Bugfix: updated integration test to new structure. [Martin Raspaud]
- Bugfix: more verbose crashing when building extensions. [Martin
  Raspaud]
- Bugfix: corrected EuropeCanary region. [Martin Raspaud]
- Bugfix: made missing areas message in projector more informative
  (includes missing area name). [Martin Raspaud]
- Bugfix: Added missing import in test_pp_core. [Martin Raspaud]
- Bugfix: fixing missing import in test_scene. [Martin Raspaud]
- Bugfix: geotiff images were all saved with the wgs84 ellipsoid even
  when another was specified... [Martin Raspaud]
- Bugfix: Corrected the formulas for area_extend computation in geos
  view. [Martin Raspaud]
- Bugfix: satellite number in cf proxy must be an int. Added also
  instrument_name. [Martin Raspaud]
- Bugfix: Erraneous on the fly area building. [Martin Raspaud]
- Bugfix: geo_image: gdal_options and tags where [] and {} by default,
  which is dangerous. [Martin Raspaud]
- Bugfix: Support for new namespace for osr. [Martin Raspaud]
- Bugfix: remove dubble test in test_channel. [Martin Raspaud]
- Bugfix: showing channels couldn't handle masked arrays. [Martin
  Raspaud]
- Bugfix: Scen tests where wrong in project. [Martin Raspaud]
- Bugfix: when loading only CTTH or CloudType, the region name was not
  defined. [Martin Raspaud]
- Bugfix: in test_channel, Channel constructor needs an argument.
  [Martin Raspaud]
- Bugfix: in test_cmp, tested GenericChannel instead of Channel. [Martin
  Raspaud]
- Bugfix: Test case for channel initialization expected the wrong error
  when wavelength argument was of the wrong size. [Martin Raspaud]
- Bugfix: Added length check for "wavelength" channel init argument.
  [Martin Raspaud]
- Bugfix: test case for channel resolution did not follow previous patch
  allowing real resolutions. [Martin Raspaud]
- Bugfix: thin modis lon/lat are now masked arrays. [Martin Raspaud]
- Bugfix: in channel constructor, wavelength triplet was not correctly
  checked for type. [Martin Raspaud]

  Just min wavelength was check three times.


Other
~~~~~
- Update changelog. [Martin Raspaud]
- Bump version: 0.1.0 → 0.2.0. [Martin Raspaud]
- Fix version number. [Martin Raspaud]
- Do not fill lon and lat masks with random values. [Martin Raspaud]
- Fix AHI reading for new rayleigh correction. [Martin Raspaud]
- Add some modifiers for AHI. [Martin Raspaud]
- Adjust to requesting rayleigh correction by wavelength. [Martin
  Raspaud]
- Add rayleigh modifier to visir. [Martin Raspaud]
- Add angles reading to nc_olci. [Martin Raspaud]
- Add pyspectral's generic rayleigh correction. [Martin Raspaud]
- Fix cosmetics in scene.py. [Martin Raspaud]
- Remove memmap from eps_l1b, use fromfile instead. [Martin Raspaud]

  This was triggering a `Too many open files` error since the memmap was
  called for every scanline.
- Fix loading for datasets with no navigation. [Martin Raspaud]
- Read start and end time from filename for eps_l1b. [Martin Raspaud]

  This avoids opening every file just for time checks.
- Rename file handler's get_area to get_lonlats. [davidh-ssec]

  There is now a get_area_def and get_lonlats method on individual file handlers

- Fix start/end/area parameters in FileYAMLReader. [davidh-ssec]
- Move start_time, end_time, area parameters to reader init instead of
  load. [davidh-ssec]

  Scenes do not change start_time, end_time, area after init so neither should readers. Same treatment is probably needed for 'sensors'.

- Fix avhrr reading. [Martin Raspaud]
- Add amsr2 composite config file. [Martin Raspaud]
- Adjust OLCI reader for reflectance calibration. [Martin Raspaud]
- Delete old reader .cfg config files that are no longer used. [davidh-
  ssec]
- Add forgotten OMPS yaml file. [davidh-ssec]
- Convert OMPS reader from .cfg/INI to YAML. [davidh-ssec]
- Provide better warning message when specified reader can't be found.
  [davidh-ssec]
- Clean up class declarations in viirs l1b yaml. [davidh-ssec]
- Fix VIIRS L1B inplace loading. [davidh-ssec]
- Remove duplicate units definition in nucaps reader. [davidh-ssec]
- Add standard_name and units to nucaps reader. [davidh-ssec]
- Convert nucaps reader to yaml. [davidh-ssec]
- Remove `dskey` from reader dataset ID dictionary. [davidh-ssec]

  The section name for each dataset was not used except to uniquely identify one dataset 'variation' from another similar dataset. For example you could technically have two sections for each calibration of a single dataset. YAML would require a different section name for each of these, but it is not used inside of satpy's readers because the `name` and DatasetID are used for that purpose.

- Rename 'navigation' section in reader configs to 'navigations'
  [davidh-ssec]

  More consistent and grammatically correct with file_types and datasets

- Rename 'corrector' and 'correction' modifiers to 'corrected' [davidh-
  ssec]

  Modifier names are applied to DatasetIDs so it was decided that 'corrected' may sound better in the majority of cases than 'corrector'.

- Add .info dictionary to SwathDefinition created by YAML Reader.
  [davidh-ssec]
- Fix standard_name of natural_color composite for VIIRS. [davidh-ssec]
- Add ratio sharpened natural color for VIIRS. [davidh-ssec]
- Rename VIIRSSharpTrueColor to RatioSharpenedRGB. [davidh-ssec]

  This includes making the ratio sharpened true color the default for VIIRS under the name 'true_color'

- Fix tuple expansion in sunz corrector. [davidh-ssec]
- Rename I and DNB angle datasets to reflect M band naming. [davidh-
  ssec]
- Allow including directories in file patterns. [Martin Raspaud]
- Add navigation to olci reader. [Martin Raspaud]
- Add support for OLCI format reading. [Martin Raspaud]
- Cleanup SunZenithCorrector. [Martin Raspaud]
- Remove some TODOs. [Martin Raspaud]
- Fix some seviri composites. [Martin Raspaud]
- Add mipp config file for MSG3. [Martin Raspaud]

  This is needed by mipp when the mipp_hrit reader is used.
- Remove `if True` from viirs sharp true color. [davidh-ssec]
- Fix small bug in scene when dataset isn't found in a reader. [davidh-
  ssec]
- Update VIIRS sharpened true color to be more flexible when upsampling.
  [davidh-ssec]
- Refactor composite config loading to allow interdependent modifiers.
  [Martin Raspaud]
- Add configuration files for HRIT H8 loading. [Martin Raspaud]
- Pass platform_name to mipp for prologue-less hrit formats. [Martin
  Raspaud]
- Provide satellite position information on load (HSD) [Martin Raspaud]
- Put AHI HSD reflectances in % [Martin Raspaud]

  They were between 0 and 1 by default
- Fix AHI HSD nav dtype. [Martin Raspaud]

  lon ssp and lat ssp where swaped
- Adjust correct standard names for seviri calibration. [Martin Raspaud]
- Fix Seviri CO2 correction buggy yaml def. [Martin Raspaud]
- Fix sunz corrector with different resolutions. [davidh-ssec]

  Includes fix to make sure composites from user-land will overwrite builtin composites.

- Update VIIRS L1B LUT variable path construction to be more flexible.
  [davidh-ssec]
- Add recursive dict updating to yaml reader configs. [davidh-ssec]

  Before this only the top level values would be updated as a whole which wasn't really the intended function of having multiple config files.

- Fix coords2area_def with rounding of x and y sizes. [Martin Raspaud]
- Fix cos zen normalisation (do not use datetime64) [Martin Raspaud]
- Fix start and end time format to use datetime.datetime. [Martin
  Raspaud]
- Add IMAPP file patterns to HDFEOS L1B reader. [davidh-ssec]
- Fix hdfeos_l1b due to missing get_area_def method. [davidh-ssec]

  The HDFEOS file handlers weren't inheriting the proper base classes

- Add sunz_corrector modifier to viirs_sdr reader. [davidh-ssec]
- Fix available_dataset_names when multiple file types are involved.
  [davidh-ssec]

  Also includes a clean up of the available_dataset_names by not providing duplicates (from multiple calibrations and resolutions)

- Allow multiple file types in yaml reader. [davidh-ssec]
- Add VIIRS SDR M-band angles and DNB angles. [davidh-ssec]
- Add VIIRS SDR reader back in [WIP] [davidh-ssec]

  I've added all the M and I bands, but need to add DNB and the various angle measurements that we use a lot. Also need to add the functionality to load/find the geolocation files from the content in the data files.

- Add reader_name and composites keywords to all/available_dataset_names
  methods. [davidh-ssec]
- Fix available_dataset_ids and all_dataset_ids methods. [davidh-ssec]

  There are not `(all/available)_dataset_(ids/names)` methods on the Scene object. Includes a fix for available composites.

- Fix multiple load calls in Scene. [davidh-ssec]

  This isn't technically a supported feature, but it was a simple fix to get it to work for my case.

- Fix compositor loading when optional_prerequisites are more than a
  name. [davidh-ssec]
- Update coord2area_def to be in sync with the mpop version. [Martin
  Raspaud]
- Fix seviri.yaml for new prerequisite syntax. [Martin Raspaud]
- Fix EPSG info in geotiffs. [Martin Raspaud]
- Adjust crefl for python 3 compatibility. [Martin Raspaud]
- Merge branch 'new_prereq_syntax' into feature-yaml. [Martin Raspaud]

  Conflicts:
  	etc/composites/viirs.yaml
  	etc/composites/visir.yaml
  	satpy/composites/__init__.py
  	satpy/scene.py
- Add support for new prerequisite syntax. [Martin Raspaud]
- Got VIIRS L1B True color working. [davidh-ssec]

  Still need work on sharpened true color when I01 is used for ratio sharpening.

- Remove unneeded quotes for python names in yaml files. [Martin
  Raspaud]
- Merge branch 'feature-ahi-no-navigation' into feature-yaml. [Martin
  Raspaud]

  Conflicts:
  	etc/composites/viirs.yaml
  	satpy/readers/yaml_reader.py
- Add viirs composites. [Martin Raspaud]
- Fix the area_def concatenation. [Martin Raspaud]
- Mask nan in ir calibration for ahi hsd. [Martin Raspaud]
- Fix out of place loading, by not using a shuttle. [Martin Raspaud]
- Make get_area_def a default method of file_handlers. [Martin Raspaud]
- Allow file handler to provide area defs instead of swath. [Martin
  Raspaud]

  This is enabled by implementing the `get_area_def` method in the file
  handler.
- Optimize AHI reading using inplace loading. [Martin Raspaud]

  Navigation is switched off for now.
- Allow area loading for the data file handlers. [Martin Raspaud]
- Use a named tuple to pass both data, mask and info dict for inplace
  loading. [Martin Raspaud]
- Fix AreaID name to AreaID. [Martin Raspaud]
- Fix AreaID name to AreaID. [Martin Raspaud]
- Add moon illumination fraction and DNB enhancements for VIIRS.
  [davidh-ssec]

  MIF needed some edits to how the reader works since it returns a Dataset (no associated navigation)

- Add other basic datasets to VIIRS L1B. [davidh-ssec]

  I only had I01 and I04 for testing, not has all I, M, and DNB datasets.

- Add enhancements configuration directory to the setup.py data_files.
  [davidh-ssec]
- Complete AHI HSD reader. [Martin Raspaud]
- Fix missing dependency and python3 compatibility in ahi_hsd. [Martin
  Raspaud]
- Add skeleton for Himawari AHI reading. [Martin Raspaud]
- Add a NIR reflectance modifier using pyspectral. [Martin Raspaud]
- Add some metadata to projectables in viirs compact. [Martin Raspaud]
- Fix optional prerequisites loading. [Martin Raspaud]
- Raise an IncompatibleArea exception on RGBCompositor. [Martin Raspaud]
- Look for local files even if base_dir and filenames are missing.
  [Martin Raspaud]
- Allow empty scene creation when neither filenames nor base_dir is
  provided. [Martin Raspaud]
- Handle incompatible areas when reading composites. [Martin Raspaud]
- Remove dead code. [Martin Raspaud]
- Add debug information in viirs compact. [Martin Raspaud]
- Get dataset key from calibration in correct order. [Martin Raspaud]
- Raise exception when no files are found. [Martin Raspaud]
- Add DNB to viirs compact. [Martin Raspaud]
- Remove old mpop legacy files. [Martin Raspaud]
- Make viirs_compact python 3 compatible. [Martin Raspaud]
- Move xmlformat.py to the readers directory, and remove a print
  statement. [Martin Raspaud]
- Fix EPSG projection definition saving to geotiff. [Martin Raspaud]
- Remove python 3 incompatible syntax (Tuple Parameter Unpacking)
  [Martin Raspaud]
- Fix crefl further to lower memory consumption. [Martin Raspaud]
- Avoid raising an error when no files are found. [Martin Raspaud]

  Instead, a warning is logged.
- Remove unused code from readers/__init__.py. [Martin Raspaud]
- Cleanup style. [Martin Raspaud]
- Fix unittests. [Martin Raspaud]
- Deactivate viirssdr testing while migrating to yaml. [Martin Raspaud]
- Refactor parts of compact viirs reader. [Martin Raspaud]
- Optimize memory for crefl computation. [Martin Raspaud]
- Allow sunz corrector to be provided the sunz angles. [Martin Raspaud]
- Make chained modifiers work. [Martin Raspaud]
- Cleanup style. [Martin Raspaud]
- Add a crefl modifier for viirs. [Martin Raspaud]
- Add loading of sun-satellite/sensor viewing angles to aapp-l1b reader.
  [Adam.Dybbroe]
- Add sensor/solar angles loading to compact viirs reader. [Martin
  Raspaud]
- Allow modifier or composites sections to be missing from config.
  [Martin Raspaud]
- Fix some composites. [Martin Raspaud]
- Port VIIRS Compact M-bands to yaml. [Martin Raspaud]
- Add modifiers feature. [Martin Raspaud]

  Now modifiers can be added to the prerequisites as dictionnaries.
- Add standard_names to channels in mipp_xrit. [Martin Raspaud]
- Add a NC4/CF writer. [Martin Raspaud]
- Use YAML instead of CFG for composites. [Martin Raspaud]
- Rename wavelength_range to wavelength in reader configs. [davidh-ssec]

  Also rewrote other yaml configs to use new dict identifiers

- Add YAML based VIIRS L1B reader (I01 and I04 only) [davidh-ssec]
- Allow dict identifiers in reader's datasets config. [davidh-ssec]

  Some metadata (standard_name, units, etc) are dependent on the calibration, resolution, or other identifying piece of info. Now these make it easier to fully identify a dataset and the multiple ways it may exist. This commit also includes small fixes for how `get_shape` is called and fixes for the netcdf4 handler to match past changes.

- Fix numpy warnings when assigning to masked arrays. [davidh-ssec]
- Add pyyaml to setup.py requires. [davidh-ssec]
- Make base file handler and abstract base class. [davidh-ssec]

  Also changed start_time and end_time to properties of the file handlers

- Make AbstractYAMLReader an actual ABCMeta abstract class. [davidh-
  ssec]
- Fix ReaderFinder when all provided filenames have been found. [davidh-
  ssec]

  Also fixed mipp_xrit reader which was providing the set of files that matched rather than the set of files that didn't match. Added start and end time to the xrit reader too.

- Rename YAMLBasedReader to FileYAMLReader. [davidh-ssec]

  As in it is a YAML Based Reader that accepts files where a dataset is not separated among multiple files.

- Merge remote-tracking branch 'origin/feature-yaml' into feature-yaml.
  [davidh-ssec]
- Port EPS l1b reader to yaml. [Martin Raspaud]
- Combine areas also in combine_info. [Martin Raspaud]
- Port mipp xrit reader to yaml. [Martin Raspaud]
- Split YAMLBasedReader to accomodate for derivatives. [Martin Raspaud]

  Some file formats split a dataset on multiple files, a situation which is
  not covered by the YAMLBasedReader. Some parts of the class being still
  valid in this situation, we split the class to avoid code duplication,
  using subclassing instead.
- Add hrpt reader. [Martin Raspaud]
- Change AMSR2 L1B reader config to be 2 spaces instead of 4. [davidh-
  ssec]
- Remove uncommented blank likes from scene header. [Martin Raspaud]
- Allow filenames to be an empty set and still look for files. [Martin
  Raspaud]
- Reorganize imports in mipp reader. [Martin Raspaud]
- Beautify resample.py. [Martin Raspaud]
- Use uncertainty flags to mask erroneous data. [Martin Raspaud]
- Optimize the loading by caching 3b flag. [Martin Raspaud]
- Stack the projectable keeping the mask. [Martin Raspaud]
- Avoid datasets from being requested multiple times. [Martin Raspaud]
- Fix aapp1b to work again. [Martin Raspaud]
- Use area ids to carry navigation needs. [Martin Raspaud]
- Get the hdfeos_l1b reader to work again. [Martin Raspaud]
- Add yaml files to setup.py included data files. [davidh-ssec]
- Move start/end/area filtering to reader init. [davidh-ssec]

  This includes moving file handler opening to the `select_files` method.

- Add combine_info method to base file handlers. [davidh-ssec]

  I needed a way to let file handlers (written by reader developers) to have control over how extra metadata is combined among all of the "joined" datasets of a swath. This should probably be a classmethod, but I worry that may complicate customization and there is always a chance that instance variables may control this behavior.

- Add more AMSR2 metadata to loaded datasets. [davidh-ssec]
- Change exception to warning when navigation information can't be
  loaded. [davidh-ssec]
- Move reader check to earlier in the file selection process. [davidh-
  ssec]

  The code was looking through each reader config file, instantiating each one, then running the `select_files` method only to return right away when the instantiated reader's name didn't equal the user's requested reader. This was a lot of wasted processing and will get worse with every new reader that's added.

- Rename amsr2 reader to amsr2_l1b. [davidh-ssec]
- Add AMSR2 36.5 channel. [davidh-ssec]
- Fix reader finder so it returns when not asked for anything. [davidh-
  ssec]

  Resampling in the Scene object requires making an empty Scene. There was an exception being raised because the reader finder was trying to search for files in path `None`.

- Add initial AMSR2 L1B reader (yaml) [davidh-ssec]
- Make lons/lats for SwathDefinition in to masked arrays. [davidh-ssec]
- Rewrite the yaml based reader loading methods. [davidh-ssec]

  Lightly tested.

- Rename utility file handlers and moved base file handlers to new
  module. [davidh-ssec]

  The base file handlers being in yaml_reader could potentially cause a circular dependency. The YAML Reader loads a file handler which subclasses one of the base handlers which are in the same module as the yaml reader.

- Fix filename_info name in file handler. [davidh-ssec]

  Oops

- Pass filename info to each file handler. [davidh-ssec]

  There is a lot of information collected while parsing filenames that wasn't being passed to file handlers, now it is. This commit also includes renaming the generic file handler's (hdf5, netcdf) data cache to `file_content` because `metadata` was too generic IMO.

- Finish merge of develop to yaml branch. [davidh-ssec]

  Starting merging develop and a few things didn't make it all the way over cleanly

- Remove redundant log message. [davidh-ssec]
- Fix reader keyword argument name change. [davidh-ssec]

  Also raise an exception if no readers are created

- Merge branch 'develop' into feature-yaml-amsr2. [davidh-ssec]

  # Conflicts:
  #	etc/readers/aapp_l1b.yaml
  #	satpy/readers/__init__.py
  #	satpy/readers/aapp_l1b.py
  #	satpy/scene.py

- Add OMPS so2_trm dataset. [davidh-ssec]
- Rename "scaling_factors" to "factor" in reader configuration. [davidh-
  ssec]
- Merge branch 'feature-omps-reader' into develop. [davidh-ssec]
- Add simple OMPS EDR Reader. [davidh-ssec]
- Clean up various reader methods. [davidh-ssec]

  In preparation for OMPS reader

- Move HDF5 file wrapper to new hdf5_utils.py. [davidh-ssec]
- Add the multiscene module to combine satellite datasets. [Martin
  Raspaud]

  The multiscene class adds the possibility to blend different datasets
  together, given a blend function.
- Add a test yaml-based reader for aapp1b. [Martin Raspaud]
- Fix manually added datasets not being resampled. [davidh-ssec]
- Merge pull request #8 from davidh-ssec/feature-ewa-resampling. [David
  Hoese]

  Feature ewa resampling
- Update EWA resampler to use new wrapper functions from pyresample.
  [davidh-ssec]
- Move resample import in resample tests. [davidh-ssec]

  The resample module import now happens inside the test so only the resample tests fail instead of halting all unittests.

- Fix resample test from moved resample import. [davidh-ssec]

  The 'resample' method imported at the top of projectable.py was moved to inside the resample method to avoid circular imports. The resample tests were still patching the global import. Now they modify the original function. I also imported unittest2 in a few modules to be more consistent.

- Fix bug in EWA output array shape. [davidh-ssec]
- Add initial EWA resampler. [davidh-ssec]
- Move resample imports in Projectable to avoid circular imports.
  [davidh-ssec]
- Rename `reader_name` scene keyword to `reader` [davidh-ssec]

  Also make it possible to pass an instance of a reader or reader-like class. Renaming is similar to how `save_datasets` takes a `writer` keyword.

- Fix loading aggregated viirs sdr metadata. [davidh-ssec]

  Aggregated VIIRS SDR files have multiple `Gran_0` groups with certain attributes and data, like G-Ring information. Loading these in a simple way is a little more complex than the normal variable load and required adding a new metadata join method.

- Refix reader_info reference in yaml base reader. [davidh-ssec]

  This fix got reverted in the last commit for some reason

- Add support for modis l1b data. [Martin Raspaud]
- Edit the wishlist only when needed. [Martin Raspaud]
- Add MODIS l1b reader, no geolocation for now. [Martin Raspaud]
- Assign right files to the reader. [Martin Raspaud]

  No matching of file was done, resulting in assigning all found files to all
  readers.
- Fix reader_info reference in yaml base reader. [davidh-ssec]
- Keep channels in the wishlist when necessary. [Martin Raspaud]

  Due to the creation of a DatasetID for each dataset key, the wishlist
  wasn't matching the actual ids of the datasets.
- Adapt reading to yaml reader way. [Martin Raspaud]

  Since there is more delegating of tasks to the reader, the reading has to
  be adapted.
- Cleanup using pep8. [Martin Raspaud]
- Allow yaml files as config files. [Martin Raspaud]
- Add the dependency tree based reading. [Martin Raspaud]
- Update the yamlbased aapp reader. [Martin Raspaud]
- Move the hdfeos reader to the readers directory. [Martin Raspaud]
- Add the multiscene module to combine satellite datasets. [Martin
  Raspaud]

  The multiscene class adds the possibility to blend different datasets
  together, given a blend function.
- Add a test yaml-based reader for aapp1b. [Martin Raspaud]
- Fix netcdf dimension use to work with older versions of netcdf-python
  library. [davidh-ssec]
- Add 'iter_by_area' method for easier grouping of datasets in special
  resampling cases. [davidh-ssec]
- Fix bug when resampling is done for specific datasets. [davidh-ssec]

  This fix addresses the case when resampling is done for a specific set of datasets. The compute method will attempt to create datasets that don't exist after resampling. Since we didn't resample all datasets it will always fail. This commit only copies the datasets that were specified in resampling. It is up to the user to care for the wishlist if not using the default (resample all datasets).

- Add dimensions to collected metadata for netcdf file wrapper. [davidh-
  ssec]

  I needed to use VIIRS L1B like I do VIIRS SDR for some GTM work and needed to copy over some of the metadata. One piece was only available as a global dimension of the NC file so I made it possible to ask for dimensions similar to how you can for attributes.

- Fix crefl searching for coefficients by dataset name. [davidh-ssec]
- Fix combining info when metadata is a numpy array. [davidh-ssec]
- Fix incorrect NUCAPS quality flag masking data. [davidh-ssec]
- Add .gitignore with python and C patterns. [davidh-ssec]
- Add 'load_tests' for easier test selection. [davidh-ssec]

  PyCharm and possibly other IDEs don't really play well with unittest TestSuites, but work as expected when `load_tests` is used.

- Fix resample hashing when area has no mask. [davidh-ssec]
- Add test for scene iter and fix it again. [davidh-ssec]
- Fix itervalues usage in scene for python 3. [davidh-ssec]
- Allow other array parameters to be passed to MaskedArray through
  Dataset. [davidh-ssec]
- Fix viirs l1b reader to handle newest change in format (no reflectance
  units) [davidh-ssec]
- Fix bug in crefl compositor not respecting input data type. [davidh-
  ssec]
- Fix NUCAPS H2O_MR Dataset to get proper field from file. [davidh-ssec]
- Add environment variable SATPY_ANCPATH for crefl composites. [davidh-
  ssec]
- Fix config files being loaded in the correct (reverse) order. [davidh-
  ssec]

  INI config files loaded from ConfigParser should be loaded in the correct order so that users' custom configs overwrite the builtin configs. For that to happen the builtin configs must be loaded first. The `config_search_paths` function had this backwards, but the compositor loading function was already reversing them. This commit puts the reverse in the config function.

- Update setup.py to always require pillow and not import PIL. [davidh-
  ssec]

  It seems that in older versions of setuptools (or maybe even easy_install) that importing certain libraries in setup.py causes an infinite loop and eats up memory until it gets killed by the kernel.

- Change NUCAPS H2O to H2O_MR to match name in file. [davidh-ssec]
- Add quality flag filtering to nucaps reader. [davidh-ssec]
- Change default units for NUCAPS H2O to g/kg. [davidh-ssec]
- Add filtering by surface pressure to NUCAPS reader. [davidh-ssec]
- Fix composite prereqs not being removed after use. [davidh-ssec]
- Update metadata combining in viirs crefl composite. [davidh-ssec]
- Perform the sharpening on unresampled data if possible. [Martin
  Raspaud]
- Set the default zero height to the right shape in crefl. [Martin
  Raspaud]
- Fix bug in viirs composites when combining infos. [davidh-ssec]
- Add the cloudtop composite for viirs. [Martin Raspaud]
- Merge pull request #7 from davidh-ssec/feature-crefl-composites.
  [David Hoese]

  Feature crefl composites
- Remove ValueError from combine_info for one argument. [davidh-ssec]
- Add info dictionary to Areas created in the base reader. [davidh-ssec]
- Modify `combine_info` to work on multiple datasets. [davidh-ssec]

  Also updated a few VIIRS composites as test usages

- Add angle datasets to viirs l1b for crefl true color to work. [davidh-
  ssec]
- Cleanup crefl code a bit. [davidh-ssec]
- Add sunz correction to CREFL compositor. [davidh-ssec]

  First attempt at adding modifiers to composites, but this method of doing it probably won't be used in the future. For now we'll keep it.

- Fix bug in Scene where composite prereqs aren't removed after
  resampling. [davidh-ssec]
- Rename VIIRS SDR solar and sensor angle datasets. [davidh-ssec]
- Update crefl true color to pan sharpen with I01 if available. [davidh-
  ssec]
- Fix crefl utils to use resolution and sensor name to find
  coefficients. [davidh-ssec]
- Fix Dataset `mask` keyword being passed to MaskedArray. [davidh-ssec]
- Remove filling masked values in crefl utils. [davidh-ssec]
- Fix crefl composite when given percentage reflectances. [davidh-ssec]
- Add basic crefl compositor. [davidh-ssec]
- Clean up crefl utils and rename main function to run_crefl. [davidh-
  ssec]
- Fix crefl utils bug and other code clean up. [davidh-ssec]
- Add M band solar angles and sensor/satellite angles. [davidh-ssec]
- Add `datasets` keyword to save_datasets to more easily filter by name.
  [davidh-ssec]
- Make crefl utils more pythonic. [davidh-ssec]
- Add original python crefl code from Ralph Kuehn. [davidh-ssec]
- Fix the viirs truecolor composite to keep mask info. [Martin Raspaud]
- Allow composites to depend on other composites. [Martin Raspaud]

  In the case of true color with crefl corrected channels for example, the
  true color needs to depend on 3 corrected channels, which in turn can now
  be composites.
- Add Scene import to __init__ for convience. [davidh-ssec]
- Add composites to 'available_datasets' [davidh-ssec]

  Additionally have Scene try to determine what sensors are involved if they weren't specified by the user.

- Add proper "available_datasets" checks in config based readers.
  [davidh-ssec]
- Move config utility functions to separate `config.py` module. [davidh-
  ssec]
- Fix the 'default' keyword not being used checking config dir
  environment variable. [davidh-ssec]
- Add H2O dataset to NUCAPS reader. [davidh-ssec]
- Merge pull request #6 from davidh-ssec/feature-nucaps-reader. [David
  Hoese]

  Add NUCAPS retrieval reader
- Cleanup code according to quantifiedcode. [davidh-ssec]

  Removed instances of checking length for 0, not using .format for strings, and various other code cleanups in the readers.

- Add documentation to various reader functions including NUCAPS reader.
  [davidh-ssec]
- Fix bug when filtering NUCAPS datasets by pressure level. [davidh-
  ssec]
- Add initial NUCAPS retrieval reader. [davidh-ssec]
- Move netcdf file handler class to separate module from VIIRS L1B
  reader. [davidh-ssec]

  Also prepare generic reader for handling other dimensions besides 2D.

- Document the __init__.py files also. [Martin Raspaud]
- Mock scipy and osgeo to fix doc generation problems. [Martin Raspaud]
- Mock more imports for doc building. [Martin Raspaud]
- Remove deprecated doc files. [Martin Raspaud]
- Mock trollsift.parser for documentation building. [Martin Raspaud]
- Update the doc conf.py file no mock trollsift. [Martin Raspaud]
- Add satpy api documentation. [Martin Raspaud]
- Post travis notifications to #satpy. [Martin Raspaud]
- Fix a few deprecation warnings. [Martin Raspaud]
- Document a few Dataset methods. [Martin Raspaud]
- Fix div test skip in py3. [Martin Raspaud]
- Skip the Dataset __div__ test in python 3. [Martin Raspaud]
- Implement numeric type methods for Dataset. [Martin Raspaud]

  In order to merge or keep metadata for Dataset during arithmetic operations
  we need to implement the numeric type methods.
- Cleanup unused arguments in base reader. [davidh-ssec]

  Also makes _load_navigation by renaming it to load_navigation to resolve some quantifiedcode code checks.

- Add documentation to setup.py data file function. [davidh-ssec]
- Fix call to netcdf4's set_auto_maskandscale in viirs l1b reader.
  [davidh-ssec]
- Fix setup.py to find all reader, writer, composite configs. [davidh-
  ssec]
- Merge pull request #5 from davidh-ssec/feature-viirs-l1b. [David
  Hoese]

  Add beta VIIRS L1B reader
- Add LZA and SZA to VIIRS L1B config for DNB composites. [davidh-ssec]

  To make certain DNB composites available I added DNB solar and lunar zenith angle as well as moon illumination fraction. This also required detecting units in the ERF DNB composite since it assumes a 0-1 range for the input DNB data.

- Remove debug_on from scene.py. [davidh-ssec]
- Fix reader not setting units. [davidh-ssec]

  The default for FileKey objects was None for "units". This means that `setdefault` would never work properly.

- Fix config parser error in python 3. [davidh-ssec]

  I tried to make typing easier by using interpolation (substitution) in the VIIRS L1B reader config, but changing from RawConfigParser to ConfigParser breaks things in python 3. I changed it back in this commit and did the config the "long way" with some find and replace.

- Add DNB and I bands to VIIRS L1B reader. [davidh-ssec]
- Fix brightness temperature M bands for VIIRS L1B. [davidh-ssec]
- Add M bands to VIIRS L1B reader. [davidh-ssec]
- Fix VIIRS L1B masking with valid_max. [davidh-ssec]
- Add initial VIIRS L1B reader. [davidh-ssec]

  Currently only supports M01.

- Revert test_viirs_sdr to np 1.7.1 compatibility. [Martin Raspaud]
- Fix gring test in viirs_sdr. [davidh-ssec]
- Add gring_lat and gring_lon as viirs_sdr metadata. [davidh-ssec]

  Also added join_method `append_granule` as a way to keep each granule's data separate.

- Fix composite kd3 resampling. [Martin Raspaud]

  3d array masks were not precomputed correctly, so we now make a workaround.
  A better solution is yet to be found.
- Fix kd3 precomputation for AreaDefinitions. [Martin Raspaud]

  The lons and lats attributes aren't defined by default in AreaDefs, so we
  now make sure to call the get_lonlats method.
- Set default format for dataset saving to geotiff. [Martin Raspaud]
- Move `save_datasets` logic from Scene to base Writer. [davidh-ssec]
- Fix bug in resample when geolocation is 2D. [davidh-ssec]

  The builtin 'any' function works for 1D numpy arrays, but raises an exception when 2D numpy arrays are provided which is the usual case for sat imagery.

- Allow geotiff creation with no 'area' [davidh-ssec]

  Geotiff creation used to depend on projection information from the `img.info['area']` object, but it is perfectly legal to make a TIFF image with GDAL by not providing this projection information. This used to raise an exception, now it just warns.

- Merge pull request #1 from pytroll/autofix/wrapped2_to3_fix. [Martin
  Raspaud]

  Fix "Consider dict comprehensions instead of using 'dict()'" issue
- Use dict comprehension instead of dict([...]) [Cody]
- Merge pull request #2 from pytroll/autofix/wrapped2_to3_fix-0. [Martin
  Raspaud]

  Fix "Explicitly number replacement fields in a format string" issue
- Explicitely numbered replacement fields. [Cody]
- Merge pull request #3 from pytroll/autofix/wrapped2_to3_fix-1. [Martin
  Raspaud]

  Fix "Use `is` or `is not` to compare with `None`" issue
- Use `is` operator for comparing with `None` (Pep8) [Cody]
- Merge pull request #4 from pytroll/autofix/wrapped2_to3_fix-2. [Martin
  Raspaud]

  Fix "Consider an iterator instead of materializing the list" issue
- Use generator expression with any/all. [Cody]
- Fix resample test for python 3. [Martin Raspaud]

  the dict `keys` method return views in py3. We now convert to list for
  consistency.
- Add a test case for resample caching. [Martin Raspaud]
- Revert resample cache changes. [Martin Raspaud]

  They didn't seem necessary in the way resampling is called.
- Rename to satpy. [Martin Raspaud]
- Remove the world_map.ascii file. [Martin Raspaud]
- Allow compressed files to be checked by hrit reader. [Martin Raspaud]
- Add number of scans metadata to viirs sdr config. [davidh-ssec]

  Also fixed rows_per_scan being a string instead of an integer when loaded from a navigation section.

- Fix bug that removed most recent cached kdtree. [davidh-ssec]

  Nearest neighbor resampling cached multiple kdtree results and cleans up the cache when there are more than CACHE_SIZE items stored. It was incorrectly cleaning out the most recent key instead of the oldest key.

- Fix bug when nearest neighbor source geo definition needs to be
  copied. [davidh-ssec]
- Fix bug when specifying what datasets to resample. [davidh-ssec]
- Move geolocation mask blending to resampling step. [davidh-ssec]

  The mask for geolocation (longitude/latitude) was being OR'd with the mask from the first dataset being loaded in the reader. This was ignoring the possibility that other loaded datasets will have different masks since AreaDefinitions are cached. This blending of the masks was moved to nearest neighbor resampling since it ignored other datasets' masks in the reader and is technically a limitation of the nearest neighbor resampling because the geolocation must be masked with the dataset mask for proper output. May still need work to optimize the resampling.

- Add spacecraft_position and midtime metadata to viirs_sdr reader.
  [davidh-ssec]
- Update changelog. [Martin Raspaud]
- Bump version: 1.1.0 → 2.0.0-alpha.1. [Martin Raspaud]
- Add config files for release utilities. [Martin Raspaud]

  We add the .bumpversion.cfg and .gitchangelog.rc for easy version bumping
  and changelog updates.
- Remove v from version string. [Martin Raspaud]
- Add str and repr methods for composites. [Martin Raspaud]

  This add simple repl and str methods for compositors.
- Restructure the documentation for mpop2. [Martin Raspaud]

  This is an attempt to reorganize the documentation to prepare for mpop2.
  Old stuff has been take away, and a fresh quickstart and api are now
  provided.
- Improve the ReaderFinder ImportError message to include original
  error. [Martin Raspaud]

  To make the ImportError more useful in ReaderFinder, the original error
  string is now provided.
- Fix save_dataset to allow both empty filename and writer. [Martin
  Raspaud]

  When saving a dataset without a filename and writer, save_dataset would
  crash. Instead, we are now putting writer to "simple_image" in that case.
- Rename projectable when assigning it through setitem. [Martin Raspaud]

  When a new dataset is added to a scene, it's name should match the string
  key provided by the user.
- Remove references to deprecated mpop.projector. [Martin Raspaud]
- Allow resample to receive strings as area identifiers. [Martin
  Raspaud]

  In resample, the interactive user would most likely use pre-defined areas
  from a custom area file. In this case, it's much easier to refer to the
  area by name, than to get the area definition object from the file. This
  patch allows the `resample` projectable method to work with string ids
  also.
- Add a dataset to whishlish when added with setitem. [Martin Raspaud]

  When adding a dataset to a scene via the datasetdict.__setitem__ method,
  it is likely that the user case about this dataset. As such, it should be
  added to the wishlist in order not to get removed accidently.
- Move composite loading out of Scene to mpop.composites. [Martin
  Raspaud]

  The loading of compositors was a part of the Scene object. However, it does
  not belong there, so we decided to move it out of Scene. The next logical
  place to have it is the mpop.composites modules.
  As a conterpart, we now provide the `available_composites` method to the
  Scene to be able to figure out what we have possibility to generate.
- Fix the travis file to allow python 2.6 to fail. [Martin Raspaud]
- Allow travis to fail on python 2.6. [Martin Raspaud]
- Install importlib for travis tests on python 2.6. [Martin Raspaud]
- Add `behave` to the pip installations in travis. [Martin Raspaud]
- Add behaviour testing to travis and coveralls. [Martin Raspaud]
- Add behaviour tests for showing and saving datasets. [Martin Raspaud]

  Three scenarios were added, testing showing a dataset, saving a dataset,
  and bulk saving datasets (`save_datasets`).
- Fix loading behaviour tests. [Martin Raspaud]

  A little cleanup, and using builtin functions for getting the dataset_names
- Fix DatasetDict's setitem to allow empty md in value. [Martin Raspaud]

  Sometimes a dataset/projectable doesn't have any info attached to it, eg
  because the dataset is synthetic. In these cases, setitem would crash.
  This is now fixed, and if a string is provided as a key in setitem it is
  used as a name if no better name is already there.
- Simplify dataset saving to disk. [Martin Raspaud]

  saving datasets can now be done one by one. If a writer is not provided,
  it is guessed from the filename extension.
- Add a show method to the Scene class. [Martin Raspaud]

  That allows the user to interactively vizualize the data
- Add a default areas.def file. [Martin Raspaud]
- Fix the manifest file to include the config files. [Martin Raspaud]
- Add missing config files to setup.py. [Martin Raspaud]
- Fix setup.py to add cfg files. [Martin Raspaud]

  This is in order to make mpop work out of the box after a pip install.
- Add a behaviour test to find out the available dataset. [Martin
  Raspaud]
- Prevent crashing when a load requirement is not available. [Martin
  Raspaud]

  When requiring a band which isn't available, mpop would crash. This is now
  fixed and replaced by a warning in the log.
- Use behave to do higher level tests. [Martin Raspaud]

  Two small scenarios for testing the loading of the data are implemented now.
- Fix import error in scene. [davidh-ssec]

  A small refactor was done and then undone to move DatasetDict and DatasetID. This little import change wasn't properly cleaned up.

- Fix scene to work with "2 part" compositors and added pan sharpened
  true color composite as an example. [davidh-ssec]
- Added log message to pillow writer to say what filename it was saving
  to. [davidh-ssec]
- Handle optional dependencies for composites (not tested) [davidh-ssec]
- Activate the remaining viirs_sdr reader test cases. [Martin Raspaud]
- Remove the overview_sun TODO item. [Martin Raspaud]
- Fix the multiple load issue for composites. [Martin Raspaud]

  The composite loading would crash when several composites would be loaded
  one after the other. This was because composite config files where loaded
  partially but were considered loaded entirely. In order to fix this
  problem and make things simpler, we removed the composite config mechanism
  entirely, so that the composites are reloaded everytime. That allows both
  config changing on the fly, but also more resilience for multiple sensor
  cases, like one sensor is loaded after another, and the composites wouldn't
  get updated.
- Fix the name issue in sensor-specific composite requests. [Martin
  Raspaud]

  The read_composite_config was requiring wrongly that the provided names
  should be empty or None, making it not read the sensor config file at all.
  In turn that meant that generic composites were used instead of sensor-
  specific ones.
- Got metadata requests working for composites. [davidh-ssec]
- Use DatasetID in composite requirements instead of names and
  wavelengths only. [davidh-ssec]
- Adds ERF DNB composite and updates compositor base to allow for
  metadata and optional requirements although they are not completely
  used yet. [davidh-ssec]
- Added adaptive DNB product. [davidh-ssec]
- Fixed bug in scene when getting writer instance in save_images.
  [davidh-ssec]
- Fix the dataset str function to allow missing name and sensor keys.
  [Martin Raspaud]
- Add quickstart seviri to the documentation. [Martin Raspaud]
- Update the documentation. [Martin Raspaud]
- Add a get_writer function to the scene object. [Martin Raspaud]
- Updating dataset displaying. [Martin Raspaud]
- Add a fixme comment. [Martin Raspaud]
- Added histogram_dnb composite as a stepping stone for getting more
  complex composites added (ex. adaptive_dnb) [davidh-ssec]
- Can now retrieve channel with incomplete DatasetID instance. [Martin
  Raspaud]
- First try at loading metadata. [davidh-ssec]
- Added python 3.5 to travis tests and removed 3.x as allowed failures.
  [davidh-ssec]
- Added basic test for DatasetDict. [davidh-ssec]
- Refactored some file reader methods to properties to be more pythonic.
  [davidh-ssec]
- Viirs test case now works with python3 hopefully. [Martin Raspaud]
- Fixed file units for eps l1b reflectances. [davidh-ssec]
- Corrected frame indicator for eps l1b band 3a. [davidh-ssec]
- Updated eps l1b config with temporary calibration information.
  [davidh-ssec]
- First attempt at rewriting eps l1b reader to be more configurable
  (overkill?) [davidh-ssec]
- Renamed Scene projectables to datasets. [davidh-ssec]
- Updated eps l1b file reader to match base class. [davidh-ssec]
- Made generic single file reader abstract base class and cleaned up
  viirs sdr tests. [davidh-ssec]
- Added a fixme comment. [Martin Raspaud]
- Enable python 3 and osx builds in travis. [Martin Raspaud]
- Config treatment for enhancements. [davidh-ssec]
- Update config handling for finding composites. [davidh-ssec]
- Small fix for dumb environment variable clear on tests. [davidh-ssec]
- First attempt at getting readers and writers using PPP_CONFIG_DIR as a
  supplement to builtin configs. [davidh-ssec]
- Fixed scene tests so they pass. [davidh-ssec]
- Added base_dir for finding input files and a separate base_dir kwargs
  on save_images. [davidh-ssec]
- Makes wishlist a set and should fix problems with multiple loads.
  [davidh-ssec]
- Fixed calibration and other DatasetID access in reader, hopefully.
  [davidh-ssec]
- Fix the xrit reader. [Martin Raspaud]
- Cleanup to prepare for handling calibration better. [davidh-ssec]
- Updated filtering based on resolution, calibration, and polarization.
  [davidh-ssec]
- Updated how readers create dataset info and dataset ids. [davidh-ssec]
- Added calibration to DatasetID (not used yet) and added helper method
  on DatasetDict for filtering retrieved items and keys. [davidh-ssec]
- Renamed BandID to DatasetID. [davidh-ssec]
- Better handling of loading composite dependencies...i think. [davidh-
  ssec]
- Got EPS L1B reader working again with readers being given BandID
  objects. [davidh-ssec]
- Fixed small bug with extra empty string being listed as reader file
  pattern. [davidh-ssec]
- Made DatasetDict accept non-BandID keys during setitem. [davidh-ssec]
- Fixed default file reader for the eps l1b reader. [davidh-ssec]
- A little more cleanup of unused code in viirs sdr. [davidh-ssec]
- More work on viirs sdr using base reader class. [davidh-ssec]
- Started using ConfigBasedReader as base class for VIIRS SDR reader.
  [davidh-ssec]
- Fixed failing scene tests. [davidh-ssec]
- Got viirs sdr reader working with namedtuple dataset keys. [davidh-
  ssec]
- Continue on python3 compatibility. [Martin Raspaud]
- Cleanup. [Martin Raspaud]
- WIP: Start python 3 support. [Martin Raspaud]
- Smoother transition in the sun zenith correct imagery. [Martin
  Raspaud]
- Move reader discovery out of the scene and into mpop.readers. [Martin
  Raspaud]

  The class ReaderFinder was created for this purpose.
- Cleanup. [Martin Raspaud]
- Fix overview and natural composites. [Martin Raspaud]
- Make read and load argument lists consistent. [Martin Raspaud]
- Fix the M01 dataset definition in viirs_sdr.cfg. [Martin Raspaud]
- Fix some viirs composites. [Martin Raspaud]
- Fix viirs_sdr loading using start and end times. [Martin Raspaud]
- Introduce BandIDs to allow for more complex referencing of datasets.
  [Martin Raspaud]

  - Add the BandID namedtuple (name, wl, resolution, polarization)
  - Fix querying for compatibility with BandIDs
  - Fix existing readers for BandIDs

  Example usage from the user side:
  scn.load([BandID(wavelength=0.67, resolution=742),
            BandID(wavelength=0.67, resolution=371),
            "natural", "true_color"])

  BandIDs are now used internally as key for the scene's projectables dict.
- Add file keys to metop's getitem. [Martin Raspaud]
- Rename metop calibration functions. [Martin Raspaud]
- Add file keys for start and end times for metop. [Martin Raspaud]
- Merge the old eps l1b reader with the new one. [Martin Raspaud]
- More work on EPS l1b reader. [Martin Raspaud]
- Initial commit for the metop eps l1b reader. [Martin Raspaud]
- New attempt at calibration keyword in viirs sdr reader. [davidh-ssec]
- Renamed 'channel' to 'dataset' [davidh-ssec]
- Added more tests for VIIRS SDR readers before making calibration or
  file discovery changes. [davidh-ssec]
- Use "super" in the readers. [Martin Raspaud]
- Hopefully fixed py2.6 incompatibility in string formatting. [davidh-
  ssec]
- Added viirs sdr tests for MultiFileReader and HDF5MetaData. [davidh-
  ssec]
- More viirs sdr file reader tests. [davidh-ssec]
- Simple proof of concept for calibration level in viirs sdr reader.
  [davidh-ssec]
- Fixed getting end orbit from last file reader in viirs sdr reader.
  [davidh-ssec]
- Use unittest2 in viirs sdr tests so we can use new features. [davidh-
  ssec]
- Added unittest2 to py26 travis build to hopefully fix h5py
  importerror. [davidh-ssec]
- Added h5py and hdf5 library to travis. [davidh-ssec]
- Started adding basic VIIRS SDR reader tests. [davidh-ssec]
- Changed scene to accept sequence instead of *args. [davidh-ssec]
- Merge branch 'feature-simplify-newreader' into feature-simplify.
  [davidh-ssec]
- Added simple method for finding geolocation files based on header
  values. [davidh-ssec]
- Added rows per scan to viirs sdr metadata. [davidh-ssec]
- Got units and file units working for VIIRS SDR reader. [davidh-ssec]
- Cleaner code for viirs sdr scaling factor check and made sure to OR
  any previous masks. [davidh-ssec]
- Better memory usage in new style viirs sdr reader. [davidh-ssec]
- First step in proof of concept with new reader design. Mostly working
  VIIRS SDR frontend. [davidh-ssec]
- Fixed get_area_file in the resample.py module. [davidh-ssec]
- Allowed sensor to be specified in the reader section. [davidh-ssec]
- Added method to base plugin to determine type of a section. [davidh-
  ssec]
- Make sunzenithnormalize a modern class. [Martin Raspaud]
- Add sunz correction feature. [Martin Raspaud]
- Avoid an infinite loop. [Martin Raspaud]
- Add travis notifications to slack. [Martin Raspaud]
- Remove unneeded code for composites. [Martin Raspaud]
- Add a few composites. [Martin Raspaud]
- Cleanup. [Martin Raspaud]
- Allow json in enhancement config files. [Martin Raspaud]
- Switch on test for writers. [Martin Raspaud]
- Move tests for image stuff to corresponding test file. [Martin
  Raspaud]
- Move image stuff out of projectable into writers/__init__.py. [Martin
  Raspaud]
- Forgot to change reader/writer base class imports. [davidh-ssec]
- Moved reader and writer base classes to subpackages. [davidh-ssec]
- Reworked configuration reading in plugins for less redundancy.
  [davidh-ssec]
- Small fixes to make VIIRS SDR reader work with new resampling.
  [davidh-ssec]
- Fix the wishlist names and removing uneeded info when building RGB
  composites. [Martin Raspaud]
- Dataset is now a subclass of np.ma.MaskedArray. [Martin Raspaud]
- Move determine_mode to projectable. [Martin Raspaud]
- Add helper function to read config files and get the area def file.
  [Martin Raspaud]
- Rename precompute kwarg to cache_dir. [Martin Raspaud]
- Convenience enhancements for resample. [Martin Raspaud]

  - we can now provide "nearest" or "kdtree" instead of a resampler class.
  - The precompute/dump kwarg is now a directory where to save the proj info,
    defaulting to '.' if precompute=True.
- Switch to containers in travis. [Martin Raspaud]
- Fix repo in .travis. [Martin Raspaud]
- Add OrderedDict for python < 2.7. [Martin Raspaud]
- Resample is now feature complete. [Martin Raspaud]

  - Dump kd_tree info to disk when asked
  - Cache the kd_tree info for later use, but cache is cleaned up.
  - OO architecture allowing other resampling methods to be implemented.
  - resampling is divided between pre- and actual computation.
  - hashing of areas is implemented, resampler-specific.
- Fixed bad patch on new scene test. [davidh-ssec]
- First try at more scene tests. [davidh-ssec]
- Move image generation methods to Dataset and move enh. application to
  enhancer. [Martin Raspaud]
- Sensor is now either None, a string, or a non-empty set. [Martin
  Raspaud]
- Forgot to actually use default writer config filename. [davidh-ssec]
- Fixed simple scene test for checking ppp_config_dir. [davidh-ssec]
- Slightly better handling of default writer configs and writer
  arguments. [davidh-ssec]
- Add a writer for png images, and move enhancer to mpop.writers.
  [Martin Raspaud]
- Detached the enhancements handling into an Enhancer class. [Martin
  Raspaud]
- Pass ppp_config_dir to writer, still needs work. [davidh-ssec]
- First attempt at configured writers and all the stuff that goes along
  with it. Renamed 'format' in configs to more logical name. [davidh-
  ssec]
- Remove the add_product method. [Martin Raspaud]
- Cleanup scene unittest. [Martin Raspaud]
- Finish testing scene.get_filenames. [Martin Raspaud]
- Testing scene.get_filenames. [Martin Raspaud]
- Updated tests to test new string messages. 100%! [davidh-ssec]
- Merge branch 'pre-master' into feature-simplify. [Martin Raspaud]

  Conflicts:
  	mpop/satellites/__init__.py
  	mpop/satin/helper_functions.py
  	mpop/satin/mipp_xrit.py
- Add algorithm version in output cloud products. [Martin Raspaud]
- Minor PEP8 tweaks. [Panu Lahtinen]
- Script to generate external calibration files for AVHRR instruments.
  [Panu Lahtinen]
- Support for external calibration coefficients for AVHRR. [Panu
  Lahtinen]
- Removed obsolete "satname" and "number" from satellite configs,
  updated documentation. [Panu Lahtinen]
- Renamed satellite configs to conform to OSCAR naming scheme. [Panu
  Lahtinen]
- Add luts to the pps products from msg format. [Martin Raspaud]
- Add metadata to nwcsaf products. [Martin Raspaud]
- Add \0 to palette strings. [Martin Raspaud]
- Fix pps format output for msg products. [Martin Raspaud]
- Remove phase palette from msg products to avoid confusion. [Martin
  Raspaud]
- Bugfix, np.string -> np.string_ [Martin Raspaud]
- Change variable length strings in h5 products to fixed. [Martin
  Raspaud]
- Fix some cloud product conversions. [Martin Raspaud]
- Fix MSG format to PPS format conversion. [Martin Raspaud]
- Merge branch 'pre-master' of github.com:mraspaud/mpop into pre-master.
  [Martin Raspaud]
- Merge pull request #16 from pnuu/simplified_platforms. [Martin
  Raspaud]

  Simplified platform names for reading custom composites
- Simplified platform names for reading custom composites. [Panu
  Lahtinen]
- Change: accept arbitrary kwargs for saving msg hdf products. [Martin
  Raspaud]
- Revert concatenation to it's original place, in order to keep the
  tests working. [Martin Raspaud]
- Fix whole globe area_extent for loading. [Martin Raspaud]
- Fix rpm building. [Martin Raspaud]
- Cleanup. [Martin Raspaud]
- Change printing of projectables and cleanup. [Martin Raspaud]
- Start testing mpop.scene. [Martin Raspaud]
- Fixed assertIn for python 2.6. [davidh-ssec]
- Added more tests for projectables and updated projectable 3d resample
  test. 100% coverage of projectable! [davidh-ssec]
- Renamed .products to .compositors and fixed unknown names bug.
  [davidh-ssec]
- Added check to see what composite configs were read already. [davidh-
  ssec]
- Do not reread already loaded projectables. [Martin Raspaud]
- Complete .gitignore. [Martin Raspaud]
- Fix unittests for python 2.6. [Martin Raspaud]
- Unittesting again... [Martin Raspaud]
- More unittesting. [Martin Raspaud]
- Fix projectables str to look better. [Martin Raspaud]
- More unittesting. [Martin Raspaud]
- Fix unittests for python 2.6. [Martin Raspaud]
- Still cleaning up. [Martin Raspaud]
- Cleanup. [Martin Raspaud]
- Add tests to the package list in setup.py. [Martin Raspaud]
- Make pylint happy. [Martin Raspaud]
- Fix tests for projectable to pass on 2.6. [Martin Raspaud]
- Start testing the new stuff in travis. [Martin Raspaud]
- Cleanup. [Martin Raspaud]
- Renamed newscene to scene. [Martin Raspaud]
- Moved updated readers from mpop.satin to mpop.readers. [Martin
  Raspaud]
- Changed 'uid' to 'name' for all new components. [davidh-ssec]
- Moved composite configs to separate subdirectory. [davidh-ssec]
- Add an RGBCompositor class and cleanup. [Martin Raspaud]
- Allow passing "areas" to mipp_xrit. [Martin Raspaud]
- Fix the overview composite giving sensible defaults. [Martin Raspaud]
- Fixed bug with RGB composites with passing the wrong info keywords.
  [davidh-ssec]
- Changed sensor keyword in scene to reader and added new sensor keyword
  behavior to find readers based on sensor names. [davidh-ssec]
- Changed new style composites to use a list of projectables instead of
  the scene object implemented __setitem__ for scene. [davidh-ssec]
- Reworked viirs and xrit reader to use .channels instead of .info.
  Simplified reader loading in newscene. [davidh-ssec]
- Test and fix projectable. [Martin Raspaud]
- Allow reading from wavelength, and add Meteosat HRIT support. [Martin
  Raspaud]
- Moved reader init to scene init. Successfully created resampled fog
  image using composite configs. [davidh-ssec]
- Added some default configs for new scene testing. [davidh-ssec]
- Started rewriting viirs sdr reader to not need scene and produce
  projectables. [davidh-ssec]
- Better config reading, and scene init. [Martin Raspaud]
- WIP: removed CONFIG_PATH and changed projectables list into dict.
  [davidh-ssec]
- Add resampling. Simple for now, with elementary caching. [Martin
  Raspaud]
- WIP. [Martin Raspaud]

  * Product dependencies
  * loading from viirs
  * generating images
- WIP: successfully loaded the first viirs granule with newscene!
  [Martin Raspaud]
- Rewriting scene. [Martin Raspaud]
- Add helper function to find files. [Martin Raspaud]
- Fix the config eval thing in scene. [Martin Raspaud]
- Fix masking of lonlats in viirs_sdr. [Martin Raspaud]
- Fixing pps-nc reader. [Adam Dybbroe]
- Clean temporary files after loading. [Adam Dybbroe]
- Pep8 stuff. [Adam Dybbroe]
- Fixed polar-stereographic projection bugs, thanks to Ron Goodson.
  [Lars Orum Rasmussen]
- Update changelog. [Martin Raspaud]
- Bump version: 1.0.2 → 1.1.0. [Martin Raspaud]
- Put config files in etc/pytroll. [Martin Raspaud]
- Fix version strings. [Martin.Raspaud]
- Don't close the h5 files too soon. [Martin Raspaud]
- Close h5 file uppon reading. [Adam Dybbroe]
- Bugfix. [Adam Dybbroe]
- Try a more clever handling of the case where more level-1b files exist
  for given sat and orbit. [Adam Dybbroe]
- Print out files matching in debug. [Martin Raspaud]
- Bugfix. [Adam Dybbroe]
- Adding debug info. [Adam Dybbroe]
- Bugfix. [Adam Dybbroe]
- Merge branch 'pre-master' of https://github.com/mraspaud/mpop into
  pre-master. [Adam Dybbroe]
- Remove ugly print statements. [Martin Raspaud]
- Load the palettes also. [Martin Raspaud]
- AAPP1b: use operational coefficients for vis calibrating per default.
  [Martin Raspaud]

   - Fallback to pre-launch if not available.
   - load(..., pre_launch_coeffs=True) to force using pre-launch coeffs)
- Correct npp name in h5 files. [Martin Raspaud]
- Add the pps v2014 h5 reader. [Martin Raspaud]
- Use h5py for lonlat reading also. [Martin Raspaud]
- Use h5py instead of netcdf for reading nc files. [Martin Raspaud]
- Fix orbit as int in nc_pps loader. [Martin Raspaud]
- Add overlay from config feature. [Martin Raspaud]
- Remove type testing for orbit number. [Martin Raspaud]
- Merge branch 'pre-master' of https://github.com/mraspaud/mpop into
  pre-master. [Adam Dybbroe]
- Allowing kwargs. [Martin Raspaud]
- Add 10 km to the area extent on each side, to avoid tangent cases.
  [Martin Raspaud]
- Orbit doesn't have to be a string anymore. [Martin Raspaud]
- Fix multiple file loading for metop l1b data. [Martin Raspaud]
- Merge branch 'pre-master' of https://github.com/mraspaud/mpop into
  pre-master. [Adam Dybbroe]
- Implement save for all cloudproducts. [Martin Raspaud]
- Change options names to cloud_product_* and add lookup in os.environ.
  [Martin Raspaud]
- Some fixes to nc_pps_l2 for correct saving. [Martin Raspaud]
- Add saving to the cloudtype object. [Martin Raspaud]
- Add the save method to cloudtype object. [Martin Raspaud]
- Rename _md attribute to mda. [Martin Raspaud]
- Mask out bowtie deleted pixels for Suomi-NPP products. [Martin
  Raspaud]
- When a file is provided in nc_pps_l2, just read this file. [Martin
  Raspaud]
- Fix nc_pps_l2 for filename input and PC readiness. [Martin Raspaud]
- ViirsSDR: Fix not to crash on single file input. [Martin Raspaud]
- Fix aapp1b to be able to run both for given filename and config.
  [Martin Raspaud]
- Try loading according to config if provided file doesn't work, aapp1b.
  [Martin Raspaud]
- Don't crash when reading non aapp1b file. [Martin Raspaud]
- Remove "/" from instrument names when loading custom composites.
  [Martin Raspaud]
- Don't say generate lon lat when returning a cached version. [Martin
  Raspaud]
- Nc_pps_l2: don't crash on multiple files, just go through them one at
  the time. [Martin Raspaud]
- Hdfeos: don't just exit when filename doesn't match, try to look for
  files. [Martin Raspaud]
- Don't crash if the file doesn't match (hdfeos) [Martin Raspaud]
- Revert nc_reader back until generalization is ready. [Martin Raspaud]
- Merge branch 'ppsv2014-reader' of github.com:mraspaud/mpop into
  ppsv2014-reader. [Martin Raspaud]
- Adding dataset attributes to pps reading. [Adam Dybbroe]
- Allow inputing filename in the nc_pps_l2 reader. [Martin Raspaud]
- Merge branch 'pre-master' into ppsv2014-reader. [Martin Raspaud]
- Viirs readers fixes. [Martin Raspaud]
- Hdf_eos now uses 1 out of 4 available cores to interpolate data.
  [Martin Raspaud]
- Merge branch 'pre-master' of github.com:mraspaud/mpop into pre-master.
  [Martin Raspaud]
- Fixed bug, now handling fill_value better. [Lars Orum Rasmussen]
- More robust tiff header file decoder. [Lars Orum Rasmussen]
- Add dnb_overview as a standard product (dnb, dnb, 10.8) [Martin
  Raspaud]
- Merge branch 'pre-master' of github.com:mraspaud/mpop into pre-master.
  [Martin Raspaud]
- Corrected the reader for SAFNWC/PPS v2014. [Sara.Hornquist]
- Allow multiresolution loading in hdf eos reader. [Martin Raspaud]
- Revert back to old nwcsaf-pps reader for hdf. The reading of the new
  netcdf format is done with another reader! [Adam Dybbroe]
- A new pps reader for the netCDF format of v2014. [Adam Dybbroe]
- Adding for new cloudmask and type formats... [Adam Dybbroe]
- Enhance nwc-pps reader to support v2014 format. [Adam Dybbroe]
- Merge branch 'pre-master' of https://github.com/mraspaud/mpop into
  pre-master. [Adam Dybbroe]
- Put the config object back in Projector. [Martin Raspaud]
- Fix area_file central search. [Martin Raspaud]
- Move the area_file search inside Projector. [Martin Raspaud]
- Error when satellite config file is not found. [Martin Raspaud]
- Get rid of the funky logging style. [Martin Raspaud]
- Log the config file used to generate the scene. [Martin Raspaud]
- Support filename list to load in viirs_sdr loader. [Martin Raspaud]
- Add avhrr/3 as aliar to avhrr in aapp reader. [Martin Raspaud]
- Fix name matching in hdfeos_l1b. [Martin Raspaud]

  The full name didn't work with fnmatch, take basename instead.
- Allows hdfeos_l1b to read a batch of files. [Martin Raspaud]
- Add delitem, and code cleanup. [Martin Raspaud]
- Merge branch 'pre-master' of github.com:mraspaud/mpop into pre-master.
  [Martin Raspaud]
- Added a reader for SAFNWC/PPS v2014 PPS v2014 has a different
  fileformat than previous SAFNWC/PPS versions. [Sara.Hornquist]
- Aapp1b reader, be more clever when (re)reading. [Martin Raspaud]
- Merge branch 'pre-master' of https://github.com/mraspaud/mpop into
  pre-master. [Adam Dybbroe]

  Conflicts:
  	mpop/satout/netcdf4.py

- Allow reading several files at once in viirs_compact. [Martin Raspaud]
- Allow reading several files at once in eps_l1b. [Martin Raspaud]
- Style: use in instead for has_key() [Martin Raspaud]
- Adding primitive umarf (native) format reader for meteosat. [Martin
  Raspaud]
- Add logging when an info field can't be save to netcdf. [Martin
  Raspaud]
- Add a name to the area when loading aapp data. [Martin Raspaud]
- Merge branch 'pre-master' of github.com:mraspaud/mpop into pre-master.
  [Martin Raspaud]
- For PNG files, geo_mage.tags will be saved a PNG metadata. [Lars Orum
  Rasmussen]
- Add a save method to cfscene objects. [Martin Raspaud]
- Don't take None as a filename in loading avhrr data. [Martin Raspaud]
- Allow loading a file directly for aapp1b and eps_l1b. [Martin Raspaud]

  Just run global_data.load(..., filename="/path/to/myfile.1b")
- Merge branch 'pre-master' of github.com:mraspaud/mpop into pre-master.
  [Martin Raspaud]
- Viirs_sdr can now load depending on an area. [Martin Raspaud]
- Pep8 cosmetics. [Adam Dybbroe]
- Merge pull request #12 from pnuu/pre-master. [Martin Raspaud]

  Fixed "logger" to "LOGGER"
- Fixed "logger" to "LOGGER" [Panu Lahtinen]
- Moving pysoectral module import down to function where pyspectral is
  used. [Adam Dybbroe]
- Merge branch 'smhi-premaster' into pre-master. [Adam Dybbroe]
- Fixing cloudtype product: palette projection. [Adam Dybbroe]
- Turned on debugging to geo-test. [Adam Dybbroe]
- Added debug printout for cloud product loading. [Adam Dybbroe]
- Make snow and microphysics transparent. [Martin Raspaud]
- Rename day_solar to snow. [Martin Raspaud]
- Keep the name of cloudtype products when projecting. [Martin Raspaud]
- Explicitly load parallax corrected files if present. [Martin Raspaud]
- Adding logging for MSG cloud products loading. [Martin Raspaud]
- Fix the parallax file sorting problem, again. [Martin Raspaud]
- Merge branch 'pre-master' of github.com:mraspaud/mpop into pre-master.
  [Martin Raspaud]
- Merge branch 'pre-master' of https://github.com/mraspaud/mpop into
  pre-master. [Adam Dybbroe]
- Bugfix. [Adam Dybbroe]
- Merge branch '3.9reflectance' into pre-master. [Adam Dybbroe]

  Conflicts:
  	mpop/channel.py
  	mpop/instruments/seviri.py
  	mpop/satin/mipp_xrit.py
  	setup.py

- Support for rgbs using the seviri 3.9 reflectance (pyspectral) [Adam
  Dybbroe]
- Adding a sun-corrected overview rgb. [Adam Dybbroe]
- Adduing for "day microphysics" RGB. [Adam Dybbroe]
- Deriving the day-solar RGB using pyspectral to derive the 3.9
  reflectance. [Adam Dybbroe]
- Use "imp" to find input plugins. [Martin Raspaud]
- Cleanup trailing whitespaces. [Martin Raspaud]
- Use cartesian coordinates for lon/lat computation if near-pole
  situations. [Martin Raspaud]
- Set alpha channel to the same type as the other channels. [Martin
  Raspaud]
- Sort the filenames in get_best_products (msg_hdf) [Martin Raspaud]
- Merge branch 'pre-master' of github.com:mraspaud/mpop into pre-master.
  [Martin Raspaud]
- Merge branch 'pre-master' of https://github.com/mraspaud/mpop into
  pre-master. [Adam Dybbroe]
- Merge pull request #10 from pnuu/pre-master. [Martin Raspaud]

  Fixed failed merging. Thanks Pnuu.
- Fixed failed merging (removed "<<<<<<< HEAD" and ">>>>>>> upstream
  /pre-master" lines) [Panu Lahtinen]
- Merge branch 'pre-master' of https://github.com/mraspaud/mpop into
  pre-master. [Adam Dybbroe]
- Merge branch 'pre-master' of https://github.com/mraspaud/mpop into
  pre-master. [Adam Dybbroe]
- Merge branch 'pre-master' of https://github.com/mraspaud/mpop into
  pre-master. [Adam Dybbroe]
- Fix terra and aqua templates for the dual gain channels (13 & 14)
  [Adam Dybbroe]
- Read both parallax corrected and usual cloudtype products. [Martin
  Raspaud]
- Merge branch 'pre-master' of github.com:mraspaud/mpop into pre-master.
  [Martin Raspaud]
- Merge pull request #9 from pnuu/pre-master. [Martin Raspaud]

  Possibility to get area_extent from area definition(s)
- Tests for mpop.satin.helper_functions.boundaries_to_extent. [Panu
  Lahtinen]
- Separated area definitions and boundary calculations. [Panu Lahtinen]
- Added test if proj string is in + -format or not. [Panu Lahtinen]
- Re-ordered the tests. [Panu Lahtinen]
- Fixed incorrect correct values. [Panu Lahtinen]
- Test using area definitions instead of definition names. [Panu
  Lahtinen]
- Possibility to give also area definition objects to
  area_def_names_to_extent() and log a warning if the area definition is
  not used. [Panu Lahtinen]
- Fixed import. [Panu Lahtinen]
- Added tests for mpop.satin.helper_functions. [Panu Lahtinen]
- Moved to mpop/tests/ [Panu Lahtinen]
- Moved to mpop/tests/ [Panu Lahtinen]
- Merge remote-tracking branch 'upstream/pre-master' into pre-master.
  [Panu Lahtinen]

  Conflicts:
  	mpop/satin/aapp1b.py

- Removed unneeded functions. [Panu Lahtinen]
- Test for area_def_names_to_extent() [Panu Lahtinen]
- Removed unnecessary functions. [Panu Lahtinen]
- Removed swath reduction functions. [Panu Lahtinen]
- Reverted not to reduce swath data. [Panu Lahtinen]
- Added possibility to do data reduction based on target area definition
  names. [Panu Lahtinen]
- Added area extent calculations based on given area definition names.
  [Panu Lahtinen]
- Helper functions for area extent and bondary calculations, and data
  reducing for swath data. [Panu Lahtinen]
- Test for mpop.satin.mipp_xrit.lonlat_to_geo_extent() [Panu Lahtinen]
- Support for lon/lat -based area extents. [Panu Lahtinen]
- Add start and end time defaults for the images (runner). [Martin
  Raspaud]
- Merge branch 'pre-master' of https://github.com/mraspaud/mpop into
  pre-master. [Lars Orum Rasmussen]
- Do not mask out negative reflectances in viirs_sdr reading. [Martin
  Raspaud]
- Added navigation to hrpt_hmf plugin. [Martin Raspaud]
- Started working on a new plugin version of hdfeos_l1b. [Martin
  Raspaud]
- Cleanup. [Martin Raspaud]
- Cleanup. [Martin Raspaud]
- Cleanup. [Martin Raspaud]
- Cleanup. [Martin Raspaud]
- Adding scene tests to the test suite. [Martin Raspaud]
- Revamped scene unittests. [Martin Raspaud]
- Don't crash on errors. [Martin Raspaud]
- Revamped projector tests. [Martin Raspaud]
- More geo_image testing. [Martin Raspaud]
- Don't use "super" in geo_image. [Martin Raspaud]
- Fix testing. [Martin Raspaud]
- Mock pyresample and mpop.projector in geo_image tests. [Martin
  Raspaud]
- More testing geo_image. [Martin Raspaud]
- Add tests for geo_image. [Martin Raspaud]
- Merge branch 'unstable' of ssh://safe/data/proj/SAF/GIT/mpop into
  unstable. [Martin Raspaud]
- Mock gdal for geo_image tests. [Martin Raspaud]
- Added netCDF read support for four more projections. [Adam Dybbroe]
- Adding support for eqc in cf format. [Adam Dybbroe]
- Added config templates for GOES and MTSAT. [Lars Orum Rasmussen]
- Copied visir.night_overview to seviri.night_overview, so
  night_overview.prerequisites is correct when night_overview is called
  from seviri.py. [ras]
- Cloutop in seviri.py now same arguments as cloudtop in visir.py. [Lars
  Orum Rasmussen]
- Fix saving as netcdf. [Martin Raspaud]
- Fix floating point tiff saving. [Martin Raspaud]
- Make pillow a requirement only if PIL is missing. [Martin Raspaud]
- Add some modules to mock in the documentation. [Martin Raspaud]
- Add pyorbital to the list of packets to install in travis. [Martin
  Raspaud]
- Merge branch 'feature-travis' into unstable. [Martin Raspaud]
- Test_projector doesn't pass. [Martin Raspaud]
- Test_projector ? [Martin Raspaud]
- Fix travis. [Martin Raspaud]
- Adding test_geoimage. [Martin Raspaud]
- Test_channel passes, test_image next. [Martin Raspaud]
- Test_pp_core crashes, test_channel on. [Martin Raspaud]
- Commenting out tests to find out the culprit. [Martin Raspaud]
- Ok, last try for travis-ci. [Martin Raspaud]
- What is happening with travis ? [Martin Raspaud]
- More fiddling to find out why travis-ci complains. [Martin Raspaud]
- Testing the simple test way (not coverage) [Martin Raspaud]
- Trying to add the tests package for travis-ci. [Martin Raspaud]
- Add the tests package. [Martin Raspaud]
- Preprare for travis-ci. [Martin Raspaud]
- Support 16 bits images (geotiff only at the moment). [Martin Raspaud]
- Merge pull request #8 from pnuu/pre-master. [Martin Raspaud]

  Sun zenith angle correction added.
- A section on mpop.tools added to documentation. [Panu Lahtinen]
- Extra tests for sun_zen_corr(). [Panu Lahtinen]
- Typo. [Panu Lahtinen]
- Channel descriptions added. [Panu Lahtinen]
- Channel desctiptions are added. [Panu Lahtinen]
- Clarification to help sunzen_corr_cos() desctiption. [Panu Lahtinen]
- Test cases for channel.sunzen_corr(). [Panu Lahtinen]
- Sun zenith angle correction split into two functions. [Panu Lahtinen]
- Revert to original version. [Panu Lahtinen]
- Initial commit of mpop.tools (with Sun zenith angle correction). [Panu
  Lahtinen]
- Sun zenith angle correction added. [Panu Lahtinen]
- Merge branch 'pre-master' of github.com:mraspaud/mpop into pre-master.
  [ras]
- Solve the multiple channel resolution with automatic resampling
  radius. [Martin Raspaud]
- Add the "nprocs" option to projector objects and scene's project
  method. [Martin Raspaud]
- Now saving orbit number (if available) as global attribute. [ras]
- Adding more files to be ignored. [ras]
- Merge branch 'pre-master' of github.com:mraspaud/mpop into pre-master.
  [ras]
- New reader for hrpt level0 format. [Martin Raspaud]
- Fix no calibration reading for aapp1b. [Martin Raspaud]
- Add the product name to the the image info. [Martin Raspaud]
- Add some debugging info about missing pixels in viirs_sdr. [Martin
  Raspaud]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Martin Raspaud]
- Corrected a comment. [Adam Dybbroe]
- Fix for M13 load problem - reported by stefano.cerino@gmail.com. [Adam
  Dybbroe]
- Use number of scan to load the right amount of data in compact viirs
  reader. [Martin Raspaud]
- Fix hook to be able to record both filename and uri. [Martin Raspaud]
- Protecting MPOP from netcdf4's unicode variables. [ras]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Martin Raspaud]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Adam Dybbroe]
- Adding a new convection RGB with co2 correction for SEVIRI. [Adam
  Dybbroe]
- Temporary hack to solve for hdf5 files with more than one granule per
  file. [Adam Dybbroe]
- Removing messaging code from saturn and added a more generic "hook"
  argument. [Martin Raspaud]
- Bumped up version. [Martin Raspaud]
- Make viirs_compact scan number independent. [Martin Raspaud]
- Cleanup: marking some deprecated modules, removing unfinished file,
  improving documentation. [Martin Raspaud]
- Adding the ears-viirs compact format reader. Untested. [Martin
  Raspaud]
- Code cleanup. [Martin Raspaud]
- Merge branch 'pre-master' into unstable. [Martin Raspaud]

  Conflicts:
  	mpop/imageo/geo_image.py
- Night_color (should had beed called night_overview) is the same as
  cloudtop. [Lars Orum Rasmussen]
- Bug fix from Bocheng. [Lars Orum Rasmussen]
- Night_overview is just like cloudtop. [Lars Orum Rasmussen]
- Now also handling Polar satellites. [Lars Orum Rasmussen]
- Cosmetic. [Lars Orum Rasmussen]
- Fixed merge conflict. [Lars Orum Rasmussen]
- Trying out a chlorophyll product. [Lars Orum Rasmussen]
- Added a night overview composite. [Lars Orum Rasmussen]
- Better check for empty array. [Lars Orum Rasmussen]
- Fix logging. [Martin Raspaud]
- Fix backward compatibility in, and deprecate image.py. [Martin
  Raspaud]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Martin Raspaud]
- Calling numpy percentile only once when doing left and right cut offs.
  [Adam Dybbroe]
- Add support for identifying npp directories by time-date as well as
  orbit number. [Adam Dybbroe]
- Fix histogram-equalization stretch test. [Adam Dybbroe]
- Bugfix in histogram equalization function. [Adam Dybbroe]
- Using percentile function to generate histogram with constant number
  of values in each bin. [Adam Dybbroe]
- Using numpy.pecentile function to cut the data in the linear stretch.
  [Adam Dybbroe]
- Fix histogram stretch unit test. [Adam Dybbroe]
- Correcting the histogram stretching. The com_histogram function was in
  error when asking for "normed" histograms. [Adam Dybbroe]
- Added histogram method that makes a more populated histogram when the
  data are heaviliy skeewed. Fixes problem seen by Bocheng in DNB
  imagery. [Adam Dybbroe]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Adam Dybbroe]
- Don't remove GeolocationFlyweight _instances, but reset it. Allowing
  for multiple "loads" [Adam Dybbroe]
- Add imageo.formats to installation. [Martin Raspaud]
- AAPP loading bug fix. [Martin Raspaud]

  the aapp1b.py loader to aapp data was broken as it was loading both
  channels 3a and 3b each time, one of them being entirely masked. This of
  course created some problem further down. Fixed by setting the not loadable
  channel to None.
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Martin Raspaud]
- Bugfix in npp.cfg template. [Adam Dybbroe]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Adam Dybbroe]
- Fixing bug concerning the identification of VIIRS geolocation files.
  Now the configuration specified in npp.cfg overwrites what is actually
  written in the metadata header of the band files. [Adam Dybbroe]
- Make saturn posttroll capable. [Martin Raspaud]
- Bump up version number. [Martin Raspaud]
- Cosmetics. [Martin Raspaud]
- Fixing test cases. [Martin Raspaud]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Martin Raspaud]
- Remove dummy test to boost projection performance. [Martin Raspaud]

  Mpop was checking in 2 different places if the source and target areas were
  different, leading to pyresample expanding the area definitions to full
  lon/lat arrays when checking against a swath definition, and then running
  an allclose. This was inefficient, and the programming team decided that it
  was the user's task to know before projection if the source and target area
  were the same. In other words, the user should be at least a little smart.
- Remove dummy test to boost projection performance. [Martin Raspaud]

  Mpop was checking in 2 different places if the source and target areas were
  different, leading to pyresample expanding the area definitions to full
  lon/lat arrays when checking against a swath definition, and then running
  an allclose. This was inefficient, and the programming team decided that it
  was the user's task to know before projection if the source and target area
  were the same. In other words, the user should be at least a little smart.
- Update channel list for modis lvl2. [Martin Raspaud]
- Bump up version number: 1.0.0. [Martin Raspaud]
- Merge branch 'pre-master' into unstable. [Martin Raspaud]
- Added Ninjo tiff example areas definitions. [Lars Orum Rasmussen]
- Cosmetic. [Lars Orum Rasmussen]
- Ninjo tiff writer now handles singel channels. [Lars Orum Rasmussen]

  Ninjo tiff meta-data can now all be passed as arguments

- Better documentation. [Lars Orum Rasmussen]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Martin Raspaud]
- Changing palette name to something more intuitive. Allow to have orbit
  number equals None. [Adam Dybbroe]
- Fixing aqua/terra template config files for dual gain channels (13&14)
  [Adam Dybbroe]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Adam Dybbroe]
- Make overview consistent with the standard overview. [Adam Dybbroe]
- Cleanup. [Martin Raspaud]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Martin Raspaud]

  Conflicts:
  	etc/npp.cfg.template

- Updated npp-template to fit the new viirs reader using the (new)
  plugin-loader system. [Adam Dybbroe]
- Minor clean up. [Adam Dybbroe]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Adam Dybbroe]

  Conflicts:
  	mpop/satin/viirs_sdr.py

- Lunar stuff... [Adam Dybbroe]
- Merge branch 'pre-master' of https://github.com/mraspaud/mpop into
  pre-master. [Adam Dybbroe]
- Changed template to fit new npp reader. [krl]
- Fix version stuff. [Martin Raspaud]
- Merge branch 'feature-optimize_viirs' into unstable. [Martin Raspaud]
- Make viirs_sdr a plugin of new format. [Martin Raspaud]
- Finalize optimisation i new viirs reader. [Martin Raspaud]
- Optimization ongoing. Mask issues. [Martin Raspaud]
- Clarify failure to load hrit data. [Martin Raspaud]
- Fix install requires. [Martin Raspaud]
- Fix projector unit test. [Martin Raspaud]
- Merge branch 'pre-master' into unstable. [Martin Raspaud]
- Merge branch 'pre-master' of git://github.com/mraspaud/mpop into pre-
  master. [Martin Raspaud]
- Fixed (temporary ?) misuse of Image.SAVE. [Lars Orum Rasmussen]
- Now config reader is a singleton. [Lars Orum Rasmussen]
- Merge branch 'pre-master' into unstable. [Martin Raspaud]
- Merge branch 'pre-master' of git://github.com/mraspaud/mpop into pre-
  master. [Martin Raspaud]
- Tmplate -> template. [Lars Orum Rasmussen]
- Added support for saving in Ninjo tiff format. [Lars Orum Rasmussen]
- Projector cleanup. [Martin Raspaud]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Martin Raspaud]
- New VIIRS reader. Better, faster, smarter (consumimg less memory)
  [Adam Dybbroe]
- Fix area hashing. [Martin Raspaud]
- Fix install dependency. [Martin Raspaud]
- Merge branch 'pre-master' into unstable. [Martin Raspaud]
- Merge branch 'pre-master' of git://github.com/mraspaud/mpop into pre-
  master. [Martin Raspaud]

  Conflicts:
  	doc/source/conf.py
  	setup.py

- Bump up version number for release. [Martin Raspaud]
- Optimize. [Martin Raspaud]
- Remove the optional ahamap requirement. [Martin Raspaud]
- Merge branch 'pre-master' of https://github.com/mraspaud/mpop into
  pre-master. [Adam Dybbroe]
- Manage version number centrally. [Martin Raspaud]
- Merge branch 'pre-master' of git://github.com/mraspaud/mpop into pre-
  master. [Martin Raspaud]
- Bump up version number. [Martin Raspaud]
- Make old plugin an info instead of a warning. [Martin Raspaud]
- Merge branch 'pre-master' of git://github.com/mraspaud/mpop into pre-
  master. [Martin Raspaud]
- Pep8. [Adam Dybbroe]
- Merge branch 'aapp1b' into unstable. [Adam Dybbroe]
- Don't mask out IR channel data where count equals zero. [Adam Dybbroe]
- Fixing the masking of the ir calibrated Tbs - count=0 not allowed.
  [Adam Dybbroe]
- Make also vis channels masked arrays. [Adam Dybbroe]
- Checking if file format is post or pre v4 : If bandcor_2 < 0 we are at
  versions higher than 4 Masking a bit more strict. [Adam Dybbroe]
- Now handle data without a mask and handling lons and lats without
  crashing. [Lars Orum Rasmussen]
- Read signed instead of unsigned (aapp1b). [Martin Raspaud]
- Style cleanup. [Martin Raspaud]
- Adding calibration type as an option to the loader. So counts,
  radiances or tbs/refl can be returned. [Adam Dybbroe]
- Better show and more cosmetic. [Lars Orum Rasmussen]
- Making pylint more happy and some cosmetic. [Lars Orum Rasmussen]
- No need to night_overview, use cloudtop with options. [Lars Orum
  Rasmussen]
- Now IR calibration returns a masked array. [Lars Orum Rasmussen]
- Added som options for overview image and added a night overview. [Lars
  Orum Rasmussen]
- Finalize aapp1b python-only reader. [Martin Raspaud]
- Working on a aapp l1b reader. [oananicola]
- Starting a aapp1b branch for directly reading aapp's l1b files. [Lars
  Orum Rasmussen]
- Adding a bit of debug info... [Adam Dybbroe]
- Adding orbit number to the cloud mask object. [Adam Dybbroe]
- Channel cleanup and tests. [Martin Raspaud]
- Merge branch 'feature_plugins' into unstable. [Martin Raspaud]
- Make orbit number an 5-character string (padding with '0') [Martin
  Raspaud]
- New plugin implementation, backward compatible. [Martin Raspaud]
- Cleanup. [Martin Raspaud]
- Add several cores for geoloc in eos. [Martin Raspaud]
- Bugfix hdfeos. [Martin Raspaud]
- Merge branch 'pre-master' of github.com:mraspaud/mpop into pre-master.
  [Lars Orum Rasmussen]
- Fix loading of terra aqua with multiple cores. [Martin Raspaud]
- Add dust, fog, ash composites to VIIRS. [Martin Raspaud]
- Enhance error messages. [Martin Raspaud]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Martin Raspaud]
- Make orbit number an 5-character string (padding with '0') [Martin
  Raspaud]
- New template files for regional EARS (AVHRR and NWC) file support.
  [Adam Dybbroe]
- Minor cosmetics. [Adam Dybbroe]
- Reverted to previous commit. [Lars Orum Rasmussen]
- Correct green-snow. [Martin Raspaud]

  Use 0.6 instead on 0.8

- Merge branch 'fixrtd' into unstable. [Martin Raspaud]
- Add pyresample to mock for doc building. [Martin Raspaud]
- Get rid of the np.inf error in rtd. [Martin Raspaud]
- Mock some import for the documentation. [Martin Raspaud]
- Now, if specified in proj4 object, add EPGS code to tiff metadata.
  [Lars Orum Rasmussen]
- Added, a poor man's version, of Adam's DNB RGB image. [Lars Orum
  Rasmussen]
- Add symlink from README.rst to README. [Martin Raspaud]
- Update download link and README. [Martin Raspaud]
- Bump up version number. [Martin Raspaud]
- Cosmetics. [Martin Raspaud]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Martin Raspaud]
- Add template file for meteosat 10. [Martin Raspaud]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Adam Dybbroe]
- Support for calibrate option. [Adam Dybbroe]
- Add debug messages to hdf-eos loader. [Martin Raspaud]
- Support pnm image formats. [Martin Raspaud]
- Introducing clip percentage for SAR average product. [Lars Orum
  Rasmussen]
- The pps palette broke msg compatibility. Now there are two palettes,
  one for msg and one for pps. [Adam Dybbroe]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Adam Dybbroe]

  Conflicts:
  	mpop/satin/viirs_sdr.py

- Adapted viirs reader to handle aggregated granule files. [Adam
  Dybbroe]
- Fixing nwcsaf-pps ctth height palette. [Adam Dybbroe]
- Take better care of the path (was uri) argument. [Martin Raspaud]
- Don't do url parsing in the hdfeos reader. [Martin Raspaud]
- Fix unit tests. [Martin Raspaud]
- Remove the deprecated append function in scene. [Martin Raspaud]
- Return when not locating hdf eos file. [Martin Raspaud]
- Remove raveling in kd_tree. [Martin Raspaud]
- Make use of the new strftime in the viirs reader. [Martin Raspaud]
- Add a custom strftime. [Martin Raspaud]

  This fixes a bug in windows that prevents running strftime on string that
  contain mapping keys conversion specifiers.
- Catch the error if there is no file to load from. [Martin Raspaud]
- Add a proper logger in hdfeos reader. [Martin Raspaud]
- Get resolution from filename for eos data. [Martin Raspaud]
- Introducing stretch argument for average product. [Lars Orum
  Rasmussen]
- Merge branch 'pre-master' of github.com:mraspaud/mpop into pre-master.
  [Lars Orum Rasmussen]
- Clean up. [Martin Raspaud]
- Bump up version number. [Martin Raspaud]
- Support passing a uri to hdfeos reader. [Martin Raspaud]
- Fix the loading of BT for VIIRS M13 channel. [Martin Raspaud]

  Has no scale and offset
- Merge branch 'pre-master' of github.com:mraspaud/mpop into pre-master.
  [Lars Orum Rasmussen]
- Refactor the unsigned netcdf packing code. [Martin Raspaud]
- Merge branch 'pre-master' of github.com:mraspaud/mpop into pre-master.
  [Lars Orum Rasmussen]
- Support packing data as unsigned in netcdf. [Martin Raspaud]
- Replace auto mask and scale from netcdf4. [Martin Raspaud]

  Eats up too much memory.
- Merge branch 'pre-master' of github.com:mraspaud/mpop into pre-master.
  [Lars Orum Rasmussen]
- Feature: Added template for electro-l satellite. [Martin Raspaud]
- Feature: taking care of missing data in the viirs reader, and allow
  for radiance retrieval. [Martin Raspaud]
- Feature: last adjustments to new netcdf format. [Martin Raspaud]
- Merge branch 'feature-netcdf-upgrade' into unstable. [Martin Raspaud]

  Conflicts:
  	mpop/satout/cfscene.py
  	mpop/satout/netcdf4.py

- Merge branch 'unstable' into feature-netcdf-upgrade. [Martin Raspaud]
- Merge branch 'unstable' into feature-netcdf-upgrade. [Martin Raspaud]

  Conflicts:
  	mpop/satin/mipp_xsar.py

- Work on new netcdf format nearing completion. [Martin Raspaud]
- Feature: wrapping up new netcdf format, cf-satellite 0.2. [Martin
  Raspaud]
- Renamed some global attributes. [Martin Raspaud]
- Netcdf: working towards better matching CF conventions. [Martin
  Raspaud]
- WIP: NetCDF cleaning. [Martin Raspaud]

  - scale_factor and add_offset are now single values.
  - vertical_perspective to geos

- Merge branch 'unstable' into feature-netcdf-upgrade. [Martin Raspaud]
- Group channels by unit and area. [Martin Raspaud]
- Do not apply scale and offset when reading. [Martin Raspaud]
- WIP: updating the netcdf interface. [Martin Raspaud]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Martin Raspaud]
- Changed handeling of "_FillValue"-attributes. Added
  find_FillValue_tags function to search for "_FillValue" attributes.
  The "_FillValue" attributes are used and set when variables are
  created. [Nina.Hakansson]
- Cosmetics. [Martin Raspaud]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Martin Raspaud]
- Fixing bug concerning viirs bandlist and the issue of preventing the
  loading of channels when only products are requested. [Adam Dybbroe]
- Fixing VIIRS reader - does not try to read SDR data if you only want
  to load a product. Minor fixes in MODIS and AAPP1b readers. [Adam
  Dybbroe]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Martin Raspaud]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Adam Dybbroe]
- Bugfix in viirs sdr reader. [Adam Dybbroe]
- Added ir108 composite to Viirs. [Martin Raspaud]
- RUN: add possibility to get prerequisites for a list of areas. [Martin
  Raspaud]
- Updating area_id for the channel during viirs loading and assembling
  of segments. [Martin Raspaud]
- Area handling in viirs and assembling segments. [Martin Raspaud]
- Viirs true color should have a transparent background. [Martin
  Raspaud]
- Added enhancements to the image.__call__ function. [Martin Raspaud]
- Fixing runner to warn for missing functions (instead of crashing).
  [Martin Raspaud]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Martin Raspaud]

  Conflicts:
  	mpop/satin/viirs_sdr.py

- Bug fix green-snow RGB. [Adam Dybbroe]
- Cleaning up a bit in viirs reader. [Adam Dybbroe]
- Temporary fix to deal with scale-factors (in CLASS archive these are
  not tuples of 2 but 6). Taken from old fix in npp-support branch.
  [Adam Dybbroe]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Adam Dybbroe]
- Support for bzip2 compressed NWCSAF products (EARS-NWC) [Adam Dybbroe]
- More flexible viirs reading, and fixes to viirs composites. [Martin
  Raspaud]
- Added a stereographic projection translation. [Lars Orum Rasmussen]
- Added modist as valid name for 'eos1' [Lars Orum Rasmussen]
- Added night_microphysics. [Lars Orum Rasmussen]
- Added stretch option. [Lars Orum Rasmussen]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Martin Raspaud]
- Feature: new function to create an image from a scene. [Martin
  Raspaud]
- Fixed a new npp template config file, with geo_filename example. [Adam
  Dybbroe]
- Adding 500meter scan area. [Adam Dybbroe]
- Fixing bug in geolocation reading and removing old style viirs
  composite file. [Adam Dybbroe]
- Using a template from configuration file to find the geolocation file
  to read - for all VIIRS bands. [Adam Dybbroe]
- Fixed bug in hr_natural and added a dnb method. [Adam Dybbroe]
- Fixing Bow-tie effects and geolocation for VIIRS when using Cloudtype.
  Needs to be generalised to all products! [Adam Dybbroe]
- Support for tiepoint grids and interpolation + masking out no-data
  geolocation (handling VIIRS Bow-tie deletetion) [Adam Dybbroe]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Adam Dybbroe]
- Adding viirs composites and pps_odim reader for avhrr and viirs
  channel data in satellite projection (swath) [Adam Dybbroe]
- Added a Geo Phys Product to modis level2. [Lars Orum Rasmussen]
- Merge branch 'pre-master' of github.com:mraspaud/mpop into pre-master.
  [Lars Orum Rasmussen]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Martin Raspaud]
- Adding support for ob_tran projection even though it is not cf-
  compatible yet. [Adam Dybbroe]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Adam Dybbroe]
- Added the reading of geolocation data from the PPS formatet level1
  file. [Adam Dybbroe]
- Added Europe Mesan area to template. [Adam Dybbroe]
- Feature: MSG hdf files are now used to determine the area. [Martin
  Raspaud]
- Fixed error message. [Martin Raspaud]
- Cleanup: clarified import error. [Martin Raspaud]
- Cleanup: More descriptive message when plugin can't be loaded. [Martin
  Raspaud]
- Raised version number. [Martin Raspaud]
- More relevant messages in msg_hdf reading. [Martin Raspaud]
- Adding a RGB for night condition. [Lars Orum Rasmussen]
- Merge branch 'pre-master' into unstable. [Martin Raspaud]
- Modis level-2 reader and netcdf writer can now handle scenes
  containing only geo-physical product (and no channels) [Lars Orum
  Rasmussen]
- Merge branch 'pre-master' into unstable. [Martin Raspaud]
- Merge pull request #2 from cheeseblok/FixViirsRedSnow. [Martin
  Raspaud]

  Fix typo in red_snow check_channels method
- Fix typo in red_snow check_channels method. [Scott Macfarlane]
- Feature: Pypi ready. [Martin Raspaud]
- Bufix: updating to use python-geotiepoints. [Martin Raspaud]
- Bumping up the version number for the next release. [Martin Raspaud]
- Doc: updating add_overlay documentation. [Martin Raspaud]
- Feature: adding interpolation to modis lon lats. [Martin Raspaud]
- Use pynav to get lon/lats if no file can be read. [Martin Raspaud]
- Hack to handle both level2 and granules. [Martin Raspaud]
- Added the possibility to provide a filename to eps_l1b loader. [Martin
  Raspaud]
- Updated npp confirg file template with geo_filename example. [Adam
  Dybbroe]
- Merge branch 'feature_new_eps_reader' into unstable. [Martin Raspaud]
- Added xml file to etc and setup.py. [Martin Raspaud]
- Bugfix in geolocation assignment. [Martin Raspaud]
- Allowing for both 3a and 3A. [Martin Raspaud]
- Put xml file in etc. [Martin Raspaud]
- New eps l1b is now feature complete. Comprehensive testing needed.
  [Martin Raspaud]
- Added a new eps l1b reader based on xml description of the format.
  [Martin Raspaud]
- Corrected longitude interpolation to work around datum shift line.
  [Martin Raspaud]
- Cloudtype channel now called "CT". [Martin Raspaud]
- Merge branch 'pre-master' of git://github.com/mraspaud/mpop into pre-
  master. [Martin Raspaud]
- SetProjCS is now correctly called after ImportFromProj4. [Lars Orum
  Rasmussen]

  Added SetWellKnownGeogCS if available

- Merge branch 'pre-master' into unstable. [Martin Raspaud]

  Conflicts:
  	mpop/satin/mipp_xsar.py

- More correct 'new area' [Lars Orum Rasmussen]
- Mipp restructure. [Lars Orum Rasmussen]
- Merge branch 'pre-master' into area-hash. [Lars Orum Rasmussen]
- Merge branch 'pre-master' into area-hash. [Lars Orum Rasmussen]
- Now more unique projection filenames (using hash of areas) [Lars Orum
  Rasmussen]
- Enhancements to pps hdf format readers. [Martin Raspaud]
- Feature: added support for geotiff float format in geo_image. [Martin
  Raspaud]
- Don't touch satscene.area if already present (mipp reading) [Martin
  Raspaud]
- Feature: get best msg hdf file using area_extent. [Martin Raspaud]
- Duck typing for channel assignation. [Martin Raspaud]
- Fixed meteosat reading. [Martin Raspaud]

  - do not change the scene metadata when no channel is loaded
  - do not crash if no PGE is present

- Added shapes in mpop.cfg.template for pycoast. [Martin Raspaud]
- Cleanup. [Martin Raspaud]
- New add_overlay function, using pycoast. [Martin Raspaud]
- Added test for __setitem__ (scene) [Martin Raspaud]
- Feature: add a global area if possible. [Martin Raspaud]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Martin Raspaud]
- Fixing so thar also other products (than Channel data) can be
  assempled. [Adam.Dybbroe]
- Adding data member to CloudType. [Adam.Dybbroe]
- Added support for trucolor image from modis. [Adam.Dybbroe]
- Cleaning up geo_image.py. [Martin Raspaud]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Martin Raspaud]

  Conflicts:
  	mpop/satin/hdfeos_l1b.py

- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Adam.Dybbroe]
- Minor cosmetic/editorial stuff. [Adam.Dybbroe]
- Small bugfix - viirs interface. [Adam.Dybbroe]
- Feature: wrapping up hdfeos upgrade. [Martin Raspaud]

  - migrated data to float32 instead of float64
  - support only geoloc a 1km resolution at the moment
  - adjust channel resolution to match loaded data
  - added template terra.cfg file.

- Trimming out dead detectors. [Adam.Dybbroe]
- WIP: hdf eos now reads only the needed channels, and can have several
  resolutions. Geoloc is missing though. [Martin Raspaud]
- WIP: Started working on supporting halv/quarter files for modis.
  [Martin Raspaud]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Martin Raspaud]
- Changed MODIS HDF-EOS level 1b reader to accomodate both the thinned
  EUMETCasted data and Direct readout data. Changed name from
  thin_modis.py to hdfeos_l1b.py. Added filename pattern to config.
  [Adam.Dybbroe]
- Fixing indexing bug: missing last line in Metop AVHRR granule.
  [Adam.Dybbroe]
- Revert "Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into
  unstable" [Martin Raspaud]

  This reverts commit 45809273f2f9670c8282c32197ef47071aecaa74, reversing
  changes made to 10ae6838131ae1b6e119e05e08496d1ec9018a4a.

- Revert "Reapplying thin_modis cleaning" [Martin Raspaud]

  This reverts commit 52c63d6fbc9f12c03b645f29dd58250da943d24a.

- Reapplying thin_modis cleaning. [Martin Raspaud]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Martin Raspaud]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Adam.Dybbroe]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Adam.Dybbroe]
- Merge branch 'pre-master' into unstable. [Adam.Dybbroe]

  Conflicts:
  	mpop/satin/eps_avhrr.py

- Minor enhancements to nwcsaf pps cloud type reading: Adding support
  for phase and quality flags. [Adam.Dybbroe]
- Fixing indexing bug: missing last line in Metop AVHRR granule.
  [Adam.Dybbroe]
- Merge branch 'unstable' of /data/proj/SAF/GIT/mpop into unstable.
  [Adam.Dybbroe]

  Conflicts:
  	doc/source/conf.py
  	mpop/instruments/mviri.py
  	mpop/instruments/seviri.py
  	mpop/instruments/test_mviri.py
  	mpop/instruments/test_seviri.py
  	mpop/instruments/test_visir.py
  	mpop/instruments/visir.py
  	mpop/satin/test_mipp.py
  	mpop/satin/thin_modis.py
  	mpop/saturn/runner.py
  	mpop/scene.py
  	setup.py
  	version.py

- Merge branch 'unstable' of https://github.com/mraspaud/mpop into
  unstable. [Adam.Dybbroe]
- Thin_modis Cleanup. [Martin Raspaud]
- Cleanup. [Martin Raspaud]
- Style: Cleaning up. [Martin Raspaud]
- Doc: added screenshots. [Martin Raspaud]
- Cleanup, switch to compositer globaly. [Martin Raspaud]
- Doc: added more documentation to polar_segments.py. [Martin Raspaud]
- Cleanup: remove old unit test for assemble_swath. [Martin Raspaud]
- Bugfix in assemble_segments. [Martin Raspaud]
- Cleanup: removed old assemble_swath function. [Martin Raspaud]
- Doc: update docstring for project. [Martin Raspaud]
- Upgrade: assemble_segments now uses scene factory. [Martin Raspaud]
- DOC: examples are now functional. [Martin Raspaud]
- Cleanup: removed old plugins directory. [Martin Raspaud]
- Merge branch 'new_plugins' into unstable. [Martin Raspaud]

  Conflicts:
  	mpop/plugin_base.py

- Init file for plugins initialization. [Adam.Dybbroe]
- Merge branch 'new_plugins' of https://github.com/mraspaud/mpop into
  new_plugins. [Adam.Dybbroe]
- Removing old deprecated and now buggy part - has been caught by the
  try-exception since long. Adding for plugins directory. [Adam.Dybbroe]
- Corrected import bug. [Adam.Dybbroe]
- Merge branch 'unstable' into new_plugins. [Adam.Dybbroe]
- Bug correction - config file reading section 'format' [Adam.Dybbroe]
- Removing old deprecated and now buggy part - has been caught by the
  try-exception since long. Adding for plugins directory. [Adam.Dybbroe]
- Merge branch 'unstable' of github.com:mraspaud/mpop into unstable.
  [Martin Raspaud]
- Merge branch 'unstable' of https://github.com/mraspaud/mpop into
  unstable. [Adam.Dybbroe]
- First time in git. [Adam.Dybbroe]
- Merge branch 'unstable' of https://github.com/mraspaud/mpop into
  unstable. [Adam.Dybbroe]
- Meris level-2 reader - first commit. [Adam.Dybbroe]
- Minor fixes. [Adam.Dybbroe]
- Fixed typo. [Adam.Dybbroe]
- Feature: updating mipp test to use factory. [Martin Raspaud]
- Cleaning up an old print. [Martin Raspaud]
- Merge branch 'v0.10.2-support' into unstable. [Martin Raspaud]
- Feature: added support for new eumetsat names (modis) and terra.
  [Martin Raspaud]
- Merge branch 'new_plugins' into unstable. [Martin Raspaud]
- Moved mipp plugin back to satin. [Martin Raspaud]
- Feature: all former plugins are adapted to newer format. [Martin
  Raspaud]
- Style: finalizing plugin system. Now plugins directories loaded from
  mpop.cfg. [Martin Raspaud]
- Cleanup: removing old stuff. [Martin Raspaud]
- Feature: added reader plugins as attributes to the scene, called
  "<format>_reader". [Martin Raspaud]
- Feature: new plugin format, added a few getters and made scene
  reference weak. [Martin Raspaud]
- New plugin system. [Martin Raspaud]

  Transfered the mipp plugin.

- DOC: fixed path for examples. [Martin Raspaud]
- DOC: Added documentation examples to the project. [Martin Raspaud]
- Merge branch 'unstable' of github.com:mraspaud/mpop into unstable.
  [Martin Raspaud]
- Using LOG call instead of print. [Adam.Dybbroe]
- Fixed missing LOG import. [Adam.Dybbroe]
- Further improvements to MODIS level2 reader and processor.
  [Adam.Dybbroe]
- Feature: Added projection to the pps_hdf channels. [Martin Raspaud]
- DOC: added use examples in the documentation directory. [Martin
  Raspaud]
- Merge branch 'master' into unstable. [Martin Raspaud]
- Added posibility to have instrument_name in the filenames.
  [Adam.Dybbroe]
- Making sure we pass on orbit number when projecting the scene.
  [Adam.Dybbroe]
- Added colour map for Modis Chlorophyl-A product. [Adam.Dybbroe]
- Taking away the alpha parameters for RGB modes. [Martin Raspaud]
- Added areas in channels for test. [Martin Raspaud]
- Added the radius parameter to runner. [Martin Raspaud]
- Adding preliminary NWCSAF pps product reader. [Adam.Dybbroe]
- Cleaning up. [Martin Raspaud]
- Updated satpos file directories. [Martin Raspaud]
- Cleaning up. [Martin Raspaud]
- Merge branch 'pre-master' into unstable. [Martin Raspaud]
- Updated copyright and version number. [Martin Raspaud]
- Merge branch 'release-0.11' [Martin Raspaud]
- Merge branch 'pre-master' into release-0.11. [Martin Raspaud]
- Updated copyright dates in setup.py. [Martin Raspaud]
- Bumped version number to 0.11.0. [Martin Raspaud]
- Updating setup stuff. [Martin Raspaud]
- Merge branch 'pre-master' into unstable. [Martin Raspaud]
- Adding Day/Night band support. [Adam.Dybbroe]
- Adding area for mapping sample data i-bands. [Adam.Dybbroe]
- Scaling reflectances to percent (%) as required in mpop.
  [Adam.Dybbroe]
- Adding support for I-bands. [Adam.Dybbroe]
- Merge branch 'pre-master' of https://github.com/mraspaud/mpop into
  pre-master. [Adam.Dybbroe]
- Merge branch 'npp-support' into pre-master. [Adam.Dybbroe]
- Renamed to npp1.cfg. [Adam.Dybbroe]
- VIIRS composites - M-bands only so far. [Adam.Dybbroe]
- Cleaning print statements. [Adam.Dybbroe]
- NPP template. [Adam.Dybbroe]
- Adding NPP/VIIRS test area for sample data: M-bands. [Adam.Dybbroe]
- Adding I-band support. [Adam.Dybbroe]
- Fixing for re-projection. [Adam.Dybbroe]
- Various small corrections. [Adam.Dybbroe]
- Corrected band widths - ned to be in microns not nm. [Adam.Dybbroe]
- Support for NPP/JPSS VIIRS. [Adam.Dybbroe]
- Updated copyright in sphinx doc. [Martin Raspaud]
- Deprecating add_overlay in favor of pycoast. [Martin Raspaud]
- Merge branch 'feature-new-nc-format' into unstable. [Martin Raspaud]
- Added support for different ordering of dimensions in band data.
  [Martin Raspaud]

  Use the band_axis keyword argument.

- NC reader support different dimension orderings for band-data. [Martin
  Raspaud]
- NC: now band data is of shape (band, x, y). [Martin Raspaud]
- Merge branch 'pre-master' into unstable. [Martin Raspaud]
- Now a channel can be added to a scene dynamically using dict notation.
  [esn]
- Added units to aapp1b reader. [Martin Raspaud]
- Deactivating mipp loading test. [Martin Raspaud]
- Adjusted tests for compositer. [Martin Raspaud]
- Merge branch 'feature-cleaning' into unstable. [Martin Raspaud]
- Merge branch 'unstable' into feature-cleaning. [Martin Raspaud]
- Merge branch 'pre-master' into unstable. [Martin Raspaud]
- Added append function to scene.py. [Esben S. Nielsen]
- New error message when no instrument-levelN section is there in the
  satellite config file. [Martin Raspaud]
- Merge branch 'feature-radius-of-influence' into unstable. [Martin
  Raspaud]
- Syntax bug fixed. [Martin Raspaud]
- Made orbit number default to None for PolarFactory's create_scene.
  [Martin Raspaud]
- Merge branch 'feature-radius-of-influence' into unstable. [Martin
  Raspaud]
- Radius of influence is now a keyword parameter to the scene.project
  method. [Martin Raspaud]
- Merge branch 'pre-master' into unstable. [Martin Raspaud]
- Can now get reader plugin from PYTHONPATH. [Esben S. Nielsen]
- Renamed asimage to as_image. [Martin Raspaud]
- Wavelength and resolution are not requirements in config files
  anymore. [Martin Raspaud]
- Merge branch 'feature-channel-to-image' into unstable. [Martin
  Raspaud]
- Feature: added the asimage method to channels, to retrieve a black and
  white image from the channel data. [Martin Raspaud]
- Merge branch 'feature-doc-examples' into unstable. [Martin Raspaud]
- Doc: added more documentation to polar_segments.py. [Martin Raspaud]
- DOC: examples are now functional. [Martin Raspaud]
- DOC: fixed path for examples. [Martin Raspaud]
- DOC: Added documentation examples to the project. [Martin Raspaud]
- DOC: added use examples in the documentation directory. [Martin
  Raspaud]
- Merge branch 'feature-project-mode' into unstable. [Martin Raspaud]
- Doc: update docstring for project. [Martin Raspaud]
- Cleanup. [Martin Raspaud]
- Switched seviri and mviri to compositer. [Martin Raspaud]
- Cleanup. [Martin Raspaud]
- Style: Cleaning up. [Martin Raspaud]
- Doc: added screenshots. [Martin Raspaud]
- Cleanup, switch to compositer globaly. [Martin Raspaud]

  Conflicts:

  	mpop/instruments/visir.py
  	mpop/satin/hrpt.py
  	mpop/saturn/runner.py

- Cleanup: remove old unit test for assemble_swath. [Martin Raspaud]
- Bugfix in assemble_segments. [Martin Raspaud]
- Cleanup: removed old assemble_swath function. [Martin Raspaud]

  Conflicts:

  	mpop/scene.py

- Upgrade: assemble_segments now uses scene factory. [Martin Raspaud]
- Fixed typo. [Adam.Dybbroe]
- Feature: updating mipp test to use factory. [Martin Raspaud]
- Cleaning up an old print. [Martin Raspaud]

  Conflicts:

  	mpop/satin/mipp.py

- Cleanup: removing old stuff. [Martin Raspaud]
- Cleaned up and updated meteosat 9 cfg template further. [Martin
  Raspaud]
- Updated templates to match pytroll MSG tutorial. [Esben S. Nielsen]
- Simplified reading of log-level. [Lars Orum Rasmussen]
- Proposal for reading loglevel from config file. [Lars Orum Rasmussen]
- Cfscene now handles channels with all masked data. [Esben S. Nielsen]
- Netcdf area fix. [Martin Raspaud]
- Syle: copyright updates. [Martin Raspaud]
- Modified the modis-lvl2 loader and extended a bit the cf-io
  interfaces. [Adam.Dybbroe]
- First time in GIT A new reader for EOS-HDF Modis level-2 files from
  NASA. See http://oceancolor.gsfc.nasa.gov/DOCS/ocformats.html#3 for
  format description. [Adam.Dybbroe]
- Added license. [Martin Raspaud]
- Merge branch 'unstable' of github.com:mraspaud/mpop into unstable.
  [Martin Raspaud]
- Info needs to be an instance attribute. [Lars Orum Rasmussen]
- Fix initialization of self.time_slot. [Lars Orum Rasmussen]
- Merge branch 'v0.10.2-support' into unstable. [Martin Raspaud]
- Added pyc and ~ files to gitignore. [Martin Raspaud]
- Updated thin modis reader for new file name. [Martin Raspaud]
- Merge branch 'v0.10.1-support' into unstable. [Martin Raspaud]
- Compression and tiling as default for geotifs. [Martin Raspaud]
- Merge branch 'v0.10.0-support' into unstable. [Martin Raspaud]
- Feauture: support for qc_straylight. [Martin Raspaud]
- Compression and tiling as default for geotifs. [Martin Raspaud]
- WIP: attempting interrupt switch for sequential runner. [Martin
  Raspaud]
- Feature: changing filewatcher from processes to threads. [Martin
  Raspaud]
- Feauture: support for qc_straylight. [Martin Raspaud]
- Compression and tiling as default for geotifs. [Martin Raspaud]
- Update: modis enhancements. [Martin Raspaud]
- Feature: filewatcher keeps arrival order. [Martin Raspaud]
- Feature: concatenation loads channels. [Martin Raspaud]
- Feature: use local tles instead of downloading systematically. [Martin
  Raspaud]
- Feature: move pyaapp as single module. [Martin Raspaud]
- Feature: added ana geoloc for hrpt and eps lvl 1a. [Martin Raspaud]
- Cosmetics. [Martin Raspaud]
- Added gatherer and two_line_elements. [Martin Raspaud]
- Merge branch 'unstable' of github.com:mraspaud/mpop into unstable.
  [Martin Raspaud]
- Moved a parenthesis six characters to the left. [Lars Orum Rasmussen]
- Feature: assemble_segments function, more clever and should replace
  assemble_swaths. [Martin Raspaud]
- Feature: thin modis reader upgrade, with lonlat estimator and channel
  trimmer for broken sensors. [Martin Raspaud]
- Merge branch 'unstable' of github.com:mraspaud/mpop into unstable.
  [Martin Raspaud]
- Netcdf bandname now only uses integer part of resolution. [Esben S.
  Nielsen]
- Improvement: made resolution int in band names, for netcdf. [Martin
  Raspaud]
- Cleaning. [Martin Raspaud]
- WIP: ears. [Martin Raspaud]
- Trying to revive the pynwclib module. [Martin Raspaud]
- Cleaning. [Martin Raspaud]
- Wip: polar hrpt 0 to 1b. [Martin Raspaud]
- Feature: Added proj4 parameters for meteosat 7. [Martin Raspaud]
- Merge branch 'unstable' of github.com:mraspaud/mpop into unstable.
  [Martin Raspaud]
- Cosmetic. [Esben S. Nielsen]
- Now channels are read and saved in order. Optimized scaling during CF
  save. [Esben S. Nielsen]
- Feature: Adding more factories. [Martin Raspaud]
- Documentation: adding something on factories and area_extent. [Martin
  Raspaud]
- Documentation: added needed files in setup.py. [Martin Raspaud]
- Style: remove a print statement and an unused import. [Martin Raspaud]
- Feature: Added natural composite to default composite list. [Martin
  Raspaud]
- Feature: made compositer sensitive to custom composites. [Martin
  Raspaud]
- Documentation: Upgraded documentation to 0.10.0. [Martin Raspaud]
- Merge branch 'unstable' of github.com:mraspaud/mpop into unstable.
  [Martin Raspaud]
- The RELEASE-VERSION file should not be checked into git. [Lars Orum
  Rasmussen]
- Optimized parts of mpop. Fixed projector caching. [Esben S. Nielsen]
- Optimized parts of mpop processing. Made projector caching functional.
  [Esben S. Nielsen]
- Ignore build directory. [Lars Orum Rasmussen]
- Check array in stretch_logarithmic. [Lars Orum Rasmussen]
- Prevent adding unintended logging handlers. [Lars Orum Rasmussen]
- Feature: Adding extra tags to the image allowed in local_runner.
  [Martin Raspaud]
- Style: lines to 80 chars. [Martin Raspaud]
- Merge branch 'unstable' [Martin Raspaud]
- Feature: pps hdf loading and polar production update. [Martin Raspaud]
- Style: cleanup. [Martin Raspaud]
- Merge branch 'unstable' of github.com:mraspaud/mpop into unstable.
  [Martin Raspaud]

  Conflicts:
  	mpop/satin/mipp.py

- Fixed memory problems. Workaround for lazy import of pyresample. Now
  uses weakref for compositor. [Esben S. Nielsen]
- Better logging in scene loading function. [Martin Raspaud]
- Remove unneeded import. [Martin Raspaud]
- New version. [Martin Raspaud]
- Merge branch 'master' of github.com:mraspaud/mpop. [Lars Orum
  Rasmussen]
- Feature: direct_readout chain in place. [Martin Raspaud]
- Removing no longer needed avhrr.py. [Martin Raspaud]
- Made scaling expression in cfscene.py nicer. [Esben S. Nielsen]
- Corrected shallow copy problem with compositor. Simplyfied usage of
  GeostationaryFactory. [Esben S. Nielsen]
- Feature: cleaner hdf reading for both pps and msg. [Martin Raspaud]
- Stability: added failsafe in case no config file is there when
  loading. [Martin Raspaud]
- Merge branch 'pps_hdf' into unstable. [Martin Raspaud]
- Feature: Support area_extent in scene.load. [Martin Raspaud]
- Feature: Cleaning and use the mipp area_extent and sublon. [Martin
  Raspaud]
- Style: Allow to exclude all the *level? sections. [Martin Raspaud]
- Redespached a few composites. [Martin Raspaud]
- Style: cosmetics. [Martin Raspaud]
- Feature: added the power operation to channels. [Martin Raspaud]
- Removed the no longer needed meteosat09.py file. [Martin Raspaud]
- Wip: iterative loading, untested. [Martin Raspaud]
- More on versionning. [Martin Raspaud]
- Merge branch 'unstable' into pps_hdf. [Martin Raspaud]
- Feature: started working on the PPS support. [Martin Raspaud]
- Spelling. [Martin Raspaud]
- Added logarithmic enhancement. [Lars Orum Rasmussen]
- Removed unneeded file. [Martin Raspaud]
- Api: new version of mipp. [Martin Raspaud]
- Added automatic version numbering. [Martin Raspaud]
- Version update to 0.10.0alpha1. [Martin Raspaud]
- Api: unload takes separate channels (not iterable) as input. [Martin
  Raspaud]
- Doc: updated the meteosat 9 template config. [Martin Raspaud]
- Merge branch 'unstable' of github.com:mraspaud/mpop into unstable.
  [Martin Raspaud]

  Conflicts:
  	mpop/satellites/meteosat09.py

- Feature: Introduced compound satscene objects. [Martin Raspaud]

  This is done through the use of an "image" attribute, created by the factory in the "satellites" package.
  The image attribute holds all the compositing functions, while the satscene object remains solely a container for satellite data and metadata.

- Feature: added the get_custom_composites function and a composites
  section in mpop.cfg to load custom made composites on the fly. [Martin
  Raspaud]
- Feature: make use of mipp's area_extent function. [Martin Raspaud]
- Style: cleanup channels_to_load after loading. [Martin Raspaud]
- Doc: introduce mpop.cfg. [Martin Raspaud]
- Feature: make use of the new mpop.cfg file to find the area file.
  Added the get_area_def helper function in projector. [Martin Raspaud]
- Feature: Added the new pge02f product for met09. [Martin Raspaud]
- Feature: New format keyword for images. [Martin Raspaud]
- Update: new version of mipp, putting the image upright when slicing.
  [Martin Raspaud]
- Merge branch 'unstable' of github.com:mraspaud/mpop into unstable.
  [Martin Raspaud]

  Conflicts:
  	mpop/satout/netcdf4.py
  	mpop/scene.py

- Corrected mipp slicing in mipp.py. Added keyword for selecting
  datatype in cfscene.py. Corrected transformation for netCDF data type
  in cfscene.py. [Esben S. Nielsen]
- New add_history function, and some changes in the netcdf handling.
  [Martin Raspaud]
- Upgrade: Upgraded the assemble_segments module to use only one
  coordinate class. [Martin Raspaud]
- Cosmetics: Added log message when slicing in mipp. [Martin Raspaud]
- Move everything to a mpop folder, so that import mpop should be used.
  [Martin Raspaud]
- WIP: Completing the nc4 reader. [Martin Raspaud]
- Doc: Added credits. [Martin Raspaud]
- Doc: updated build for github. [Martin Raspaud]
- Feature: Started to support arithmetic operations on channels. [Martin
  Raspaud]
- Feature: support for calibration flag for met 9. [Martin Raspaud]
- Cosmetics: Added names to copyrigths. [Martin Raspaud]
- Changed default logging. [Esben S. Nielsen]
- Merge branch 'dmi_fix' into unstable. [Martin Raspaud]

  Conflicts:
  	pp/scene.py

- Added fill_valued as a keyworded argument. [Lars Orum Rasmussen]
- Fixed oversampling error when pyresample is not present. Added
  compression as default option when writing netCDF files. [Esben S.
  Nielsen]
- Moved pyresample and osgeo dependency in geo_image.py. [Esben S.
  Nielsen]
- Feature: support umarf files for eps avhrr. [Martin Raspaud]
- Feature: support the load_again flag for meteosat 9. [Martin Raspaud]
- Feature: Allows passing arguments to reader plugins in
  SatelliteScene.load, and in particular "calibrate" to mipp. [Martin
  Raspaud]
- Feature: added the fill_value argument to channel_image function.
  [Martin Raspaud]
- Cosmetics: reorganized imports. [Martin Raspaud]
- Cosmetics: Updated some template files. [Martin Raspaud]
- Feature: Added the resave argument for saving projector objects.
  [Martin Raspaud]
- Installation: Updated version number, removed obsolete file to
  install, and made the package non zip-safe. [Martin Raspaud]
- Testing: Added tests for pp.satellites, and some cosmetics. [Martin
  Raspaud]
- Feature: Handled the case of several instruments for
  get_satellite_class. [Martin Raspaud]
- Cosmetics: changed the name of the satellite classes generated on the
  fly. [Martin Raspaud]
- Testing: more on scene unit tests. [Martin Raspaud]
- Testing: started integration testing of pp core parts. [Martin
  Raspaud]
- Testing: completed seviri tests. [Martin Raspaud]
- Testing: completed avhrr test. [Martin Raspaud]
- Testing: Added tests for instruments : seviri, mviri, avhrr. [Martin
  Raspaud]
- Testing: took away prerequisites tests for python 2.4 compatibility.
  [Martin Raspaud]
- Testing: final adjustments for visir. [Martin Raspaud]
- Testing: visir tests complete. [Martin Raspaud]
- Testing: fixed nosetest running in test_visir. [Martin Raspaud]
- Testing: corrected scene patching for visir tests. [Martin Raspaud]
- Tests: started testing the visir instrument. [Martin Raspaud]
- Cosmetics and documentation in the scene module. [Martin Raspaud]
- Feature: better handling of tags and gdal options in geo_images.
  [Martin Raspaud]
- Cleanup: removed uneeded hardcoded satellites and instruments. [Martin
  Raspaud]
- Documentation: Updated readme, with link to the documentation. [Martin
  Raspaud]
- Documentation: Added a paragraph on geolocalisation. [Martin Raspaud]
- Refactoring: took away the precompute flag from the projector
  constructor, added the save method instead. [Martin Raspaud]
- Cosmetics. [Martin Raspaud]
- Cosmetics. [Martin Raspaud]
- Feature: pyresample 0.7 for projector, and enhanced unittesting.
  [Martin Raspaud]
- New template file for areas. [Martin Raspaud]
- Feature: First draft for the hrpt reading (using aapp) and eps1a
  reading (using aapp and kai). [Martin Raspaud]
- Cosmetics: cleaning up the etc directory. [Martin Raspaud]
- Testing: Basic mipp testing. [Martin Raspaud]
- Cosmetics: cfscene. [Martin Raspaud]
- Feature: One mipp reader fits all :) [Martin Raspaud]
- Feature: helper "debug_on" function. [Martin Raspaud]
- Feature: save method for satscene. Supports only netcdf4 for now.
  [Martin Raspaud]
- Feature: reload keyword for loading channels. [Martin Raspaud]
- Documentation: better pp.satellites docstring. [Martin Raspaud]
- Testing: updated the test_scene file to reflect scene changes. [Martin
  Raspaud]
- Documentation: changed a couple of docstrings. [Martin Raspaud]
- Feature: support pyresample areas in geo images. [Martin Raspaud]
- Cosmetics: changing area_id to area. [Martin Raspaud]
- Feature: adding metadata handling to channels. [Martin Raspaud]
- Feature: now scene and channel accept a pyresample area as area
  attribute. [Martin Raspaud]
- Enhancement: making a better mipp plugin. [Martin Raspaud]
- Feature: Finished the netcdf writer. [Martin Raspaud]
- Feature: updated the netcdf writer and added a proxy scene class for
  cf conventions. [Martin Raspaud]
- Documentation: big update. [Martin Raspaud]
- Documentation: quickstart now passes the doctest. [Martin Raspaud]
- Documentation: reworking. [Martin Raspaud]
- Feature: Moved get_satellite_class and build_satellite_class to
  pp.satellites. [Martin Raspaud]
- Doc: starting documentation update. [Martin Raspaud]
- Enhanced mipp reader. [Martin Raspaud]

  * Added metadata when loading scenes.
  * Added slicing when reading data from seviri
  * Added a draft generic reader

- Cosmetics: enhanced error description and debug message in aapp1b,
  giving names to loaded/missing files. [Martin Raspaud]
- Testing: updated test_scene. [Martin Raspaud]
- Feature: Added automatic retreiving of product list for a given
  satellite. [Martin Raspaud]
- Cleaning: remove class retrieving and building from runner.py. [Martin
  Raspaud]
- Cosmetics: Better error message in scene when a reader is not found,
  plus some code enbelishment. [Martin Raspaud]
- Feature: made scene object iteratable (channels are iterated). [Martin
  Raspaud]
- Feature: Adding functions to retreive a satellite class from the
  satellites name and to build it on the fly from a configuration file.
  [Martin Raspaud]
- Testing: more on channel. [Martin Raspaud]
- Testing: added test for pp.scene.assemble_swaths. [Martin Raspaud]
- Testing: scene loading tested. [Martin Raspaud]
- Cleaning: test_scene is now more pylint friendly. [Martin Raspaud]
- Feature: extended scene test. [Martin Raspaud]
- Feature: more testing of scene.py. [Martin Raspaud]
- Merge branch 'unstable' of github.com:mraspaud/mpop into unstable.
  [Martin Raspaud]

  Conflicts:
  	pp/test_scene.py

- Feature: Enhanced unitests for scene. [Martin Raspaud]
- Feature: Enhanced unitests for scene. [Martin Raspaud]
- Tests: Improving unittests for channel classes. [Martin Raspaud]
- Feature: Project function won't crash if pyresample can't be loaded.
  Returns the untouched scene instead. [Martin Raspaud]
- Rewrote Filewatcher code. [Martin Raspaud]
- Feature: added the refresh option to filewatcher to call the
  processing function even if no new file has come. [Martin Raspaud]
- Refactoring: satellite, number, variant arguments to runner __init__
  are now a single list argument. [Martin Raspaud]
- Cleaning: Removing pylint errors from runner.py code. [Martin Raspaud]
- Resolution can now be a floating point number. [Martin Raspaud]
- Added the osgeo namespace when importing gdal. [Martin Raspaud]
- Warning: Eps spline interpolation does not work around poles. [Martin
  Raspaud]
- Added the "info" attribute to channel and scene as metadata holder.
  [Martin Raspaud]
- Functionality: Automatically build satellite classes from config
  files. [Martin Raspaud]
- Added copyright notices and updated version. [Martin Raspaud]
- Changed channel names for seviri. [Martin Raspaud]
- Added info stuff in mipp reader. [Martin Raspaud]
- Added info.area_name update on projection. [Martin Raspaud]
- Added quick mode for projecting fast and dirty. [Martin Raspaud]
- Added single channel image building. [Martin Raspaud]
- Added support for gdal_options when saving a geo_image. [Martin
  Raspaud]
- Made satout a package. [Martin Raspaud]
- Added a few information tags. [Martin Raspaud]
- Added support for mipp reading of met 09. [Martin Raspaud]
- Added reader and writer to netcdf format. [Martin Raspaud]
- Added info object to the scene object in preparation for the netCDF/CF
  writer. [Adam Dybbroe]
- Added support for FY3 satellite and MERSI instrument. [Adam Dybbroe]
- Merge branch 'unstable' of git@github.com:mraspaud/mpop into unstable.
  [Martin Raspaud]

  Conflicts:
  	imageo/test_image.py

  Conflicts:
  	imageo/test_image.py

- Bugfix in image unit test: testing "almost equal" instead of "equal"
  for image inversion (floating point errors). [Martin Raspaud]
- Bugfix in image unit test: testing "almost equal" instead of "equal"
  for image inversion (floating point errors). [Martin Raspaud]
- Modified image inversion unit test to reflect new behaviour. [Martin
  Raspaud]
- New rebase. [Martin Raspaud]
