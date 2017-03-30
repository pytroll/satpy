.. NWCSAF/satpy documentation master file, created by
   sphinx-quickstart on Fri Sep 25 16:58:28 2009.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==================================
 Welcome to satpy's documentation!
==================================

The Meteorological Post-Processing package is a python library for generating
RGB products for meteorological remote sensing. As such it can create RGB
composites directly from satellite instrument channels, or take advantage of
precomputed PGEs.

Get to the project_ page, with source and downloads.

It is designed to be easily extendable to support any meteorological satellite
by the creation of plugins. In the base distribution, we provide support for:


.. list-table:: Data formats supported in satpy
   :header-rows: 1

   * - Description
     - Reader name
   * - MSG (Meteosat 8 to 11) Seviri data in HRIT format
     - `hrit_msg`
   * - Himawari 8 and 9 AHI data in HSD format
     - `ahi_hsd`
   * - Himawari 8 and 9 AHI data in HRIT format
     - `hrit_jma`
   * - GOES 16 imager data in netcdf format
     - `abi_l1b`
   * - GOES 11 to 15 imager data in HRIT format
     - `mipp_xrit`
   * - Electro-L N2 MSU-GS data in HRIT format
     - `hrit_electrol`
   * - NOAA 15 to 19, Metop A to C AVHRR data in AAPP format
     - `aapp_l1b`
   * - Metop A to C AVHRR in native level 1 format
     - `epsl1b`
   * - Tiros-N, NOAA 7 to 19 AVHRR data in GAC and LAC format
     - `gac_lac_l1b`
   * - NOAA 15 to 19 AVHRR data in raw HRPT format
     - `hrpt`
   * - GCOM-W1 AMSR2 data in HDF5 format
     - `amsr2_l1b`
   * - MTG FCI data in netcdf format
     - `fci_fdhsi`
   * - Callipso Caliop data in EOS-hdf4 format
     - `hdf4_caliopv3`
   * - Terra and Aqua MODIS data in EOS-hdf4 format
     - `hdfeos_l1b`
   * - NWCSAF MSG 2016 products in netCDF4 format
     - `nc_nwcsaf_msg`
   * - Sentinel-1 A and B SAR-C data in SAFE format
     - `sar_c`
   * - Sentinel-2 A and B MSI data in SAFE format
     - `safe_msi`
   * - Sentinel-3 A and B OLCI data in netCDF4 format
     - `nc_olci`
   * - Sentinel-3 A and B SLSTR data in netCDF4 format
     - `nc_slstr`
   * - OSISAF SST data in netcdf format
     - `ghrsst_osisaf`
   * - NUCAPS EDR Retrieval in NetCDF4 format
     - `nucaps`
   * - SNPP VIIRS data in hdf5 SDR format
     - `viirs_sdr`
   * - SNPP VIIRS data in netCDF4 L1B format
     - `viirs_sdr`
   * - SNPP VIIRS data in hdf5 Compact format
     - `viirs_compact`

Reprojection of data is also available through the use of pyresample_.

.. _project: http://github.com/pytroll/satpy
.. _pyresample: http://pyresample.googlecode.com

.. toctree::
   :maxdepth: 2

   install
   quickstart
   satpy

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
