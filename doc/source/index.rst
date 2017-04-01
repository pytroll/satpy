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
     - Status
   * - MSG (Meteosat 8 to 11) Seviri data in HRIT format
     - `hrit_msg`
     - Nominal
   * - MSG (Meteosat 8 to 11) SEVIRI data in native format
     - `native_msg`
     - | No support for reading sub-section of the
       | full disk. HRV data cannot be remapped.
   * - Himawari 8 and 9 AHI data in HSD format
     - `ahi_hsd`
     - Nominal
   * - Himawari 8 and 9 AHI data in HRIT format
     - `hrit_jma`
     - Nominal
   * - GOES 16 imager data in netcdf format
     - `abi_l1b`
     - Nominal
   * - GOES 11 to 15 imager data in HRIT format
     - `mipp_xrit`
     - Nominal
   * - Electro-L N2 MSU-GS data in HRIT format
     - `hrit_electrol`
     - Nominal
   * - NOAA 15 to 19, Metop A to C AVHRR data in AAPP format
     - `aapp_l1b`
     - Nominal
   * - Metop A to C AVHRR in native level 1 format
     - `epsl1b`
     - Nominal
   * - Tiros-N, NOAA 7 to 19 AVHRR data in GAC and LAC format
     - `gac_lac_l1b`
     - Nominal
   * - NOAA 15 to 19 AVHRR data in raw HRPT format
     - `hrpt`
     - Nominal
   * - GCOM-W1 AMSR2 data in HDF5 format
     - `amsr2_l1b`
     - Nominal
   * - MTG FCI data in netcdf format
     - `fci_fdhsi`
     - Nominal
   * - Callipso Caliop data in EOS-hdf4 format
     - `hdf4_caliopv3`
     - Nominal
   * - Terra and Aqua MODIS data in EOS-hdf4 format
     - `hdfeos_l1b`
     - Nominal
   * - NWCSAF MSG 2016 products in netCDF4 format
     - `nc_nwcsaf_msg`
     - Nominal
   * - Sentinel-1 A and B SAR-C data in SAFE format
     - `sar_c`
     - Nominal
   * - Sentinel-2 A and B MSI data in SAFE format
     - `safe_msi`
     - Nominal
   * - Sentinel-3 A and B OLCI data in netCDF4 format
     - `nc_olci`
     - Nominal
   * - Sentinel-3 A and B SLSTR data in netCDF4 format
     - `nc_slstr`
     - Nominal
   * - OSISAF SST data in GHRSST (netcdf) format
     - `ghrsst_osisaf`
     - Nominal
   * - NUCAPS EDR Retrieval in NetCDF4 format
     - `nucaps`
     - Nominal
   * - SNPP VIIRS data in hdf5 SDR format
     - `viirs_sdr`
     - Nominal
   * - SNPP VIIRS data in netCDF4 L1B format
     - `viirs_sdr`
     - Nominal
   * - SNPP VIIRS data in hdf5 Compact format
     - `viirs_compact`
     - Nominal

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
