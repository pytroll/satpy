=====================
SatPy's Documentation
=====================

SatPy is a python library for reading and manipulating
meteorological remote sensing data and writing it to various image and
data file formats. SatPy comes with the ability to make various RGB
composites directly from satellite instrument channel data or higher level
processing output. The
`pyresample <http://pyresample.readthedocs.io/en/latest/>`_ package is used
to resample data to different uniform areas or grids. Various atmospheric
corrections and visual enhancements are also provided, either directly in
SatPy or from those in the
`PySpectral <https://pyspectral.readthedocs.io/en/develop/>`_ and
`TrollImage <http://trollimage.readthedocs.io/en/latest/>`_ packages.

Go to the project_ page for source code and downloads.

It is designed to be easily extendable to support any meteorological satellite
by the creation of plugins (readers, compositors, writers, etc). The table at
the bottom of this page shows the input formats supported by the base SatPy
installation.

.. note::

    SatPy's interfaces are not guaranteed stable and may change until version
    1.0 when backwards compatibility will be a main focus.

.. _project: http://github.com/pytroll/satpy

.. toctree::
    :maxdepth: 2

    overview
    install
    examples
    quickstart
    readers
    composites
    resample
    writers
    multiscene
    dev_guide/index
    SatPy API <api/satpy>


.. _reader_table:

.. list-table:: SatPy Readers
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
      - `hrit_goes`
      - Nominal
    * - Electro-L N2 MSU-GS data in HRIT format
      - `hrit_electrol`
      - Nominal
    * - NOAA 15 to 19, Metop A to C AVHRR data in AAPP format
      - `avhrr_aapp_l1b`
      - Nominal
    * - Metop A to C AVHRR in native level 1 format
      - `avhrr_eps_l1b`
      - Nominal
    * - Tiros-N, NOAA 7 to 19 AVHRR data in GAC and LAC format
      - `gac_lac_l1b`
      - Nominal
    * - NOAA 15 to 19 AVHRR data in raw HRPT format
      - `avhrr_hrpt_l1b`
      - In development
    * - GCOM-W1 AMSR2 data in HDF5 format
      - `amsr2_l1b`
      - Nominal
    * - MTG FCI data in netcdf format
      - `fci_fdhsi`
      - In development
    * - Callipso Caliop data in EOS-hdf4 format
      - `hdf4_caliopv3`
      - In development
    * - Terra and Aqua MODIS data in EOS-hdf4 level-1 format as produced by IMAPP and IPOPP or downloaded from LAADS
      - `hdfeos_l1b`
      - Nominal
    * - NWCSAF MSG 2016 products in netCDF4 format
      - `nc_nwcsaf_msg`
      - In development
    * - NWCSAF PPS 2014 products in netCDF4 format
      - `nc_nwcsaf_pps`
      - | Not yet support for remapped netCDF products. 
        | Only the standard swath based output is supported.
        | CPP products not supported yet
    * - Sentinel-1 A and B SAR-C data in SAFE format
      - `safe_sar_c`
      - Nominal
    * - Sentinel-2 A and B MSI data in SAFE format
      - `safe_msi`
      - Nominal
    * - Sentinel-3 A and B OLCI data in netCDF4 format
      - `nc_olci`
      - Nominal
    * - Sentinel-3 A and B SLSTR data in netCDF4 format
      - `nc_slstr`
      - In development
    * - OSISAF SST data in GHRSST (netcdf) format
      - `ghrsst_osisaf`
      - In development
    * - NUCAPS EDR Retrieval in NetCDF4 format
      - `nucaps`
      - Nominal
    * - NOAA Level 2 ACSPO SST data in netCDF4 format
      - `acspo`
      - Nominal
    * - GEOstationary Cloud Algorithm Test-bed (GEOCAT)
      - `geocat`
      - Nominal
    * - The Clouds from AVHRR Extended (CLAVR-x)
      - `clavrx`
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
    * - AAPP MAIA VIIRS and AVHRR products in hdf5 format
      - `maia`
      - Nominal
    * - GRIB2 format
      - `grib`
      - Beta

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
