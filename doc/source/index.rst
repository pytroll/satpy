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
    :widths: 45 25 30

    * - Description
      - Reader name
      - Status
    * - MSG (Meteosat 8 to 11) SEVIRI data in HRIT format
      - `seviri_l1b_hrit`
      - Nominal
    * - MSG (Meteosat 8 to 11) SEVIRI data in native format
      - `seviri_l1b_native`
      - HRV full disk data cannot be remapped.
    * - MSG (Meteosat 8 to 11) SEVIRI data in netCDF format
      - `seviri_l1b_nc`
      - | HRV channel not supported, incomplete metadata
        | in the files. EUMETSAT has been notified.
    * - Himawari 8 and 9 AHI data in HSD format
      - `ahi_hsd`
      - Nominal
    * - Himawari 8 and 9 AHI data in HRIT format
      - `ahi_hrit`
      - Nominal
    * - MTSAT-1R JAMI data in JMA HRIT format
      - `jami_hrit`
      - Beta
    * - MTSAT-2 Imager data in JMA HRIT format
      - `mtsat2-imager_hrit`
      - Beta
    * - GOES 16 imager data in netcdf format
      - `abi_l1b`
      - Nominal
    * - GOES 11 to 15 imager data in HRIT format
      - `goes-imager_hrit`
      - Nominal
    * - GOES 8 to 15 imager data in netCDF format (from NOAA CLASS)
      - `goes-imager_nc`
      - Beta
    * - Electro-L N2 MSU-GS data in HRIT format
      - `electrol_hrit`
      - Nominal
    * - NOAA 15 to 19, Metop A to C AVHRR data in AAPP format
      - `avhrr_l1b_aapp`
      - Nominal
    * - Metop A to C AVHRR in native level 1 format
      - `avhrr_l1b_eps`
      - Nominal
    * - Tiros-N, NOAA 7 to 19 AVHRR data in GAC and LAC format
      - `avhrr_l1b_gaclac`
      - Nominal
    * - NOAA 15 to 19 AVHRR data in raw HRPT format
      - `avhrr_l1b_hrpt`
      - In development
    * - GCOM-W1 AMSR2 data in HDF5 format
      - `amsr2_l1b`
      - Nominal
    * - MTG FCI Level 1C data for Full Disk High Spectral Imagery (FDHSI) in netcdf format
      - `fci_l1c_fdhsi`
      - In development
    * - Callipso Caliop Level 2 Cloud Layer data (v3) in EOS-hdf4 format
      - `caliop_l2_cloud`
      - In development
    * - Terra and Aqua MODIS data in EOS-hdf4 level-1 format as produced by IMAPP and IPOPP or downloaded from LAADS
      - `modis_l1b`
      - Nominal
    * - NWCSAF GEO 2016 products in netCDF4 format (limited to SEVIRI)
      - `nwcsaf-geo`
      - In development
    * - NWCSAF PPS 2014, 2018 products in netCDF4 format
      - `nwcsaf-pps_nc`
      - | Not yet support for remapped netCDF products.
        | Only the standard swath based output is supported.
        | CPP products not supported yet
    * - Sentinel-1 A and B SAR-C data in SAFE format
      - `sar-c_safe`
      - Nominal
    * - Sentinel-2 A and B MSI data in SAFE format
      - `msi_safe`
      - Nominal
    * - Sentinel-3 A and B OLCI Level 1B data in netCDF4 format
      - `olci_l1b`
      - Nominal
    * - Sentinel-3 A and B OLCI Level 2 data in netCDF4 format
      - `olci_l2`
      - Nominal
    * - Sentinel-3 A and B SLSTR data in netCDF4 format
      - `slstr_l1b`
      - In development
    * - OSISAF SST data in GHRSST (netcdf) format
      - `ghrsst_l3c_sst`
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
    * - SNPP VIIRS data in HDF5 SDR format
      - `viirs_sdr`
      - Nominal
    * - SNPP VIIRS data in netCDF4 L1B format
      - `viirs_l1b`
      - Nominal
    * - SNPP VIIRS SDR data in HDF5 Compact format
      - `viirs_compact`
      - Nominal
    * - AAPP MAIA VIIRS and AVHRR products in hdf5 format
      - `maia`
      - Nominal
    * - VIIRS EDR Flood data in hdf4 format
      - `viirs_edr_flood`
      - Beta
    * - GRIB2 format
      - `grib`
      - Beta
    * - SCMI ABI L1B format
      - `abi_l1b_scmi`
      - Beta

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
