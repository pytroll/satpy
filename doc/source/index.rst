=====================
Satpy's Documentation
=====================

Satpy is a python library for reading, manipulating, and writing data from
remote-sensing earth-observing meteorological satellite instruments. Satpy
provides users with readers that convert geophysical parameters from various
file formats to the common Xarray :class:`~xarray.DataArray` and
:class:`~xarray.Dataset` classes for easier interoperability with other
scientific python libraries. Satpy also provides interfaces for creating
RGB (Red/Green/Blue) images and other composite types by combining data
from multiple instrument bands or products. Various atmospheric corrections
and visual enhancements are provided for improving the usefulness and quality
of output images. Output data can be written to
multiple output file formats such as PNG, GeoTIFF, and CF standard NetCDF
files. Satpy also allows users to resample data to geographic projected grids
(areas). Satpy is maintained by the open source
`Pytroll <http://pytroll.github.io/>`_ group.

The Satpy library acts as a high-level abstraction layer on top of other
libraries maintained by the Pytroll group including:

- `Pyresample <http://pyresample.readthedocs.io/en/latest/>`_
- `PySpectral <https://pyspectral.readthedocs.io/en/develop/>`_
- `Trollimage <http://trollimage.readthedocs.io/en/latest/>`_
- `Pycoast <https://pycoast.readthedocs.io/en/latest/>`_
- `Pydecorate <https://pydecorate.readthedocs.io/en/latest/>`_
- `python-geotiepoints <https://python-geotiepoints.readthedocs.io/en/latest/>`_
- `pyninjotiff <https://github.com/pytroll/pyninjotiff>`_

Go to the Satpy project_ page for source code and downloads.

Satpy is designed to be easily extendable to support any meteorological
satellite by the creation of plugins (readers, compositors, writers, etc).
The table at the bottom of this page shows the input formats supported by
the base Satpy installation.

.. note::

    Satpy's interfaces are not guaranteed stable and may change until version
    1.0 when backwards compatibility will be a main focus.

.. _project: http://github.com/pytroll/satpy

.. toctree::
    :maxdepth: 2

    overview
    install
    data_download
    examples
    quickstart
    readers
    composites
    resample
    enhancements
    writers
    multiscene
    dev_guide/index

.. toctree::
    :maxdepth: 1

    Satpy API <api/satpy>
    faq

.. _reader_table:

.. list-table:: Satpy Readers
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
    * - VIIRS EDR Active Fires data in NetCDF4 & CSV .txt format
      - `viirs_edr_active_fires`
      - Beta
    * - VIIRS EDR Flood data in hdf4 format
      - `viirs_edr_flood`
      - Beta
    * - GRIB2 format
      - `grib`
      - Beta
    * - SCMI ABI L1B format
      - `abi_l1b_scmi`
      - Beta
    * - VIRR data in HDF5 format
      - `virr_l1b`
      - Beta
    * - MERSI-2 L1B data in HDF5 format
      - `mersi2_l1b`
      - Beta
    * - Vaisala Global Lightning Dataset GLD360 data in ASCII format
      - `vaisala_gld360`
      - Beta
    * - TROPOMI L2 data in NetCDF4 format
      - `tropomi_l2`
      - Beta
    * - Hydrology SAF products in GRIB format
      - `hsaf_grib`
      - | Beta
        | Only the h03, h03b, h05 and h05B products are supported at-present


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
