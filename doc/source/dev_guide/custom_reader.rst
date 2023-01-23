=================================
 Adding a Custom Reader to Satpy
=================================

In order to add a reader to satpy, you will need to create two files:

 - a YAML file for describing the files to read and the datasets that are available
 - a python file implementing the actual reading of the datasets and metadata

Satpy implements readers by defining a single "reader" object that pulls
information from one or more file handler objects. The base reader class
provided by Satpy is enough for most cases and does not need to be modified.
The individual file handler classes do need to be created due to the small
differences between file formats.

The below documentation will walk through each part of making a reader in
detail. To do this we will implement a reader for the EUMETSAT NetCDF
format for SEVIRI data.

.. _reader_naming:

Naming your reader
------------------

Satpy tries to follow a standard scheme for naming its readers. These names
are used in filenames, but are also used by users so it is important that
the name be recognizable and clear. Although some
special cases exist, most fit in to the following naming scheme:

.. parsed-literal::

    <sensor>[_<processing level>[_<level detail>]][_<file format>]

All components of the name should be lowercase and use underscores as the main
separator between fields. Hyphens should be used as an intra-field separator
if needed (ex. goes-imager).

:sensor: The first component of the name represents the sensor or
    instrument that observed the data stored in the files being read. If
    the files are the output of a specific processing software or a certain
    algorithm implementation that supports multiple sensors then a lowercase
    version of that software's name should be used (e.g. clavrx for CLAVR-x,
    nucaps for NUCAPS). The ``sensor`` field is the only required field of
    the naming scheme. If it is actually an instrument name then the reader
    name should include one of the other optional fields. If sensor is a
    software package then that may be enough without any additional
    information to uniquely identify the reader.
:processing level: This field marks the specific level of processing or
    calibration that has been performed to produce the data in the files being
    read. Common values of this field include: ``sdr`` for Sensor Data
    Record (SDR), ``edr`` for Environmental Data Record (EDR), ``l1b`` for
    Level 1B, and ``l2`` for Level 2.
:level detail: In cases where the processing level is not enough to completely
    define the reader this field can be used to provide a little more context.
    For example, some VIIRS EDR products are specific to a particular field
    of study or type of scientific event, like a flood or cloud product. In
    these cases the detail field can be added to produce a name like
    ``viirs_edr_flood``. This field shouldn't be used unless processing level
    is also specified.
:file format: If the file format of the files is informative to the user or
    can distinguish one reader from another then this field should be
    specified. Common format names should be abbreviated following existing
    abbreviations like ``nc`` for NetCDF3 or NetCDF4, ``hdf`` for HDF4, ``h5`` for
    HDF5.

The existing :ref:`reader's table <reader_table>` can be used for reference.
When in doubt, reader names can be discussed in the GitHub pull
request when this reader is added to Satpy, or in a GitHub issue.

The YAML file
-------------

If your reader is going to be part of Satpy, the YAML file should be
located in the ``satpy/etc/readers`` directory, along with the YAML
files for all other readers.  If you are developing a reader for internal
purposes (such as for unpublished data), the YAML file should be located
in any directory in ``$SATPY_CONFIG_PATH`` within the subdirectory
``readers/`` (see :doc:`../../config`).

The YAML file is composed of three sections:

 - the :ref:`reader <custom_reader_reader_section>` section,
   that provides basic parameters for the reader
 - the :ref:`file_types <custom_reader_file_types_section>` section,
   that gives the patterns of the files this reader can handle
 - the :ref:`datasets <custom_reader_datasets_section>` section,
   that describes the datasets available from this reader

.. _custom_reader_reader_section:

The ``reader`` section
~~~~~~~~~~~~~~~~~~~~~~

The ``reader`` section provides basic parameters for the overall reader.

The parameters to provide in this section are:

 name
    This is the name of the reader, it should be the same as the
    filename (without the .yaml extension). The naming convention for
    this is described above in the :ref:`reader_naming` section above.
    short_name (optional): Human-readable version of the reader 'name'.
    If not provided, applications using this can default to taking the 'name',
    replacing ``_`` with spaces and uppercasing every letter.
 long_name
    Human-readable title for the reader. This may be used as a
    section title on a website or in GUI applications using Satpy. Default
    naming scheme is ``<space program> <sensor> Level <level> [<format>]``.
    For example, for the ``abi_l1b`` reader this is ``"GOES-R ABI Level 1b"``
    where "GOES-R" is the name of the program and **not** the name of the
    platform/satellite. This scheme may not work for all readers, but in
    general should be followed. See existing readers for more examples.
 description
    General description of the reader. This may include any
    `restructuredtext <http://docutils.sourceforge.net/docs/user/rst/quickref.html>`_
    formatted text like links to PDFs or sites with more information on the
    file format. This can be multiline if formatted properly in YAML (see
    example below).
 status
    The status of the reader (one of: Nominal, Beta, Alpha)
 supports_fsspec
    If the reader supports reading data via fsspec (either true or false).
 sensors
    The list of sensors this reader will support. This must be
    all lowercase letters for full support throughout in Satpy.
 reader
    The main python reader class to use, in most cases the
    ``FileYAMLReader`` is a good choice.

.. code:: yaml

    reader:
      name: seviri_l1b_nc
      short_name: SEVIRI L1b NetCDF4
      long_name: MSG SEVIRI Level 1b (NetCDF4)
      description: >
        NetCDF4 reader for EUMETSAT MSG SEVIRI Level 1b files.
      sensors: [seviri]
      reader: !!python/name:satpy.readers.yaml_reader.FileYAMLReader

Optionally, if you need to customize the ``DataID`` for this reader, you can provide the
relevant keys with a ``data_identification_keys`` item here. See the :doc:`satpy_internals`
section for more information.

.. _custom_reader_file_types_section:

The ``file_types`` section
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each file type needs to provide:

 - ``file_reader``, the class that will
   handle the files for this reader, that you will implement in the
   corresponding python file. See the :ref:`custom_reader_python`
   section below.
 - ``file_patterns``, the
   patterns to match to find files this reader can handle. The syntax to
   use is basically the same as ``format`` with the addition of time. See
   the `trollsift package documentation <https://trollsift.readthedocs.io/en/latest/usage.html>`__
   for more details.
 - Optionally, a file type can have a ``requires``
   field: it is a list of file types that the current file types needs to
   function. For example, the HRIT MSG format segment files each need a
   prologue and epilogue file to be read properly, hence in this case we
   have added ``requires: [HRIT_PRO, HRIT_EPI]`` to the file type
   definition.

.. code:: yaml

    file_types:
        nc_seviri_l1b:
            file_reader: !!python/name:satpy.readers.nc_seviri_l1b.NCSEVIRIFileHandler
            file_patterns: ['W_XX-EUMETSAT-Darmstadt,VIS+IR+IMAGERY,{satid:4s}+SEVIRI_C_EUMG_{processing_time:%Y%m%d%H%M%S}.nc']
        nc_seviri_l1b_hrv:
            file_reader: !!python/name:satpy.readers.nc_seviri_l1b.NCSEVIRIHRVFileHandler
            file_patterns: ['W_XX-EUMETSAT-Darmstadt,HRV+IMAGERY,{satid:4s}+SEVIRI_C_EUMG_{processing_time:%Y%m%d%H%M%S}.nc']

.. _custom_reader_datasets_section:

The ``datasets`` section
~~~~~~~~~~~~~~~~~~~~~~~~

The datasets section describes each dataset available in the files. The
parameters provided are made available to the methods of the
implemented python class.

If your input files contain all the necessary metadata or you have a lot
of datasets to configure look at the :ref:`custom_reader_available_datasets`
section below. Implementing this will save you from having to write
a lot of configuration in the YAML files.

Parameters you can define for example are:

 - name
 - sensor
 - resolution
 - wavelength
 - polarization
 - standard\_name: The
   `CF standard name <http://cfconventions.org/Data/cf-standard-names/70/build/cf-standard-name-table.html>`_
   for the dataset that will be used to determine the type of data. See
   existing readers for common standard names in Satpy or the CF standard name
   documentation for other available names or how to define your own. Satpy
   does not currently have a hard requirement on these names being completely
   CF compliant, but consistency across readers is important.
 - units: The units of the data when returned by the file handler. Although
   not technically a requirement, it is common for Satpy datasets to use "%"
   for reflectance fields and "K" for brightness temperature fields.
 - modifiers: The modification(s) that have already been applied to the data
   when it is returned by the file handler. Only a few of these have been
   standardized across Satpy, but are based on the names of the modifiers
   configured in the "composites" YAML files. Examples include
   ``sunz_corrected`` or ``rayleigh_corrected``. See the
   `metadata wiki <https://github.com/pytroll/satpy/wiki/Metadata-names>`_
   for more information.
 - file\_type: Name of file type (see above).
 - coordinates: An optional two-element list with the names of the longitude
   and latitude datasets describing the location of this dataset. This
   is optional if the data being read is gridded already. Swath data,
   from example data from some polar-orbiting satellites, should have these
   defined or no geolocation information will be available when the data
   are loaded. For gridded datasets a ``get_area_def`` function will be
   implemented in python (see below) to define geolocation information.
 - Any other field that is relevant for the reader or could be useful metadata
   provided to the user.

This section can be copied and adapted simply from existing seviri
readers, like for example the ``msg_native`` reader.

.. code:: yaml

    datasets:
      HRV:
        name: HRV
        resolution: 1000.134348869
        wavelength: [0.5, 0.7, 0.9]
        calibration:
          reflectance:
            standard_name: toa_bidirectional_reflectance
            units: "%"
          radiance:
            standard_name: toa_outgoing_radiance_per_unit_wavelength
            units: W m-2 um-1 sr-1
          counts:
            standard_name: counts
            units: count
        file_type: nc_seviri_l1b_hrv

      IR_016:
        name: IR_016
        resolution: 3000.403165817
        wavelength: [1.5, 1.64, 1.78]
        calibration:
          reflectance:
            standard_name: toa_bidirectional_reflectance
            units: "%"
          radiance:
            standard_name: toa_outgoing_radiance_per_unit_wavelength
            units: W m-2 um-1 sr-1
          counts:
            standard_name: counts
            units: count
        file_type: nc_seviri_l1b
        nc_key: 'ch3'

      IR_039:
        name: IR_039
        resolution: 3000.403165817
        wavelength: [3.48, 3.92, 4.36]
        calibration:
          brightness_temperature:
            standard_name: toa_brightness_temperature
            units: K
          radiance:
            standard_name: toa_outgoing_radiance_per_unit_wavelength
            units: W m-2 um-1 sr-1
          counts:
            standard_name: counts
            units: count
        file_type: nc_seviri_l1b
        nc_key: 'ch4'

      IR_087:
        name: IR_087
        resolution: 3000.403165817
        wavelength: [8.3, 8.7, 9.1]
        calibration:
          brightness_temperature:
            standard_name: toa_brightness_temperature
            units: K
          radiance:
            standard_name: toa_outgoing_radiance_per_unit_wavelength
            units: W m-2 um-1 sr-1
          counts:
            standard_name: counts
            units: count
        file_type: nc_seviri_l1b

      IR_097:
        name: IR_097
        resolution: 3000.403165817
        wavelength: [9.38, 9.66, 9.94]
        calibration:
          brightness_temperature:
            standard_name: toa_brightness_temperature
            units: K
          radiance:
            standard_name: toa_outgoing_radiance_per_unit_wavelength
            units: W m-2 um-1 sr-1
          counts:
            standard_name: counts
            units: count
        file_type: nc_seviri_l1b

      IR_108:
        name: IR_108
        resolution: 3000.403165817
        wavelength: [9.8, 10.8, 11.8]
        calibration:
          brightness_temperature:
            standard_name: toa_brightness_temperature
            units: K
          radiance:
            standard_name: toa_outgoing_radiance_per_unit_wavelength
            units: W m-2 um-1 sr-1
          counts:
            standard_name: counts
            units: count
        file_type: nc_seviri_l1b

      IR_120:
        name: IR_120
        resolution: 3000.403165817
        wavelength: [11.0, 12.0, 13.0]
        calibration:
          brightness_temperature:
            standard_name: toa_brightness_temperature
            units: K
          radiance:
            standard_name: toa_outgoing_radiance_per_unit_wavelength
            units: W m-2 um-1 sr-1
          counts:
            standard_name: counts
            units: count
        file_type: nc_seviri_l1b

      IR_134:
        name: IR_134
        resolution: 3000.403165817
        wavelength: [12.4, 13.4, 14.4]
        calibration:
          brightness_temperature:
            standard_name: toa_brightness_temperature
            units: K
          radiance:
            standard_name: toa_outgoing_radiance_per_unit_wavelength
            units: W m-2 um-1 sr-1
          counts:
            standard_name: counts
            units: count
        file_type: nc_seviri_l1b

      VIS006:
        name: VIS006
        resolution: 3000.403165817
        wavelength: [0.56, 0.635, 0.71]
        calibration:
          reflectance:
            standard_name: toa_bidirectional_reflectance
            units: "%"
          radiance:
            standard_name: toa_outgoing_radiance_per_unit_wavelength
            units: W m-2 um-1 sr-1
          counts:
            standard_name: counts
            units: count
        file_type: nc_seviri_l1b

      VIS008:
        name: VIS008
        resolution: 3000.403165817
        wavelength: [0.74, 0.81, 0.88]
        calibration:
          reflectance:
            standard_name: toa_bidirectional_reflectance
            units: "%"
          radiance:
            standard_name: toa_outgoing_radiance_per_unit_wavelength
            units: W m-2 um-1 sr-1
          counts:
            standard_name: counts
            units: count
        file_type: nc_seviri_l1b

      WV_062:
        name: WV_062
        resolution: 3000.403165817
        wavelength: [5.35, 6.25, 7.15]
        calibration:
          brightness_temperature:
            standard_name: toa_brightness_temperature
            units: "K"
          radiance:
            standard_name: toa_outgoing_radiance_per_unit_wavelength
            units: W m-2 um-1 sr-1
          counts:
            standard_name: counts
            units: count
        file_type: nc_seviri_l1b

      WV_073:
        name: WV_073
        resolution: 3000.403165817
        wavelength: [6.85, 7.35, 7.85]
        calibration:
          brightness_temperature:
            standard_name: toa_brightness_temperature
            units: "K"
          radiance:
            standard_name: toa_outgoing_radiance_per_unit_wavelength
            units: W m-2 um-1 sr-1
          counts:
            standard_name: counts
            units: count
        file_type: nc_seviri_l1b

The YAML file is now ready and you can move on to writing your python code.

.. _custom_reader_available_datasets:

Dynamic Dataset Configuration
-----------------------------

The above "datasets" section for reader configuration is the most explicit
method for specifying metadata about possible data that can be loaded from
input files. It is also the easiest way for people with little python
experience to customize or add new datasets to a reader. However, some file
formats may have 10s or even 100s of datasets or variations of datasets.
Writing the metadata and access information for every one of these datasets
can easily become a problem. To help in these cases the
:meth:`~satpy.readers.file_handlers.BaseFileHandler.available_datasets`
file handler interface can be used.

This method, if needed, should be implemented in your reader's file handler
classes. The best information for what this method does and how to use it
is available in the
:meth:`API documentation <satpy.readers.file_handlers.BaseFileHandler.available_datasets>`.
This method is good when you want to:

1. Define datasets dynamically without needing to define them in the YAML.
2. Supplement metadata from the YAML file with information from the file
   content (ex. ``resolution``).
3. Determine if a dataset is available by the file contents. This differs from
   the default behavior of a dataset being considered loadable if its
   "file_type" is loaded.

Note that this is considered an advanced interface and involves more advanced
Python concepts like generators. If you need help with anything feel free
to ask questions in your pull request or on the :ref:`Pytroll Slack <dev_help>`.

.. _custom_reader_python:

The python file
---------------

The python files needs to implement a file
handler class for each file type that we want to read. Such a class
needs to implement a few methods:

 - the ``__init__`` method, that takes as arguments

   - the filename (string)
   - the filename info (dict) that we get by parsing the filename using the pattern defined in the yaml file
   - the filetype info that we get from the filetype definition in the yaml file

   This method can also receive other file handler instances as parameter
   if the filetype at hand has requirements. (See the explanation in the
   YAML file filetype section above)

 - the ``get_dataset`` method, which takes as arguments

   - the dataset ID of the dataset to load
   - the dataset info that is the description of the channel in the YAML file

   This method has to return an xarray.DataArray instance if the loading is
   successful, containing the data and :ref:`metadata <dataset_metadata>` of the
   loaded dataset, or return None if the loading was unsuccessful.

   The DataArray should at least have a ``y`` dimension. For data covering
   a 2D region on the Earth, their should be at least a ``y`` and ``x``
   dimension. This applies to
   non-gridded data like that of a polar-orbiting satellite instrument. The
   latitude dimension is typically named ``y`` and longitude named ``x``.
   This may require renaming dimensions from the file, see for the
   :meth:`xarray.DataArray.rename` method for more information and its use
   in the example below.

   If the reader should be compatible with opening remote files see
   :doc:`remote_file_support`.

 - the ``get_area_def`` method, that takes as single argument the
   :class:`~satpy.dataset.DataID` for which we want
   the area. It should return a :class:`~pyresample.geometry.AreaDefinition`
   object. For data that cannot be geolocated with an area
   definition, the pixel coordinates will be loaded using the
   ``get_dataset`` method for the resulting scene to be navigated.
   The names of the datasets to be loaded should be specified as a special
   ``coordinates`` attribute in the YAML file. For example, by specifying
   ``coordinates: [longitude_dataset, latitude_dataset]`` in the YAML, Satpy
   will call ``get_dataset`` twice, once to load the dataset named
   ``longitude_dataset`` and once to load ``latitude_dataset``. Satpy will
   then create a :class:`~pyresample.geometry.SwathDefinition` with this
   coordinate information and assign it to the dataset's
   ``.attrs['area']`` attribute.

 - Optionally, the
   ``get_bounding_box`` method can be implemented if filtering files by
   area is desirable for this data type

On top of that, two attributes need to be defined: ``start_time`` and
``end_time``, that define the start and end times of the sensing.
See the :ref:`time_metadata` section for a description of the different
times that Satpy readers typically use and what times should be used
for the ``start_time`` and ``end_time``. Note that these properties will
be assigned to the ``start_time`` and ``end_time`` metadata of any DataArrays
returned by ``get_dataset``, any existing values will be overwritten.

If you are writing a file handler for more common formats like HDF4, HDF5, or
NetCDF4 you may want to consider using the utility base classes for each:
:class:`satpy.readers.hdf4_utils.HDF4FileHandler`,
:class:`satpy.readers.hdf5_utils.HDF5FileHandler`, and
:class:`satpy.readers.netcdf_utils.NetCDF4FileHandler`. These were added as
a convenience and are not required to read these formats. In many cases using
the :func:`xarray.open_dataset` function in a custom file handler is a much
better idea.

.. note::
   Be careful about the data types of the DataArray attributes (`.attrs`) your reader is
   returning. Satpy or other tools may attempt to serialize these attributes (ex. hashing for cache keys). For example, Numpy types don't serialize into JSON and
   should therefore be cast to basic Python types (`float`, `int`, etc) before being
   assigned to the attributes.

.. note::
   Be careful about the types of the data your reader is returning.
   It is easy to let the data be coerced into double precision floats (`np.float64`). At the
   moment, satellite instruments are rarely measuring in a resolution greater
   than what can be encoded in 16 bits. As such, to preserve processing power,
   please consider carefully what data type you should scale or calibrate your
   data to.

   Single precision floats (`np.float32`) is a good compromise, as it has 23
   significant bits (mantissa) and can thus represent 16 bit integers exactly,
   as well as keeping the memory footprint half of a double precision float.

   One commonly used method in readers is :meth:`xarray.DataArray.where` (to
   mask invalid data) which can be coercing the data to `np.float64`. To ensure
   for example that integer data is coerced to `np.float32` when
   :meth:`xarray.DataArray.where` is used, you can do::

     my_float_dataarray = my_int_dataarray.where(some_condition, np.float32(np.nan))

One way of implementing a file handler is shown below:

.. code:: python

    # this is seviri_l1b_nc.py
    from satpy.readers.file_handlers import BaseFileHandler
    from pyresample.geometry import AreaDefinition

    class NCSEVIRIFileHandler(BaseFileHandler):
        def __init__(self, filename, filename_info, filetype_info):
            super(NCSEVIRIFileHandler, self).__init__(filename, filename_info, filetype_info)
            self.nc = None

        def get_dataset(self, dataset_id, dataset_info):
            if dataset_id['calibration'] != 'radiance':
                # TODO: implement calibration to reflectance or brightness temperature
                return
            if self.nc is None:
                self.nc = xr.open_dataset(self.filename,
                                          decode_cf=True,
                                          mask_and_scale=True,
                                          chunks={'num_columns_vis_ir': CHUNK_SIZE,
                                                  'num_rows_vis_ir': CHUNK_SIZE})
                self.nc = self.nc.rename({'num_columns_vir_ir': 'x', 'num_rows_vir_ir': 'y'})
            dataset = self.nc[dataset_info['nc_key']]
            dataset.attrs.update(dataset_info)
            return dataset

        def get_area_def(self, dataset_id):
            return pyresample.geometry.AreaDefinition(
                "some_area_name",
                "on-the-fly area",
                "geos",
                "+a=6378169.0 +h=35785831.0 +b=6356583.8 +lon_0=0 +proj=geos",
                3636,
                3636,
                [-5456233.41938636, -5453233.01608472, 5453233.01608472, 5456233.41938636])

    class NCSEVIRIHRVFileHandler():
      # left as an exercise to the reader :)

If you have any questions, please contact the
:ref:`Satpy developers <dev_help>`.

Auxiliary File Download
-----------------------

If your reader needs additional data files to do calibrations, corrections,
or anything else see the :doc:`aux_data` document for more information on
how to download and cache these files without including them in the Satpy
python package.
