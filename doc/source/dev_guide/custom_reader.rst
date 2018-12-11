.. _custom-reader:

=================================
 Adding a Custom Reader to SatPy
=================================

In order to add a reader to satpy, you will need to create two files:
 - a YAML file for describing the files to read and the datasets that are available
 - a python file implementing the actual reading of the datasets and metadata

For this tutorial, we will implement a reader for the Eumetsat NetCDF
format for SEVIRI data

The YAML file
-------------

The yaml file is composed of three sections:
 - the ``reader`` section, that provides basic parameters for the reader
 - the ``file_types`` section, which gives the patterns of the files this reader can handle
 - the ``datasets`` section, describing the datasets available from this reader

The ``reader`` section
~~~~~~~~~~~~~~~~~~~~~~

The ``reader`` section, that provides basic parameters for the reader.

The parameters to provide in this section are:
 - description: General description of the reader
 - name: this is the name of the reader, it
   should be the same as the filename (without the .yaml extension
   obviously). This is the name used interactively in satpy, so choose it
   well! A loose convention is to use ``<format>_<instrument>_<level>`` as
   a template for the name
 - sensors: the list of sensors this reader will support
 - reader: the metareader to use, in most cases the
   ``FileYAMLReader`` is a good choice.

.. code:: yaml

    reader:
      description: NetCDF4 reader for the Eumetsat MSG format
      name: nc_seviri_l1b
      sensors: [seviri]
      reader: !!python/name:satpy.readers.yaml_reader.FileYAMLReader

The ``file_types`` section
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each file type needs to provide:
 - ``file_reader``, the class that will
   handle the files for this reader, that you will implement in the
   corresponding python file (see next section)
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

The ``datasets`` section
~~~~~~~~~~~~~~~~~~~~~~~~

The datasets section describes each dataset available in the files. The
parameters provided are made available to the methods of the
implementing class.

Parameters you can define for example are:
 - name
 - sensor
 - resolution
 - wavelength
 - polarization
 - standard\_name: the name used for the
   dataset, that will be used for knowing what kind of data it is and
   handle it appropriately
 - units: the units of the data, important to get
   consistent processing across multiple platforms/instruments
 - modifiers: what modification have already been applied to the data, eg
   ``sunz_corrected``
 - file\_type
 - coordinates: this tells which datasets
   to load to navigate the current dataset
 - and any other field that is
   relevant for the reader

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



The YAML file is now ready, let's go on with the corresponding python
file.

The python file
---------------

The python files needs to implement a file
handler class for each file type that we want to read. Such a class
needs to implement a few methods:

 - the ``__init__`` method, that takes as arguments

   - the filename (string)
   - the filename info (dict) that we get by parsing the filename using the pattern defined in the yaml file
   - the filetype info that we get from the filetype definition in the yaml file

  This method can also recieve other file handler instances as parameter
  if the filetype at hand has requirements. (See the explanation in the
  YAML file filetype section above)

 - the ``get_dataset`` method, which takes as arguments

   - the dataset ID of the dataset to load
   - the dataset info that is the description of the channel in the YAML file

  This method has to return an xarray.DataArray instance if the loading is
  successful, containing the data and metadata of the loaded dataset, or
  return None if the loading was unsuccessful.

 - the ``get_area_def`` method, that takes as single argument the dataset ID for which we want
   the area. For the data that cannot be geolocated with an area
   definition, the pixel coordinates need to be loadable from
   ``get_dataset`` for the resulting scene to be navigated. That is, if the
   data cannot be geolocated with an area definition then the dataset
   section should specify ``coordinates: [longitude_dataset, latitude_dataset]``
 - Optionally, the
   ``get_bounding_box`` method can be implemented if filtering files by
   area is desirable for this data type

On top of that, two attributes need to be defined: ``start_time`` and
``end_time``, that define the start and end times of the sensing.

.. code:: python

    # this is nc_seviri_l1b.py
    class NCSEVIRIFileHandler():
        def __init__(self, filename, filename_info, filetype_info):
            super(NCSEVIRIFileHandler, self).__init__(filename, filename_info, filetype_info)
            self.nc = None

        def get_dataset(self, dataset_id, dataset_info):
            if dataset_id.calibration != 'radiance':
                # TODO: implement calibration to relfectance or brightness temperature
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
            # TODO
            pass

    class NCSEVIRIHRVFileHandler():
      # left as an exercise to the reader :)
