======================================================
 Satpy internal workings: having a look under the hood
======================================================

Querying and identifying data arrays
====================================

DataQuery
---------

The loading of data in Satpy is usually done through giving the name or the wavelength of the data arrays we are interested
in. This way, the highest, most calibrated data arrays is often returned.

However, in some cases, we need more control over the loading of the data arrays. The way to accomplish this is to load
data arrays using queries, eg::

  scn.load([DataQuery(name='channel1', resolution=400)]

Here a data array with name `channel1` and of resolution `400` will be loaded if available.

Note that None is not a valid value, and keys having a value set to None will simply be ignored.

If one wants to use wildcards to query data, just provide `'*'`, eg::

  scn.load([DataQuery(name='channel1', resolution=400, calibration='*')]

Alternatively, one can provide a list as parameter to query data, like this::

  scn.load([DataQuery(name='channel1', resolution=[400, 800])]



DataID
------

Satpy stores loaded data arrays in a special dictionary (`DatasetDict`) inside scene objects.
In order to identify each data array uniquely, Satpy is assigning an ID to each data array, which is then used as the key in
the scene object. These IDs are of type `DataID` and are immutable. They are not supposed to be used by regular users and should only be
created in special circumstances. Satpy should take care of creating and assigning these automatically. They are also stored in the
`attrs` of each data array as `_satpy_id`.

Default and custom metadata keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One thing however that the user has control over is which metadata keys are relevant to which datasets. Satpy provides two default sets
of metadata key (or ID keys), one for regular imager bands, and the other for composites.
The first one contains: name, wavelength, resolution, calibration, modifiers.
The second one contains: name, resolution.

As an example here is the definition of the first one in yaml:

  .. code-block:: yaml

    data_identification_keys:
      name:
        required: true
      wavelength:
        type: !!python/name:satpy.dataset.WavelengthRange
      resolution:
      calibration:
        enum:
            - reflectance
            - brightness_temperature
            - radiance
            - counts
        transitive: true
      modifiers:
        required: true
        default: []
        type: !!python/name:satpy.dataset.ModifierTuple

To create a new set, the user can provide indications in the relevant yaml file.
It has to be provided in header of the reader configuration file, under the `reader`
section, as `data_identification_keys`. Each key under this is the name of relevant
metadata key that will used to find relevant information in the attributes of the data
arrays. Under each of this, a few options are available:

 - `required`: if the item is required, False by default
 - `type`: the type to use. More on this further down.
 - `enum`: if the item has to be limited to a finite number of options, an enum can be used.
   Be sure to place the options in the order of preference, with the most desirable option on top.
 - `default`: the default value to assign to the item if nothing (or None) is provided. If this
   option isn't provided, the key will simply be omitted if it is not present in the attrs or if it
   is None. It will be passed to the type's `convert` method if available.
 - `transitive`: whether the key is to be passed when looking for dependencies of composites/modifiers.
   Here for example, a composite that has in a given calibration type will pass this calibration
   type requirement to its dependencies.


If the definition of the metadata keys need to be done in python rather than in a yaml file, it will
be a dictionary very similar to the yaml code. Here is the same example as above in python:

  .. code-block:: python

    from satpy.dataset import WavelengthRange, ModifierTuple

    id_keys_config = {'name': {
                          'required': True,
                      },
                      'wavelength': {
                          'type': WavelengthRange,
                      },
                      'resolution': None,
                      'calibration': {
                          'enum': [
                              'reflectance',
                              'brightness_temperature',
                              'radiance',
                              'counts'
                              ],
                          'transitive': True,
                      },
                      'modifiers': {
                          'required': True,
                          'default': ModifierTuple(),
                          'type': ModifierTuple,
                      },
                      }

Types
~~~~~
Types are classes that implement a type to be used as value for metadata in the `DataID`. They have
to implement a few methods:

 - a `convert` class method that returns it's argument as an instance of the class
 - `__hash__`, `__eq__` and `__ne__` methods
 - a `distance` method the tells how "far" an instance of this class is from it's argument.

An example of such a class is the :class:`WavelengthRange <satpy.dataset.WavelengthRange>` class.
Through its implementation, it allows us to use the wavelength in a query to find out which of the
DataID in a list which has its central wavelength closest to that query for example.


DataID and DataQuery interactions
=================================

Different DataIDs and DataQuerys can have different metadata items defined. As such
we define equality between different instances of these classes, and across the classes
as equality between the sorted key/value pairs shared between the instances.
If a DataQuery has one or more values set to `'*'`, the corresponding key/value pair will be omitted from the comparison.
Instances sharing no keys will no be equal.


Breaking changes from DatasetIDs
================================

 - The way to access values from the DataID and DataQuery is through getitem: `my_dataid['resolution']`
 - For checking if a dataset is loaded, use `'mydataset' in scene`, as `'mydataset' in scene.keys()` will always return `False`:
   the `DatasetDict` instance only supports `DataID` as key type.

Creating DataID for tests
=========================

Sometimes, it is useful to create `DataID` instances for testing purposes. For these cases, the `satpy.tests.utils` module
now has a `make_dsid` function that can be used just for this::

  from satpy.tests.utils import make_dataid
  did = make_dataid(name='camembert', modifiers=('runny',))
