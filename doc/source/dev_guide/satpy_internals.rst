================================================
 Satpy internal workings: having a look under the hood
================================================

Querying and identifying datasets

DatasetQuery's
--------------

The loading of datasets in Satpy is usually done through giving the name or the wavelength of the dataset. This way, the highest, most calibrated dataset is often returned.

However, in some cases, we need more control over the loading of the datasets. The way to accomplish this is to load using queries, eg:
```
scn.load([DatasetQuery(name='channel1', resolution=400)]
```


DatasetID's
-----------

Not to be used by users

enums, put the most desirable item on top (lowest number, distance function)