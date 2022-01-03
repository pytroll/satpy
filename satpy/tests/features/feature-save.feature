# Created by a001673 at 2015-12-07
Feature: Simple and intuitive saving

  Visualization of the data is important and should be an easy one-line, like eg
  show(my_dataset). In a similar way, saving the data to disk should be simple,
  for example save(dataset, filename), with sensible defaults provided depending
  on the filename extension (eg. geotiff for .tif, netcdf for .nc). Saving
  several datasets at once would be nice to have.

  Scenario: 1-step showing dataset
    Given a dataset is available
    When the show command is called
    Then an image should pop up

  Scenario: 1-step saving dataset
    Given a dataset is available
    When the save_dataset command is called
    Then a file should be saved on disk

  Scenario: 1-step saving all datasets
    Given a bunch of datasets are available
    When the save_datasets command is called
    Then a bunch of files should be saved on disk
