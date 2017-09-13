#!/usr/bin/env bash

set -ex

# needed to build Python binding for GDAL
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
# the libhdf4-alt-dev package puts libraries in a weird place
# it says so that it doesn't conflict with netcdf...let's see what happens
sudo ln -s libmfhdfalt.so /usr/lib/libmfhdf.so
sudo ln -s libdfalt.so /usr/lib/libdf.so

set +ex