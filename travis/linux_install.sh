#!/usr/bin/env bash

set -ex

# needed to build Python binding for GDAL
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal

set +ex