#!/usr/bin/env bash

set -ex

# needed to build Python binding for GDAL
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
sudo mv /etc/apt/sources.list.d/pgdg-source.list* /tmp
sudo apt-get -qq remove postgis
sudo add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable
sudo apt-get update -qq
sudo apt-get install -qq bison flex python-lxml libfribidi-dev swig cmake \
librsvg2-dev colordiff postgis postgresql-9.1-postgis-2.0-scripts libpq-dev \
libpng12-dev libjpeg-dev libgif-dev libgeos-dev libgd2-xpm-dev \
libfreetype6-dev libfcgi-dev libcurl4-gnutls-dev libcairo2-dev libgdal1-dev \
libproj-dev libxml2-dev python-dev php5-dev libexempi-dev lcov lftp
pip install gdal

set +ex