#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Fetch avhrr calibration coefficients."""
import datetime as dt
import os.path
import sys

import h5py
import urllib2

BASE_URL = "http://www.star.nesdis.noaa.gov/smcd/spb/fwu/homepage/" + \
           "AVHRR/Op_Cal_AVHRR/"

URLS = {
    "Metop-B":
    {"ch1": BASE_URL + "Metop1_AVHRR_Libya_ch1.txt",
     "ch2": BASE_URL + "Metop1_AVHRR_Libya_ch2.txt",
     "ch3a": BASE_URL + "Metop1_AVHRR_Libya_ch3a.txt"},
    "Metop-A":
    {"ch1": BASE_URL + "Metop2_AVHRR_Libya_ch1.txt",
     "ch2": BASE_URL + "Metop2_AVHRR_Libya_ch2.txt",
     "ch3a": BASE_URL + "Metop2_AVHRR_Libya_ch3a.txt"},
    "NOAA-16":
    {"ch1": BASE_URL + "N16_AVHRR_Libya_ch1.txt",
     "ch2": BASE_URL + "N16_AVHRR_Libya_ch2.txt"},
    "NOAA-17":
    {"ch1": BASE_URL + "N17_AVHRR_Libya_ch1.txt",
     "ch2": BASE_URL + "N17_AVHRR_Libya_ch2.txt",
     "ch3a": BASE_URL + "N17_AVHRR_Libya_ch3a.txt"},
    "NOAA-18":
    {"ch1": BASE_URL + "N18_AVHRR_Libya_ch1.txt",
     "ch2": BASE_URL + "N18_AVHRR_Libya_ch2.txt"},
    "NOAA-19":
    {"ch1": BASE_URL + "N19_AVHRR_Libya_ch1.txt",
     "ch2": BASE_URL + "N19_AVHRR_Libya_ch2.txt"}
}


def get_page(url):
    """Retrieve the given page."""
    return urllib2.urlopen(url).read()


def get_coeffs(page):
    """Parse coefficients from the page."""
    coeffs = {}
    coeffs['datetime'] = []
    coeffs['slope1'] = []
    coeffs['intercept1'] = []
    coeffs['slope2'] = []
    coeffs['intercept2'] = []

    slope1_idx, intercept1_idx, slope2_idx, intercept2_idx = \
        None, None, None, None

    date_idx = 0
    for row in page.lower().split('\n'):
        row = row.split()
        if len(row) == 0:
            continue
        if row[0] == 'update':
            # Get the column indices from the header line
            slope1_idx = row.index('slope_lo')
            intercept1_idx = row.index('int_lo')
            slope2_idx = row.index('slope_hi')
            intercept2_idx = row.index('int_hi')
            continue

        if slope1_idx is None:
            continue

        # In some cases the fields are connected, skip those rows
        if max([slope1_idx, intercept1_idx,
                slope2_idx, intercept2_idx]) >= len(row):
            continue

        try:
            dat = dt.datetime.strptime(row[date_idx], "%m/%d/%Y")
        except ValueError:
            continue

        coeffs['datetime'].append([dat.year, dat.month, dat.day])
        coeffs['slope1'].append(float(row[slope1_idx]))
        coeffs['intercept1'].append(float(row[intercept1_idx]))
        coeffs['slope2'].append(float(row[slope2_idx]))
        coeffs['intercept2'].append(float(row[intercept2_idx]))

    return coeffs


def get_all_coeffs():
    """Get all available calibration coefficients for the satellites."""
    coeffs = {}

    for platform in URLS:
        if platform not in coeffs:
            coeffs[platform] = {}
        for chan in URLS[platform].keys():
            url = URLS[platform][chan]
            print(url)
            page = get_page(url)
            coeffs[platform][chan] = get_coeffs(page)

    return coeffs


def save_coeffs(coeffs, out_dir=''):
    """Save calibration coefficients to HDF5 files."""
    for platform in coeffs.keys():
        fname = os.path.join(out_dir, "%s_calibration_data.h5" % platform)
        fid = h5py.File(fname, 'w')

        for chan in coeffs[platform].keys():
            fid.create_group(chan)
            fid[chan]['datetime'] = coeffs[platform][chan]['datetime']
            fid[chan]['slope1'] = coeffs[platform][chan]['slope1']
            fid[chan]['intercept1'] = coeffs[platform][chan]['intercept1']
            fid[chan]['slope2'] = coeffs[platform][chan]['slope2']
            fid[chan]['intercept2'] = coeffs[platform][chan]['intercept2']

        fid.close()
        print("Calibration coefficients saved for %s" % platform)


def main():
    """Create calibration coefficient files for AVHRR."""
    out_dir = sys.argv[1]
    coeffs = get_all_coeffs()
    save_coeffs(coeffs, out_dir=out_dir)


if __name__ == "__main__":
    main()
