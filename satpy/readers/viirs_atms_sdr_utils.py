#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2022 Satpy Developers

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Common utilities for reading VIIRS and ATMS SDR data."""


DATASET_KEYS = {'GDNBO': 'VIIRS-DNB-GEO',
                'SVDNB': 'VIIRS-DNB-SDR',
                'GITCO': 'VIIRS-IMG-GEO-TC',
                'GIMGO': 'VIIRS-IMG-GEO',
                'SVI01': 'VIIRS-I1-SDR',
                'SVI02': 'VIIRS-I2-SDR',
                'SVI03': 'VIIRS-I3-SDR',
                'SVI04': 'VIIRS-I4-SDR',
                'SVI05': 'VIIRS-I5-SDR',
                'GMTCO': 'VIIRS-MOD-GEO-TC',
                'GMODO': 'VIIRS-MOD-GEO',
                'SVM01': 'VIIRS-M1-SDR',
                'SVM02': 'VIIRS-M2-SDR',
                'SVM03': 'VIIRS-M3-SDR',
                'SVM04': 'VIIRS-M4-SDR',
                'SVM05': 'VIIRS-M5-SDR',
                'SVM06': 'VIIRS-M6-SDR',
                'SVM07': 'VIIRS-M7-SDR',
                'SVM08': 'VIIRS-M8-SDR',
                'SVM09': 'VIIRS-M9-SDR',
                'SVM10': 'VIIRS-M10-SDR',
                'SVM11': 'VIIRS-M11-SDR',
                'SVM12': 'VIIRS-M12-SDR',
                'SVM13': 'VIIRS-M13-SDR',
                'SVM14': 'VIIRS-M14-SDR',
                'SVM15': 'VIIRS-M15-SDR',
                'SVM16': 'VIIRS-M16-SDR',
                'IVCDB': 'VIIRS-DualGain-Cal-IP',
                'SATMS': 'ATMS-SDR',
                'GATMO': 'ATMS-SDR-GEO',
                'TATMS': 'ATMS-TDR'}
