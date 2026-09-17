
"""EUMETSAT base reader tests package."""

import datetime as dt
import unittest

import numpy as np

from satpy.readers.core.eum import (
    get_service_mode,
    recarray2dict,
    time_cds,
    time_cds_expanded,
    time_cds_short,
    timecds2datetime,
)
from satpy.readers.core.seviri import mpef_product_header


class TestMakeTimeCdsDictionary(unittest.TestCase):
    """Test TestMakeTimeCdsDictionary."""

    def test_fun(self):
        """Test function for TestMakeTimeCdsDictionary."""
        # time_cds_short
        tcds = {"Days": np.array(1), "Milliseconds": np.array(2)}
        expected = dt.datetime(1958, 1, 2, 0, 0, 0, 2000)
        assert timecds2datetime(tcds) == expected

        # time_cds
        tcds = {"Days": np.array(1), "Milliseconds": np.array(2), "Microseconds": np.array(3)}
        expected = dt.datetime(1958, 1, 2, 0, 0, 0, 2003)
        assert timecds2datetime(tcds) == expected

        # time_cds_expanded
        tcds = {"Days": np.array(1), "Milliseconds": np.array(2), "Microseconds": np.array(3),
                "Nanoseconds": np.array(4)}
        expected = dt.datetime(1958, 1, 2, 0, 0, 0, 2003)
        assert timecds2datetime(tcds) == expected


class TestMakeTimeCdsRecarray(unittest.TestCase):
    """Test TestMakeTimeCdsRecarray."""

    def test_fun(self):
        """Test function for TestMakeTimeCdsRecarray."""
        # time_cds_short
        tcds = np.array([(1, 2)], dtype=np.dtype(time_cds_short))
        expected = dt.datetime(1958, 1, 2, 0, 0, 0, 2000)
        assert timecds2datetime(tcds) == expected

        # time_cds
        tcds = np.array([(1, 2, 3)], dtype=np.dtype(time_cds))
        expected = dt.datetime(1958, 1, 2, 0, 0, 0, 2003)
        assert timecds2datetime(tcds) == expected

        # time_cds_expanded
        tcds = np.array([(1, 2, 3, 4)], dtype=np.dtype(time_cds_expanded))
        expected = dt.datetime(1958, 1, 2, 0, 0, 0, 2003)
        assert timecds2datetime(tcds) == expected


class TestRecarray2Dict(unittest.TestCase):
    """Test TestRecarray2Dict."""

    def test_timestamps(self):
        """Test function for TestRecarray2Dict."""
        # datatype definition

        pat_dt = np.dtype([
            ("TrueRepeatCycleStart", time_cds_expanded),
            ("PlanForwardScanEnd", time_cds_expanded),
            ("PlannedRepeatCycleEnd", time_cds_expanded)
        ])

        # planned acquisition time, add extra dimensions
        # these should be removed by recarray2dict
        pat = np.array([[[(
            (21916, 41409544, 305, 262),
            (21916, 42160340, 659, 856),
            (21916, 42309417, 918, 443))]]], dtype=pat_dt)

        expected = {
            "TrueRepeatCycleStart": dt.datetime(2018, 1, 2, 11, 30, 9, 544305),
            "PlanForwardScanEnd": dt.datetime(2018, 1, 2, 11, 42, 40, 340660),
            "PlannedRepeatCycleEnd": dt.datetime(2018, 1, 2, 11, 45, 9, 417918)
        }

        assert recarray2dict(pat) == expected

    def test_mpef_product_header(self):
        """Test function for TestRecarray2Dict and mpef product header."""
        names = ["ImageLocation", "GsicsCalMode", "GsicsCalValidity",
                 "Padding", "OffsetToData", "Padding2"]
        mpef_header = np.dtype([(name, mpef_product_header.fields[name][0])
                                for name in names])
        mph_struct = np.array([("OPE", True, False, "XX", 1000, "12345678")], dtype=mpef_header)
        test_mph = {"ImageLocation": "OPE",
                    "GsicsCalMode": True,
                    "GsicsCalValidity": False,
                    "Padding": "XX",
                    "OffsetToData": 1000,
                    "Padding2": "12345678"
                    }
        assert recarray2dict(mph_struct) == test_mph


class TestGetServiceMode(unittest.TestCase):
    """Test the get_service_mode function."""

    def test_get_seviri_service_mode_fes(self):
        """Test fetching of SEVIRI service mode information for FES."""
        ssp_lon = 0.0
        name = "fes"
        desc = "Full Earth Scanning service"
        res = get_service_mode("seviri", ssp_lon)
        assert res["service_name"] == name
        assert res["service_desc"] == desc

    def test_get_seviri_service_mode_rss(self):
        """Test fetching of SEVIRI service mode information for RSS."""
        ssp_lon = 9.5
        name = "rss"
        desc = "Rapid Scanning Service"
        res = get_service_mode("seviri", ssp_lon)
        assert res["service_name"] == name
        assert res["service_desc"] == desc

    def test_get_seviri_service_mode_iodc_E0415(self):
        """Test fetching of SEVIRI service mode information for IODC at 41.5 degrees East."""
        ssp_lon = 41.5
        name = "iodc"
        desc = "Indian Ocean Data Coverage service"
        res = get_service_mode("seviri", ssp_lon)
        assert res["service_name"] == name
        assert res["service_desc"] == desc

    def test_get_seviri_service_mode_iodc_E0455(self):
        """Test fetching of SEVIRI service mode information for IODC at 45.5 degrees East."""
        ssp_lon = 45.5
        name = "iodc"
        desc = "Indian Ocean Data Coverage service"
        res = get_service_mode("seviri", ssp_lon)
        assert res["service_name"] == name
        assert res["service_desc"] == desc

    def test_get_fci_service_mode_fdss(self):
        """Test fetching of FCI service mode information for FDSS."""
        ssp_lon = 0.0
        name = "fdss"
        desc = "Full Disk Scanning Service"
        res = get_service_mode("fci", ssp_lon)
        assert res["service_name"] == name
        assert res["service_desc"] == desc

    def test_get_fci_service_mode_rss(self):
        """Test fetching of FCI service mode information for RSS."""
        ssp_lon = 9.5
        name = "rss"
        desc = "Rapid Scanning Service"
        res = get_service_mode("fci", ssp_lon)
        assert res["service_name"] == name
        assert res["service_desc"] == desc

    def test_get_unknown_lon_service_mode(self):
        """Test fetching of service mode information for unknown input longitude."""
        ssp_lon = 13
        name = "unknown"
        desc = "unknown"
        res = get_service_mode("fci", ssp_lon)
        assert res["service_name"] == name
        assert res["service_desc"] == desc

    def test_get_unknown_instrument_service_mode(self):
        """Test fetching of service mode information for unknown input instrument."""
        ssp_lon = 0
        name = "unknown"
        desc = "unknown"
        res = get_service_mode("test", ssp_lon)
        assert res["service_name"] == name
        assert res["service_desc"] == desc
