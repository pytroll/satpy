"""Fixtures with fake pre-GVAR AREA files for the reader tests."""

import pytest

from satpy.tests.reader_tests.goes_pregvar.fake_area_files import (
    GOES3_IR,
    GOES3_VIS,
    GOES7_IR,
    IR_WINDOW,
    VIS_WINDOW,
    FakeArea,
    aaa_spec,
    mode_a_data,
    mode_aa_spec,
    write_area_file,
)


@pytest.fixture
def mode_a_ir_file(tmp_path):
    """Mode A IR file where the counts cover both branches of the formula."""
    spec = FakeArea(GOES3_IR, mode_a_data([0, 100, 175, 176, 200, 255]), **IR_WINDOW)
    return write_area_file(tmp_path / "goes03.1979.234.1515.AREA_IR", spec)


@pytest.fixture
def mode_a_vis_file(tmp_path):
    """Mode A VIS file; counts are 6-bit values shifted left by two."""
    spec = FakeArea(GOES3_VIS, mode_a_data([0, 4, 128, 252, 252, 255]), cal_type="RAW", **VIS_WINDOW)
    return write_area_file(tmp_path / "goes03.1979.234.1515.AREA_VIS", spec)


@pytest.fixture
def mode_aa_file(tmp_path):
    """Mode AA file with band 8 calibrated on every line."""
    return write_area_file(tmp_path / "goes05.1983.229.2231-2.AREA_IR", mode_aa_spec())


@pytest.fixture
def goes7_aaa_file(tmp_path):
    """GOES-7 AAA file with channels 8, 7 and 10 in changing order."""
    return write_area_file(tmp_path / "goes07.1992.236.1301.AREA_IR", aaa_spec(GOES7_IR))
