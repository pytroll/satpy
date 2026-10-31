"""Tests for the pre-GVAR AREA file parser (``goes_pregvar_base``)."""

import logging

import numpy as np
import pytest

from satpy.tests.utils import skip_numba_unstable_if_missing

try:
    from satpy.readers.goes_pregvar.goes_pregvar_base import AreaFile
except ImportError as err:
    if skip_numba_unstable_if_missing():
        pytest.skip(f"Numba is not compatible with unstable NumPy: {err!s}", allow_module_level=True)
    raise

from satpy.tests.reader_tests.goes_pregvar.fake_area_files import (
    CAL_WORDS,
    DIR_WORDS,
    GOES3_IR,
    GOES7_IR,
    IR_WINDOW,
    NAV_WORDS,
    SHAPE,
    W_CAL_OFFSET,
    W_NAV_OFFSET,
    FakeArea,
    aaa_spec,
    mode_a_data,
    mode_aa_spec,
    patch_dir_word,
    write_area_file,
)

# NOTE: the ``mode_a_ir_file``, ``mode_a_vis_file``, ``mode_aa_file`` and ``goes7_aaa_file``
# fixtures come from the ``conftest.py`` next to this module.


class TestMalformedFiles:
    """Files with missing or broken blocks."""

    def test_undetectable_byte_order_warns(self, tmp_path, caplog):
        """Fall back to big endian, with a warning, when neither byte order looks right."""
        spec = FakeArea(99, mode_a_data([100]), **IR_WINDOW)
        with caplog.at_level(logging.WARNING):
            area = AreaFile(write_area_file(tmp_path / "unknown.AREA_IR", spec))
        assert "could not confidently detect byte order" in caplog.text
        assert area.dir["satellite"] is None

    def test_truncated_directory(self, tmp_path):
        """Refuse a file that ends inside the directory."""
        path = tmp_path / "short.AREA_IR"
        path.write_bytes(bytes(100))
        with pytest.raises(ValueError, match="truncated area directory"):
            AreaFile(str(path))

    def test_no_nav_block(self, mode_a_ir_file):
        """Without a nav block there is nothing to earth-locate with."""
        patch_dir_word(mode_a_ir_file, W_NAV_OFFSET, 0)
        area = AreaFile(mode_a_ir_file)
        assert area.nav is None
        with pytest.raises(ValueError, match="no GOES-type nav block"):
            area.get_lonlats()
        with pytest.raises(ValueError, match="no GOES-type nav block"):
            area.get_nadir_sampling()

    def test_nav_block_of_another_type(self, mode_a_ir_file, caplog):
        """Warn about a nav block that is not of the GOES type and do not use it."""
        with open(mode_a_ir_file, "r+b") as f:
            f.seek(DIR_WORDS * 4)
            f.write(b"XXXX")
        with caplog.at_level(logging.WARNING):
            area = AreaFile(mode_a_ir_file)
        assert "does not start with b'GOES'" in caplog.text
        with pytest.raises(ValueError, match="no GOES-type nav block"):
            area.get_lonlats()

    def test_short_aaa_calibration_block(self, tmp_path):
        """An AAA file with a calibration block that is too short has no radiance calibration."""
        path = write_area_file(tmp_path / "goes07.1992.236.1301.AREA_IR", aaa_spec(GOES7_IR))
        data_offset = (DIR_WORDS + NAV_WORDS + CAL_WORDS) * 4
        patch_dir_word(path, W_CAL_OFFSET, data_offset - 40)
        area = AreaFile(path)
        assert area.cal is None
        with pytest.raises(ValueError, match="calibration block"):
            area.read_ir_radiance(8)

    def test_short_mode_aa_calibration_region(self, tmp_path, caplog):
        """A Mode AA file without room for the calibration entries cannot be calibrated."""
        spec = mode_aa_spec()
        spec.cal_len = 50
        path = write_area_file(tmp_path / "goes05.1983.229.2231-2.AREA_IR", spec)
        with caplog.at_level(logging.WARNING):
            area = AreaFile(path)
        assert "Mode AA calibration region is only 50 bytes" in caplog.text
        with pytest.raises(ValueError, match="no Mode AA calibration"):
            area.read_ir_temperature_mode_aa(8)

    def test_mode_aa_without_band_list(self, tmp_path, mode_aa_file, caplog):
        """A single-band Mode AA file with a short prefix is read without the band list."""
        spec = mode_aa_spec()
        spec.prefixes = spec.prefixes[:, :632]
        path = write_area_file(tmp_path / "goes05.1983.229.2231-2.AREA_IR", spec)
        with caplog.at_level(logging.WARNING):
            area = AreaFile(path)
        assert "too short for the documented AAA line-prefix layout" in caplog.text
        assert area.band_list is None
        expected = AreaFile(mode_aa_file).read_ir_temperature_mode_aa(8)
        np.testing.assert_allclose(area.read_ir_temperature_mode_aa(8), expected)

    def test_band_list_shorter_than_the_number_of_bands(self, tmp_path, caplog):
        """Without a usable band list the channels of an AAA file cannot be told apart."""
        spec = aaa_spec(GOES7_IR)
        spec.prefixes = spec.prefixes[:, :633]
        path = write_area_file(tmp_path / "goes07.1992.236.1301.AREA_IR", spec)
        with caplog.at_level(logging.WARNING):
            area = AreaFile(path)
        assert "band list region is only 1 bytes" in caplog.text
        with pytest.raises(ValueError, match="band list"):
            area.read_channel(8)

    def test_unsupported_element_size(self, tmp_path):
        """Only one and two byte elements are supported."""
        spec = FakeArea(GOES3_IR, mode_a_data([100]), bytes_per_elem=4, **IR_WINDOW)
        area = AreaFile(write_area_file(tmp_path / "goes03.1979.234.1515.AREA_IR", spec))
        with pytest.raises(NotImplementedError, match="bytes/element"):
            area.read_data()


class TestAreaFileRequests:
    """Requests that the file or the calibration cannot satisfy."""

    def test_single_band_read_of_a_multiband_file(self, goes7_aaa_file):
        """Multiband files are read per channel."""
        with pytest.raises(ValueError, match="single-band"):
            AreaFile(goes7_aaa_file).read_data()

    def test_channel_without_calibration(self, goes7_aaa_file):
        """Refuse channels that the calibration block does not describe."""
        with pytest.raises(ValueError, match="not present"):
            AreaFile(goes7_aaa_file).read_ir_radiance(39)

    def test_channel_without_planck_constants(self, goes7_aaa_file):
        """Refuse channels that have no verified Planck constants."""
        with pytest.raises(NotImplementedError, match="No verified Planck"):
            AreaFile(goes7_aaa_file).read_ir_temperature_aaa(14)

    def test_mode_aa_request_for_another_format(self, mode_a_ir_file):
        """Mode AA temperatures only exist for Mode AA files."""
        with pytest.raises(ValueError, match="not a Mode AA"):
            AreaFile(mode_a_ir_file).read_ir_temperature_mode_aa(8)

    def test_mode_aa_band_out_of_range(self, mode_aa_file):
        """The visible channel is not covered by the Mode AA formula."""
        with pytest.raises(ValueError, match="1-12"):
            AreaFile(mode_aa_file).read_ir_temperature_mode_aa(13)

    def test_mode_a_temperature_of_a_visible_file(self, mode_a_vis_file):
        """The IR formula does not apply to visible data."""
        with pytest.raises(ValueError, match="not IR"):
            AreaFile(mode_a_vis_file).read_ir_temperature()

    def test_mode_a_temperature_of_a_multiband_file(self, goes7_aaa_file):
        """The single-band formula does not apply to multiband files."""
        with pytest.raises(NotImplementedError, match="multiband"):
            AreaFile(goes7_aaa_file).read_ir_temperature()

    def test_mode_a_temperature_of_wide_counts(self, tmp_path):
        """The formula assumes 8-bit counts."""
        data = np.full((SHAPE[0], SHAPE[1], 1), 300)
        spec = FakeArea(GOES3_IR, data, bytes_per_elem=2, **IR_WINDOW)
        area = AreaFile(write_area_file(tmp_path / "goes03.1979.234.1515.AREA_IR", spec))
        with pytest.raises(ValueError, match="8 bits"):
            area.read_ir_temperature()

    def test_visible_counts_of_an_infrared_file(self, mode_a_ir_file):
        """Visible counts only exist for visible files."""
        with pytest.raises(ValueError, match="not VIS"):
            AreaFile(mode_a_ir_file).read_vis_counts()

    def test_unsupported_visible_calibration(self, mode_a_vis_file):
        """Only counts and unnormalized reflectance are available."""
        with pytest.raises(ValueError, match="Unsupported VIS calibration"):
            AreaFile(mode_a_vis_file).read_vis_counts("bogus")
