"""GOES pre-GVAR AREA base reader.

Reference: `SSEC New Area Format`_.

.. _SSEC New Area Format:
    https://www.ssec.wisc.edu/mcidas/doc/misc_doc/area2.html
"""

import functools
import logging

import numpy as np

from satpy.readers.goes_pregvar.goes_pregvar_navigation import GOESNav

logger = logging.getLogger(__name__)

DIR_WORDS = 64
DIR_BYTES = DIR_WORDS * 4

# Area Directory word indices (1-based in the spec -> 0-based here)
W = {
    "valid_code": 0,  # W1
    "area_format": 1,  # W2
    "sensor_source": 2,  # W3
    "nom_start_date": 3,  # W4  YYDDD
    "nom_start_time": 4,  # W5  HHMMSS
    "ul_line": 5,  # W6  image line of area (0,0)
    "ul_element": 6,  # W7  image element of area (0,0)
    "num_lines": 8,  # W9
    "num_elements": 9,  # W10
    "bytes_per_elem": 10,  # W11
    "line_res": 11,  # W12
    "elem_res": 12,  # W13
    "num_bands": 13,  # W14
    "prefix_len": 14,  # W15
    "band_map": 18,  # W19
    "data_offset": 33,  # W34
    "nav_offset": 34,  # W35
    "valid_code2": 35,  # W36
    "actual_start_date": 45,  # W46
    "actual_start_time": 46,  # W47
    "doc_len": 48,  # W49
    "cal_len": 49,  # W50
    "level_len": 50,  # W51
    "source_type": 51,  # W52  ASCII 4 chars, e.g. "VISR"
    "cal_type": 52,  # W53  ASCII 4 chars, e.g. "RAW " / "BRIT"
    "cal_offset": 62,  # W63
}

# Appendix A sensor source numbers relevant to the pre-GVAR VISSR era
SENSOR_SOURCE = {
    16: ("SMS-1", "VIS"), 17: ("SMS-1", "IR"),
    18: ("SMS-2", "VIS"), 19: ("SMS-2", "IR"),
    20: ("GOES-1", "VIS"), 21: ("GOES-1", "IR"),
    22: ("GOES-2", "VIS"), 23: ("GOES-2", "IR"),
    24: ("GOES-3", "VIS"), 25: ("GOES-3", "IR"),
    26: ("GOES-4", "VIS"), 27: ("GOES-4", "IR"),  # VAS
    28: ("GOES-5", "VIS"), 29: ("GOES-5", "IR"),  # VAS
    30: ("GOES-6", "VIS"), 31: ("GOES-6", "IR"),
    32: ("GOES-7", "VIS"), 33: ("GOES-7", "IR"),
}


def _ascii(raw):
    return raw.decode("ascii", errors="replace").strip("\x00").strip()


def _yyddd_to_iso(val):
    """W4/W46-style nominal/actual date: YYDDD -> (year, day-of-year)."""
    val = int(val)
    yy, ddd = divmod(val, 1000)
    year = 1900 + yy if yy >= 70 else 2000 + yy
    return year, ddd


def _directory_plausibility(words):
    """Score how plausible a directory read in one byte order is (higher is better)."""
    sensor = int(words[W["sensor_source"]])
    nlines = int(words[W["num_lines"]])
    nelem = int(words[W["num_elements"]])
    score = 10 if sensor in SENSOR_SOURCE else 0
    return score + int(0 < nlines < 200000) + int(0 < nelem < 200000)


def _nav_word(nav_words, i):
    """Return the 1-based word ``i`` of the nav block, or None if the block is too short."""
    idx = i - 1
    return int(nav_words[idx]) if idx < nav_words.size else None


def _scaled(value, divisor):
    """Divide a nav word by ``divisor``, passing a missing word (None) through."""
    return None if value is None else value / divisor


class AreaFile:
    """Lazy-ish reader: parses directory + nav eagerly, data on demand."""

    def __init__(self, path):
        """Read an AREA file and parse its directory, navigation and calibration.

        Args:
            path: Path to the AREA file.
        """
        self.path = path
        with open(path, "rb") as f:
            self._raw = f.read()
        self._actual_num_lines_cache = None
        self._parse_directory()
        self._parse_nav()
        self._parse_cal_block()
        self._parse_band_list()
        self._parse_mode_aa_calibration()

    def _actual_num_lines(self):
        """Return the number of lines actually present in the DATA block.

        This can be less than the directory's claimed num_lines (W9) if the
        file was truncated (seen in practice on at least one 1980s-era GOES-6
        Mode AA VIS file). It is computed once, cached, and used as the single
        source of truth for the full-grid line count, both in read_data()'s
        pixel array and in get_lonlats()'s default grid. Using the claimed
        count in one place and the actual count in the other gives pixel data
        and navigation/earth-mask arrays with mismatched shapes.
        """
        if self._actual_num_lines_cache is None:
            d = self.dir
            nelem = d["num_elements"]
            nbytes = d["bytes_per_elem"]
            nbands = d["num_bands"]
            prefix_len = d["prefix_len"]
            line_data_len = nelem * nbytes * nbands
            line_len = prefix_len + line_data_len
            start = d["data_offset"]
            nlines = d["num_lines"]
            block_len = len(self._raw[start:start + nlines * line_len])
            actual = block_len // line_len
            if actual != nlines:
                logger.warning("%s: directory claims %d lines but data block only has %d; using %d.",
                               self.path, nlines, actual, actual)
            self._actual_num_lines_cache = actual
        return self._actual_num_lines_cache

    def _detect_endianness(self):
        """Detect the byte order of the file.

        The 1997 spec says AREA files are always big-endian, but real files
        floating around today are often little-endian, presumably from having
        passed through x86 tooling at some point since. Rather than trust the
        spec, both interpretations are checked against the known sensor_source
        table and whichever produces a valid, plausible directory wins.
        """
        raw = self._raw[:DIR_BYTES]
        candidates = {">": np.frombuffer(raw, dtype=">i4"),
                      "<": np.frombuffer(raw, dtype="<i4")}
        scored = {order: _directory_plausibility(words) for order, words in candidates.items()}
        best = max(scored, key=scored.get)
        if scored[best] < 10:
            logger.warning("%s: could not confidently detect byte order (scores: %s); defaulting to '>' "
                           "(big-endian) per the written spec, but this file may parse incorrectly.",
                           self.path, scored)
            best = ">"
        self._endian = best
        return candidates[best]

    def _parse_directory(self):
        words = self._detect_endianness()
        if words.size != DIR_WORDS:
            raise ValueError(
                f"{self.path}: truncated area directory "
                f"({words.size * 4} bytes, expected {DIR_BYTES})"
            )
        self._dir_words = words

        d = {}
        d["sensor_source"] = int(words[W["sensor_source"]])
        d["num_lines"] = int(words[W["num_lines"]])
        d["num_elements"] = int(words[W["num_elements"]])
        d["bytes_per_elem"] = int(words[W["bytes_per_elem"]])
        d["line_res"] = int(words[W["line_res"]])
        d["elem_res"] = int(words[W["elem_res"]])
        d["num_bands"] = int(words[W["num_bands"]])
        d["prefix_len"] = int(words[W["prefix_len"]])
        d["data_offset"] = int(words[W["data_offset"]])
        d["nav_offset"] = int(words[W["nav_offset"]])
        d["cal_offset"] = int(words[W["cal_offset"]])
        d["ul_line"] = int(words[W["ul_line"]])
        d["ul_element"] = int(words[W["ul_element"]])
        d["band_filter_map"] = int(words[W["band_map"]])  # W19
        d["valcode"] = int(words[W["valid_code2"]]) != 0   # W36: is a 4B validity code present
        d["doc_len"] = int(words[W["doc_len"]])    # W49: prefix documentation length
        d["cal_len"] = int(words[W["cal_len"]])    # W50: prefix calibration length
        d["level_len"] = int(words[W["level_len"]])  # W51: prefix band-list/"level" length

        # source_type / cal_type are 4 ASCII chars packed into one 4-byte
        # word each -- reinterpret those two words as raw bytes.
        raw_bytes = self._raw[:DIR_BYTES]
        st_off = W["source_type"] * 4
        ct_off = W["cal_type"] * 4
        d["source_type"] = _ascii(raw_bytes[st_off:st_off + 4])
        d["cal_type"] = _ascii(raw_bytes[ct_off:ct_off + 4])

        nom_year, nom_doy = _yyddd_to_iso(words[W["nom_start_date"]])
        d["nominal_year"] = nom_year
        d["nominal_doy"] = nom_doy
        d["nominal_time_hhmmss"] = int(words[W["nom_start_time"]])

        sat_name, band = SENSOR_SOURCE.get(d["sensor_source"], (None, None))
        d["satellite"] = sat_name
        d["band"] = band  # "VIS" or "IR"

        self.dir = d

    def _parse_nav(self):
        """GOES-7-documented NAV block layout."""
        off = self.dir["nav_offset"]
        if off == 0:
            self.nav = None
            return

        # Nav block runs until CAL block (W63) or DATA block (W34),
        # whichever comes first / is nonzero.
        end = self.dir["cal_offset"] or self.dir["data_offset"]
        dtype = ">i4" if self._endian == ">" else "<i4"
        nav_bytes = self._raw[off:end]
        nav_bytes = nav_bytes[:len(nav_bytes) - (len(nav_bytes) % 4)]
        nav_words = np.frombuffer(nav_bytes, dtype=dtype)

        if nav_words.size < 4 or self._raw[off:off + 4] != b"GOES":
            # Not fatal -- some deliveries may differ -- but flag it since
            # silent bad navigation is worse than a loud warning.
            logger.warning("%s: NAV block does not start with b'GOES' (got %r); navigation "
                           "parameters below may be wrong for this file.",
                           self.path, self._raw[off:off + 4])

        w = functools.partial(_nav_word, nav_words)

        self._nav_words = nav_words  # raw signed-32 words, kept for GOESNav

        self.nav = {
            "nav_type": self._raw[off:off + 4],
            "sat_year_doy": w(2),           # SSYYDDD
            "nominal_start_time": w(3),     # HHMMSS
            "epoch_date": w(5),             # YYMMDD
            "epoch_time": w(6),             # HHMMSS
            "semimajor_axis_km": _scaled(w(7), 100.0),
            "eccentricity": _scaled(w(8), 100000.0),
            "inclination_deg": _scaled(w(9), 1000.0),
            "mean_anomaly_deg": _scaled(w(10), 1000.0),
            "arg_perigee_deg": _scaled(w(11), 1000.0),
            "right_asc_asc_node_deg": _scaled(w(12), 1000.0),
            "declination_ddmmss": w(13),
            "right_ascension_ddmmss": w(14),
            "picture_center_line": w(15),
            "spin_period_us": w(16),
            "sweep_angle_line_ddmmss": w(17),
            "num_scan_lines_raw": w(18),    # NNLLLLL
            "sweep_angle_elem_ddmmss": w(19),
            "num_elements_per_line": w(20),
            "pitch_ddmmss": w(21),
            "yaw_ddmmss": w(22),
            "roll_ddmmss": w(23),
        }

    def _parse_cal_block(self):
        """GOES-7 AAA calibration block.

        Only meaningful for source_type in ('AAA', 'VAS') -- 'AAA' is
        Mode AAA (1987-1994); 'VAS' refers to Mode AA (1980-1986).
        For Mode A/'VISR' files there's no VAS radiance equation
        and this is left unparsed (self.cal is None).
        """
        self.cal = None
        if self.dir.get("source_type") not in ("AAA", "VAS"):
            return
        off = self.dir["cal_offset"]
        end = self.dir["data_offset"]
        dtype = ">i4" if self._endian == ">" else "<i4"
        cal_bytes = self._raw[off:end]
        cal_bytes = cal_bytes[: len(cal_bytes) - (len(cal_bytes) % 4)]
        words = np.frombuffer(cal_bytes, dtype=dtype)
        if words.size < 117:
            return  # too short to be a real AAA cal block

        # The 117-word minimum above covers every index used below.
        # channel (1-based) -> (AB1, AB2); word4 == idx3
        iab = {chan: (int(words[3 + (chan - 1) * 2]), int(words[4 + (chan - 1) * 2])) for chan in range(1, 39)}
        # channel (1-based) -> IFAB scale-factor word; word80 == idx79
        ifab = {chan: int(words[79 + (chan - 1)]) for chan in range(1, 39)}

        self.cal = {"sensor_source": int(words[0]), "iab": iab, "ifab": ifab}

    # -----------------------------------------------------------------
    # Mode AA calibration. Courtesy of SSEC via NOAA CLASS.

    # channel_code (1-38, 1-based) -> IR band (1-12). Index 0 = code 1.
    _MODE_AA_CHANNEL_BANDS = tuple(range(1, 13)) * 2 + (3, 4, 5, 7, 8, 9, 10) * 2
    # deltaF prefix code (0-15) -> actual deltaF (-5..+5).
    _MODE_AA_DELTA_F = (0, -1, -2, -3, -4, -5, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0)
    _MODE_AA_COEFFS = {
        1:  (7, 3723, 976.5, -0.0003905001, 1.0),
        2:  (7, 3910, 992.5, 0.002225, 1.0),
        3:  (7, 4097, 1008, 0.00236, 1.0),
        4:  (7, 4352, 1029, 0.004170999, 0.9999),
        5:  (7, 5049, 1081, 0.004202001, 0.9999),
        6:  (11, 128100, 3176, 0.06761003, 0.9998),
        7:  (7, 5874, 1137, 0.004214, 0.9999),
        8:  (7, 8512, 1286, 0.3212, 0.9974),
        9:  (8, 30940, 1978, 0.06927001, 0.9997),
        10: (8, 38880, 2134, 0.7513, 0.9973),
        11: (11, 136000, 3240, 0.05738, 0.9999),
        12: (13, 195900, 3659, 0.4237, 0.9991),
    }

    def _mode_aa_code_to_band(self, codes):
        """Map channel codes (1-38) to IR bands (1-12), vectorized.

        Uses _MODE_AA_CHANNEL_BANDS. Codes outside 1-38 (e.g. the 0 used for
        unused slots) map to 0, which never matches a real band.
        """
        band_lookup = np.array((0,) + self._MODE_AA_CHANNEL_BANDS)  # shift so index==code
        out = np.zeros_like(codes)
        valid = (codes >= 1) & (codes <= 38)
        out[valid] = band_lookup[codes[valid]]
        return out

    def _parse_mode_aa_calibration(self):
        """Parse Mode AA's per-line calibration entries."""
        self.mode_aa_cal = None
        if self.dir.get("source_type") != "VAS":
            return
        d = self.dir
        prefix_len = d["prefix_len"]
        cal_start = (4 if d["valcode"] else 0) + d["doc_len"]
        cal_len = d["cal_len"]
        n_entries, entry_bytes, header_bytes = 13, 8, 12
        needed = header_bytes + n_entries * entry_bytes  # 116
        if cal_len < needed:
            logger.warning("%s: Mode AA calibration region is only %d bytes, need >= %d for the "
                           "documented 13-entry layout -- Mode AA calibration unavailable for this file.",
                           self.path, cal_len, needed)
            return

        nelem = d["num_elements"]
        nbytes = d["bytes_per_elem"]
        nbands = d["num_bands"]
        line_data_len = nelem * nbytes * nbands
        line_len = prefix_len + line_data_len
        data_offset = d["data_offset"]
        dtype16s = ">i2" if self._endian == ">" else "<i2"

        actual_lines = self._actual_num_lines()
        block = self._raw[data_offset:data_offset + actual_lines * line_len]
        arr = np.frombuffer(block[:actual_lines * line_len], dtype=np.uint8)
        arr = arr.reshape(actual_lines, line_len)

        entries_start = cal_start + header_bytes
        entries_bytes = arr[:, entries_start:entries_start + n_entries * entry_bytes]
        entries = entries_bytes.copy().view(dtype16s).reshape(actual_lines, n_entries, 4)
        self.mode_aa_cal = {
            "channel": entries[:, :, 0],
            "spin": entries[:, :, 1],       # unused by the temperature formula
            "deltaf_code": entries[:, :, 2],
            "ysubz": entries[:, :, 3],
        }

    def _check_mode_aa_request(self, band):
        """Raise ValueError unless this is a Mode AA file with calibration data and ``band`` is IR 1-12."""
        if self.dir.get("source_type") != "VAS":
            raise ValueError(f"{self.path}: not a Mode AA ('VAS') file.")
        if not (1 <= band <= 12):
            raise ValueError(
                f"Mode AA IR band must be 1-12 (got {band}); band 13 is "
                f"visible and not covered by this formula."
            )
        if self.mode_aa_cal is None:
            raise ValueError(
                f"{self.path}: no Mode AA calibration data available. "
            )

    def _mode_aa_line_calibration(self, band):
        """Return per-line ``(cal_present, delta_f, ysubz)`` for IR ``band``.

        ``cal_present`` marks the lines whose calibration entries include the
        band; ``delta_f`` and ``ysubz`` are the entry's values on those lines.
        """
        cal = self.mode_aa_cal
        match = self._mode_aa_code_to_band(cal["channel"]) == band  # (nlines, 13)
        cal_present = match.any(axis=1)
        slot = np.where(cal_present, match.argmax(axis=1), 0)
        lines_idx = np.arange(match.shape[0])
        deltaf_code = cal["deltaf_code"][lines_idx, slot]
        ysubz = cal["ysubz"][lines_idx, slot].astype(np.float64)

        deltaf_table = np.array(self._MODE_AA_DELTA_F)
        valid_code = (deltaf_code >= 0) & (deltaf_code < len(deltaf_table))
        delta_f = np.where(valid_code, deltaf_table[np.clip(deltaf_code, 0, len(deltaf_table) - 1)], 0)
        return cal_present, delta_f, ysubz

    def _mode_aa_raw_counts(self, band):
        """Return ``(raw, raw_present)``: the raw counts of IR ``band`` and the lines that carry it.

        The band list also stores raw channel codes, not bands, so the same
        channel-code aliasing as the calibration entries applies.
        """
        if self.band_list is None:
            # Single-band file: this file's one band already is the
            # requested band (per the directory's band_filter_map);
            # the caller is responsible for having checked that.
            raw = self.read_data().astype(np.float64)
            return raw, np.ones(raw.shape[0], dtype=bool)

        match = self._mode_aa_code_to_band(self.band_list) == band
        present = match.any(axis=1)
        slot = np.where(present, match.argmax(axis=1), 0)
        pixels = self._read_pixel_array()
        nlines, nelem, _ = pixels.shape
        raw = np.take_along_axis(
            pixels, np.broadcast_to(slot[:, None, None], (nlines, nelem, 1)), axis=2
        )[:, :, 0].astype(np.float64)
        return raw, present[:nlines]

    def read_ir_temperature_mode_aa(self, band):
        """Mode AA brightness temperature for IR band 1-12.

        Band 13 is visible and not covered by this formula.

        Returns float32 Kelvin, NaN for missing pixels (raw count 0,
        or the offset-corrected count rounds down to 0).
        """
        self._check_mode_aa_request(band)
        f, fk1, fk2, tc1, tc2 = self._MODE_AA_COEFFS[band]
        cal_present, delta_f, ysubz = self._mode_aa_line_calibration(band)
        raw, raw_present = self._mode_aa_raw_counts(band)

        n_common = min(len(cal_present), len(raw_present), raw.shape[0])
        valid_line = cal_present[:n_common] & raw_present[:n_common]

        corrected = np.maximum(0.0, raw[:n_common] - ysubz[:n_common, None])
        corrected = 8.0 * np.floor(corrected / 8.0)
        radiance = corrected * (2.0 ** (delta_f[:n_common, None] - f))

        with np.errstate(invalid="ignore", divide="ignore"):
            expn = fk1 / radiance + 1.0
            t = (fk2 / np.log(expn) - tc1) / tc2

        missing = (raw[:n_common] == 0) | (corrected == 0) | ~valid_line[:, None]
        t = np.where(missing, np.nan, t)
        return t.astype(np.float32)

    def _read_pixel_array(self):
        """Return the raw (nlines, nelem, nbands) pixel array in BIP order.

        BIP is the McIDAS AREA convention: each pixel's num_bands values are
        stored consecutively, then the next pixel's, etc. Shared by read_data()
        and the per-line band-list machinery so both use exactly the same byte
        layout.
        """
        d = self.dir
        nelem = d["num_elements"]
        nbytes = d["bytes_per_elem"]
        nbands = d["num_bands"]
        prefix_len = d["prefix_len"]

        if nbytes not in (1, 2):
            raise NotImplementedError(
                f"{self.path}: {nbytes} bytes/element -- only 1- or 2-byte "
                f"elements are handled."
            )

        line_data_len = nelem * nbytes * nbands
        line_len = prefix_len + line_data_len
        start = d["data_offset"]
        actual_lines = self._actual_num_lines()  # single source of truth -- see its docstring
        end = start + actual_lines * line_len

        block = self._raw[start:end]
        arr = np.frombuffer(block[:actual_lines * line_len], dtype=np.uint8)
        arr = arr.reshape(actual_lines, line_len)
        pixel_bytes = arr[:, prefix_len:prefix_len + line_data_len]

        if nbytes == 1:
            pixels = pixel_bytes.reshape(actual_lines, nelem, nbands)
        else:
            dtype = ">u2" if self._endian == ">" else "<u2"
            pixels = pixel_bytes.copy().view(dtype).reshape(actual_lines, nelem, nbands)
        return pixels

    def read_data(self):
        """Return a 2D array (lines x elements) for a single-band file (Mode A IR, VIS).

        Multiband VAS files (AAA / Mode AA) are read per channel instead: see
        read_channel() (AAA) and read_ir_temperature_mode_aa() (Mode AA), which
        resolve each line's channel from the file itself.
        """
        nbands = self.dir["num_bands"]
        if nbands != 1:
            raise ValueError(
                f"{self.path}: read_data() is for single-band files, but "
                f"this file has {nbands} bands -- use read_channel() "
                f"(AAA) or read_ir_temperature_mode_aa() (Mode AA)."
            )
        return self._read_pixel_array()[:, :, 0]

    _AAA_VALIDITY_LEN = 4
    _AAA_IR_COMMON_DOC_LEN = 512
    _AAA_VAS_CAL_INFO_LEN = 116  # 4 (day) + 4 (time) + 4 (scan#) + 104 (13 x 8B channel/spin groups)
    _AAA_PRE_BANDLIST_LEN = _AAA_VALIDITY_LEN + _AAA_IR_COMMON_DOC_LEN + _AAA_VAS_CAL_INFO_LEN  # 632

    def _parse_band_list(self):
        """Parse each line's band list out of its prefix.

        This is the authoritative source for the band slot -> channel
        mapping. The doc explicitly warns "every line may not contain
        all the spectral bands indicated in the band map" and "the
        order of the bands may not be the same on every line" -- so
        each line's channels are resolved independently, never from a
        single assumed global order.

        Sets:
          self.band_list -- (nlines, nbands) uint8 array, channel number
            (1-38) per band slot per line, or 0 if that slot is unused
            on that line. None if this isn't an AAA file, or if
            prefix_len is too short for this layout to apply.
          self.line_channel_spin -- dict with "channel" and "num_spins"
            (nlines, 13) arrays parsed from the redundant channel/spin-
            count groups earlier in the prefix, kept as an independent
            cross-check on the band list above.
        """
        self.band_list = None
        self.line_channel_spin = None
        if self.dir.get("source_type") not in ("AAA", "VAS"):
            return

        d = self.dir
        nelem = d["num_elements"]
        nbytes = d["bytes_per_elem"]
        nbands = d["num_bands"]
        prefix_len = d["prefix_len"]

        band_list_len = prefix_len - self._AAA_PRE_BANDLIST_LEN
        if band_list_len <= 0:
            logger.warning("%s: prefix_len=%d is too short for the documented AAA line-prefix layout "
                           "(needs > %d bytes before any band list) -- per-line band identification is "
                           "unavailable, so AAA channels cannot be read.",
                           self.path, prefix_len, self._AAA_PRE_BANDLIST_LEN)
            return
        if band_list_len < nbands:
            logger.warning("%s: band list region is only %d bytes but num_bands=%d needs at "
                           "least that many -- per-line band identification is "
                           "unavailable, so AAA channels cannot be read.",
                           self.path, band_list_len, nbands)
            return
        # NOTE: the actual band-list region size doesn't always match a
        # simple "1 byte/band rounded up to the nearest 4-byte word"
        # formula (e.g. one real file was observed with 12 bytes for
        # num_bands=3, not 4) -- rather than guess at the padding
        # scheme, this just takes the region's first nbands bytes and
        # ignores whatever the remaining padding bytes are for.

        line_data_len = nelem * nbytes * nbands
        line_len = prefix_len + line_data_len
        start = d["data_offset"]
        actual_lines = self._actual_num_lines()  # single source of truth -- see its docstring
        block = self._raw[start:start + actual_lines * line_len]
        arr = np.frombuffer(block[:actual_lines * line_len], dtype=np.uint8)
        arr = arr.reshape(actual_lines, line_len)

        band_list_bytes = arr[:, self._AAA_PRE_BANDLIST_LEN:self._AAA_PRE_BANDLIST_LEN + band_list_len]
        # Only the first nbands bytes of the (possibly padded) band list
        # are real per-slot data -- the rest is round-to-a-word filler.
        self.band_list = band_list_bytes[:, :nbands].copy()

        # Independent cross-check: the 13 eight-byte channel/num-spins
        # groups earlier in the prefix (day/time/scan header + 13 x
        # [channel u2, num_spins u2, unused 4B]).
        groups_off = self._AAA_VALIDITY_LEN + self._AAA_IR_COMMON_DOC_LEN + 12  # skip day/time/scan
        dtype16 = ">u2" if self._endian == ">" else "<u2"
        groups_bytes = arr[:, groups_off:groups_off + 104]
        groups = groups_bytes.copy().view(dtype16).reshape(actual_lines, 13, 4)
        self.line_channel_spin = {
            "channel": groups[:, :, 0],
            "num_spins": groups[:, :, 1],
        }

    def read_channel(self, channel):
        """Extract raw counts for a specific VAS channel number (1-38).

        Which band slot holds the channel is resolved ON EACH LINE
        independently via the per-line band list.

        Returns (data, present):
          data    -- (nlines, nelem) raw counts, same dtype as
                     read_data(). Pixels on lines where this channel is
                     absent are 0 and should not be trusted -- check
                     `present`.
          present -- (nlines,) bool, True where `channel` actually
                     appeared in that line's band list.
        """
        if self.band_list is None:
            raise ValueError(
                f"{self.path}: no per-line band list available (see "
                f"_parse_band_list() warnings)."
            )
        pixels = self._read_pixel_array()  # (nlines, nelem, nbands)
        nlines, nelem, nbands = pixels.shape

        match = (self.band_list == channel)  # (nlines, nbands)
        present = match.any(axis=1)
        slot = np.where(present, match.argmax(axis=1), 0)  # 0 is a dummy index when absent

        # Gather each line's chosen band slot without a Python loop.
        out = np.take_along_axis(
            pixels, np.broadcast_to(slot[:, None, None], (nlines, nelem, 1)), axis=2
        )[:, :, 0]
        out = out.copy()
        out[~present] = 0
        return out, present

    def read_ir_radiance(self, channel):
        """AAA-format VAS radiance for a VAS channel number.

        The channel's raw counts are resolved per line via the band
        list (read_channel()), so this is correct even where channel
        order/presence varies line-to-line. Pixels on lines missing
        this channel are NaN. Requires self.band_list (see
        _parse_band_list()).
        """
        if self.cal is None:
            raise ValueError(
                f"{self.path}: no AAA-format calibration block available "
                f"(source_type={self.dir.get('source_type')!r}, expected "
                f"'AAA')."
            )
        lookup_channel = 8 if channel == 13 else channel
        if lookup_channel not in self.cal["iab"] or lookup_channel not in self.cal["ifab"]:
            raise ValueError(
                f"{self.path}: channel {lookup_channel} not present in "
                f"this file's calibration block."
            )
        ab1, ab2 = self.cal["iab"][lookup_channel]
        ifab = self.cal["ifab"][lookup_channel]
        fab = 2.0 ** (15 - ifab)

        p, present = self.read_channel(channel)
        p = p.astype(np.float64)
        r = (ab2 * p / 32.0 - ab1) / fab
        r = np.where(present[:, None], r, np.nan)
        return r.astype(np.float32)

    def read_ir_temperature_aaa(self, channel):
        """AAA-format VAS brightness temperature (Kelvin) for a VAS channel number.

        Goes via radiance (read_ir_radiance()) -> Planck inversion with the
        per-satellite constants. NaN where the inversion is undefined (very
        low radiance, near the noise floor) or the channel is absent on that
        line.
        """
        fk1, fk2, tc1, tc2 = self._vas_planck_coeffs(channel)
        planck_channel = 8 if channel == 13 else channel
        r = self.read_ir_radiance(planck_channel).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            expn = fk1 / r + 1.0
            t = fk2 / np.log(expn)
        t = np.where((r > -fk1) & (expn > 0), (t - tc1) / tc2, np.nan)
        return t.astype(np.float32)

    def _goes_nav(self):
        """Build the GOESNav for this file, raising if it has no GOES-type nav block."""
        if self.nav is None or self.nav["nav_type"] != b"GOES":
            raise ValueError(
                f"{self.path}: no GOES-type nav block available for "
                f"earth location (nav_type={self.nav and self.nav['nav_type']!r})"
            )
        return GOESNav(self._nav_words, self.dir, endian=self._endian)

    def get_nadir_sampling(self):
        """Return the sub-satellite point, altitude and nadir line/element spacing.

        See GOESNav.get_nadir_sampling() for the returned dict.
        """
        return self._goes_nav().get_nadir_sampling()

    def get_lonlats(self, lines=None, elems=None):
        """Earth-locate this area's pixel grid with the pre-GVAR VISSR 'GOES' navigation.

        Returns (lon, lat) arrays in degrees.

        By default earth-locates every pixel in the data block (full
        nlines x nelements grid, in AREA/file line-element coordinates
        i.e. 0-based offsets into *this* area, matching read_data()).
        Pass explicit 1-D `lines`/`elems` arrays (0-based, local to this
        area, same convention as read_data()) to earth-locate a subset,
        e.g. for a coarser nav grid.
        """
        gnav = self._goes_nav()

        d = self.dir
        if lines is None:
            # Use the actually-present line count, not the directory's
            # claimed num_lines -- they can disagree for truncated
            # files, and read_data()'s pixel array always uses the
            # actual count (see _actual_num_lines() docstring). Using
            # the claimed count here would silently produce a lon/lat
            # (and earth-space mask) grid taller than the real pixel
            # data.
            lines = np.arange(self._actual_num_lines())
        if elems is None:
            elems = np.arange(d["num_elements"])
        line_grid, elem_grid = np.meshgrid(lines, elems, indexing="ij")

        # These are AREA (this-file-local) coordinates, 1-based per the
        # McIDAS convention (AREA line 1 == this area's ul_line). GOESNav
        # ._area_to_image() applies the ul_line/ul_element offset and the
        # line_res/elem_res blowdown factor itself -- do NOT add ul_line/
        # ul_element again here, or the offset gets double-counted.
        area_line = line_grid + 1
        area_elem = elem_grid + 1

        lat, lon = gnav.to_latlon(area_line, area_elem)
        return lon, lat

    # -----------------------------------------------------------------
    # Mode AAA radiance-to-brightness-temperature Planck-inversion constants.
    # Courtesy of SSEC via NOAA CLASS.
    #
    # Only channels 1-12 confirmed here -- 13+ weren't validated and
    # may use a different array layout (channel 13 is separately
    # handled below by borrowing channel 8's values, per the McIDAS-X
    # User's Guide VAS channel table).
    _VAS_PLANCK_COEFFS_GOES6 = {
        1: (3716.0, 975.8, -0.0002859, 1.0),
        2: (3936.0, 994.8, 0.002472, 1.0),
        3: (4089.0, 1007.0, 0.002334, 1.0),
        4: (4336.0, 1027.0, 0.003948, 0.9999),
        5: (5019.0, 1079.0, 0.004031, 0.9999),
        6: (128000.0, 3175.0, 0.06857, 0.9998),
        7: (5855.0, 1136.0, 0.004256, 0.9999),
        8: (8438.0, 1283.0, 0.3263, 0.9973),
        9: (31030.0, 1980.0, 0.06822, 0.9997),
        10: (38790.0, 2133.0, 0.736, 0.9973),
        11: (136100.0, 3241.0, 0.05787, 0.9999),
        12: (195000.0, 3653.0, 0.4213, 0.9991),
    }
    _VAS_PLANCK_COEFFS_GOES7 = {
        1: (3767.0, 980.3, -0.001267, 1.0),
        2: (3940.0, 995.1, 0.00318, 0.9999),
        3: (4167.0, 1014.0, 0.002383, 1.0),
        4: (4364.0, 1030.0, 0.002739, 0.9999),
        5: (5025.0, 1079.0, 0.00536, 0.9999),
        6: (129100.0, 3185.0, 0.07353, 0.9998),
        7: (5851.0, 1135.0, 0.004447, 0.9999),
        8: (8525.0, 1287.0, 0.3408, 0.9973),
        9: (31070.0, 1981.0, 0.06478, 0.9998),
        10: (39200.0, 2140.0, 0.6448, 0.9977),
        11: (132500.0, 3212.0, 0.0583, 0.9999),
        12: (193200.0, 3642.0, 0.415, 0.9991),
    }
    # Satellite name (self.dir["satellite"]) -> that satellite's table.
    _VAS_PLANCK_COEFFS_BY_SATELLITE = {
        "GOES-6": _VAS_PLANCK_COEFFS_GOES6,
        "GOES-7": _VAS_PLANCK_COEFFS_GOES7,
    }

    def _vas_planck_coeffs(self, channel):
        """Look up the Planck-inversion coefficients for ``channel``.

        The table is chosen by the file's satellite (self.dir["satellite"]).
        AAA-format VAS data only exists for GOES-6 and GOES-7, so any other
        satellite here means a mislabeled/unsupported file. Channel 13
        borrows channel 8's constants.
        """
        sat = self.dir.get("satellite")
        table = self._VAS_PLANCK_COEFFS_BY_SATELLITE.get(sat)
        if table is None:
            raise NotImplementedError(
                f"{self.path}: no verified VAS Planck-inversion constants "
                f"for satellite {sat!r} -- AAA-format VAS calibration is "
                f"only confirmed for GOES-6 and GOES-7."
            )
        planck_channel = 8 if channel == 13 else channel
        if planck_channel not in table:
            raise NotImplementedError(
                f"No verified Planck-inversion constants for channel "
                f"{channel} on {sat} (only channels 1-12 have been "
                f"checked against real data so far)."
            )
        return table[planck_channel]

    def read_ir_temperature(self):
        """Mode A IR brightness temperature, as float32 Kelvin.

        Raises if this file isn't an IR band.
        """
        if self.dir["band"] != "IR":
            raise ValueError(
                f"{self.path}: band is {self.dir['band']!r}, not IR -- "
                f"the brightness-temperature formula only applies to IR."
            )
        if self.dir.get("source_type") in ("AAA", "VAS"):
            raise NotImplementedError(
                f"{self.path}: this is a {self.dir['source_type']!r}-format "
                f"multiband VAS file -- the single-band 8-bit T=418-B "
                f"formula doesn't apply (it assumes one band, 0-255 raw "
                f"values)."
            )
        b = self.read_data()
        if b.max() > 255:
            raise ValueError(
                f"{self.path}: raw values exceed 8 bits (max={b.max()}). "
                f"The documented T=418-B formula assumes 8-bit VISSR data "
                f"and cannot be applied blindly to this file."
            )
        b = b.astype(np.float32)
        t = np.where(b >= 176, 418.0 - b, 330.0 - b / 2.0)
        return t

    def read_vis_counts(self, calibration="counts"):
        """Read VIS data as raw counts or unnormalized reflectance.

        calibration="counts" gives raw 0-255 counts;
        calibration="unnormalized_reflectance" is a linear rescale to
        percent (0-100).
        """
        if self.dir["band"] != "VIS":
            raise ValueError(f"{self.path}: band is {self.dir['band']!r}, not VIS.")
        raw = self.read_data().astype(np.float32)
        if calibration == "counts":
            return raw
        elif calibration == "unnormalized_reflectance":
            return np.clip(raw / 252.0 * 100.0, 0.0, 100.0)
        else:
            raise ValueError(f"Unsupported VIS calibration {calibration!r}.")
