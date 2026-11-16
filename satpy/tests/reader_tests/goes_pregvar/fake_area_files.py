"""Fake pre-GVAR AREA files for the reader tests."""

import dataclasses

import numpy as np

# Area directory word indices (0-based)
W_SENSOR = 2
W_DATE = 3
W_TIME = 4
W_UL_LINE = 5
W_UL_ELEM = 6
W_NLINES = 8
W_NELEM = 9
W_BYTES = 10
W_LINE_RES = 11
W_ELEM_RES = 12
W_NBANDS = 13
W_PREFIX_LEN = 14
W_DATA_OFFSET = 33
W_NAV_OFFSET = 34
W_VALCODE = 35
W_DOC_LEN = 48
W_CAL_LEN = 49
W_SOURCE_TYPE = 51
W_CAL_TYPE = 52
W_CAL_OFFSET = 62
DIR_WORDS = 64
NAV_WORDS = 128
CAL_WORDS = 128
SUB_LON = -112.1
EARTH_SEMIMAJOR_KM = 6378.137
MCIDAS_MISSING = -2139062144  # 0x80808080 as a signed 32-bit integer

# sensor source numbers from the AREA directory
GOES3_IR = 25
GOES3_VIS = 24
GOES5_IR = 29
GOES6_IR = 31
GOES7_IR = 33


@dataclasses.dataclass
class FakeArea:
    """Description of a fake AREA file."""

    sensor_source: int
    data: np.ndarray  # (lines, elements, bands)
    source_type: str = "VISR"
    cal_type: str = "BRIT"
    ul_line: int = 1
    ul_element: int = 1
    line_res: int = 8
    elem_res: int = 4
    bytes_per_elem: int = 1
    prefixes: np.ndarray | None = None  # (lines, prefix_len) uint8
    cal_words: np.ndarray | None = None
    valcode: int = 0
    doc_len: int = 0
    cal_len: int = 0
    claimed_lines: int | None = None
    endian: str = ">"
    nav_words: np.ndarray | None = None


def make_nav_words(skew_word=0):
    """Nav block of a circular equatorial orbit with a polar spin axis.

    The scan geometry gives about 0.86 km between full-resolution lines and
    0.75 km between full-resolution elements at nadir, like the real files.
    """
    words = np.zeros(NAV_WORDS, dtype=np.int64)
    words[0] = int.from_bytes(b"GOES", "big")
    words[1] = 92236  # SSYYDDD
    words[2] = 130100  # nominal HHMMSS
    words[4] = 920823  # epoch YYMMDD
    words[5] = 130000  # epoch HHMMSS
    words[6] = 4216417  # semimajor axis [km * 100]
    words[7] = 0  # eccentricity * 1e6
    words[8] = 0  # inclination * 1000
    words[9] = 0  # mean anomaly * 1000
    words[10] = 0  # argument of perigee * 1000
    words[11] = 54973  # right ascension of ascending node * 1000
    words[12] = 900000  # spin axis declination, DDDMMSS
    words[13] = 0  # spin axis right ascension, DDDMMSS
    words[14] = 7285  # picture center line
    words[15] = 600000  # spin period [us]
    words[16] = 200000  # degrees of line sweep, DDDMMSS (20 deg)
    words[17] = 8 * 100000 + 1821  # NNLLLLL: 8 detectors, 1821 scans
    words[18] = 182100  # degrees of element sweep, DDDMMSS (18 deg 21 min)
    words[19] = 15288  # elements per line
    words[28] = skew_word
    return words


def _pack(words, endian):
    return np.asarray(words).astype(f"{endian}i4").tobytes()


def _directory(spec, nav_offset, cal_offset, data_offset):
    words = np.zeros(DIR_WORDS, dtype=np.int64)
    nlines = spec.data.shape[0]
    words[W_SENSOR] = spec.sensor_source
    words[W_DATE] = 92236
    words[W_TIME] = 130100
    words[W_UL_LINE] = spec.ul_line
    words[W_UL_ELEM] = spec.ul_element
    words[W_NLINES] = spec.claimed_lines or nlines
    words[W_NELEM] = spec.data.shape[1]
    words[W_BYTES] = spec.bytes_per_elem
    words[W_LINE_RES] = spec.line_res
    words[W_ELEM_RES] = spec.elem_res
    words[W_NBANDS] = spec.data.shape[2]
    words[W_PREFIX_LEN] = 0 if spec.prefixes is None else spec.prefixes.shape[1]
    words[W_DATA_OFFSET] = data_offset
    words[W_NAV_OFFSET] = nav_offset
    words[W_VALCODE] = spec.valcode
    words[W_DOC_LEN] = spec.doc_len
    words[W_CAL_LEN] = spec.cal_len
    words[W_CAL_OFFSET] = cal_offset
    raw = bytearray(_pack(words, spec.endian))
    # source and calibration types are plain ASCII, independent of byte order
    raw[W_SOURCE_TYPE * 4:W_SOURCE_TYPE * 4 + 4] = spec.source_type.ljust(4).encode("ascii")
    raw[W_CAL_TYPE * 4:W_CAL_TYPE * 4 + 4] = spec.cal_type.ljust(4).encode("ascii")
    return bytes(raw)


def write_area_file(path, spec):
    """Write a fake AREA file and return its path as a string."""
    nav_offset = DIR_WORDS * 4
    cal_offset = nav_offset + NAV_WORDS * 4 if spec.cal_words is not None else 0
    data_offset = (cal_offset + CAL_WORDS * 4) if spec.cal_words is not None else nav_offset + NAV_WORDS * 4

    nav_words = make_nav_words() if spec.nav_words is None else spec.nav_words
    nav = b"GOES" + _pack(nav_words[1:], spec.endian)

    blocks = [_directory(spec, nav_offset, cal_offset, data_offset), nav]
    if spec.cal_words is not None:
        blocks.append(_pack(spec.cal_words, spec.endian))
    pixels = spec.data.astype(f"{spec.endian}u{spec.bytes_per_elem}")
    for line in range(spec.data.shape[0]):
        if spec.prefixes is not None:
            blocks.append(spec.prefixes[line].astype(np.uint8).tobytes())
        blocks.append(pixels[line].tobytes())
    path.write_bytes(b"".join(blocks))
    return str(path)


def patch_dir_word(path, index, value):
    """Overwrite one (big-endian) directory word of a fake file."""
    with open(path, "r+b") as f:
        f.seek(index * 4)
        f.write(int(value).to_bytes(4, "big", signed=True))


# ---------------------------------------------------------------------------
# Fake files
# ---------------------------------------------------------------------------
# 16 lines x 24 elements around nadir (nadir is at row 8, column 12 for IR)
IR_WINDOW = {"ul_line": 7213, "ul_element": 7592, "line_res": 8, "elem_res": 4}
VIS_WINDOW = {"ul_line": 7276, "ul_element": 7631, "line_res": 1, "elem_res": 1}
# 16 lines straddling the northern limb; the first rows look into space
LIMB_WINDOW = {"ul_line": 900, "ul_element": 7592, "line_res": 8, "elem_res": 4}
SHAPE = (16, 24)
NADIR_ROW, NADIR_COL = 8, 12


def mode_a_data(values):
    """Single band image where each column holds one of the given values."""
    columns = np.resize(np.asarray(values, dtype=np.uint8), SHAPE[1])
    return np.tile(columns[None, :, None], (SHAPE[0], 1, 1))


def mode_aa_spec(raw=10000, channel_codes=(8,) + (0,) * 12):
    """Mode AA file: calibration entries in every line's prefix, 2-byte data."""
    nlines, nelem = SHAPE
    prefix_len = 4 + 512 + 116 + 4
    prefixes = np.zeros((nlines, prefix_len), dtype=np.uint8)
    for line in range(nlines):
        entries = np.zeros((13, 4), dtype=">i2")
        entries[:, 0] = channel_codes
        entries[0, 2] = 0  # delta-F code 0 -> delta-F = 0
        entries[0, 3] = 400  # space-view offset
        prefixes[line, 4 + 512 + 12:4 + 512 + 116] = np.frombuffer(entries.tobytes(), dtype=np.uint8)
        prefixes[line, 632] = channel_codes[0]  # band list: raw channel code of the one band
    data = np.full((nlines, nelem, 1), raw)
    return FakeArea(
        GOES5_IR, data, source_type="VAS", cal_type="RAW", bytes_per_elem=2, prefixes=prefixes,
        valcode=1, doc_len=512, cal_len=116, **IR_WINDOW)


# AAA fake: three bands whose order differs between lines
AAA_BAND_ORDERS = ((8, 7, 10), (7, 8, 10))
AAA_COUNTS = {7: 3000, 8: 4000, 10: 5000}
AAA_AB1, AAA_AB2, AAA_IFAB = 0, 32, 15  # radiance = counts for these coefficients


def _aaa_cal_words(sensor_source):
    words = np.zeros(CAL_WORDS, dtype=np.int64)
    words[0] = sensor_source
    for channel in range(1, 39):
        words[3 + 2 * (channel - 1)] = AAA_AB1
        words[3 + 2 * (channel - 1) + 1] = AAA_AB2
        words[79 + channel - 1] = AAA_IFAB
    return words


def aaa_spec(sensor_source):
    """Describe an AAA file with channels 8, 7 and 10 in an order that changes between lines."""
    nlines, nelem = SHAPE
    prefix_len = 632 + 4
    prefixes = np.zeros((nlines, prefix_len), dtype=np.uint8)
    data = np.zeros((nlines, nelem, 3), dtype=np.int64)
    for line in range(nlines):
        order = AAA_BAND_ORDERS[line % 2]
        prefixes[line, 632:632 + 3] = order
        for slot, channel in enumerate(order):
            data[line, :, slot] = AAA_COUNTS[channel]
    return FakeArea(
        sensor_source, data, source_type="AAA", cal_type="RAW", bytes_per_elem=2,
        prefixes=prefixes, cal_words=_aaa_cal_words(sensor_source), **IR_WINDOW)


def planck_temperature(radiance, constants):
    """Planck inversion with a band correction, written out independently of the reader."""
    fk1, fk2, tc1, tc2 = constants
    return (fk2 / np.log(1.0 + fk1 / radiance) - tc1) / tc2
