"""Unittests for Landsat image readers.

The tests are data-driven. Every tested product is described by one :class:`LandsatProduct` entry in ``PRODUCTS``,
holding the values expected from its metadata file in ``landsat_metadata``. To test a new product, add an entry there.

Tests select the products they apply to with :func:`product_params`, e.g. ``product_params("thermal")`` for the
products having a thermal band. For every product a complete synthetic granule is written once per module: one
GeoTIFF for each dataset configured in the reader YAML file, plus the MTL metadata file (see :class:`LandsatFiles`).
"""

import contextlib
import functools
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pytest
import rioxarray  # noqa: F401  # registers the xarray ``.rio`` accessor
import xarray as xr
from pyresample.geometry import AreaDefinition
from trollsift import compose, parse

from satpy import Scene
from satpy.readers.core.config import configs_for_reader
from satpy.readers.core.landsat import (
    ETMCHReader,
    ETML2CHReader,
    LandsatL1MDReader,
    LandsatL2MDReader,
    MSSCHReader,
    OLITIRSCHReader,
    OLITIRSL2CHReader,
    TMCHReader,
    TML2CHReader,
)
from satpy.readers.core.loading import load_reader

# ----------------------------------------------------
# Constants ------------------------------------------
# ----------------------------------------------------
# Values shared by the products and the tests.

ETC_DIR = os.path.join(os.path.dirname(__file__), "landsat_metadata")

# Image sizes set in the test metadata files (REFLECTIVE_LINES/SAMPLES and PANCHROMATIC_LINES/SAMPLES)
SHAPE = (100, 100)
PAN_SHAPE = (200, 200)
# The spectral band used in the tests, present in all products
SPECTRAL = "B4"
# Level-2 datasets stored as int16 with a fill value of -9999, all others use a fill value of 0
FILL_9999_DATASETS = {"TRAD", "DRAD", "URAD", "ATRAN", "EMIS", "EMSD", "CDIST", "qa_st", "qa_atmos_opacity"}
# Angle dataset names and their standard names
ANGLES = {"solar_zenith_angle": "solar_zenith_angle",
          "solar_azimuth_angle": "solar_azimuth_angle",
          "satellite_zenith_angle": "viewing_zenith_angle",
          "satellite_azimuth_angle": "viewing_azimuth_angle"}
RADIANCE_UNITS = "W m-2 um-1 sr-1"
TOA_RADIANCE = "toa_outgoing_radiance_per_unit_wavelength"


# ----------------------------------------------------
# Products -------------------------------------------
# ----------------------------------------------------
# The tested products, one ``LandsatProduct`` per product in ``PRODUCTS``, with the values expected from their
# metadata files and the features (thermal band, angles, panchromatic band...) the tests are selected by.

@dataclass(frozen=True)
class LandsatProduct:
    """A Landsat product and the values expected from its test metadata file.

    The fields after ``saturation`` describe what the product has. They default to "not available" and are used by
    :func:`product_params` to select the products a test applies to.
    """

    # What to test and where its files come from
    id: str  # test id, also the name of the temporary directory of the granule
    reader: str
    prefix: str  # filename prefix shared by all files of the granule, filename info is parsed from it
    mtl: str  # metadata file in ``ETC_DIR``
    channel_handler: type
    metadata_handler: type

    # Values expected from the metadata file
    platform_name: str
    sensor: str
    start_time: datetime  # observation date from the filename combined with the scene center time
    earth_sun_distance: float
    cloud_cover: float
    zone: int  # UTM zone
    extent: tuple  # area extent of the reflective and thermal bands
    # Calibration coefficients by band, a subset of the metadata file which must include B4 and the thermal band.
    # Level 1: (radiance gain, radiance offset, reflectance gain, reflectance offset) for spectral bands and
    #          (radiance gain, radiance offset, K1, K2) for thermal bands.
    # Level 2: (gain, offset) of the reflectance or brightness temperature.
    calibration: dict
    saturation: dict  # all saturation flags by band, thermal bands of some products have none

    # What the product has
    thermal: str | None = None  # thermal band used in the tests, MSS has none
    has_angles: bool = False  # level-1 products except MSS have angle files
    pan_extent: tuple | None = None  # area extent of the panchromatic band B8, if any
    wavelengths: dict = field(default_factory=dict)  # band wavelengths which depend on the platform (MSS)
    absent_datasets: tuple = ()  # datasets of the reader not produced by this platform (MSS)
    # 8< v1.0
    # Products offering both the deprecated 'reflectance' and the 'unnormalized_reflectance' calibrations
    has_unnormalized_reflectance: bool = False
    # >8 v1.0

    @property
    def level(self):
        """Processing level of the product."""
        return 2 if "_l2_" in self.reader else 1

    @property
    def is_oli_tirs(self):
        """Whether the granule can contain only OLI or only TIRS data."""
        return self.sensor == "OLI_TIRS"

    @property
    def band_names(self):
        """The spectral and (if any) thermal band used in the tests."""
        return [SPECTRAL, self.thermal] if self.thermal else [SPECTRAL]

    @property
    def area(self):
        """Area of the reflective and thermal bands."""
        return self._area(SHAPE, self.extent)

    @property
    def pan_area(self):
        """Area of the panchromatic band."""
        return self._area(PAN_SHAPE, self.pan_extent)

    def area_for(self, name):
        """Area of the dataset ``name``."""
        return self.pan_area if name == "B8" and self.pan_extent else self.area

    def _area(self, shape, extent):
        # Built independently of the reader, which uses an EPSG code instead of a PROJ dict
        pcs_id = f"WGS84 / UTM zone {self.zone}N"
        proj_dict = {"proj": "utm", "zone": self.zone, "datum": "WGS84", "units": "m", "no_defs": None, "type": "crs"}
        return AreaDefinition("geotiff_area", pcs_id, pcs_id, proj_dict, shape[1], shape[0], extent)


# The expected values are those of the metadata files in ``ETC_DIR``
PRODUCTS = [
    LandsatProduct(
        id="oli_tirs_l1", reader="oli_tirs_l1_tif", prefix="LC08_L1TP_026200_20240502_20240513_02_T2",
        mtl="oli_tirs_l1_metadata.xml", channel_handler=OLITIRSCHReader, metadata_handler=LandsatL1MDReader,
        platform_name="Landsat-8", sensor="OLI_TIRS", start_time=datetime(2024, 5, 2, 18, 0, 24, tzinfo=timezone.utc),
        earth_sun_distance=1.0079981, cloud_cover=0.85,
        zone=40, extent=(619485.0, 2440485.0, 850515.0, 2675715.0),
        pan_extent=(619492.5, 2440492.5, 850507.5, 2675707.5),
        calibration={"B1": (0.012357, -61.78647, 2e-05, -0.1),
                     "B4": (0.0098329, -49.16426, 2e-05, -0.1),
                     "B10": (0.0003342, 0.1, 774.8853, 1321.0789),
                     "B11": (0.0003342, 0.1, 480.8883, 1201.1442)},
        saturation={"B1": False, "B2": False, "B3": False, "B4": True, "B5": False, "B6": False, "B7": False,
                    "B8": False, "B9": False},
        thermal="B11", has_angles=True, has_unnormalized_reflectance=True,
    ),
    LandsatProduct(
        id="oli_tirs_l2", reader="oli_tirs_l2_tif", prefix="LC09_L2SP_029030_20240616_20240617_02_T1",
        mtl="oli_tirs_l2_metadata.xml", channel_handler=OLITIRSL2CHReader, metadata_handler=LandsatL2MDReader,
        platform_name="Landsat-9", sensor="OLI_TIRS", start_time=datetime(2024, 6, 16, 17, 10, 58, tzinfo=timezone.utc),
        earth_sun_distance=1.0158933, cloud_cover=29.27,
        zone=14, extent=(534885.0, 4665585.0, 765015.0, 4899315.0),
        calibration={"B1": (2.75e-05, -0.2), "B4": (2.75e-05, -0.2), "B10": (0.00341802, 149.0)},
        saturation={"B1": False, "B2": False, "B3": False, "B4": True, "B5": False, "B6": True, "B7": True,
                    "B8": False, "B9": False},
        thermal="B10",
    ),
    LandsatProduct(
        id="etm_l1", reader="etm_l1_tif", prefix="LE07_L1TP_230080_20231208_20240103_02_T1",
        mtl="etm_l1_metadata.xml", channel_handler=ETMCHReader, metadata_handler=LandsatL1MDReader,
        platform_name="Landsat-7", sensor="ETM+", start_time=datetime(2023, 12, 8, 11, 44, 51, tzinfo=timezone.utc),
        earth_sun_distance=0.9850987, cloud_cover=10.0,
        zone=20, extent=(240885.0, -3301215.0, 483015.0, -3091485.0),
        pan_extent=(240892.5, -3301207.5, 483007.5, -3091492.5),
        calibration={"B1": (0.77874, -6.97874, 0.0011661, -0.01045),
                     "B4": (0.96929, -6.06929, 0.0027591, -0.017277),
                     "B6_VCID_1": (0.067087, -0.06709, 666.09, 1282.71),
                     "B6_VCID_2": (0.037205, 3.1628, 666.09, 1282.71)},
        saturation={"B1": False, "B2": True, "B3": True, "B4": True, "B5": False, "B6_VCID_1": False,
                    "B6_VCID_2": False, "B7": True, "B8": False},
        thermal="B6_VCID_1", has_angles=True,
    ),
    LandsatProduct(
        id="etm_l2", reader="etm_l2_tif", prefix="LE07_L2SP_028030_20230817_20230912_02_T1",
        mtl="etm_l2_metadata.xml", channel_handler=ETML2CHReader, metadata_handler=LandsatL2MDReader,
        platform_name="Landsat-7", sensor="ETM+", start_time=datetime(2023, 8, 17, 14, 54, 20, tzinfo=timezone.utc),
        earth_sun_distance=1.0124651, cloud_cover=0.0,
        zone=15, extent=(85785.0, 4680885.0, 336915.0, 4904115.0),
        calibration={"B1": (2.75e-05, -0.2), "B4": (2.75e-05, -0.2), "B6": (0.00341802, 149.0)},
        saturation={"B1": False, "B2": True, "B3": True, "B4": True, "B5": False, "B6_VCID_1": False,
                    "B6_VCID_2": True, "B7": True, "B8": False},
        thermal="B6",
    ),
    LandsatProduct(
        id="tm_l1", reader="tm_l1_tif", prefix="LT04_L1TP_143021_19890818_20200916_02_T1",
        mtl="tm_l1_metadata.xml", channel_handler=TMCHReader, metadata_handler=LandsatL1MDReader,
        platform_name="Landsat-4", sensor="TM", start_time=datetime(1989, 8, 18, 4, 26, 11, tzinfo=timezone.utc),
        earth_sun_distance=1.0122057, cloud_cover=10.0,
        zone=46, extent=(322185.0, 6085185.0, 567615.0, 6311415.0),
        calibration={"B1": (0.67921, -2.19921, 0.0011252, -0.003643),
                     "B4": (0.87602, -2.38602, 0.0027296, -0.007435),
                     "B6": (0.055375, 1.18243, 671.62, 1284.3)},
        saturation={"B1": False, "B2": True, "B3": True, "B4": True, "B5": False, "B6": False, "B7": True},
        thermal="B6", has_angles=True,
    ),
    LandsatProduct(
        id="tm_l2", reader="tm_l2_tif", prefix="LT05_L2SP_165054_20110817_20200820_02_T1",
        mtl="tm_l2_metadata.xml", channel_handler=TML2CHReader, metadata_handler=LandsatL2MDReader,
        platform_name="Landsat-5", sensor="TM", start_time=datetime(2011, 8, 17, 7, 10, 40, tzinfo=timezone.utc),
        earth_sun_distance=1.0125021, cloud_cover=1.0,
        zone=38, extent=(258285.0, 855285.0, 495315.0, 1064415.0),
        calibration={"B1": (2.75e-05, -0.2), "B4": (2.75e-05, -0.2), "B6": (0.00341802, 149.0)},
        saturation={"B1": False, "B2": False, "B3": True, "B4": True, "B5": False, "B6": False, "B7": True},
        thermal="B6",
    ),
    LandsatProduct(
        id="mss_l1_landsat1", reader="mss_l1_tif", prefix="LM01_L1TP_032030_19720729_20200909_02_T2",
        mtl="mss_l1_landsat1_metadata.xml", channel_handler=MSSCHReader, metadata_handler=LandsatL1MDReader,
        platform_name="Landsat-1", sensor="MSS", start_time=datetime(1972, 7, 29, 16, 49, 31, tzinfo=timezone.utc),
        earth_sun_distance=1.0152109, cloud_cover=0.0,
        zone=14, extent=(442110.0, 4667550.0, 673530.0, 4887990.0),
        calibration={"B4": (0.95591, -18.55591, 0.0017282, -0.033547),
                     "B5": (0.64843, -0.74843, 0.001366, -0.001577)},
        saturation={"B4": True, "B5": False, "B6": False, "B7": False},
        wavelengths={"B4": (0.5, 0.55, 0.6), "B5": (0.6, 0.65, 0.7)},
        absent_datasets=("B1", "B2", "B3"),
    ),
    LandsatProduct(
        id="mss_l1_landsat4", reader="mss_l1_tif", prefix="LM04_L1TP_029030_19840415_20200903_02_T2",
        mtl="mss_l1_landsat4_metadata.xml", channel_handler=MSSCHReader, metadata_handler=LandsatL1MDReader,
        platform_name="Landsat-4", sensor="MSS", start_time=datetime(1984, 4, 15, 16, 38, 15, tzinfo=timezone.utc),
        earth_sun_distance=1.0035512, cloud_cover=2.0,
        zone=14, extent=(540030.0, 4679970.0, 774450.0, 4888350.0),
        calibration={"B1": (0.8752, 2.9248, 0.001568, 0.00524),
                     "B4": (0.47638, 3.82362, 0.0017954, 0.014411)},
        saturation={"B1": False, "B2": False, "B3": False, "B4": True},
        wavelengths={"B4": (0.8, 0.95, 1.1), "B3": (0.7, 0.75, 0.8)},
        absent_datasets=("B5", "B6", "B7"),
    ),
]


# ----------------------------------------------------
# Calibration cases ----------------------------------
# ----------------------------------------------------
# The calibrations tested for each product: which dataset is loaded with which calibration, the expected units and
# standard name, and the formula for the expected values.

@dataclass(frozen=True)
class CalibrationCase:
    """A calibration of a dataset and the function computing its expected values from counts and coefficients."""

    dataset: str
    calibration: str
    units: str
    standard_name: str
    expected: Callable  # expected(counts, coefficients of the dataset in ``LandsatProduct.calibration``)
    deprecated: bool = False  # whether loading it raises a DeprecationWarning


# Functions computing the expected calibrated values, see ``LandsatProduct.calibration`` for the coefficients.
# They are applied to the counts as float64, so they are an independent reference for the float32 reader output.

    @property
    def id(self):
        """Test id."""
        return f"{self.dataset}-{self.calibration}"


def _counts(counts, coefs):
    return counts


def _linear(counts, coefs):
    # Level-1 radiance, level-2 brightness temperature
    return counts * coefs[0] + coefs[1]


def _l1_reflectance(counts, coefs):
    # In percent, without the solar zenith angle normalization
    return (counts * coefs[2] + coefs[3]) * 100


def _l1_brightness_temperature(counts, coefs):
    # Inverse Planck function: K2 / ln(K1 / radiance + 1)
    return coefs[3] / np.log(coefs[2] / _linear(counts, coefs) + 1)


def _l2_reflectance(counts, coefs):
    # Surface reflectance in percent
    return _linear(counts, coefs) * 100


def _scaled(factor):
    # Level-2 auxiliary datasets have a fixed scale factor instead of coefficients in the metadata file
    return lambda counts, coefs: counts * factor


# Functions returning the calibration cases of a product. Each one covers one kind of dataset and returns no cases
# when the product doesn't have it.

def _spectral_cases(product):
    cases = [CalibrationCase(SPECTRAL, "counts", "1", "counts", _counts)]
    if product.level == 2:
        return cases + [CalibrationCase(SPECTRAL, "reflectance", "%", "surface_bidirectional_reflectance",
                                        _l2_reflectance)]
    cases += [CalibrationCase(SPECTRAL, "radiance", RADIANCE_UNITS, TOA_RADIANCE, _linear),
              # 8< v1.0
              CalibrationCase(SPECTRAL, "reflectance", "%", "toa_bidirectional_reflectance", _l1_reflectance,
                              deprecated=True),
              # >8 v1.0
              ]
    if product.has_unnormalized_reflectance:
        cases.append(CalibrationCase(
            SPECTRAL, "unnormalized_reflectance", "%",
            "product_of_cosine_solar_zenith_angle_and_toa_bidirectional_reflectance", _l1_reflectance))
    return cases


def _thermal_cases(product):
    if product.thermal is None:
        return []
    cases = [CalibrationCase(product.thermal, "counts", "1", "counts", _counts)]
    if product.level == 2:
        return cases + [CalibrationCase(product.thermal, "brightness_temperature", "K", "brightness_temperature",
                                        _linear)]
    return cases + [
        CalibrationCase(product.thermal, "radiance", RADIANCE_UNITS, TOA_RADIANCE, _linear),
        CalibrationCase(product.thermal, "brightness_temperature", "K", "brightness_temperature",
                        _l1_brightness_temperature),
    ]


def _l2_auxiliary_cases(product):
    if product.level != 2:
        return []
    return [
        CalibrationCase("TRAD", "radiance", RADIANCE_UNITS, TOA_RADIANCE, _scaled(0.001)),
        CalibrationCase("DRAD", "radiance", RADIANCE_UNITS, "downwelling_radiance_per_unit_wavelength_in_air",
                        _scaled(0.001)),
        CalibrationCase("URAD", "radiance", RADIANCE_UNITS, "surface_upwelling_radiance_per_unit_wavelength_in_air",
                        _scaled(0.001)),
        CalibrationCase("ATRAN", "atmospheric_transmittance", "1", "atmospheric_transmittance", _scaled(0.0001)),
        CalibrationCase("EMIS", "emissivity", "1", "surface_longwave_emissivity", _scaled(0.0001)),
        CalibrationCase("EMSD", "emissivity", "1", "surface_longwave_emissivity", _scaled(0.0001)),
        CalibrationCase("CDIST", "cloud_distance", "km", "distance_to_cloud", _scaled(0.01)),
        CalibrationCase("qa_st", "qa_brightness_temperature", "K", "brightness_temperature", _scaled(0.01)),
    ]


def _calibration_cases(product):
    return _spectral_cases(product) + _thermal_cases(product) + _l2_auxiliary_cases(product)


def _angle_cases(product):
    return list(ANGLES) if product.has_angles else []


# ----------------------------------------------------
# Test parametrization -------------------------------
# ----------------------------------------------------
# The tests are parametrized with the helpers below and ``indirect=True``, so the products are passed through the
# ``product`` fixture, e.g. to run a test for all products having a thermal band::
#
#     @pytest.mark.parametrize("product", product_params("thermal"), indirect=True)
#     def test_something(product, landsat_files):

def product_params(*required):
    """Get the products having all ``required`` attributes set, as parameters for the ``product`` fixture."""
    return [pytest.param(product, id=product.id)
            for product in PRODUCTS if all(getattr(product, attr) for attr in required)]


def product_case_params(cases_for):
    """Get (product, case) parameters for each product and each of its cases returned by ``cases_for(product)``.

    The test ids combine both, e.g. ``etm_l1-B4-radiance``.
    """
    return [pytest.param(product, case, id=f"{product.id}-{getattr(case, 'id', case)}")
            for product in PRODUCTS for case in cases_for(product)]


# ----------------------------------------------------
# Test files -----------------------------------------
# ----------------------------------------------------
# Writing the synthetic granule of a product: a copy of its metadata file and one GeoTIFF of random counts per
# dataset of the reader, named after the reader's file patterns. ``LandsatFiles`` gives the tests access to the files,
# the counts written to them and file handlers created from them.

@functools.cache
def _reader_config(reader):
    # The content of the reader YAML file: its file types (with file patterns) and datasets
    return load_reader(next(configs_for_reader(reader))).config


def _filename_info(reader, filename):
    # Parse the filename with the reader's own pattern, as the reader does when it is given the file
    pattern = _reader_config(reader)["file_types"]["l1_metadata"]["file_patterns"][0]
    return parse(pattern, os.path.basename(filename))


def _write_tif(filename, data, area):
    # Georeferenced like the real files, although the reader takes the area from the metadata file
    x, y = area.get_proj_vectors()
    data_array = xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x})
    data_array.rio.write_crs(area.crs).rio.to_raster(filename)


def _make_counts(name, shape, rng):
    # Random counts, different for each dataset so that loading the wrong file is detected
    if name in FILL_9999_DATASETS:
        data, fill_value = rng.integers(1, 10000, shape, dtype=np.int16), -9999
    else:
        data, fill_value = rng.integers(1000, 16000, shape, dtype=np.uint16), 0
    # The first row holds fill values only
    data[0] = fill_value
    return data


def _to_fsfiles(filenames):
    from fsspec.implementations.local import LocalFileSystem

    from satpy.readers.core.remote import FSFile

    return [FSFile(os.path.abspath(filename), fs=LocalFileSystem()) for filename in filenames]


class LandsatFiles:
    """A synthetic granule of a Landsat product: one GeoTIFF per dataset of the reader and the MTL file."""

    def __init__(self, product, directory):
        """Write the granule of ``product`` to ``directory``."""
        self.product = product
        self.config = _reader_config(product.reader)
        self.mtl = os.fspath(directory / f"{product.prefix}_MTL.xml")
        shutil.copyfile(os.path.join(ETC_DIR, product.mtl), self.mtl)
        self.filename_info = _filename_info(product.reader, self.mtl)
        # Both by dataset name
        self.paths = {}
        self.counts = {}  # the data written to the files, to compute the expected values from
        rng = np.random.default_rng(42)
        # One file per dataset of the reader, so new datasets in the YAML file get tested automatically
        for name, ds_info in self.config["datasets"].items():
            if name not in product.absent_datasets:
                self._write_dataset(directory, name, ds_info["file_type"], rng)

    def _write_dataset(self, directory, name, file_type, rng):
        area = self.product.area_for(name)
        # The filename is built from the reader's file pattern, so the reader is sure to recognise it
        pattern = self.config["file_types"][file_type]["file_patterns"][0]
        self.paths[name] = os.fspath(directory / compose(pattern, self.filename_info))
        self.counts[name] = _make_counts(name, area.shape, rng)
        _write_tif(self.paths[name], self.counts[name], area)

    @property
    def all_paths(self):
        """All the files of the granule."""
        return list(self.paths.values()) + [self.mtl]

    @property
    def uncalibrated_names(self):
        """Names of the QA and gap mask datasets which are returned as stored in the file."""
        return [name for name in self.paths
                if "calibration" not in self.config["datasets"][name] and name not in ANGLES]

    def select(self, *names):
        """Get the files of the datasets ``names`` together with the MTL file."""
        return [self.paths[name] for name in names] + [self.mtl]

    def load(self, names, **kwargs):
        """Load the datasets ``names`` from their files."""
        scn = Scene(reader=self.product.reader, filenames=self.select(*names))
        scn.load(names, **kwargs)
        return scn

    def counts_with_nan(self, name):
        """Get the counts of dataset ``name`` as float, with fill values replaced by NaN."""
        # This is what the reader returns for the counts, and the input for computing calibrated values
        counts = self.counts[name].astype(np.float64)
        counts[0] = np.nan
        return counts

    # The file handlers are created directly, without a Scene, to test their errors.
    # ``info_overrides`` replace values of the filename info, e.g. ``platform_type="B"`` for a non-Landsat file.

    def metadata_handler(self, **info_overrides):
        """Create the metadata file handler."""
        return self.product.metadata_handler(self.mtl, self.filename_info | info_overrides, {})

    def channel_handler(self, name, file_type=None, **info_overrides):
        """Create the file handler of dataset ``name``, by default with the file type matching its name."""
        ftype_info = {"file_type": file_type or f"granule_{name}"}
        return self.product.channel_handler(self.paths[name], self.filename_info | info_overrides, ftype_info,
                                            self.metadata_handler())


# ----------------------------------------------------
# Fixtures -------------------------------------------
# ----------------------------------------------------
# ``product`` is the product under test and ``landsat_files`` its granule, written once per module.

@pytest.fixture(scope="module")
def granules(tmp_path_factory):
    """Get a function returning the granule of a product, writing it only on first use."""
    # A cache instead of a module-scoped fixture parametrized by product: each test selects its own products, so
    # pytest may not group the tests by product, and would then write the same granule again. With the cache, each
    # granule is written once whatever the test order.
    cache = {}

    def get_granule(product):
        if product.id not in cache:
            cache[product.id] = LandsatFiles(product, tmp_path_factory.mktemp(product.id))
        return cache[product.id]

    return get_granule


@pytest.fixture
def product(request):
    """Get the product under test, set by parametrizing the test with :func:`product_params`."""
    return request.param


@pytest.fixture
def landsat_files(product, granules):
    """Get the granule of the product under test."""
    return granules(product)


# ----------------------------------------------------
# Loading tests --------------------------------------
# ----------------------------------------------------
# Loading datasets through a Scene: availability of every dataset, and the attributes of the loaded bands.

# Attributes compared as dicts in a single assert, so that a failure shows all the differences at once

def _expected_attrs(product, name):
    return {"shape": product.area_for(name).shape, "area": product.area_for(name),
            "platform_name": product.platform_name, "sensor": product.sensor,
            "perc_cloud_cover": product.cloud_cover, "start_time": product.start_time,
            "saturated": product.saturation.get(name)}


def _actual_attrs(data_arr):
    return {"shape": data_arr.shape, "area": data_arr.attrs["area"],
            "platform_name": data_arr.attrs["platform_name"], "sensor": data_arr.attrs["sensor"],
            "perc_cloud_cover": data_arr.attrs["perc_cloud_cover"], "start_time": data_arr.attrs["start_time"],
            "saturated": data_arr.attrs.get("saturated")}


# Loading 'reflectance', the default calibration of the spectral bands, is deprecated for level-1 products
@pytest.mark.filterwarnings("ignore:The 'reflectance' calibration:DeprecationWarning")
@pytest.mark.parametrize("product", product_params(), indirect=True)
def test_all_datasets_load(product, landsat_files):
    """Test that every dataset of the reader is available and loads, not only the bands used in other tests."""
    # Bugs in less used datasets, like the one fixed in #3300 for MSS bands other than B4, show up here
    scn = Scene(reader=product.reader, filenames=landsat_files.all_paths)
    assert set(scn.available_dataset_names()) == set(landsat_files.paths)

    scn.load(scn.available_dataset_names())
    assert {data_id["name"] for data_id in scn.keys()} == set(landsat_files.paths)


# The files are given to the Scene as paths, or as FSFile objects like files on remote file systems
@pytest.mark.parametrize("to_filenames", [list, _to_fsfiles], ids=["local", "remote"])
@pytest.mark.parametrize("product", product_params(), indirect=True)
def test_basic_load(product, landsat_files, to_filenames):
    """Test loading the bands and their attributes, from local paths and from FSFile objects."""
    scn = Scene(reader=product.reader, filenames=to_filenames(landsat_files.select(*product.band_names)))
    scn.load(product.band_names, calibration="counts")

    assert scn.start_time == scn.end_time == product.start_time
    for name in product.band_names:
        assert _actual_attrs(scn[name]) == _expected_attrs(product, name)


# ----------------------------------------------------
# Calibration tests ----------------------------------
# ----------------------------------------------------
# The values of the loaded datasets: calibrated bands, angles, and QA and gap mask datasets returned as stored.

def _expect_deprecation(deprecated):
    if deprecated:
        return pytest.warns(DeprecationWarning, match="'reflectance' is deprecated")
    return contextlib.nullcontext()


@pytest.mark.parametrize(("product", "case"), product_case_params(_calibration_cases), indirect=["product"])
def test_calibration(product, case, landsat_files):
    """Test the calibrated values, units and standard names, with fill values masked."""
    # NaN in the counts stays NaN in the expected values, so the masking of fill values is checked too
    counts = landsat_files.counts_with_nan(case.dataset)
    expected = case.expected(counts, product.calibration.get(case.dataset))

    # The calibration code runs when the dataset is loaded, so this is where a DeprecationWarning is raised
    with _expect_deprecation(case.deprecated):
        data_arr = landsat_files.load([case.dataset], calibration=case.calibration)[case.dataset]

    assert (data_arr.dtype, data_arr.attrs["units"], data_arr.attrs["standard_name"]) == (
        np.float32, case.units, case.standard_name)
    # absolute tolerance for the float32 rounding of values close to zero
    np.testing.assert_allclose(data_arr.values, expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize(("product", "angle"), product_case_params(_angle_cases), indirect=["product"])
def test_angles(angle, landsat_files):
    """Test the angles are scaled to degrees."""
    # Angles have no calibration, the files store hundredths of a degree
    data_arr = landsat_files.load([angle])[angle]

    assert (data_arr.attrs["units"], data_arr.attrs["standard_name"]) == ("degrees", ANGLES[angle])
    np.testing.assert_allclose(data_arr.values, landsat_files.counts_with_nan(angle) * 0.01, rtol=1e-5)


@pytest.mark.parametrize("product", product_params(), indirect=True)
def test_uncalibrated_datasets(landsat_files):
    """Test the QA and gap mask datasets are returned as stored in the file, with fill values masked."""
    names = landsat_files.uncalibrated_names
    scn = landsat_files.load(names)

    for name in names:
        np.testing.assert_array_equal(scn[name].values, landsat_files.counts_with_nan(name), err_msg=name)


# ----------------------------------------------------
# Metadata and area tests ----------------------------
# ----------------------------------------------------
# Values read from the metadata file, and the areas built from it: UTM, panchromatic and polar stereographic.

@pytest.mark.parametrize("product", product_params(), indirect=True)
def test_metadata(product, landsat_files):
    """Test the values read from the metadata file."""
    mda = landsat_files.metadata_handler()

    assert (mda.platform_name, mda.earth_sun_distance(), mda.cloud_cover) == (
        product.platform_name, product.earth_sun_distance, product.cloud_cover)
    assert mda.band_saturation == product.saturation
    assert product.calibration.items() <= mda.band_calibration.items()


@pytest.mark.parametrize("product", product_params(), indirect=True)
def test_area_def(product, landsat_files):
    """Test the area of the bands built from the metadata file."""
    # The panchromatic band B8 is covered by test_pan_band
    assert landsat_files.metadata_handler().build_area_def(SPECTRAL) == product.area


@pytest.mark.parametrize("product", product_params("pan_extent"), indirect=True)
def test_pan_band(product, landsat_files):
    """Test the panchromatic band gets its own, finer area."""
    data_arr = landsat_files.load(["B8"], calibration="counts")["B8"]

    assert (data_arr.shape, data_arr.attrs["area"]) == (PAN_SHAPE, product.pan_area)


@pytest.mark.parametrize(("true_scale_lat", "epsg"), [("-71.00000", 3031), ("71.00000", 3995)])
def test_polar_area_def(tmp_path, true_scale_lat, epsg):
    """Test the polar stereographic areas used instead of UTM in the Antarctic and the Arctic."""
    # The reader picks the projection from TRUE_SCALE_LAT. There is no Arctic test file, so it is made from the
    # Antarctic one by changing the sign of the latitude, which keeps the extents the same.
    mtl = tmp_path / "LC08_L1GT_132122_20220301_20220308_02_T2_MTL.xml"
    with open(os.path.join(ETC_DIR, "antarctic_metadata.xml")) as antarctic_mtl:
        mtl.write_text(antarctic_mtl.read().replace("-71.00000", true_scale_lat))
    mda = LandsatL1MDReader(os.fspath(mtl), _filename_info("oli_tirs_l1_tif", mtl), {})

    areas = {name: mda.build_area_def(name) for name in ("B4", "B8")}
    assert {name: (area.crs.to_epsg(), area.area_extent) for name, area in areas.items()} == {
        "B4": (epsg, (-138315.0, 590385.0, 61215.0, 808815.0)),
        "B8": (epsg, (-138307.5, 590392.5, 61207.5, 808807.5))}


# ----------------------------------------------------
# File handler error tests ---------------------------
# ----------------------------------------------------
# The errors raised by the file handlers, created directly without a Scene: files of other platforms, invalid file
# types, and channels which are not in the file.

@pytest.mark.parametrize("product", product_params(), indirect=True)
def test_metadata_handler_rejects_non_landsat(landsat_files):
    """Test the metadata file handler refuses files of other platforms."""
    with pytest.raises(ValueError, match="This reader only supports Landsat data"):
        landsat_files.metadata_handler(platform_type="B")


@pytest.mark.parametrize(("handler_kwargs", "match"), [
    ({"platform_type": "B"}, "This reader only supports Landsat data"),
    ({"file_type": "granule-b05"}, "Invalid file type: granule-b05"),
])
@pytest.mark.parametrize("product", product_params(), indirect=True)
def test_channel_handler_rejects_bad_files(landsat_files, handler_kwargs, match):
    """Test the channel file handler refuses files of other platforms and invalid file types."""
    with pytest.raises(ValueError, match=match):
        landsat_files.channel_handler(SPECTRAL, **handler_kwargs)


@pytest.mark.parametrize("product", product_params(), indirect=True)
def test_channel_handler_rejects_other_channel(landsat_files):
    """Test the channel file handler refuses to load a band from the file of another band."""
    handler = landsat_files.channel_handler(SPECTRAL)

    with pytest.raises(ValueError, match=f"Requested channel B5 does not match the reader channel {SPECTRAL}"):
        handler.get_dataset({"name": "B5", "calibration": "counts"}, {})


@pytest.mark.parametrize("product", product_params("is_oli_tirs"), indirect=True)
def test_oli_tirs_granule_content(product, landsat_files):
    """Test OLI-only granules (O) have no thermal bands and TIRS-only granules (T) have no spectral bands."""
    # The granule content is the second letter of the filename, e.g. "C" (combined) in LC08_...
    for data_type, name in (("T", SPECTRAL), ("O", product.thermal)):
        handler = landsat_files.channel_handler(name, data_type=data_type)
        with pytest.raises(ValueError, match=f"Requested channel {name} is not available in this granule"):
            handler.get_dataset({"name": name, "calibration": "counts"}, {})


# ----------------------------------------------------
# MSS tests ------------------------------------------
# ----------------------------------------------------
# The MSS bands, whose wavelengths depend on the platform, and the MSS file handler's ``available_datasets``.

@pytest.mark.parametrize("product", product_params("wavelengths"), indirect=True)
def test_platform_dependent_wavelengths(product, landsat_files):
    """Test the MSS bands get the wavelengths of the platform and can be loaded by their central wavelength."""
    # B4 is green on Landsat-1 to 3 but near infrared on Landsat-4 and 5, while the YAML file has the green one
    scn = Scene(reader=product.reader, filenames=landsat_files.select(*product.wavelengths))
    scn.load([central for _, central, _ in product.wavelengths.values()], calibration="counts")

    # WavelengthRange also holds the unit, only (min, central, max) are compared
    assert {data_id["name"]: tuple(data_id["wavelength"])[:3] for data_id in scn.keys()} == product.wavelengths


@pytest.mark.parametrize("product", product_params("wavelengths"), indirect=True)
def test_mss_available_datasets(product, landsat_files):
    """Test the MSS file handler claims only its own dataset and yields (availability, info) pairs (see #3300)."""
    other_band = next(name for name in product.wavelengths if name != SPECTRAL)
    own_info = {"name": other_band, "file_type": f"granule_{other_band}"}
    foreign_info = {"name": SPECTRAL, "file_type": f"granule_{SPECTRAL}"}
    handler = landsat_files.channel_handler(other_band)

    # (None, info): nobody claimed the dataset yet, the handler of its file sets it to available.
    # (False, info): another handler already decided, which must be kept.
    # Before #3300 the info of bands other than B4 was replaced by an (availability, info) tuple.
    configured = [(None, own_info), (None, foreign_info), (False, own_info)]
    assert list(handler.available_datasets(configured)) == [(True, own_info), (None, foreign_info), (False, own_info)]
