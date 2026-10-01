"""Unit tests for the calibration, space masking and area definition shared by the GMS VISSR readers."""

import dask.array as da
import numpy as np
import pytest

import satpy.readers.gms.gms_vissr_common as common
import satpy.readers.gms.gms_vissr_navigation as nav
from satpy.tests.utils import make_dataid


class TestCalibrator:
    """Test calibration by table lookup."""

    @pytest.fixture
    def counts(self):
        """Get counts as a dask array."""
        return da.from_array(np.array([[0, 1], [2, 3]], dtype=np.uint8), chunks=1)

    @pytest.fixture
    def table(self):
        """Get a lookup table mapping count N to N / 10."""
        return np.arange(256, dtype=np.float32) / 10

    def test_counts_unchanged(self, counts, table):
        """Test that counts are returned as is."""
        res = common.Calibrator(table).calibrate(counts, "counts")
        assert res is counts

    def test_lookup(self, counts, table):
        """Test lookup without percent conversion."""
        res = common.Calibrator(table).calibrate(counts, "brightness_temperature")
        np.testing.assert_allclose(res.compute(), [[0, 0.1], [0.2, 0.3]], rtol=1e-6)
        assert res.dtype == np.float32

    def test_percent_conversion(self, counts, table):
        """Test conversion to percent for reflectance-like calibrations."""
        res = common.Calibrator(table).calibrate(counts, "unnormalized_reflectance")
        np.testing.assert_allclose(res.compute(), [[0, 10], [20, 30]], rtol=1e-6)

    def test_custom_percent_calibrations(self, counts, table):
        """Test that readers can declare additional calibrations as percent."""
        cal = common.Calibrator(table, percent_calibrations=("reflectance",))
        res = cal.calibrate(counts, "reflectance")
        np.testing.assert_allclose(res.compute(), [[0, 10], [20, 30]], rtol=1e-6)

    def test_mask(self, table):
        """Test that the mask is applied before the lookup (6-bit counts)."""
        counts = da.from_array(np.array([[0b11000001, 0b00000010]], dtype=np.uint8), chunks=1)
        res = common.Calibrator(table, mask=0x3F).calibrate(counts, "brightness_temperature")
        np.testing.assert_allclose(res.compute(), [[0.1, 0.2]], rtol=1e-6)


class TestEarthMask:
    """Test getting the earth mask."""

    def test_get_earth_mask(self):
        """Test getting the earth mask."""
        edges = np.array([-1, 1, 0, -1]), np.array([-1, 3, 2, -1])
        mask_exp = np.array(
            [[0, 0, 0, 0],
             [0, 1, 1, 1],
             [1, 1, 1, 0],
             [0, 0, 0, 0]],
            dtype=bool
        )
        mask = common.get_earth_mask(mask_exp.shape, edges)
        np.testing.assert_array_equal(mask, mask_exp)

    def test_fill_value_on_one_side_only(self):
        """Test that a scanline with only one fill value edge is fully masked."""
        mask = common.get_earth_mask((1, 4), (np.array([-1]), np.array([2])))
        assert not mask.any()

    def test_edges_outside_image_are_clamped(self):
        """Test that edges beyond the image border don't break the mask."""
        mask = common.get_earth_mask((2, 4), (np.array([0, 10]), np.array([10, 12])))
        np.testing.assert_array_equal(mask, [[1, 1, 1, 1], [0, 0, 0, 0]])

    def test_scale_earth_edges(self):
        """Test that edges are scaled but fill values are preserved."""
        edges = np.array([-1, 2, 5], dtype=np.int32)
        res = common.scale_earth_edges(edges, 4)
        np.testing.assert_array_equal(res, [-1, 8, 20])
        assert res.dtype == np.int32

    def test_scale_earth_edges_fractional_ratio(self):
        """Test scaling with a non-integer ratio truncates towards zero."""
        res = common.scale_earth_edges(np.array([-1, 3]), 1.9)
        np.testing.assert_array_equal(res, [-1, 5])


class TestAreaDefEstimator:
    """Test estimating a full disk area definition with uniform sampling."""

    @pytest.fixture
    def estimator(self):
        """Get an estimator with nominal satellite parameters."""
        return common.AreaDefEstimator(
            platform_name="GMS-5", sensor_name="VISSR", ssp_lon=140.0, satellite_height=35785831.0
        )

    @pytest.fixture
    def area_def(self, estimator):
        """Get an area definition."""
        dataset_id = make_dataid(name="IR1", resolution=5000)
        return estimator.get_area_def_uniform_sampling(dataset_id, size=100, stepping_angle=1.4e-4)

    def test_naming(self, area_def):
        """Test that the area is named after platform, service and resolution."""
        assert area_def.area_id == "gms-5_vissr_western-pacific_5km"
        assert area_def.description == "GMS-5 VISSR Western Pacific area definition with 5 km resolution"

    def test_shape_is_square(self, area_def):
        """Test that the requested size is used for both dimensions."""
        assert area_def.shape == (100, 100)

    def test_projection(self, area_def):
        """Test that the nominal satellite position and the earth ellipsoid are used."""
        proj_dict = area_def.crs.to_dict()
        assert proj_dict["lon_0"] == 140.0
        assert proj_dict["h"] == 35785831.0
        assert area_def.crs.ellipsoid.semi_major_metre == pytest.approx(nav.EARTH_EQUATORIAL_RADIUS)
        assert area_def.crs.ellipsoid.semi_minor_metre == pytest.approx(nav.EARTH_POLAR_RADIUS)

    def test_centered_on_subsatellite_point(self, area_def):
        """Test that the area is centered on the nominal sub-satellite point (within one pixel)."""
        x_min, y_min, x_max, y_max = area_def.area_extent
        assert abs(x_min + x_max) <= area_def.pixel_size_x * 1.01
        assert abs(y_min + y_max) <= area_def.pixel_size_y * 1.01

    def test_uniform_sampling(self, area_def):
        """Test that pixel size is the same along lines and pixels."""
        assert area_def.pixel_size_x == pytest.approx(area_def.pixel_size_y)

    def test_stepping_angle_determines_extent(self, estimator):
        """Test that a smaller stepping angle yields a proportionally smaller area."""
        dataset_id = make_dataid(name="VIS", resolution=1250)
        coarse = estimator.get_area_def_uniform_sampling(dataset_id, size=100, stepping_angle=1.4e-4)
        fine = estimator.get_area_def_uniform_sampling(dataset_id, size=100, stepping_angle=3.5e-5)
        assert fine.area_extent[2] == pytest.approx(coarse.area_extent[2] / 4, rel=1e-6)
