"""Tests for the CREFL ReflectanceCorrector modifier."""
import datetime as dt
from contextlib import contextmanager
from unittest import mock

import numpy as np
import pytest
import xarray as xr
from dask import array as da
from pyresample.geometry import AreaDefinition

from satpy.tests.utils import assert_maximum_dask_computes

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmpdir


@contextmanager
def mock_cmgdem(tmpdir, url):
    """Create fake file representing CMGDEM.hdf."""
    yield from _mock_and_create_dem_file(tmpdir, url, "averaged elevation", fill_value=-9999)


@contextmanager
def mock_tbase(tmpdir, url):
    """Create fake file representing tbase.hdf."""
    yield from _mock_and_create_dem_file(tmpdir, url, "Elevation")


def _mock_and_create_dem_file(tmpdir, url, var_name, fill_value=None):
    if not url:
        yield None
        return

    rmock_obj, dem_fn = _mock_dem_retrieve(tmpdir, url)
    _create_fake_dem_file(dem_fn, var_name, fill_value)

    try:
        yield rmock_obj
    finally:
        rmock_obj.stop()


def _mock_dem_retrieve(tmpdir, url):
    rmock_obj = mock.patch("satpy.modifiers._crefl.retrieve")
    rmock = rmock_obj.start()
    dem_fn = str(tmpdir.join(url))
    rmock.return_value = dem_fn
    return rmock_obj, dem_fn


def _create_fake_dem_file(dem_fn, var_name, fill_value):
    from pyhdf.SD import SD, SDC
    h = SD(dem_fn, SDC.WRITE | SDC.CREATE)
    dem_var = h.create(var_name, SDC.INT16, (10, 10))
    dem_var[:] = np.zeros((10, 10), dtype=np.int16)
    if fill_value is not None:
        dem_var.setfillvalue(fill_value)
    h.end()


def _make_viirs_xarray(data, area, name, standard_name, wavelength=None, units="degrees", calibration=None,
                       sensor="viirs", platform_name="Suomi-NPP", resolution=371):
    return xr.DataArray(data, dims=("y", "x"),
                        attrs={
                            "start_orbit": 1708, "end_orbit": 1708, "wavelength": wavelength,
                            "modifiers": None, "calibration": calibration,
                            "resolution": resolution, "name": name,
                            "standard_name": standard_name, "platform_name": platform_name,
                            "polarization": None, "sensor": sensor, "units": units,
                            "start_time": dt.datetime(2012, 2, 25, 18, 1, 24, 570942),
                            "end_time": dt.datetime(2012, 2, 25, 18, 11, 21, 175760), "area": area,
                            "ancillary_variables": []
                        })


def _create_ref_cor(**kwargs):
    """Create a ReflectanceCorrector with the standard angle prerequisites."""
    from satpy.modifiers._crefl import ReflectanceCorrector
    from satpy.tests.utils import make_dsq
    return ReflectanceCorrector(
        optional_prerequisites=[
            make_dsq(name="satellite_azimuth_angle"),
            make_dsq(name="satellite_zenith_angle"),
            make_dsq(name="solar_azimuth_angle"),
            make_dsq(name="solar_zenith_angle")
        ],
        prerequisites=[],
        calibration="reflectance",
        **kwargs)


def _make_band_and_angles(data, area, sensor, platform_name, name, wavelength, resolution):
    """Create a reflectance band and the four angle arrays needed by the modifier."""
    kwargs = {"sensor": sensor, "platform_name": platform_name, "resolution": resolution}
    band = _make_viirs_xarray(data, area, name, "toa_bidirectional_reflectance",
                              wavelength=wavelength, units="%", calibration="reflectance", **kwargs)
    angles = [
        _make_viirs_xarray(data, area, "satellite_azimuth_angle", "sensor_azimuth_angle", **kwargs),
        _make_viirs_xarray(data, area, "satellite_zenith_angle", "sensor_zenith_angle", **kwargs),
        _make_viirs_xarray(data, area, "solar_azimuth_angle", "solar_azimuth_angle", **kwargs),
        _make_viirs_xarray(data, area, "solar_zenith_angle", "solar_zenith_angle", **kwargs),
    ]
    return band, angles


class TestReflectanceCorrectorModifier:
    """Test the CREFL modifier."""

    @staticmethod
    def data_area_ref_corrector():
        """Create test area definition and data."""
        rows = 3
        cols = 5
        area = AreaDefinition(
            "some_area_name", "On-the-fly area", "geosabii",
            {"a": "6378137.0", "b": "6356752.31414", "h": "35786023.0", "lon_0": "-89.5", "proj": "geos", "sweep": "x",
             "units": "m"},
            cols, rows,
            (-5434894.954752679, -5434894.964451744, 5434894.964451744, 5434894.954752679))

        data = np.zeros((rows, cols)) + 25
        data[1, :] += 25
        data[2, :] += 50
        data = da.from_array(data, chunks=2)
        return area, data

    @pytest.mark.parametrize(
        ("name", "wavelength", "resolution", "exp_mean", "exp_unique"),
        [
            ("C01", (0.45, 0.47, 0.49), 1000, 44.757951,
             np.array([12.83774603, 14.38767557, 17.24258084, 41.87806142, 44.42472192, 47.89958451,
                       48.23343427, 48.53847386, 71.52916035, 72.26078684, 73.10523784])),
            ("C02", (0.59, 0.64, 0.69), 500, 51.4901,
             np.array([23.69999579, 24.00407203, 24.49390685, 51.4304448, 51.64271324, 51.70519738,
                       51.70942859, 51.76064747, 78.37182815, 78.77078522, 78.80199923])),
            ("C03", (0.8455, 0.865, 0.8845), 1000, 50.7243,
             np.array([24.78444631, 24.86790679, 24.99481254, 50.69670516, 50.72983327, 50.73601728,
                       50.75685498, 50.83136276, 76.39973287, 76.5714688, 76.59856607])),
            # ("C04", (1.3705, 1.378, 1.3855), 2000, 55.973458829136796, None),
            ("C05", (1.58, 1.61, 1.64), 1000, 52.7231,
             np.array([26.26568157, 26.43230852, 26.48936244, 52.00527783, 52.13043172, 52.20176747,
                       53.01505657, 53.29017112, 78.93907987, 79.49089239, 79.69387535])),
            ("C06", (2.225, 2.25, 2.275), 2000, 55.9735,
             np.array([27.82291562, 28.2268102, 28.37246323, 54.33639308, 54.61451818, 54.77543748,
                       56.62284858, 57.27288821, 83.57235975, 84.81324822, 85.27816457])),
        ]
    )
    def test_reflectance_corrector_abi(self, name, wavelength, resolution, exp_mean, exp_unique):
        """Test ReflectanceCorrector modifier with ABI data."""
        from satpy.modifiers._crefl import ReflectanceCorrector
        from satpy.tests.utils import make_dsq
        ref_cor = ReflectanceCorrector(optional_prerequisites=[
            make_dsq(name="satellite_azimuth_angle"),
            make_dsq(name="satellite_zenith_angle"),
            make_dsq(name="solar_azimuth_angle"),
            make_dsq(name="solar_zenith_angle")], name=name, prerequisites=[],
                                       wavelength=wavelength, resolution=resolution, calibration="reflectance",
                                       modifiers=("sunz_corrected", "rayleigh_corrected_crefl",), sensor="abi")

        assert ref_cor.attrs["modifiers"] == ("sunz_corrected", "rayleigh_corrected_crefl")
        assert ref_cor.attrs["calibration"] == "reflectance"
        assert ref_cor.attrs["wavelength"] == wavelength
        assert ref_cor.attrs["name"] == name
        assert ref_cor.attrs["resolution"] == resolution
        assert ref_cor.attrs["sensor"] == "abi"
        assert ref_cor.attrs["prerequisites"] == []
        assert ref_cor.attrs["optional_prerequisites"] == [
            make_dsq(name="satellite_azimuth_angle"),
            make_dsq(name="satellite_zenith_angle"),
            make_dsq(name="solar_azimuth_angle"),
            make_dsq(name="solar_zenith_angle")]

        area, dnb = self.data_area_ref_corrector()
        c01 = xr.DataArray(dnb,
                           dims=("y", "x"),
                           attrs={
                               "platform_name": "GOES-16",
                               "calibration": "reflectance", "units": "%", "wavelength": wavelength,
                               "name": name, "resolution": resolution, "sensor": "abi",
                               "start_time": "2017-09-20 17:30:40.800000", "end_time": "2017-09-20 17:41:17.500000",
                               "area": area, "ancillary_variables": [],
                               "orbital_parameters": {
                                   "satellite_nominal_longitude": -89.5,
                                   "satellite_nominal_latitude": 0.0,
                                   "satellite_nominal_altitude": 35786023.4375,
                               },
                           })
        with assert_maximum_dask_computes(0):
            res = ref_cor([c01], [])

        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs["modifiers"] == ("sunz_corrected", "rayleigh_corrected_crefl")
        assert res.attrs["platform_name"] == "GOES-16"
        assert res.attrs["calibration"] == "reflectance"
        assert res.attrs["units"] == "%"
        assert res.attrs["wavelength"] == wavelength
        assert res.attrs["name"] == name
        assert res.attrs["resolution"] == resolution
        assert res.attrs["sensor"] == "abi"
        assert res.attrs["start_time"] == "2017-09-20 17:30:40.800000"
        assert res.attrs["end_time"] == "2017-09-20 17:41:17.500000"
        assert res.attrs["area"] == area
        assert res.attrs["ancillary_variables"] == []
        data = res.values
        unique = np.unique(data[~np.isnan(data)])
        np.testing.assert_allclose(np.nanmean(data), exp_mean, rtol=1e-5)
        assert data.shape == (3, 5)
        np.testing.assert_allclose(unique, exp_unique, rtol=1e-5)

    @pytest.mark.parametrize(
        ("url", "dem_mock_cm", "dem_sds"),
        [
            (None, mock_cmgdem, "average elevation"),
            ("CMGDEM.hdf", mock_cmgdem, "averaged elevation"),
            ("tbase.hdf", mock_tbase, "Elevation"),
        ])
    @pytest.mark.parametrize(
        ("sensor", "platform_name", "name", "wavelength", "resolution", "modifiers", "exp_mean", "exp_unique"),
        [
            pytest.param("viirs", "Suomi-NPP", "I01", (0.6, 0.64, 0.68), 371,
                         ("sunz_corrected_iband", "rayleigh_corrected_crefl_iband"),
                         51.12750267805715, [25.20341703, 52.38819447, 75.79089654], id="viirs"),
            pytest.param("modis", "EOS-Aqua", "1", (0.62, 0.645, 0.67), 500,
                         ("sunz_corrected", "rayleigh_corrected_crefl"),
                         52.09372623964498, [25.43670075, 52.93221561, 77.91226236], id="modis"),
            # METimage borrows the MODIS coefficients and the MODIS atmosphere
            # equations, so "vii_668" must produce exactly the same result as
            # MODIS band "1".
            pytest.param("metimage", "Metop-SG-A1", "vii_668", (0.658, 0.668, 0.678), 500,
                         ("sunz_corrected", "rayleigh_corrected_crefl"),
                         52.09372623964498, [25.43670075, 52.93221561, 77.91226236], id="metimage"),
        ])
    def test_reflectance_corrector_polar(self, tmpdir, sensor, platform_name, name, wavelength, resolution,
                                         modifiers, exp_mean, exp_unique, url, dem_mock_cm, dem_sds):
        """Test ReflectanceCorrector modifier with polar-orbiter data (VIIRS, MODIS, METimage)."""
        from satpy.tests.utils import make_dsq

        ref_cor = _create_ref_cor(name=name, wavelength=wavelength, resolution=resolution, modifiers=modifiers,
                                  sensor=sensor, url=url, dem_sds=dem_sds)

        assert ref_cor.attrs["modifiers"] == modifiers
        assert ref_cor.attrs["calibration"] == "reflectance"
        assert ref_cor.attrs["wavelength"] == wavelength
        assert ref_cor.attrs["name"] == name
        assert ref_cor.attrs["resolution"] == resolution
        assert ref_cor.attrs["sensor"] == sensor
        assert ref_cor.attrs["prerequisites"] == []
        assert ref_cor.attrs["optional_prerequisites"] == [
            make_dsq(name="satellite_azimuth_angle"),
            make_dsq(name="satellite_zenith_angle"),
            make_dsq(name="solar_azimuth_angle"),
            make_dsq(name="solar_zenith_angle")]

        area, data = self.data_area_ref_corrector()
        c01, angles = _make_band_and_angles(data, area, sensor, platform_name, name, wavelength, resolution)

        with dem_mock_cm(tmpdir, url), assert_maximum_dask_computes(0):
            res = ref_cor([c01], angles)

        assert isinstance(res, xr.DataArray)
        assert isinstance(res.data, da.Array)
        assert res.attrs["wavelength"] == wavelength
        assert res.attrs["modifiers"] == modifiers
        assert res.attrs["calibration"] == "reflectance"
        assert res.attrs["resolution"] == resolution
        assert res.attrs["name"] == name
        assert res.attrs["standard_name"] == "toa_bidirectional_reflectance"
        assert res.attrs["platform_name"] == platform_name
        assert res.attrs["sensor"] == sensor
        assert res.attrs["units"] == "%"
        assert res.attrs["start_time"] == dt.datetime(2012, 2, 25, 18, 1, 24, 570942)
        assert res.attrs["end_time"] == dt.datetime(2012, 2, 25, 18, 11, 21, 175760)
        assert res.attrs["area"] == area
        assert res.attrs["ancillary_variables"] == []
        data = res.values
        assert data.shape == (3, 5)
        np.testing.assert_allclose(np.mean(data), exp_mean, rtol=1e-8)
        np.testing.assert_allclose(np.unique(data), exp_unique)

    def test_reflectance_corrector_bad_prereqs(self):
        """Test ReflectanceCorrector modifier with wrong number of inputs."""
        from satpy.modifiers._crefl import ReflectanceCorrector
        ref_cor = ReflectanceCorrector("test")
        pytest.raises(ValueError, ref_cor, [1], [2, 3, 4], match="Not sure how to handle provided dependencies..*")
        pytest.raises(ValueError, ref_cor, [1, 2, 3, 4], [], match="Not sure how to handle provided dependencies..*")
        pytest.raises(ValueError, ref_cor, [], [1, 2, 3, 4], match="Not sure how to handle provided dependencies..*")


    @pytest.mark.parametrize(
        ("url", "dem_mock_cm", "dem_sds"),
        [
            (None, mock_cmgdem, "average elevation"),
            ("CMGDEM.hdf", mock_cmgdem, "averaged elevation"),
            ("tbase.hdf", mock_tbase, "Elevation"),
        ])
    def test_reflectance_corrector_different_chunks(self, tmpdir, url, dem_mock_cm, dem_sds):
        """Test that the modifier works with different chunk sizes for inputs.

        The modifier uses dask's "map_blocks". If the input chunks aren't the
        same an error is raised.

        """
        ref_cor = _create_ref_cor(name="I01", wavelength=(0.6, 0.64, 0.68), resolution=371,
                                  modifiers=("sunz_corrected_iband", "rayleigh_corrected_crefl_iband"),
                                  sensor="viirs", url=url, dem_sds=dem_sds)

        area, data = self.data_area_ref_corrector()
        c01, angles = _make_band_and_angles(data, area, "viirs", "Suomi-NPP", "I01", (0.6, 0.64, 0.68), 371)
        angles[0].data = angles[0].data.rechunk((1, -1))

        with dem_mock_cm(tmpdir, url):
            res = ref_cor([c01], angles)

        # make sure it can actually compute
        res.compute()
