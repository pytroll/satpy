"""Unit testing the enhancements functions, e.g. cira_stretch."""

import os
from typing import NamedTuple
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def fake_area():
    """Return a fake 2×2 area."""
    from pyresample.geometry import create_area_def
    return create_area_def("wingertsberg", 4087, area_extent=[-2_000, -2_000, 2_000, 2_000], shape=(2, 2))


class _NWCSAFCase(NamedTuple):
    """Properties of a fake NWCSAF product used to test its composite and enhancement."""

    flavor: str
    varname: str
    file_varname: str | None  # name in the file if it differs from varname
    palette_name: str
    status_name: str | None
    composite: str
    file_label: str
    dtype: str

    @property
    def valid_range(self):
        """Get the valid range of the fake data."""
        return (0, 100) if self.dtype == "uint8" else (-100, 1000)


_nwcsaf_geo_props = {
    "cma_geo": ("geo", "cma", None, "cma_pal", None, "cloudmask", "CMA", "uint8"),
    "cma_pps": ("pps", "cma", None, "cma_pal", None, "cloudmask", "CMA", "uint8"),
    "cma_extended_pps": ("pps", "cma_extended", None, "cma_extended_pal", None,
                         "cloudmask_extended", "CMA", "uint8"),
    "cmaprob_pps": ("pps", "cmaprob", None, "cmaprob_pal", None, "cloudmask_probability",
                    "CMAPROB", "uint8"),
    "ct_geo": ("geo", "ct", None, "ct_pal", None, "cloudtype", "CT", "uint8"),
    "ct_pps": ("pps", "ct", None, "ct_pal", None, "cloudtype", "CT", "uint8"),
    "ctth_alti_geo": ("geo", "ctth_alti", None, "ctth_alti_pal", None, "cloud_top_height",
                      "CTTH", "float64"),
    "ctth_alti_pps": ("pps", "ctth_alti", None, "ctth_alti_pal", "ctth_status_flag",
                      "cloud_top_height", "CTTH", "float64"),
    "ctth_pres_geo": ("geo", "ctth_pres", None, "ctth_pres_pal", None, "cloud_top_pressure",
                      "CTTH", "float64"),
    "ctth_pres_pps": ("pps", "ctth_pres", None, "ctth_pres_pal", None, "cloud_top_pressure",
                      "CTTH", "float64"),
    "ctth_tempe_geo": ("geo", "ctth_tempe", None, "ctth_tempe_pal", None, "cloud_top_temperature",
                       "CTTH", "float64"),
    "ctth_tempe_pps": ("pps", "ctth_tempe", None, "ctth_tempe_pal", None, "cloud_top_temperature",
                       "CTTH", "float64"),
    "cmic_phase_geo": ("geo", "cmic_phase", None, "cmic_phase_pal", None, "cloud_top_phase",
                       "CMIC", "uint8"),
    "cmic_phase_pps": ("pps", "cmic_phase", None, "cmic_phase_pal", "cmic_status_flag", "cloud_top_phase",
                       "CMIC", "uint8"),
    "cmic_reff_geo": ("geo", "cmic_reff", None, "cmic_reff_pal", None, "cloud_drop_effective_radius",
                      "CMIC", "float64"),
    "cmic_reff_pps": ("pps", "cmic_reff", "cmic_cre", "cmic_cre_pal", "cmic_status_flag",
                      "cloud_drop_effective_radius", "CMIC", "float64"),
    "cmic_cot_geo": ("geo", "cmic_cot", None, "cmic_cot_pal", None, "cloud_optical_thickness",
                     "CMIC", "float64"),
    "cmic_cot_pps": ("pps", "cmic_cot", None, "cmic_cot_pal", None, "cloud_optical_thickness",
                     "CMIC", "float64"),
    "cmic_cwp_pps": ("pps", "cmic_cwp", None, "cmic_cwp_pal", None, "cloud_water_path",
                     "CMIC", "float64"),
    "cmic_lwp_geo": ("geo", "cmic_lwp", None, "cmic_lwp_pal", None, "cloud_liquid_water_path",
                     "CMIC", "float64"),
    "cmic_lwp_pps": ("pps", "cmic_lwp", None, "cmic_lwp_pal", None, "liquid_water_path",
                     "CMIC", "float64"),
    "cmic_iwp_geo": ("geo", "cmic_iwp", None, "cmic_iwp_pal", None, "cloud_ice_water_path",
                     "CMIC", "float64"),
    "cmic_iwp_pps": ("pps", "cmic_iwp", None, "cmic_iwp_pal", None, "ice_water_path",
                     "CMIC", "float64"),
    "pc": ("geo", "pc", None, "pc_pal", None, "precipitation_probability", "PC", "uint8"),
    "crr": ("geo", "crr", None, "crr_pal", None, "convective_rain_rate", "CRR", "uint8"),
    "crr_accum": ("geo", "crr_accum", None, "crr_pal", None,
                  "convective_precipitation_hourly_accumulation", "CRR", "uint8"),
    "ishai_tpw": ("geo", "ishai_tpw", None, "ishai_tpw_pal", None, "total_precipitable_water",
                  "iSHAI", "float64"),
    "ishai_shw": ("geo", "ishai_shw", None, "ishai_shw_pal", None, "showalter_index",
                  "iSHAI", "float64"),
    "ishai_li": ("geo", "ishai_li", None, "ishai_li_pal", None, "lifted_index",
                 "iSHAI", "float64"),
    "ci_prob30": ("geo", "ci_prob30", None, "ci_pal", None, "convection_initiation_prob30",
                  "CI", "float64"),
    "ci_prob60": ("geo", "ci_prob60", None, "ci_pal", None, "convection_initiation_prob60",
                  "CI", "float64"),
    "ci_prob90": ("geo", "ci_prob90", None, "ci_pal", None, "convection_initiation_prob90",
                  "CI", "float64"),
    "asii_turb_trop_prob": ("geo", "asii_turb_trop_prob", None, "asii_turb_prob_pal", None,
                            "asii_prob", "ASII-NG", "float64"),
    "MapCellCatType": ("geo", "MapCellCatType", None, "MapCellCatType_pal", None,
                       "rdt_cell_type", "RDT-CW", "uint8"),
}

_nwcsaf_colorized = {"ctth_alti_pps", "cmic_reff_pps", "cmic_cot_pps", "cmic_cwp_pps",
                     "cmic_lwp_pps", "cmic_iwp_pps"}


@pytest.mark.parametrize("data", _nwcsaf_geo_props.keys())
def test_nwcsaf_comps(fake_area, tmp_path, data):
    """Test loading NWCSAF composites."""
    from satpy.enhancements.enhancer import get_enhanced_image
    case = _NWCSAFCase(*_nwcsaf_geo_props[data])
    sc = _create_fake_nwcsaf_scene(tmp_path, case)
    fake_data = _add_fake_nwcsaf_datasets(sc, case, fake_area)

    def _fake_get_varname(info, info_type="file_key"):
        return case.file_varname or case.varname

    with mock.patch("satpy.readers.nwcsaf_nc.NcNWCSAF._get_varname_in_file") as srnN_:
        srnN_.side_effect = _fake_get_varname
        sc.load([case.composite])
    im = get_enhanced_image(sc[case.composite])
    if data in _nwcsaf_colorized:
        _assert_nwcsaf_colorized(im, case, fake_data)
    else:
        _assert_nwcsaf_palettized(im, case, fake_data)


def _create_fake_nwcsaf_scene(tmp_path, case):
    """Create a Scene from a minimal fake NWCSAF file, otherwise satpy won't load the composite."""
    from satpy import Scene
    if case.flavor == "geo":
        fn = f"S_NWC_{case.file_label:s}_MSG2_MSG-N-VISIR_20220124T094500Z.nc"
        reader = "nwcsaf-geo"
        id_ = {"satellite_identifier": "MSG4"}
    else:
        fn = f"S_NWC_{case.file_label:s}_noaa20_00000_20230301T1200213Z_20230301T1201458Z.nc"
        reader = "nwcsaf-pps_nc"
        id_ = {"platform": "NOAA-20"}
    fk = tmp_path / fn
    ds = xr.Dataset(
        coords={"nx": [0], "ny": [0]},
        attrs={
            "source": "satpy unit test",
            "time_coverage_start": "0001-01-01T00:00:00Z",
            "time_coverage_end": "0001-01-01T01:00:00Z",
            **id_,
        }
    )
    ds.to_netcdf(fk)
    return Scene(filenames=[os.fspath(fk)], reader=[reader])


def _add_fake_nwcsaf_datasets(sc, case, fake_area):
    """Add the fake data, palette and optional status flag datasets to the Scene.

    Returns the fake data array so the enhanced image can be compared against it.
    """
    vmin, vmax = case.valid_range
    fake_data = da.linspace(vmin, vmax, 4, chunks=2, dtype=case.dtype).reshape(2, 2)
    sc[case.palette_name] = xr.DataArray(
        da.tile(da.arange(256), [3, 1]).T,
        dims=("pal02_colors", "pal_RGB"))
    ancvars = [sc[case.palette_name]]
    if case.status_name is not None:
        sc[case.status_name] = xr.DataArray(
            da.zeros(shape=(2, 2), dtype="uint8"),
            attrs={
                "area": fake_area,
                "_FillValue": 123},
            dims=("y", "x"))
        ancvars.append(sc[case.status_name])
    sc[case.varname] = xr.DataArray(
        fake_data,
        dims=("y", "x"),
        attrs={
            "area": fake_area,
            "scaled_FillValue": 123,
            "ancillary_variables": ancvars,
            "valid_range": case.valid_range})
    return fake_data


def _assert_nwcsaf_colorized(im, case, fake_data):
    """Check a colorized image, where the fake gray ramp palette makes every band the normalized data."""
    vmin, vmax = case.valid_range
    assert im.mode == "RGB"
    expected = (fake_data - vmin) / (vmax - vmin)
    for band in "RGB":
        np.testing.assert_allclose(im.data.sel(bands=band), expected, rtol=1e-6)


def _assert_nwcsaf_palettized(im, case, fake_data):
    """Check a palettized image.

    Only the GEO values are checked as some PPS composites alter the data before palettizing.
    """
    assert im.mode == "P"
    np.testing.assert_array_equal(im.data.coords["bands"], ["P"])
    if case.flavor != "geo":
        return
    vmin, vmax = case.valid_range
    expected = ((fake_data - vmin) * (255 / (vmax - vmin))).round() if case.dtype == "float64" else fake_data
    np.testing.assert_allclose(im.data.sel(bands="P"), expected)


@pytest.mark.parametrize("name",
                         ["stretch",
                          "gamma",
                          "invert",
                          "piecewise_linear_stretch",
                          "cira_stretch",
                          "reinhard_to_srgb",
                          "btemp_threshold",
                         ]
                         )
def test_stretching_warns(name):
    """Test that there's a warning when importing stretching functions from old location."""
    from satpy import enhancements
    with pytest.warns(UserWarning, match="has been moved to"):
        getattr(enhancements, name)


def test_jma_true_color_repropdution_warns():
    """Test that there's a warning when importing jma_true_color_reproduction from old location."""
    with pytest.warns(UserWarning, match="has been moved to"):
        from satpy.enhancements import jma_true_color_reproduction  # noqa


def test_convolution_warns():
    """Test that there's a warning when importing three_d_effect from old location."""
    with pytest.warns(UserWarning, match="has been moved to"):
        from satpy.enhancements import three_d_effect  # noqa


@pytest.mark.parametrize("name",
                         ["exclude_alpha",
                          "on_separate_bands",
                          "using_map_blocks",
                         ]
                         )
def test_wrappers_warns(name):
    """Test that there's a warning when importing wrapper functions from old location."""
    from satpy import enhancements
    with pytest.warns(UserWarning, match="has been moved to"):
        getattr(enhancements, name)


@pytest.mark.parametrize("name",
                         ["lookup",
                          "colorize",
                          "palettize",
                          "create_colormap",
                         ]
                         )
def test_color_mapping_warns(name):
    """Test that there's a warning when importing color mapping functions from old location."""
    from satpy import enhancements
    with pytest.warns(UserWarning, match="has been moved to"):
        getattr(enhancements, name)
