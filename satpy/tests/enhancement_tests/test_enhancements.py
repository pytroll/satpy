# Copyright (c) 2017-2025 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Unit testing the enhancements functions, e.g. cira_stretch."""

import os
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


@pytest.mark.parametrize(
    "data",
    ["cma_geo", "cma_pps", "cma_extended_pps", "cmaprob_pps", "ct_geo",
     "ct_pps", "ctth_alti_geo", "ctth_alti_pps", "ctth_pres_geo",
     "ctth_pres_pps", "ctth_tempe_geo", "ctth_tempe_pps",
     "cmic_phase_geo", "cmic_phase_pps", "cmic_reff_geo",
     "cmic_reff_pps", "cmic_cot_geo", "cmic_cot_pps", "cmic_cwp_pps",
     "cmic_lwp_geo", "cmic_lwp_pps", "cmic_iwp_geo", "cmic_iwp_pps",
     "pc", "crr", "crr_accum", "ishai_tpw", "ishai_shw", "ishai_li",
     "ci_prob30", "ci_prob60", "ci_prob90", "asii_turb_trop_prob",
     "MapCellCatType"]
)
def test_nwcsaf_comps(fake_area, tmp_path, data):
    """Test loading NWCSAF composites."""
    from satpy import Scene
    from satpy.enhancements.enhancer import get_enhanced_image
    (flavour, dvname, altname, palettename, statusname, comp, filelabel, dtp) = _nwcsaf_geo_props[data]
    rng = (0, 100) if dtp == "uint8" else (-100, 1000)
    if flavour == "geo":
        fn = f"S_NWC_{filelabel:s}_MSG2_MSG-N-VISIR_20220124T094500Z.nc"
        reader = "nwcsaf-geo"
        id_ = {"satellite_identifier": "MSG4"}
    else:
        fn = f"S_NWC_{filelabel:s}_noaa20_00000_20230301T1200213Z_20230301T1201458Z.nc"
        reader = "nwcsaf-pps_nc"
        id_ = {"platform": "NOAA-20"}
    fk = tmp_path / fn
    # create a minimally fake netCDF file, otherwise satpy won't load the
    # composite
    ds = xr.Dataset(
        coords={"nx": [0], "ny": [0]},
        attrs={
            "source": "satpy unit test",
            "time_coverage_start": "0001-01-01T00:00:00Z",
            "time_coverage_end": "0001-01-01T01:00:00Z",
        }
    )
    ds.attrs.update(id_)
    ds.to_netcdf(fk)
    sc = Scene(filenames=[os.fspath(fk)], reader=[reader])
    sc[palettename] = xr.DataArray(
        da.tile(da.arange(256), [3, 1]).T,
        dims=("pal02_colors", "pal_RGB"))
    fake_alti = da.linspace(rng[0], rng[1], 4, chunks=2, dtype=dtp).reshape(2, 2)
    ancvars = [sc[palettename]]
    if statusname is not None:
        sc[statusname] = xr.DataArray(
            da.zeros(shape=(2, 2), dtype="uint8"),
            attrs={
                "area": fake_area,
                "_FillValue": 123},
            dims=("y", "x"))
        ancvars.append(sc[statusname])
    sc[dvname] = xr.DataArray(
        fake_alti,
        dims=("y", "x"),
        attrs={
            "area": fake_area,
            "scaled_FillValue": 123,
            "ancillary_variables": ancvars,
            "valid_range": rng})

    def _fake_get_varname(info, info_type="file_key"):
        return altname or dvname

    with mock.patch("satpy.readers.nwcsaf_nc.NcNWCSAF._get_varname_in_file") as srnN_:
        srnN_.side_effect = _fake_get_varname
        sc.load([comp])
    im = get_enhanced_image(sc[comp])
    if flavour == "geo":
        assert im.mode == "P"
        np.testing.assert_array_equal(im.data.coords["bands"], ["P"])
        if dtp == "float64":
            np.testing.assert_allclose(
                im.data.sel(bands="P"),
                ((fake_alti - rng[0]) * (255 / np.ptp(rng))).round())
        else:
            np.testing.assert_allclose(im.data.sel(bands="P"), fake_alti)


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
