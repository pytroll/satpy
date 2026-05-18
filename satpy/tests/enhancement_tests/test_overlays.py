# Copyright (c) 2025 Satpy developers
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
"""Tests for the enhancements overlays module."""
from __future__ import annotations

import warnings
from unittest import mock

import xarray as xr
from dask import array as da
from trollimage.colormap import greys


class TestOverlays:
    """Tests for add_overlay and add_decorate functions."""

    def setup_method(self):
        """Create test data and mock pycoast/pydecorate."""
        from pyresample.geometry import AreaDefinition
        from trollimage.xrimage import XRImage

        proj_dict = {"proj": "lcc", "datum": "WGS84", "ellps": "WGS84",
                     "lon_0": -95., "lat_0": 25, "lat_1": 25,
                     "units": "m", "no_defs": True}
        self.area_def = AreaDefinition(
            "test", "test", "test", proj_dict,
            200, 400, (-1000., -1500., 1000., 1500.),
        )
        self.orig_rgb_img = XRImage(
            xr.DataArray(da.arange(75., chunks=10).reshape(3, 5, 5) / 75.,
                         dims=("bands", "y", "x"),
                         coords={"bands": ["R", "G", "B"]},
                         attrs={"name": "test_ds", "area": self.area_def})
        )
        self.orig_l_img = XRImage(
            xr.DataArray(da.arange(25., chunks=10).reshape(5, 5) / 75.,
                         dims=("y", "x"),
                         attrs={"name": "test_ds", "area": self.area_def})
        )

        self.decorate = {
            "decorate": [
                {"logo": {"logo_path": "", "height": 143, "bg": "white", "bg_opacity": 255}},
                {"text": {
                    "txt": "TEST",
                    "align": {"top_bottom": "bottom", "left_right": "right"},
                    "font": "",
                    "font_size": 22,
                    "height": 30,
                    "bg": "black",
                    "bg_opacity": 255,
                    "line": "white"}},
                {"scale": {
                    "colormap": greys,
                    "extend": False,
                    "width": 1670, "height": 110,
                    "tick_marks": 5, "minor_tick_marks": 1,
                    "cursor": [0, 0], "bg": "white",
                    "title": "TEST TITLE OF SCALE",
                    "fontsize": 110, "align": "cc"
                }}
            ]
        }

        import_mock = mock.MagicMock()
        modules = {"pycoast": import_mock.pycoast,
                   "pydecorate": import_mock.pydecorate}
        self.module_patcher = mock.patch.dict("sys.modules", modules)
        self.module_patcher.start()

    def teardown_method(self):
        """Turn off pycoast/pydecorate mocking."""
        self.module_patcher.stop()

    def test_add_overlay_basic_rgb(self):
        """Test basic add_overlay usage with RGB data."""
        from pycoast import ContourWriterAGG

        from satpy.enhancements.overlays import _burn_overlay, add_overlay
        coast_dir = "/path/to/coast/data"
        with mock.patch.object(self.orig_rgb_img, "apply_pil") as apply_pil:
            apply_pil.return_value = self.orig_rgb_img
            new_img = add_overlay(self.orig_rgb_img, self.area_def, coast_dir, fill_value=0)
            assert self.orig_rgb_img.mode == new_img.mode
            new_img = add_overlay(self.orig_rgb_img, self.area_def, coast_dir)
            assert self.orig_rgb_img.mode + "A" == new_img.mode

            with mock.patch.object(self.orig_rgb_img, "convert") as convert:
                convert.return_value = self.orig_rgb_img
                overlays = {"coasts": {"outline": "red"}}
                new_img = add_overlay(self.orig_rgb_img, self.area_def, coast_dir,
                                      overlays=overlays, fill_value=0)
                pil_args = None
                pil_kwargs = {"fill_value": 0}
                fun_args = (self.orig_rgb_img.data.area, ContourWriterAGG.return_value, overlays)
                fun_kwargs = None
                apply_pil.assert_called_with(_burn_overlay, self.orig_rgb_img.mode,
                                             pil_args, pil_kwargs, fun_args, fun_kwargs)
                ContourWriterAGG.assert_called_with(coast_dir)

                # test legacy call

                grid = {"minor_is_tick": True}
                color = "red"
                expected_overlays = {"coasts": {"outline": color, "width": 0.5, "level": 1},
                                     "borders": {"outline": color, "width": 0.5, "level": 1},
                                     "grid": grid}
                with warnings.catch_warnings(record=True) as wns:
                    warnings.simplefilter("always")
                    new_img = add_overlay(self.orig_rgb_img, self.area_def, coast_dir,
                                          color=color, grid=grid, fill_value=0)
                    assert len(wns) == 1
                    assert issubclass(wns[0].category, DeprecationWarning)
                    assert "deprecated" in str(wns[0].message)

                pil_args = None
                pil_kwargs = {"fill_value": 0}
                fun_args = (self.orig_rgb_img.data.area, ContourWriterAGG.return_value, expected_overlays)
                fun_kwargs = None
                apply_pil.assert_called_with(_burn_overlay, self.orig_rgb_img.mode,
                                             pil_args, pil_kwargs, fun_args, fun_kwargs)
                ContourWriterAGG.assert_called_with(coast_dir)

    def test_add_overlay_basic_l(self):
        """Test basic add_overlay usage with L data."""
        from satpy.enhancements.overlays import add_overlay
        new_img = add_overlay(self.orig_l_img, self.area_def, "", fill_value=0)
        assert "RGB" == new_img.mode
        new_img = add_overlay(self.orig_l_img, self.area_def, "")
        assert "RGBA" == new_img.mode

    def test_add_decorate_basic_rgb(self):
        """Test basic add_decorate usage with RGB data."""
        from satpy.enhancements.overlays import add_decorate
        new_img = add_decorate(self.orig_rgb_img, **self.decorate)
        assert "RGBA" == new_img.mode

    def test_add_decorate_basic_l(self):
        """Test basic add_decorate usage with L data."""
        from satpy.enhancements.overlays import add_decorate
        new_img = add_decorate(self.orig_l_img, **self.decorate)
        assert "RGBA" == new_img.mode
