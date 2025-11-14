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
"""Unit tests for VectorScene."""

import os
import unittest.mock

import pytest

from satpy.readers.core.file_handlers import BaseFileHandler


class DummyFileHandler(BaseFileHandler):
    """Dummy file handler for testing vector datasets."""

    def get_dataset(self, data_id, data_info):
        """Dummy get dataset."""
        from geopandas import GeoDataFrame
        from shapely import Point
        return GeoDataFrame(
                {"col1": ["name1", "name2"],
                 "geometry": [Point(1, 2), Point(2, 1)]},
                crs="EPSG:4326")

dummy_config = f"""reader:
    name: fake_l99_dummy
    reader: !!python/name:satpy.readers.core.yaml_reader.VectorFileYAMLReader
    sensors: [dummy_vector_sensor]
file_types:
    dummy_vector_filetype:
        file_reader: !!python/name:{DummyFileHandler.__module__:s}.DummyFileHandler
        file_patterns: ["grenadines"]
datasets:
    dummy:
        name: dummy_vector_dataset
        file_type: dummy_vector_filetype
"""

def test_init():
    """Test VectorScene initialisation."""
    from satpy.vectorscene import VectorScene
    VectorScene()

def test_load(tmp_path):
    """Test loading some vector data."""
    import geopandas

    from satpy.vectorscene import VectorScene
    dummy_reader_path = tmp_path / "dummy_vector_reader.yaml"
    with dummy_reader_path.open("w") as fp:
        fp.write(dummy_config)
    with unittest.mock.patch("satpy.readers.core.loading.configs_for_reader") as srclc:
        srclc.return_value = [[os.fspath(dummy_reader_path)]]
        vs = VectorScene(filenames=["grenadines"], reader=["dummy_vector_reader"])
        vs.load(["dummy_vector_dataset"])
        assert isinstance(vs["dummy_vector_dataset"], geopandas.GeoDataFrame)

@pytest.fixture
def dummy_vector_scene():
    """Return a dummy vector scene with one dataset."""
    from geopandas import GeoDataFrame
    from shapely import Point

    from satpy.vectorscene import VectorScene
    vs = VectorScene()
    vs["dummy_vector_dataset"] = GeoDataFrame(
                {"col1": ["name1", "name2"],
                 "geometry": [Point(1, 2), Point(2, 1)]},
                crs="EPSG:4326")
    return vs

def test_save(dummy_vector_scene, tmp_path):
    """Test saving a dummy vector scene."""
    dummy_vector_scene.save_dataset(
            "dummy_vector_dataset",
            writer="feature",
            filename=os.fspath(tmp_path / "feature.sqlite"))
