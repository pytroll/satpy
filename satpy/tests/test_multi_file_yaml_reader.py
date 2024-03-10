"""Tests for a multi file reader."""

import numpy as np
import pytest

from satpy.readers.yaml_reader import MultiFileYAMLReader

yaml_reader_config = {
  "reader": {
    "name": "json_npz",
    "short_name": "Simple JSON and npz reader",
    "sensors": ["vvhrr"],
    "reader": MultiFileYAMLReader
  },
  "file_types": {"metadata": {"file_patterns": ["{start_time:%Y%m%dT%H%M%S}_vvhrr.json"]},
                 "data": {"file_patterns": ["{start_time:%Y%m%dT%H%M%S}_vvhrr.data.npz"]}}
                 }


@pytest.fixture()
def vvhrr_files(tmp_path):
    """Create a fake file."""
    import json
    metadata = dict(sensor="vvhrr")
    data = np.zeros((10, 10))
    json_file = tmp_path / "20231211T111111_vvhrr.json"
    data_file = tmp_path / "20231211T111111_vvhrr.data.npz"
    with open(json_file, "w") as fp:
        json.dump(metadata, fp)
    np.savez(data_file, chanel_5=data)
    return json_file, data_file


def test_read_using_yaml_reader_interface(vvhrr_files):
    """Test loading from storage using the YAMLReader interface."""
    reader = MultiFileYAMLReader(yaml_reader_config)
    reader.assign_storage_items(vvhrr_files)
    from satpy.dataset.dataid import DataID, default_id_keys_config
    dataarray_key = DataID(default_id_keys_config, name="chanel_5", resolution=400)
    res = reader.load([dataarray_key])[dataarray_key]
    expected = np.zeros((10, 10))
    np.testing.assert_allclose(res, expected)
    assert res.attrs["sensor"] == "vvhrr"

def test_read_using_scene_interface(vvhrr_files, tmp_path):
    """Test the reader interface."""
    import yaml

    from satpy import Scene, config
    config_dir = tmp_path / "readers"
    config_dir.mkdir()
    with open(config_dir / "json_npz.yaml", "w") as fd:
        fd.write(yaml.dump(yaml_reader_config))
    with config.set(config_path=[tmp_path]):
        scn = Scene(vvhrr_files, reader="json_npz")
        scn.load(["chanel_5"])
