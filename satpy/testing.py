"""Testing helpers for satpy."""

from contextlib import contextmanager
from unittest import mock

import pytest

import satpy.scene


@contextmanager
def fake_satpy_reading(scene_dict):
    """Fake the satpy reading and populate the returned scene with the contents of *scene_dict*.

    This allows users to test their programs that use satpy without actually needing to read files, eg::

        scene_dict = {channel: somedata}

        with fake_satpy_reading(scene_dict):
            scene = Scene(input_files, reader="dummy_reader")
            scene.load([channel])

    """
    with pytest.MonkeyPatch().context() as monkeypatch:
        reader_instance = mock.Mock()
        reader_instance.sensor_names = ["dummy_sensor"]
        fake_load_readers = mock.Mock()
        fake_load_readers.return_value = {"dummy_reader": reader_instance}
        monkeypatch.setattr(satpy.scene, "load_readers", fake_load_readers)

        def fake_load(self, channels):
            for channel in channels:
                self[channel] = scene_dict[channel]


        monkeypatch.setattr(satpy.scene.Scene, "load", fake_load)
        yield
