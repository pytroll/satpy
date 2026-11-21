"""Tests for non-discoverable reader support."""

import pytest

from satpy.readers.core.loading import load_readers


class _DummyReader:
    """Minimal reader stub used for discoverability tests."""

    def __init__(self, discoverable=True):
        """Create a test reader."""
        self.name = "dummy"
        self.info = {"discoverable": discoverable}
        self.file_handlers = {}

    def select_files_from_pathnames(self, filenames):
        """Return all supplied filenames as loadable."""
        return list(filenames)

    def create_storage_items(self, loadables, fh_kwargs=None):
        """Simulate reader storage-item creation."""

    @property
    def available_dataset_ids(self):
        """Return a single available dataset."""
        return [{"name": "test"}]


def test_non_discoverable_reader_skipped_during_auto_discovery(monkeypatch):
    """Verify non-discoverable readers are ignored during auto-discovery."""
    monkeypatch.setattr(
        "satpy.readers.core.loading.configs_for_reader",
        lambda reader: [("dummy.yaml",)],
    )

    monkeypatch.setattr(
        "satpy.readers.core.loading._get_reader_instance",
        lambda *args, **kwargs: _DummyReader(discoverable=False),
    )

    with pytest.raises(ValueError, match="No supported files found"):
        load_readers(
            filenames=["dummy.nc"],
        )


def test_non_discoverable_reader_used_when_explicitly_requested(monkeypatch):
    """Verify explicit reader selection bypasses discoverability filtering."""
    monkeypatch.setattr(
        "satpy.readers.core.loading.configs_for_reader",
        lambda reader: [("dummy.yaml",)],
    )

    monkeypatch.setattr(
        "satpy.readers.core.loading._get_reader_instance",
        lambda *args, **kwargs: _DummyReader(discoverable=False),
    )

    readers = load_readers(
        filenames=["dummy.nc"],
        reader="dummy",
    )

    assert list(readers.keys()) == ["dummy"]


def test_reader_discoverable_defaults_true(monkeypatch):
    """Verify readers remain discoverable by default."""
    monkeypatch.setattr(
        "satpy.readers.core.loading.configs_for_reader",
        lambda reader: [("dummy.yaml",)],
    )

    monkeypatch.setattr(
        "satpy.readers.core.loading._get_reader_instance",
        lambda *args, **kwargs: _DummyReader(),
    )

    readers = load_readers(
        filenames=["dummy.nc"],
    )

    assert list(readers.keys()) == ["dummy"]
