"""Pytest configuration and setup functions."""
from pathlib import Path

import pytest


def pytest_configure(config):
    """Set test configuration."""
    from satpy import aux_download
    aux_download.RUNNING_TESTS = True


def pytest_unconfigure(config):
    """Undo previous configurations."""
    from satpy import aux_download
    aux_download.RUNNING_TESTS = False


@pytest.fixture(scope="session")
def session_tmp_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a single temp path to use for the entire session."""
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="module")
def module_tmp_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a single temp path to use for the entire session."""
    return tmp_path_factory.mktemp("data")
