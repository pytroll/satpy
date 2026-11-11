"""Utility functions for area definitions."""
from ._config import config_search_paths, get_config_path


def get_area_file():
    """Find area file(s) to use.

    The files are to be named `areas.yaml` or `areas.def`.
    """
    paths = config_search_paths("areas.yaml")
    if paths:
        return paths
    else:
        return get_config_path("areas.def")


def get_area_def(area_name):
    """Get the definition of *area_name* from file.

    The file is defined to use is to be placed in the $SATPY_CONFIG_PATH
    directory, and its name is defined in satpy's configuration file.
    """
    try:
        from pyresample import parse_area_file
    except ImportError:
        from pyresample.utils import parse_area_file
    return parse_area_file(get_area_file(), area_name)[0]
