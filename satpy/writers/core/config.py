"""Helpers for loading writer config files."""
from __future__ import annotations

import logging
import os

import yaml
from yaml import UnsafeLoader

from satpy._config import config_search_paths, get_entry_points_config_dirs, glob_config

LOG = logging.getLogger(__name__)


def read_writer_config(config_files, loader=UnsafeLoader):
    """Read the writer `config_files` and return the info extracted."""
    conf = {}
    LOG.debug("Reading %s", str(config_files))
    for config_file in config_files:
        with open(config_file) as fd:
            conf.update(yaml.load(fd.read(), Loader=loader))

    try:
        writer_info = conf["writer"]
    except KeyError:
        raise KeyError(
            "Malformed config file {}: missing writer 'writer'".format(
                config_files))
    writer_info["config_files"] = config_files
    return writer_info


def load_writer_configs(writer_configs, **writer_kwargs):
    """Load the writer from the provided `writer_configs`."""
    try:
        writer_info = read_writer_config(writer_configs)
        writer_class = writer_info["writer"]
    except (ValueError, KeyError, yaml.YAMLError):
        raise ValueError("Invalid writer configs: "
                         "'{}'".format(writer_configs))
    init_kwargs, kwargs = writer_class.separate_init_kwargs(writer_kwargs)
    writer = writer_class(config_files=writer_configs,
                          **init_kwargs)
    return writer, kwargs


def load_writer(writer, **writer_kwargs):
    """Find and load writer `writer` in the available configuration files."""
    config_fn = writer + ".yaml" if "." not in writer else writer
    config_files = config_search_paths(os.path.join("writers", config_fn))
    writer_kwargs.setdefault("config_files", config_files)
    if not writer_kwargs["config_files"]:
        raise ValueError("Unknown writer '{}'".format(writer))

    try:
        return load_writer_configs(writer_kwargs["config_files"],
                                   **writer_kwargs)
    except ValueError:
        raise ValueError("Writer '{}' does not exist or could not be "
                         "loaded".format(writer))


def configs_for_writer(writer=None):
    """Generate writer configuration files for one or more writers.

    Args:
        writer (Optional[str]): Yield configs only for this writer

    Returns: Generator of lists of configuration files

    """
    if writer is not None:
        if not isinstance(writer, (list, tuple)):
            writer = [writer]
        # given a config filename or writer name
        config_files = [w if w.endswith(".yaml") else w + ".yaml" for w in writer]
    else:
        paths = get_entry_points_config_dirs("satpy.writers")
        writer_configs = glob_config(os.path.join("writers", "*.yaml"), search_dirs=paths)
        config_files = set(writer_configs)

    for config_file in config_files:
        config_basename = os.path.basename(config_file)
        paths = get_entry_points_config_dirs("satpy.writers")
        writer_configs = config_search_paths(
            os.path.join("writers", config_basename),
            search_dirs=paths,
        )

        if not writer_configs:
            LOG.warning("No writer configs found for '%s'", writer)
            continue

        yield writer_configs


def available_writers(as_dict=False):
    """Available writers based on current configuration.

    Args:
        as_dict (bool): Optionally return writer information as a dictionary.
                        Default: False

    Returns: List of available writer names. If `as_dict` is `True` then
             a list of dictionaries including additionally writer information
             is returned.

    """
    writers = []
    for writer_configs in configs_for_writer():
        try:
            writer_info = read_writer_config(writer_configs)
        except (KeyError, IOError, yaml.YAMLError):
            LOG.warning("Could not import writer config from: %s", writer_configs)
            LOG.debug("Error loading YAML", exc_info=True)
            continue
        writers.append(writer_info if as_dict else writer_info["name"])
    return writers
