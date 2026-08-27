"""Classes and utilities for defining generic "plugin" components."""

import logging

import yaml
from yaml import UnsafeLoader

from satpy._config import config_search_paths
from satpy.utils import recursive_dict_update

LOG = logging.getLogger(__name__)


class Plugin(object):
    """Base plugin class for all dynamically loaded and configured objects."""

    def __init__(self, default_config_filename=None, config_files=None, **kwargs):
        """Load configuration files related to this plugin.

        This initializes a `self.config` dictionary that can be used to customize the subclass.

        Args:
            default_config_filename (str): Configuration filename to use if
                no other files have been specified with `config_files`.
            config_files (list or str): Configuration files to load instead
                of those automatically found in `SATPY_CONFIG_PATH` and other
                default configuration locations.
            kwargs (dict): Unused keyword arguments.

        """
        self.default_config_filename = default_config_filename
        self.config_files = config_files
        if self.config_files is None and self.default_config_filename is not None:
            # Specify a default
            self.config_files = config_search_paths(self.default_config_filename)
        if not isinstance(self.config_files, (list, tuple)):
            self.config_files = [self.config_files]

        self.config = {}
        if self.config_files:
            for config_file in self.config_files:
                self.load_yaml_config(config_file)

    def load_yaml_config(self, conf):
        """Load a YAML configuration file and recursively update the overall configuration."""
        with open(conf, "r", encoding="utf-8") as fd:
            recursive_dict_update(self.config, yaml.load(fd, Loader=UnsafeLoader))
