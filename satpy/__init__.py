"""Satpy Package initializer."""

try:
    from satpy.version import version as __version__  # noqa
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "No module named satpy.version. This could mean "
        "you didn't install 'satpy' properly. Try reinstalling ('pip "
        "install').")

from satpy._config import config  # noqa
from satpy.dataset import DataID, DataQuery  # noqa
from satpy.dataset.data_dict import DatasetDict  # noqa
from satpy.multiscene import MultiScene  # noqa
from satpy.readers.core.config import available_readers  # noqa
from satpy.readers.core.grouping import find_files_and_readers  # noqa
from satpy.scene import Scene  # noqa
from satpy.utils import get_logger  # noqa
from satpy.writers.core.config import available_writers  # noqa

log = get_logger("satpy")
