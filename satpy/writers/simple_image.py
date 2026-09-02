"""Generic PIL/Pillow image format writer."""
from __future__ import annotations

import logging
import typing

from satpy.writers.core.image import ImageWriter

if typing.TYPE_CHECKING:
    from os import PathLike
    from typing import Any

    import dask.array as da
    from dask.delayed import Delayed
    from trollimage.xrimage import XRImage

LOG = logging.getLogger(__name__)


class PillowWriter(ImageWriter):
    """Generic PIL image format writer."""

    def __init__(self, **kwargs):
        """Initialize image writer plugin."""
        ImageWriter.__init__(
            self,
            default_config_filename="writers/simple_image.yaml",
            **kwargs)

    def save_image(
        self,
        img: XRImage,
        filename: str | None = None,
        compute: bool = True,
        **kwargs,
    ) -> list[da.Array | Delayed] | tuple[list[da.Array], list[Any]] | list[str | PathLike | None]:
        """Save Image object to a given ``filename``.

        Args:
            img: Image object to save to disk.
            filename: Optionally specify the filename to save this
                dataset to. It may include string formatting
                patterns that will be filled in by dataset
                attributes.
            compute: If `True` (default), compute and save the dataset.
                If `False` return either a `dask.delayed.Delayed`
                object or tuple of (source, target). See the
                return values below for more information.
            **kwargs: Keyword arguments to pass to the images `save` method.

        Returns:
            Value returned depends on `compute`. If `compute` is `True` then
            the return value is the result of computing a
            `dask.delayed.Delayed` object or running `dask.array.store`. If
            `compute` is `False` then the returned value is either a
            `dask.delayed.Delayed` object that can be computed using
            `delayed.compute()` or a tuple of (source, target) that should be
            passed to `dask.array.store`. If target is provided the the caller
            is responsible for calling `target.close()` if the target has
            this method.

        """
        filename = filename or self.get_filename(**img.data.attrs)

        LOG.debug("Saving to image: %s", filename)
        res = img.save(filename, compute=compute, **kwargs)
        # old trollimage <1.27.0: res is None or Delayed
        # new trollimage: res is str | Path or dask Array
        return [res]
