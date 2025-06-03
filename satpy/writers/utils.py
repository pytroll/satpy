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
"""Utilities for writers."""
from __future__ import annotations

import logging

import dask
from dask import array as da

from satpy.writers.overlay_utils import add_decorate, add_overlay

LOG = logging.getLogger(__name__)


def get_enhanced_image(dataset, enhance=None, overlay=None, decorate=None,
                       fill_value=None):
    """Get an enhanced version of `dataset` as an :class:`~trollimage.xrimage.XRImage` instance.

    Args:
        dataset (xarray.DataArray): Data to be enhanced and converted to an image.
        enhance (bool or satpy.enhancements.enhancer.Enhancer): Whether to automatically enhance
            data to be more visually useful and to fit inside the file
            format being saved to. By default, this will default to using
            the enhancement configuration files found using the default
            :class:`~satpy.enhancements.enhancer.Enhancer` class. This can be set to
            `False` so that no enhancments are performed. This can also
            be an instance of the :class:`~satpy.enhancements.enhancer.Enhancer` class
            if further custom enhancement is needed.
        overlay (dict): Options for image overlays. See :func:`add_overlay`
            for available options.
        decorate (dict): Options for decorating the image. See
            :func:`add_decorate` for available options.
        fill_value (int or float): Value to use when pixels are masked or
            invalid. Default of `None` means to create an alpha channel.
            See :meth:`~trollimage.xrimage.XRImage.finalize` for more
            details. Only used when adding overlays or decorations. Otherwise
            it is up to the caller to "finalize" the image before using it
            except if calling ``img.show()`` or providing the image to
            a writer as these will finalize the image.
    """
    if enhance is False:
        # no enhancement
        enhancer = None
    elif enhance is None or enhance is True:
        # default enhancement
        from satpy.enhancements.enhancer import Enhancer

        enhancer = Enhancer()
    else:
        # custom enhancer
        enhancer = enhance

    # Create an image for enhancement
    img = to_image(dataset)

    if enhancer is None or enhancer.enhancement_tree is None:
        LOG.debug("No enhancement being applied to dataset")
    else:
        if dataset.attrs.get("sensor", None):
            enhancer.add_sensor_enhancements(dataset.attrs["sensor"])

        enhancer.apply(img, **dataset.attrs)

    if overlay is not None:
        img = add_overlay(img, dataset.attrs["area"], fill_value=fill_value, **overlay)

    if decorate is not None:
        img = add_decorate(img, fill_value=fill_value, **decorate)

    return img


def show(dataset, **kwargs):
    """Display the dataset as an image."""
    img = get_enhanced_image(dataset.squeeze(), **kwargs)
    img.show()
    return img


def to_image(dataset):
    """Convert ``dataset`` into a :class:`~trollimage.xrimage.XRImage` instance.

    Convert the ``dataset`` into an instance of the
    :class:`~trollimage.xrimage.XRImage` class.  This function makes no other
    changes.  To get an enhanced image, possibly with overlays and decoration,
    see :func:`~get_enhanced_image`.

    Args:
        dataset (xarray.DataArray): Data to be converted to an image.

    Returns:
        Instance of :class:`~trollimage.xrimage.XRImage`.

    """
    from trollimage.xrimage import XRImage

    dataset = dataset.squeeze()
    if dataset.ndim < 2:
        raise ValueError("Need at least a 2D array to make an image.")
    return XRImage(dataset)


def split_results(results):
    """Split results.

    Get sources, targets and delayed objects to separate lists from a list of
    results collected from (multiple) writer(s).
    """
    from dask.delayed import Delayed

    def flatten(results):
        out = []
        if isinstance(results, (list, tuple)):
            for itm in results:
                out.extend(flatten(itm))
            return out
        return [results]

    sources = []
    targets = []
    delayeds = []

    for res in flatten(results):
        if isinstance(res, da.Array):
            sources.append(res)
        elif isinstance(res, Delayed):
            delayeds.append(res)
        else:
            targets.append(res)
    return sources, targets, delayeds


def group_results_by_output_file(sources, targets):
    """Group results by output file.

    For writers that return sources and targets for ``compute=False``, split
    the results by output file.

    When not only the data but also GeoTIFF tags are dask arrays, then
    ``save_datasets(..., compute=False)``` returns a tuple of flat lists,
    where the second list consists of a mixture of ``RIOTag`` and ``RIODataset``
    objects (from trollimage).  In some cases, we may want to get a seperate
    delayed object for each file; for example, if we want to add a wrapper to do
    something with the file as soon as it's finished.  This function unflattens
    the flat lists into a list of (src, target) tuples.

    For example, to close files as soon as computation is completed::

        >>> @dask.delayed
        >>> def closer(obj, targs):
        ...     for targ in targs:
        ...         targ.close()
        ...     return obj
        >>> (srcs, targs) = sc.save_datasets(writer="ninjogeotiff", compute=False, **ninjo_tags)
        >>> for (src, targ) in group_results_by_output_file(srcs, targs):
        ...     delayed_store = da.store(src, targ, compute=False)
        ...     wrapped_store = closer(delayed_store, targ)
        ...     wrapped.append(wrapped_store)
        >>> compute_writer_results(wrapped)

    In the wrapper you can do other useful tasks, such as writing a log message
    or moving files to a different directory.

    .. warning::

        Adding a callback may impact runtime and RAM.  The pattern or cause is
        unclear.  Tests with FCI data show that for resampling with high RAM
        use (from around 15 GB), runtime increases when a callback is added.
        Tests with ABI or low RAM consumption rather show a decrease in runtime.
        More information, see `these GitHub comments
        <https://github.com/pytroll/satpy/pull/2281#issuecomment-1324910253>`_
        Users who find out more are encouraged to contact the Satpy developers
        with clues.

    Args:
        sources: List of sources (typically dask.array) as returned by
            :meth:`Scene.save_datasets`.
        targets: List of targets (should be ``RIODataset`` or ``RIOTag``) as
            returned by :meth:`Scene.save_datasets`.

    Returns:
        List of ``Tuple(List[sources], List[targets])`` with a length equal to
        the number of output files planned to be written by
        :meth:`Scene.save_datasets`.
    """
    ofs = {}
    for (src, targ) in zip(sources, targets):
        fn = targ.rfile.path
        if fn not in ofs:
            ofs[fn] = ([], [])
        ofs[fn][0].append(src)
        ofs[fn][1].append(targ)
    return list(ofs.values())


def compute_writer_results(results):
    """Compute all the given dask graphs `results` so that the files are saved.

    Args:
        results (iterable): Iterable of dask graphs resulting from calls to
                            `scn.save_datasets(..., compute=False)`
    """
    if not results:
        return

    sources, targets, delayeds = split_results(results)

    # one or more writers have targets that we need to close in the future
    if targets:
        delayeds.append(da.store(sources, targets, compute=False))

    if delayeds:
        # replace Delayed's graph optimization function with the Array function
        # since a Delayed object here is only from the writer but the rest of
        # the tasks are dask array operations we want to fully optimize all
        # array operations. At the time of writing Array optimizations seem to
        # include the optimizations done for Delayed objects alone.
        with dask.config.set(delayed_optimization=dask.config.get("array_optimize", da.optimize)):
            da.compute(delayeds)

    if targets:
        for target in targets:
            if hasattr(target, "close"):
                target.close()
