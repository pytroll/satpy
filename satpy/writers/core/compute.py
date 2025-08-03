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

import dask
from dask import array as da


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
            :meth:`satpy.scene.Scene.save_datasets`.
        targets: List of targets (should be ``RIODataset`` or ``RIOTag``) as
            returned by :meth:`satpy.scene.Scene.save_datasets`.

    Returns:
        List of ``Tuple(List[sources], List[targets])`` with a length equal to
        the number of output files planned to be written by
        :meth:`satpy.scene.Scene.save_datasets`.
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
        results (Iterable): Iterable of dask graphs resulting from calls to
                            `scn.save_datasets(..., compute=False)`
    """
    if not results:
        return

    sources, targets, delayeds = split_results(results)

    # one or more writers have targets that we need to close in the future
    if targets:
        delayeds.append(da.store(sources, targets, compute=False))
    elif sources:
        # array-like only, no targets (ex. reduce to a single map_blocks/blockwise func call)
        da.compute(sources)

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
