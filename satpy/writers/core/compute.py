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

import os
from collections.abc import Iterable
from typing import Any

import dask
import numpy as np
from dask import array as da
from dask.delayed import Delayed


def split_results(
        results: Iterable[list[da.Array | Delayed] | tuple[list[da.Array], list[Any]]],
) -> tuple[list[da.Array], list[Any], list[da.Array | Delayed]]:
    """Split results.

    Get sources, targets, and objects to be computed into separate lists from a list of
    results collected from one or more writers. This function will treat dask
    Arrays and Delayed objects with no targets as dask collections to be
    computed. Otherwise dask Arrays can be supplied with an associated target
    array-like file object and will be passed to :func:`dask.array.core.store`.

    We assume that the provided input is a list containing any combination of:

    1. List of dask Arrays (to be computed, not stored).
    2. List of Delayed objects.
    3. A two-element tuple where the first element is a collection of dask
       Arrays and the second is a collection of array-like file objects of
       the same size.
    4. A list made up of the above.

    """
    sources = []
    targets = []
    delayeds_or_arrays: list[da.Array | Delayed] = []

    for result in results:
        if not result:
            continue
        if isinstance(result, tuple) and len(result) == 2:
            sources.extend(result[0])
            targets.extend(result[1])
        elif isinstance(result, list):
            delayeds_or_arrays.extend(result)
        else:
            raise ValueError(f"Unexpected result from Satpy writer: {result!r}")
    return sources, targets, delayeds_or_arrays


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


def compute_writer_results(
        results: Iterable[list[da.Array | Delayed] | tuple[list[da.Array], list[Any]]],
) -> list[np.ndarray | str | os.PathLike | None]:
    """Compute all the given dask graphs `results` so that the files are saved.

    Args:
        results: Iterable of dask collections resulting from calls to
        ``scn.save_datasets(..., compute=False)``. The iterable passed
        should always **contain** the results of the writer/save_datasets
        call, not be the results themselves. Each collection of results from
        a writer should be either a list of dask containers to be computed
        (ex. Array or Delayed) or a 2-element tuple where the first element
        is a list of dask Array objects and the second element is a list of
        target file-like objects supporting ``__setitem__`` syntax. In the
        case of the 2-element tuple, these lists will be passed to
        :func:`dask.array.core.store` to write the Arrays to the targets.

    Returns:
        A list of the computed results. These are typically string or Path
        filenames that were created or None if the ``store`` function
        (see above) was called. If the dask collection to be computed
        is an array then the computed array will be in the returned list.

    """
    computed_results: list[np.ndarray | str | os.PathLike | None] = []
    if not results:
        return computed_results

    sources, targets, delayeds_or_arrays = split_results(results)

    # one or more writers have targets that we need to close in the future
    if targets:
        delayeds_or_arrays.append(da.store(sources, targets, compute=False))

    if delayeds_or_arrays:
        # replace Delayed's graph optimization function with the Array function
        # since a Delayed object here is only from the writer but the rest of
        # the tasks are dask array operations we want to fully optimize all
        # array operations. At the time of writing Array optimizations seem to
        # include the optimizations done for Delayed objects alone.
        with dask.config.set(delayed_optimization=dask.config.get("array_optimize", da.optimize)):
            computed_results.extend(da.compute(delayeds_or_arrays))

    if targets:
        for target in targets:
            if hasattr(target, "close"):
                target.close()

    return computed_results
