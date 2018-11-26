#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""MultiScene object to blend satellite data.
"""

import logging
import numpy as np
import dask
import dask.array as da
import xarray as xr
from satpy.scene import Scene
from satpy.writers import get_enhanced_image
from itertools import chain

try:
    from itertools import zip_longest
except ImportError:
    # python 2.7
    from itertools import izip_longest as zip_longest

try:
    import imageio
except ImportError:
    imageio = None

log = logging.getLogger(__name__)


def cascaded_compute(callback, arrays, batch_size=None, optimize=True):
    """Dask helper function for iterating over computed dask arrays.

    Args:
        callback (callable): Called with a single numpy array computed from
                             the provided dask arrays.
        arrays (list, tuple): Dask arrays to pass to callback.
        batch_size (int): Group computation in to this many arrays at a time.
        optimize (bool): Whether to try to optimize the dask graphs of the
                         provided arrays.

    Returns: `dask.Delayed` object to be computed

    """
    def _callback_wrapper(arr, previous_call, cb=callback):
        del previous_call  # used only for task ordering
        return cb(arr)

    array_batches = []
    if not batch_size:
        array_batches.append(arrays)
    else:
        arr_gens = iter(arrays)
        array_batches = (arrs for arrs in zip_longest(*([arr_gens] * batch_size)))

    for batch_arrs in array_batches:
        batch_arrs = [x for x in batch_arrs if x is not None]
        if optimize:
            # optimize Dask graph over all objects
            dsk = da.Array.__dask_optimize__(
                # combine all Dask Array graphs
                dask.sharedict.merge(*[e.__dask_graph__() for e in batch_arrs]),
                # get Dask Array keys in result
                list(dask.core.flatten([e.__dask_keys__() for e in batch_arrs]))
            )
            # rebuild Dask Arrays
            batch_arrs = [da.Array(dsk, e.name, e.chunks, e.dtype) for e in batch_arrs]

        current_write = None
        for dask_arr in batch_arrs:
            current_write = dask.delayed(_callback_wrapper)(
                dask_arr, current_write)
        yield current_write


def stack(datasets):
    """First dataset at the bottom."""
    base = datasets[0].copy()
    for dataset in datasets[1:]:
        base = base.where(dataset.isnull(), dataset)
    return base


class _SceneGenerator(object):
    """Fancy way of caching Scenes from a generator."""

    def __init__(self, scene_gen):
        self._scene_gen = scene_gen
        self._scene_cache = []
        self._dataset_idx = {}
        # this class itself is not an iterator, make one
        self._self_iter = iter(self)

    def __iter__(self):
        """Iterate over the provided scenes, caching them for later."""
        for scn in self._scene_gen:
            self._scene_cache.append(scn)
            yield scn

    def __getitem__(self, ds_id):
        """Get a specific dataset from the scenes."""
        if ds_id in self._dataset_idx:
            raise RuntimeError("Cannot get SceneGenerator item multiple times")
        self._dataset_idx[ds_id] = idx = 0
        while True:
            if idx >= len(self._scene_cache):
                scn = next(self._self_iter)
            else:
                scn = self._scene_cache[idx]
            yield scn.get(ds_id)
            idx += 1
            self._dataset_idx[ds_id] = idx


class MultiScene(object):
    """Container for multiple `Scene` objects."""

    def __init__(self, scenes=None):
        """Initialize MultiScene and validate sub-scenes.

        Args:
            scenes (iterable):
                `Scene` objects to operate on (optional)

        .. note::

            If the `scenes` passed to this object are a generator then certain
            operations performed will try to preserve that generator state.
            This may limit what properties or methods are available to the
            user. To avoid this behavior compute the passed generator by
            converting the passed scenes to a list first:
            ``MultiScene(list(scenes))``.

        """
        self._scenes = scenes or []

    def __iter__(self):
        """Iterate over the provided Scenes once."""
        return self.scenes

    @property
    def scenes(self):
        """Get list of Scene objects contained in this MultiScene.

        .. note::

            If the Scenes contained in this object are stored in a
            generator (not list or tuple) then accessing this property
            will load/iterate through the generator possibly

        """
        if self.is_generator:
            log.debug("Forcing iteration of generator-like object of Scenes")
            self._scenes = list(self._scenes)
        return self._scenes

    @property
    def is_generator(self):
        """Contained Scenes are stored as a generator."""
        return not isinstance(self._scenes, (list, tuple))

    @property
    def loaded_dataset_ids(self):
        """Union of all Dataset IDs loaded by all children."""
        return set(ds_id for scene in self.scenes for ds_id in scene.keys())

    @property
    def shared_dataset_ids(self):
        """Dataset IDs shared by all children."""
        shared_ids = set(self.scenes[0].keys())
        for scene in self.scenes[1:]:
            shared_ids &= set(scene.keys())
        return shared_ids

    def _all_same_area(self, dataset_ids):
        """Return True if all areas for the provided IDs are equal."""
        all_areas = []
        for ds_id in dataset_ids:
            for scn in self.scenes:
                ds = scn.get(ds_id)
                if ds is None:
                    continue
                all_areas.append(ds.attrs.get('area'))
        all_areas = [area for area in all_areas if area is not None]
        return all(all_areas[0] == area for area in all_areas[1:])

    @property
    def all_same_area(self):
        return self._all_same_area(self.loaded_dataset_ids)

    def _gen_load(self, gen, *args, **kwargs):
        """Perform a load in a generator so it is delayed."""
        for scn in gen:
            scn.load(*args, **kwargs)
            yield scn

    def load(self, *args, **kwargs):
        """Load the required datasets from the multiple scenes."""
        scene_gen = self._gen_load(self._scenes, *args, **kwargs)
        self._scenes = scene_gen if self.is_generator else list(scene_gen)

    def _gen_resample(self, gen, destination=None, **kwargs):
        for scn in gen:
            new_scn = scn.resample(destination, **kwargs)
            yield new_scn

    def resample(self, destination=None, **kwargs):
        """Resample the multiscene."""
        new_scenes = self._gen_resample(self._scenes, destination=destination, **kwargs)
        new_scenes = new_scenes if self.is_generator else list(new_scenes)
        return self.__class__(new_scenes)

    def blend(self, blend_function=stack):
        """Blend the datasets into one scene.

        .. note::

            Blending is not currently optimized for generator-based
            MultiScene.

        """
        new_scn = Scene()
        common_datasets = self.shared_dataset_ids
        for ds_id in common_datasets:
            datasets = [scn[ds_id] for scn in self.scenes if ds_id in scn]
            new_scn[ds_id] = blend_function(datasets)

        return new_scn

    def _get_animation_info(self, all_datasets, filename, fill_value=None):
        """Determine filename and shape of animation to be created."""
        valid_datasets = [ds for ds in all_datasets if ds is not None]
        first_dataset = valid_datasets[0]
        last_dataset = valid_datasets[-1]
        first_img = get_enhanced_image(first_dataset)
        first_img_data = first_img.finalize(fill_value=fill_value)[0]
        shape = tuple(first_img_data.sizes.get(dim_name)
                      for dim_name in ('y', 'x', 'bands'))
        if fill_value is None and filename.endswith('gif'):
            log.warning("Forcing fill value to '0' for GIF Luminance images")
            fill_value = 0
            shape = shape[:2]

        attrs = first_dataset.attrs.copy()
        if 'end_time' in last_dataset.attrs:
            attrs['end_time'] = last_dataset.attrs['end_time']
        this_fn = filename.format(**attrs)
        return this_fn, shape, fill_value

    def _get_animation_frames(self, all_datasets, shape, fill_value=None,
                              ignore_missing=False):
        """Create enhanced image frames to save to a file."""
        for idx, ds in enumerate(all_datasets):
            if ds is None and ignore_missing:
                continue
            elif ds is None:
                log.debug("Missing frame: %d", idx)
                data = da.zeros(shape, dtype=np.uint8, chunks=shape)
                data = xr.DataArray(data)
            else:
                img = get_enhanced_image(ds)
                data, mode = img.finalize(fill_value=fill_value)
                if data.ndim == 3:
                    # assume all other shapes are (y, x)
                    # we need arrays grouped by pixel so
                    # transpose if needed
                    data = data.transpose('y', 'x', 'bands')
            yield data.data

    def save_animation(self, filename, datasets=None, fps=10, fill_value=None,
                       batch_size=None, ignore_missing=False, **kwargs):
        """Helper method for saving to movie or GIF formats.

        Supported formats are dependent on the `imageio` library and are
        determined by filename extension by default.

        By default all datasets available will be saved to individual files
        using the first Scene's datasets metadata to format the filename
        provided. If a dataset is not available from a Scene then a black
        array is used instead (np.zeros(shape)).

        Args:
            filename (str): Filename to save to. Can include python string
                            formatting keys from dataset ``.attrs``
                            (ex. "{name}_{start_time:%Y%m%d_%H%M%S.gif")
            datasets (list): DatasetIDs to save (default: all datasets)
            fps (int): Frames per second for produced animation
            fill_value (int): Value to use instead creating an alpha band.
            batch_size (int): Group array computation in to this many arrays
                              at a time. This is useful to avoid memory
                              issues. Defaults to all of the arrays at once.
            ignore_missing (bool): Don't include a black frame when a dataset
                                   is missing from a child scene.
            kwargs: Additional keyword arguments to pass to
                   `imageio.get_writer`.

        """
        if imageio is None:
            raise ImportError("Missing required 'imageio' library")

        scenes = iter(self._scenes)
        first_scene = next(scenes)
        info_scenes = [first_scene]
        if 'end_time' in filename:
            # if we need the last scene to generate the filename
            # then compute all the scenes so we can figure it out
            log.debug("Generating scenes to compute end_time for filename")
            scenes = list(scenes)
            info_scenes.append(scenes[-1])
        scene_gen = _SceneGenerator(chain([first_scene], scenes))

        if not self.is_generator:
            available_ds = self.loaded_dataset_ids
        else:
            available_ds = list(first_scene.keys())
        dataset_ids = datasets or available_ds

        writers = []
        delayeds = []
        for dataset_id in dataset_ids:
            if not self.is_generator and not self._all_same_area([dataset_id]):
                raise ValueError("Sub-scene datasets must all be on the same "
                                 "area (see the 'resample' method).")

            all_datasets = scene_gen[dataset_id]
            info_datasets = [scn.get(dataset_id) for scn in info_scenes]
            this_fn, shape, this_fill = self._get_animation_info(info_datasets, filename, fill_value=fill_value)
            data_to_write = self._get_animation_frames(all_datasets, shape, this_fill, ignore_missing)

            writer = imageio.get_writer(this_fn, fps=fps, **kwargs)
            delayed = cascaded_compute(writer.append_data, data_to_write,
                                       batch_size=batch_size)
            # Save delayeds and writers to compute and close later
            delayeds.append(delayed)
            writers.append(writer)
        # compute all the datasets at once to combine any computations that can be shared
        iter_delayeds = [iter(x) for x in delayeds]
        for delayed_batch in zip_longest(*iter_delayeds):
            delayed_batch = [x for x in delayed_batch if x is not None]
            dask.compute(delayed_batch)
        for writer in writers:
            writer.close()
