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
import dask
import dask.array as da
import xarray as xr
from satpy.scene import Scene
from satpy.writers import get_enhanced_image

try:
    import imageio
except ImportError:
    imageio = None

log = logging.getLogger(__name__)


def cascaded_compute(callback, arrays, optimize=True):
    """Dask helper function for iterating over computed dask arrays.

    Args:
        callback (callable): Called with a single numpy array computed from
                             the provided dask arrays.
        arrays (list, tuple): Dask arrays to pass to callback.
        optimize (bool): Whether to try to optimize the dask graphs of the
                         provided arrays.

    Returns: `dask.Delayed` object to be computed

    """
    if optimize:
        # optimize Dask graph over all objects
        dsk = da.Array.__dask_optimize__(
            # combine all Dask Array graphs
            dask.sharedict.merge(*[e.__dask_graph__() for e in arrays]),
            # get Dask Array keys in result
            list(dask.core.flatten([e.__dask_keys__() for e in arrays]))
        )
        # rebuild Dask Arrays
        arrays = [da.Array(dsk, e.name, e.chunks, e.dtype) for e in arrays]

    def _callback_wrapper(arr, cb=callback, previous_call=None):
        del previous_call  # used only for task ordering
        return cb(arr)

    current_write = None
    for dask_arr in arrays:
        current_write = dask.delayed(_callback_wrapper)(
            dask_arr, previous_call=current_write)
    return current_write


def stack(datasets):
    """First dataset at the bottom."""
    base = datasets[0].copy()
    for dataset in datasets[1:]:
        base = base.where(dataset.isnull(), dataset)
    return base


class MultiScene(object):
    """Container for multiple `Scene` objects."""

    def __init__(self, scenes=None):
        """Initialize MultiScene and validate sub-scenes"""
        self.scenes = scenes or []

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

    @property
    def all_same_area(self):
        all_areas = [ds.attrs.get('area', None)
                     for scn in self.scenes for ds in scn]
        all_areas = [area for area in all_areas if area is not None]
        return all(all_areas[0] == area for area in all_areas[1:])

    def load(self, *args, **kwargs):
        """Load the required datasets from the multiple scenes."""
        for layer in self.scenes:
            layer.load(*args, **kwargs)

    def resample(self, destination, **kwargs):
        """Resample the multiscene."""
        return self.__class__([scn.resample(destination, **kwargs)
                               for scn in self.scenes])

    def blend(self, blend_function=stack):
        """Blend the datasets into one scene."""
        new_scn = Scene()
        common_datasets = self.shared_dataset_ids
        for ds_id in common_datasets:
            datasets = [scn[ds_id] for scn in self.scenes if ds_id in scn]
            new_scn[ds_id] = blend_function(datasets)

        return new_scn

    def _get_animation_info(self, all_datasets, filename, fill_value=None):
        """Determine filename and shape of animation to be created."""
        first_dataset = [ds for ds in all_datasets if ds is not None][0]
        first_img = get_enhanced_image(first_dataset)
        first_img_data = first_img._finalize(fill_value=fill_value)[0]
        shape = tuple(first_img_data.sizes.get(dim_name)
                      for dim_name in ('y', 'x', 'bands'))
        if fill_value is None and filename.endswith('gif'):
            log.warning("Forcing fill value to '0' for GIF Luminance images")
            fill_value = 0
            shape = shape[:2]

        this_fn = filename.format(**first_dataset.attrs)
        return this_fn, shape, fill_value

    def save(self, filename, datasets=None, fps=10, fill_value=None,
             ignore_missing=False, **kwargs):
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
            ignore_missing (bool): Don't include a black frame when a dataset
                                   is missing from a child scene.
            **kwargs: Additional keyword arguments to pass to
                     `imageio.get_writer`.

        """
        if not self.all_same_area:
            raise ValueError("Sub-scenes must all be on the same area "
                             "(see the 'resample' method).")
        if imageio is None:
            raise ImportError("Missing required 'imageio' library")

        dataset_ids = datasets or self.loaded_dataset_ids

        for dataset_id in dataset_ids:
            all_datasets = [scn.datasets.get(dataset_id) for scn in self.scenes]
            this_fn, shape, fill_value = self._get_animation_info(
                all_datasets, filename, fill_value=fill_value)
            writer = imageio.get_writer(this_fn, fps=fps, **kwargs)

            def _append_wrapper(data):
                writer.append_data(data)

            data_to_write = []
            for ds in all_datasets:
                if ds is None and ignore_missing:
                    continue
                elif ds is None:
                    data = da.zeros(shape, chunks=shape)
                    data = xr.DataArray(data)
                else:
                    img = get_enhanced_image(ds)
                    data, mode = img._finalize(fill_value=fill_value)
                    if data.ndim == 3:
                        # assume all other shapes are (y, x)
                        # we need arrays grouped by pixel so
                        # transpose if needed
                        data = data.transpose('y', 'x', 'bands')
                data_to_write.append(data.data)
            cascaded_compute(_append_wrapper, data_to_write).compute()
            writer.close()
