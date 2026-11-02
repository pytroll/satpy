"""Base modifier classes and utilities."""

from satpy.composites.core import CompositeBase


class ModifierBase(CompositeBase):
    """Base class for all modifiers.

    A modifier in Satpy is a class that takes one input DataArray to be
    changed along with zero or more other input DataArrays used to perform
    these changes. The result of a modifier typically has a lot of the same
    metadata (name, units, etc) as the original DataArray, but the data is
    different. A modified DataArray can be differentiated from the original
    DataArray by the `modifiers` property of its `DataID`.

    See the :class:`~satpy.composites.core.CompositeBase` class for information
    on the similar concept of "compositors".

    """

    def __call__(self, datasets, optional_datasets=None, **info):
        """Generate a modified copy of the first provided dataset."""
        raise NotImplementedError()
