"""Classes and functions related to a dictionary with DataID keys."""

from __future__ import annotations

import numbers
from collections.abc import Iterable
from typing import Any, Literal, TypeVar, overload

import numpy as np

from .dataid import DataID, DataQuery, create_filtered_query, minimal_default_keys_config

_V = TypeVar("_V")
_T = TypeVar("_T")

#: Anything that can be used to look up a dataset: a full or partial
#: :class:`~satpy.dataset.dataid.DataID`, a :class:`~satpy.dataset.dataid.DataQuery`,
#: a dataset name, or a wavelength.
DataKey = DataID | DataQuery | str | float | numbers.Number


class TooManyResults(KeyError):
    """Special exception when one key maps to multiple items in the container."""


def get_best_dataset_key(key: DataQuery, choices: Iterable[DataID]) -> list[DataID]:
    """Choose the "best" `DataID` from `choices` based on `key`.

    To see how the keys are sorted, refer to `:meth:satpy.datasets.DataQuery.sort_dataids`.

    This function assumes `choices` has already been filtered to only
    include datasets that match the provided `key`.

    Args:
        key (DataQuery): Query parameters to sort `choices` by.
        choices (Iterable): `DataID` objects to sort through to determine
                            the best dataset.

    Returns: List of best `DataID`s from `choices`. If there is more
             than one element this function could not choose between the
             available datasets.

    """
    sorted_choices, distances = key.sort_dataids(choices)
    if len(sorted_choices) == 0 or distances[0] is np.inf:
        return []
    else:
        return [choice for choice, distance in zip(sorted_choices, distances) if distance == distances[0]]


@overload
def get_key(key: DataKey, key_container: Iterable[DataID], num_results: Literal[1] = 1, best: bool = True,
            query: DataQuery | dict[str, Any] | None = None, **kwargs: Any) -> DataID: ...


@overload
def get_key(key: DataKey, key_container: Iterable[DataID], num_results: int, best: bool = True,
            query: DataQuery | dict[str, Any] | None = None, **kwargs: Any) -> list[DataID]: ...


def get_key(key: DataKey, key_container: Iterable[DataID], num_results: int = 1, best: bool = True,  # noqa: D417
            query: DataQuery | dict[str, Any] | None = None, **kwargs: Any) -> DataID | list[DataID]:
    """Get the fully-specified key best matching the provided key.

    Only the best match is returned if `best` is `True` (default). See
    `get_best_dataset_key` for more information on how this is determined.

    `query` is provided as a convenience to filter by multiple parameters
    at once without having to filter by multiple `key` inputs.

    Args:
        key (DataID, DataQuery, str, or float): DataID or DataQuery of
                         query parameters to use for searching. Any
                         parameter that is `None` is considered a wild
                         card and any match is accepted. Can also be a
                         string representing the dataset name or a
                         number representing the dataset wavelength.
        key_container (dict or set): Container of DataID objects that
                                     uses hashing to quickly access items.
        num_results (int): Number of results to return. Use `0` for all
                           matching results. If `1` then the single matching
                           key is returned instead of a list of length 1.
                           (default: 1)
        best (bool): Sort results to get "best" result first
                     (default: True). See `get_best_dataset_key` for details.
        query (DataQuery): filter for the key which can contain for example:

            resolution (float, int, or list): Resolution of the dataset in
                                            dataset units (typically
                                            meters). This can also be a
                                            list of these numbers.
            calibration (str or list): Dataset calibration
                                    (ex.'reflectance'). This can also be a
                                    list of these strings.
            polarization (str or list): Dataset polarization
                                        (ex.'V'). This can also be a
                                        list of these strings.
            level (number or list): Dataset level (ex. 100). This can also be a
                                    list of these numbers.
            modifiers (list): Modifiers applied to the dataset. Unlike
                            resolution and calibration this is the exact
                            desired list of modifiers for one dataset, not
                            a list of possible modifiers.

    Returns:
        list or DataID: Matching key(s)

    Raises: KeyError if no matching results or if more than one result is
            found when `num_results` is `1`.

    """
    data_query = create_filtered_query(key, query)

    res = data_query.filter_dataids(key_container)
    if not res:
        raise KeyError("No dataset matching '{}' found".format(str(data_query)))

    if best:
        res = get_best_dataset_key(data_query, res)

    if num_results == 1 and not res:
        raise KeyError("No dataset matching '{}' found".format(str(data_query)))
    if num_results == 1 and len(res) != 1:
        raise TooManyResults("No unique dataset matching {}".format(str(data_query)))
    if num_results == 1:
        return res[0]
    if num_results == 0:
        return res

    return res[:num_results]


class DatasetDict(dict[DataID, _V]):
    """Special dictionary object that can handle dict operations based on dataset name, wavelength, or DataID.

    Note: Internal dictionary keys are `DataID` objects.

    """

    @overload
    def get_key(self, match_key: DataKey, num_results: Literal[1] = 1, best: bool = True, **dfilter: Any) -> DataID: ...

    @overload
    def get_key(self, match_key: DataKey, num_results: int, best: bool = True, **dfilter: Any) -> list[DataID]: ...

    def get_key(self, match_key: DataKey, num_results: int = 1, best: bool = True,  # noqa: D417
                **dfilter: Any) -> DataID | list[DataID]:
        """Get multiple fully-specified keys that match the provided query.

        Args:
            key (DataID, DataQuery, str, or float): DataID or DataQuery of
                          query parameters to use for searching. Any
                          parameter that is `None` is considered a wild
                          card and any match is accepted. Can also be a
                          string representing the dataset name or a number
                          representing the dataset wavelength.
            num_results (int): Number of results to return. If `0` return all,
                               if `1` return only that element, otherwise
                               return a list of matching keys.
            **dfilter (dict): See `get_key` function for more information.

        """
        return get_key(match_key, self.keys(), num_results=num_results,
                       best=best, **dfilter)

    def getitem(self, item: DataID) -> _V:
        """Get Node when we know the *exact* DataID."""
        return super(DatasetDict, self).__getitem__(item)

    def __getitem__(self, item: DataKey) -> _V:
        """Get item from container."""
        if isinstance(item, DataID):
            try:
                # short circuit - try to get the object without more work
                return super(DatasetDict, self).__getitem__(item)
            except KeyError:
                pass
        key = self.get_key(item)
        return super(DatasetDict, self).__getitem__(key)

    @overload
    def get(self, key: DataKey, default: None = None) -> _V | None: ...

    @overload
    def get(self, key: DataKey, default: _V | _T) -> _V | _T: ...

    def get(self, key: DataKey, default: _V | _T | None = None) -> _V | _T | None:
        """Get value with optional default."""
        try:
            dataid = self.get_key(key)
        except KeyError:
            return default
        return super(DatasetDict, self).get(dataid, default)

    def __setitem__(self, key: DataKey, value: _V) -> None:
        """Support assigning 'Dataset' objects or dictionaries of metadata."""
        if hasattr(value, "attrs"):
            # xarray.DataArray objects
            value_info = value.attrs
        else:
            value_info = value
        # use value information to make a more complete DataID
        if not isinstance(key, DataID):
            key = self._create_dataid_key(key, value_info)

        # update the 'value' with the information contained in the key
        if isinstance(value_info, dict):
            value_info.update(key.to_dict())
            value_info["_satpy_id"] = key

        return super(DatasetDict, self).__setitem__(key, value)

    def _create_dataid_key(self, key: DataKey, value_info: Any) -> DataID:
        """Create a DataID key from dictionary."""
        if not isinstance(value_info, dict):
            raise ValueError("Key must be a DataID when value is not an xarray DataArray or dict")
        try:
            return self.get_key(key)
        except KeyError:
            new_name: str | None
            if isinstance(key, str):
                new_name = key
            else:
                new_name = value_info.get("name")
            # this is a new key and it's not a full DataID tuple
            if new_name is None and value_info.get("wavelength") is None:
                raise ValueError("One of 'name' or 'wavelength' attrs "
                                 "values should be set.")
            id_keys = self._create_id_keys_from_dict(value_info)
            value_info["name"] = new_name
            return DataID(id_keys, **value_info)

    def _create_id_keys_from_dict(self, value_info_dict: dict[str, Any]) -> dict[str, Any]:
        """Create id_keys from dict."""
        try:
            id_keys = value_info_dict["_satpy_id"].id_keys
        except KeyError:
            try:
                id_keys = value_info_dict["_satpy_id_keys"]
            except KeyError:
                id_keys = minimal_default_keys_config
        return id_keys

    def contains(self, item: DataID) -> bool:
        """Check contains when we know the *exact* DataID."""
        return super(DatasetDict, self).__contains__(item)

    def __contains__(self, item: object) -> bool:
        """Check if item exists in container."""
        if not isinstance(item, (DataID, DataQuery, str, float, numbers.Number)):
            return super(DatasetDict, self).__contains__(item)
        try:
            key = self.get_key(item)
        except KeyError:
            return False
        return super(DatasetDict, self).__contains__(key)

    def __delitem__(self, key: DataKey) -> None:
        """Delete item from container."""
        if isinstance(key, DataID):
            try:
                # short circuit - try to get the object without more work
                return super(DatasetDict, self).__delitem__(key)
            except KeyError:
                pass
        dataid = self.get_key(key)
        return super(DatasetDict, self).__delitem__(dataid)
