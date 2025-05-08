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
"""Simple decision tree functionality."""
from __future__ import annotations

from typing import Callable

from satpy.utils import get_logger, recursive_dict_update

LOG = get_logger(__name__)


class DecisionTree:
    """Structure to search for nearest match from a set of parameters.

    This class is used to find the best configuration section by matching
    a set of attributes. The provided dictionary contains a mapping of
    "section name" to "decision" dictionaries. Each decision dictionary
    contains the attributes that will be used for matching plus any
    additional keys that could be useful when matched. This class will
    search these decisions and return the one with the most matching
    parameters to the attributes passed to the
    :meth:`~satpy.writers.DecisionTree.find_match` method.

    Note that decision sections are provided as a dict instead of a list
    so that they can be overwritten or updated by doing the equivalent
    of a ``current_dicts.update(new_dicts)``.

    Examples:
        Decision sections are provided as a dictionary of dictionaries.
        The returned match will be the first result found by searching
        provided `match_keys` in order.

        ::

            decisions = {
                'first_section': {
                    'a': 1,
                    'b': 2,
                    'useful_key': 'useful_value',
                },
                'second_section': {
                    'a': 5,
                    'useful_key': 'other_useful_value1',
                },
                'third_section': {
                    'b': 4,
                    'useful_key': 'other_useful_value2',
                },
            }
            tree = DecisionTree(decisions, ('a', 'b'))
            tree.find_match(a=5, b=2)  # second_section dict
            tree.find_match(a=1, b=2)  # first_section dict
            tree.find_match(a=5, b=4)  # second_section dict
            tree.find_match(a=3, b=2)  # no match

    """

    any_key = None
    _indent = "  "

    def __init__(self, decision_dicts, match_keys, multival_keys=None):
        """Init the decision tree.

        Args:
            decision_dicts (dict): Dictionary of dictionaries. Each
                sub-dictionary contains key/value pairs that can be
                matched from the `find_match` method. Sub-dictionaries
                can include additional keys outside the ``match_keys``
                provided to act as the "result" of a query. The keys of
                the root dict are arbitrary.
            match_keys (list): Keys of the provided dictionary to use for
                matching.
            multival_keys (list): Keys of `match_keys` that can be provided
                as multiple values.
                A multi-value key can be specified as a single value
                (typically a string) or a set. If a set, it will be sorted
                and converted to a tuple and then used for matching.
                When querying the tree, these keys will
                be searched for exact multi-value results (the sorted tuple)
                and if not found then each of the values will be searched
                individually in alphabetical order.

        """
        self._match_keys = match_keys
        self._multival_keys = multival_keys or []
        self._log = get_logger(__name__ + f".{self.__class__.__name__}")
        self._tree = _DecisionDict(self._match_keys[0], 0)
        if not isinstance(decision_dicts, (list, tuple)):
            decision_dicts = [decision_dicts]
        self.add_config_to_tree(*decision_dicts)

    def _indented_trace(self, indent_level: int, msg: str):
        indent = self._indent * indent_level
        self._log.trace(f"{indent}{msg}")

    def _indented_print(self, indent_level: int, msg: str):
        indent = self._indent * indent_level
        print(f"{indent}{msg}")  # noqa: T201

    def print_tree(self):
        """Print the decision tree in a structured human-readable format."""
        self._print_tree_level(0, self._tree)

    def _print_tree_level(self, level: int, curr_level: dict):
        if len(self._match_keys) == level:
            # final component
            self._print_matched_info(len(self._match_keys), curr_level, self._indented_print)
            return

        match_key = self._match_keys[level]
        for match_val, next_level in curr_level.items():
            if match_val is None:
                match_val = "<wildcard>"
            self._indented_print(level, f"{match_key}={match_val}")
            self._print_tree_level(level + 1, next_level)

    def _print_matched_info(self, level: int, decision_info: dict, print_func: Callable) -> None:
        any_keys = False
        for key in self._match_keys:
            if key not in decision_info:
                continue
            any_keys = True
            print_func(level, f"| {key}={decision_info[key]}")
        if not any_keys:
            print_func(level, "| <global wildcard match>")

    def add_config_to_tree(self, *decision_dicts):
        """Add a configuration to the tree."""
        conf = {}
        for decision_dict in decision_dicts:
            recursive_dict_update(conf, decision_dict)
        self._build_tree(conf)

    def _build_tree(self, conf):
        """Build the tree.

        Create a tree structure of dicts where each level represents the
        possible matches for a specific ``match_key``. When finding matches
        we will iterate through the tree matching each key that we know about.
        The last dict in the "tree" will contain the configure section whose
        match values led down that path in the tree.

        See :meth:`DecisionTree.find_match` for more information.

        """
        for _section_name, sect_attrs in conf.items():
            # Set a path in the tree for each section in the config files
            curr_level = self._tree
            for match_level, match_key in enumerate(self._match_keys):
                # or None is necessary if they have empty strings
                this_attr_val = sect_attrs.get(match_key, self.any_key) or None
                if match_key in self._multival_keys and isinstance(this_attr_val, list):
                    this_attr_val = tuple(sorted(this_attr_val))
                is_last_key = match_key == self._match_keys[-1]
                level_needs_init = this_attr_val not in curr_level
                if is_last_key:
                    # if we are at the last attribute, then assign the value
                    # set the dictionary of attributes because the config is
                    # not persistent
                    curr_level[this_attr_val] = sect_attrs
                elif level_needs_init:
                    curr_level[this_attr_val] = _DecisionDict(self._match_keys[match_level + 1], match_level + 1)
                curr_level = curr_level[this_attr_val]

    @staticmethod
    def _convert_query_val_to_hashable(query_val):
        _sorted_query_val = sorted(query_val)
        query_vals = [tuple(_sorted_query_val)] + _sorted_query_val
        query_vals += query_val
        return query_vals

    def _get_query_values(self, query_dict, curr_match_key):
        query_val = query_dict[curr_match_key]
        if curr_match_key in self._multival_keys and isinstance(query_val, set):
            query_vals = self._convert_query_val_to_hashable(query_val)
        else:
            query_vals = [query_val]
        return query_vals

    def _find_match_if_known(self, curr_level, remaining_match_keys, query_dict):
        match = None
        curr_match_key = remaining_match_keys[0]
        if curr_match_key not in query_dict:
            self._indented_trace(
                len(self._match_keys) - len(remaining_match_keys),
                f"Match key {curr_match_key!r} not in query dict"
            )
            return match

        query_vals = self._get_query_values(query_dict, curr_match_key)
        for query_val in query_vals:
            if not curr_level.traced_contains(query_val):
                continue
            match = self._find_match(curr_level[query_val],
                                     remaining_match_keys[1:],
                                     query_dict)
            if match is not None:
                break
        return match

    def _find_match(self, curr_level, remaining_match_keys, query_dict):
        """Find a match."""
        if len(remaining_match_keys) == 0:
            # we're at the bottom level, we must have found something
            self._indented_trace(len(self._match_keys), "Found match!")
            self._print_matched_info(
                len(self._match_keys),
                curr_level,
                self._indented_trace,
            )
            return curr_level

        match = self._find_match_if_known(
            curr_level, remaining_match_keys, query_dict)

        if match is None and curr_level.traced_contains(self.any_key):
            # if we couldn't find it using the attribute then continue with
            # the other attributes down the 'any' path
            match = self._find_match(
                curr_level[self.any_key],
                remaining_match_keys[1:],
                query_dict)
        return match

    def find_match(self, **query_dict):
        """Find a match.

        Recursively search through the tree structure for a path that matches
        the provided match parameters.

        """
        try:
            match = self._find_match(self._tree, self._match_keys, query_dict)
        except (KeyError, IndexError, ValueError, TypeError):
            LOG.debug("Match exception:", exc_info=True)
            LOG.error("Error when finding matching decision section")
            match = None

        if match is None:
            # only possible if no default section was provided
            raise KeyError("No decision section found for %s" %
                           (query_dict.get("uid", None),))
        return match


class _DecisionDict(dict):
    """Helper class for debugging decision tree choices.

    At the time of writing this class does not do anything extra to the
    behavior of what choices are made. It is only a record keeper to make
    log messages and debugging operations easier. A simple `dict` should
    be useable in its place.

    """
    def __init__(self, match_key: str, level: int):
        self.match_key = match_key
        self.level = level
        self._log = get_logger(__name__ + f".{self.__class__.__name__}.{match_key}")
        self._indent = DecisionTree._indent * self.level
        super().__init__()

    def _log_trace(self, msg: str):
        self._log.trace(f"{self._indent}{msg}")

    def traced_contains(self, item: str) -> bool:
        contains = super().__contains__(item)
        item_str = "<wildcard>" if item is DecisionTree.any_key else repr(item)
        self._log_trace(f"Checking {self.match_key!r} level for {item_str}: {contains}")
        return contains
