"""Default ID key sets and types for DataID keys."""
import numbers
from collections import namedtuple
from contextlib import suppress
from enum import IntEnum

import numpy as np


def get_keys_from_config(common_id_keys: dict, config: dict) -> dict:
    """Gather keys for a new DataID from the ones available in configured dataset."""
    id_keys = {}
    for key, val in common_id_keys.items():
        has_key = key in config
        is_required_or_default = val is not None and (val.get("required") is True or val.get("default") is not None)
        if has_key or is_required_or_default:
            id_keys[key] = val
    if not id_keys:
        raise ValueError("Metadata does not contain enough information to create a DataID.")
    return id_keys


class ValueList(IntEnum):
    """A static value list.

    This class is meant to be used for dynamically created Enums. Due to this
    it should not be used as a normal Enum class or there may be some
    unexpected behavior. For example, this class contains custom pickling and
    unpickling handling that may break in subclasses.

    """

    @classmethod
    def convert(cls, value):
        """Convert value to an instance of this class."""
        try:
            return cls[value]
        except KeyError:
            raise ValueError("{} invalid value for {}".format(value, cls))

    @classmethod
    def _unpickle(cls, enum_name, enum_members, enum_member):
        """Create dynamic class that was previously pickled.

        See :meth:`__reduce_ex__` for implementation details.

        """
        enum_cls = cls(enum_name, enum_members)
        return enum_cls[enum_member]

    def __reduce_ex__(self, proto):
        """Reduce the object for pickling."""
        return (ValueList._unpickle,
                (self.__class__.__name__, list(self.__class__.__members__.keys()), self.name))

    def __eq__(self, other):
        """Check equality."""
        return self.name == other

    def __ne__(self, other):
        """Check non-equality."""
        return self.name != other

    def __hash__(self):
        """Hash the object."""
        return hash(self.name)

    def __repr__(self):
        """Represent the values."""
        return "<" + str(self) + ">"


wlklass = namedtuple("WavelengthRange", "min central max unit", defaults=("µm",))  # type: ignore


class WavelengthRange(wlklass):
    """A named tuple for wavelength ranges.

    The elements of the range are min, central and max values, and optionally a unit
    (defaults to µm). No clever unit conversion is done here, it's just used for checking
    that two ranges are comparable.
    """

    def __eq__(self, other):
        """Return if two wavelengths are equal.

        Args:
            other (tuple or scalar): (min wl, nominal wl, max wl) or scalar wl

        Return:
            True if other is a scalar and min <= other <= max, or if other is
            a tuple equal to self, False otherwise.

        """
        if other is None:
            return False
        if isinstance(other, numbers.Number):
            return other in self
        if isinstance(other, (tuple, list)) and len(other) == 3:
            return self[:3] == other
        return super().__eq__(other)

    def __ne__(self, other):
        """Return the opposite of `__eq__`."""
        return not self == other

    def __lt__(self, other):
        """Compare to another wavelength."""
        if other is None:
            return False
        return super().__lt__(other)

    def __gt__(self, other):
        """Compare to another wavelength."""
        if other is None:
            return True
        return super().__gt__(other)

    def __hash__(self):
        """Hash this tuple."""
        return tuple.__hash__(self)

    def __str__(self):
        """Format for print out."""
        return "{0.central} {0.unit} ({0.min}-{0.max} {0.unit})".format(self)

    def __contains__(self, other):
        """Check if this range contains *other*."""
        if other is None:
            return False
        if isinstance(other, numbers.Number):
            return self.min <= other <= self.max
        with suppress(AttributeError):
            if self.unit != other.unit:
                raise NotImplementedError("Can't compare wavelength ranges with different units.")
            return self.min <= other.min and self.max >= other.max
        return False

    def distance(self, value):
        """Get the distance from value."""
        if self == value:
            try:
                return abs(value.central - self.central)
            except AttributeError:
                if isinstance(value, (tuple, list)):
                    return abs(value[1] - self.central)
                return abs(value - self.central)
        else:
            return np.inf

    @classmethod
    def convert(cls, wl):
        """Convert `wl` to this type if possible."""
        if isinstance(wl, (tuple, list)):
            return cls(*wl)
        return wl

    def to_cf(self):
        """Serialize for cf export."""
        return str(self)

    @classmethod
    def from_cf(cls, blob):
        """Return a WavelengthRange from a cf blob."""
        try:
            obj = cls._read_cf_from_string_export(blob)
        except TypeError:
            obj = cls._read_cf_from_string_list(blob)
        return obj

    @classmethod
    def _read_cf_from_string_export(cls, blob):
        """Read blob as a string created by `to_cf`."""
        pattern = "{central:f} {unit:s} ({min:f}-{max:f} {unit2:s})"
        from trollsift import Parser
        parser = Parser(pattern)
        res_dict = parser.parse(blob)
        res_dict.pop("unit2")
        obj = cls(**res_dict)
        return obj

    @classmethod
    def _read_cf_from_string_list(cls, blob):
        """Read blob as a list of strings (legacy formatting)."""
        min_wl, central_wl, max_wl, unit = blob
        obj = cls(float(min_wl), float(central_wl), float(max_wl), unit)
        return obj


class ModifierTuple(tuple):
    """A tuple holder for modifiers."""

    @classmethod
    def convert(cls, modifiers):
        """Convert `modifiers` to this type if possible."""
        if modifiers is None:
            return None
        if not isinstance(modifiers, (cls, tuple, list)):
            raise TypeError("'DataID' modifiers must be a tuple or None, "
                            "not {}".format(type(modifiers)))
        return cls(modifiers)

    def __eq__(self, other):
        """Check equality."""
        if isinstance(other, list):
            other = tuple(other)
        return super().__eq__(other)

    def __ne__(self, other):
        """Check non-equality."""
        if isinstance(other, list):
            other = tuple(other)
        return super().__ne__(other)

    def __hash__(self):
        """Hash this tuple."""
        return tuple.__hash__(self)


#: Default ID keys DataArrays.
default_id_keys_config = {
    "name": {
        "required": True,
    },
    "wavelength": {
        "type": WavelengthRange,
    },
    "resolution": {
        "transitive": False,
    },
    "calibration": {
        "enum": [
            "reflectance",
            "brightness_temperature",
            "radiance",
            "radiance_wavenumber",
            "counts",
        ],
        "transitive": True,
    },
    "modifiers": {
        "default": ModifierTuple(),
        "type": ModifierTuple,
    },
}


#: Default ID keys for coordinate DataArrays.
default_co_keys_config = {
    "name": default_id_keys_config["name"],
    "resolution": default_id_keys_config["resolution"],
}


#: Minimal ID keys for DataArrays, for example composites.
minimal_default_keys_config = {
    "name": default_id_keys_config["name"],
    "resolution": default_id_keys_config["resolution"],
}
