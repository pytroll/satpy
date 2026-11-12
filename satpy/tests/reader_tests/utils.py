"""Utilities for reader tests."""

import inspect


def default_attr_processor(root, attr):
    """Do not change the attribute."""
    return attr


def fill_h5(root, contents, attr_processor=default_attr_processor):
    """Fill hdf5 file with the given contents.

    Args:
        root: hdf5 file rott
        contents: Contents to be written into the file
        attr_processor: A method for modifying attributes before they are
          written to the file.
    """
    for key, val in contents.items():
        if key in ["value", "attrs"]:
            continue
        if "value" in val:
            root[key] = val["value"]
        else:
            grp = root.create_group(key)
            fill_h5(grp, contents[key])
        if "attrs" in val:
            for attr_name, attr_val in val["attrs"].items():
                root[key].attrs[attr_name] = attr_processor(root, attr_val)


def get_jit_methods(module):
    """Get all jit-compiled methods in a module."""
    res = {}
    module_name = module.__name__
    members = inspect.getmembers(module)
    for member_name, obj in members:
        if _is_jit_method(obj):
            full_name = f"{module_name}.{member_name}"
            res[full_name] = obj
    return res


def _is_jit_method(obj):
    return hasattr(obj, "py_func")
