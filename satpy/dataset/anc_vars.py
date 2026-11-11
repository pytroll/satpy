"""Utilities for dealing with ancillary variables."""

from .dataid import DataID, default_id_keys_config


def dataset_walker(datasets):
    """Walk through *datasets* and their ancillary data.

    Yields datasets and their parent.
    """
    for dataset in datasets:
        yield dataset, None
        for anc_ds in dataset.attrs.get("ancillary_variables", []):
            try:
                anc_ds.attrs
                yield anc_ds, dataset
            except AttributeError:
                continue


def replace_anc(dataset, parent_dataset):
    """Replace *dataset* the *parent_dataset*'s `ancillary_variables` field."""
    if parent_dataset is None:
        return
    id_keys = parent_dataset.attrs.get(
            "_satpy_id_keys",
            dataset.attrs.get(
                "_satpy_id_keys",
                default_id_keys_config))
    current_dataid = DataID(id_keys, **dataset.attrs)
    for idx, ds in enumerate(parent_dataset.attrs["ancillary_variables"]):
        if current_dataid == DataID(id_keys, **ds.attrs):
            parent_dataset.attrs["ancillary_variables"][idx] = dataset
            return
