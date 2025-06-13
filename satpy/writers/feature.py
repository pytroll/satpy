"""Module for writers for feature datasets."""
from satpy.writers import Writer
from pathlib import Path

class feature(Writer):
    """Write features in geodataframe with to_file method."""
    def __init__(self, name=None, filename=None, base_dir=None, layer_name=None, **kwargs):
        """Init."""
        super().__init__(name, filename, base_dir,  **kwargs)

    def save_dataset(self, dataset, filename=None, layer_name=None, **kwargs):
        """Save the geodataframe to sqlite."""
        if layer_name is None:
            layer_name = dataset.attrs["name"]

        filename = filename or self.get_filename()
        extension = Path(filename).suffix

        if extension == ".sqlite":
            kwargs = {"driver": "SQLite", "spatialite": True, "layer": layer_name}

        dataset.data.to_file(filename, **kwargs)
