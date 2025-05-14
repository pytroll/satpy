from satpy.writers import Writer


class sqlite(Writer):
    """Write features in geodataframe to sqlite."""
    def __init__(self, name=None, filename=None, base_dir=None, layer_name=None, **kwargs):
        super().__init__(name, filename, base_dir,  **kwargs)

    def save_dataset(self, dataset, filename=None, layer_name=None, **kwargs):
        """Save the geodataframe to sqlite"""
        if layer_name is None:
            layer_name = dataset.attrs["name"]

        print(layer_name, filename)
        dataset.data.to_file(filename, driver="SQLite", spatialite=True, layer=layer_name)
