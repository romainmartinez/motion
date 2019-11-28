import xarray as xr


@xr.register_dataset_accessor("meca")
class MecaDataSetAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @staticmethod
    def plot():
        """Plot data on a map."""
        return "plotting!"
