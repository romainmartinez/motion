import xarray as xr


@xr.register_dataarray_accessor("misc")
class _MiscAccessor(object):
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj
