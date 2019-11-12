import numpy as np
import xarray as xr


def _abs(array: xr.DataArray) -> xr.DataArray:
    return np.abs(array)


def matmul(array: xr.DataArray, other: xr.DataArray) -> xr.DataArray:
    return array @ other


def square(array: xr.DataArray, *args, **kwargs) -> xr.DataArray:
    return np.square(array, *args, **kwargs)


def sqrt(array: xr.DataArray, *args, **kwargs) -> xr.DataArray:
    return np.sqrt(array, *args, **kwargs)


def rms(array: xr.DataArray) -> xr.DataArray:
    return array.proc.square().mean().proc.sqrt()
