from typing import Union

import numpy as np
import xarray as xr


def _abs(array: xr.DataArray) -> xr.DataArray:
    return np.abs(array)


def matmul(array: xr.DataArray, other: xr.DataArray) -> xr.DataArray:
    return array @ other


def square(array: xr.DataArray, *args, **kwargs) -> xr.DataArray:
    return np.square(array, *args, **kwargs)


def norm(array: xr.DataArray, dim: Union[str, list], ord: int = None) -> xr.DataArray:
    if isinstance(dim, str):
        dim = [dim]
    return xr.apply_ufunc(
        np.linalg.norm, array, input_core_dims=[dim], kwargs={"ord": ord, "axis": -1}
    )


def sqrt(array: xr.DataArray, *args, **kwargs) -> xr.DataArray:
    return np.sqrt(array, *args, **kwargs)


def rms(array: xr.DataArray) -> xr.DataArray:
    return array.proc.square().mean().proc.sqrt()


def center(
    array: xr.DataArray, mu: Union[xr.DataArray, np.array, float, int] = None
) -> xr.DataArray:
    if mu is None:
        return array - array.mean(dim="time_frame")
    else:
        return array - mu


def normalize(
    array: xr.DataArray,
    ref: Union[xr.DataArray, np.array, float, int] = None,
    scale: Union[int, float] = 100,
) -> xr.DataArray:
    if ref is None:
        ref = array.max(dim="time_frame")
    return array / (ref / scale)
