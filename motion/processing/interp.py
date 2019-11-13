from typing import Union

import numpy as np
import xarray as xr


def time_normalization(
    array: xr.DataArray,
    time_vector: Union[xr.DataArray, np.array] = None,
    n_frames: int = 100,
) -> xr.DataArray:
    if time_vector is None:
        time_vector = np.linspace(array.time_frame[0], array.time_frame[-1], n_frames)
    return array.interp(time_frame=time_vector)
