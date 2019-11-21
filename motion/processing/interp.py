from typing import Union

import numpy as np
import xarray as xr


def time_normalize(
    array: xr.DataArray,
    time_vector: Union[xr.DataArray, np.array] = None,
    n_frames: int = 100,
    norm_time_frame: bool = False,
) -> xr.DataArray:
    if time_vector is None:
        if norm_time_frame:
            first_last_time_frames = (0, 99)
            array["time_frame"] = np.linspace(
                first_last_time_frames[0],
                first_last_time_frames[1],
                array["time_frame"].shape[0],
            )
        else:
            first_last_time_frames = (array.time_frame[0], array.time_frame[-1])
        time_vector = np.linspace(
            first_last_time_frames[0], first_last_time_frames[1], n_frames
        )
    return array.interp(time_frame=time_vector)
