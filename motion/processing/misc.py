from typing import Union

import numpy as np
import xarray as xr
from scipy import fftpack


def fft(
    array: xr.DataArray, freq: Union[int, float], only_positive=True
) -> xr.DataArray:
    n = array.time_frame.shape[0]
    yfft = fftpack.fft(array, n)
    freqs = fftpack.fftfreq(n, 1 / freq)
    if only_positive:
        amp = 2 * np.abs(yfft) / n
        half = int(np.floor(n / 2))
        amp = amp[..., :half]
        freqs = freqs[:half]
    else:
        amp = np.abs(yfft) / n

    coords = {}
    if "axis" in array.dims:
        coords["axis"] = array.axis
    coords["channel"] = array.channel
    coords["freqs"] = freqs

    return xr.DataArray(data=amp, dims=coords.keys(), coords=coords)
