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


def detect_outliers(array: xr.DataArray, threshold: int = 3):
    m = array.mean(dim='time_frame')
    s = array.std(dim='time_frame')
    array.where(array.proc.abs() > m + (threshold * s))
    if np.any(onset_idx):
        mask = np.zeros(self.shape, dtype="bool")
        for (inf, sup) in onset_idx:
            mask[inf:sup] = 1
        sigma = np.nanstd(self[mask])
        mu = np.nanmean(self[mask])
    else:
        sigma = np.nanstd(self)
        mu = np.nanmean(self)
    return np.ma.masked_where(np.abs(self) > mu + (threshold * sigma), self)
