from typing import Union, Optional

import numpy as np
import pandas as pd
import xarray as xr

from motion.processing.rototrans import rototrans_from_euler_angles


class Rototrans:
    def __new__(
        cls,
        data: Optional[Union[np.array, np.ndarray, xr.DataArray, list]] = None,
        time_frames: Optional[Union[np.array, list, pd.Series]] = None,
        *args,
        **kwargs,
    ) -> xr.DataArray:
        """
        Rototrans DataArray with `axis`, `channel` and `time_frame` dimensions

        To instantiate a `Rototrans` 4 by 4 and 100 frames filled with some random data:

        ```python
        import numpy as np
        from motion import Rototrans

        n_row = 4
        n_col = 4
        n_frames = 100
        data = np.random.random(size=(n_row, n_col, n_frames))
        rt = Rototrans(data)
        ```

        You can an associate time vector:

        ```python
        rate = 100  # Hz
        time_frames = np.arange(start=0, stop=n_frames / rate, step=1 / rate)
        rt = Rototrans(data, time_frames=time_frames)
        ```

        Calling `Rototrans()` generate an empty array.

        Arguments:
            data: Array to be passed to xarray.DataArray
            time_frames: Time vector in seconds associated with the `data` parameter
            args: Positional argument(s) to be passed to xarray.DataArray
            kwargs: Keyword argument(s) to be passed to xarray.DataArray

        Returns:
            Rototrans `xarray.DataArray` with the specified data and coordinates
        """
        coords = {}
        if data is None:
            data = np.eye(4)
        if data.shape[0] != 4 or data.shape[1] != 4:
            raise IndexError(
                f"data must have first and second dimensions of length 4, you have: {data.shape}"
            )
        if data.ndim == 2:
            data = data[..., np.newaxis]
        if time_frames is not None:
            coords["time_frame"] = time_frames
        return xr.DataArray(
            data=data,
            dims=("row", "col", "time_frame"),
            coords=coords,
            name="rototrans",
            *args,
            **kwargs,
        )

    @classmethod
    def from_random_data(
        cls, distribution: str = "normal", size: tuple = (4, 4, 100), *args, **kwargs
    ) -> xr.DataArray:
        """
        Create random data from a specified distribution (normal by default) using random walk

        TODO: example

        Parameters:
            distribution: Distribution available in
              [numpy.random](https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html#distributions)
            size: Shape of the desired array
            args: Positional argument(s) to be passed to numpy.random.`distribution`
            kwargs: Keyword argument(s) to be passed to numpy.random.`distribution`

        Returns:
            Random rototrans `xarray.DataArray` sampled from a given distribution
        """
        return Rototrans(
            getattr(np.random, distribution)(size=size, *args, **kwargs).cumsum(-1)
        )

    @classmethod
    def from_euler_angles(
        cls,
        angles: Optional[xr.DataArray] = None,
        angle_sequence: Optional[str] = None,
        translations: Optional[xr.DataArray] = None,
    ) -> xr.DataArray:
        """
        Rototrans DataArray from a rototranslation matrix and specified angle sequence

        TODO: example with code

        Arguments:
            angles: Euler angles of the rototranslation matrix
            angle_sequence: Euler sequence of angles. Valid values are all permutations of "xyz"
            translations: Translation part of the Rototrans matrix

        Returns:
            Rototrans `xarray.DataArray` from the specified angles and angles sequence.
        """
        return rototrans_from_euler_angles(cls, angles, angle_sequence, translations)
