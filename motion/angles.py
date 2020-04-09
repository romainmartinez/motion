from typing import Union, Optional

import numpy as np
import pandas as pd
import xarray as xr

from motion.processing import angles


class Angles:
    def __new__(
        cls,
        data: Optional[Union[np.array, np.ndarray, xr.DataArray, list]] = None,
        time_frames: Optional[Union[np.array, list, pd.Series]] = None,
        *args,
        **kwargs,
    ) -> xr.DataArray:
        """
        Angles DataArray with `axis`, `channel` and `time_frame` dimensions
         used for joint angles.

        Arguments:
            data: Array to be passed to xarray.DataArray
            time_frames: Time vector in seconds associated with the `data` parameter
            args: Positional argument(s) to be passed to xarray.DataArray
            kwargs: Keyword argument(s) to be passed to xarray.DataArray

        Returns:
            Angles `xarray.DataArray` with the specified data and coordinates

        !!! example
            To instantiate an `Angles` 4 by 4 and 100 frames filled with some random data:

            ```python
            import numpy as np
            from motion import Angles

            n_axis = 3
            n_channel = 3
            n_frames = 100
            data = np.random.random(size=(n_axis, n_channel, n_frames))
            angles = Angles(data)
            ```

            You can an associate time vector:

            ```python
            rate = 100  # Hz
            time_frames = np.arange(start=0, stop=n_frames / rate, step=1 / rate)
            angles = Angles(data, time_frames=time_frames)
            ```

        !!! note
            Calling `Angles()` generate an empty array.
        """
        coords = {}
        if data is None:
            data = np.ndarray((0, 0, 0))
        if time_frames is not None:
            coords["time_frame"] = time_frames
        return xr.DataArray(
            data=data,
            dims=("axis", "channel", "time_frame"),
            coords=coords,
            name="angles",
            *args,
            **kwargs,
        )

    @classmethod
    def from_random_data(
        cls, distribution: str = "normal", size: tuple = (3, 10, 100), *args, **kwargs
    ) -> xr.DataArray:
        """
        Create random data from a specified distribution (normal by default) using random walk.

        Arguments:
            distribution: Distribution available in
              [numpy.random](https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html#distributions)
            size: Shape of the desired array
            args: Positional argument(s) to be passed to numpy.random.`distribution`
            kwargs: Keyword argument(s) to be passed to numpy.random.`distribution`

        Returns:
            Random angles `xarray.DataArray` sampled from a given distribution

        !!! example
            To instantiate an `Angles` with some random data sampled from a normal distribution:

            ```python
            from motion import Angles

            n_frames = 100
            size = 10, 10, n_frames
            angles = Angles.from_random_data(size=size)
            ```

            You can choose any distribution available in
                [numpy.random](https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html#distributions):

            ```python
            angles = Angles.from_random_data(distribution="uniform", size=size, low=1, high=10)
            ```
        """
        return Angles(
            getattr(np.random, distribution)(size=size, *args, **kwargs).cumsum(-1)
        )

    @classmethod
    def from_rototrans(
        cls, rototrans: xr.DataArray, angle_sequence: str
    ) -> xr.DataArray:
        """
        Angles DataArray from a rototranslation matrix and specified angle sequence.

        Arguments:
            rototrans: Rototranslation matrix created with motion.Rototrans()
            angle_sequence: Euler sequence of angles. Valid values are all permutations of "xyz"

        Returns:
            Angles `xarray.DataArray` from the specified rototrans and angles sequence

        !!! example
            To get the euler angles from a random rototranslation matrix with a given angle sequence type:

            ```python
            from motion import Angles, Rototrans

            size = (4, 4, 100)
            rt = Rototrans.from_random_data(size=size)
            angles_sequence = "xyz"

            angles = Angles.from_rototrans(rototrans=rt, angle_sequence=angles_sequence)
            ```
        """
        return angles.angles_from_rototrans(cls, rototrans, angle_sequence)
