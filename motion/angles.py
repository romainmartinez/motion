from typing import Union, Optional

import numpy as np
import pandas as pd
import xarray as xr

from motion.processing.angles import angles_from_rototrans


class Angles:
    def __new__(
        cls,
        data: Optional[Union[np.array, np.ndarray, xr.DataArray, list]] = None,
        time_frames: Optional[Union[np.array, list, pd.Series]] = None,
        *args,
        **kwargs,
    ) -> xr.DataArray:
        """
        Angles array with `axis`, `channel` and `time_frame` dimensions
        Parameters
        ----------
        data
            Array to be passed to xarray.DataArray
        args
            Positional argument(s) to be passed to xarray.DataArray
        kwargs
            Keyword argument(s) to be passed to xarray.DataArray
        Returns
        -------
        Angles xarray.DataArray
        """
        coords = {}
        if data is None:
            data = np.ndarray((0, 0, 0))
        if time_frames is not None:
            coords["time_frame"] = time_frames
        return xr.DataArray(
            data=data,
            dims=("row", "col", "time_frame"),
            coords=coords,
            name="angles",
            *args,
            **kwargs,
        )

    @classmethod
    def from_random_data(
        cls, distribution: str = "normal", size: tuple = (10, 10, 100), *args, **kwargs
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
            Random angles `xarray.DataArray` sampled from a given distribution
        """
        return Angles(
            getattr(np.random, distribution)(size=size, *args, **kwargs).cumsum(-1)
        )

    @classmethod
    def from_rototrans(
        cls, rototrans: xr.DataArray, angle_sequence: str
    ) -> xr.DataArray:
        """
        Angles DataArray from a rototranslation matrix and specified angle sequence

        TODO: example with code

        Arguments:
            rototrans: Rototranslation matrix created with motion.Rototrans()
            angle_sequence: Euler sequence of angles. Valid values are all permutations of "xyz"

        Returns:
            Angles `xarray.DataArray` from the specified rototrans and angles sequence.
        """
        return angles_from_rototrans(cls, rototrans, angle_sequence)
