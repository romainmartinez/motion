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
    def from_rototrans(cls, rototrans, angle_sequence):
        """
        Get Angles DataArray from rototrans and specified angle sequence
        Parameters
        ----------
        rototrans
            Rototrans created with motion.create_rototrans()
        angle_sequence
            Euler sequence of angles. Valid values are all permutations of "xyz"
        Returns
        -------
        Angles DataArray with the euler angles associated
        """
        return angles_from_rototrans(cls, rototrans, angle_sequence)
