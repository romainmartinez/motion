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
        Rototrans array with `axis`, `channel` and `time_frame` dimensions
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
        Rototrans xarray.DataArray
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
    def from_euler_angles(
        cls,
        angles: Optional[xr.DataArray] = None,
        angle_sequence: Optional[str] = None,
        translations: Optional[xr.DataArray] = None,
    ):
        """
        Get rototrans DataArray from angles/translations
        Parameters
        ----------
        angles:
            Euler angles of the rototranslation
        angle_sequence:
            Euler sequence of angles; valid values are all permutation of axes (e.g. "xyz", "yzx", ...)
        translations
            Translation part of the Rototrans matrix
        """
        return rototrans_from_euler_angles(cls, angles, angle_sequence, translations)
