from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import xarray as xr

from motion.io.read_all import read_csv_or_excel


class Markers:
    def __new__(
        cls,
        data: Optional[Union[np.array, np.ndarray, xr.DataArray, list]] = None,
        *args,
        **kwargs
    ) -> xr.DataArray:
        """
        Markers array with `axis`, `channel` and `time_frame` dimensions
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
        xarray.DataArray
        """
        if data is None:
            data = np.ndarray((0, 0, 0))
        array = xr.DataArray(
            data=data, dims=("axis", "channel", "time_frame"), *args, **kwargs
        )
        return array

    @staticmethod
    def from_2d(array: Union[np.array, np.ndarray]) -> xr.DataArray:
        """
        Takes a tabular matrix and return a Markers DataArray
        Parameters
        ----------
        array:
            A tabular array (frames x N) with N = 3 x marker
        """
        print("debug")
        if array.shape[1] % 3 != 0:
            raise IndexError(
                'Array second dimension should be divisible by 3'
                f"You provided an array with this shape {array.shape}"
            )
        return array.reshape((3, int(array.shape[1]/3), array.shape[0]), order='F')

    @classmethod
    def from_csv(
        cls,
        filename: Union[str, Path],
        usecols: Optional[List[Union[str, int]]] = None,
        header: Optional[int] = None,
        first_row: int = 0,
        first_column: Optional[Union[str, int]] = None,
        time_column: Optional[Union[str, int]] = None,
        last_column_to_remove: Optional[Union[str, int]] = None,
        prefix_delimiter: Optional[str] = None,
        suffix_delimiter: Optional[str] = None,
        skiprows: Optional[List[int]] = None,
        pandas_kwargs: Optional[dict] = None,
    ) -> xr.DataArray:
        """
        Read csv data and convert to Markers DataArray
        Parameters
        ----------
        filename:
            Any valid string path
        usecols:
            All elements must either be positional or strings that correspond to column names.
            For example, a valid list-like usecols parameter would be [0, 1, 2] or ['foo', 'bar', 'baz'].
        header:
            Row of the header (0-indexed)
        first_row:
            First row of the data (0-indexed)
        first_column:
            First column of the data (0-indexed)
        time_column:
            Column of the time column. If None, we associate the index
        last_column_to_remove:
            If for some reason the csv reads extra columns, how many should be ignored
        prefix_delimiter:
            Delimiter that split each column name by its prefix (we keep only the column name)
        suffix_delimiter:
            Delimiter that split each column name by its suffix (we keep only the column name)
        skiprows:
            Line numbers to skip (0-indexed)
        pandas_kwargs:
            Keyword arguments to be passed to pandas.read_csv
        """
        return read_csv_or_excel(
            cls,
            "csv",
            filename=filename,
            usecols=usecols,
            header=header,
            first_row=first_row,
            first_column=first_column,
            time_column=time_column,
            last_column_to_remove=last_column_to_remove,
            prefix_delimiter=prefix_delimiter,
            suffix_delimiter=suffix_delimiter,
            skiprows=skiprows,
            pandas_kwargs=pandas_kwargs,
        )
