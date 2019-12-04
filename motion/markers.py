from pathlib import Path
from typing import Union, Optional, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from motion.io.read import read_csv_or_excel, read_c3d, read_trc
from motion.io.utils import col_spliter


class Markers:
    def __new__(
        cls,
        data: Optional[Union[np.array, np.ndarray, xr.DataArray, list]] = None,
        channels: Optional[list] = None,
        time_frames: Optional[Union[np.array, list, pd.Series]] = None,
        *args,
        **kwargs,
    ) -> xr.DataArray:
        """
        Markers array with `axis`, `channel` and `time_frame` dimensions
        Parameters
        ----------
        data
            Array to be passed to xarray.DataArray
        channels
            Channel names
        args
            Positional argument(s) to be passed to xarray.DataArray
        kwargs
            Keyword argument(s) to be passed to xarray.DataArray
        Returns
        -------
        Markers xarray.DataArray
        """
        coords = {}
        if data is None:
            data = np.ndarray((0, 0, 0))
        else:
            coords["axis"] = ["x", "y", "z", "ones"]
        if data.shape[0] == 3:
            data = np.insert(data, obj=3, values=1, axis=0)
        if channels:
            coords["channel"] = channels
        if time_frames is not None:
            coords["time_frame"] = time_frames
        return xr.DataArray(
            data=data,
            dims=("axis", "channel", "time_frame"),
            coords=coords,
            name="markers",
            *args,
            **kwargs,
        )

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
        attrs: Optional[dict] = None,
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
        attrs:
            attrs to be passed to xr.DataArray. If attrs['rate'] is provided, compute the time_frame accordingly
        """
        return read_csv_or_excel(
            cls,
            "csv",
            filename,
            usecols,
            header,
            first_row,
            first_column,
            time_column,
            last_column_to_remove,
            prefix_delimiter,
            suffix_delimiter,
            skiprows,
            pandas_kwargs,
            attrs,
        )

    @classmethod
    def from_excel(
        cls,
        filename: Union[str, Path],
        sheet_name: Union[str, int] = 0,
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
        attrs: Optional[dict] = None,
    ) -> xr.DataArray:
        """
        Read excel data and convert to Markers DataArray
        Parameters
        ----------
        filename:
            Any valid string path
        sheet_name:
            Strings are used for sheet names. Integers are used in zero-indexed sheet positions
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
        attrs:
            attrs to be passed to xr.DataArray. If attrs['rate'] is provided, compute the time_frame accordingly
        """
        return read_csv_or_excel(
            cls,
            "excel",
            filename,
            usecols,
            header,
            first_row,
            first_column,
            time_column,
            last_column_to_remove,
            prefix_delimiter,
            suffix_delimiter,
            skiprows,
            pandas_kwargs,
            attrs,
            sheet_name,
        )

    @classmethod
    def from_c3d(
        cls,
        filename: Union[str, Path],
        usecols: Optional[List[Union[str, int]]] = None,
        prefix_delimiter: Optional[str] = None,
        suffix_delimiter: Optional[str] = None,
        attrs: Optional[dict] = None,
    ):
        """
        Read c3d data and convert to Markers DataArray
        Parameters
        ----------
        filename
            Any valid string path
        usecols
            All elements must either be positional or strings that correspond to column names.
            For example, a valid list-like usecols parameter would be [0, 1, 2] or ['foo', 'bar', 'baz'].
        prefix_delimiter:
            Delimiter that split each column name by its prefix (we keep only the column name)
        suffix_delimiter:
            Delimiter that split each column name by its suffix (we keep only the column name)
        attrs
            attrs to be passed to xr.DataArray
        Returns
        -------
        Markers xarray.DataArray
        """
        return read_c3d(
            cls, filename, usecols, prefix_delimiter, suffix_delimiter, attrs
        )

    @classmethod
    def from_trc(cls, filename: Union[str, Path], **kwargs):
        """
        Read a TRC file and return a Markers DataArray
        Parameters
        ----------
        filename
            Any valid string path
        kwargs
            Keyword arguments to be passed to `from_csv`
        Returns
        -------
        Markers xarray.DataArray
        """
        return read_trc(cls, filename, **kwargs)

    @staticmethod
    def reshape_flat_array(array: Union[np.array, np.ndarray]) -> xr.DataArray:
        """
        Takes a tabular numpy array (frames x [N * 3]) and return a (3 x N x frames) numpy array
        Parameters
        ----------
        array:
            A tabular array (frames x N) with N = 3 x marker
        """
        if array.shape[1] % 3 != 0:
            raise IndexError(
                "Array second dimension should be divisible by 3. "
                f"You provided an array with this shape {array.shape}"
            )
        return array.T.reshape((3, int(array.shape[1] / 3), array.shape[0]), order="F")

    @staticmethod
    def get_requested_channels_from_pandas(
        columns, header, usecols, prefix_delimiter: str, suffix_delimiter: str
    ) -> Tuple[Optional[list], Optional[list]]:
        if usecols:
            idx, channels = [], []
            if isinstance(usecols[0], int):
                for i in usecols:
                    real_idx = i * 3
                    idx.extend([real_idx, real_idx + 1, real_idx + 2])
                    channels.append(
                        col_spliter(
                            columns[real_idx], prefix_delimiter, suffix_delimiter
                        )
                    )
            elif isinstance(usecols[0], str):
                columns_split = [
                    col_spliter(col, prefix_delimiter, suffix_delimiter)
                    for col in columns
                ]
                for col in usecols:
                    i = columns_split.index(col)
                    idx.extend([i, i + 1, i + 2])
                    channels.append(col)
            else:
                raise ValueError(
                    "usecols should be None, list of string or list of int."
                    f"You provided {type(usecols)}"
                )
            return channels, idx

        if header is None:
            return None, None

        channels = [
            col_spliter(col, prefix_delimiter, suffix_delimiter)
            for col in columns
            if "Unnamed" not in col
        ]
        return channels, None
