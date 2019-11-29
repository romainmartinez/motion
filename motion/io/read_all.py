from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import pandas as pd
import xarray as xr



def read_csv_or_excel(
    caller,
    extension: str,
    filename: Union[str, Path],
    usecols: Optional[List[Union[str, int]]] = None,
    sheet_name: Union[int, str] = 0,
    header: Optional[int] = None,
    first_row: int = 0,
    first_column: Optional[Union[str, int]] = None,
    time_column: Optional[Union[str, int]] = None,
    last_column_to_remove: Optional[Union[str, int]] = None,
    prefix_delimiter: Optional[str] = None,
    suffix_delimiter: Optional[str] = None,
    skiprows: Optional[List[int]] = None,
    pandas_kwargs: Optional[dict] = None,
):
    if skiprows is None:
        skiprows = (
            np.arange(header + 1, first_row) if header else np.arange(1, first_row)
        )

    if pandas_kwargs is None:
        pandas_kwargs = {}

    if extension == "csv":
        data = pd.read_csv(filename, header=header, skiprows=skiprows, **pandas_kwargs)
    else:
        data = pd.read_excel(
            filename,
            sheet_name=sheet_name,
            header=header,
            skiprows=skiprows,
            **pandas_kwargs,
        )

    if time_column:
        if isinstance(time_column, int):
            time_frames = data.iloc[:, time_column]
            data = data.drop(data.columns[time_column], axis=1)
        elif isinstance(time_column, str):
            time_frames = data[time_column]
            data = data.drop(time_column, axis=1)
        else:
            raise ValueError(
                f"time_column should be str or int. It is {type(time_column)}"
            )
    else:
        time_frames = None

    if first_column:
        data = data.drop(data.columns[:first_column], axis=1)

    if last_column_to_remove:
        data = data.drop(data.columns[-last_column_to_remove], axis=1)

    caller.from_2d(data.values)
    idx = []
    channels = []
    if usecols:
        if isinstance(usecols[0], int):
            for i in usecols:
                idx.extend([i, i + 1, i + 2])
                channels.append(
                    _col_spliter(data.columns[i], prefix_delimiter, suffix_delimiter)
                )
        if isinstance(usecols[0], str):
            for i, col in enumerate(data.columns):
                s = _col_spliter(col, prefix_delimiter, suffix_delimiter)
                if s in usecols:
                    idx.extend([i, i + 1, i + 2])
                    channels.append(s)
        else:
            raise ValueError(
                "usecols should be None, list of string or list of int."
                f"You provided {type(usecols)}"
            )
    else:
        channels = [
            _col_spliter(col, prefix_delimiter, suffix_delimiter)
            for col in data.columns
            if "Unnamed" not in col
        ]

    coords = {
        "channel": channels,
    }
    if time_frames:
        coords["time_frame"] = time_frames
    return xr.DataArray(data=data.T, dims=("channel", "time_frame"), coords=coords)


def _col_spliter(x, p, s):
    return x.split(p)[-1].split(s)[0]
