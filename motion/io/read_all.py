from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import pandas as pd
import xarray as xr


def __from_csv_or_excel(
    caller: str,
    group: str,
    filename: Union[str, Path],
    usecols: Optional[List[Union[str, int]]] = None,
    sheet_name: Union[int, str] = 0,
    header: Optional[int] = None,
    first_row: int = 0,
    first_column: Optional[Union[str, int]] = None,
    time_column: Optional[Union[str, int]] = None,
    last_column_to_remove: Optional[Union[str, int]] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    skiprows: Optional[List[int]] = None,
    pandas_kwargs: Optional[dict] = None,
):
    if skiprows is None:
        skiprows = (
            np.arange(header + 1, first_row) if header else np.arange(1, first_row)
        )

    if pandas_kwargs is None:
        pandas_kwargs = {}

    if caller == "csv":
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
            time_frames = data.loc[:, time_column]
            data = data.drop(time_column, axis=1)
        else:
            raise ValueError(
                f"time_column should be str or int. It is {type(time_column)}"
            )
    else:
        time_frame = np.arange(0, data.shape[0])

    if first_column:
        data = data.drop(data.columns[:first_column], axis=1)

    if last_column_to_remove:
        data = data.drop(data.columns[-last_column_to_remove], axis=1)

    column_names = [
        col.split(prefix)[-1].split(suffix)[0]
        for col in data.columns
        if "Unnamed" not in col
    ]


def read_analogs_csv2(
    filename: Union[str, Path],
    usecols: Optional[List[Union[str, int]]] = None,
    header: Optional[int] = None,
    first_row: int = 0,
    first_column: Optional[Union[str, int]] = None,
    time_column: Optional[Union[str, int]] = None,
    last_column_to_remove: Optional[Union[str, int]] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    skiprows: Optional[List[int]] = None,
    pandas_kwargs: Optional[dict] = None,
) -> xr.DataArray:
    return __from_csv_or_excel(
        "csv",
        "ANALOG",
        filename=filename,
        usecols=usecols,
        header=header,
        first_row=first_row,
        first_column=first_column,
        time_column=time_column,
        last_column_to_remove=last_column_to_remove,
        prefix=prefix,
        suffix=suffix,
        skiprows=skiprows,
        pandas_kwargs=pandas_kwargs,
    )


def read_markers_csv2(
    filename: Union[str, Path],
    usecols: Optional[List[Union[str, int]]] = None,
    header: Optional[int] = None,
    first_row: int = 0,
    first_column: Optional[Union[str, int]] = None,
    time_column: Optional[Union[str, int]] = None,
    last_column_to_remove: Optional[Union[str, int]] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    skiprows: Optional[List[int]] = None,
    pandas_kwargs: Optional[dict] = None,
) -> xr.DataArray:
    return __from_csv_or_excel(
        "csv",
        "POINT",
        filename=filename,
        usecols=usecols,
        header=header,
        first_row=first_row,
        first_column=first_column,
        time_column=time_column,
        last_column_to_remove=last_column_to_remove,
        prefix=prefix,
        suffix=suffix,
        skiprows=skiprows,
        pandas_kwargs=pandas_kwargs,
    )
