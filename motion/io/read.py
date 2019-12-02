from pathlib import Path
from typing import Union, Optional, List

import ezc3d
import numpy as np
import pandas as pd
import xarray as xr

from motion.io.utils import col_spliter


def read_c3d(
    caller,
    filename: Union[str, Path],
    usecols: Optional[List[Union[str, int]]] = None,
    prefix_delimiter: Optional[str] = None,
    suffix_delimiter: Optional[str] = None,
    attrs: Optional[dict] = None,
) -> xr.DataArray:
    group = "ANALOG" if caller.__name__ == "Analogs" else "POINT"

    reader = ezc3d.c3d(f"{filename}").c3d_swig
    columns = [
        col_spliter(label, prefix_delimiter, suffix_delimiter)
        for label in reader.parameters()
        .group(group)
        .parameter("LABELS")
        .valuesAsString()
    ]

    get_data_function = getattr(reader, f"get_{group.lower()}s")

    if usecols:
        if isinstance(usecols[0], str):
            idx = [columns.index(channel) for channel in usecols]
        elif isinstance(usecols[0], int):
            idx = usecols
        else:
            raise ValueError(
                "usecols should be None, list of string or list of int."
                f"You provided {type(usecols)}"
            )
        data = get_data_function()[:, idx, :]
        channels = [columns[i] for i in idx]
    else:
        data = get_data_function()
        channels = columns

    data_by_frame = 1 if group == "POINT" else reader.header().nbAnalogByFrame()

    attrs = attrs if attrs else {}
    attrs["first_frame"] = reader.header().firstFrame() * data_by_frame
    attrs["last_frame"] = reader.header().lastFrame() * data_by_frame
    attrs["rate"] = reader.header().frameRate() * data_by_frame
    attrs["unit"] = (
        reader.parameters().group(group).parameter("UNITS").valuesAsString()[0]
    )

    time_frames = np.arange(
        start=0, stop=data.shape[-1] / attrs["rate"], step=1 / attrs["rate"]
    )
    return caller(
        data[0, ...] if group == "ANALOG" else data, channels, time_frames, attrs=attrs
    )


def read_csv_or_excel(
    caller,
    extension: str,
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
    sheet_name: Union[int, str] = 0,
):
    if skiprows is None:
        skiprows = np.arange(header + 1, first_row) if header else np.arange(first_row)

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

    if time_column is not None:
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
        data = data.drop(data.columns[-last_column_to_remove:], axis=1)

    channels, idx = caller.get_requested_channels_from_pandas(
        data.columns, header, usecols, prefix_delimiter, suffix_delimiter
    )
    data = caller.reshape_flat_array(data.values[:, idx] if idx else data.values)

    attrs = attrs if attrs else {}
    if "rate" in attrs and time_frames is None:
        time_frames = np.arange(
            start=0, stop=data.shape[-1] / attrs["rate"], step=1 / attrs["rate"]
        )
    return caller(data, channels, time_frames, attrs=attrs)
